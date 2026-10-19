"""Draft evaluation questions from sampled evidence, then filter the drafts."""

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field

import psycopg
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from kgrag.config import get_settings
from kgrag.extract.llm import COST_SAFETY_MARGIN
from kgrag.observability import langchain_callbacks
from kgrag.questions.sample import (
    MIN_TEMPORAL_GAP,
    EvidenceSet,
    QuestionType,
    iter_evidence_sets,
    thread_topic,
)
from kgrag.store.repo import Document, QuestionRow, get_documents, insert_question, log_failure
from kgrag.tokens import count_tokens

DRAFT_PROMPT_VERSION = "draft-v1"
MULTI_HOP: tuple[QuestionType, ...] = ("inference", "comparison", "temporal")
# Randomly sampled single-hop goes first so ID-linked relabels never displace it.
DRAFT_ORDER: tuple[QuestionType, ...] = ("single_hop", *MULTI_HOP, "null")
NULL_ANSWER = "The mailing list does not contain this information."
MAX_CONSECUTIVE_FAILURES = 5  # a systemic error (bad key, outage) must not burn the whole corpus
MAX_KEY_TERMS = 4
MIN_KEY_TERMS = 2
AVG_GOLD_DOCS = {"single_hop": 1, "inference": 2.5, "comparison": 2, "temporal": 2.5, "null": 0}
# Leave-one-out calls: only 3-doc sets (about half of inference/temporal) get one call per doc.
AVG_LOO_CALLS = {"inference": 1.5, "temporal": 1.5}
# Assumptions for the cost estimate: a draft is a short JSON object, a filter verdict is tiny.
EST_DRAFT_OUTPUT_TOKENS = 200
EST_FILTER_OUTPUT_TOKENS = 30
log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You write evaluation questions about the Apache Kafka developer mailing list (dev@). "
    "Questions must be self-contained: name the KIP, ticket, release or topic explicitly, "
    "never refer to 'the email', 'the document' or 'the thread'. Keep answers short: an "
    "entity, a date, a version, yes/no, or one sentence. Also return 2-4 distinctive "
    "key_terms (words or identifiers that would appear in a document stating the answer)."
)
TYPE_INSTRUCTIONS: dict[str, str] = {
    "single_hop": (
        "Write a question that can be answered from the single document below, about a "
        "concrete fact it states."
    ),
    "inference": (
        "Write a question whose answer requires combining facts from EVERY document below; "
        "no single document may be enough. The documents are linked by a shared KIP or "
        "ticket, so the question should draw on what each one contributes."
    ),
    "comparison": (
        "The documents concern two different items. Write a question that compares them "
        "(for example which came first, which is broader, whether they conflict) and needs "
        "BOTH documents to answer. Name both items in the question."
    ),
    "temporal": (
        "The documents are from different dates. Write a question about how something "
        "changed or what happened first or later, which needs EVERY document to answer."
    ),
    "null": (
        "You are given only the topic of a discussion. Write a question that is plausible "
        "for this mailing list and related to the topic, but whose answer is NOT stated "
        "anywhere on the list (for example a detail nobody discussed, or a decision never "
        "made). Do not invent a premise that is false; ask for something that simply was "
        "never said. The answer field is ignored: set it to "
        f"'{NULL_ANSWER}'. Make the key_terms the words a document answering the question "
        "would have to contain."
    ),
}
OTHERS_PROMPT = (
    "Decide whether the question can be fully answered from the documents below alone. They "
    "are only some of the evidence, so answer true only if they by themselves contain "
    "everything needed to give the reference answer."
)
FILTER_PROMPT = (
    "Decide whether the question can be fully answered from the single document below "
    "alone, with no other source. Answer true only if this document by itself contains "
    "everything needed to give the reference answer."
)


class DraftSchema(BaseModel):
    """Structured output requested from the drafting model."""

    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    key_terms: list[str] = Field(default_factory=list)


class SingleSourceSchema(BaseModel):
    answerable: bool


class OtherDocsSchema(BaseModel):
    answerable: bool


class DraftQuestion(BaseModel):
    question: str
    answer: str
    type: QuestionType
    gold_doc_ids: list[str]
    key_terms: list[str] = Field(max_length=MAX_KEY_TERMS)


@dataclass
class DraftStats:
    drafted: int = 0
    kept: int = 0
    relabeled: int = 0
    dropped_single_source: int = 0
    dropped_null_present: int = 0
    dropped_shrunk: int = 0
    skipped_existing: int = 0
    failed: int = 0
    kept_by_type: dict[str, int] = field(default_factory=dict)
    kept_by_origin: dict[str, int] = field(default_factory=dict)
    shortfall: dict[str, int] = field(default_factory=dict)


def _default_llm() -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=get_settings().models.draft, max_tokens=4000)


def _format_doc(doc: Document) -> str:
    return f"[{doc.doc_id}] {doc.date:%Y-%m-%d} · {doc.sender} · {doc.subject}\n{doc.text}"


def _invoke(llm: BaseChatModel, schema: type[BaseModel], system: str, user: str):
    chain = llm.with_structured_output(schema, method="json_schema")
    return chain.invoke(
        [SystemMessage(system), HumanMessage(user)], config={"callbacks": langchain_callbacks()}
    )


def draft_question(
    es: EvidenceSet, docs: list[Document], llm: BaseChatModel, *, topic: str | None = None
) -> DraftQuestion:
    """Draft one question/answer for an evidence set (null sets use `topic` instead of docs)."""
    if es.type == "null":
        context = f"Topic context (thread subject and first message):\n\n{topic or ''}"
    else:
        context = "Documents:\n\n" + "\n\n".join(_format_doc(d) for d in docs)
    out = _invoke(
        llm, DraftSchema, f"{SYSTEM_PROMPT}\n\n{TYPE_INSTRUCTIONS[es.type]}", context
    )
    terms = [t.strip() for t in out.key_terms if t.strip()][:MAX_KEY_TERMS]
    if len(terms) < MIN_KEY_TERMS:
        raise ValueError(f"draft returned {len(terms)} key terms, need at least {MIN_KEY_TERMS}")
    return DraftQuestion(
        question=out.question.strip(),
        answer=NULL_ANSWER if es.type == "null" else out.answer.strip(),
        type=es.type, gold_doc_ids=list(es.doc_ids), key_terms=terms,
    )


def single_source_doc(draft: DraftQuestion, docs: list[Document], llm: BaseChatModel) -> str | None:
    """The first gold doc that answers the question on its own, or None."""
    for doc in docs:
        user = (
            f"Document:\n\n{_format_doc(doc)}\n\nQuestion: {draft.question}\n"
            f"Reference answer: {draft.answer}"
        )
        if _invoke(llm, SingleSourceSchema, FILTER_PROMPT, user).answerable:
            return doc.doc_id
    return None


def leave_one_out(draft: DraftQuestion, docs: list[Document], llm: BaseChatModel) -> list[Document]:
    """For a 3-doc set, drop the first doc the other two can answer without; else keep all."""
    if len(docs) < 3:
        return docs
    for k in range(len(docs)):
        others = docs[:k] + docs[k + 1 :]
        user = (
            "Documents:\n\n" + "\n\n".join(_format_doc(d) for d in others)
            + f"\n\nQuestion: {draft.question}\nReference answer: {draft.answer}"
        )
        if _invoke(llm, OtherDocsSchema, OTHERS_PROMPT, user).answerable:
            return others
    return docs


def answerable_from_single_source(
    draft: DraftQuestion, docs: list[Document], llm: BaseChatModel
) -> bool:
    return single_source_doc(draft, docs, llm) is not None


def null_question_absent(draft: DraftQuestion, conn: psycopg.Connection) -> bool:
    """True when no single document matches ALL the key terms (full-text search)."""
    if not draft.key_terms:
        return False  # nothing to verify absence with
    cond = " and ".join(
        "to_tsvector('english', text) @@ plainto_tsquery('english', %s)" for _ in draft.key_terms
    )
    row = conn.execute(f"select 1 from documents where {cond} limit 1", draft.key_terms).fetchone()
    return row is None


def _question_id(d: DraftQuestion) -> str:
    raw = d.type + "".join(sorted(d.gold_doc_ids)) + d.question
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _already_drafted(conn: psycopg.Connection, es: EvidenceSet) -> bool:
    """True if questions already hold this evidence (null: same thread link)."""
    if es.type == "null":
        row = conn.execute(
            "select 1 from questions where type = 'null' and link = %s limit 1", (es.link,)
        ).fetchone()
    else:
        row = conn.execute(
            "select 1 from questions where type = %s and gold_doc_ids @> %s and "
            "gold_doc_ids <@ %s limit 1", (es.type, es.doc_ids, es.doc_ids),
        ).fetchone()
    return row is not None


def draft_questions(
    conn: psycopg.Connection,
    counts: dict[QuestionType, int],
    *,
    seed: int,
    llm: BaseChatModel | None = None,
    oversample: float = 1.3,
) -> DraftStats:
    """Sample, draft, filter and store drafts (status 'draft').

    Each type keeps sampling until ceil(count * oversample) drafts are kept or the corpus runs
    out of evidence sets. Randomly sampled single-hop goes first; a multi-hop draft that one
    doc answers alone is relabeled single_hop only into remaining single-hop headroom.
    """
    stats = DraftStats()
    model = get_settings().models.draft
    targets = {t: math.ceil(c * oversample) for t, c in counts.items() if c > 0}
    if not targets:
        return stats
    if llm is None:
        llm = _default_llm()
    single_target = targets.get("single_hop", 0)
    single_kept = 0

    def store(d: DraftQuestion, es: EvidenceSet, origin: str) -> None:
        row = QuestionRow(
            id=_question_id(d), type=d.type, question=d.question, answer=d.answer,
            gold_doc_ids=d.gold_doc_ids, status="draft", draft_model=model,
            prompt_version=DRAFT_PROMPT_VERSION, link=es.link, origin=origin,
            key_terms=d.key_terms,
        )
        if insert_question(conn, row):
            stats.kept += 1
            stats.kept_by_type[d.type] = stats.kept_by_type.get(d.type, 0) + 1
            stats.kept_by_origin[origin] = stats.kept_by_origin.get(origin, 0) + 1
        conn.commit()

    def process(es: EvidenceSet) -> bool:
        """Draft and filter one set; True when it counts toward its own type's target."""
        nonlocal single_kept
        docs = get_documents(conn, es.doc_ids)
        if len(docs) != len(es.doc_ids):
            raise ValueError(f"missing documents for {es.doc_ids}")
        topic = None
        if es.type == "null":
            topic = thread_topic(conn, es.link.removeprefix("thread:"))
        d = draft_question(es, docs, llm, topic=topic)
        stats.drafted += 1
        if es.type == "single_hop":
            store(d, es, "sampled")
            single_kept += 1
            return True
        if es.type == "null":
            if not null_question_absent(d, conn):
                stats.dropped_null_present += 1
                return False
            store(d, es, "sampled")
            return True
        alone = single_source_doc(d, docs, llm)
        if alone is not None:
            if single_kept >= single_target:
                stats.dropped_single_source += 1
                return False
            stats.relabeled += 1
            single_kept += 1
            store(d.model_copy(update={"type": "single_hop", "gold_doc_ids": [alone]}),
                  es, "relabeled")
            return False
        kept_docs = leave_one_out(d, docs, llm)
        if len(kept_docs) < len(docs):
            dates = [x.date for x in kept_docs]
            if es.type == "temporal" and max(dates) - min(dates) < MIN_TEMPORAL_GAP:
                stats.dropped_shrunk += 1
                return False
            d = d.model_copy(update={"gold_doc_ids": [x.doc_id for x in kept_docs]})
        store(d, es, "sampled")
        return True

    for qtype in DRAFT_ORDER:
        target = targets.get(qtype)
        if target is None:
            continue
        kept = failures = 0
        for es in iter_evidence_sets(conn, qtype, seed):
            if kept >= target or failures >= MAX_CONSECUTIVE_FAILURES:
                break
            if _already_drafted(conn, es):
                stats.skipped_existing += 1
                continue
            try:
                counted = process(es)
                failures = 0
            except Exception as exc:  # noqa: BLE001 - one bad draft must not lose the others
                conn.rollback()
                log_failure(conn, "draft", es.link, f"{type(exc).__name__}: {exc}")
                conn.commit()
                stats.failed += 1
                failures += 1
                continue
            kept += counted
        if kept < target:
            stats.shortfall[qtype] = target - kept
            log.warning("%s: kept %d of %d drafts (ran out of evidence or failing)",
                        qtype, kept, target)
    return stats


def avg_doc_tokens(conn: psycopg.Connection) -> int:
    row = conn.execute("select avg(token_count) from documents").fetchone()
    return round(row[0] or 0)


def estimate_draft_cost_usd(
    counts: dict[QuestionType, int], avg_tokens: int, model: str, *, oversample: float = 1.3
) -> float:
    """Draft call per set plus one filter call per gold doc of each multi-hop set."""
    price = get_settings().pricing[model]
    draft_overhead = count_tokens(SYSTEM_PROMPT + json.dumps(DraftSchema.model_json_schema()))
    filter_overhead = count_tokens(
        FILTER_PROMPT + json.dumps(SingleSourceSchema.model_json_schema())
    )
    in_tok = out_tok = 0.0
    for qtype, count in counts.items():
        n = math.ceil(count * oversample)
        docs = AVG_GOLD_DOCS[qtype]
        in_tok += n * (draft_overhead + max(docs, 1) * avg_tokens)
        out_tok += n * EST_DRAFT_OUTPUT_TOKENS
        if qtype in MULTI_HOP:
            # one single-doc call per gold doc, plus leave-one-out calls on 3-doc sets
            filters = n * docs
            in_tok += filters * (filter_overhead + avg_tokens + 100)
            loo = n * AVG_LOO_CALLS.get(qtype, 0)
            in_tok += loo * (filter_overhead + 2 * avg_tokens + 100)
            out_tok += (filters + loo) * EST_FILTER_OUTPUT_TOKENS
    return (in_tok * price.input + out_tok * price.output) / 1_000_000 * COST_SAFETY_MARGIN

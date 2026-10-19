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
    EvidenceSet,
    QuestionType,
    sample_evidence_sets,
    thread_topic,
)
from kgrag.store.repo import Document, QuestionRow, get_documents, insert_question, log_failure
from kgrag.tokens import count_tokens

DRAFT_PROMPT_VERSION = "draft-v1"
MULTI_HOP: tuple[QuestionType, ...] = ("inference", "comparison", "temporal")
# Single-hop is drafted last so relabeled multi-hop drafts count toward its quota.
DRAFT_ORDER: tuple[QuestionType, ...] = (*MULTI_HOP, "null", "single_hop")
AVG_GOLD_DOCS = {"single_hop": 1, "inference": 2.5, "comparison": 2, "temporal": 2.5, "null": 0}
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
        "never said."
    ),
}
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


class DraftQuestion(BaseModel):
    question: str
    answer: str
    type: QuestionType
    gold_doc_ids: list[str]
    key_terms: list[str]


@dataclass
class DraftStats:
    drafted: int = 0
    kept: int = 0
    relabeled: int = 0
    dropped_single_source: int = 0
    dropped_null_present: int = 0
    failed: int = 0
    kept_by_type: dict[str, int] = field(default_factory=dict)


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
    return DraftQuestion(
        question=out.question.strip(), answer=out.answer.strip(), type=es.type,
        gold_doc_ids=list(es.doc_ids), key_terms=[t.strip() for t in out.key_terms if t.strip()],
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


def draft_questions(
    conn: psycopg.Connection,
    counts: dict[QuestionType, int],
    *,
    seed: int,
    llm: BaseChatModel | None = None,
    oversample: float = 1.3,
) -> DraftStats:
    """Sample, draft, filter and store drafts (status 'draft'); ~oversample x each count."""
    stats = DraftStats()
    model = get_settings().models.draft
    targets = {t: math.ceil(c * oversample) for t, c in counts.items() if c > 0}
    if not targets:
        return stats
    if llm is None:
        llm = _default_llm()
    single_quota = targets.get("single_hop", 0)

    def process(es: EvidenceSet) -> None:
        nonlocal single_quota
        try:
            docs = get_documents(conn, es.doc_ids)
            if len(docs) != len(es.doc_ids):
                raise ValueError(f"missing documents for {es.doc_ids}")
            topic = None
            if es.type == "null":
                topic = thread_topic(conn, es.link.removeprefix("thread:"))
            d = draft_question(es, docs, llm, topic=topic)
            stats.drafted += 1
            if es.type in MULTI_HOP:
                alone = single_source_doc(d, docs, llm)
                if alone is not None:
                    if single_quota <= 0:
                        stats.dropped_single_source += 1
                        return
                    d = d.model_copy(update={"type": "single_hop", "gold_doc_ids": [alone]})
                    single_quota -= 1
                    stats.relabeled += 1
            elif es.type == "null" and not null_question_absent(d, conn):
                stats.dropped_null_present += 1
                return
            row = QuestionRow(
                id=_question_id(d), type=d.type, question=d.question, answer=d.answer,
                gold_doc_ids=d.gold_doc_ids, status="draft", draft_model=model,
                prompt_version=DRAFT_PROMPT_VERSION,
            )
            if insert_question(conn, row):
                stats.kept += 1
                stats.kept_by_type[d.type] = stats.kept_by_type.get(d.type, 0) + 1
            conn.commit()
        except Exception as exc:  # noqa: BLE001 - one bad draft must not lose the others
            conn.rollback()
            log_failure(conn, "draft", es.link, f"{type(exc).__name__}: {exc}")
            conn.commit()
            stats.failed += 1

    for qtype in DRAFT_ORDER:
        if qtype not in targets:
            continue
        n = targets[qtype]
        if qtype == "single_hop":
            n = max(0, n - stats.relabeled)  # relabeled multi-hop drafts already fill slots
        for es in sample_evidence_sets(conn, qtype, n, seed):
            process(es)
    log.info("drafted %d, single-hop slots filled by relabels: %d", stats.drafted, stats.relabeled)
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
            filters = n * docs
            in_tok += filters * (filter_overhead + avg_tokens + 100)
            out_tok += filters * EST_FILTER_OUTPUT_TOKENS
    return (in_tok * price.input + out_tok * price.output) / 1_000_000 * COST_SAFETY_MARGIN

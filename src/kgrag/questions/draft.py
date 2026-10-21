"""Draft evaluation questions from sampled evidence, then filter the drafts."""

import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Literal

import psycopg
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from kgrag.config import get_settings
from kgrag.extract.llm import COST_SAFETY_MARGIN
from kgrag.extract.regex_ids import extract_ids
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

DRAFT_PROMPT_VERSION = "draft-v2"
MULTI_HOP: tuple[QuestionType, ...] = ("inference", "comparison", "temporal")
# Randomly sampled single-hop goes first so ID-linked relabels never displace it.
DRAFT_ORDER: tuple[QuestionType, ...] = ("single_hop", *MULTI_HOP, "null")
NULL_ANSWER = "The mailing list does not contain this information."
MAX_CONSECUTIVE_FAILURES = 5  # a systemic error (bad key, outage) must not burn the whole corpus
MAX_KEY_TERMS = 4
MIN_KEY_TERMS = 2
MAX_QUESTION_WORDS = 30
DATE_SHARE_CAP = 0.15  # at most this share of a run's kept drafts may have a date-centric answer
ATTEMPT_CAP_FACTOR = 3  # per type, at most this many drafting attempts per kept-draft target
NULL_CHECK_MAX_DOCS = 20  # candidate docs an LLM reads to confirm a null question is unanswered
NULL_CHECK_BATCH_TOKENS = 6000
AVG_GOLD_DOCS = {"single_hop": 1, "inference": 2.25, "comparison": 2, "temporal": 2.4, "null": 0}
# Leave-one-out calls: only 3-doc sets (a minority of inference/temporal) get one call per doc.
AVG_LOO_CALLS = {"inference": 0.75, "temporal": 1.2}
# Assumptions for the cost estimate: a draft is a short JSON object, a filter verdict is tiny.
EST_DRAFT_OUTPUT_TOKENS = 200
EST_FILTER_OUTPUT_TOKENS = 30
EST_NULL_CHECK_CALLS = 1.5
log = logging.getLogger(__name__)

AnswerKind = Literal["date", "entity", "reason", "yes_no", "other"]

SYSTEM_PROMPT = (
    "You write evaluation questions about the Apache Kafka developer mailing list (dev@).\n"
    "Rules for every question:\n"
    "1. Ask exactly ONE question with one short answer. Never join two asks with 'and' or "
    "'or', and never add a second clause asking for another fact (no 'who ... and when', no "
    "'which ticket ..., and was it a blocker', no 'how long ... and in which release').\n"
    "2. At most 25 words.\n"
    "3. Self-contained: identify the KIP, ticket, release or discussion so a reader knows what "
    "is meant, but never refer to 'the email', 'the document', 'the message' or 'the thread'.\n"
    "4. Do not copy sentences, example values or numbers, creators, or dates from the documents "
    "into the question; paraphrase only what is needed to identify the subject.\n"
    "5. Nothing in the question may name or imply the answer (no identifier, file name or "
    "list of options that contains it).\n"
    "6. Prefer substance: who proposed, objected to or decided something, what was decided and "
    "why, which approach, config, API or release. Ask about a date, a duration or an ordering "
    "only when the type instructions below explicitly allow it.\n"
    "Return: question; answer (short: an entity, a version, yes/no, or one sentence); "
    "answer_kind: 'date' if the answer is primarily a date, a duration or an ordering "
    "(which came first), 'entity' (a person, ticket, KIP, release, config, API, component), "
    "'reason' (why, what approach, what changed), 'yes_no', or 'other'; key_terms: 2-4 "
    "distinctive words or identifiers that a document stating the answer would contain, most "
    "specific first; viable: false when these documents cannot support a question that "
    "follows every rule (the other fields are then ignored)."
)
TYPE_INSTRUCTIONS: dict[str, str] = {
    "single_hop": (
        "Write a question that can be answered from the single document below, about a "
        "concrete fact it states: preferably a decision, proposal, objection, reason, "
        "behaviour or the person behind one. If the document is a Jira event (created, "
        "resolved or reopened), ask what the ticket is about, who created or resolved it, "
        "which ticket covers a described problem, or which release fixed it; never ask when "
        "it happened. Describe the subject in your own words instead of copying its title."
    ),
    "inference": (
        "Write a BRIDGE question that needs both kinds of document below. The first document "
        "mentions an entity (a KIP or ticket) inside a different discussion; the other "
        "document(s) are about that entity itself. Identify the entity ONLY through what the "
        "first document says about it (for example 'the ticket Luke Chen said blocked the "
        "3.7.0 release candidate' or 'the KIP Jun Rao cited as already propagating the high "
        "watermark'), never by its ID or title, and ask for ONE fact that only the other "
        "document(s) state (who proposed or objected, what it changed, why, which release "
        "fixed it). Prefer facts from discussion: decisions, objections, reasons, who said, "
        "proposed or objected. A question that joins two separate facts with 'and' is NOT "
        "acceptable. Set viable=false if the first document says nothing distinctive about "
        "the entity, or if every document states the same thing about it."
    ),
    "comparison": (
        "The two documents are about two RELATED items. Write ONE question that compares them "
        "on substance (scope, approach, outcome, who supported or objected, which release "
        "shipped each, whether they overlap or conflict) and needs BOTH documents to answer. "
        "Name both items. Do NOT ask which came first and do not compare dates. Set "
        "viable=false if dates are the only dimension the documents share."
    ),
    "temporal": (
        "The documents are about one KIP, ticket or discussion at different dates. Write ONE "
        "question about what changed between the earlier and the later document: how the "
        "design evolved, a decision or position that was reversed, or a status change and its "
        "reason; it must need EVERY document to answer. Never ask which came first. Only if "
        "nothing changed in substance may you ask how long something took (answer_kind "
        "'date'). Set viable=false if neither works."
    ),
    "null": (
        "You are given only the topic of a discussion. Write a question that is plausible "
        "for this mailing list and related to the topic, but whose answer is NOT stated "
        "anywhere on the list (an open detail nobody gave: a value, a release, a rationale). "
        "Every named entity, KIP, ticket, config and person in the question must appear in the "
        "topic context below, and belong together there; do not import configs, KIPs or claims "
        "from elsewhere. Do not assume anything was decided, chosen, assigned, scheduled or "
        "benchmarked unless the topic context says so; the question must not have a false "
        "premise. Avoid details a later reply in the same discussion would likely settle. "
        f"The answer field is ignored: set it to '{NULL_ANSWER}'. Make the key_terms the words "
        "a document answering the question would have to contain, most specific first, with "
        "identifiers spelled as in the topic context."
    ),
}
SINGLE_ASK_PROMPT = (
    "Decide whether the question below asks for exactly ONE thing (a single ask with one "
    "answer). Answer false if it asks for two or more things, joined by 'and', 'or', a comma "
    "or a second clause (for example 'who ... and when', 'which ticket ... and was it ...', "
    "'how long ... and which release'). A descriptive clause that only identifies the subject "
    "does not count as a second ask."
)
OTHERS_PROMPT = (
    "Decide whether the documents below alone answer the core of the question: the main fact "
    "it asks for. They are only some of the evidence. Answer true if a reader with only these "
    "documents could give essentially the reference answer, even if a minor detail, a "
    "qualifier or the exact wording is missing."
)
FILTER_PROMPT = (
    "Decide whether the single document below, alone, answers the core of the question: the "
    "main fact it asks for. Answer true if a reader with only this document could give "
    "essentially the reference answer, even if a minor detail, a qualifier or the exact "
    "wording is missing; false only if another source is genuinely needed."
)
NULL_CHECK_PROMPT = (
    "The question below is meant to be unanswerable from the Apache Kafka dev mailing list. "
    "Decide whether ANY of the documents below answers it, even partially or approximately "
    "(states the value, release, decision, person or reason asked for), or shows that the "
    "question's premise is false. Answer true if so."
)


class DraftSchema(BaseModel):
    """Structured output requested from the drafting model."""

    question: str  # may be empty when viable is false
    answer: str
    answer_kind: AnswerKind
    key_terms: list[str] = Field(default_factory=list)
    viable: bool


class SingleSourceSchema(BaseModel):
    answerable: bool


class OtherDocsSchema(BaseModel):
    answerable: bool


class SingleAskSchema(BaseModel):
    single_ask: bool


class NullCheckSchema(BaseModel):
    answered: bool


class DraftQuestion(BaseModel):
    question: str
    answer: str
    type: QuestionType
    gold_doc_ids: list[str]
    key_terms: list[str] = Field(max_length=MAX_KEY_TERMS)
    answer_kind: AnswerKind = "other"
    viable: bool = True


@dataclass
class DraftStats:
    drafted: int = 0
    kept: int = 0
    relabeled: int = 0
    dropped_single_source: int = 0
    dropped_null_present: int = 0
    dropped_null_premise: int = 0
    dropped_shrunk: int = 0
    dropped_compound: int = 0
    dropped_too_long: int = 0
    dropped_date_cap: int = 0
    dropped_unviable: int = 0
    skipped_existing: int = 0
    failed: int = 0
    date_cap: int = 0
    kept_by_type: dict[str, int] = field(default_factory=dict)
    kept_by_origin: dict[str, int] = field(default_factory=dict)
    kept_by_answer_kind: dict[str, int] = field(default_factory=dict)
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
        if es.hint:
            context = f"{es.hint}\n\n{context}"
    out = _invoke(
        llm, DraftSchema, f"{SYSTEM_PROMPT}\n\n{TYPE_INSTRUCTIONS[es.type]}", context
    )
    terms = [t.strip() for t in out.key_terms if t.strip()][:MAX_KEY_TERMS]
    if out.viable and not (out.question.strip() and out.answer.strip()):
        raise ValueError("viable draft with an empty question or answer")
    if out.viable and len(terms) < MIN_KEY_TERMS:
        raise ValueError(f"draft returned {len(terms)} key terms, need at least {MIN_KEY_TERMS}")
    return DraftQuestion(
        question=out.question.strip(),
        answer=NULL_ANSWER if es.type == "null" else out.answer.strip(),
        type=es.type, gold_doc_ids=list(es.doc_ids), key_terms=terms,
        answer_kind=out.answer_kind, viable=out.viable,
    )


def too_long(draft: DraftQuestion) -> bool:
    return len(draft.question.split()) > MAX_QUESTION_WORDS


def is_single_question(draft: DraftQuestion, llm: BaseChatModel) -> bool:
    """Ask the filter model whether the question is one ask (compound questions are rejected)."""
    return _invoke(llm, SingleAskSchema, SINGLE_ASK_PROMPT, f"Question: {draft.question}").single_ask


def single_source_doc(draft: DraftQuestion, docs: list[Document], llm: BaseChatModel) -> str | None:
    """The first gold doc that answers the core of the question on its own, or None.

    For a 2-doc set this is the leave-one-out check: either doc alone answering means the
    question does not need both.
    """
    for doc in docs:
        user = (
            f"Document:\n\n{_format_doc(doc)}\n\nQuestion: {draft.question}\n"
            f"Reference answer: {draft.answer}"
        )
        if _invoke(llm, SingleSourceSchema, FILTER_PROMPT, user).answerable:
            return doc.doc_id
    return None


def leave_one_out(draft: DraftQuestion, docs: list[Document], llm: BaseChatModel) -> list[Document]:
    """For a 3-doc set, drop the first doc the other two can answer without; else keep all.

    2-doc sets are covered by `single_source_doc` (leaving one out leaves a single doc).
    """
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


# --- null questions ----------------------------------------------------------------------

_SEP = "[-_. ]?"


def term_pattern(term: str) -> str | None:
    """Regex body matching a term's surface variants: case, -/_/./space or no separator,
    camelCase split, singular/plural (KIP-1087 ~ kip 1087; intermediate-topics ~
    intermediate_topic ~ intermediateTopics). Same syntax in Python and Postgres."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", term)
    parts = re.findall(r"[a-z0-9]+", spaced.lower())
    if not parts:
        return None
    out = []
    for p in parts:
        if p.isdigit():
            out.append(p)
        elif len(p) > 3 and p.endswith("s"):
            out.append(p[:-1] + "s?")
        else:
            out.append(p + "s?")
    return _SEP.join(out)


def _py(body: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])", re.IGNORECASE)


def _pg(body: str) -> str:
    return rf"\y{body}\y"


def _term_bodies(terms: list[str]) -> list[str]:
    return [b for b in (term_pattern(t) for t in terms) if b]


def null_question_absent(draft: DraftQuestion, conn: psycopg.Connection) -> bool:
    """True when no single document contains ALL the key terms, by full-text search or by
    surface-variant regex."""
    if not draft.key_terms:
        return False  # nothing to verify absence with
    cond = " and ".join(
        "to_tsvector('english', text) @@ plainto_tsquery('english', %s)" for _ in draft.key_terms
    )
    if conn.execute(f"select 1 from documents where {cond} limit 1", draft.key_terms).fetchone():
        return False
    bodies = _term_bodies(draft.key_terms)
    if len(bodies) == len(draft.key_terms):
        cond = " and ".join("text ~* %s" for _ in bodies)
        if conn.execute(
            f"select 1 from documents where {cond} limit 1", [_pg(b) for b in bodies]
        ).fetchone():
            return False
    return True


_IDENT = re.compile(
    r"--[a-z][\w-]+|\b[a-z][a-z0-9]+(?:[._][a-z0-9]+)+\b|\b[a-z]+[A-Z][A-Za-z0-9]*\b"
    r"|\b[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*\b"
)


def foreign_identifiers(question: str, topic: str) -> list[str]:
    """KIPs, tickets, releases and config/API identifiers named in a null question but absent
    from its topic context (an imported premise)."""
    topic_ids = {h.id for h in extract_ids(topic)}
    out = [h.id for h in extract_ids(question) if h.id not in topic_ids]
    for ident in _IDENT.findall(question):
        body = term_pattern(ident)
        if body and not _py(body).search(topic):
            out.append(ident)
    return sorted(set(out))


def null_candidates(
    conn: psycopg.Connection, draft: DraftQuestion, thread_id: str | None
) -> list[Document]:
    """Docs that could answer a null question: those with its most specific key term (any
    surface variant) alone, those naming its KIPs / tickets, and its whole linked thread;
    ranked by how many key terms they contain (thread docs break ties), capped."""
    bodies = _term_bodies(draft.key_terms)
    counts = [
        conn.execute("select count(*) from documents where text ~* %s", (_pg(b),)).fetchone()[0]
        for b in bodies
    ]
    present = [(n, b) for n, b in zip(counts, bodies, strict=True) if n]
    id_bodies = [
        term_pattern(h.id) for h in extract_ids(draft.question) if h.kind in ("KIP", "JiraTicket")
    ]
    conds, params = [], []
    if present:
        conds.append("d.text ~* %s")
        params.append(_pg(min(present)[1]))
    for b in id_bodies:
        conds.append("d.text ~* %s")
        params.append(_pg(b))
    if thread_id:
        conds.append("e.thread_id = %s")
        params.append(thread_id)
    if not conds:
        return []
    rows = conn.execute(
        "select d.doc_id, d.text, e.thread_id from documents d left join emails e on "
        f"d.source_kind = 'email' and e.id = d.source_id where {' or '.join(conds)}",
        params,
    ).fetchall()
    term_pats = [_py(b) for b in bodies]
    id_pats = [_py(b) for b in id_bodies]
    scored = []
    for doc_id, text, thread in rows:
        score = sum(1 for p in term_pats if p.search(text))
        score += any(p.search(text) for p in id_pats)
        score += 0.5 * (thread_id is not None and thread == thread_id)
        if score > 0:  # every doc of the linked thread; elsewhere only docs with a match
            scored.append((-score, doc_id))
    top = [doc_id for _, doc_id in sorted(scored)[:NULL_CHECK_MAX_DOCS]]
    return get_documents(conn, top)


def null_answered(draft: DraftQuestion, docs: list[Document], llm: BaseChatModel) -> bool:
    """Ask the filter model, in token-bounded batches, whether any candidate doc answers."""
    batch: list[str] = []
    batches: list[list[str]] = []
    used = 0
    for d in docs:
        text = _format_doc(d)
        n = count_tokens(text)
        if batch and used + n > NULL_CHECK_BATCH_TOKENS:
            batches.append(batch)
            batch, used = [], 0
        batch.append(text)
        used += n
    if batch:
        batches.append(batch)
    for b in batches:
        user = "Documents:\n\n" + "\n\n".join(b) + f"\n\nQuestion: {draft.question}"
        if _invoke(llm, NullCheckSchema, NULL_CHECK_PROMPT, user).answered:
            return True
    return False


def _question_id(d: DraftQuestion) -> str:
    raw = d.type + "".join(sorted(d.gold_doc_ids)) + d.question
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def attempt_cap(target: int) -> int:
    return ATTEMPT_CAP_FACTOR * target


def _already_attempted(conn: psycopg.Connection, es: EvidenceSet) -> bool:
    """True if this exact set was drafted before; failed attempts are retried."""
    row = conn.execute(
        "select 1 from evidence_attempts where type = %s and evidence_doc_ids = %s "
        "and link = %s and outcome <> 'failed' limit 1",
        (es.type, sorted(es.doc_ids), es.link),
    ).fetchone()
    return row is not None


def _record_attempt(
    conn: psycopg.Connection, es: EvidenceSet, outcome: str, question_id: str | None
) -> None:
    conn.execute(
        "insert into evidence_attempts (type, link, evidence_doc_ids, outcome, question_id) "
        "values (%s, %s, %s, %s, %s) on conflict (type, evidence_doc_ids, link) do update "
        "set outcome = excluded.outcome, question_id = excluded.question_id, "
        "attempted_at = now()",
        (es.type, es.link, sorted(es.doc_ids), outcome, question_id),
    )


def date_cap(targets: dict[str, int]) -> int:
    """Most date-centric drafts one run may keep: DATE_SHARE_CAP of its total target."""
    return math.floor(DATE_SHARE_CAP * sum(targets.values()))


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

    Every draft must be viable, at most MAX_QUESTION_WORDS words and a single ask; once the
    run has kept its date cap of date-centric drafts, further date answers are dropped and the
    type tops up from other sets.
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
    stats.date_cap = date_cap(targets)

    def store(d: DraftQuestion, es: EvidenceSet, origin: str) -> str:
        row = QuestionRow(
            id=_question_id(d), type=d.type, question=d.question, answer=d.answer,
            gold_doc_ids=d.gold_doc_ids, status="draft", draft_model=model,
            prompt_version=DRAFT_PROMPT_VERSION, link=es.link, origin=origin,
            key_terms=d.key_terms, answer_kind=d.answer_kind,
        )
        if insert_question(conn, row):
            stats.kept += 1
            stats.kept_by_type[d.type] = stats.kept_by_type.get(d.type, 0) + 1
            stats.kept_by_origin[origin] = stats.kept_by_origin.get(origin, 0) + 1
            stats.kept_by_answer_kind[d.answer_kind] = (
                stats.kept_by_answer_kind.get(d.answer_kind, 0) + 1
            )
        return row.id

    def drop(outcome: str) -> tuple[bool, str, None]:
        setattr(stats, outcome, getattr(stats, outcome) + 1)
        return False, outcome, None

    def process(es: EvidenceSet) -> tuple[bool, str, str | None]:
        """Draft and filter one set: (counts toward its type's target, outcome, question id)."""
        nonlocal single_kept
        docs = get_documents(conn, es.doc_ids)
        if len(docs) != len(es.doc_ids):
            raise ValueError(f"missing documents for {es.doc_ids}")
        topic = thread = None
        if es.type == "null":
            thread = es.link.removeprefix("thread:")
            topic = thread_topic(conn, thread)
        d = draft_question(es, docs, llm, topic=topic)
        stats.drafted += 1
        if not d.viable:
            return drop("dropped_unviable")
        if too_long(d):
            return drop("dropped_too_long")
        if d.answer_kind == "date" and stats.kept_by_answer_kind.get("date", 0) >= stats.date_cap:
            return drop("dropped_date_cap")
        if not is_single_question(d, llm):
            return drop("dropped_compound")
        if es.type == "single_hop":
            single_kept += 1
            return True, "kept", store(d, es, "sampled")
        if es.type == "null":
            if foreign_identifiers(d.question, topic or ""):
                return drop("dropped_null_premise")
            if not null_question_absent(d, conn) or null_answered(
                d, null_candidates(conn, d, thread), llm
            ):
                return drop("dropped_null_present")
            return True, "kept", store(d, es, "sampled")
        alone = single_source_doc(d, docs, llm)
        if alone is not None:
            if single_kept >= single_target:
                return drop("dropped_single_source")
            stats.relabeled += 1
            single_kept += 1
            qid = store(d.model_copy(update={"type": "single_hop", "gold_doc_ids": [alone]}),
                        es, "relabeled")
            return False, "relabeled", qid
        kept_docs = leave_one_out(d, docs, llm)
        outcome = "kept"
        if len(kept_docs) < len(docs):
            dates = [x.date for x in kept_docs]
            if es.type == "temporal" and max(dates) - min(dates) < MIN_TEMPORAL_GAP:
                return drop("dropped_shrunk")
            d = d.model_copy(update={"gold_doc_ids": [x.doc_id for x in kept_docs]})
            outcome = "shrunk"
        return True, outcome, store(d, es, "sampled")

    for qtype in DRAFT_ORDER:
        target = targets.get(qtype)
        if target is None:
            continue
        kept = failures = attempts = 0
        cap = attempt_cap(target)
        for es in iter_evidence_sets(conn, qtype, seed):
            if kept >= target or failures >= MAX_CONSECUTIVE_FAILURES or attempts >= cap:
                break
            if _already_attempted(conn, es):
                stats.skipped_existing += 1
                continue
            attempts += 1
            try:
                counted, outcome, qid = process(es)
                _record_attempt(conn, es, outcome, qid)
                conn.commit()
                failures = 0
            except Exception as exc:  # noqa: BLE001 - one bad draft must not lose the others
                conn.rollback()
                log_failure(conn, "draft", es.link, f"{type(exc).__name__}: {exc}")
                _record_attempt(conn, es, "failed", None)
                conn.commit()
                stats.failed += 1
                failures += 1
                continue
            kept += counted
        if kept < target:
            stats.shortfall[qtype] = target - kept
            log.warning("%s: kept %d of %d drafts after %d attempts (cap %d)",
                        qtype, kept, target, attempts, cap)
    return stats


def avg_doc_tokens(conn: psycopg.Connection) -> int:
    row = conn.execute("select avg(token_count) from documents").fetchone()
    return round(row[0] or 0)


def estimate_draft_cost_usd(
    counts: dict[QuestionType, int], avg_tokens: int, model: str, *, oversample: float = 1.3,
    worst_case: bool = False,
) -> float:
    """Draft call and single-ask check per set, filter calls per multi-hop set, candidate
    checks per null set.

    Expected = one attempt per target draft; worst case = every type uses its full attempt cap.
    """
    price = get_settings().pricing[model]
    longest = max(TYPE_INSTRUCTIONS.values(), key=len)
    draft_overhead = count_tokens(
        SYSTEM_PROMPT + longest + json.dumps(DraftSchema.model_json_schema())
    )
    filter_overhead = count_tokens(
        FILTER_PROMPT + json.dumps(SingleSourceSchema.model_json_schema())
    )
    ask_tokens = count_tokens(SINGLE_ASK_PROMPT + json.dumps(SingleAskSchema.model_json_schema()))
    ask_tokens += 2 * MAX_QUESTION_WORDS
    null_overhead = count_tokens(NULL_CHECK_PROMPT + json.dumps(NullCheckSchema.model_json_schema()))
    null_docs = min(NULL_CHECK_MAX_DOCS * avg_tokens, NULL_CHECK_BATCH_TOKENS)
    in_tok = out_tok = 0.0
    for qtype, count in counts.items():
        n = math.ceil(count * oversample)
        if worst_case:
            n = attempt_cap(n)
        docs = AVG_GOLD_DOCS[qtype]
        in_tok += n * (draft_overhead + max(docs, 1) * avg_tokens + ask_tokens)
        out_tok += n * (EST_DRAFT_OUTPUT_TOKENS + EST_FILTER_OUTPUT_TOKENS)
        if qtype in MULTI_HOP:
            # one single-doc call per gold doc, plus leave-one-out calls on 3-doc sets
            filters = n * docs
            in_tok += filters * (filter_overhead + avg_tokens + 100)
            loo = n * AVG_LOO_CALLS.get(qtype, 0)
            in_tok += loo * (filter_overhead + 2 * avg_tokens + 100)
            out_tok += (filters + loo) * EST_FILTER_OUTPUT_TOKENS
        if qtype == "null":
            calls = n * EST_NULL_CHECK_CALLS
            in_tok += calls * (null_overhead + null_docs + 100)
            out_tok += calls * EST_FILTER_OUTPUT_TOKENS
    return (in_tok * price.input + out_tok * price.output) / 1_000_000 * COST_SAFETY_MARGIN

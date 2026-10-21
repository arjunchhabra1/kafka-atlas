"""Evidence sampling for the evaluation question set.

Reads Postgres only (documents, emails, threads, jira_events, regex mentions); it must never
touch the knowledge graph, so the question set is not biased toward the graph system.
"""

import random
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import islice
from typing import Literal

import psycopg
from pydantic import BaseModel

from kgrag.extract.regex_ids import extract_ids

QuestionType = Literal["single_hop", "inference", "comparison", "temporal", "null"]

MIN_DOC_TOKENS = 40
HUMAN_SHARE = 0.85  # single-hop: share of human email chunks vs Jira docs
MIN_TEMPORAL_GAP = timedelta(days=7)
LINKABLE_KINDS = ("KIP", "JiraTicket")  # releases are too generic to link documents
MAX_PASSES = 50  # safety cap on shuffled passes over the candidate keys
STALE_LIMIT = 2  # consecutive passes with nothing new
# A chunk is mostly machine output when this share of its non-blank lines are log/trace lines.
NOISY_LINE = re.compile(
    r"\[stack trace|\[log:|ducktape|^\s*at [\w.$<>]+\(|^\s*Caused by:", re.IGNORECASE
)
NOISY_SHARE = 0.3
BRIDGE_THREE_DOCS = 0.25  # inference: share of bridge sets with a second "about" doc
KIP_SHARE = 0.75  # comparison / temporal: share of sets built around KIPs rather than tickets
MAX_WORD_BUCKET = 40  # title words shared by more items than this are too generic to relate them
MAX_COMENTION = 4  # a doc naming more ids of one kind than this does not relate them
MIN_SHARED_WORDS = 2  # titles must share this many distinctive words (1 with a common release)
_TITLE_NOISE = re.compile(
    r"\b(re|fwd?|aw)\s*:|\[(discuss|vote|result|results|announce|jira|created|resolved|reopened)"
    r"\]|\bkip[- ]?\d+\s*:?|\bkafka-\d+", re.IGNORECASE,
)
_STOP = frozenset(
    "about after also allow allows apache before being between broker brokers change changes "
    "class could does doesn during fail failed failing fails failure fix fixes fixed flaky from "
    "have improve improvement into issue kafka make method more need needs only other should "
    "some support supports test tests testing than that their them then there these they this "
    "thread unable update upon used using when where which while will with without would discuss "
    "vote result results proposal".split()
)


class EvidenceSet(BaseModel):
    type: QuestionType
    doc_ids: list[str]
    link: str
    hint: str = ""  # how the docs relate, shown to the drafting model (never stored)


@dataclass(frozen=True)
class _Doc:
    doc_id: str
    kind: str  # 'email' | 'jira'
    source_id: str
    thread: str  # all events of one Jira ticket count as a single thread
    date: datetime
    ids: frozenset[str]  # canonical KIP / ticket ids
    subject_ids: frozenset[str] = frozenset()  # ids in the subject: the doc is ABOUT them
    title: str = ""  # subject, used to relate items by topic


def is_noisy(text: str) -> bool:
    """True for chunks that are mostly logs, stack traces or ducktape errors."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True
    return sum(1 for ln in lines if NOISY_LINE.search(ln)) / len(lines) >= NOISY_SHARE


def _ids_in(text: str) -> frozenset[str]:
    return frozenset(h.id for h in extract_ids(text) if h.kind in LINKABLE_KINDS)


def _id_kind(ident: str) -> str:
    return "KIP" if ident.startswith("KIP-") else "JiraTicket"


def _canonical_id(surface: str) -> str | None:
    for hit in extract_ids(surface):
        if hit.kind in LINKABLE_KINDS:
            return hit.id
    return None


def _load_docs(conn: psycopg.Connection) -> list[_Doc]:
    """Substantial, non-log documents with their canonical ids, ordered by doc_id."""
    mention_ids: dict[str, dict[str, set[str]]] = {}  # email -> id -> surfaces
    for email_id, surface in conn.execute(
        "select email_id, surface from mentions where source = 'regex' "
        "and type in ('KIP', 'JiraTicket') order by mention_id"
    ):
        ident = _canonical_id(surface)
        if ident:
            mention_ids.setdefault(email_id, {}).setdefault(ident, set()).add(surface.lower())
    tickets = {
        str(i): _canonical_id(t) for i, t in conn.execute(
            "select id, ticket from jira_events order by id"
        ) if t
    }
    rows = conn.execute(
        "select d.doc_id, d.source_kind, d.source_id, d.chunk_index, d.text, d.token_count, "
        "d.date, e.thread_id, d.subject from documents d "
        "left join emails e on d.source_kind = 'email' and e.id = d.source_id "
        "order by d.doc_id"
    ).fetchall()

    # An email's id belongs to every chunk containing its surface, else to chunk 0.
    by_email: dict[str, list[tuple]] = {}
    for row in rows:
        if row[1] == "email":
            by_email.setdefault(row[2], []).append(row)
    chunk_ids: dict[str, set[str]] = {}
    for email_id, chunks in by_email.items():
        first = min(chunks, key=lambda r: r[3])
        for ident, surfaces in mention_ids.get(email_id, {}).items():
            pats = [re.compile(rf"(?<!\w){re.escape(s)}(?!\w)", re.I) for s in surfaces]
            hits = [c for c in chunks if any(p.search(c[4]) for p in pats)] or [first]
            for c in hits:
                chunk_ids.setdefault(c[0], set()).add(ident)

    docs = []
    for doc_id, kind, source_id, _chunk, text, tokens, date, thread, subject in rows:
        if tokens < MIN_DOC_TOKENS or is_noisy(text):
            continue
        subject = subject or ""
        if kind == "jira":
            ticket = tickets.get(source_id)
            ids = frozenset({ticket} if ticket else ())
            thread = f"jira:{ticket or doc_id}"
            about = ids
        else:
            ids = frozenset(chunk_ids.get(doc_id, ()))
            about = _ids_in(subject)
        docs.append(_Doc(doc_id, kind, source_id, thread, date, ids | about, about, subject))
    return docs


def _ordered(docs: list[_Doc]) -> list[str]:
    return [d.doc_id for d in sorted(docs, key=lambda d: (d.date, d.doc_id))]


def _rng(seed: int, qtype: str) -> random.Random:
    """Independent, deterministic random stream per (seed, question type)."""
    return random.Random(f"{seed}:{qtype}")


def _collect(keys: list[str], build, rng: random.Random) -> Iterator[EvidenceSet]:
    """Yield distinct sets from shuffled passes over keys until passes stop producing new ones."""
    seen: set[frozenset[str]] = set()
    stale = 0
    for _ in range(MAX_PASSES):
        order = list(keys)
        rng.shuffle(order)
        fresh = False
        for key in order:
            es = build(key)
            if es is not None and frozenset(es.doc_ids) not in seen:
                seen.add(frozenset(es.doc_ids))
                fresh = True
                yield es
        stale = 0 if fresh else stale + 1
        if stale >= STALE_LIMIT:
            return


def _single_hop(docs: list[_Doc], rng: random.Random) -> Iterator[EvidenceSet]:
    pools = {
        "email": [d for d in docs if d.kind == "email"],
        "jira": [d for d in docs if d.kind == "jira"],
    }
    while pools["email"] or pools["jira"]:
        want = "email" if rng.random() < HUMAN_SHARE else "jira"
        pool = pools[want] or pools["email" if want == "jira" else "jira"]
        d = pool.pop(rng.randrange(len(pool)))
        yield EvidenceSet(type="single_hop", doc_ids=[d.doc_id], link=f"doc:{d.doc_id}")


def _mix(rng: random.Random, streams: list[tuple[Iterator[EvidenceSet], float]]):
    """Interleave streams, drawing from each with its weight until all are exhausted."""
    live = [(it, w) for it, w in streams if w > 0]
    while live:
        pick = rng.random() * sum(w for _, w in live)
        for i, (it, w) in enumerate(live):
            pick -= w
            if pick < 0 or i == len(live) - 1:
                break
        es = next(it, None)
        if es is None:
            live.pop(i)
        else:
            yield es


def _about(docs: list[_Doc]) -> dict[str, list[_Doc]]:
    """Docs grouped by the ids they are about (id in the subject or the Jira ticket)."""
    out: dict[str, list[_Doc]] = {}
    for d in docs:
        for ident in sorted(d.subject_ids):
            out.setdefault(ident, []).append(d)
    return out


def _prefer_email(pool: list[_Doc], rng: random.Random, share: float = 0.8) -> list[_Doc]:
    """Shuffled pool, with human emails first with probability `share`."""
    pool = list(pool)
    rng.shuffle(pool)
    if rng.random() < share:
        pool.sort(key=lambda d: d.kind != "email")
    return pool


def _is_root(d: _Doc) -> bool:
    """First chunk of a thread-starting email (states a proposal's scope and approach)."""
    return (d.kind == "email" and d.doc_id.endswith(":0")
            and re.match(r"\s*(re|aw|fwd?)\s*:", d.title, re.IGNORECASE) is None)


def _prefer_root(pool: list[_Doc], rng: random.Random, share: float = 0.6) -> list[_Doc]:
    pool = _prefer_email(pool, rng)
    if rng.random() < share:
        pool.sort(key=lambda d: not _is_root(d))
    return pool


def _inference(docs: list[_Doc], rng: random.Random) -> Iterator[EvidenceSet]:
    """Bridge sets: a doc that mentions an id inside another discussion, plus doc(s) about it.

    The first doc identifies the bridge entity through its context; the other(s) state facts
    about it. Sets where every doc is about the same id (same content restated) never occur.
    """
    about = _about(docs)
    citing: dict[str, list[_Doc]] = {}
    for d in docs:
        for ident in sorted(d.ids - d.subject_ids):
            if ident in about:
                citing.setdefault(ident, []).append(d)

    def build(ident: str) -> EvidenceSet | None:
        for a in _prefer_email(citing[ident], rng):
            want = 2 if rng.random() >= BRIDGE_THREE_DOCS else 3
            picked = [a]
            for b in _prefer_email(about[ident], rng, 0.7):
                if len(picked) < want and b.thread not in {p.thread for p in picked}:
                    picked.append(b)
            if len(picked) >= 2:
                return EvidenceSet(
                    type="inference", doc_ids=[a.doc_id, *_ordered(picked[1:])],
                    link=f"bridge:{ident}",
                    hint=f"The first document mentions {ident} inside a different discussion "
                         f"(subject: {a.title!r}); the other document(s) are about {ident} itself.",
                )
        return None

    return _collect(sorted(citing), build, rng)


def _split_kinds(keys) -> tuple[list[str], list[str]]:
    keys = sorted(keys)
    return [k for k in keys if _id_kind(k) == "KIP"], [k for k in keys if _id_kind(k) != "KIP"]


def _temporal(docs: list[_Doc], rng: random.Random) -> Iterator[EvidenceSet]:
    """Docs about one KIP / ticket / discussion, at least a week apart (KIPs preferred)."""
    groups = _about(docs)

    def build(ident: str) -> EvidenceSet | None:
        pool = _prefer_email(groups[ident], rng)
        for a in pool:
            for b in pool:
                if (a.kind, a.source_id) != (b.kind, b.source_id) and (
                    b.date - a.date >= MIN_TEMPORAL_GAP
                ):
                    picked = [a, b]
                    if rng.random() < 0.4:
                        mid = [
                            m for m in pool
                            if a.date <= m.date <= b.date
                            and (m.kind, m.source_id) not in {(p.kind, p.source_id) for p in picked}
                        ]
                        if mid:
                            picked.append(rng.choice(mid))
                    return EvidenceSet(
                        type="temporal", doc_ids=_ordered(picked), link=f"temporal:{ident}",
                        hint=f"Every document is about {ident}; they are from different dates.",
                    )
        return None

    kips, tickets = _split_kinds(groups)
    return _mix(rng, [(_collect(kips, build, rng), KIP_SHARE),
                      (_collect(tickets, build, rng), 1 - KIP_SHARE)])


def _title_words(title: str) -> set[str]:
    title = _TITLE_NOISE.sub(" ", title)
    title = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", title)  # split camelCase
    return {w for w in re.findall(r"[a-z][a-z0-9]{3,}", title.lower()) if w not in _STOP}


def related_pairs(
    docs: list[_Doc], releases: dict[str, str] | None = None
) -> dict[tuple[str, str], str]:
    """Pairs of same-kind items that are related, with the reason.

    Related means: mentioned together in one document, or titles sharing distinctive words
    (topic / component), or fixed in the same release with a shared title word. Uses only
    headers, regex ids and Jira fields.
    """
    releases = releases or {}
    about = _about(docs)
    pairs: dict[tuple[str, str], str] = {}
    for d in docs:
        for kind in LINKABLE_KINDS:
            ids = sorted(i for i in d.ids if _id_kind(i) == kind and i in about)
            if len(ids) > MAX_COMENTION:
                continue  # release notes and digests list many unrelated items
            for n, a in enumerate(ids):
                for b in ids[n + 1:]:
                    pairs.setdefault((a, b), "mentioned together in one message")
    words = {i: set().union(*(_title_words(d.title) for d in ds)) for i, ds in about.items()}
    buckets: dict[str, list[str]] = {}
    for ident in sorted(words):
        for w in words[ident]:
            buckets.setdefault(w, []).append(ident)
    shared: dict[tuple[str, str], set[str]] = {}
    for w, items in buckets.items():
        if len(items) > MAX_WORD_BUCKET:
            continue
        for i, a in enumerate(items):
            for b in items[i + 1:]:
                if _id_kind(a) == _id_kind(b):
                    shared.setdefault((min(a, b), max(a, b)), set()).add(w)
    for (a, b), ws in sorted(shared.items()):
        same_release = releases.get(a) is not None and releases.get(a) == releases.get(b)
        if len(ws) >= MIN_SHARED_WORDS:
            pairs.setdefault((a, b), f"same topic: {', '.join(sorted(ws))}")
        elif same_release:
            pairs.setdefault((a, b), f"both fixed in {releases[a]}; shared topic: {min(ws)}")
    return pairs


def _comparison(
    docs: list[_Doc], rng: random.Random, releases: dict[str, str] | None = None
) -> Iterator[EvidenceSet]:
    """Pairs of RELATED items, one doc about each (never two unrelated items by date)."""
    about = _about(docs)
    pairs = related_pairs(docs, releases)

    def build(key: tuple[str, str]) -> EvidenceSet | None:
        a, b = key
        for da in _prefer_root(about[a], rng):
            for db in _prefer_root(about[b], rng):
                # each doc must concern exactly one side, so the comparison needs both
                if b in da.ids or a in db.ids:
                    continue
                if (da.kind, da.source_id) != (db.kind, db.source_id):
                    return EvidenceSet(
                        type="comparison", doc_ids=[da.doc_id, db.doc_id],
                        link=f"related:{a}|{b}",
                        hint=f"The documents are about two related items, {a} and {b} "
                             f"({pairs[key]}).",
                    )
        return None

    keys = sorted(pairs)
    kips = [k for k in keys if _id_kind(k[0]) == "KIP"]
    tickets = [k for k in keys if _id_kind(k[0]) != "KIP"]
    return _mix(rng, [(_collect(kips, build, rng), KIP_SHARE),
                      (_collect(tickets, build, rng), 1 - KIP_SHARE)])


def _releases(conn: psycopg.Connection) -> dict[str, str]:
    """Ticket -> fix version, from Jira resolution events."""
    out: dict[str, str] = {}
    for ticket, version in conn.execute(
        "select ticket, fix_version from jira_events where fix_version is not null "
        "and ticket is not null order by id"
    ):
        ident = _canonical_id(ticket)
        if ident:
            out[ident] = version
    return out


def _null(conn: psycopg.Connection, rng: random.Random) -> Iterator[EvidenceSet]:
    threads = [r[0] for r in conn.execute(
        "select distinct e.thread_id from emails e join documents d "
        "on d.source_kind = 'email' and d.source_id = e.id where e.thread_id is not null "
        "order by e.thread_id"
    )]
    rng.shuffle(threads)
    for t in threads:
        yield EvidenceSet(type="null", doc_ids=[], link=f"thread:{t}")


def iter_evidence_sets(
    conn: psycopg.Connection, qtype: QuestionType, seed: int
) -> Iterator[EvidenceSet]:
    """Lazily yield distinct evidence sets until the corpus can build no more.

    Deterministic for a corpus, seed and type; each type has its own random stream.
    """
    rng = _rng(seed, qtype)
    if qtype == "null":
        return _null(conn, rng)
    docs = _load_docs(conn)
    if qtype == "comparison":
        return _comparison(docs, rng, _releases(conn))
    builders = {"single_hop": _single_hop, "inference": _inference, "temporal": _temporal}
    return builders[qtype](docs, rng)


def sample_evidence_sets(
    conn: psycopg.Connection, qtype: QuestionType, n: int, seed: int
) -> list[EvidenceSet]:
    """Up to n evidence sets of the given type (fewer if the corpus cannot build n)."""
    return list(islice(iter_evidence_sets(conn, qtype, seed), n))


def thread_topic(conn: psycopg.Connection, thread_id: str) -> str:
    """Subject plus first chunk of a thread: the only context a null question is drafted from."""
    subject = conn.execute(
        "select subject from threads where thread_id = %s", (thread_id,)
    ).fetchone()
    first = conn.execute(
        "select d.text from documents d join emails e on d.source_kind = 'email' "
        "and d.source_id = e.id where e.thread_id = %s and d.chunk_index = 0 "
        "order by e.date, e.id limit 1",
        (thread_id,),
    ).fetchone()
    return f"Subject: {subject[0] if subject else ''}\n\n{first[0] if first else ''}"

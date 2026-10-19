"""Evidence sampling for the evaluation question set.

Reads Postgres only (documents, emails, threads, jira_events, regex mentions); it must never
touch the knowledge graph, so the question set is not biased toward the graph system.
"""

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import psycopg
from pydantic import BaseModel

from kgrag.extract.regex_ids import extract_ids

QuestionType = Literal["single_hop", "inference", "comparison", "temporal", "null"]

MIN_DOC_TOKENS = 40
HUMAN_SHARE = 0.7  # single-hop: share of human email chunks vs Jira docs
MIN_TEMPORAL_GAP = timedelta(days=7)
LINKABLE_KINDS = ("KIP", "JiraTicket")  # releases are too generic to link documents


class EvidenceSet(BaseModel):
    type: QuestionType
    doc_ids: list[str]
    link: str


@dataclass(frozen=True)
class _Doc:
    doc_id: str
    kind: str  # 'email' | 'jira'
    source_id: str
    thread: str  # a Jira doc is its own thread
    date: datetime
    ids: frozenset[str]  # canonical KIP / ticket ids


def _id_kind(ident: str) -> str:
    return "KIP" if ident.startswith("KIP-") else "JiraTicket"


def _canonical_id(surface: str) -> str | None:
    for hit in extract_ids(surface):
        if hit.kind in LINKABLE_KINDS:
            return hit.id
    return None


def _load_docs(conn: psycopg.Connection) -> list[_Doc]:
    """Substantial documents with their canonical ids, ordered by doc_id."""
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
        "d.date, e.thread_id from documents d "
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
            hits = [c for c in chunks if any(s in c[4].lower() for s in surfaces)] or [first]
            for c in hits:
                chunk_ids.setdefault(c[0], set()).add(ident)

    docs = []
    for doc_id, kind, source_id, _chunk, _text, tokens, date, thread in rows:
        if tokens < MIN_DOC_TOKENS:
            continue
        if kind == "jira":
            ticket = tickets.get(source_id)
            ids = frozenset({ticket} if ticket else ())
            thread = f"jira:{doc_id}"
        else:
            ids = frozenset(chunk_ids.get(doc_id, ()))
        docs.append(_Doc(doc_id, kind, source_id, thread, date, ids))
    return docs


def _by_id(docs: list[_Doc]) -> dict[str, list[_Doc]]:
    out: dict[str, list[_Doc]] = {}
    for d in docs:
        for ident in sorted(d.ids):
            out.setdefault(ident, []).append(d)
    return out


def _ordered(docs: list[_Doc]) -> list[str]:
    return [d.doc_id for d in sorted(docs, key=lambda d: (d.date, d.doc_id))]


def _collect(keys: list[str], build, n: int, rng: random.Random) -> list[EvidenceSet]:
    """Try each key (shuffled) up to three passes, keeping distinct sets, until n are built."""
    out: list[EvidenceSet] = []
    seen: set[frozenset[str]] = set()
    for _ in range(3):
        order = list(keys)
        rng.shuffle(order)
        for key in order:
            if len(out) >= n:
                return out
            es = build(key)
            if es is not None and frozenset(es.doc_ids) not in seen:
                seen.add(frozenset(es.doc_ids))
                out.append(es)
    return out


def _single_hop(docs: list[_Doc], n: int, rng: random.Random) -> list[EvidenceSet]:
    pools = {
        "email": [d for d in docs if d.kind == "email"],
        "jira": [d for d in docs if d.kind == "jira"],
    }
    out = []
    while len(out) < n and (pools["email"] or pools["jira"]):
        want = "email" if rng.random() < HUMAN_SHARE else "jira"
        pool = pools[want] or pools["email" if want == "jira" else "jira"]
        d = pool.pop(rng.randrange(len(pool)))
        out.append(EvidenceSet(type="single_hop", doc_ids=[d.doc_id], link=f"doc:{d.doc_id}"))
    return out


def _inference(docs: list[_Doc], n: int, rng: random.Random) -> list[EvidenceSet]:
    groups = _by_id(docs)

    def build(ident: str) -> EvidenceSet | None:
        pool = list(groups[ident])
        rng.shuffle(pool)
        want = rng.choice((2, 3))
        picked: list[_Doc] = []
        for d in pool:
            if len(picked) < want and d.thread not in {p.thread for p in picked}:
                picked.append(d)
        if len(picked) < 2:
            return None
        return EvidenceSet(type="inference", doc_ids=_ordered(picked), link=f"shared:{ident}")

    return _collect(sorted(groups), build, n, rng)


def _temporal(docs: list[_Doc], n: int, rng: random.Random) -> list[EvidenceSet]:
    groups = _by_id(docs)

    def build(ident: str) -> EvidenceSet | None:
        pool = list(groups[ident])
        rng.shuffle(pool)
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
                        type="temporal", doc_ids=_ordered(picked), link=f"temporal:{ident}"
                    )
        return None

    return _collect(sorted(groups), build, n, rng)


def _comparison(docs: list[_Doc], n: int, rng: random.Random) -> list[EvidenceSet]:
    # Docs that mention exactly one id of their kind, grouped by that id.
    single: dict[str, dict[str, list[_Doc]]] = {k: {} for k in LINKABLE_KINDS}
    for d in docs:
        for kind in LINKABLE_KINDS:
            of_kind = [i for i in d.ids if _id_kind(i) == kind]
            if len(of_kind) == 1:
                single[kind].setdefault(of_kind[0], []).append(d)
    kinds = [k for k in LINKABLE_KINDS if len(single[k]) >= 2]
    out: list[EvidenceSet] = []
    seen: set[frozenset[str]] = set()
    for _ in range(n * 20):
        if len(out) >= n or not kinds:
            break
        kind = rng.choice(kinds)
        a, b = sorted(rng.sample(sorted(single[kind]), 2))
        da, db = rng.choice(single[kind][a]), rng.choice(single[kind][b])
        key = frozenset((da.doc_id, db.doc_id))
        if (da.kind, da.source_id) == (db.kind, db.source_id) or key in seen:
            continue
        seen.add(key)
        out.append(EvidenceSet(
            type="comparison", doc_ids=[da.doc_id, db.doc_id], link=f"compare:{a}|{b}"
        ))
    return out


def _null(conn: psycopg.Connection, n: int, rng: random.Random) -> list[EvidenceSet]:
    threads = [r[0] for r in conn.execute(
        "select distinct e.thread_id from emails e join documents d "
        "on d.source_kind = 'email' and d.source_id = e.id where e.thread_id is not null "
        "order by e.thread_id"
    )]
    rng.shuffle(threads)
    return [EvidenceSet(type="null", doc_ids=[], link=f"thread:{t}") for t in threads[:n]]


def sample_evidence_sets(
    conn: psycopg.Connection, qtype: QuestionType, n: int, seed: int
) -> list[EvidenceSet]:
    """Up to n evidence sets of the given type; deterministic for a corpus and seed."""
    rng = random.Random(seed)
    if qtype == "null":
        return _null(conn, n, rng)
    docs = _load_docs(conn)
    builders = {
        "single_hop": _single_hop, "inference": _inference, "temporal": _temporal,
        "comparison": _comparison,
    }
    return builders[qtype](docs, n, rng)


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

"""Postgres access for emails, Jira events and the retrievable `documents` table."""

import re
from datetime import UTC, datetime
from typing import Literal

import psycopg
from pydantic import BaseModel

from kgrag.extract.schema import AnchorHints, Mention
from kgrag.ingest.jira import JiraEvent, render_event
from kgrag.resolve.interface import Resolution
from kgrag.tokens import count_tokens, split_tokens

_PARAGRAPH = re.compile(r"\n[ \t]*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class EmailRow(BaseModel):
    id: str
    message_id: str
    thread_id: str
    date: datetime
    sender_name: str
    sender_address: str
    subject: str
    body_clean: str
    kind: Literal["human", "jira"]
    parent_message_id: str | None = None


class Document(BaseModel):
    doc_id: str
    source_kind: Literal["email", "jira"]
    source_id: str
    chunk_index: int
    text: str
    token_count: int
    date: datetime
    sender: str
    subject: str


class QuestionRow(BaseModel):
    """One row of the `questions` table."""

    id: str
    type: Literal["single_hop", "inference", "comparison", "temporal", "null"]
    question: str
    answer: str
    gold_doc_ids: list[str]
    status: Literal["draft", "kept", "edited", "dropped"]
    split: str | None = None
    draft_model: str | None = None
    prompt_version: str | None = None
    link: str | None = None
    origin: Literal["sampled", "relabeled"] | None = None
    key_terms: list[str] | None = None
    reviewed_at: datetime | None = None


def _pack(units: list[str], sep: str, max_tokens: int) -> list[str]:
    """Greedily join units with sep while the joined text stays within max_tokens."""
    out: list[str] = []
    cur = ""
    for u in units:
        cand = f"{cur}{sep}{u}" if cur else u
        if cur and count_tokens(cand) > max_tokens:
            out.append(cur)
            cur = u
        else:
            cur = cand
    if cur:
        out.append(cur)
    return out


def chunk_text(text: str, max_tokens: int) -> list[str]:
    """Split text into chunks of at most max_tokens tokens.

    Whole paragraphs are packed together; an oversized paragraph is split on
    sentence boundaries, and an oversized sentence at token boundaries.
    """
    units: list[str] = []
    for para in (p.strip() for p in _PARAGRAPH.split(text)):
        if not para:
            continue
        if count_tokens(para) <= max_tokens:
            units.append(para)
            continue
        sentences = []
        for s in (s.strip() for s in _SENTENCE.split(para)):
            if s:
                sentences.extend([s] if count_tokens(s) <= max_tokens else split_tokens(s, max_tokens))
        units.extend(_pack(sentences, " ", max_tokens))
    return _pack(units, "\n\n", max_tokens)


def upsert_threads(conn: psycopg.Connection, rows: list[tuple[str, str, str]]) -> int:
    """Upsert (thread_id, root_message_id, subject) rows."""
    with conn.cursor() as cur:
        cur.executemany(
            "insert into threads (thread_id, root_message_id, subject) values (%s, %s, %s) "
            "on conflict (thread_id) do update set root_message_id = excluded.root_message_id, "
            "subject = excluded.subject",
            rows,
        )
    return len(rows)


def upsert_emails(conn: psycopg.Connection, rows: list[EmailRow]) -> int:
    with conn.cursor() as cur:
        cur.executemany(
            "insert into emails (id, message_id, thread_id, date, sender_name, sender_address, "
            "subject, body_clean, kind, parent_message_id) values (%(id)s, %(message_id)s, %(thread_id)s, %(date)s, "
            "%(sender_name)s, %(sender_address)s, %(subject)s, %(body_clean)s, %(kind)s, %(parent_message_id)s) "
            "on conflict (id) do update set message_id = excluded.message_id, "
            "thread_id = excluded.thread_id, date = excluded.date, "
            "sender_name = excluded.sender_name, sender_address = excluded.sender_address, "
            "subject = excluded.subject, body_clean = excluded.body_clean, kind = excluded.kind, "
            "parent_message_id = coalesce(excluded.parent_message_id, emails.parent_message_id)",
            [r.model_dump() for r in rows],
        )
    return len(rows)


def insert_jira_events(conn: psycopg.Connection, events: list[JiraEvent]) -> int:
    """Upsert events (one per source email), keeping the serial id; returns rows written."""
    inserted = 0
    with conn.cursor() as cur:
        for e in events:
            cur.execute(
                "insert into jira_events (ticket, action, actor, date, status, fix_version, "
                "title, source_email_id) values (%s, %s, %s, %s, %s, %s, %s, %s) "
                "on conflict (source_email_id) do update set ticket = excluded.ticket, "
                "action = excluded.action, actor = excluded.actor, date = excluded.date, "
                "status = excluded.status, fix_version = excluded.fix_version, "
                "title = excluded.title",
                (e.ticket, e.action, e.actor, e.date, e.status, e.fix_version, e.title,
                 e.source_email_id),
            )
            inserted += cur.rowcount
    return inserted


def delete_stale_jira_events(
    conn: psycopg.Connection, email_ids: list[str], keep_email_ids: list[str]
) -> int:
    """Delete events whose source email is in email_ids but is not in keep_email_ids."""
    cur = conn.execute(
        "delete from jira_events where source_email_id = any(%s) "
        "and not (source_email_id = any(%s))",
        (email_ids, keep_email_ids),
    )
    return cur.rowcount


def rebuild_documents(conn: psycopg.Connection, chunk_tokens: int) -> int:
    """Delete and regenerate every document from emails (human) and jira_events."""
    docs: list[Document] = []
    with conn.cursor() as cur:
        cur.execute(
            "select id, date, sender_name, sender_address, subject, body_clean from emails "
            "where kind = 'human' order by id"
        )
        for eid, date, name, address, subject, body in cur.fetchall():
            for i, chunk in enumerate(chunk_text(body or "", chunk_tokens)):
                docs.append(Document(
                    doc_id=f"email:{eid}:{i}", source_kind="email", source_id=eid,
                    chunk_index=i, text=chunk, token_count=count_tokens(chunk), date=date,
                    sender=name or address, subject=subject or "",
                ))
        cur.execute(
            "select id, ticket, action, actor, date, status, fix_version, title, source_email_id "
            "from jira_events order by id"
        )
        for eid, ticket, action, actor, date, status, fix, title, src in cur.fetchall():
            event = JiraEvent(
                ticket=ticket, action=action, actor=actor, date=date.astimezone(UTC),
                status=status, fix_version=fix, source_email_id=src, title=title or "",
            )
            text = render_event(event)
            docs.append(Document(
                doc_id=f"jira:{eid}", source_kind="jira", source_id=str(eid), chunk_index=0,
                text=text, token_count=count_tokens(text), date=date, sender=actor,
                subject=f"{ticket} {title or ''}".strip(),
            ))
        cur.execute("delete from documents")
        cur.executemany(
            "insert into documents (doc_id, source_kind, source_id, chunk_index, text, "
            "token_count, date, sender, subject) values (%(doc_id)s, %(source_kind)s, "
            "%(source_id)s, %(chunk_index)s, %(text)s, %(token_count)s, %(date)s, %(sender)s, "
            "%(subject)s)",
            [d.model_dump() for d in docs],
        )
    return len(docs)


def get_documents(conn: psycopg.Connection, doc_ids: list[str]) -> list[Document]:
    """Fetch documents by id, in the order requested (missing ids are omitted)."""
    with conn.cursor() as cur:
        cur.execute(
            "select doc_id, source_kind, source_id, chunk_index, text, token_count, date, "
            "sender, subject from documents where doc_id = any(%s)",
            (doc_ids,),
        )
        by_id = {r[0]: Document(**dict(zip(Document.model_fields, r, strict=True)))
                 for r in cur.fetchall()}
    return [by_id[i] for i in doc_ids if i in by_id]


def first_document_text(conn: psycopg.Connection) -> str | None:
    """Text of the first document by doc_id (the fixture's recorded-embedding query probe)."""
    row = conn.execute("select text from documents order by doc_id limit 1").fetchone()
    conn.commit()
    return row[0] if row else None


def log_failure(conn: psycopg.Connection, stage: str, item_id: str, error: str) -> None:
    conn.execute(
        "insert into failures (stage, item_id, error) values (%s, %s, %s)",
        (stage, item_id, error),
    )


def load_mentions(conn: psycopg.Connection) -> list[Mention]:
    """All persisted mentions (every source), ordered by mention_id."""
    cur = conn.execute(
        "select mention_id, surface, type, email_id, context from mentions order by mention_id"
    )
    return [
        Mention(mention_id=r[0], surface=r[1], type=r[2], email_id=r[3], context=r[4])
        for r in cur.fetchall()
    ]


def load_anchor_hints(conn: psycopg.Connection) -> AnchorHints:
    pairs = conn.execute(
        "select distinct sender_name, sender_address from emails where kind = 'human' "
        "order by sender_name, sender_address"
    ).fetchall()
    actors = conn.execute(
        "select distinct actor from jira_events where actor is not null order by actor"
    ).fetchall()
    return AnchorHints(
        sender_pairs=[(n, a) for n, a in pairs], jira_actors=[r[0] for r in actors]
    )


def select_unextracted(
    conn: psycopg.Connection, prompt_version: str, model: str, limit: int | None = None
) -> list[EmailRow]:
    """Human emails with no llm_cache row for (prompt_version, model), by date then id."""
    sql = (
        "select id, message_id, thread_id, date, sender_name, sender_address, subject, "
        "body_clean, kind, parent_message_id from emails e where kind = 'human' and not exists ("
        "select 1 from llm_cache c where c.email_id = e.id and c.prompt_version = %s "
        "and c.model = %s) order by date, id"
    )
    params: list = [prompt_version, model]
    if limit is not None:
        sql += " limit %s"
        params.append(limit)
    cur = conn.execute(sql, params)
    return [EmailRow(**dict(zip(EmailRow.model_fields, r, strict=True))) for r in cur.fetchall()]


def insert_question(conn: psycopg.Connection, q: QuestionRow) -> bool:
    """Insert a question; False when its id already exists (existing rows are never touched)."""
    cur = conn.execute(
        "insert into questions (id, type, question, answer, gold_doc_ids, status, split, "
        "draft_model, prompt_version, link, origin, key_terms, reviewed_at) values "
        "(%(id)s, %(type)s, %(question)s, "
        "%(answer)s, %(gold_doc_ids)s, %(status)s, %(split)s, %(draft_model)s, "
        "%(prompt_version)s, %(link)s, %(origin)s, %(key_terms)s, %(reviewed_at)s) on conflict (id) do nothing",
        q.model_dump(),
    )
    return cur.rowcount == 1


def save_resolution(conn: psycopg.Connection, resolver_name: str, resolution: Resolution) -> None:
    """Replace all stored rows for `resolver_name` with `resolution`, in one transaction."""
    rows = [
        (resolver_name, mid, eid, resolution.canonical_names[eid], resolution.entity_types[eid])
        for mid, eid in sorted(resolution.assignments.items())
    ]
    with conn.transaction(), conn.cursor() as cur:
        cur.execute("delete from resolutions where resolver = %s", (resolver_name,))
        cur.executemany(
            "insert into resolutions (resolver, mention_id, entity_id, canonical_name, "
            "entity_type) values (%s, %s, %s, %s, %s)",
            rows,
        )


def load_resolution(conn: psycopg.Connection, resolver_name: str) -> Resolution:
    cur = conn.execute(
        "select mention_id, entity_id, canonical_name, entity_type from resolutions "
        "where resolver = %s order by mention_id",
        (resolver_name,),
    )
    out = Resolution(assignments={}, canonical_names={}, entity_types={})
    for mid, eid, name, etype in cur.fetchall():
        out.assignments[mid] = eid
        out.canonical_names[eid] = name
        out.entity_types[eid] = etype
    return out


def resolution_hash(conn: psycopg.Connection, resolver_name: str) -> str | None:
    """Content hash of the stored resolution rows for `resolver_name` (None if there are none).

    Recorded in GraphMeta at build time; eval compares it to detect a re-resolve without rebuild.
    """
    row = conn.execute(
        "select encode(sha256(convert_to(string_agg("
        "concat_ws(E'\\t', mention_id, entity_id, canonical_name, entity_type), E'\\n' "
        "order by mention_id), 'UTF8')), 'hex') from resolutions where resolver = %s",
        (resolver_name,),
    ).fetchone()
    return row[0][:16] if row and row[0] else None

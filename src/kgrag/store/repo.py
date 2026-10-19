"""Postgres access for emails, Jira events and the retrievable `documents` table."""

import re
from datetime import UTC, datetime
from typing import Literal

import psycopg
from pydantic import BaseModel

from kgrag.ingest.jira import JiraEvent, render_event
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
            "subject, body_clean, kind) values (%(id)s, %(message_id)s, %(thread_id)s, %(date)s, "
            "%(sender_name)s, %(sender_address)s, %(subject)s, %(body_clean)s, %(kind)s) "
            "on conflict (id) do update set message_id = excluded.message_id, "
            "thread_id = excluded.thread_id, date = excluded.date, "
            "sender_name = excluded.sender_name, sender_address = excluded.sender_address, "
            "subject = excluded.subject, body_clean = excluded.body_clean, kind = excluded.kind",
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


def log_failure(conn: psycopg.Connection, stage: str, item_id: str, error: str) -> None:
    conn.execute(
        "insert into failures (stage, item_id, error) values (%s, %s, %s)",
        (stage, item_id, error),
    )

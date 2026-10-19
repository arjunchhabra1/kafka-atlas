"""Ingest mbox files: parse, clean, thread and store emails and Jira events, then build documents."""

import hashlib
from pathlib import Path

import psycopg
from pydantic import BaseModel

from kgrag.config import get_settings
from kgrag.ingest.clean import clean_body
from kgrag.ingest.jira import JiraEvent, parse_jira_notification
from kgrag.ingest.parse import RawEmail, assign_threads, classify_kind, dedupe, parse_mbox
from kgrag.store import repo


class IngestStats(BaseModel):
    emails_human: int
    emails_jira: int
    jira_events: int
    unparsed_jira: int
    failures: int
    documents: int


def email_id(message_id: str) -> str:
    return hashlib.sha256(message_id.encode()).hexdigest()[:16]


def ingest(conn: psycopg.Connection, mbox_paths: list[Path]) -> IngestStats:
    """Ingest the given mbox files as one corpus (threading spans all of them).

    Idempotent: emails upsert on id, Jira events are unique per source email, documents are
    rebuilt from scratch, and previous `ingest` failures for these files are cleared first so
    re-runs do not duplicate failure rows.
    """
    paths = [Path(p) for p in mbox_paths]
    conn.execute(
        "delete from failures where stage = 'ingest' and split_part(item_id, '#', 1) = any(%s)",
        ([p.name for p in paths],),
    )
    skipped = 0

    def on_skip(item_id: str, reason: str) -> None:
        nonlocal skipped
        skipped += 1
        repo.log_failure(conn, "ingest", item_id, reason)

    raw: list[RawEmail] = []
    for p in paths:
        raw.extend(parse_mbox(p, on_skip=on_skip))
    emails = dedupe(raw)
    thread_of = assign_threads(emails)

    by_id = {e.message_id: e for e in emails}
    members: dict[str, list[RawEmail]] = {}
    for e in emails:
        members.setdefault(thread_of[e.message_id], []).append(e)
    threads = []
    for root, group in members.items():
        rootmail = by_id.get(root) or min(group, key=lambda e: e.date)
        threads.append((root, root, rootmail.subject))

    rows: list[repo.EmailRow] = []
    events: list[JiraEvent] = []
    human = jira = unparsed = 0
    for e in emails:
        eid = email_id(e.message_id)
        kind = classify_kind(e.sender_name, e.sender_address)
        if kind == "human":
            human += 1
            body = clean_body(e.body)
        else:
            jira += 1
            body = e.body
            event = parse_jira_notification(e, eid)
            if event is None:
                unparsed += 1
            else:
                events.append(event)
        rows.append(repo.EmailRow(
            id=eid, message_id=e.message_id, thread_id=thread_of[e.message_id], date=e.date,
            sender_name=e.sender_name, sender_address=e.sender_address, subject=e.subject,
            body_clean=body, kind=kind,
        ))

    repo.upsert_threads(conn, threads)
    repo.upsert_emails(conn, rows)
    repo.insert_jira_events(conn, events)
    documents = repo.rebuild_documents(conn, get_settings().chunk_tokens)
    conn.commit()
    return IngestStats(
        emails_human=human, emails_jira=jira, jira_events=len(events),
        unparsed_jira=unparsed, failures=skipped, documents=documents,
    )

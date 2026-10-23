"""Rebuild `mentions` and `relations` from cached LLM output plus deterministic sources."""

from dataclasses import dataclass, field

import psycopg

from kgrag.config import get_settings
from kgrag.extract.llm import ExtractionError, parse_response
from kgrag.extract.prompts import PROMPT_VERSION
from kgrag.extract.regex_ids import extract_ids
from kgrag.store.repo import log_failure

MAX_CONTEXT = 300
SOURCES = ("llm", "regex", "header", "jira")


@dataclass
class PersistStats:
    mentions_by_source: dict[str, int] = field(default_factory=lambda: dict.fromkeys(SOURCES, 0))
    relations: int = 0


def _line_around(text: str, start: int, end: int) -> str:
    lo = text.rfind("\n", 0, start) + 1
    hi = text.find("\n", end)
    return text[lo : len(text) if hi == -1 else hi].strip()[:MAX_CONTEXT]


def _persist(conn: psycopg.Connection) -> PersistStats:
    stats = PersistStats()
    model = get_settings().models.extract
    rows: list[tuple[str, str, str, str, str, str]] = []  # mention row + source

    def add(source: str, mention_id: str, surface: str, typ: str, email_id: str, ctx: str) -> None:
        rows.append((mention_id, surface, typ, email_id, ctx, source))
        stats.mentions_by_source[source] += 1

    relations: list[tuple] = []
    cache = conn.execute(
        "select email_id, response from llm_cache where prompt_version = %s and model = %s "
        "order by email_id",
        (PROMPT_VERSION, model),
    ).fetchall()
    for email_id, response in cache:
        try:
            result = parse_response(email_id, response)
        except ExtractionError as e:
            log_failure(conn, "persist", email_id, f"cached response unparseable: {e}")
            continue
        for m in result.mentions:
            add("llm", m.mention_id, m.surface, m.type, m.email_id, m.context)
        for r in result.relations:
            relations.append((
                r.type, r.source_mention_id, r.target_mention_id, r.vote_value, r.binding,
                r.confidence, email_id,
            ))

    humans = conn.execute(
        "select id, sender_name, sender_address, subject, body_clean from emails "
        "where kind = 'human' order by id"
    ).fetchall()
    for eid, name, address, subject, body in humans:
        add("header", f"{eid}:h0", name or "", "Person", eid, f"{name} {address}")
        text = f"{subject or ''}\n{body or ''}"
        for n, hit in enumerate(extract_ids(text)):
            add("regex", f"{eid}:r{n}", text[hit.span[0] : hit.span[1]], hit.kind, eid,
                _line_around(text, *hit.span))

    for event_id, actor, src in conn.execute(
        "select id, actor, source_email_id from jira_events where actor is not null order by id"
    ).fetchall():
        add("jira", f"jira:{event_id}:a0", actor, "Person", src, actor)

    conn.execute("delete from relations")
    conn.execute("delete from mentions")
    with conn.cursor() as cur:
        cur.executemany(
            "insert into mentions (mention_id, surface, type, email_id, context, source) "
            "values (%s, %s, %s, %s, %s, %s)", rows,
        )
        cur.executemany(
            "insert into relations (type, source_mention_id, target_mention_id, vote_value, "
            "binding, confidence, email_id) values (%s, %s, %s, %s, %s, %s, %s)", relations,
        )
    stats.relations = len(relations)
    return stats


def persist_extractions(conn: psycopg.Connection) -> PersistStats:
    """Delete and regenerate all mentions and relations; commit, or roll back and re-raise."""
    try:
        stats = _persist(conn)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return stats

"""Build the dev fixture slice in tests/fixtures from the populated production database.

    uv run python scripts/make_fixture.py --kips KIP-1032,KIP-1043

Selects every human email in threads whose root subject names one of the KIPs, plus the Jira
notification emails of tickets linked to them (a Jira event title names the KIP, or a slice
email mentions the ticket). Writes those messages byte-for-byte from data/raw into
dev_slice.mbox, and their cached Sonnet responses (llm_cache.response, unparsed) into
recorded_extractions.json. It then ingests dev_slice.mbox alone into a fresh, throwaway
database, loads the recorded responses and runs persist_extractions, so that every mention id
and doc id the hand-written gold files reference is the id a test database will produce.
The production database is only read.
"""

import argparse
import json
import mailbox
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from kgrag.config import get_settings
from kgrag.extract.llm import PROMPT_VERSION
from kgrag.extract.pipeline import persist_extractions
from kgrag.ingest.parse import _header, _ids
from kgrag.ingest.pipeline import email_id, ingest
from kgrag.store.db import apply_migrations

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
FIXTURE_DB = "kgrag_fixture"


def select_slice(conn: psycopg.Connection, kips: list[str]) -> tuple[list[str], list[str]]:
    """(human email ids, Jira notification email ids) for the KIPs, both sorted."""
    pattern = "(" + "|".join(re.escape(k) for k in kips) + r")\M"
    humans = [r[0] for r in conn.execute(
        "select e.id from emails e join threads t using (thread_id) "
        "where e.kind = 'human' and t.subject ~ %s order by e.id", (pattern,))]
    tickets = {r[0] for r in conn.execute(
        "select ticket from jira_events where title ~ %s", (pattern,))}
    tickets |= {r[0] for r in conn.execute(
        "select distinct upper(m.surface) from mentions m join jira_events j "
        "on j.ticket = upper(m.surface) where m.type = 'JiraTicket' and m.email_id = any(%s)",
        (humans,))}
    jira = [r[0] for r in conn.execute(
        "select source_email_id from jira_events where ticket = any(%s) order by source_email_id",
        (sorted(tickets),))]
    return humans, jira


def raw_messages(raw_dir: Path, wanted: set[str]) -> bytes:
    """The wanted messages (by email id) as stored in the monthly mboxes, From_ line included.

    The first occurrence wins, matching ingest's dedupe; output is in archive order.
    """
    out: list[bytes] = []
    found: set[str] = set()
    for path in sorted(raw_dir.glob("dev-kafka-*.mbox")):
        mbx = mailbox.mbox(str(path), create=False)
        try:
            keys = mbx.keys()  # builds the table of contents
            with path.open("rb") as f:
                for key in keys:
                    ids = _ids(_header(mbx.get_message(key), "Message-ID"))
                    eid = email_id(ids[0]) if ids else None
                    if eid in wanted and eid not in found:
                        start, stop = mbx._toc[key]
                        f.seek(start)
                        out.append(f.read(stop - start))
                        found.add(eid)
        finally:
            mbx.close()
    if missing := wanted - found:
        raise SystemExit(f"messages not found in {raw_dir}: {sorted(missing)}")
    return b"".join(chunk if chunk.endswith(b"\n\n") else chunk.rstrip(b"\n") + b"\n\n"
                    for chunk in out)


def recorded_responses(conn: psycopg.Connection, humans: list[str]) -> dict[str, dict]:
    rows = dict(conn.execute(
        "select email_id, response from llm_cache where prompt_version = %s and model = %s "
        "and email_id = any(%s)", (PROMPT_VERSION, get_settings().models.extract, humans),
    ).fetchall())
    if missing := set(humans) - set(rows):
        raise SystemExit(f"no cached extraction for: {sorted(missing)}")
    return {eid: rows[eid] for eid in sorted(rows)}


@contextmanager
def fresh_database(admin_dsn: str, name: str) -> Iterator[psycopg.Connection]:
    """Create database `name` on the admin DSN's server, migrate it, yield a connection, drop it."""
    ident = sql.Identifier(name)
    with psycopg.connect(admin_dsn, autocommit=True) as admin:
        admin.execute(sql.SQL("drop database if exists {}").format(ident))
        admin.execute(sql.SQL("create database {}").format(ident))
        try:
            with psycopg.connect(admin_dsn, dbname=name) as conn:
                apply_migrations(conn)
                yield conn
        finally:
            admin.execute(sql.SQL("drop database if exists {}").format(ident))


def load_slice(conn: psycopg.Connection, mbox: Path, recorded: dict[str, dict]) -> None:
    """Ingest the slice mbox, cache the recorded responses, persist mentions and relations."""
    ingest(conn, [mbox])
    model = get_settings().models.extract
    with conn.cursor() as cur:
        cur.executemany(
            "insert into llm_cache (email_id, prompt_version, model, response) "
            "values (%s, %s, %s, %s)",
            [(eid, PROMPT_VERSION, model, Jsonb(resp)) for eid, resp in recorded.items()],
        )
    conn.commit()
    persist_extractions(conn)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kips", required=True, help="comma-separated, e.g. KIP-1032,KIP-1043")
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, default=FIXTURES)
    args = ap.parse_args()
    kips = [k.strip().upper() for k in args.kips.split(",") if k.strip()]

    dsn = get_settings().pg_dsn
    with psycopg.connect(dsn) as prod:
        humans, jira = select_slice(prod, kips)
        recorded = recorded_responses(prod, humans)
    mbox = args.out / "dev_slice.mbox"
    mbox.write_bytes(raw_messages(args.raw_dir, set(humans) | set(jira)))
    (args.out / "recorded_extractions.json").write_text(
        json.dumps(recorded, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    )

    with fresh_database(dsn, FIXTURE_DB) as conn:
        load_slice(conn, mbox, recorded)
        got = {r[0]: r[1] for r in conn.execute("select id, kind from emails")}
        if set(got) != set(humans) | set(jira):
            raise SystemExit("re-ingested slice does not match the selection")
        sources = dict(conn.execute("select source, count(*) from mentions group by 1").fetchall())
        docs = conn.execute("select count(*) from documents").fetchone()[0]
    print(f"{kips}: {len(humans)} human + {len(jira)} jira emails -> {mbox}")
    print(f"fresh-db check: {docs} documents, mentions by source {sources}")


if __name__ == "__main__":
    main()

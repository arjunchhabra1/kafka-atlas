import json
from datetime import UTC, datetime

import anthropic
import pytest
from psycopg.types.json import Jsonb
from typer.testing import CliRunner

from kgrag import cli
from kgrag.config import get_settings
from kgrag.extract.pipeline import persist_extractions
from kgrag.extract.prompts import PROMPT_VERSION
from kgrag.store import repo
from kgrag.store.repo import EmailRow

MODEL = "claude-sonnet-5-5"
TABLES = ("relations", "mentions", "llm_cache", "jira_events", "emails", "threads", "failures")


def _email(eid, name, address, body, day, subject="Re: [VOTE] KIP-1", kind="human"):
    return EmailRow(
        id=eid, message_id=f"<{eid}@x>", thread_id="t1",
        date=datetime(2026, 1, day, tzinfo=UTC), sender_name=name, sender_address=address,
        subject=subject, body_clean=body, kind=kind,
    )


EMAILS = [
    _email("e1", "Jun Rao", "jun@confluent.io", "I propose KIP-1.\nSee KAFKA-100 for details.", 1),
    _email("e2", "Ann Dev", "ann@example.org", "+1 (binding) on KIP-1. Targets 3.7.0 release.", 2),
    _email("e3", "Jun Rao", "jun@confluent.io", "Thanks all, merging soon.", 3),
    _email("j1", "Jane Dev (Jira)", "jira@apache.org", "Created.", 4,
           subject="[jira] [Created] (KAFKA-100) Fix", kind="jira"),
]


def _message(mentions, relations):
    payload = {"mentions": mentions, "relations": relations}
    return {
        "id": "msg_1", "type": "message", "role": "assistant", "model": MODEL,
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": json.dumps(payload)}],
        "usage": {"input_tokens": 10, "output_tokens": 10},
    }


def _m(lid, surface, typ):
    return {"local_id": lid, "surface": surface, "type": typ, "context": f"{surface} here."}


def _r(typ, src, dst, vote=None, binding=None):
    return {"type": typ, "source": src, "target": dst, "vote_value": vote,
            "binding": binding, "confidence": 0.9}


RECORDED = {
    "e1": _message([_m("m1", "Jun", "Person"), _m("m2", "KIP-1", "KIP")],
                   [_r("PROPOSED", "m1", "m2")]),
    "e2": _message([_m("m1", "Ann", "Person"), _m("m2", "KIP-1", "KIP")],
                   [_r("VOTED", "m1", "m2", vote=1, binding=True)]),
    "e3": _message([_m("m1", "Jun", "Person")], []),
}
N_LLM_MENTIONS = 5
N_RELATIONS = 2


@pytest.fixture
def db(pg_conn):
    pg_conn.execute(f"truncate {', '.join(TABLES)}")
    repo.upsert_threads(pg_conn, [("t1", "<e1@x>", "[VOTE] KIP-1")])
    repo.upsert_emails(pg_conn, EMAILS)
    pg_conn.execute(
        "insert into jira_events (ticket, action, actor, date, source_email_id) "
        "values ('KAFKA-100', 'created', 'Jane Dev', %s, 'j1')",
        (datetime(2026, 1, 4, tzinfo=UTC),),
    )
    for eid, msg in RECORDED.items():
        pg_conn.execute(
            "insert into llm_cache (email_id, prompt_version, model, response) "
            "values (%s, %s, %s, %s)", (eid, PROMPT_VERSION, MODEL, Jsonb(msg)),
        )
    pg_conn.commit()
    return pg_conn


def _count(conn, sql, *args):
    return conn.execute(sql, args).fetchone()[0]


def test_persist_from_cache(db):
    stats = persist_extractions(db)
    assert stats.mentions_by_source["llm"] == N_LLM_MENTIONS
    assert stats.relations == N_RELATIONS
    assert _count(db, "select count(*) from mentions where source = 'llm'") == N_LLM_MENTIONS
    assert _count(db, "select count(*) from relations") == N_RELATIONS
    vote = db.execute(
        "select vote_value, binding, source_mention_id, target_mention_id, email_id "
        "from relations where type = 'VOTED'"
    ).fetchone()
    assert vote == (1, True, "e2:m1", "e2:m2", "e2")
    # deterministic rebuild: running again yields the same rows, not duplicates
    persist_extractions(db)
    assert _count(db, "select count(*) from mentions") == sum(stats.mentions_by_source.values())
    assert _count(db, "select count(*) from relations") == N_RELATIONS


def test_header_and_jira_mentions_created(db):
    stats = persist_extractions(db)
    assert stats.mentions_by_source["header"] == 3  # human emails only
    assert stats.mentions_by_source["jira"] == 1
    header = db.execute(
        "select surface, type, email_id, context from mentions where mention_id = 'e1:h0'"
    ).fetchone()
    assert header == ("Jun Rao", "Person", "e1", "Jun Rao jun@confluent.io")
    jira = db.execute(
        "select surface, type, email_id from mentions where mention_id like 'jira:%'"
    ).fetchall()
    assert jira == [("Jane Dev", "Person", "j1")]


def test_regex_mentions(db):
    persist_extractions(db)
    rows = db.execute(
        "select mention_id, surface, type, context from mentions where source = 'regex' "
        "order by mention_id"
    ).fetchall()
    assert [(r[0], r[1], r[2]) for r in rows] == [
        ("e1:r0", "KIP-1", "KIP"),  # subject is scanned too
        ("e1:r1", "KIP-1", "KIP"),
        ("e1:r2", "KAFKA-100", "JiraTicket"),
        ("e2:r0", "KIP-1", "KIP"),
        ("e2:r1", "KIP-1", "KIP"),
        ("e2:r2", "3.7.0", "Release"),
        ("e3:r0", "KIP-1", "KIP"),
    ]
    assert rows[0][3] == "Re: [VOTE] KIP-1"
    assert rows[2][3] == "See KAFKA-100 for details."


def test_anchor_hints(db):
    persist_extractions(db)
    hints = repo.load_anchor_hints(db)
    assert ("Jun Rao", "jun@confluent.io") in hints.sender_pairs
    assert hints.sender_pairs == sorted(set(hints.sender_pairs)) and len(hints.sender_pairs) == 2
    assert hints.jira_actors == ["Jane Dev"]
    mentions = repo.load_mentions(db)
    jira_id = _count(db, "select id from jira_events")
    assert {m.mention_id for m in mentions} >= {"e1:h0", "e1:m1", f"jira:{jira_id}:a0"}


class _Conn:
    """Shares the session connection but ignores close()."""

    def __init__(self, conn):
        self._c = conn

    def __getattr__(self, name):
        return getattr(self._c, name)

    def close(self):
        pass


@pytest.fixture
def cli_env(db, monkeypatch):
    monkeypatch.setattr(cli, "connect", lambda: _Conn(db))
    monkeypatch.setattr(anthropic, "Anthropic", _boom)
    yield
    get_settings.cache_clear()


def _boom(*a, **k):
    raise AssertionError("Anthropic client must not be constructed")


def test_cost_guard_blocks_without_yes(db, cli_env, monkeypatch):
    db.execute("delete from llm_cache")
    db.commit()
    monkeypatch.setenv("KGRAG_COST_GUARD_USD", "0.0000001")
    get_settings.cache_clear()
    res = CliRunner().invoke(cli.app, ["extract"])
    assert res.exit_code == 2, res.output
    assert "estimated cost" in res.output.lower()
    assert _count(db, "select count(*) from llm_cache") == 0


def test_extract_nothing_to_do_persists(db, cli_env):
    res = CliRunner().invoke(cli.app, ["extract"])  # all 3 emails already cached
    assert res.exit_code == 0, res.output
    assert _count(db, "select count(*) from relations") == N_RELATIONS


def test_show_prints_cached_results_without_api(db, cli_env):
    res = CliRunner().invoke(cli.app, ["extract", "--show", "2"])
    assert res.exit_code == 0, res.output
    assert "KIP-1" in res.output and "Re: [VOTE] KIP-1" in res.output


def test_resume_reports_emails_outside_batch(db, cli_env, monkeypatch, tmp_path):
    db.execute("delete from llm_cache where email_id in ('e2', 'e3')")
    db.commit()
    monkeypatch.setenv("KGRAG_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    (tmp_path / "batches").mkdir()
    (tmp_path / "batches" / "b1.json").write_text(json.dumps({"e0": "e3"}))
    monkeypatch.setattr(cli, "_make_client", lambda: object())
    calls = {}

    def fake_run(conn, emails, client, **kw):
        calls.update(kw, n=len(emails))
        from kgrag.extract.llm import ExtractionStats
        return ExtractionStats(done=1)

    monkeypatch.setattr(cli, "run_extraction", fake_run)
    res = CliRunner().invoke(cli.app, ["extract", "--resume-batch", "b1", "--yes"])
    assert res.exit_code == 0, res.output
    assert calls["resume_batch_id"] == "b1"
    assert "1 selected emails were not part of batch b1" in res.output

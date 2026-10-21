from collections import Counter

import pytest
from typer.testing import CliRunner

from kgrag import cli
from kgrag.questions.split import AlreadySplitError, assign_splits, persist_splits
from kgrag.store.repo import QuestionRow, insert_question

TYPES = ("single_hop", "inference", "comparison", "temporal", "null")


def _q(qid, qtype="single_hop", status="kept", split=None):
    return QuestionRow(
        id=qid, type=qtype, question="q?", answer="a", gold_doc_ids=[], status=status, split=split,
    )


def _many(per_type=100, status="kept"):
    return [_q(f"{t}-{i:03d}", t, status) for t in TYPES for i in range(per_type)]


def test_split_stratified():
    result = assign_splits(_many())
    for t in TYPES:
        got = Counter(v for k, v in result.items() if k.startswith(t))
        assert got == {"dev": 20, "test": 80}


def test_split_deterministic():
    qs = _many(30)
    assert assign_splits(qs) == assign_splits(list(reversed(qs)))
    assert assign_splits(qs) != assign_splits(qs, seed=14)


def test_split_excludes_dropped():
    qs = _many(10)
    qs += [_q("x-dropped", status="dropped"), _q("x-draft", status="draft")]
    qs.append(_q("x-edited", status="edited"))
    result = assign_splits(qs)
    assert "x-dropped" not in result and "x-draft" not in result
    assert "x-edited" in result
    assert len(result) == 51


@pytest.fixture
def qdb(pg_conn):
    pg_conn.execute("truncate questions")
    pg_conn.commit()
    return pg_conn


def _load(conn, qs):
    for q in qs:
        insert_question(conn, q)
    conn.commit()


def test_split_refuses_reassign(qdb):
    _load(qdb, _many(10))
    counts = persist_splits(qdb)
    assert counts["single_hop"] == {"dev": 2, "test": 8, "new": 10}
    before = dict(qdb.execute("select id, split from questions").fetchall())
    assert set(before.values()) == {"dev", "test"}
    with pytest.raises(AlreadySplitError):
        persist_splits(qdb)
    persist_splits(qdb, force=True)
    assert dict(qdb.execute("select id, split from questions").fetchall()) == before


def test_split_force_assigns_only_unsplit_questions(qdb):
    _load(qdb, _many(10))
    persist_splits(qdb)
    before = dict(qdb.execute("select id, split from questions").fetchall())
    _load(qdb, [_q(f"single_hop-new{i}") for i in range(10)])
    counts = persist_splits(qdb, force=True)
    after = dict(qdb.execute("select id, split from questions").fetchall())
    assert {k: after[k] for k in before} == before  # nothing moved between dev and test
    new = [after[f"single_hop-new{i}"] for i in range(10)]
    assert Counter(new) == {"dev": 2, "test": 8}
    assert counts["single_hop"] == {"dev": 4, "test": 16, "new": 10}
    assert counts["inference"]["new"] == 0


def test_split_reshuffle_reassigns_all(qdb):
    _load(qdb, _many(10))
    qdb.execute("update questions set split = 'test'")
    qdb.commit()
    counts = persist_splits(qdb, reshuffle=True)
    assert counts["single_hop"] == {"dev": 2, "test": 8, "new": 10}


def test_split_ignores_drafts_and_clears_dropped(qdb):
    _load(qdb, _many(10) + [_q("d1", status="draft"), _q("x1", status="dropped", split="dev")])
    persist_splits(qdb, force=True)
    rows = dict(qdb.execute("select id, split from questions").fetchall())
    assert rows["d1"] is None
    assert rows["x1"] is None


class _Conn:
    def __init__(self, c):
        self._c = c

    def __getattr__(self, name):
        return getattr(self._c, name)

    def close(self):
        pass


def test_split_cli(qdb, monkeypatch):
    monkeypatch.setattr(cli, "connect", lambda: _Conn(qdb))
    _load(qdb, _many(10) + [_q("d1", status="draft")])
    res = CliRunner().invoke(cli.app, ["questions", "split"])
    assert res.exit_code == 0, res.output
    assert "single_hop" in res.output and "1 draft" in res.output
    again = CliRunner().invoke(cli.app, ["questions", "split"])
    assert again.exit_code == 1 and "--force" in again.output
    forced = CliRunner().invoke(cli.app, ["questions", "split", "--force"])
    assert forced.exit_code == 0 and "(0 newly assigned)" in forced.output
    assert "leakage" not in forced.output
    shuffled = CliRunner().invoke(cli.app, ["questions", "split", "--reshuffle"])
    assert shuffled.exit_code == 0 and "leakage" in shuffled.output
    both = CliRunner().invoke(cli.app, ["questions", "split", "--force", "--reshuffle"])
    assert both.exit_code == 2

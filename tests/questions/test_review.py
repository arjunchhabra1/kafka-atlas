from datetime import UTC, datetime
from io import StringIO

import pytest
import yaml
from rich.console import Console

from kgrag.questions.draft import NULL_ANSWER
from kgrag.questions.review import progress, review_session, validate_edit
from kgrag.store.repo import QuestionRow, insert_question


def _doc(i):
    return (f"d{i}", "email", f"e{i}", 0, f"text {i} " + "x" * 2000, 10,
            datetime(2026, 1, i, tzinfo=UTC), "Dev", f"subject {i}")


def _q(qid, qtype="single_hop", gold=("d1",), answer="a", **kw):
    return QuestionRow(id=qid, type=qtype, question=f"{qid}?", answer=answer,
                       gold_doc_ids=list(gold), status="draft", **kw)


@pytest.fixture
def db(pg_conn):
    pg_conn.execute("truncate questions, documents")
    pg_conn.execute("truncate threads cascade")
    with pg_conn.cursor() as cur:
        cur.executemany(
            "insert into documents (doc_id, source_kind, source_id, chunk_index, text, "
            "token_count, date, sender, subject) values (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            [_doc(i) for i in (1, 2, 3)],
        )
    for q in (_q("a1", gold=("d1",)), _q("a2", gold=("d2",)), _q("a3", gold=("d3",)),
              _q("b1", "inference", gold=("d1", "d2")),
              _q("n1", "null", gold=(), answer=NULL_ANSWER, link="thread:t1")):
        insert_question(pg_conn, q)
    pg_conn.commit()
    return pg_conn


def keys(s):
    it = iter(s)
    return lambda: next(it)


def run(db, script, editor=None, qtype=None):
    out = StringIO()
    console = Console(file=out, width=100, force_terminal=False)
    review_session(db, read_key=keys(script), editor=editor or (lambda t: None),
                   console=console, qtype=qtype)
    return out.getvalue()


def status(db):
    return dict(db.execute("select id, status from questions").fetchall())


def test_review_keep_drop_persist(db):
    run(db, "kdq")
    st = status(db)
    assert st["a1"] == "kept" and st["a2"] == "dropped" and st["a3"] == "draft"
    assert db.execute("select reviewed_at from questions where id='a1'").fetchone()[0] is not None
    assert db.execute("select reviewed_at from questions where id='a3'").fetchone()[0] is None
    # restart resumes at the first remaining draft
    out = run(db, "q")
    assert "a3?" in out and "a1?" not in out


def test_progress_header_and_doc_rendering(db):
    out = run(db, "q")
    assert "single_hop" in out and "remaining" in out
    assert "[d1] 2026-01-01 · Dev · subject 1" in out
    assert "truncated" in out


def test_commit_after_every_decision(db, pg_conn):
    # a decision followed by a crash (no q) must already be durable
    with pytest.raises(StopIteration):
        run(db, "k")
    db.rollback()
    assert status(db)["a1"] == "kept"


def test_skip_leaves_draft(db):
    run(db, "ssq")
    assert status(db)["a1"] == "draft" and status(db)["a2"] == "draft"
    assert "a1?" in run(db, "q")  # skipped items reappear next session


def test_unknown_key_reprompts(db):
    run(db, "zkq")
    assert status(db)["a1"] == "kept"


def test_type_filter(db):
    run(db, "kq", qtype="inference")
    st = status(db)
    assert st["b1"] == "kept" and st["a1"] == "draft"


def test_null_shows_thread_subject(db):
    db.execute("insert into threads (thread_id, root_message_id, subject) values "
               "('t1', '<t1@x>', 'Thread about nothing')")
    db.commit()
    out = run(db, "q", qtype="null")
    assert "Thread about nothing" in out


def test_edit_updates_fields(db):
    def editor(text):
        data = yaml.safe_load(text)
        assert data["question"] == "a1?" and data["gold_doc_ids"] == ["d1"]
        data.update(question="Better?", answer="Better answer", gold_doc_ids=["d2"])
        return yaml.safe_dump(data)

    run(db, "eq", editor=editor)
    row = db.execute("select status, question, answer, gold_doc_ids, reviewed_at from questions "
                     "where id='a1'").fetchone()
    assert row[:4] == ("edited", "Better?", "Better answer", ["d2"])
    assert row[4] is not None


def test_edit_invalid_then_reedit(db):
    seen = []

    def editor(text):
        seen.append(text)
        data = yaml.safe_load(text)
        if len(seen) == 1:
            data["gold_doc_ids"] = ["nope"]
        else:
            data.update(gold_doc_ids=["d1"], question="Fixed?")
        return yaml.safe_dump(data)

    out = run(db, "erq", editor=editor)  # r = re-open editor after error
    assert "nope" in out
    assert status(db)["a1"] == "edited"


def test_edit_invalid_then_cancel(db):
    out = run(db, "ecdq", editor=lambda t: yaml.safe_dump({**yaml.safe_load(t), "question": " "}))
    assert "question" in out
    st = status(db)
    assert st["a1"] == "dropped"  # cancel returned to k/e/d prompt, then d


def test_edit_unchanged_or_aborted_editor_cancels(db):
    run(db, "ekq", editor=lambda t: None)  # editor returned nothing -> back to prompt, then keep
    assert status(db)["a1"] == "kept"


def test_validate_edit_rules(db):
    sh, inf, null = _q("s"), _q("i", "inference", ("d1", "d2")), _q("n", "null", (), NULL_ANSWER)
    ok = {"question": "q?", "answer": "a", "gold_doc_ids": ["d1"]}
    assert validate_edit(db, sh, ok) is None
    assert "exactly 1" in validate_edit(db, sh, {**ok, "gold_doc_ids": ["d1", "d2"]})
    assert "exactly 1" in validate_edit(db, sh, {**ok, "gold_doc_ids": []})
    assert "at least 2" in validate_edit(db, inf, {**ok, "gold_doc_ids": ["d1"]})
    assert validate_edit(db, inf, {**ok, "gold_doc_ids": ["d1", "d3"]}) is None
    assert "unknown" in validate_edit(db, inf, {**ok, "gold_doc_ids": ["d1", "zz"]})
    assert "answer" in validate_edit(db, sh, {**ok, "answer": ""})
    nok = {"question": "q?", "answer": NULL_ANSWER, "gold_doc_ids": []}
    assert validate_edit(db, null, nok) is None
    assert "answer" in validate_edit(db, null, {**nok, "answer": "something"})
    assert "exactly 0" in validate_edit(db, null, {**nok, "gold_doc_ids": ["d1"]})
    assert "mapping" in validate_edit(db, sh, "just a string")


def test_progress_counts(db):
    run(db, "kdq")
    p = progress(db)
    assert p["single_hop"] == {"kept": 1, "edited": 0, "dropped": 1, "remaining": 1}
    assert p["inference"]["remaining"] == 1


def test_unchanged_edit_is_cancel(db):
    out = run(db, "ekq", editor=lambda t: t)
    assert "No changes" in out and status(db)["a1"] == "kept"


def test_edit_strips_text(db):
    def editor(text):
        return yaml.safe_dump({**yaml.safe_load(text), "question": "  Padded?  \n"})

    run(db, "eq", editor=editor)
    assert db.execute("select question from questions where id='a1'").fetchone()[0] == "Padded?"


def test_duplicate_gold_ids_rejected(db):
    inf = _q("i", "inference", ("d1", "d2"))
    err = validate_edit(db, inf, {"question": "q?", "answer": "a", "gold_doc_ids": ["d1", "d1"]})
    assert "duplicate" in err and "d1" in err


def test_emoji_shortcodes_survive(db):
    db.execute("update questions set question = 'is :warning: shown?' where id='a1'")
    db.commit()
    assert ":warning:" in run(db, "q")


def test_missing_editor_binary_cancels(monkeypatch, capsys):
    from kgrag.questions.review import external_editor

    monkeypatch.setenv("EDITOR", "/nonexistent/editor-xyz")
    assert external_editor("x") is None
    assert "Could not run editor" in capsys.readouterr().err

import hashlib
import mailbox
from email.message import EmailMessage
from pathlib import Path

import pytest

from kgrag.ingest.pipeline import ingest
from kgrag.store import repo
from kgrag.tokens import count_tokens

TABLES = ("documents", "jira_events", "emails", "threads", "failures")


def _msg(mid, body, frm="Jun Rao <jun@confluent.io>", subject="Re: hi",
         date="Tue, 05 Mar 2024 10:00:00 +0000", **headers):
    m = EmailMessage()
    if mid:
        m["Message-ID"] = mid
    m["From"] = frm
    m["Subject"] = subject
    if date:
        m["Date"] = date
    for k, v in headers.items():
        m[k.replace("_", "-")] = v
    m.set_content(body)
    return m


def _write(path: Path, msgs) -> Path:
    mb = mailbox.mbox(str(path))
    for m in msgs:
        mb.add(m)
    mb.flush()
    mb.close()
    return path


def _long_body() -> str:
    paras = [
        " ".join(f"Broker {i}-{j} restarted cleanly after the rolling upgrade finished."
                 for j in range(18))
        for i in range(6)
    ]
    return "\n\n".join(paras)


JIRA_FROM = "Jane Dev (Jira) <jira@apache.org>"


@pytest.fixture
def mbox(tmp_path):
    return _write(tmp_path / "dev-kafka-2024-03.mbox", [
        _msg("<h1@x>", "First human mail.", subject="KIP-1 discussion"),
        _msg("<h2@x>", "A reply.", subject="Re: KIP-1 discussion", In_Reply_To="<h1@x>"),
        _msg("<h3@x>", _long_body(), subject="Long one"),
        _msg("<h4@x>", "Another one.", frm="Ann <ann@example.com>", subject="Other"),
        _msg("<j1@x>", "Created.\n", frm=JIRA_FROM,
             subject="[jira] [Created] (KAFKA-1) Fix the thing"),
        _msg("<j2@x>", "Resolved.\n", frm=JIRA_FROM,
             subject="[jira] [Resolved] (KAFKA-1) Fix the thing"),
    ])


@pytest.fixture
def db(pg_conn):
    pg_conn.execute(f"truncate {', '.join(TABLES)}")
    pg_conn.commit()
    return pg_conn


def _count(conn, table):
    return conn.execute(f"select count(*) from {table}").fetchone()[0]


@pytest.mark.integration
def test_ingest_counts(db, mbox):
    stats = ingest(db, [mbox])
    assert stats.emails_human == 4
    assert stats.emails_jira == 2
    assert stats.jira_events == 2
    assert stats.unparsed_jira == 0
    assert stats.failures == 0
    assert _count(db, "emails") == 6
    assert _count(db, "jira_events") == 2
    assert stats.documents == _count(db, "documents") == 3 + 3 + 2
    thread = db.execute("select thread_id from emails where message_id='h2@x'").fetchone()[0]
    assert thread == "h1@x"
    assert db.execute("select subject from threads where thread_id='h1@x'").fetchone()[0] \
        == "KIP-1 discussion"


@pytest.mark.integration
def test_ingest_idempotent(db, mbox):
    first = ingest(db, [mbox])
    before = {t: _count(db, t) for t in TABLES}
    second = ingest(db, [mbox])
    assert {t: _count(db, t) for t in TABLES} == before
    assert second == first


@pytest.mark.integration
def test_skips_logged_once_across_runs(db, tmp_path):
    p = _write(tmp_path / "dev-kafka-2024-04.mbox", [_msg("<a@x>", "ok"), _msg(None, "no id")])
    assert ingest(db, [p]).failures == 1
    assert ingest(db, [p]).failures == 1
    assert _count(db, "failures") == 1


@pytest.mark.integration
def test_cross_file_threading(db, tmp_path):
    a = _write(tmp_path / "dev-kafka-2024-03.mbox", [_msg("<r@x>", "root", subject="Topic")])
    b = _write(tmp_path / "dev-kafka-2024-04.mbox",
               [_msg("<c@x>", "child", subject="Re: Topic", In_Reply_To="<r@x>")])
    ingest(db, [a, b])
    assert db.execute("select thread_id from emails where message_id='c@x'").fetchone()[0] == "r@x"


@pytest.mark.integration
def test_unparsed_jira_counted(db, tmp_path):
    p = _write(tmp_path / "dev-kafka-2024-05.mbox", [
        _msg("<b@x>", "build", frm="Apache Jenkins <jenkins@builds.apache.org>",
             subject="Build failed")])
    stats = ingest(db, [p])
    assert (stats.emails_jira, stats.jira_events, stats.unparsed_jira) == (1, 0, 1)
    assert stats.documents == 0


@pytest.mark.integration
def test_long_email_chunked(db, mbox):
    assert count_tokens(_long_body()) >= 1100
    ingest(db, [mbox])
    eid = hashlib.sha256(b"h3@x").hexdigest()[:16]
    rows = db.execute(
        "select token_count, text from documents where source_id=%s order by chunk_index", (eid,)
    ).fetchall()
    assert len(rows) == 3
    for tc, text in rows:
        assert tc == count_tokens(text) <= 500


@pytest.mark.integration
def test_doc_ids_format(db, mbox):
    ingest(db, [mbox])
    ids = [r[0] for r in db.execute("select doc_id from documents order by doc_id")]
    h1 = hashlib.sha256(b"h1@x").hexdigest()[:16]
    assert f"email:{h1}:0" in ids
    event_ids = [r[0] for r in db.execute("select id from jira_events")]
    for i in event_ids:
        assert f"jira:{i}" in ids
    docs = repo.get_documents(db, [f"jira:{event_ids[0]}", f"email:{h1}:0"])
    assert {d.source_kind for d in docs} == {"jira", "email"}
    jd = next(d for d in docs if d.source_kind == "jira")
    assert jd.sender == "Jane Dev" and jd.subject == "KAFKA-1 Fix the thing"
    assert "KAFKA-1" in jd.text


def test_chunk_text_splits_oversized_paragraph_and_sentence():
    para = " ".join(f"Sentence number {i} is here." for i in range(300))
    chunks = repo.chunk_text(para, 100)
    assert len(chunks) > 1
    assert all(count_tokens(c) <= 100 for c in chunks)
    blob = "word" * 2000
    chunks = repo.chunk_text(blob, 100)
    assert len(chunks) > 1 and all(count_tokens(c) <= 100 for c in chunks)
    assert "".join(chunks) == blob
    assert repo.chunk_text("   \n\n ", 100) == []

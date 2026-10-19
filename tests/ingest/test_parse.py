import mailbox
from datetime import UTC, datetime
from email.message import EmailMessage
from pathlib import Path

from kgrag.ingest.parse import (
    RawEmail,
    assign_threads,
    classify_kind,
    dedupe,
    parse_mbox,
)


def _msg(mid="<a@x>", body="hello", frm="Jun Rao <Jun@Confluent.io>", subject="Re: hi",
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


def _write(path: Path, msgs: list) -> Path:
    mb = mailbox.mbox(str(path))
    for m in msgs:
        mb.add(m)
    mb.flush()
    mb.close()
    return path


def _raw(mid, irt=None, refs=()):
    return RawEmail(message_id=mid, in_reply_to=irt, references=list(refs),
                    date=datetime(2024, 3, 5, tzinfo=UTC), sender_name="n",
                    sender_address="a@b", subject="s", body="b")


def test_parse_basic_fields(tmp_path):
    p = _write(tmp_path / "m.mbox", [_msg(In_Reply_To="<p@x>", References="<r@x> <p@x>")])
    [e] = list(parse_mbox(p))
    assert e.message_id == "a@x"
    assert e.in_reply_to == "p@x"
    assert e.references == ["r@x", "p@x"]
    assert e.sender_name == "Jun Rao" and e.sender_address == "jun@confluent.io"
    assert e.date == datetime(2024, 3, 5, 10, tzinfo=UTC)
    assert e.body.strip() == "hello"


def test_parse_latin1_and_html_only(tmp_path):
    latin = EmailMessage()
    latin["Message-ID"] = "<l@x>"
    latin["From"] = "A <a@x>"
    latin["Date"] = "Tue, 05 Mar 2024 10:00:00 +0000"
    latin.set_content("caf\xe9 au lait", charset="latin-1", cte="8bit")
    html = EmailMessage()
    html["Message-ID"] = "<h@x>"
    html["From"] = "B <b@x>"
    html["Date"] = "Tue, 05 Mar 2024 10:00:00 +0000"
    html.set_content("<html><body><p>Hello <b>world</b></p><script>x()</script></body></html>",
                     subtype="html")
    raw = (b"Message-ID: <bad@x>\nFrom: C <c@x>\nDate: Tue, 05 Mar 2024 10:00:00 +0000\n"
           b"Content-Type: text/plain; charset=nonsense-9\nContent-Transfer-Encoding: 8bit\n\n"
           b"caf\xe9 \xff ok\n")
    p = tmp_path / "m.mbox"
    mb = mailbox.mbox(str(p))
    mb.add(latin)
    mb.add(html)
    mb.add(mailbox.mboxMessage(raw))
    mb.flush()
    mb.close()
    emails = {e.message_id: e for e in parse_mbox(p)}
    assert "café au lait" in emails["l@x"].body
    assert emails["h@x"].body.split() == ["Hello", "world"]
    assert emails["bad@x"].body.strip() != ""


def test_missing_message_id_and_bad_date_reported(tmp_path):
    p = _write(tmp_path / "m.mbox", [_msg(mid=None), _msg(mid="<ok@x>"), _msg(mid="<d@x>", date="garbage")])
    skipped = []
    emails = list(parse_mbox(p, on_skip=lambda i, r: skipped.append((i, r))))
    assert [e.message_id for e in emails] == ["ok@x"]
    assert [i for i, _ in skipped] == ["m.mbox#0", "m.mbox#2"]
    assert all(r for _, r in skipped)


def test_naive_date_assumed_utc_and_subject_decoded(tmp_path):
    p = _write(tmp_path / "m.mbox", [_msg(date="Tue, 05 Mar 2024 10:00:00 -0000",
                                          subject="=?utf-8?q?caf=C3=A9?=   topic")])
    [e] = list(parse_mbox(p))
    assert e.date.tzinfo is not None
    assert e.subject == "café topic"


def test_classify_kind():
    assert classify_kind("Chia-Ping Tsai (Jira)", "jira@apache.org") == "jira"
    assert classify_kind("Jun Rao", "jun@confluent.io") == "human"
    assert classify_kind("x", "jira@apache.org") == "jira"
    assert classify_kind("Apache Jenkins Server", "jenkins@builds.apache.org") == "jira"
    assert classify_kind("GitHub", "noreply@github.com") == "jira"
    assert classify_kind("Bob", "notifications@x.org") == "jira"


def test_dedupe_keeps_first():
    a1, a2, b = _raw("a"), _raw("a"), _raw("b")
    a2.body = "second"
    out = dedupe([a1, a2, b])
    assert out == [a1, b] and out[0].body == "b"


def test_assign_threads():
    emails = [_raw("c", irt="b", refs=["a", "b"]), _raw("a"), _raw("b", irt="a"),
              _raw("o"), _raw("r", refs=["x", "b"]), _raw("m", irt="gone")]
    t = assign_threads(emails)
    assert t == {"a": "a", "b": "a", "c": "a", "o": "o", "r": "a", "m": "m"}


def test_assign_threads_cycle_terminates():
    t = assign_threads([_raw("a", irt="b"), _raw("b", irt="a")])
    assert set(t) == {"a", "b"}

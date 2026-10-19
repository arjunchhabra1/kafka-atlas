import json
import re
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from kgrag.config import get_settings
from kgrag.extract import llm
from kgrag.extract.llm import (
    ExtractionError,
    build_request,
    estimate_cost_usd,
    parse_response,
    run_extraction,
)
from kgrag.extract.prompts import PROMPT_VERSION, SYSTEM_PROMPT
from kgrag.store.repo import EmailRow

MODEL = "claude-sonnet-5-5"


def make_email(eid="e1", body="I propose KIP-1 and +1 from me (binding)."):
    return EmailRow(
        id=eid,
        message_id=f"<{eid}@x>",
        thread_id="t1",
        date=datetime(2026, 1, 1, tzinfo=UTC),
        sender_name="Ann Dev",
        sender_address="ann@example.org",
        subject="Re: [VOTE] KIP-1",
        body_clean=body,
        kind="human",
    )


GOOD = {
    "mentions": [
        {"local_id": "m1", "surface": "Ann", "type": "Person", "context": "Ann proposes it."},
        {"local_id": "m2", "surface": "KIP-1", "type": "KIP", "context": "See KIP-1."},
    ],
    "relations": [
        {
            "type": "PROPOSED",
            "source": "m1",
            "target": "m2",
            "vote_value": None,
            "binding": None,
            "confidence": 0.9,
        }
    ],
}


def message(payload, stop_reason="end_turn", as_text=None):
    text = as_text if as_text is not None else json.dumps(payload)
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "stop_reason": stop_reason,
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


class Obj(SimpleNamespace):
    def model_dump(self, mode=None):
        return self._dump


def sdk_message(payload, stop_reason="end_turn", as_text=None):
    d = message(payload, stop_reason, as_text)
    return Obj(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=d["content"][0]["text"])],
        _dump=d,
    )


class FakeClient:
    """Implements just messages.create and messages.batches.{create,retrieve,results}."""

    def __init__(self, responder=None, batch_responder=None, polls_before_end=1):
        self.responder = responder or (lambda params: sdk_message(GOOD))
        self.batch_responder = batch_responder or (lambda params: sdk_message(GOOD))
        self.polls_before_end = polls_before_end
        self.create_calls = []
        self.batch_requests = []
        self.retrieve_calls = 0
        self.retrieve_raises = False
        self.messages = SimpleNamespace(
            create=self._create,
            batches=SimpleNamespace(
                create=self._batch_create, retrieve=self._retrieve, results=self._results
            ),
        )

    def _create(self, **params):
        self.create_calls.append(params)
        return self.responder(params)

    def _batch_create(self, *, requests):
        self.batch_requests.append(requests)
        return SimpleNamespace(id="batch_1", processing_status="in_progress")

    def _retrieve(self, batch_id):
        self.retrieve_calls += 1
        if self.retrieve_raises:
            raise RuntimeError("network down")
        done = self.retrieve_calls > self.polls_before_end
        return SimpleNamespace(
            id=batch_id, processing_status="ended" if done else "in_progress"
        )

    def _results(self, batch_id):
        out = []
        for req in reversed(self.batch_requests[-1]):  # arrive out of order
            msg = self.batch_responder(req["params"])
            if isinstance(msg, str):  # "api_error", "invalid_request_error", "expired", ...
                if msg in ("expired", "canceled"):
                    result = SimpleNamespace(type=msg)
                else:
                    err = SimpleNamespace(type="error", error=SimpleNamespace(type=msg))
                    result = SimpleNamespace(type="errored", error=err)
            elif msg is None:
                result = SimpleNamespace(
                    type="errored",
                    error=SimpleNamespace(type="error", error=SimpleNamespace(type="api_error")),
                )
            else:
                result = SimpleNamespace(type="succeeded", message=msg)
            out.append(SimpleNamespace(custom_id=req["custom_id"], result=result))
        return iter(out)


def no_sleep(_):
    pass


@pytest.fixture(autouse=True)
def batch_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "_batch_dir", lambda: tmp_path / "batches")
    return tmp_path / "batches"


def test_build_request_uses_structured_output():
    req = build_request(make_email(), "[VOTE] KIP-1")
    assert req["model"] == MODEL
    assert req["max_tokens"] == 8000
    assert req["output_config"]["format"]["type"] == "json_schema"
    assert req["output_config"]["format"]["schema"]["additionalProperties"] is False
    assert req["output_config"]["effort"] == "low"
    assert "tool_choice" not in req
    assert "temperature" not in req
    assert "fallbacks" not in req
    [block] = req["system"]
    assert block["text"] == SYSTEM_PROMPT
    assert block["cache_control"] == {"type": "ephemeral"}
    [msg] = req["messages"]
    content = msg["content"]
    for needle in ("Ann Dev", "ann@example.org", "2026-01-01", "[VOTE] KIP-1", "Re: [VOTE] KIP-1"):
        assert needle in content
    assert "I propose KIP-1" in content


def test_parse_valid_rewrites_ids():
    res = parse_response("e1", message(GOOD))
    assert [m.mention_id for m in res.mentions] == ["e1:m1", "e1:m2"]
    assert all(m.email_id == "e1" for m in res.mentions)
    [r] = res.relations
    assert (r.source_mention_id, r.target_mention_id) == ("e1:m1", "e1:m2")
    assert r.vote_value is None and r.binding is None


def test_parse_accepts_sdk_object():
    res = parse_response("e1", sdk_message(GOOD))
    assert len(res.mentions) == 2


def test_parse_vote_values():
    payload = json.loads(json.dumps(GOOD))
    payload["relations"] = [
        {"type": "VOTED", "source": "m1", "target": "m2", "vote_value": -1,
         "binding": True, "confidence": 1.0}
    ]
    [r] = parse_response("e1", message(payload)).relations
    assert r.vote_value == -1 and r.binding is True


def bad(**changes):
    p = json.loads(json.dumps(GOOD))
    p.update(changes)
    return p


@pytest.mark.parametrize(
    "msg",
    [
        message(None, as_text="not json at all"),
        message(None, as_text="[1, 2]"),
        message(bad(mentions="nope")),
        message(bad(relations=[{**GOOD["relations"][0], "target": "m99"}])),
        message(bad(relations=[{**GOOD["relations"][0], "source": "m99"}])),
        message(bad(relations=[{**GOOD["relations"][0], "type": "LIKES"}])),
        message(bad(relations=[{**GOOD["relations"][0], "vote_value": 2}])),
        message(bad(mentions=GOOD["mentions"] + [GOOD["mentions"][0]])),
        message(bad(mentions=[{**GOOD["mentions"][0], "type": "Dog"}, GOOD["mentions"][1]])),
        message(bad(mentions=[{**GOOD["mentions"][0], "extra": 1}, GOOD["mentions"][1]])),
        message(GOOD, stop_reason="refusal"),
        message(GOOD, stop_reason="max_tokens"),
        {"stop_reason": "end_turn", "content": []},
    ],
)
def test_parse_rejects(msg):
    with pytest.raises(ExtractionError):
        parse_response("e1", msg)


@pytest.mark.parametrize("raw,clamped", [(1.5, 1.0), (-0.2, 0.0), (0.4, 0.4)])
def test_parse_clamps_confidence(raw, clamped):
    p = bad(relations=[{**GOOD["relations"][0], "confidence": raw}])
    [r] = parse_response("e1", message(p)).relations
    assert r.confidence == clamped


def test_parse_rejects_dangling_relation():
    p = bad(relations=[{**GOOD["relations"][0], "target": "m99"}])
    with pytest.raises(ExtractionError, match="m99"):
        parse_response("e1", message(p))


def test_estimate_cost_arithmetic():
    s = get_settings()
    assert s.pricing[MODEL].input == 2.00 and s.batch_discount == 0.5
    n, avg = 1000, 400
    per = (avg + llm.PROMPT_OVERHEAD_TOKENS) * 2.00 + llm.EST_OUTPUT_TOKENS * 10.00
    expected = n * per / 1_000_000 * llm.COST_SAFETY_MARGIN
    assert llm.COST_SAFETY_MARGIN == 1.3
    assert estimate_cost_usd(n, avg, MODEL, batch=False) == pytest.approx(expected)
    assert estimate_cost_usd(n, avg, MODEL) == pytest.approx(expected * 0.5)
    assert llm.PROMPT_OVERHEAD_TOKENS > 0 and llm.EST_OUTPUT_TOKENS == 600
    assert estimate_cost_usd(0, avg, MODEL) == 0
    assert estimate_cost_usd(10, 100, "claude-haiku-5-5", batch=False) == pytest.approx(
        10 * ((100 + llm.PROMPT_OVERHEAD_TOKENS) * 0.10 + 600 * 0.50) / 1_000_000 * 1.3
    )


def cache_count(conn, eid=None):
    return conn.execute(
        "select count(*) from llm_cache where (%s::text is null or email_id = %s)", (eid, eid)
    ).fetchone()[0]


@pytest.fixture
def clean(pg_conn):
    for t in ("llm_cache", "failures"):
        pg_conn.execute(f"delete from {t}")
    pg_conn.commit()
    return pg_conn


def test_batch_happy_path_stores_cache(clean):
    emails = [make_email("e1"), make_email("e2"), make_email("e3")]
    client = FakeClient(polls_before_end=2)
    stats = run_extraction(clean, emails, client, sleep=no_sleep)
    assert (stats.done, stats.cached, stats.failed) == (3, 0, 0)
    assert client.create_calls == []
    assert client.retrieve_calls == 3
    reqs = client.batch_requests[0]
    assert len(reqs) == 3
    assert all(r["params"]["model"] == MODEL for r in reqs)
    assert cache_count(clean) == 3
    row = clean.execute(
        "select response from llm_cache where email_id='e2' and prompt_version=%s and model=%s",
        (PROMPT_VERSION, MODEL),
    ).fetchone()[0]
    assert row["stop_reason"] == "end_turn"
    assert parse_response("e2", row).mentions[0].mention_id == "e2:m1"


def test_sync_mode(clean):
    client = FakeClient()
    stats = run_extraction(clean, [make_email("s1")], client, batch=False, sleep=no_sleep)
    assert (stats.done, stats.failed) == (1, 0)
    assert len(client.create_calls) == 1 and client.batch_requests == []


def test_batch_ids_unsafe_for_custom_id(clean):
    eid = "<weird id@x.org>/1"
    client = FakeClient()
    stats = run_extraction(clean, [make_email(eid)], client, sleep=no_sleep)
    assert stats.done == 1
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", client.batch_requests[0][0]["custom_id"])
    assert cache_count(clean, eid) == 1


def test_cache_hit_skips_call(clean):
    emails = [make_email("c1")]
    run_extraction(clean, emails, FakeClient(), sleep=no_sleep)
    client = FakeClient()
    stats = run_extraction(clean, emails, client, sleep=no_sleep)
    assert (stats.done, stats.cached, stats.failed) == (0, 1, 0)
    assert client.batch_requests == [] and client.create_calls == []


def test_invalid_then_failure_logged(clean):
    client = FakeClient(
        batch_responder=lambda p: sdk_message(None, as_text="garbage"),
        responder=lambda p: sdk_message(None, as_text="garbage"),
    )
    stats = run_extraction(clean, [make_email("f1")], client, sleep=no_sleep)
    assert (stats.done, stats.failed) == (0, 1)
    assert len(client.create_calls) == 1  # one sync retry
    rows = clean.execute("select stage, item_id, error from failures").fetchall()
    assert len(rows) == 1 and rows[0][:2] == ("extract", "f1")
    assert cache_count(clean) == 0


def test_invalid_then_retry_succeeds(clean):
    client = FakeClient(batch_responder=lambda p: sdk_message(None, as_text="garbage"))
    stats = run_extraction(clean, [make_email("r1")], client, sleep=no_sleep)
    assert (stats.done, stats.failed) == (1, 0)
    assert len(client.create_calls) == 1
    assert cache_count(clean, "r1") == 1


def test_refusal_and_max_tokens_are_failures(clean):
    for reason in ("refusal", "max_tokens"):
        clean.execute("delete from failures")
        clean.commit()
        client = FakeClient(
            batch_responder=lambda p, r=reason: sdk_message(GOOD, stop_reason=r),
            responder=lambda p, r=reason: sdk_message(GOOD, stop_reason=r),
        )
        stats = run_extraction(clean, [make_email("x1")], client, sleep=no_sleep)
        assert stats.failed == 1 and stats.done == 0
        assert cache_count(clean) == 0


def test_errored_batch_item_retried(clean):
    client = FakeClient(batch_responder=lambda p: None)
    stats = run_extraction(clean, [make_email("b1")], client, sleep=no_sleep)
    assert (stats.done, stats.failed) == (1, 0)
    assert len(client.create_calls) == 1


def test_sync_exception_logged_as_failure(clean):
    def boom(params):
        raise RuntimeError("503 upstream")

    client = FakeClient(batch_responder=lambda p: None, responder=boom)
    stats = run_extraction(clean, [make_email("b2")], client, sleep=no_sleep)
    assert stats.failed == 1
    [(err,)] = clean.execute("select error from failures").fetchall()
    assert "503" in err


def test_prompt_version_change_reprocesses(clean, monkeypatch):
    emails = [make_email("p1")]
    run_extraction(clean, emails, FakeClient(), sleep=no_sleep)
    monkeypatch.setattr(llm, "PROMPT_VERSION", "extract-v2")
    client = FakeClient()
    stats = run_extraction(clean, emails, client, sleep=no_sleep)
    assert (stats.done, stats.cached) == (1, 0)
    assert len(client.batch_requests[0]) == 1
    assert cache_count(clean, "p1") == 2


def test_batch_results_committed_before_retry_raises(clean):
    emails = [make_email("k1"), make_email("k2")]
    garbage = sdk_message(None, as_text="garbage")
    # k1's request is last in the (reversed) result stream; make only k2 unparseable
    client = FakeClient(
        batch_responder=lambda p: garbage if "k2" in p["messages"][0]["content"] else sdk_message(GOOD)
    )
    emails[1] = make_email("k2", body="body mentions k2")

    def boom(params):
        raise KeyboardInterrupt

    client.responder = boom
    with pytest.raises(KeyboardInterrupt):
        run_extraction(clean, emails, client, sleep=no_sleep)
    clean.rollback()  # anything not committed is gone
    assert cache_count(clean, "k1") == 1  # the good paid result survived


def test_poll_failure_then_resume_without_new_create(clean, batch_dir):
    emails = [make_email("q1"), make_email("q2")]
    client = FakeClient()
    client.retrieve_raises = True
    with pytest.raises(RuntimeError):
        run_extraction(clean, emails, client, sleep=no_sleep)
    assert len(client.batch_requests) == 1
    assert (batch_dir / "batch_1.json").exists()
    assert cache_count(clean) == 0

    client.retrieve_raises = False
    stats = run_extraction(clean, emails, client, sleep=no_sleep, resume_batch_id="batch_1")
    assert (stats.done, stats.failed) == (2, 0)
    assert len(client.batch_requests) == 1  # no new create
    assert cache_count(clean) == 2


def test_resume_unknown_batch_raises(clean):
    with pytest.raises(FileNotFoundError):
        run_extraction(clean, [make_email("u1")], FakeClient(), resume_batch_id="nope")


def test_invalid_request_error_not_retried(clean):
    client = FakeClient(batch_responder=lambda p: "invalid_request_error")
    stats = run_extraction(clean, [make_email("i1")], client, sleep=no_sleep)
    assert (stats.done, stats.failed) == (0, 1)
    assert client.create_calls == []
    [(err,)] = clean.execute("select error from failures").fetchall()
    assert "invalid_request_error" in err


@pytest.mark.parametrize("kind", ["expired", "canceled", "overloaded_error", "rate_limit_error"])
def test_retryable_batch_outcomes_retried(clean, kind):
    client = FakeClient(batch_responder=lambda p: kind)
    stats = run_extraction(clean, [make_email("v1")], client, sleep=no_sleep)
    assert (stats.done, stats.failed) == (1, 0)
    assert len(client.create_calls) == 1


def test_sync_retry_cap_enforced(clean):
    emails = [make_email(f"z{i}") for i in range(5)]
    client = FakeClient(batch_responder=lambda p: "expired")
    stats = run_extraction(clean, emails, client, sleep=no_sleep, max_sync_retries=2)
    assert len(client.create_calls) == 2
    assert (stats.done, stats.failed) == (2, 3)
    errs = [r[0] for r in clean.execute("select error from failures")]
    assert errs.count("retry cap reached") == 3


def test_default_retry_cap_is_at_least_20(clean):
    emails = [make_email(f"d{i}") for i in range(25)]
    client = FakeClient(batch_responder=lambda p: "expired")
    stats = run_extraction(clean, emails, client, sleep=no_sleep)
    assert len(client.create_calls) == 20
    assert stats.failed == 5


def test_thread_subject_read_from_db(clean):
    clean.execute("delete from threads where thread_id = 'tx'")
    clean.execute("insert into threads (thread_id, subject) values ('tx', 'Original root subject')")
    clean.commit()
    reply = make_email("t9").model_copy(update={"thread_id": "tx", "subject": "Re: something"})
    client = FakeClient()
    run_extraction(clean, [reply], client, sleep=no_sleep)
    content = client.batch_requests[0][0]["params"]["messages"][0]["content"]
    assert "Thread subject: Original root subject" in content
    assert "Subject: Re: something" in content

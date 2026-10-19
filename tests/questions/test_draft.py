import re

import pytest
from langchain_core.runnables import RunnableLambda
from typer.testing import CliRunner

from kgrag import cli
from kgrag.config import get_settings
from kgrag.questions import draft as draft_mod
from kgrag.questions.draft import (
    DRAFT_PROMPT_VERSION,
    DraftQuestion,
    NULL_ANSWER,
    DraftSchema,
    OtherDocsSchema,
    SingleSourceSchema,
    TYPE_INSTRUCTIONS,
    answerable_from_single_source,
    avg_doc_tokens,
    draft_question,
    draft_questions,
    estimate_draft_cost_usd,
    leave_one_out,
    null_question_absent,
)
from kgrag.questions.sample import EvidenceSet, sample_evidence_sets
from kgrag.store.repo import get_documents


def _text(messages) -> str:
    return "\n".join(m.content for m in messages)


class FakeLLM:
    """Stub chat model: dispatches structured calls on the requested schema."""

    def __init__(self, *, key_terms=("protocol", "broker"), answers_alone=(),
                 others_answerable=None, fail=False):
        self.key_terms = key_terms  # list, or callable(draft call number) -> list
        self.answers_alone = set(answers_alone)  # doc ids that answer on their own
        self.others_answerable = others_answerable or (lambda ids: False)
        self.fail = fail
        self.calls: list[tuple[str, str]] = []
        self.kwargs: list[dict] = []

    def with_structured_output(self, schema, **kwargs):
        self.kwargs.append(kwargs)

        def run(messages, config=None):
            text = _text(messages)
            self.calls.append((schema.__name__, text))
            if self.fail:
                raise RuntimeError("API down")
            if schema is DraftSchema:
                terms = self.key_terms(self.count("DraftSchema")) if callable(
                    self.key_terms) else self.key_terms
                return DraftSchema(
                    question=f"Question #{len(self.calls)}?", answer="An answer.",
                    key_terms=list(terms),
                )
            if schema is OtherDocsSchema:
                ids = frozenset(re.findall(r"^\[([^\]]+)\]", text, re.M))
                return OtherDocsSchema(answerable=self.others_answerable(ids))
            assert schema is SingleSourceSchema
            return SingleSourceSchema(
                answerable=any(f"[{d}]" in text for d in self.answers_alone)
            )

        return RunnableLambda(run)

    def count(self, name: str) -> int:
        return sum(1 for n, _ in self.calls if n == name)


def _rows(conn):
    return conn.execute(
        "select type, gold_doc_ids, status, draft_model, prompt_version from questions "
        "order by id"
    ).fetchall()


def test_prompt_version():
    assert DRAFT_PROMPT_VERSION == "draft-v1"


def test_draft_question_builds_from_set_and_docs(corpus):
    es = EvidenceSet(type="inference", doc_ids=["email:e1:0", "email:e2:0"], link="shared:KIP-100")
    docs = get_documents(corpus, es.doc_ids)
    llm = FakeLLM()
    d = draft_question(es, docs, llm)
    assert isinstance(d, DraftQuestion)
    assert (d.type, d.gold_doc_ids, d.key_terms) == (
        "inference", ["email:e1:0", "email:e2:0"], ["protocol", "broker"]
    )
    assert llm.kwargs == [{"method": "json_schema"}]
    sent = llm.calls[0][1]
    assert "[email:e1:0]" in sent and "[email:e2:0]" in sent and "Proposal for KIP-100" in sent


def test_null_draft_gets_topic_context_only(corpus):
    es = EvidenceSet(type="null", doc_ids=[], link="thread:t4")
    llm = FakeLLM()
    d = draft_question(es, [], llm, topic="Subject: [DISCUSS] topic 4\nDiscussion of KIP-200")
    assert d.gold_doc_ids == [] and d.type == "null"
    assert "[DISCUSS] topic 4" in llm.calls[0][1]


def test_single_hop_sampled_first_and_relabels_dropped_when_full(corpus):
    every_doc = {r[0] for r in corpus.execute("select doc_id from documents")}
    llm = FakeLLM(answers_alone=every_doc)
    stats = draft_questions(
        corpus, {"inference": 1, "single_hop": 2}, seed=3, llm=llm, oversample=1.0
    )
    draft_prompts = [t for n, t in llm.calls if n == "DraftSchema"]
    assert "single document below" in draft_prompts[0]  # single_hop drafted first
    rows = corpus.execute("select type, origin from questions").fetchall()
    assert rows == [("single_hop", "sampled")] * 2
    assert stats.relabeled == 0 and stats.dropped_single_source >= 1
    assert stats.kept_by_origin == {"sampled": 2}


def test_relabel_fills_only_remaining_single_hop_headroom(corpus):
    every_doc = {r[0] for r in corpus.execute("select doc_id from documents")}
    llm = FakeLLM(answers_alone=every_doc)
    # 12 eligible docs; a target of 14 leaves headroom of 2 after sampling exhausts them.
    stats = draft_questions(
        corpus, {"inference": 1, "single_hop": 14}, seed=3, llm=llm, oversample=1.0
    )
    assert stats.kept_by_origin == {"sampled": 12, "relabeled": 2}
    assert stats.shortfall == {"single_hop": 2, "inference": 1}
    row = corpus.execute(
        "select type, link, origin from questions where origin = 'relabeled' limit 1"
    ).fetchone()
    assert row[0] == "single_hop" and row[1].startswith("shared:")


def test_multi_hop_kept_when_no_single_doc_answers(corpus):
    llm = FakeLLM()
    stats = draft_questions(corpus, {"inference": 2}, seed=3, llm=llm, oversample=1.0)
    rows = _rows(corpus)
    assert stats.kept == len(rows) >= 1
    assert all(r[0] == "inference" and len(r[1]) >= 2 for r in rows)
    assert llm.count("SingleSourceSchema") >= 2  # one per gold doc


def test_answerable_from_single_source_asks_per_doc(corpus):
    docs = get_documents(corpus, ["email:e1:0", "email:e2:0"])
    d = DraftQuestion(question="q", answer="a", type="inference",
                      gold_doc_ids=["email:e1:0", "email:e2:0"], key_terms=[])
    no, yes = FakeLLM(), FakeLLM(answers_alone={"email:e2:0"})
    assert answerable_from_single_source(d, docs, no) is False
    assert no.count("SingleSourceSchema") == 2
    assert answerable_from_single_source(d, docs, yes) is True


def test_relabel_dropped_when_single_hop_not_requested(corpus):
    every_doc = {r[0] for r in corpus.execute("select doc_id from documents")}
    llm = FakeLLM(answers_alone=every_doc)
    stats = draft_questions(corpus, {"inference": 2}, seed=3, llm=llm, oversample=1.0)
    assert _rows(corpus) == []
    assert stats.dropped_single_source >= 1 and stats.kept == 0


def _docs(conn, *ids):
    return get_documents(conn, list(ids))


def test_leave_one_out_drops_redundant_doc(corpus):
    docs = _docs(corpus, "email:e1:0", "email:e2:0", "email:e3:0")
    d = DraftQuestion(question="q", answer="a", type="inference",
                      gold_doc_ids=[x.doc_id for x in docs], key_terms=[])
    llm = FakeLLM(others_answerable=lambda ids: ids == {"email:e1:0", "email:e2:0"})
    kept = leave_one_out(d, docs, llm)
    assert [x.doc_id for x in kept] == ["email:e1:0", "email:e2:0"]
    assert llm.count("OtherDocsSchema") == 3
    assert llm.kwargs[0] == {"method": "json_schema"}
    none = FakeLLM()
    assert len(leave_one_out(d, docs, none)) == 3
    assert leave_one_out(d, docs[:2], none) == docs[:2]  # 2-doc sets are not re-tested
    assert none.count("OtherDocsSchema") == 3


def test_leave_one_out_shrinks_stored_gold(corpus):
    llm = FakeLLM(others_answerable=lambda ids: len(ids) == 2)
    draft_questions(corpus, {"inference": 6}, seed=3, llm=llm, oversample=1.0)
    golds = [r[1] for r in _rows(corpus)]
    assert golds and all(len(g) == 2 for g in golds)


def test_null_keyword_check(corpus):
    def d(*terms):
        return DraftQuestion(question="q", answer="a", type="null", gold_doc_ids=[],
                             key_terms=list(terms))

    assert null_question_absent(d("zebra", "unicorn"), corpus) is True
    assert null_question_absent(d("broker", "protocol"), corpus) is False  # one doc has both
    assert null_question_absent(d("broker", "unicorn"), corpus) is True  # no doc has both
    assert null_question_absent(d("KIP-100", "rollout"), corpus) is False
    assert null_question_absent(d(), corpus) is False  # nothing to verify against


def test_null_drafts_filtered_by_keyword_check(corpus):
    present = FakeLLM(key_terms=["broker", "protocol"])
    stats = draft_questions(corpus, {"null": 2}, seed=1, llm=present, oversample=1.0)
    assert stats.dropped_null_present == 6 and _rows(corpus) == []  # tops up to the cap (6)
    assert stats.shortfall == {"null": 2}

    absent = FakeLLM(key_terms=["zebra", "unicorn"])
    stats = draft_questions(corpus, {"null": 2}, seed=1, llm=absent, oversample=1.0)
    assert stats.kept == 2
    assert [r[:2] for r in _rows(corpus)] == [("null", []), ("null", [])]
    assert absent.count("SingleSourceSchema") == 0
    assert any("[DISCUSS] topic" in t for _, t in absent.calls)


def test_oversample_scales_draft_count(corpus):
    llm = FakeLLM()
    draft_questions(corpus, {"single_hop": 3}, seed=1, llm=llm, oversample=1.3)
    assert llm.count("DraftSchema") == 4  # ceil(3 * 1.3)


def test_rerun_skips_evidence_already_drafted(corpus):
    llm = FakeLLM()  # question text changes on every call
    draft_questions(corpus, {"single_hop": 3}, seed=1, llm=llm, oversample=1.0)
    stats = draft_questions(corpus, {"single_hop": 3}, seed=1, llm=llm, oversample=1.0)
    assert stats.skipped_existing == 3
    golds = [tuple(r[1]) for r in _rows(corpus)]
    assert len(golds) == 6 and len(set(golds)) == 6
    nulls = FakeLLM(key_terms=["zebra", "unicorn"])
    draft_questions(corpus, {"null": 2}, seed=1, llm=nulls, oversample=1.0)
    again = draft_questions(corpus, {"null": 2}, seed=1, llm=nulls, oversample=1.0)
    links = [r[0] for r in corpus.execute("select link from questions where type = 'null'")]
    assert len(links) == len(set(links)) == 4 and again.skipped_existing == 2


def test_top_up_replaces_rejected_drafts(corpus):
    # Every other draft names terms that exist in the corpus, so the null check rejects it.
    llm = FakeLLM(key_terms=lambda n: ["broker", "protocol"] if n % 2 else ["zebra", "unicorn"])
    stats = draft_questions(corpus, {"null": 3}, seed=2, llm=llm, oversample=1.0)
    assert stats.kept == 3 and stats.drafted == 6 and stats.dropped_null_present == 3
    assert stats.shortfall == {}


def test_shortfall_reported_when_evidence_runs_out(corpus, caplog):
    stats = draft_questions(corpus, {"null": 20}, seed=2, llm=FakeLLM(key_terms=["zebra", "unicorn"]),
                            oversample=1.0)
    assert stats.kept == 9  # nine threads with documents
    assert stats.shortfall == {"null": 11}
    assert "null" in caplog.text


def test_null_answer_is_canonical(corpus):
    draft_questions(corpus, {"null": 1}, seed=1, llm=FakeLLM(key_terms=["zebra", "unicorn"]),
                    oversample=1.0)
    assert corpus.execute("select answer from questions").fetchone()[0] == NULL_ANSWER
    assert "ignored" in TYPE_INSTRUCTIONS["null"] and NULL_ANSWER in TYPE_INSTRUCTIONS["null"]


def test_provenance_and_key_terms_persisted(corpus):
    draft_questions(corpus, {"inference": 1}, seed=3, llm=FakeLLM(), oversample=1.0)
    link, origin, terms = corpus.execute("select link, origin, key_terms from questions").fetchone()
    assert link.startswith("shared:") and origin == "sampled" and terms == ["protocol", "broker"]


def test_too_few_key_terms_is_a_failure(corpus):
    stats = draft_questions(corpus, {"single_hop": 1}, seed=1, llm=FakeLLM(key_terms=["one"]),
                            oversample=1.0)
    assert stats.kept == 0 and stats.failed >= 1


def test_key_terms_truncated_to_four(corpus):
    llm = FakeLLM(key_terms=list("abcdef"))
    draft_questions(corpus, {"single_hop": 1}, seed=1, llm=llm, oversample=1.0)
    assert corpus.execute("select key_terms from questions").fetchone()[0] == list("abcd")


def _drafted_sets(llm, marker):
    """Doc-id tuples of every draft call whose prompt contains marker."""
    return [
        tuple(sorted(re.findall(r"^\[([^\]]+)\]", text, re.M)))
        for name, text in llm.calls if name == "DraftSchema" and marker in text
    ]


def _every_doc(conn):
    return {r[0] for r in conn.execute("select doc_id from documents")}


def test_rerun_after_relabel_does_not_redraft(corpus):
    llm = FakeLLM(answers_alone=_every_doc(corpus))
    counts = {"inference": 1, "single_hop": 13}  # headroom of 1 -> one relabel
    first = draft_questions(corpus, counts, seed=3, llm=llm, oversample=1.0)
    assert first.relabeled == 1
    draft_questions(corpus, counts, seed=3, llm=llm, oversample=1.0)
    sets = _drafted_sets(llm, "combining facts")
    assert sets and len(sets) == len(set(sets))
    outcomes = {r[0] for r in corpus.execute("select outcome from evidence_attempts")}
    assert {"relabeled", "dropped_single_source", "kept"} <= outcomes


def test_rerun_after_leave_one_out_shrink_does_not_redraft(corpus):
    llm = FakeLLM(others_answerable=lambda ids: len(ids) == 2)
    draft_questions(corpus, {"inference": 4}, seed=3, llm=llm, oversample=1.0)
    assert corpus.execute(
        "select count(*) from evidence_attempts where outcome = 'shrunk'"
    ).fetchone()[0] >= 1
    draft_questions(corpus, {"inference": 4}, seed=3, llm=llm, oversample=1.0)
    sets = _drafted_sets(llm, "combining facts")
    assert len(sets) == len(set(sets))


def test_rerun_after_drop_does_not_redraft(corpus):
    llm = FakeLLM(answers_alone=_every_doc(corpus))
    draft_questions(corpus, {"inference": 2}, seed=3, llm=llm, oversample=1.0)
    n_first = len(_drafted_sets(llm, "combining facts"))
    assert n_first >= 1
    draft_questions(corpus, {"inference": 2}, seed=3, llm=llm, oversample=1.0)
    sets = _drafted_sets(llm, "combining facts")
    assert len(sets) == len(set(sets))


def test_failed_attempts_are_retried(corpus):
    draft_questions(corpus, {"single_hop": 1}, seed=1, llm=FakeLLM(fail=True), oversample=1.0)
    failed = corpus.execute(
        "select evidence_doc_ids from evidence_attempts where outcome = 'failed'"
    ).fetchall()
    assert len(failed) == 3  # attempt cap for a target of 1
    llm = FakeLLM()
    stats = draft_questions(corpus, {"single_hop": 1}, seed=1, llm=llm, oversample=1.0)
    assert stats.skipped_existing == 0 and stats.kept == 1
    assert _drafted_sets(llm, "single document")[0] == tuple(failed[0][0])
    assert corpus.execute(
        "select count(*) from evidence_attempts where outcome = 'failed'"
    ).fetchone()[0] == 2


def test_attempt_cap_enforced(corpus, caplog):
    llm = FakeLLM(key_terms=["broker", "protocol"])  # every null draft is rejected
    stats = draft_questions(corpus, {"null": 2}, seed=2, llm=llm, oversample=1.0)
    assert stats.drafted == 6 == llm.count("DraftSchema")  # ceil(3 x target)
    assert stats.kept == 0 and stats.shortfall == {"null": 2}
    assert "cap 6" in caplog.text


def test_guard_compares_worst_case(corpus, cli_env, monkeypatch):
    counts = {"inference": 5, "null": 5}
    avg = avg_doc_tokens(corpus)
    model = get_settings().models.draft
    expected = estimate_draft_cost_usd(counts, avg, model, oversample=1.3)
    worst = estimate_draft_cost_usd(counts, avg, model, oversample=1.3, worst_case=True)
    assert expected < worst
    monkeypatch.setattr(draft_mod, "_default_llm", _boom)
    monkeypatch.setenv("KGRAG_COST_GUARD_USD", str((expected + worst) / 2))
    get_settings.cache_clear()
    res = CliRunner().invoke(cli.app, ["questions", "draft", "--counts", "inference=5,null=5"])
    assert res.exit_code == 2, res.output
    assert "worst-case" in res.output and "expected" in res.output


# --- CLI -------------------------------------------------------------------------------


class _Conn:
    def __init__(self, conn):
        self._c = conn

    def __getattr__(self, name):
        return getattr(self._c, name)

    def close(self):
        pass


@pytest.fixture
def cli_env(corpus, monkeypatch):
    monkeypatch.setattr(cli, "connect", lambda: _Conn(corpus))
    yield
    get_settings.cache_clear()


def _boom():
    raise AssertionError("LLM client must not be constructed")


def test_cost_guard_exit_2_without_client(corpus, cli_env, monkeypatch):
    monkeypatch.setattr(draft_mod, "_default_llm", _boom)
    monkeypatch.setenv("KGRAG_COST_GUARD_USD", "0.0000001")
    get_settings.cache_clear()
    res = CliRunner().invoke(cli.app, ["questions", "draft", "--pilot"])
    assert res.exit_code == 2, res.output
    assert "estimated cost" in res.output.lower()
    assert corpus.execute("select count(*) from questions").fetchone()[0] == 0


def test_cli_counts_with_yes_runs_draft(corpus, cli_env, monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr(draft_mod, "_default_llm", lambda: llm)
    monkeypatch.setenv("KGRAG_COST_GUARD_USD", "0.0000001")
    get_settings.cache_clear()
    res = CliRunner().invoke(
        cli.app, ["questions", "draft", "--counts", "single_hop=2", "--yes", "--seed", "5"]
    )
    assert res.exit_code == 0, res.output
    assert llm.count("DraftSchema") == 3  # ceil(2 * 1.3)
    assert corpus.execute("select count(*) from questions").fetchone()[0] == 3


def test_cli_pilot_uses_six_per_type(corpus, cli_env, monkeypatch):
    seen = {}

    def fake_draft(conn, counts, **kw):
        seen.update(counts=counts, **kw)
        return draft_mod.DraftStats()

    monkeypatch.setattr(cli, "draft_questions", fake_draft)
    res = CliRunner().invoke(cli.app, ["questions", "draft", "--pilot", "--yes"])
    assert res.exit_code == 0, res.output
    assert seen["counts"] == dict.fromkeys(
        ["single_hop", "inference", "comparison", "temporal", "null"], 6
    )
    assert seen["seed"] == 13 and seen["oversample"] == 1.0


@pytest.mark.parametrize("spec", ["bogus=3", "null=-1", "null=1,null=2", "null"])
def test_cli_rejects_bad_counts(corpus, cli_env, spec):
    res = CliRunner().invoke(cli.app, ["questions", "draft", "--counts", spec, "--yes"])
    assert res.exit_code != 0


def test_cli_exits_nonzero_when_every_draft_fails(corpus, cli_env, monkeypatch):
    monkeypatch.setattr(draft_mod, "_default_llm", lambda: FakeLLM(fail=True))
    res = CliRunner().invoke(cli.app, ["questions", "draft", "--counts", "single_hop=2", "--yes"])
    assert res.exit_code == 1, res.output
    assert corpus.execute("select count(*) from failures where stage = 'draft'").fetchone()[0] >= 1

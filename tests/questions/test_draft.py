import re
from datetime import UTC, datetime

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
    BridgeSourceSchema,
    DraftSchema,
    NullCheckSchema,
    OtherDocsSchema,
    SingleAskSchema,
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
from kgrag.store.repo import Document, get_documents


def _text(messages) -> str:
    return "\n".join(m.content for m in messages)


class FakeLLM:
    """Stub chat model: dispatches structured calls on the requested schema."""

    def __init__(self, *, key_terms=("protocol", "broker"), answers_alone=(),
                 others_answerable=None, fail=False, answer_kind="entity", single_ask=True,
                 viable=True, question=None, null_answered=False, answer="An answer.",
                 bridge_clue=None):
        self.key_terms = key_terms  # list, or callable(draft call number) -> list
        self.answers_alone = set(answers_alone)  # doc ids that answer on their own
        self.others_answerable = others_answerable or (lambda ids: False)
        self.fail = fail
        self.answer_kind = answer_kind  # str, or callable(draft call number) -> str
        self.single_ask = single_ask
        self.viable = viable
        self.question = question  # None -> a unique short question per call
        self.null_answered = null_answered  # callable(prompt text) -> bool, or bool
        self.answer = answer
        # bridge filter: docs holding the identifying clue (default: those that answer alone)
        self.bridge_clue = self.answers_alone if bridge_clue is None else set(bridge_clue)
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
                n = self.count("DraftSchema")
                terms = self.key_terms(n) if callable(self.key_terms) else self.key_terms
                kind = self.answer_kind(n) if callable(self.answer_kind) else self.answer_kind
                return DraftSchema(
                    question=f"Question #{len(self.calls)}?" if self.question is None
                    else self.question,
                    answer=self.answer if self.viable else "", key_terms=list(terms), answer_kind=kind,
                    viable=self.viable,
                )
            if schema is SingleAskSchema:
                return SingleAskSchema(single_ask=self.single_ask)
            if schema is NullCheckSchema:
                hit = self.null_answered
                return NullCheckSchema(answered=hit(text) if callable(hit) else hit)
            if schema is OtherDocsSchema:
                ids = frozenset(re.findall(r"^\[([^\]]+)\]", text, re.M))
                return OtherDocsSchema(answerable=self.others_answerable(ids))
            if schema is BridgeSourceSchema:
                doc = re.search(r"^\[([^\]]+)\]", text, re.M).group(1)
                return BridgeSourceSchema(has_clue=doc in self.bridge_clue,
                                          has_fact=doc in self.answers_alone)
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
    assert DRAFT_PROMPT_VERSION == "draft-v5"


def test_draft_question_builds_from_set_and_docs(corpus):
    es = EvidenceSet(type="inference", doc_ids=["email:e1:0", "email:e2:0"], link="bridge:KIP-100",
                     hint="The first document mentions KIP-100 inside a different discussion")
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
    assert "mentions KIP-100 inside a different discussion" in sent  # the hint reaches the model


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
        corpus, {"comparison": 1, "single_hop": 14}, seed=3, llm=llm, oversample=1.0
    )
    assert stats.kept_by_origin == {"sampled": 12, "relabeled": 2}
    assert stats.shortfall == {"single_hop": 2, "comparison": 1}
    row = corpus.execute(
        "select type, link, origin from questions where origin = 'relabeled' limit 1"
    ).fetchone()
    assert row[0] == "single_hop" and row[1].startswith("related:")


def test_multi_hop_kept_when_no_single_doc_answers(corpus):
    llm = FakeLLM()
    stats = draft_questions(corpus, {"inference": 2}, seed=3, llm=llm, oversample=1.0)
    rows = _rows(corpus)
    assert stats.kept == len(rows) >= 1
    assert all(r[0] == "inference" and len(r[1]) >= 2 for r in rows)
    assert llm.count("BridgeSourceSchema") >= 2  # one per gold doc


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


CITING = {"email:e5:0", "email:e6:0", "email:e7:0"}  # fixture docs that cite a bridge id


def _without_one_about_doc(ids):
    return len(ids) == 2 and bool(ids & CITING)


def test_leave_one_out_shrinks_stored_gold(corpus):
    llm = FakeLLM(others_answerable=_without_one_about_doc)
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
    assert link.startswith("bridge:") and origin == "sampled" and terms == ["protocol", "broker"]


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
    counts = {"comparison": 1, "single_hop": 13}  # headroom of 1 -> one relabel
    first = draft_questions(corpus, counts, seed=3, llm=llm, oversample=1.0)
    assert first.relabeled == 1
    draft_questions(corpus, counts, seed=3, llm=llm, oversample=1.0)
    sets = _drafted_sets(llm, "two RELATED items")
    assert sets and len(sets) == len(set(sets))
    outcomes = {r[0] for r in corpus.execute("select outcome from evidence_attempts")}
    assert {"relabeled", "dropped_single_source", "kept"} <= outcomes


def test_rerun_after_leave_one_out_shrink_does_not_redraft(corpus):
    llm = FakeLLM(others_answerable=_without_one_about_doc)
    draft_questions(corpus, {"inference": 4}, seed=3, llm=llm, oversample=1.0)
    assert corpus.execute(
        "select count(*) from evidence_attempts where outcome = 'shrunk'"
    ).fetchone()[0] >= 1
    draft_questions(corpus, {"inference": 4}, seed=3, llm=llm, oversample=1.0)
    sets = _drafted_sets(llm, "BRIDGE question")
    assert len(sets) == len(set(sets))


def test_rerun_after_drop_does_not_redraft(corpus):
    llm = FakeLLM(answers_alone=_every_doc(corpus))
    draft_questions(corpus, {"inference": 2}, seed=3, llm=llm, oversample=1.0)
    n_first = len(_drafted_sets(llm, "BRIDGE question"))
    assert n_first >= 1
    draft_questions(corpus, {"inference": 2}, seed=3, llm=llm, oversample=1.0)
    sets = _drafted_sets(llm, "BRIDGE question")
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
    assert "worst case" in res.output and "expected" in res.output


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


# --- draft-v2 rules ------------------------------------------------------------------------


def _outcomes(conn):
    return [r[0] for r in conn.execute("select outcome from evidence_attempts order by id")]


def test_date_cap_is_share_of_run_target():
    assert draft_mod.date_cap(dict.fromkeys(["a", "b", "c", "d", "e"], 6)) == 4  # 15% of 30
    assert draft_mod.date_cap({"single_hop": 4}) == 0


def test_date_cap_stops_accepting_date_answers(corpus):
    stats = draft_questions(corpus, {"single_hop": 10}, seed=1,
                            llm=FakeLLM(answer_kind="date"), oversample=1.0)
    assert stats.date_cap == 1 and stats.kept == 1 and stats.kept_by_answer_kind == {"date": 1}
    assert stats.dropped_date_cap >= 1 and "dropped_date_cap" in _outcomes(corpus)
    assert corpus.execute("select answer_kind from questions").fetchall() == [("date",)]


def test_date_cap_tops_up_with_other_answers(corpus):
    llm = FakeLLM(answer_kind=lambda n: "date" if n % 2 else "reason")
    stats = draft_questions(corpus, {"single_hop": 4}, seed=1, llm=llm, oversample=1.0)
    assert stats.date_cap == 0 and stats.kept == 4 and stats.dropped_date_cap == 4
    assert stats.kept_by_answer_kind == {"reason": 4} and stats.shortfall == {}


def test_compound_drafts_rejected(corpus):
    llm = FakeLLM(single_ask=False)
    stats = draft_questions(corpus, {"single_hop": 1, "inference": 1}, seed=1, llm=llm,
                            oversample=1.0)
    assert stats.kept == 0 and stats.dropped_compound == 6 and _rows(corpus) == []
    assert set(_outcomes(corpus)) == {"dropped_compound"}
    asks = [t for n, t in llm.calls if n == "SingleAskSchema"]
    assert asks and all("Question #" in t and "ONE thing" in t for t in asks)
    assert llm.count("SingleSourceSchema") == 0  # cheap rejection before the per-doc filter


def test_too_long_drafts_rejected_without_filter_calls(corpus):
    llm = FakeLLM(question=" ".join(["word"] * 31) + "?")
    stats = draft_questions(corpus, {"single_hop": 1}, seed=1, llm=llm, oversample=1.0)
    assert stats.dropped_too_long == 3 and stats.kept == 0
    assert llm.count("SingleAskSchema") == 0


def test_unviable_drafts_dropped_not_failed(corpus):
    llm = FakeLLM(viable=False, key_terms=[], question="")  # empty fields, as the model sends
    stats = draft_questions(corpus, {"comparison": 1}, seed=1, llm=llm, oversample=1.0)
    assert stats.failed == 0 and stats.dropped_unviable >= 1 and stats.kept == 0
    assert set(_outcomes(corpus)) == {"dropped_unviable"}


def test_two_doc_set_dropped_when_either_doc_answers_core(corpus):
    # The about-docs of the 2-doc ticket bridges answer the core question on their own.
    llm = FakeLLM(answers_alone={"jira:1", "jira:2"})
    draft_questions(corpus, {"inference": 6}, seed=1, llm=llm, oversample=1.0)
    rows = corpus.execute(
        "select evidence_doc_ids, outcome from evidence_attempts where link like 'bridge:KAFKA%'"
    ).fetchall()
    assert rows and all(len(ids) == 2 and out == "dropped_single_source" for ids, out in rows)
    assert not corpus.execute(
        "select 1 from questions where gold_doc_ids && array['jira:1', 'jira:2']"
    ).fetchone()
    assert "core of the question" in draft_mod.FILTER_PROMPT


def test_null_premise_from_elsewhere_rejected(corpus):
    q = "Which default did metadata.recovery.rebootstrap.trigger.ms get for KIP-899?"
    assert draft_mod.foreign_identifiers(q, "Subject: KIP-899 rebootstrap") == [
        "metadata.recovery.rebootstrap.trigger.ms"]
    assert draft_mod.foreign_identifiers(q, "Subject: KIP-1102") == [
        "KIP-899", "metadata.recovery.rebootstrap.trigger.ms"]
    ok = "Which release removes numberOfOpenFiles?"
    assert draft_mod.foreign_identifiers(ok, "the number-of-open-files metric") == []
    stats = draft_questions(corpus, {"null": 1}, seed=1, oversample=1.0,
                            llm=FakeLLM(question=q, key_terms=["zebra", "unicorn"]))
    assert stats.dropped_null_premise == 3 and stats.kept == 0


def test_term_pattern_surface_variants():
    import re as _re

    def hit(term, text):
        return _re.search(rf"(?<![a-z0-9]){draft_mod.term_pattern(term)}(?![a-z0-9])", text,
                          _re.IGNORECASE) is not None

    assert hit("intermediate-topics", "the --intermediate-topic option")
    assert hit("intermediate_topics", "intermediateTopics")
    assert hit("intermediateTopic", "intermediate topics")
    assert hit("KIP-1087", "see kip 1087") and not hit("KIP-1087", "KIP-10870")


def test_null_absence_matches_surface_variants(corpus):
    def d(*terms):
        return DraftQuestion(question="q", answer="a", type="null", gold_doc_ids=[],
                             key_terms=list(terms))

    # e8: "Cleaning up KAFKA-700 follow-ups" — present only via the variant regex.
    assert null_question_absent(d("followUps", "kafka_700"), corpus) is False
    assert null_question_absent(d("followUps", "kafka_701"), corpus) is True


def test_null_candidates_cover_specific_term_thread_and_kip_mentions(corpus):
    q = DraftQuestion(question="Which release ships KIP-200?", answer=NULL_ANSWER, type="null",
                      gold_doc_ids=[], key_terms=["zebra", "regressions"])
    ids = {x.doc_id for x in draft_mod.null_candidates(corpus, q, "t7")}
    assert {"email:e4:0", "email:e5:0"} <= ids  # name KIP-200 (in any thread)
    assert "email:e6:0" in ids  # the most specific term alone, via its singular variant
    assert "email:e7:0" in ids  # the whole linked thread
    assert "email:e1:0" not in ids  # matches nothing, other thread
    many = q.model_copy(update={"question": "q", "key_terms": ["protocol", "zebra"]})
    assert len(draft_mod.null_candidates(corpus, many, None)) == 12  # every doc has the filler


def test_null_dropped_when_a_candidate_doc_answers(corpus):
    seen = []

    def answered(text):
        seen.append(text)
        return "[email:e5:0]" in text

    llm = FakeLLM(key_terms=["compatibility", "zebra"], null_answered=answered)
    stats = draft_questions(corpus, {"null": 1}, seed=1, llm=llm, oversample=1.0)
    assert stats.dropped_null_present == 3 and stats.kept == 0 and seen
    keep = FakeLLM(key_terms=["compatibility", "zebra"], null_answered=False)
    corpus.execute("truncate evidence_attempts")
    corpus.commit()
    assert draft_questions(corpus, {"null": 1}, seed=1, llm=keep, oversample=1.0).kept == 1


# --- draft-v3 rules ------------------------------------------------------------------------


@pytest.mark.parametrize("question,answer", [
    ("When did the vote on KIP-1079 pass?", "It passed with three binding votes."),
    ("when was KAFKA-19229 resolved?", "Fixed."),
    ("How long did KAFKA-18773 stay open after creation?", "Two weeks."),
    ("How many days after creation was KAFKA-19229 resolved?", "Seven."),
    ("How many weeks did the KIP-1166 discussion run?", "Three."),
    ("On what date did Jun Rao start the KIP-1166 discussion?", "He started it in a reply."),
    ("Which came first, the KIP-848 vote or the 3.7.0 release?", "The vote."),
    ("Which Jira ticket was resolved earlier: KAFKA-16935 or KAFKA-17681?", "KAFKA-16935."),
    ("Was the KIP-405 vote held before or after the 3.6.0 code freeze?", "Before it."),
    ("Which ticket did Luke Chen say blocked the release candidate?", "On 2024-05-03."),
    ("What was decided about KIP-500 in the follow-up?", "7 days: created on 2025-05-02 "
     "and resolved on 2025-05-09"),
    ("What did the proposer say about the rollout timing?", "About three weeks later."),
    ("What did the reply say about the timeline?", "March 14, 2024"),
])
def test_date_classifier_positive(question, answer):
    assert draft_mod.is_date_centric(question, answer)


@pytest.mark.parametrize("question,answer", [
    ("When the vote failed, who proposed the alternative design?", "Colin McCabe."),
    ("Which release fixed the ticket Luke Chen said blocked the release candidate?",
     "Kafka 3.7.1"),
    ("Who objected to the KIP-1166 high-watermark change?", "Jun Rao objected."),
    ("Why was the KIP-405 design changed after the first vote?",
     "Because the RLMM API leaked implementation details."),
    ("Which release shipped the fix?", "Kafka 3.7.0, released in March 2024."),
    ("How many brokers does the KIP-1150 benchmark use?", "Six."),
    ("Which approach did the later reply prefer?", "The first one, a pull-based design."),
])
def test_date_classifier_negative(question, answer):
    assert not draft_mod.is_date_centric(question, answer)


def _question(conn, qid, question, answer, kind, status="draft"):
    conn.execute(
        "insert into questions (id, type, question, answer, gold_doc_ids, status, answer_kind) "
        "values (%s, 'single_hop', %s, %s, '{}', %s, %s)", (qid, question, answer, status, kind),
    )


def test_date_cap_seeded_from_existing_questions(corpus):
    _question(corpus, "q1", "Which release fixed it?", "3.7.0", "date")  # stored label
    _question(corpus, "q2", "How many days after creation was KAFKA-1 resolved?", "7", None)
    _question(corpus, "q3", "When was KAFKA-2 created?", "In May", "entity")  # classifier wins
    _question(corpus, "q4", "Who proposed KIP-1?", "Jun Rao", None)
    _question(corpus, "q5", "When was KIP-2 accepted?", "Late", None, status="dropped")
    _question(corpus, "q6", "Who wrote KIP-3?", "Ann", "entity", status="kept")
    corpus.commit()
    assert draft_mod.existing_date_counts(corpus) == (5, 3)
    # 5 existing + 8 targeted -> cap floor(0.15 * 13) = 1, already exceeded by the 3 stored.
    stats = draft_questions(corpus, {"single_hop": 8}, seed=1,
                            llm=FakeLLM(answer_kind="date"), oversample=1.0)
    assert stats.date_cap == 1 and stats.dates_existing == 3
    assert stats.kept == 0 and stats.dropped_date_cap >= 8


def test_date_cap_counts_classifier_matches_regardless_of_label(corpus):
    llm = FakeLLM(answer_kind="entity", question="How long did the change take to land?")
    stats = draft_questions(corpus, {"single_hop": 10}, seed=1, llm=llm, oversample=1.0)
    assert stats.date_cap == 1 and stats.kept_by_answer_kind == {"date": 1}
    assert corpus.execute("select answer_kind from questions").fetchall() == [("date",)]
    assert stats.dropped_date_cap >= 1


def test_jira_single_hop_when_questions_rejected(corpus):
    llm = FakeLLM(question="When was the ticket resolved?")
    draft_questions(corpus, {"single_hop": 12}, seed=1, llm=llm, oversample=1.0)
    rows = dict(corpus.execute(
        "select evidence_doc_ids[1], outcome from evidence_attempts"
    ).fetchall())
    assert rows["jira:1"] == rows["jira:2"] == "dropped_jira_date"
    assert all(o != "dropped_jira_date" for d, o in rows.items() if d.startswith("email:"))


def _dq(question, answer, qtype="single_hop", kind="entity", gold=("email:e1:0",)):
    return DraftQuestion(question=question, answer=answer, type=qtype, gold_doc_ids=list(gold),
                         key_terms=["a", "b"], answer_kind=kind)


def test_giveaway_answer_in_question(corpus):
    docs = get_documents(corpus, ["email:e1:0"])
    es = EvidenceSet(type="single_hop", doc_ids=["email:e1:0"], link="doc:email:e1:0")
    g = draft_mod.giveaway
    assert g(_dq("Did Jun Rao, not Colin, object to KIP-1?", "Jun Rao"), es, docs) == "answer"
    assert g(_dq("Which release, Kafka 4.3.0 or 4.2.0, fixed it?", "Kafka 4.3.0"), es, docs)
    # id-like answer, matched by surface variant (the Kip405.tla case)
    assert g(_dq("Which KIP does the Kip405.tla spec model?", "KIP-405"), es, docs) == "answer_id"
    assert g(_dq("Which config replaced log.retention.ms?", "log.retention.ms"), es, docs)
    # not giveaways: yes/no answers, an answer that only shares its subject id, paraphrase
    assert g(_dq("Did Jun Rao object?", "Yes", kind="yes_no"), es, docs) is None
    assert g(_dq("Which release fixed KAFKA-500?", "KAFKA-500 was fixed in 3.7.0"), es, docs) \
        is None
    assert g(_dq("Who proposed the broker protocol rollout?", "Dev"), es, docs) is None


def test_giveaway_bridge_id_in_question(corpus):
    docs = get_documents(corpus, ["email:e6:0", "email:e1:0"])
    es = EvidenceSet(type="inference", doc_ids=["email:e6:0", "email:e1:0"],
                     link="bridge:KIP-100", hint="")
    d = _dq("Who proposed kip 100, the KIP cited in the regression report?", "Dev",
            qtype="inference", gold=es.doc_ids)
    assert draft_mod.giveaway(d, es, docs) == "bridge_id"
    ok = d.model_copy(update={"question": "Who proposed the KIP cited in the regression report?"})
    assert draft_mod.giveaway(ok, es, docs) is None


def _doc(doc_id, sender, subject, text="Some discussion text."):
    return Document(doc_id=doc_id, source_kind="jira" if doc_id.startswith("jira") else "email",
                    source_id=doc_id, chunk_index=0, text=text, token_count=5,
                    date=datetime(2026, 1, 1, tzinfo=UTC), sender=sender, subject=subject)


def test_giveaway_bridge_title():
    citing = _doc("email:c:0", "Luke Chen", "Re: [VOTE] 3.7.1 RC1")
    about = _doc("jira:1", "Satish Duggana",
                 "KAFKA-900 Remote storage metadata dump tool for log segments")
    es = EvidenceSet(type="inference", doc_ids=[citing.doc_id, about.doc_id],
                     link="bridge:KAFKA-900")

    def g(question):
        return draft_mod.giveaway(_dq(question, "4.0.0", qtype="inference", gold=es.doc_ids),
                                  es, [citing, about])

    assert g("Which release fixed the remote metadata dump tool ticket?") == "bridge_title"
    # Two title words are not enough; the clue from the citing doc is fine.
    assert g("Which release fixed the storage ticket Luke Chen flagged in the RC1 vote?") is None
    assert g("Which release fixed the remote dump ticket?") is None


def test_giveaway_metadata_only_comparison():
    docs = [_doc("jira:1", "A", "KAFKA-1 x"), _doc("jira:2", "B", "KAFKA-2 y")]
    es = EvidenceSet(type="comparison", doc_ids=["jira:1", "jira:2"], link="related:x")

    def g(question):
        return draft_mod.giveaway(_dq(question, "No", qtype="comparison", kind="yes_no",
                                      gold=es.doc_ids), es, docs)

    assert g("Did KAFKA-1 and KAFKA-2 ship in the same Kafka release?") == "metadata_comparison"
    assert g("Did the same person resolve both KAFKA-1 and KAFKA-2?") == "metadata_comparison"
    assert g("Which release shipped fixes for both KAFKA-1 and KAFKA-2?") \
        == "metadata_comparison"
    assert g("Did KAFKA-1 and KAFKA-2 share the same resolution?") == "metadata_comparison"
    assert g("How does the scope of KAFKA-1 differ from KAFKA-2?") is None
    assert g("Did the same person start the vote for both KIPs?") is None


def test_giveaway_long_ngram_copied_from_gold_doc(corpus):
    docs = get_documents(corpus, ["email:e1:0"])
    es = EvidenceSet(type="single_hop", doc_ids=["email:e1:0"], link="doc:email:e1:0")
    copied = _dq("Which KIP covers the testing that Operators should perform, before enabling "
                 "the change?", "KIP-100")
    assert draft_mod.giveaway(copied, es, docs) == "ngram"  # 8 tokens, case/punct differ
    seven = _dq("Which KIP covers the testing that operators should perform first?", "KIP-100")
    assert draft_mod.giveaway(seven, es, docs) is None


def test_giveaway_drafts_dropped(corpus):
    llm = FakeLLM(question="What did the rollout plan across several releases, including "
                           "compatibility notes, say?")
    stats = draft_questions(corpus, {"single_hop": 1}, seed=1, llm=llm, oversample=1.0)
    assert stats.dropped_giveaway == 3 and stats.kept == 0
    assert set(_outcomes(corpus)) == {"dropped_giveaway"}
    assert llm.count("SingleAskSchema") == 0  # deterministic, before any filter call


def test_bridge_filter_needs_clue_and_fact_and_never_relabels(corpus):
    every = _every_doc(corpus)
    # Every doc states the fact, but none carries the identifying clue -> bridges survive.
    llm = FakeLLM(answers_alone=every, bridge_clue=())
    stats = draft_questions(corpus, {"inference": 2}, seed=3, llm=llm, oversample=1.0)
    assert stats.kept == 2 and stats.dropped_single_source == 0
    prompts = [t for n, t in llm.calls if n == "BridgeSourceSchema"]
    assert prompts and all("different discussion" in t for t in prompts)  # hint is passed
    assert llm.count("SingleSourceSchema") == 0  # bridges never use the core check
    # A doc with both clue and fact answers alone: dropped, even with single-hop headroom.
    corpus.execute("truncate questions, evidence_attempts")
    corpus.commit()
    llm = FakeLLM(answers_alone=every)
    stats = draft_questions(corpus, {"inference": 2, "single_hop": 30}, seed=3, llm=llm,
                            oversample=1.0)
    assert stats.relabeled == 0 and stats.dropped_single_source >= 1
    assert not corpus.execute(
        "select 1 from questions where link like 'bridge:%' and type <> 'inference'"
    ).fetchone()
    assert "relabeled" not in _outcomes(corpus)


def test_comparison_keeps_core_single_source_check(corpus):
    llm = FakeLLM(answers_alone={"email:e4:0"})
    stats = draft_questions(corpus, {"comparison": 1}, seed=3, llm=llm, oversample=1.0)
    assert llm.count("SingleSourceSchema") >= 1 and llm.count("BridgeSourceSchema") == 0
    assert stats.dropped_single_source >= 1


def test_bridge_dropped_when_citing_doc_is_redundant(corpus):
    # The about-docs alone answer: the question is not a bridge any more.
    llm = FakeLLM(others_answerable=lambda ids: not ids & CITING)
    draft_questions(corpus, {"inference": 6}, seed=3, llm=llm, oversample=1.0)
    three = corpus.execute(
        "select outcome from evidence_attempts where cardinality(evidence_doc_ids) = 3"
    ).fetchall()
    assert three and {o for (o,) in three} == {"dropped_shrunk"}
    assert all(len(r[1]) == 2 for r in _rows(corpus))


def test_temporal_prompt_asks_what_changed():
    text = TYPE_INSTRUCTIONS["temporal"]
    assert "what changed" in text and "why it was changed" in text
    assert "ORIGINAL" in text and "REVISION" in text
    assert "comparing the two" in text
    assert "Never ask which came first, when something happened or how long" in text


def test_attempts_recorded_with_prompt_version(corpus):
    draft_questions(corpus, {"single_hop": 1}, seed=1, llm=FakeLLM(), oversample=1.0)
    assert corpus.execute("select prompt_version from evidence_attempts").fetchall() == [
        (DRAFT_PROMPT_VERSION,)]


def test_attempts_per_kept_defaults_and_history(corpus):
    assert draft_mod.attempts_per_kept(corpus) == draft_mod.DEFAULT_ATTEMPTS_PER_KEPT
    assert draft_mod.DEFAULT_ATTEMPTS_PER_KEPT == {
        "single_hop": 1.3, "null": 1.8, "comparison": 2.0, "inference": 3.5, "temporal": 3.5}
    rows = (
        [("comparison", "kept", DRAFT_PROMPT_VERSION)] * 4 + [("comparison", "dropped_unviable", DRAFT_PROMPT_VERSION)]
        * 6  # 10 attempts, 4 kept -> 2.5
        + [("null", "dropped_null_present", DRAFT_PROMPT_VERSION)] * 10  # nothing kept -> the attempt cap
        + [("temporal", "kept", "draft-v2")] * 20  # other prompt version: ignored
        + [("single_hop", "kept", DRAFT_PROMPT_VERSION)] * 3  # too few attempts: default
    )
    for n, (t, outcome, version) in enumerate(rows):
        corpus.execute(
            "insert into evidence_attempts (type, link, evidence_doc_ids, outcome, "
            "prompt_version) values (%s, %s, '{}', %s, %s)", (t, f"l{n}", outcome, version),
        )
    corpus.commit()
    f = draft_mod.attempts_per_kept(corpus)
    assert f["comparison"] == 2.5 and f["null"] == draft_mod.ATTEMPT_CAP_FACTOR
    assert f["temporal"] == 3.5 and f["single_hop"] == 1.3


def test_cost_estimate_scales_with_attempts_per_kept():
    model = get_settings().models.draft
    one = dict.fromkeys(draft_mod.DEFAULT_ATTEMPTS_PER_KEPT, 1.0)
    base = estimate_draft_cost_usd({"comparison": 6}, 400, model, oversample=1.0, factors=one)
    double = estimate_draft_cost_usd({"comparison": 6}, 400, model, oversample=1.0)
    assert double == pytest.approx(2.0 * base)  # default comparison factor 2.0
    worst = estimate_draft_cost_usd({"comparison": 6}, 400, model, oversample=1.0,
                                    worst_case=True)
    assert worst == pytest.approx(3.0 * base)
    # a factor above the attempt cap is capped: expected never exceeds the worst case
    capped = estimate_draft_cost_usd({"inference": 6}, 400, model, oversample=1.0)
    assert capped == pytest.approx(
        estimate_draft_cost_usd({"inference": 6}, 400, model, oversample=1.0, worst_case=True))
    est = draft_mod.estimate_draft_costs({"comparison": 6, "single_hop": 6}, 400, model,
                                         oversample=1.0)
    assert est.low < est.expected < est.high <= est.worst
    line = draft_mod.format_cost_line(est)
    assert re.fullmatch(r"expected \$\d+\.\d\d \(typical range \$\d+\.\d\d–\$\d+\.\d\d\), "
                        r"worst case \$\d+\.\d\d", line)


def test_cli_types_per_type_targets_only_those_types(corpus, cli_env, monkeypatch):
    seen = {}

    def fake_draft(conn, counts, **kw):
        seen.update(counts=counts, **kw)
        return draft_mod.DraftStats()

    monkeypatch.setattr(cli, "draft_questions", fake_draft)
    res = CliRunner().invoke(cli.app, ["questions", "draft", "--types", "inference,temporal",
                                       "--per-type", "4", "--yes"])
    assert res.exit_code == 0, res.output
    assert seen["counts"] == {"inference": 4, "temporal": 4} and seen["oversample"] == 1.0
    assert "typical range" in res.output


def test_cli_dry_run_prints_estimate_without_client(corpus, cli_env, monkeypatch):
    monkeypatch.setattr(draft_mod, "_default_llm", _boom)
    monkeypatch.setattr(cli, "draft_questions", _boom)
    res = CliRunner().invoke(cli.app, ["questions", "draft", "--types", "inference,temporal",
                                       "--per-type", "6", "--dry-run"])
    assert res.exit_code == 0, res.output
    assert "12 questions" in res.output and "worst case" in res.output


@pytest.mark.parametrize("args", [
    ["--types", "inference", "--pilot"],
    ["--types", "inference", "--counts", "null=1"],
    ["--types", "bogus"],
    ["--types", "inference,inference"],
    ["--per-type", "3", "--pilot"],
])
def test_cli_rejects_bad_type_selection(corpus, cli_env, args):
    res = CliRunner().invoke(cli.app, ["questions", "draft", *args, "--yes"])
    assert res.exit_code != 0

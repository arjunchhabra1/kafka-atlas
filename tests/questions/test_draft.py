import pytest
from langchain_core.runnables import RunnableLambda
from typer.testing import CliRunner

from kgrag import cli
from kgrag.config import get_settings
from kgrag.questions import draft as draft_mod
from kgrag.questions.draft import (
    DRAFT_PROMPT_VERSION,
    DraftQuestion,
    DraftSchema,
    SingleSourceSchema,
    answerable_from_single_source,
    draft_question,
    draft_questions,
    null_question_absent,
)
from kgrag.questions.sample import EvidenceSet, sample_evidence_sets
from kgrag.store.repo import get_documents


def _text(messages) -> str:
    return "\n".join(m.content for m in messages)


class FakeLLM:
    """Stub chat model: dispatches structured calls on the requested schema."""

    def __init__(self, *, key_terms=("protocol", "broker"), answers_alone=()):
        self.key_terms = list(key_terms)
        self.answers_alone = set(answers_alone)  # doc ids that answer on their own
        self.calls: list[tuple[str, str]] = []
        self.kwargs: list[dict] = []

    def with_structured_output(self, schema, **kwargs):
        self.kwargs.append(kwargs)

        def run(messages, config=None):
            text = _text(messages)
            self.calls.append((schema.__name__, text))
            if schema is DraftSchema:
                return DraftSchema(
                    question=f"Question #{len(self.calls)}?", answer="An answer.",
                    key_terms=self.key_terms,
                )
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


def test_single_source_filter_relabels(corpus):
    es = sample_evidence_sets(corpus, "inference", 1, 3)[0]
    answering = es.doc_ids[1]  # fake: this doc alone answers the question
    llm = FakeLLM(answers_alone={answering})
    stats = draft_questions(
        corpus, {"inference": 1, "single_hop": 2}, seed=3, llm=llm, oversample=1.0
    )
    assert stats.relabeled == 1
    rows = _rows(corpus)
    assert [r[0] for r in rows] == ["single_hop", "single_hop"]  # 1 relabeled + 1 sampled
    assert [answering] in [r[1] for r in rows]
    assert all(r[2] == "draft" and r[3] == get_settings().models.draft
               and r[4] == DRAFT_PROMPT_VERSION for r in rows)


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


def test_relabel_dropped_when_single_hop_quota_zero(corpus):
    every_doc = {r[0] for r in corpus.execute("select doc_id from documents")}
    llm = FakeLLM(answers_alone=every_doc)
    stats = draft_questions(corpus, {"inference": 2}, seed=3, llm=llm, oversample=1.0)
    assert _rows(corpus) == []
    assert stats.dropped_single_source >= 1 and stats.kept == 0


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
    assert stats.dropped_null_present == 2 and _rows(corpus) == []

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


def test_rerun_does_not_duplicate_ids(corpus):
    for _ in range(2):
        draft_questions(corpus, {"single_hop": 3}, seed=1, llm=FakeLLM(), oversample=1.0)
    assert corpus.execute("select count(*) from questions").fetchone()[0] == 3


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


def test_cli_rejects_bad_counts(corpus, cli_env):
    res = CliRunner().invoke(cli.app, ["questions", "draft", "--counts", "bogus=3", "--yes"])
    assert res.exit_code != 0

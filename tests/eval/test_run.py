import json

import psycopg
import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from neo4j import GraphDatabase
from psycopg import sql

from kgrag.answer.chain import AnswerSchema
from kgrag.eval import run as run_mod
from kgrag.eval.judge import Verdict
from kgrag.eval.run import EvalPreconditionError, run_eval
from kgrag.graph.types import GraphMeta
from kgrag.retrieve.types import Evidence, RetrievalResult
from kgrag.store.db import apply_migrations
from kgrag.store.repo import QuestionRow, insert_question


def _closed_port_driver():
    return GraphDatabase.driver("bolt://127.0.0.1:1", auth=("neo4j", "x"), connection_timeout=2)


def _meta(resolver):
    from datetime import datetime

    return GraphMeta(resolver, datetime(2026, 1, 1), "abc", 1, 1)


def test_graph_eval_fails_fast_when_neo4j_down(tmp_path):
    def boom(*a, **k):
        raise AssertionError("no question may run")

    driver = _closed_port_driver()
    try:
        with pytest.raises(EvalPreconditionError, match="not reachable"):
            run_eval(None, driver, system="graph", split="dev", resolver="exact",
                     results_dir=tmp_path, retriever=boom)
    finally:
        driver.close()
    assert list(tmp_path.iterdir()) == []


def test_graph_eval_rejects_resolver_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mod, "read_graph_meta", lambda d: _meta("exact"))
    with pytest.raises(EvalPreconditionError, match="exact.*user"):
        run_eval(None, object(), system="graph", split="dev", resolver="user", results_dir=tmp_path)
    monkeypatch.setattr(run_mod, "read_graph_meta", lambda d: None)
    with pytest.raises(EvalPreconditionError, match="GraphMeta"):
        run_eval(None, object(), system="graph", split="dev", resolver="user", results_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_test_split_one_shot(tmp_path):
    (tmp_path / "old.json").write_text(json.dumps(
        {"header": {"system": "vector", "resolver": None, "split": "test"}, "questions": []}))
    (tmp_path / "junk.json").write_text("not json")
    with pytest.raises(EvalPreconditionError, match="one-shot"):
        run_eval(None, None, system="vector", split="test", resolver=None, results_dir=tmp_path)


# ---- runs against a real Postgres with fake retriever / LLMs -----------------------------


@pytest.fixture
def eval_conn(pg_conn):
    i = pg_conn.info
    admin = f"postgresql://{i.user}:{i.password}@{i.host}:{i.port}/{i.dbname}"
    name = "kgrag_eval_task20"
    ident = sql.Identifier(name)
    with psycopg.connect(admin, autocommit=True) as a:
        a.execute(sql.SQL("drop database if exists {}").format(ident))
        a.execute(sql.SQL("create database {}").format(ident))
    try:
        with psycopg.connect(admin, dbname=name) as conn:
            apply_migrations(conn)
            yield conn
    finally:
        with psycopg.connect(admin, autocommit=True) as a:
            a.execute(sql.SQL("drop database if exists {} with (force)").format(ident))


def _seed(conn, specs):
    for qid, qtype, split, gold in specs:
        insert_question(conn, QuestionRow(
            id=qid, type=qtype, question=f"question {qid}", answer=f"gold {qid}",
            gold_doc_ids=gold, status="kept", split=split))
    insert_question(conn, QuestionRow(  # never evaluated: wrong status / wrong split
        id="dropped", type="single_hop", question="d", answer="d", gold_doc_ids=["x"],
        status="dropped", split="dev"))
    conn.commit()


def retriever(question, budget):
    return RetrievalResult(evidence=[Evidence(doc_id="d1", text="text one", score=1.0),
                                     Evidence(doc_id="d2", text="text two", score=0.5)],
                           trace={"q": question})


class FakeLLM:
    """Answers or judges; `fail_on` makes the answer call raise for matching questions."""

    def __init__(self, fail_on=(), abstain=False, verdict="correct"):
        self.fail_on, self.abstain, self.verdict = tuple(fail_on), abstain, verdict

    def with_structured_output(self, schema, **kwargs):
        def run(messages, config=None):
            text = next(m.content for m in messages if isinstance(m, HumanMessage))
            if any(f in text for f in self.fail_on):
                raise ValueError("unparseable model output")
            if schema is Verdict:
                return Verdict(verdict=self.verdict)
            return AnswerSchema(answer="an answer", cited_doc_ids=["d1"], insufficient=self.abstain)

        return RunnableLambda(run)


pg = pytest.mark.integration


@pg
def test_results_file_header_fields(eval_conn, tmp_path):
    _seed(eval_conn, [("s1", "single_hop", "dev", ["d1"]), ("m1", "inference", "dev", ["d1", "d9"]),
                      ("n1", "null", "dev", []), ("t1", "single_hop", "test", ["d1"])])
    path = run_eval(eval_conn, None, system="vector", split="dev", resolver="ignored",
                    results_dir=tmp_path, retriever=retriever, answer_llm=FakeLLM(),
                    judge_llm=FakeLLM(verdict="partial"))
    data = json.loads(path.read_text())
    h = data["header"]
    assert {"run_id", "system", "resolver", "split", "models", "prompt_versions", "run_hash",
            "base_run_hash", "build_hash",
            "started_at", "finished_at", "degraded"} <= set(h)
    assert h["resolver"] is None and h["degraded"] is False
    assert h["run_id"].startswith("vector-na-dev-") and path.name == f"{h['run_id']}.json"
    assert h["models"]["judge"] and h["prompt_versions"]["answer"] == "answer-v1"
    assert h["prompt_versions"]["judge"] == "judge-v1"
    qs = {q["question_id"]: q for q in data["questions"]}
    assert set(qs) == {"s1", "m1", "n1"}  # only kept/edited questions of the requested split
    for q in qs.values():
        assert {"question_id", "type", "retrieved_doc_ids", "recall", "full_evidence", "answer",
                "verdict", "latency_ms", "tokens_in", "tokens_out"} <= set(q)
    assert qs["s1"]["recall"] == 1.0 and qs["s1"]["full_evidence"] is True
    assert qs["m1"]["recall"] == 0.5 and qs["m1"]["full_evidence"] is False
    assert qs["m1"]["verdict"] == "partial" and qs["m1"]["judge_rationale"] == ""
    assert qs["n1"]["recall"] is None and qs["n1"]["verdict"] == "incorrect"  # answered, not abstained
    assert qs["s1"]["tokens_in"] > 0 and qs["s1"]["tokens_out"] > 0


@pg
def test_null_question_abstaining_is_correct(eval_conn, tmp_path):
    _seed(eval_conn, [("n1", "null", "dev", [])])
    path = run_eval(eval_conn, None, system="vector", split="dev", resolver=None,
                    results_dir=tmp_path, retriever=retriever, answer_llm=FakeLLM(abstain=True),
                    judge_llm=FakeLLM(verdict="incorrect"))
    assert json.loads(path.read_text())["questions"][0]["verdict"] == "correct"


@pg
def test_per_question_error_is_recorded_not_fatal(eval_conn, tmp_path):
    specs = [(f"q{i:02d}", "single_hop", "dev", ["d1"]) for i in range(20)]
    _seed(eval_conn, specs)
    path = run_eval(eval_conn, None, system="vector", split="dev", resolver=None,
                    results_dir=tmp_path, retriever=retriever,
                    answer_llm=FakeLLM(fail_on=["question q03"]), judge_llm=FakeLLM())
    data = json.loads(path.read_text())
    bad = next(q for q in data["questions"] if q["question_id"] == "q03")
    assert bad["verdict"] == "error" and "unparseable" in bad["error"]
    assert sum(q["verdict"] == "correct" for q in data["questions"]) == 19  # run continued
    assert data["header"]["n_errors"] == 1 and data["header"]["degraded"] is False


@pg
def test_degraded_flag_when_over_ten_percent_error(eval_conn, tmp_path):
    _seed(eval_conn, [(f"q{i:02d}", "single_hop", "dev", ["d1"]) for i in range(10)])

    def flaky(question, budget):
        if question.endswith(("q00", "q01")):  # 2 of 10 retrievals raise
            raise RuntimeError("pgvector down")
        return retriever(question, budget)

    path = run_eval(eval_conn, None, system="vector", split="dev", resolver=None,
                    results_dir=tmp_path, retriever=flaky, answer_llm=FakeLLM(), judge_llm=FakeLLM())
    data = json.loads(path.read_text())
    assert data["header"]["degraded"] is True and data["header"]["n_errors"] == 2
    err = next(q for q in data["questions"] if q["question_id"] == "q00")
    # a failed retrieval counts against the system: recall 0, not None
    assert err["verdict"] == "error" and err["recall"] == 0.0 and err["full_evidence"] is False
    assert "retrieve" in err["error"]
    assert not list(tmp_path.glob("*.partial.jsonl"))  # removed once the final file is written


@pg
def test_partial_jsonl_written_per_question_and_kept_on_crash(eval_conn, tmp_path):
    _seed(eval_conn, [(f"q{i}", "single_hop", "dev", ["d1"]) for i in range(3)])

    def crash_on_third(question, budget):
        if question.endswith("q2"):
            raise KeyboardInterrupt
        return retriever(question, budget)

    with pytest.raises(KeyboardInterrupt):
        run_eval(eval_conn, None, system="vector", split="dev", resolver=None,
                 results_dir=tmp_path, retriever=crash_on_third, answer_llm=FakeLLM(),
                 judge_llm=FakeLLM())
    assert not list(tmp_path.glob("*.json"))  # no final file...
    lines = next(tmp_path.glob("*.partial.jsonl")).read_text().splitlines()  # ...but the partial
    assert json.loads(lines[0])["header"]["system"] == "vector"
    assert [json.loads(x)["question_id"] for x in lines[1:]] == ["q0", "q1"]


@pg
def test_notimplemented_from_answer_is_a_question_error(eval_conn, tmp_path):
    _seed(eval_conn, [("q1", "single_hop", "dev", ["d1"])])

    class NotImpl(FakeLLM):
        def with_structured_output(self, schema, **kwargs):
            def run(messages, config=None):
                raise NotImplementedError("nope")
            return RunnableLambda(run)

    path = run_eval(eval_conn, None, system="vector", split="dev", resolver=None,
                    results_dir=tmp_path, retriever=retriever, answer_llm=NotImpl(),
                    judge_llm=FakeLLM())
    q = json.loads(path.read_text())["questions"][0]
    assert q["verdict"] == "error" and q["error"].startswith("answer:")


@pg
def test_degraded_test_run_does_not_consume_one_shot(eval_conn, tmp_path):
    _seed(eval_conn, [("t1", "single_hop", "test", ["d1"])])

    def broken(question, budget):
        raise RuntimeError("down")

    kw = dict(split="test", resolver=None, results_dir=tmp_path, answer_llm=FakeLLM(),
              judge_llm=FakeLLM())
    p = run_eval(eval_conn, None, system="vector", retriever=broken, **kw)
    assert json.loads(p.read_text())["header"]["degraded"] is True
    run_eval(eval_conn, None, system="vector", retriever=retriever, **kw)  # allowed
    with pytest.raises(EvalPreconditionError, match="one-shot"):
        run_eval(eval_conn, None, system="vector", retriever=retriever, **kw)


def test_results_dir_anchored_to_project_not_cwd(tmp_path, monkeypatch):
    from kgrag.eval import report, run

    monkeypatch.chdir(tmp_path)
    assert run.RESULTS_DIR.is_absolute() and run.RESULTS_DIR.name == "results"
    assert run.RESULTS_DIR == report.RESULTS_DIR
    assert (run.RESULTS_DIR.parent / "config.yaml").exists()


@pg
def test_second_test_run_needs_force(eval_conn, tmp_path):
    _seed(eval_conn, [("t1", "single_hop", "test", ["d1"])])
    kw = dict(split="test", resolver=None, results_dir=tmp_path, retriever=retriever,
              answer_llm=FakeLLM(), judge_llm=FakeLLM())
    run_eval(eval_conn, None, system="vector", **kw)
    with pytest.raises(EvalPreconditionError, match="one-shot"):
        run_eval(eval_conn, None, system="vector", **kw)
    assert len(list(tmp_path.glob("*.json"))) == 1
    run_eval(eval_conn, None, system="vector", force=True, **kw)


# ---- graph configs as part of the run identity ------------------------------------------


def test_unknown_graph_config_is_a_precondition_error(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mod, "read_graph_meta", lambda d: _meta("user"))
    with pytest.raises(EvalPreconditionError, match="unknown graph config 'nope'"):
        run_eval(None, object(), system="graph", split="dev", resolver="user",
                 graph_config="nope", results_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_describe_and_identity():
    assert run_mod.describe("vector", None, None) == "vector"
    assert run_mod.describe("graph", "user", "hybrid") == "graph-user"
    assert run_mod.describe("graph", "user", "graph-only") == "graph-user (graph-only)"
    assert run_mod.run_identity({"system": "graph", "resolver": "exact"}) == ("graph", "exact", "hybrid")
    assert run_mod.run_identity({"system": "vector", "resolver": None}) == ("vector", None, None)


def _no_freshness_checks(monkeypatch):
    monkeypatch.setattr(run_mod, "read_graph_meta", lambda d: _meta("user"))


@pg
def test_graph_config_is_recorded_and_part_of_the_one_shot_key(eval_conn, tmp_path, monkeypatch):
    from kgrag.retrieve.graph_config import GRAPH_ONLY_CONFIG

    _no_freshness_checks(monkeypatch)
    _seed(eval_conn, [("t1", "single_hop", "test", ["d1"])])
    kw = dict(split="test", resolver="user", results_dir=tmp_path, retriever=retriever,
              answer_llm=FakeLLM(), judge_llm=FakeLLM())
    p = run_eval(eval_conn, object(), system="graph", graph_config="graph-only", **kw)
    h = json.loads(p.read_text())["header"]
    assert h["graph_config"] == "graph-only"
    assert h["graph_config_hash"] == GRAPH_ONLY_CONFIG.fingerprint()
    assert h["run_id"].startswith("graph-user-graph-only-test-")
    run_eval(eval_conn, object(), system="graph", **kw)  # hybrid is a different run: allowed
    for cfg in ("graph-only", "hybrid"):
        with pytest.raises(EvalPreconditionError, match="one-shot"):
            run_eval(eval_conn, object(), system="graph", graph_config=cfg, **kw)
    v = run_eval(eval_conn, None, system="vector", **{**kw, "resolver": None})
    vh = json.loads(v.read_text())["header"]
    assert vh["graph_config"] is None and vh["graph_config_hash"] is None


@pg
def test_interrupted_test_run_spends_the_one_shot(eval_conn, tmp_path):
    _seed(eval_conn, [(f"t{i}", "single_hop", "test", ["d1"]) for i in range(2)])

    def crash_on_second(question, budget):
        if question.endswith("t1"):
            raise KeyboardInterrupt
        return retriever(question, budget)

    kw = dict(split="test", resolver=None, results_dir=tmp_path, answer_llm=FakeLLM(),
              judge_llm=FakeLLM())
    with pytest.raises(KeyboardInterrupt):
        run_eval(eval_conn, None, system="vector", retriever=crash_on_second, **kw)
    assert list(tmp_path.glob("*.partial.jsonl"))
    with pytest.raises(EvalPreconditionError, match="interrupted.*already"):
        run_eval(eval_conn, None, system="vector", retriever=retriever, **kw)
    run_eval(eval_conn, None, system="vector", retriever=retriever, force=True, **kw)

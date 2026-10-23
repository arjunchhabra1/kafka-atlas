import json

from typer.testing import CliRunner

from kgrag.cli import app


def _run(tmp_path, system, resolver):
    q = {"question_id": "q1", "type": "single_hop", "question": "Q", "gold_answer": "G",
         "gold_doc_ids": ["d1"], "retrieved_doc_ids": ["d1"], "recall": 1.0, "full_evidence": True,
         "answer": {"text": "a", "cited_doc_ids": [], "insufficient": False}, "verdict": "correct",
         "latency_ms": 1.0, "tokens_in": 10, "tokens_out": 5, "error": None}
    h = {"run_id": f"{system}-{resolver}", "system": system, "resolver": resolver, "split": "dev",
         "models": {"answer": "claude-sonnet-5-5", "judge": "claude-opus-5-5"},
         "prompt_versions": {}, "run_hash": "h", "base_run_hash": "h", "build_hash": "b", "n_questions": 1, "n_errors": 0,
         "degraded": False}
    p = tmp_path / f"{h['run_id']}.json"
    p.write_text(json.dumps({"header": h, "questions": [q]}))
    return p


def test_report_cli_skips_judge_check_file(tmp_path):
    paths = [_run(tmp_path, "vector", None), _run(tmp_path, "graph", "exact")]
    check = tmp_path / "judge_check.json"
    check.write_text(json.dumps({"run_id": "r", "items": []}))
    result = CliRunner().invoke(app, ["report", "--split", "dev", *map(str, paths), str(check)])
    assert result.exit_code == 0, result.output
    assert "Kafka Atlas evaluation report" in result.output


def test_report_cli_errors_cleanly_without_matching_split(tmp_path):
    paths = [_run(tmp_path, "vector", None)]  # a dev run; default split is test
    result = CliRunner().invoke(app, ["report", *map(str, paths)])
    assert result.exit_code == 1 and "cannot build report" in result.output
    assert "Traceback" not in result.output
    ok = CliRunner().invoke(app, ["report", "--split", "dev", *map(str, paths)])
    assert ok.exit_code == 0, ok.output


def test_eval_cli_validates_graph_config():
    bad = CliRunner().invoke(app, ["eval", "--system", "graph", "--split", "dev", "--resolver",
                                   "user", "--graph-config", "nope"])
    assert bad.exit_code == 2 and "unknown graph config" in bad.output
    vec = CliRunner().invoke(app, ["eval", "--system", "vector", "--split", "dev",
                                   "--graph-config", "graph-only"])
    assert vec.exit_code == 2 and "graph system only" in vec.output


def test_report_cli_skips_ablation_files(tmp_path):
    paths = [_run(tmp_path, "vector", None)]
    abl = tmp_path / "ablation-20260101T000000Z.json"
    abl.write_text(json.dumps({"header": {"kind": "ablation"}, "results": {}}))
    result = CliRunner().invoke(app, ["report", "--split", "dev", *map(str, paths), str(abl)])
    assert result.exit_code == 0, result.output


def _eval_cli_fakes(monkeypatch, tmp_path, n=3):
    from kgrag import cli
    from kgrag.eval import run as run_mod
    from kgrag.store.repo import QuestionRow

    class Conn:
        def close(self):
            pass

    calls = []
    qs = [QuestionRow(id=f"q{i}", type="single_hop", question="Who proposed KIP-1?", answer="Jane",
                      gold_doc_ids=["d"], status="kept", split="dev") for i in range(n)]
    monkeypatch.setattr(cli, "connect", lambda: Conn())
    monkeypatch.setattr(cli, "apply_migrations", lambda c: [])
    monkeypatch.setattr(run_mod, "load_questions", lambda conn, split: qs)

    def fake_run_eval(conn, driver, **kw):
        calls.append(kw)
        return _run(tmp_path, "vector", None)

    monkeypatch.setattr(run_mod, "run_eval", fake_run_eval)
    return calls


def test_eval_cli_prints_estimate_and_guards_cost(monkeypatch, tmp_path):
    from kgrag.eval import run as run_mod

    calls = _eval_cli_fakes(monkeypatch, tmp_path)
    ok = CliRunner().invoke(app, ["eval", "--system", "vector", "--split", "dev"])
    assert ok.exit_code == 0, ok.output
    assert "3 questions; estimated cost $" in ok.output and len(calls) == 1
    monkeypatch.setattr(run_mod, "estimate_eval_cost_usd", lambda qs: (60.0, 40.0))
    refused = CliRunner().invoke(app, ["eval", "--system", "vector", "--split", "dev"])
    assert refused.exit_code == 2 and "pass --yes" in refused.output and len(calls) == 1
    forced = CliRunner().invoke(app, ["eval", "--system", "vector", "--split", "dev", "--yes"])
    assert forced.exit_code == 0 and "$100.00" in forced.output and len(calls) == 2


def test_estimate_eval_cost_scales_and_skips_judging_null_questions():
    from kgrag.eval.run import estimate_eval_cost_usd
    from kgrag.store.repo import QuestionRow

    def q(i, qtype="single_hop"):
        return QuestionRow(id=f"q{i}", type=qtype, question="Who proposed KIP-1?", answer="Jane",
                           gold_doc_ids=[], status="kept", split="test")

    a1, j1 = estimate_eval_cost_usd([q(0)])
    a2, j2 = estimate_eval_cost_usd([q(0), q(1)])
    assert 0 < a1 and 0 < j1 and a2 == 2 * a1 and j2 == 2 * j1
    an, jn = estimate_eval_cost_usd([q(0), q(1, "null")])
    assert an == a2 and jn == j1  # null questions are judged by rule, not by the judge model

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
         "prompt_versions": {}, "config_hash": "h", "n_questions": 1, "n_errors": 0,
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

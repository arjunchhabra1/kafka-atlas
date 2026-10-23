import json

import pytest

from kgrag.eval.report import render_report


def _q(qid, qtype, recall, full, verdict, gold=("d1",)):
    return {"question_id": qid, "type": qtype, "question": f"Q {qid}", "gold_answer": "G",
            "gold_doc_ids": list(gold), "retrieved_doc_ids": ["d1"], "recall": recall,
            "full_evidence": full, "answer": {"text": f"ans {qid}", "cited_doc_ids": [],
                                              "insufficient": False},
            "verdict": verdict, "latency_ms": 100.0, "tokens_in": 1000, "tokens_out": 50,
            "error": None}


def _write(tmp_path, system, resolver, questions, split="test", stamp="x", **header):
    h = {"run_id": f"{system}-{resolver or 'na'}-{split}-{stamp}", "started_at": stamp, "system": system, "resolver": resolver,
         "split": split, "models": {"answer": "claude-sonnet-5-5", "judge": "claude-opus-5-5"},
         "prompt_versions": {"answer": "answer-v1", "judge": "judge-v1"}, "run_hash": "h", "base_run_hash": "h", "build_hash": "b",
         "n_questions": len(questions),
         "n_errors": sum(q["verdict"] == "error" for q in questions), "degraded": False, **header}
    path = tmp_path / f"{h['run_id']}.json"
    path.write_text(json.dumps({"header": h, "questions": questions}))
    return path


def _questions(better):
    out = []
    for i in range(12):
        qtype = ["single_hop", "inference", "comparison", "temporal"][i % 4]
        good = better or i % 2 == 0
        out.append(_q(f"q{i:02d}", qtype, 1.0 if good else 0.5, good,
                      "correct" if good else "partial", gold=("d1", "d2")))
    out.append(_q("n00", "null", None, None, "correct", gold=()))
    return out


def test_report_tables(tmp_path):
    a = _write(tmp_path, "vector", None, _questions(False))
    b = _write(tmp_path, "graph", "exact", _questions(True))
    md = render_report([b, a])
    assert "| Question group | vector | graph-exact |" in md  # fixed system order
    multi = next(line for line in md.splitlines() if line.startswith("| Multi-hop |"))
    assert "[" in multi and "]" in multi
    import re

    assert re.search(r"\d\.\d\d \[\d\.\d\d, \d\.\d\d\]", multi)
    assert "*(indicative)*" in md
    assert any(line.startswith("| Null") for line in md.splitlines())
    assert "graph-exact minus vector" in md
    assert "DEGRADED" not in md


def test_paired_diff_uses_intersection(tmp_path):
    base = _questions(False)
    a = _write(tmp_path, "vector", None, base)
    graph_q = _questions(True)[:6]  # graph only answered 6 questions
    b = _write(tmp_path, "graph", "exact", graph_q)
    md = render_report([a, b])
    assert "(paired, 6 shared questions)" in md
    paired = md.split("graph-exact minus vector")[1].split("## Cost")[0]
    multi = next(line for line in paired.splitlines() if line.startswith("| Multi-hop |"))
    assert multi.count("(n=4)") == 3  # shared questions 0-5 hold 4 multi-hop ones, never 9 or 13
    assert "n=13" not in paired


def test_errors_excluded_and_degraded_banner(tmp_path):
    qs = _questions(True)
    for q in qs[:3]:
        q.update(verdict="error", answer=None, error="answer: boom", tokens_in=None,
                 tokens_out=None)
    path = _write(tmp_path, "vector", None, qs, degraded=True)
    md = render_report([path])
    assert "DEGRADED RUN: vector" in md
    accuracy = md.split("### Answer accuracy")[1].split("##")[0]
    row = next(line for line in accuracy.splitlines() if line.startswith("| Single-hop |"))
    # errored single_hop questions are excluded from the accuracy n
    n_single = sum(q["type"] == "single_hop" for q in qs)
    n_err_single = sum(q["type"] == "single_hop" and q["verdict"] == "error" for q in qs)
    assert f"(n={n_single - n_err_single})" in row


def test_mixed_splits_rejected(tmp_path):
    a = _write(tmp_path, "vector", None, _questions(False), split="dev")
    b = _write(tmp_path, "graph", "exact", _questions(True), split="test")
    with pytest.raises(ValueError, match="splits"):
        render_report([a, b])


def test_traces_and_judge_section(tmp_path):
    a = _write(tmp_path, "vector", None, _questions(False))
    b = _write(tmp_path, "graph", "exact", _questions(True))
    (tmp_path / "judge_check.json").write_text(json.dumps({"run_id": "r", "items": [
        {"question_id": "q1", "judge": "correct", "human": "correct"},
        {"question_id": "q2", "judge": "correct", "human": "partial"},
        {"question_id": "q3", "judge": "partial", "human": None}]}))
    md = render_report([a, b], trace_examples=2)
    assert "Side-by-side traces" in md and md.count("**Question:**") == 2
    assert "Judge validation" in md and "Agreement 50%" in md and "incomplete" in md
    assert "Judge validation" not in render_report([a, b], judge_check_path=tmp_path / "none.json")


def test_paired_difference_sign_and_value(tmp_path):
    def qs(verdicts):
        return [_q(f"q{i}", "single_hop", 1.0, True, v) for i, v in enumerate(verdicts)]

    vector = _write(tmp_path, "vector", None, qs(["correct", "partial", "partial", "incorrect"]))
    graph = _write(tmp_path, "graph", "exact", qs(["correct", "correct", "partial", "partial"]))
    md = render_report([vector, graph])
    paired = md.split("graph-exact minus vector")[1]
    row = next(line for line in paired.splitlines() if line.startswith("| Single-hop |"))
    assert "+0.25 [" in row.split("|")[4]  # accuracy: mean(.75) - mean(.5), graph minus vector
    assert "+0.00 [+0.00, +0.00]" in row.split("|")[2]  # recall identical


def test_latest_nondegraded_run_per_system_and_split_filter(tmp_path):
    dev = _write(tmp_path, "vector", None, _questions(False), split="dev", stamp="1")
    old = _write(tmp_path, "vector", None, _questions(False), stamp="1")
    new = _write(tmp_path, "vector", None, _questions(False), stamp="2")
    bad = _write(tmp_path, "vector", None, _questions(False), stamp="3", degraded=True)
    md = render_report([dev, old, new, bad], split="test")
    assert "vector-na-test-2" in md and "Ignored runs" in md
    assert "vector-na-test-1" in md.split("Ignored runs")[1].split("\n")[0]
    assert "vector-na-test-3" in md.split("Ignored runs")[1].split("\n")[0]  # degraded ignored
    assert "vector-na-dev-1" not in md
    with pytest.raises(ValueError, match="no result files"):
        render_report([dev], split="test")


def test_mismatch_banner(tmp_path):
    a = _write(tmp_path, "vector", None, _questions(False))
    b = _write(tmp_path, "graph", "exact", _questions(True), base_run_hash="other",
               models={"answer": "m2", "judge": "claude-opus-5-5"},
               graph_meta={"resolver": "exact", "built_at": "t", "build_hash": "older"})
    md = render_report([a, b])
    assert "MISMATCH: graph-exact vs vector: models differ" in md
    assert "MISMATCH: graph-exact vs vector: base_run_hash differ" in md
    assert "graph built with build settings older" in md
    assert "MISMATCH" not in render_report([a])


def test_no_mismatch_for_like_runs_with_different_graph_configs(tmp_path):
    # the run hash differs by graph config; only the system-independent part is compared
    a = _write(tmp_path, "vector", None, _questions(False), run_hash="v")
    b = _write(tmp_path, "graph", "user", _questions(True), run_hash="g", graph_config="graph-only",
               graph_meta={"resolver": "user", "built_at": "t", "build_hash": "b"})
    assert "MISMATCH" not in render_report([a, b])


def test_judge_section_notes_unreported_run(tmp_path):
    a = _write(tmp_path, "vector", None, _questions(False))
    (tmp_path / "judge_check.json").write_text(json.dumps({"run_id": "elsewhere", "items": []}))
    assert "not among the runs reported" in render_report([a])


def test_graph_configs_are_separate_systems(tmp_path):
    a = _write(tmp_path, "vector", None, _questions(False))
    hybrid = _write(tmp_path, "graph", "user", _questions(True), graph_config="hybrid")
    only = _write(tmp_path, "graph", "user", _questions(False), stamp="y", graph_config="graph-only")
    old = _write(tmp_path, "graph", "exact", _questions(True), stamp="z")  # no graph_config field
    md = render_report([only, hybrid, a, old])
    assert "| Question group | vector | graph-exact | graph-user | graph-user (graph-only) |" in md
    assert "graph-user (graph-only) minus vector" in md and "graph-user minus vector" in md
    assert "Ignored runs" not in md  # different configs never supersede each other

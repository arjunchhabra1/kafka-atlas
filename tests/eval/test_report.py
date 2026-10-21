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


def _write(tmp_path, system, resolver, questions, split="test", **header):
    h = {"run_id": f"{system}-{resolver or 'na'}-{split}-x", "system": system, "resolver": resolver,
         "split": split, "models": {"answer": "claude-sonnet-5-5", "judge": "claude-opus-5-5"},
         "prompt_versions": {"answer": "answer-v1", "judge": "judge-v1"}, "config_hash": "h",
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

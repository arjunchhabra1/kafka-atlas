import json

import pytest
from langchain_core.runnables import RunnableLambda

from kgrag.answer.chain import Answer
from kgrag.eval.judge import (
    JUDGE_PROMPT_VERSION,
    Verdict,
    agreement_stats,
    cohens_kappa,
    judge,
    run_judge_check,
)


class FakeJudgeLLM:
    def __init__(self, verdict: str):
        self.verdict, self.kwargs, self.messages = verdict, None, None

    def with_structured_output(self, schema, **kwargs):
        self.kwargs = kwargs
        assert schema is Verdict

        def run(messages, config=None):
            self.messages = messages
            return Verdict(verdict=self.verdict, rationale="because")

        return RunnableLambda(run)


def _ans(text="Use KIP-848.", insufficient=False):
    return Answer(text=text, cited_doc_ids=[], insufficient=insufficient)


def test_prompt_version():
    assert JUDGE_PROMPT_VERSION == "judge-v1"


def test_judge_parses_verdict():
    llm = FakeJudgeLLM("partial")
    assert judge("Which KIP?", "KIP-848 and KIP-932", _ans(), llm) == "partial"
    assert llm.kwargs == {"method": "json_schema"}  # same pattern as the answer chain
    prompt = "\n".join(str(m.content) for m in llm.messages)
    assert "KIP-848 and KIP-932" in prompt and "Use KIP-848." in prompt


def test_null_scored_by_abstention():
    class Boom:
        def with_structured_output(self, *a, **k):
            raise AssertionError("null questions must not call the LLM")

    assert judge("q", "no answer", _ans("I don't know", insufficient=True), Boom(), is_null=True) == "correct"
    assert judge("q", "no answer", _ans("It is X."), Boom(), is_null=True) == "incorrect"


def test_cohens_kappa_hand_computed():
    # 7/10 agree; marginals a: c4 p3 i3, b: c4 p4 i2 -> pe = (16+12+6)/100 = .34
    a = ["c", "c", "p", "i", "c", "p", "i", "i", "c", "p"]
    b = ["c", "p", "p", "i", "c", "c", "i", "p", "c", "p"]
    assert cohens_kappa(a, b) == pytest.approx(6 / 11)  # (0.7 - 0.34) / 0.66
    assert cohens_kappa(a, a) == pytest.approx(1.0)
    assert cohens_kappa(["c", "c"], ["c", "c"]) is None  # chance agreement 1: undefined
    assert cohens_kappa([], []) is None


def _run_file(tmp_path, n=6):
    questions = [{"question_id": f"q{i}", "type": "single_hop", "question": f"Q{i}",
                  "gold_answer": "G", "answer": {"text": f"A{i}"},
                  "verdict": ["correct", "partial", "incorrect"][i % 3]} for i in range(n)]
    questions.append({"question_id": "n0", "type": "null", "question": "N", "gold_answer": "",
                      "answer": {"text": "x"}, "verdict": "correct"})
    questions.append({"question_id": "e0", "type": "single_hop", "question": "E", "gold_answer": "",
                      "answer": None, "verdict": "error"})
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"header": {"run_id": "r1"}, "questions": questions}))
    return path


def test_judge_check_is_resumable_and_blind(tmp_path):
    run, state_path = _run_file(tmp_path), tmp_path / "judge_check.json"
    seen = []

    def ask_two(item):
        assert "verdict" not in item and "judge" not in item  # human must not see the judge label
        if len(seen) == 2:
            return None  # quit after two labels
        seen.append(item["question_id"])
        return "correct"

    state = run_judge_check(run, state_path, 4, ask_two)
    assert len(state["items"]) == 4 and not ({"n0", "e0"} & {i["question_id"] for i in state["items"]})
    assert [i["human"] for i in state["items"]].count("correct") == 2

    seen2 = []
    state = run_judge_check(run, state_path, 4, lambda item: seen2.append(item["question_id"]) or "incorrect")
    assert len(seen2) == 2 and not set(seen) & set(seen2)  # only unlabelled items asked again
    stats = agreement_stats(state["items"])
    assert stats["labelled"] == 4 and 0 <= stats["agreement"] <= 1
    assert json.loads(state_path.read_text())["items"] == state["items"]


def test_judge_check_rejects_other_run(tmp_path):
    run, state_path = _run_file(tmp_path), tmp_path / "judge_check.json"
    state_path.write_text(json.dumps({"run_id": "other", "items": []}))
    with pytest.raises(ValueError, match="other"):
        run_judge_check(run, state_path, 3, lambda item: "correct")

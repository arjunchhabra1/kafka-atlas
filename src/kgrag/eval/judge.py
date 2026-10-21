"""LLM judge, plus the human-vs-judge agreement check that validates it."""

import json
import random
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from kgrag.answer.chain import Answer
from kgrag.config import get_settings
from kgrag.observability import langchain_callbacks

JUDGE_PROMPT_VERSION = "judge-v1"

Label = Literal["correct", "partial", "incorrect"]
LABELS: tuple[Label, ...] = ("correct", "partial", "incorrect")
VERDICT_SCORE: dict[str, float] = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}

JUDGE_SYSTEM_PROMPT = (
    "You grade a system's answer to a question about the Apache Kafka mailing lists against a "
    "reviewed gold answer. Judge only whether the system answer states the facts the gold answer "
    "requires; ignore style, length and citations. "
    "'correct': every required fact is present and nothing contradicts the gold answer. "
    "'partial': some but not all required facts are present, with no contradiction. "
    "'incorrect': the key facts are missing, wrong, or contradict the gold answer, or the system "
    "declined to answer."
)


class Verdict(BaseModel):
    """Structured output requested from the judge."""

    verdict: Label
    rationale: str = ""


def judge(
    question: str,
    gold_answer: str,
    answer: Answer,
    llm: BaseChatModel | None = None,
    *,
    is_null: bool = False,
) -> Label:
    """Grade `answer` against `gold_answer`.

    Null questions (no answer exists in the corpus) are scored by rule, with no LLM call:
    correct iff the system abstained.
    """
    if is_null:
        return "correct" if answer.insufficient else "incorrect"
    if llm is None:
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model=get_settings().models.judge, max_tokens=1000)
    chain = llm.with_structured_output(Verdict, method="json_schema")
    messages = [
        SystemMessage(JUDGE_SYSTEM_PROMPT),
        HumanMessage(
            f"Question: {question}\n\nGold answer: {gold_answer}\n\n"
            f"System answer: {answer.text}"
        ),
    ]
    parsed = chain.invoke(messages, config={"callbacks": langchain_callbacks()})
    return parsed.verdict


# ---- judge validation: human labels vs judge labels -------------------------------------


def cohens_kappa(a: list[str], b: list[str]) -> float | None:
    """Cohen's kappa between two raters; None when undefined (empty, or chance agreement is 1)."""
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if pe == 1:
        return None
    return (po - pe) / (1 - pe)


def agreement_stats(items: list[dict]) -> dict:
    """Agreement % and kappa over the items a human has labelled so far."""
    done = [i for i in items if i.get("human") is not None]
    humans, judges = [i["human"] for i in done], [i["judge"] for i in done]
    agree = sum(h == j for h, j in zip(humans, judges, strict=True))
    return {
        "labelled": len(done),
        "total": len(items),
        "agreement": agree / len(done) if done else None,
        "kappa": cohens_kappa(humans, judges),
    }


def _load_state(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)  # atomic, so an interrupt never corrupts the saved labels


def run_judge_check(
    run_path: Path,
    state_path: Path,
    n: int,
    ask: Callable[[dict], str | None],
    seed: int = 0,
) -> dict:
    """Sample `n` judged answers from a run and collect a human verdict for each.

    Resumable: the sample and every label are saved to `state_path` as they are given. `ask`
    receives the question item (the judge's verdict is deliberately not shown) and returns a
    label, or None to stop. Null questions are excluded: they are scored by rule, not judged.
    """
    run = json.loads(run_path.read_text())
    run_id = run["header"]["run_id"]
    state = _load_state(state_path)
    if state is not None and state["run_id"] != run_id:
        raise ValueError(
            f"{state_path} holds a check for run {state['run_id']}; remove it to start a new one"
        )
    by_id = {q["question_id"]: q for q in run["questions"]}
    if state is None:
        pool = sorted(
            q["question_id"] for q in run["questions"]
            if q["type"] != "null" and q["verdict"] in LABELS
        )
        chosen = random.Random(seed).sample(pool, min(n, len(pool)))
        state = {
            "run_id": run_id, "seed": seed, "n": len(chosen),
            "items": [{"question_id": i, "judge": by_id[i]["verdict"], "human": None}
                      for i in chosen],
        }
        _save_state(state_path, state)
    for item in state["items"]:
        if item["human"] is not None:
            continue
        q = by_id[item["question_id"]]
        label = ask({"question_id": q["question_id"], "question": q.get("question", ""),
                     "gold_answer": q.get("gold_answer", ""), "answer": q["answer"]["text"]})
        if label is None:
            break
        if label not in LABELS:
            raise ValueError(f"invalid label {label!r}")
        item["human"] = label
        _save_state(state_path, state)
    return state

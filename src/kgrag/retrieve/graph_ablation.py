"""Ablations: gold-source recall per question type under named retriever configurations.

This is what the dev-split ablation (`kgrag ablate --split dev`) runs before the headline
run. Example:

    results = ablate(kg, questions, {"hybrid": DEFAULT_CONFIG, "graph-only": GRAPH_ONLY_CONFIG},
                     budget=4000, baselines={"vector": VectorRetriever(...).retrieve})
    print(format_table(results))

`questions` are dicts with `question`, `type` and `gold_doc_ids`, as in
tests/fixtures/dev_questions.json. Questions without gold docs (null questions) are skipped.
Retrieval only: no LLM calls.
"""

from collections.abc import Callable, Mapping

from kgrag.graph.check import MULTI_HOP_TYPES
from kgrag.graph.kg import KnowledgeGraph
from kgrag.retrieve import graph_retriever
from kgrag.retrieve.graph_config import GraphRetrieverConfig
from kgrag.retrieve.types import RetrievalResult

Retrieve = Callable[[str, int], RetrievalResult]
GROUPS = {"single-hop": ("single_hop",), "multi-hop": MULTI_HOP_TYPES}
TYPE_ORDER = ("single_hop", "inference", "comparison", "temporal")


def recall(got: set[str], gold: list[str]) -> float:
    return len(got & set(gold)) / len(gold)


def evaluate(questions: list[dict], retrieve: Retrieve, budget: int) -> dict:
    """One retriever over the questions with gold docs:

    {"recall": {bucket: mean}, "n": {bucket: count}, "unlinked_rate", "fallback_rate"}.
    Buckets are each question type, the groups (single-hop, multi-hop) and "all". The rates
    are the share of questions whose trace says nothing linked / plain vector search was used
    (None for a retriever whose trace has no such fields, e.g. the vector baseline).
    """
    scored, unlinked, fallback, traced = [], 0, 0, 0
    for q in questions:
        if not q["gold_doc_ids"]:
            continue
        res = retrieve(q["question"], budget)
        scored.append((q["type"], recall({e.doc_id for e in res.evidence}, q["gold_doc_ids"])))
        if "unlinked" in res.trace:
            traced += 1
            unlinked += bool(res.trace["unlinked"])
            fallback += bool(res.trace.get("fallback"))
    buckets: dict[str, list[float]] = {"all": [r for _, r in scored]}
    for qtype, r in scored:
        buckets.setdefault(qtype, []).append(r)
        for group, types in GROUPS.items():
            if qtype in types:
                buckets.setdefault(group, []).append(r)
    return {
        "recall": {k: sum(v) / len(v) for k, v in sorted(buckets.items()) if v},
        "n": {k: len(v) for k, v in sorted(buckets.items()) if v},
        "unlinked_rate": unlinked / traced if traced else None,
        "fallback_rate": fallback / traced if traced else None,
    }


def recall_by_type(questions: list[dict], retrieve: Retrieve, budget: int) -> dict[str, float]:
    """Mean recall per question type, per group (single-hop, multi-hop) and overall ("all")."""
    return evaluate(questions, retrieve, budget)["recall"]


def ablate(kg: KnowledgeGraph, questions: list[dict], configs: Mapping[str, GraphRetrieverConfig],
           budget: int, baselines: Mapping[str, Retrieve] | None = None) -> dict[str, dict]:
    """{name: evaluate(...)} for each graph config, then each baseline retriever."""
    out = {}
    for name, cfg in configs.items():
        out[name] = evaluate(
            questions, lambda q, b, cfg=cfg: graph_retriever.retrieve(q, kg, b, cfg), budget)
    for name, fn in (baselines or {}).items():
        out[name] = evaluate(questions, fn, budget)
    return out


def _rate(v: float | None) -> str:
    return "-" if v is None else f"{v:.0%}"


def format_table(results: dict[str, dict], rows: tuple[str, ...] | None = None) -> str:
    """Recall per bucket (with n), then the unlinked and vector-fallback rates per config."""
    names = list(results)
    if rows is None:
        present = {b for r in results.values() for b in r["recall"]}
        rows = tuple(b for b in (*TYPE_ORDER, *GROUPS, "all") if b in present)
    width = max([13, *(len(n) + 2 for n in names)])
    lines = [f"{'type':<14}{'n':>5}" + "".join(f"{n:>{width}}" for n in names)]
    for row in rows:
        n = next((r["n"][row] for r in results.values() if row in r["n"]), 0)
        lines.append(f"{row:<14}{n:>5}" + "".join(
            f"{results[c]['recall'][row]:>{width}.2f}" if row in results[c]["recall"]
            else f"{'-':>{width}}" for c in names))
    for key, label in (("unlinked_rate", "unlinked"), ("fallback_rate", "vector fallback")):
        lines.append(f"{label:<19}" + "".join(f"{_rate(results[c][key]):>{width}}" for c in names))
    return "\n".join(lines)

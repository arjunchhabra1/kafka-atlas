"""Ablations: gold-source recall per question type under named retriever configurations.

This is what the dev-split ablation runs before the headline run. Example:

    results = ablate(kg, questions, {"hybrid": DEFAULT_CONFIG, "graph-only": GRAPH_ONLY_CONFIG},
                     budget=4000, baselines={"vector": VectorRetriever(...).retrieve})
    print(format_table(results))

`questions` are dicts with `question`, `type` and `gold_doc_ids`, as in
tests/fixtures/dev_questions.json. Questions without gold docs (null questions) are skipped.
"""

from collections.abc import Callable, Mapping

from kgrag.graph.check import MULTI_HOP_TYPES
from kgrag.graph.kg import KnowledgeGraph
from kgrag.retrieve import graph_retriever
from kgrag.retrieve.graph_config import GraphRetrieverConfig
from kgrag.retrieve.types import RetrievalResult

Retrieve = Callable[[str, int], RetrievalResult]
GROUPS = {"single-hop": ("single_hop",), "multi-hop": MULTI_HOP_TYPES}


def recall(got: set[str], gold: list[str]) -> float:
    return len(got & set(gold)) / len(gold)


def recall_by_type(questions: list[dict], retrieve: Retrieve, budget: int) -> dict[str, float]:
    """Mean recall per question type, per group (single-hop, multi-hop) and overall ("all")."""
    scored = [(q["type"], recall({e.doc_id for e in retrieve(q["question"], budget).evidence},
                                 q["gold_doc_ids"]))
              for q in questions if q["gold_doc_ids"]]
    buckets: dict[str, list[float]] = {"all": [r for _, r in scored]}
    for qtype, r in scored:
        buckets.setdefault(qtype, []).append(r)
        for group, types in GROUPS.items():
            if qtype in types:
                buckets.setdefault(group, []).append(r)
    return {k: sum(v) / len(v) for k, v in sorted(buckets.items()) if v}


def ablate(kg: KnowledgeGraph, questions: list[dict], configs: Mapping[str, GraphRetrieverConfig],
           budget: int, baselines: Mapping[str, Retrieve] | None = None) -> dict[str, dict[str, float]]:
    """{name: recall_by_type} for each graph config, then each baseline retriever."""
    out = {}
    for name, cfg in configs.items():
        out[name] = recall_by_type(
            questions, lambda q, b, cfg=cfg: graph_retriever.retrieve(q, kg, b, cfg), budget)
    for name, fn in (baselines or {}).items():
        out[name] = recall_by_type(questions, fn, budget)
    return out


def format_table(results: dict[str, dict[str, float]], rows: tuple[str, ...] = ("single-hop", "multi-hop")) -> str:
    names = list(results)
    lines = [f"{'type':<12}" + "".join(f"{n:>13}" for n in names)]
    for row in rows:
        lines.append(f"{row:<12}" + "".join(
            f"{results[n][row]:>13.2f}" if row in results[n] else f"{'-':>13}" for n in names))
    return "\n".join(lines)

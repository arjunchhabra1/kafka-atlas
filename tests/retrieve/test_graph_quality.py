"""Graph vs vector-baseline gold-source recall on the dev questions. Prints a table, asserts nothing.

xfails while graph_retriever.retrieve is the stub. Tune on the fixture/dev split only.
"""

import json
from pathlib import Path

import pytest

from kgrag.graph.check import MULTI_HOP_TYPES
from kgrag.retrieve.vector import VectorRetriever

pytestmark = [pytest.mark.integration, pytest.mark.quality]

QUESTIONS = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "dev_questions.json").read_text())
BUDGET = 4000


def _recall(got: set[str], gold: list[str]) -> float:
    return len(got & set(gold)) / len(gold)


def test_graph_vs_vector_recall(graph_retrieve, fixture_kg, capsys):
    baseline = VectorRetriever(fixture_kg.chunk_store, fixture_kg.conn)
    groups = {"single-hop": ("single_hop",), "multi-hop": MULTI_HOP_TYPES}
    rows = []
    for name, types in groups.items():
        qs = [q for q in QUESTIONS if q["type"] in types and q["gold_doc_ids"]]
        scores = {"graph": [], "vector": []}
        for q in qs:
            scores["graph"].append(_recall(
                {e.doc_id for e in graph_retrieve(q["question"], BUDGET).evidence}, q["gold_doc_ids"]))
            scores["vector"].append(_recall(
                {e.doc_id for e in baseline.retrieve(q["question"], BUDGET).evidence}, q["gold_doc_ids"]))
        rows.append((name, len(qs), sum(scores["graph"]) / len(qs), sum(scores["vector"]) / len(qs)))
    with capsys.disabled():
        print(f"\ngold-source recall within {BUDGET} tokens (dev questions)")
        print(f"{'type':<12}{'n':>3}{'graph':>9}{'vector':>9}")
        for name, n, g, v in rows:
            print(f"{name:<12}{n:>3}{g:>9.2f}{v:>9.2f}")

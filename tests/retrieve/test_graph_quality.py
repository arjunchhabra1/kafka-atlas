"""Graph vs vector-baseline gold-source recall on the dev questions. Prints a table, asserts nothing.

Rows: graph (hybrid: graph + semantic rerank + vector backfill), graph-only (no embedding
signal at all) and the vector baseline. xfails while graph_retriever.retrieve is the stub.
Tune on the fixture/dev split only.
"""

import json
from pathlib import Path

import pytest

from kgrag.retrieve.graph_ablation import GROUPS, ablate, format_table
from kgrag.retrieve.graph_config import DEFAULT_CONFIG, GRAPH_ONLY_CONFIG
from kgrag.retrieve.vector import VectorRetriever

pytestmark = [pytest.mark.integration, pytest.mark.quality]

QUESTIONS = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "dev_questions.json").read_text())
BUDGET = 4000


def test_graph_vs_vector_recall(graph_retrieve, fixture_kg, capsys):
    baseline = VectorRetriever(fixture_kg.chunk_store, fixture_kg.conn)
    results = ablate(fixture_kg, QUESTIONS, {"graph": DEFAULT_CONFIG, "graph-only": GRAPH_ONLY_CONFIG},
                     BUDGET, baselines={"vector": baseline.retrieve})
    n = {g: sum(1 for q in QUESTIONS if q["gold_doc_ids"] and q["type"] in types)
         for g, types in GROUPS.items()}
    with capsys.disabled():
        print(f"\ngold-source recall within {BUDGET} tokens (dev questions; n: {n})")
        print(format_table(results))

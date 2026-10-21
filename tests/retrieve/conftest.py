"""Fixtures for the graph-retriever tests: skip (xfail) cleanly while retrieve() is the stub."""

import json
from pathlib import Path

import pytest

from kgrag.retrieve import graph_retriever

PROBE = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "test_queries.json").read_text())["probe"]


def graph_retriever_implemented(kg) -> bool:
    """False only if the stub raises NotImplementedError; any other outcome means implemented."""
    try:
        graph_retriever.retrieve(PROBE, kg, 100)
    except NotImplementedError:
        return False
    except Exception:  # noqa: BLE001 - implemented; the real tests will surface the error
        return True
    return True


@pytest.fixture(scope="session")
def graph_retrieve(fixture_kg):
    """`retrieve(question, budget_tokens)` bound to the fixture KG, or an xfail while it is the stub."""
    if not graph_retriever_implemented(fixture_kg):
        pytest.xfail("graph retriever not implemented yet (see src/kgrag/retrieve/graph_retriever.py)")
    return lambda question, budget: graph_retriever.retrieve(question, fixture_kg, budget)

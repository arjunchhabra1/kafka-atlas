"""Fixtures for the graph-retriever tests."""

import pytest

from kgrag.retrieve import graph_retriever


@pytest.fixture(scope="session")
def graph_retrieve(fixture_kg):
    """`retrieve(question, budget_tokens)` bound to the fixture KG (default config)."""
    return lambda question, budget: graph_retriever.retrieve(question, fixture_kg, budget)

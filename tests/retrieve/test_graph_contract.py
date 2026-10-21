"""Contract for graph_retriever.retrieve. xfails while it is the NotImplementedError stub."""

import pytest

from kgrag.retrieve.types import RetrievalResult
from kgrag.tokens import count_tokens

pytestmark = pytest.mark.integration

QUESTIONS = [
    "Who proposed KIP-1043 and which release does it affect?",
    "Which Jira ticket did Ismael Juma file?",
]


def _tokens(res: RetrievalResult) -> int:
    return sum(count_tokens(e.text) for e in res.evidence)


@pytest.mark.parametrize("budget", [500, 4000])
def test_respects_budget(graph_retrieve, budget):
    for q in QUESTIONS:
        assert _tokens(graph_retrieve(q, budget)) <= budget


def test_returns_retrieval_result_type(graph_retrieve):
    res = graph_retrieve(QUESTIONS[0], 4000)
    assert isinstance(res, RetrievalResult)
    assert res.evidence, "expected some evidence for an on-topic question"
    assert isinstance(res.trace, dict)


def test_doc_ids_exist(graph_retrieve, fixture_kg):
    for q in QUESTIONS:
        res = graph_retrieve(q, 4000)
        ids = [e.doc_id for e in res.evidence]
        assert len(ids) == len(set(ids)), "duplicate doc_ids"
        docs = {d.doc_id: d.text for d in fixture_kg.documents(ids)}
        assert set(ids) == set(docs)
        assert all(e.text == docs[e.doc_id] for e in res.evidence), "evidence text must be the stored text"


def test_deterministic(graph_retrieve):
    for q in QUESTIONS:
        a, b = graph_retrieve(q, 4000), graph_retrieve(q, 4000)
        assert [(e.doc_id, e.text, e.score) for e in a.evidence] == [
            (e.doc_id, e.text, e.score) for e in b.evidence]


def test_no_entity_question_returns_valid_result(graph_retrieve):
    res = graph_retrieve("What's a good pizza recipe?", 4000)
    assert isinstance(res, RetrievalResult)
    assert _tokens(res) <= 4000

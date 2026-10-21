from datetime import datetime, timezone

import psycopg
import pytest
from fastapi.testclient import TestClient
from neo4j.exceptions import ServiceUnavailable

from kgrag.answer.chain import Answer
from kgrag.api.main import AURA_HINT, Deps, create_app
from kgrag.retrieve.types import Evidence, RetrievalResult


def _row(doc_id: str, text: str):
    return (doc_id, "email", doc_id, 0, text, 10, datetime(2024, 5, 1, tzinfo=timezone.utc),
            f"sender-{doc_id}", f"subject-{doc_id}")


ROWS = {
    "email:a:0": _row("email:a:0", "A" * 500),
    "email:b:0": _row("email:b:0", "short text"),
}


class FakeCursor:
    def __init__(self, ids):
        self.ids = ids

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params):
        self.ids = params[0]

    def fetchall(self):
        return [ROWS[i] for i in self.ids if i in ROWS]


class FakeConn:
    def __init__(self):
        self.closed = False

    def cursor(self):
        return FakeCursor([])

    def close(self):
        self.closed = True


class FakeRetriever:
    def retrieve(self, question, budget_tokens):
        return RetrievalResult(
            evidence=[
                Evidence(doc_id="email:a:0", text="A" * 500, score=0.9),
                Evidence(doc_id="email:b:0", text="short text", score=0.8),
                Evidence(doc_id="email:a:0", text="A" * 500, score=0.7),
            ],
            trace={"k": 3},
        )


def fake_answer(question, result, docs):
    return Answer(text="forty-two", cited_doc_ids=["email:a:0"], insufficient=False)


def make_client(**over):
    conns: list[FakeConn] = []

    def conn_factory():
        conns.append(FakeConn())
        return conns[-1]

    def graph_retrieve(question, kg, budget):
        raise NotImplementedError("stub")

    base = dict(vector_retriever=FakeRetriever(), kg=object(), graph_retrieve=graph_retrieve,
                answer_fn=fake_answer, conn_factory=conn_factory, neo4j_ping=lambda: None)
    base.update(over)
    return TestClient(create_app(Deps(**base))), conns


def test_ask_vector_ok():
    client, conns = make_client()
    r = client.post("/ask", json={"question": "who?", "system": "vector"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "forty-two" and body["insufficient"] is False
    assert body["trace"] == {"k": 3}
    assert body["latency_ms"] >= 0
    assert [s["doc_id"] for s in body["sources"]] == ["email:a:0", "email:b:0"]
    first = body["sources"][0]
    assert first["subject"] == "subject-email:a:0" and first["sender"] == "sender-email:a:0"
    assert first["date"].startswith("2024-05-01")
    assert first["snippet"] == "A" * 300
    assert body["sources"][1]["snippet"] == "short text"
    assert all(c.closed for c in conns) and conns


def test_ask_unknown_system_422():
    client, _ = make_client()
    assert client.post("/ask", json={"question": "q", "system": "bm25"}).status_code == 422


@pytest.mark.parametrize("q", ["", "x" * 1001])
def test_ask_empty_question_422(q):
    client, _ = make_client()
    assert client.post("/ask", json={"question": q, "system": "vector"}).status_code == 422


def test_ask_graph_not_implemented_501():
    client, _ = make_client()
    r = client.post("/ask", json={"question": "q", "system": "graph"})
    assert r.status_code == 501
    assert r.json() == {"detail": "graph retrieval not implemented yet"}


def test_ask_graph_neo4j_down_503_with_hint():
    def boom(question, kg, budget):
        raise ServiceUnavailable("no route")

    client, _ = make_client(graph_retrieve=boom)
    r = client.post("/ask", json={"question": "q", "system": "graph"})
    assert r.status_code == 503
    assert r.json()["neo4j"] == "unavailable" and r.json()["hint"] == AURA_HINT


def test_ask_postgres_down_503():
    def no_pg():
        raise psycopg.OperationalError("connection refused")

    client, _ = make_client(conn_factory=no_pg)
    r = client.post("/ask", json={"question": "q", "system": "vector"})
    assert r.status_code == 503
    assert r.json()["postgres"] == "unavailable"


def test_ask_unexpected_error_500_is_generic():
    class Bad:
        def retrieve(self, question, budget_tokens):
            raise RuntimeError("secret internal detail")

    client, _ = make_client(vector_retriever=Bad())
    r = client.post("/ask", json={"question": "q", "system": "vector"})
    assert r.status_code == 500
    assert "secret" not in r.text


def test_health_ok():
    client, conns = make_client()
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"postgres": "ok", "neo4j": "ok"}
    assert all(c.closed for c in conns)


def test_health_neo4j_down_503_with_hint():
    def down():
        raise ServiceUnavailable("paused")

    client, _ = make_client(neo4j_ping=down)
    r = client.get("/health")
    assert r.status_code == 503
    assert r.json() == {"neo4j": "unavailable", "hint": AURA_HINT}


def test_health_postgres_down_503():
    def no_pg():
        raise psycopg.OperationalError("refused")

    client, _ = make_client(conn_factory=no_pg)
    r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["postgres"] == "unavailable"

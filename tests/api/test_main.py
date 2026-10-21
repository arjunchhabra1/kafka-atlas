from datetime import datetime, timezone
from pathlib import Path

import psycopg
import sqlalchemy.exc
import pytest
from fastapi.testclient import TestClient
from neo4j.exceptions import ServiceUnavailable

from kgrag.answer.chain import Answer
from kgrag.api.main import AURA_HINT, Deps, create_app
from kgrag.retrieve.types import Evidence, RetrievalResult


@pytest.fixture(autouse=True)
def _fresh_settings():
    from kgrag.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


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

    base = dict(vector_retriever=FakeRetriever(), graph_retrieve=FakeRetriever().retrieve,
                answer_fn=fake_answer, conn_factory=conn_factory, neo4j_ping=lambda: None,
                embeddings_configured=lambda: True)
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


def test_ask_graph_ok():
    client, _ = make_client()
    r = client.post("/ask", json={"question": "q", "system": "graph"})
    assert r.status_code == 200 and r.json()["answer"] == "forty-two"


def test_ask_notimplemented_is_a_plain_500():
    def stub(question, budget):
        raise NotImplementedError("x")

    client, _ = make_client(graph_retrieve=stub)
    r = client.post("/ask", json={"question": "q", "system": "graph"})
    assert r.status_code == 500 and r.json() == {"detail": "internal server error"}


def test_ask_graph_neo4j_down_503_with_hint():
    def boom(question, budget):
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
    assert r.json() == {"postgres": "ok", "neo4j": "ok", "embeddings": "ok"}
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


def test_health_embeddings_not_configured_stays_200():
    client, _ = make_client(embeddings_configured=lambda: False)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"postgres": "ok", "neo4j": "ok", "embeddings": "not configured"}


def test_health_any_exception_is_unavailable():
    def weird():
        raise RuntimeError("boom")

    client, _ = make_client(neo4j_ping=weird)
    assert client.get("/health").status_code == 503
    client, _ = make_client(conn_factory=weird)
    r = client.get("/health")
    assert r.status_code == 503 and r.json() == {"postgres": "unavailable"}


@pytest.mark.parametrize("system", ["vector", "graph"])
def test_ask_embeddings_not_configured_503(system):
    from kgrag.retrieve.embeddings import EmbeddingsNotConfiguredError

    class NoEmb:
        def retrieve(self, question, budget_tokens):
            raise EmbeddingsNotConfiguredError("set VOYAGE_API_KEY secret-hint")

    def no_emb(question, budget):
        raise EmbeddingsNotConfiguredError("set VOYAGE_API_KEY secret-hint")

    client, _ = make_client(vector_retriever=NoEmb(), graph_retrieve=no_emb)
    r = client.post("/ask", json={"question": "q", "system": system})
    assert r.status_code == 503
    assert r.json() == {"detail": "embeddings not configured"}


def test_ask_sqlalchemy_operational_error_503():
    class Down:
        def retrieve(self, question, budget_tokens):
            raise sqlalchemy.exc.OperationalError("select 1", {}, Exception("refused"))

    client, _ = make_client(vector_retriever=Down())
    r = client.post("/ask", json={"question": "q", "system": "vector"})
    assert r.status_code == 503 and r.json() == {"postgres": "unavailable"}


def test_ask_insufficient_true():
    def insufficient(question, result, docs):
        return Answer(text="not enough", cited_doc_ids=[], insufficient=True)

    client, _ = make_client(answer_fn=insufficient)
    body = client.post("/ask", json={"question": "q", "system": "vector"}).json()
    assert body["insufficient"] is True and body["answer"] == "not enough"


def test_lifespan_closes_deps():
    closed = []
    client, _ = make_client(close=lambda: closed.append(1))
    with client:
        assert not closed
    assert closed == [1]


def test_default_app_health_without_api_keys(monkeypatch):
    """Importing the app and hitting /health needs no API keys (only reachable DBs, here stubbed)."""
    import kgrag.api.main as m

    for k in ("ANTHROPIC_API_KEY", "VOYAGE_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(k, raising=False)
    closed = []

    class FakeDriver:
        def verify_connectivity(self):
            pass

        def close(self):
            closed.append(1)

    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *a, **k: FakeDriver())
    monkeypatch.setattr("kgrag.store.db.connect", lambda *a, **k: FakeConn())
    with TestClient(m.create_app()) as client:
        r = client.get("/health")
    assert r.status_code == 200 and r.json()["embeddings"] == "not configured"
    assert closed == [1]


def test_default_resources_created_once_under_concurrency(monkeypatch):
    import threading
    import time
    from contextlib import contextmanager

    import kgrag.api.main as m

    calls = {"stores": 0, "driver": 0}

    @contextmanager
    def fake_open_stores(*a, **k):
        calls["stores"] += 1
        time.sleep(0.05)
        yield ("chunks", "nodes")

    class FakeDriver:
        def close(self):
            pass

    def fake_driver(*a, **k):
        calls["driver"] += 1
        time.sleep(0.05)
        return FakeDriver()

    monkeypatch.setattr("kgrag.retrieve.index.open_stores", fake_open_stores)
    monkeypatch.setattr("neo4j.GraphDatabase.driver", fake_driver)
    res = m._Resources()
    threads = [threading.Thread(target=lambda: (res.stores(), res.driver())) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert calls == {"stores": 1, "driver": 1}
    res.close()


def _ask(client, headers=None):
    return client.post("/ask", json={"question": "q", "system": "vector"}, headers=headers or {})


def test_api_key_unset_ask_is_open(monkeypatch):
    monkeypatch.delenv("KGRAG_API_KEY", raising=False)
    client, _ = make_client()
    assert _ask(client).status_code == 200


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}, {"X-API-Key": ""}])
def test_api_key_set_rejects_missing_or_wrong(monkeypatch, headers):
    monkeypatch.setenv("KGRAG_API_KEY", "s3cret")
    client, conns = make_client()
    r = _ask(client, headers)
    assert r.status_code == 401
    assert r.json() == {"detail": "invalid or missing API key"}
    assert conns == []  # rejected before any retrieval or database work


def test_api_key_set_accepts_match(monkeypatch):
    monkeypatch.setenv("KGRAG_API_KEY", "s3cret")
    client, _ = make_client()
    assert _ask(client, {"X-API-Key": "s3cret"}).status_code == 200


def test_api_key_set_health_stays_open(monkeypatch):
    monkeypatch.setenv("KGRAG_API_KEY", "s3cret")
    client, _ = make_client()
    assert client.get("/health").status_code == 200


def test_api_key_compare_is_constant_time(monkeypatch):
    import kgrag.api.main as m

    monkeypatch.setenv("KGRAG_API_KEY", "s3cret")
    seen = []
    real = m.secrets.compare_digest
    monkeypatch.setattr(m.secrets, "compare_digest", lambda a, b: seen.append((a, b)) or real(a, b))
    client, _ = make_client()
    _ask(client, {"X-API-Key": "wrong"})
    assert seen == [(b"wrong", b"s3cret")]


def _cloud(monkeypatch, **env):
    for k in ("KGRAG_API_KEY", "KGRAG_ALLOW_OPEN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("KGRAG_DB_MODE", "cloudsql")
    monkeypatch.setenv("KGRAG_NEO4J_PASSWORD", "prod-pw")
    monkeypatch.setenv("KGRAG_NEO4J_URI", "neo4j+s://x.databases.neo4j.io")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    from kgrag.config import get_settings
    get_settings.cache_clear()


def test_empty_configured_key_refuses_startup(monkeypatch):
    monkeypatch.setenv("KGRAG_API_KEY", "")
    with pytest.raises(RuntimeError, match="KGRAG_API_KEY"):
        make_client()


def test_cloudsql_without_key_refuses_startup(monkeypatch):
    _cloud(monkeypatch)
    with pytest.raises(RuntimeError, match="KGRAG_ALLOW_OPEN"):
        make_client()


def test_cloudsql_allow_open_starts(monkeypatch):
    _cloud(monkeypatch, KGRAG_ALLOW_OPEN="1")
    client, _ = make_client()
    assert _ask(client).status_code == 200


def test_cloudsql_with_key_starts_and_enforces(monkeypatch):
    _cloud(monkeypatch, KGRAG_API_KEY="k")
    client, _ = make_client()
    assert _ask(client).status_code == 401
    assert _ask(client, {"X-API-Key": "k"}).status_code == 200


def test_docs_disabled_when_key_configured(monkeypatch):
    monkeypatch.setenv("KGRAG_API_KEY", "k")
    client, _ = make_client()
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


def test_docs_available_without_key(monkeypatch):
    monkeypatch.delenv("KGRAG_API_KEY", raising=False)
    client, _ = make_client()
    assert client.get("/openapi.json").status_code == 200


def test_serve_binds_loopback_by_default_and_container_opts_in(monkeypatch):
    import uvicorn
    from typer.testing import CliRunner

    from kgrag.cli import app as cli_app

    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda target, host, port: seen.update(host=host, port=port))
    monkeypatch.delenv("PORT", raising=False)
    assert CliRunner().invoke(cli_app, ["serve"]).exit_code == 0
    assert seen == {"host": "127.0.0.1", "port": 8000}
    dockerfile = (Path(__file__).resolve().parents[2] / "Dockerfile").read_text()
    assert 'CMD ["kgrag", "serve", "--host", "0.0.0.0"]' in dockerfile

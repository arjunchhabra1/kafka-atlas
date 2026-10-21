"""HTTP service: `POST /ask` (vector or graph system) and `GET /health`."""

import logging
import os
import secrets
import time
from collections.abc import AsyncIterator, Callable
from contextlib import ExitStack, asynccontextmanager
from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal

import psycopg
import sqlalchemy.exc
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from neo4j.exceptions import ServiceUnavailable, SessionExpired
from pydantic import BaseModel, Field

from kgrag.answer.chain import Answer
from kgrag.config import get_settings, load_env
from kgrag.observability import tag_current_trace, traced
from kgrag.retrieve.embeddings import EmbeddingsNotConfiguredError, embeddings_configured
from kgrag.retrieve.types import RetrievalResult
from kgrag.store import repo

log = logging.getLogger(__name__)

AURA_HINT = "AuraDB Free instances pause after inactivity; resume it in the Aura console."
SNIPPET_CHARS = 300
NEO4J_ERRORS = (ServiceUnavailable, SessionExpired)
PG_ERRORS = (psycopg.OperationalError, psycopg.InterfaceError,
             sqlalchemy.exc.OperationalError, sqlalchemy.exc.DBAPIError)
NEO4J_TIMEOUT_S = 5


@dataclass
class Deps:
    """Everything the endpoints need; tests inject fakes.

    `vector_retriever` has `.retrieve(question, budget_tokens)`. `graph_retrieve(question, budget)`
    is the graph retriever. `answer_fn(question, result, docs)` produces the Answer.
    `conn_factory()` returns a fresh Postgres connection (one per request, closed after).
    `neo4j_ping()` raises if Neo4j is unreachable.
    """

    vector_retriever: Any
    graph_retrieve: Callable[[str, int], RetrievalResult]
    answer_fn: Callable[[str, RetrievalResult, dict], Answer]
    conn_factory: Callable[[], psycopg.Connection]
    neo4j_ping: Callable[[], None]
    embeddings_configured: Callable[[], bool] = embeddings_configured
    close: Callable[[], None] = lambda: None


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    system: Literal["vector", "graph"]


class Source(BaseModel):
    doc_id: str
    subject: str
    date: str
    sender: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    insufficient: bool
    sources: list[Source]
    trace: dict
    latency_ms: int


class _Resources:
    """Lazily opened vector stores, Neo4j driver and per-request adapters for the default deps."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._stack = ExitStack()
        self._stores: tuple[Any, Any] | None = None
        self._driver: Any = None

    def driver(self) -> Any:
        with self._lock:
            if self._driver is None:
                from neo4j import GraphDatabase

                s = get_settings()
                self._driver = GraphDatabase.driver(
                    s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password),
                    connection_timeout=NEO4J_TIMEOUT_S, connection_acquisition_timeout=NEO4J_TIMEOUT_S,
                )
            return self._driver

    def stores(self) -> tuple[Any, Any]:
        with self._lock:
            if self._stores is None:
                from kgrag.retrieve.index import open_stores

                self._stores = self._stack.enter_context(open_stores())
            return self._stores

    def close(self) -> None:
        with self._lock:
            self._stack.close()
            self._stores = None
            if self._driver is not None:
                self._driver.close()
                self._driver = None

    def ping(self) -> None:
        self.driver().verify_connectivity()

    def retrieve_vector(self, question: str, budget_tokens: int) -> RetrievalResult:
        from kgrag.retrieve.vector import VectorRetriever
        from kgrag.store.db import connect

        conn = connect()
        try:
            return VectorRetriever(self.stores()[0], conn).retrieve(question, budget_tokens)
        finally:
            conn.close()

    def retrieve_graph(self, question: str, budget_tokens: int) -> RetrievalResult:
        from kgrag.graph.kg import KnowledgeGraph
        from kgrag.retrieve import graph_retriever
        from kgrag.store.db import connect

        conn = connect()
        try:
            chunks, nodes = self.stores()
            kg = KnowledgeGraph(self.driver(), conn, chunks, nodes)
            return graph_retriever.retrieve(question, kg, budget_tokens)
        finally:
            conn.close()


class _VectorAdapter:
    def __init__(self, res: _Resources) -> None:
        self.retrieve = res.retrieve_vector


def _default_answer(question: str, result: RetrievalResult, docs: dict) -> Answer:
    from kgrag.answer.chain import answer

    return answer(question, result, docs=docs)


def default_deps() -> Deps:
    """Cheap to build: nothing connects (or needs API keys) until a request uses it."""
    from kgrag.store.db import connect

    res = _Resources()
    return Deps(
        vector_retriever=_VectorAdapter(res), graph_retrieve=res.retrieve_graph,
        answer_fn=_default_answer, conn_factory=connect, neo4j_ping=res.ping, close=res.close,
    )


def _neo4j_down() -> JSONResponse:
    return JSONResponse({"neo4j": "unavailable", "hint": AURA_HINT}, status_code=503)


def _postgres_down() -> JSONResponse:
    return JSONResponse({"postgres": "unavailable"}, status_code=503)


def require_api_key(x_api_key: str = Header(default="")) -> None:
    """Optional shared secret: when env KGRAG_API_KEY is set, /ask needs a matching X-API-Key."""
    expected = os.environ.get("KGRAG_API_KEY")
    if expected is None:
        return
    # present-but-empty never matches (fail closed); startup also refuses to run that way
    if not expected or not secrets.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def check_access_config() -> bool:
    """Refuse unsafe auth configuration at startup; returns True when /ask is key-protected."""
    key = os.environ.get("KGRAG_API_KEY")
    if key is not None and not key.strip():
        raise RuntimeError("KGRAG_API_KEY is set but empty; unset it or give it a value")
    if key is None and get_settings().db_mode == "cloudsql" and os.environ.get("KGRAG_ALLOW_OPEN") != "1":
        raise RuntimeError(
            "cloudsql mode requires KGRAG_API_KEY (or KGRAG_ALLOW_OPEN=1 to run /ask unauthenticated)"
        )
    return key is not None


def create_app(deps: Deps | None = None) -> FastAPI:
    load_env()
    protected = check_access_config()
    d = deps or default_deps()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            d.close()

    hidden = protected or get_settings().db_mode == "cloudsql"  # no public API docs on a deployed service
    app = FastAPI(
        title="Kafka Atlas", lifespan=lifespan,
        docs_url=None if hidden else "/docs", redoc_url=None if hidden else "/redoc",
        openapi_url=None if hidden else "/openapi.json",
    )

    @traced("api.ask")
    def run_ask(req: AskRequest) -> AskResponse:
        tag_current_trace(system=req.system)
        start = time.perf_counter()
        budget = get_settings().budget_tokens
        if req.system == "vector":
            result = d.vector_retriever.retrieve(req.question, budget)
        else:
            result = d.graph_retrieve(req.question, budget)
        conn = d.conn_factory()
        try:
            ids = list(dict.fromkeys(e.doc_id for e in result.evidence))
            docs = {x.doc_id: x for x in repo.get_documents(conn, ids)}
        finally:
            conn.close()
        out = d.answer_fn(req.question, result, docs)
        sources = [
            Source(doc_id=i, subject=docs[i].subject, date=docs[i].date.isoformat(),
                   sender=docs[i].sender, snippet=docs[i].text[:SNIPPET_CHARS])
            for i in ids if i in docs
        ]
        return AskResponse(
            answer=out.text, insufficient=out.insufficient, sources=sources, trace=result.trace,
            latency_ms=int((time.perf_counter() - start) * 1000),
        )

    @app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
    def ask(req: AskRequest):
        try:
            return run_ask(req)
        except NEO4J_ERRORS:
            log.warning("neo4j unavailable during /ask", exc_info=True)
            return _neo4j_down()
        except EmbeddingsNotConfiguredError:
            return JSONResponse({"detail": "embeddings not configured"}, status_code=503)
        except PG_ERRORS:
            log.warning("postgres unavailable during /ask", exc_info=True)
            return _postgres_down()
        except Exception:  # noqa: BLE001 - never leak internals to clients
            log.exception("unexpected error in /ask")
            return JSONResponse({"detail": "internal server error"}, status_code=500)

    @app.get("/health")
    def health():
        embeddings = "ok" if d.embeddings_configured() else "not configured"
        try:
            d.conn_factory().close()
        except Exception:  # noqa: BLE001 - any failure means the dependency is unavailable
            log.warning("postgres health check failed", exc_info=True)
            return _postgres_down()
        try:
            d.neo4j_ping()
        except Exception:  # noqa: BLE001
            log.warning("neo4j health check failed", exc_info=True)
            return _neo4j_down()
        return {"postgres": "ok", "neo4j": "ok", "embeddings": embeddings}

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error", exc_info=exc)
        return JSONResponse({"detail": "internal server error"}, status_code=500)

    return app


app = create_app()

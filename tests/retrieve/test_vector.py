"""Vector baseline integration tests.

Uses RecordedEmbeddings when tests/fixtures/recorded_embeddings.json exists, otherwise the
deterministic HashEmbeddings (plumbing only; retrieval quality is asserted weakly).
"""

import json
from pathlib import Path

import pytest

from kgrag.graph.build import build_graph
from kgrag.retrieve.embeddings import (
    HashEmbeddings,
    RecordedEmbeddings,
    set_embeddings_for_tests,
)
from kgrag.retrieve.index import build_index, open_stores
from kgrag.retrieve.vector import VectorRetriever
from kgrag.tokens import count_tokens

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
RECORDED = FIXTURES / "recorded_embeddings.json"

pytestmark = pytest.mark.integration


class CountingEmbeddings(HashEmbeddings):
    def __init__(self, inner):
        super().__init__(dim=getattr(inner, "dim", 64))
        self.inner = inner
        self.docs_embedded = 0

    def embed_documents(self, texts):
        self.docs_embedded += len(texts)
        return self.inner.embed_documents(texts)

    def embed_query(self, text):
        return self.inner.embed_query(text)


@pytest.fixture(scope="module")
def embeddings():
    inner = RecordedEmbeddings(RECORDED) if RECORDED.exists() else HashEmbeddings(dim=64)
    emb = CountingEmbeddings(inner)
    set_embeddings_for_tests(emb)
    yield emb
    set_embeddings_for_tests(None)


@pytest.fixture(scope="module")
def dsn(pg_conn, slice_conn):
    i = pg_conn.info
    return f"postgresql://{i.user}:{i.password}@{i.host}:{i.port}/{slice_conn.info.dbname}"


@pytest.fixture(scope="module")
def indexed(slice_conn, neo4j_driver, embeddings, dsn):
    build_graph(slice_conn, neo4j_driver, "exact")
    return build_index(slice_conn, neo4j_driver, connection=dsn)


def test_index_counts(indexed, slice_conn, neo4j_driver):
    n_docs = slice_conn.execute("select count(*) from documents").fetchone()[0]
    with neo4j_driver.session() as s:
        n_nodes = s.run(
            "MATCH (n) WHERE NOT n:GraphMeta AND NOT n:Email RETURN count(n) AS c"
        ).single()["c"]
    assert indexed.chunks == n_docs > 0
    assert indexed.node_names == n_nodes > 0


def test_retrieve_returns_gold_for_single_hop(indexed, slice_conn, dsn, embeddings):
    qs = [q for q in json.loads((FIXTURES / "dev_questions.json").read_text())
          if q["type"] == "single_hop"]
    hits = 0
    for q in qs:
        with open_stores(dsn) as (chunks, _):
            res = VectorRetriever(chunks, slice_conn).retrieve(q["question"], 4000)
        got = {e.doc_id for e in res.evidence}
        hit = bool(got & set(q["gold_doc_ids"]))
        hits += hit
        if RECORDED.exists():
            assert hit, q["id"]
    print(f"single-hop recall (>=1 gold in results): {hits}/{len(qs)}")
    assert hits > 0


def test_retrieve_within_budget(indexed, slice_conn, dsn, embeddings):
    with open_stores(dsn) as (chunks, _):
        res = VectorRetriever(chunks, slice_conn).retrieve("KIP-1043 consumer groups", 300)
    assert sum(count_tokens(e.text) for e in res.evidence) <= 300
    assert res.trace["k"] == 50 and res.trace["considered"] >= len(res.evidence)
    assert all(0.0 <= abs(e.score) <= 1.0 + 1e-6 for e in res.evidence)


def test_index_incremental(indexed, slice_conn, neo4j_driver, dsn, embeddings):
    before = embeddings.docs_embedded
    again = build_index(slice_conn, neo4j_driver, connection=dsn)
    assert embeddings.docs_embedded == before
    assert again == indexed

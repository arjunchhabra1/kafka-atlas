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
from kgrag.store.repo import first_document_text, get_documents
from kgrag.tokens import count_tokens

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
RECORDED = FIXTURES / "recorded_embeddings.json"
TEST_QUERIES = json.loads((FIXTURES / "test_queries.json").read_text())

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
            "MATCH (n) WHERE NOT n:GraphMeta AND NOT n:Email AND n.name IS NOT NULL "
            "RETURN count(n) AS c"
        ).single()["c"]
    assert indexed.chunks.total == n_docs > 0
    assert indexed.node_names.total == n_nodes > 0


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
        res = VectorRetriever(chunks, slice_conn).retrieve(TEST_QUERIES["kip_chunks"], 300)
    assert sum(count_tokens(e.text) for e in res.evidence) <= 300
    assert res.trace["k"] == 50 and res.trace["considered"] >= len(res.evidence)
    assert res.trace["missing"] == 0
    assert all(0.0 <= abs(e.score) <= 1.0 + 1e-6 for e in res.evidence)


def test_index_incremental(indexed, slice_conn, neo4j_driver, dsn, embeddings):
    before = embeddings.docs_embedded
    again = build_index(slice_conn, neo4j_driver, connection=dsn)
    assert embeddings.docs_embedded == before
    assert again.chunks.unchanged == indexed.chunks.total
    assert (again.chunks.added, again.chunks.updated, again.chunks.deleted) == (0, 0, 0)
    assert (again.node_names.added, again.node_names.updated, again.node_names.deleted) == (0, 0, 0)


def test_evidence_text_and_score_order(indexed, slice_conn, dsn, embeddings):
    with open_stores(dsn) as (chunks, _):
        res = VectorRetriever(chunks, slice_conn).retrieve(TEST_QUERIES["kip_chunks"], 4000)
    assert len(res.evidence) > 1
    scores = [e.score for e in res.evidence]
    assert scores == sorted(scores, reverse=True)
    stored = {d.doc_id: d.text for d in get_documents(slice_conn, [e.doc_id for e in res.evidence])}
    assert all(e.text == stored[e.doc_id] for e in res.evidence)


def test_missing_documents_are_reported(indexed, slice_conn, dsn, embeddings, caplog):
    victim = slice_conn.execute("select doc_id from documents order by doc_id limit 1").fetchone()[0]
    probe = first_document_text(slice_conn)  # scripts/record_embeddings.py records this as a query
    row, cols = _take_row(slice_conn, victim)
    assert row[cols.index("text")] == probe
    try:
        with open_stores(dsn) as (chunks, _), caplog.at_level("WARNING"):
            res = VectorRetriever(chunks, slice_conn).retrieve(row[cols.index("text")], 4000)
        assert res.trace["missing"] == 1
        assert "no row in documents" in caplog.text
        assert victim not in {e.doc_id for e in res.evidence}
    finally:
        _put_row(slice_conn, row, cols)


def _take_row(conn, doc_id):
    cur = conn.execute("select * from documents where doc_id = %s", (doc_id,))
    cols = [c.name for c in cur.description]
    row = cur.fetchone()
    conn.execute("delete from documents where doc_id = %s", (doc_id,))
    conn.commit()
    return row, cols


def _put_row(conn, row, cols):
    conn.execute(
        f"insert into documents ({', '.join(cols)}) values ({', '.join(['%s'] * len(cols))})", row
    )
    conn.commit()


def test_changed_text_reembeds_only_that_doc(indexed, slice_conn, neo4j_driver, dsn, embeddings):
    doc_id, text = slice_conn.execute("select doc_id, text from documents order by doc_id limit 1").fetchone()
    before = embeddings.docs_embedded
    try:
        slice_conn.execute("update documents set text = %s where doc_id = %s", (text + " changed", doc_id))
        slice_conn.commit()
        stats = build_index(slice_conn, neo4j_driver, connection=dsn)
        assert (stats.chunks.updated, stats.chunks.added, stats.chunks.deleted) == (1, 0, 0)
        assert stats.node_names.updated == 0
        assert embeddings.docs_embedded - before == 1
    finally:
        slice_conn.execute("update documents set text = %s where doc_id = %s", (text, doc_id))
        slice_conn.commit()
        build_index(slice_conn, neo4j_driver, connection=dsn)


def test_deleted_doc_is_removed(indexed, slice_conn, neo4j_driver, dsn, embeddings):
    victim = slice_conn.execute("select doc_id from documents order by doc_id limit 1").fetchone()[0]
    row, cols = _take_row(slice_conn, victim)
    try:
        stats = build_index(slice_conn, neo4j_driver, connection=dsn)
        assert stats.chunks.deleted == 1 and stats.chunks.total == indexed.chunks.total - 1
        with open_stores(dsn) as (chunks, _):
            hits = chunks.similarity_search(TEST_QUERIES["vector_scan"], k=500)
        assert victim not in {h.metadata["doc_id"] for h in hits}
    finally:
        _put_row(slice_conn, row, cols)
        assert build_index(slice_conn, neo4j_driver, connection=dsn).chunks.added == 1


def test_model_change_reembeds_everything(indexed, slice_conn, neo4j_driver, dsn, embeddings):
    stats = build_index(slice_conn, neo4j_driver, connection=dsn, model="other-model")
    try:
        assert stats.chunks.updated == indexed.chunks.total and stats.chunks.unchanged == 0
        assert stats.node_names.updated == indexed.node_names.total
    finally:
        back = build_index(slice_conn, neo4j_driver, connection=dsn)
        assert back.chunks.updated == indexed.chunks.total

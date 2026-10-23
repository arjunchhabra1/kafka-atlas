"""Freshness preflight: eval refuses stale documents, index or graph before any question runs."""

import json

import pytest

from kgrag.eval.preflight import EvalPreconditionError, check_freshness
from kgrag.eval.run import run_eval
from kgrag.graph.build import build_graph, read_graph_meta
from kgrag.resolve.exact_match import ExactMatchResolver
from kgrag.resolve.interface import resolve_all
from kgrag.retrieve.embeddings import HashEmbeddings, set_embeddings_for_tests
from kgrag.retrieve.index import build_index
from kgrag.store.repo import (
    QuestionRow,
    load_anchor_hints,
    load_mentions,
    load_resolution,
    save_resolution,
)
from tests.conftest import FIXTURES, make_fixture_module

pytestmark = pytest.mark.integration
MODEL = "embed-model-under-test"


@pytest.fixture(scope="module")
def fresh(pg_conn, neo4j_driver):
    """(conn, driver, dsn) for the dev slice: exact resolution, graph built, index in sync."""
    mf = make_fixture_module()
    i = pg_conn.info
    admin = f"postgresql://{i.user}:{i.password}@{i.host}:{i.port}/{i.dbname}"
    recorded = json.loads((FIXTURES / "recorded_extractions.json").read_text())
    set_embeddings_for_tests(HashEmbeddings(dim=16))
    try:
        with mf.fresh_database(admin, "kgrag_preflight") as conn:
            dsn = f"postgresql://{i.user}:{i.password}@{i.host}:{i.port}/{conn.info.dbname}"
            mf.load_slice(conn, FIXTURES / "dev_slice.mbox", recorded)
            save_resolution(conn, "exact", resolve_all(
                ExactMatchResolver(), load_mentions(conn), load_anchor_hints(conn)))
            conn.commit()
            build_graph(conn, neo4j_driver, "exact")
            build_index(conn, neo4j_driver, connection=dsn, model=MODEL)
            yield conn, neo4j_driver, dsn
    finally:
        set_embeddings_for_tests(None)


def _questions(conn):
    gold = [r[0] for r in conn.execute("select doc_id from documents order by doc_id limit 2")]
    conn.commit()
    return [QuestionRow(id="q1", type="inference", question="q", answer="a", gold_doc_ids=gold,
                        status="kept", split="dev")]


def _check(fresh, questions=None, system="graph", **kw):
    conn, driver, _ = fresh
    check_freshness(conn, driver, _questions(conn) if questions is None else questions,
                    system=system, resolver="exact", embed_model=kw.pop("embed_model", MODEL), **kw)


def test_fresh_state_passes(fresh):
    _check(fresh, system="vector")
    _check(fresh, system="graph")


def test_missing_gold_doc(fresh):
    q = QuestionRow(id="q", type="single_hop", question="q", answer="a",
                    gold_doc_ids=["email:nope:0"], status="kept", split="dev")
    with pytest.raises(EvalPreconditionError, match="gold doc.*email:nope:0"):
        _check(fresh, questions=[q], system="vector")


def test_chunks_embedded_with_another_model(fresh):
    with pytest.raises(EvalPreconditionError, match=f"embedded with {MODEL}.*kgrag index"):
        _check(fresh, system="vector", embed_model="voyage-new")


def test_chunks_count_must_equal_documents(fresh):
    conn = fresh[0]
    conn.execute("insert into documents (doc_id, source_kind, source_id, chunk_index, text) "
                 "values ('email:extra:0', 'email', 'extra', 0, 'x')")
    conn.commit()
    try:
        with pytest.raises(EvalPreconditionError, match="vectors for .* documents.*kgrag index"):
            _check(fresh, system="vector")
    finally:
        conn.execute("delete from documents where doc_id = 'email:extra:0'")
        conn.commit()


def test_no_index_at_all(fresh, pg_conn):
    conn = fresh[0]
    conn.execute("alter table langchain_pg_embedding rename to lpe_hidden")
    conn.commit()
    try:
        with pytest.raises(EvalPreconditionError, match="no `chunks` vector index.*kgrag index"):
            _check(fresh, system="vector")
    finally:
        conn.execute("alter table lpe_hidden rename to langchain_pg_embedding")
        conn.commit()


def test_graph_rebuilt_after_index(fresh):
    _, driver, _ = fresh
    with driver.session() as s:
        s.run("CREATE (:Person {id: 'Person:new person', name: 'New Person'})").consume()
    try:
        with pytest.raises(EvalPreconditionError, match="node_names.*1 graph node.*kgrag index"):
            _check(fresh)
        _check(fresh, system="vector")  # the vector system does not use the graph
    finally:
        with driver.session() as s:
            s.run("MATCH (n {id: 'Person:new person'}) DETACH DELETE n").consume()


def test_resolution_changed_since_build(fresh):
    conn = fresh[0]
    original = load_resolution(conn, "exact")
    changed = original.model_copy(deep=True)
    mid = next(iter(changed.assignments))
    other = next(e for e in changed.canonical_names if e != changed.assignments[mid])
    changed.assignments[mid] = other
    save_resolution(conn, "exact", changed)
    conn.commit()
    try:
        with pytest.raises(EvalPreconditionError,
                           match="resolution changed.*kgrag build --resolver exact` then `kgrag index"):
            _check(fresh)
    finally:
        save_resolution(conn, "exact", original)
        conn.commit()
    _check(fresh)


def test_graph_without_resolution_hash(fresh):
    _, driver, _ = fresh
    meta = read_graph_meta(driver)
    old = type(meta)(meta.resolver, meta.built_at, meta.build_hash, meta.node_count,
                     meta.rel_count, None)
    with pytest.raises(EvalPreconditionError, match="no resolution hash.*kgrag build"):
        _check(fresh, meta=old)


def test_run_eval_fails_fast_without_results(fresh, tmp_path):
    conn, driver, _ = fresh
    conn.execute("insert into questions (id, type, question, answer, gold_doc_ids, status, split) "
                 "values ('pf1', 'single_hop', 'q', 'a', '{email:gone:0}', 'kept', 'dev')")
    conn.commit()

    def boom(*a, **k):
        raise AssertionError("no question may run")

    try:
        with pytest.raises(EvalPreconditionError, match="email:gone:0"):
            run_eval(conn, driver, system="graph", split="dev", resolver="exact",
                     results_dir=tmp_path, retriever=boom, answer_llm=boom, judge_llm=boom)
    finally:
        conn.execute("delete from questions where id = 'pf1'")
        conn.commit()
    assert list(tmp_path.iterdir()) == []

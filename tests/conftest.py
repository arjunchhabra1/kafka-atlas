import importlib.util
import json
from pathlib import Path

import pytest

from kgrag.resolve.exact_match import ExactMatchResolver
from kgrag.resolve.interface import resolve_all
from kgrag.store.db import apply_migrations, connect
from kgrag.store.repo import load_anchor_hints, load_mentions, save_resolution

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def make_fixture_module():
    spec = importlib.util.spec_from_file_location("make_fixture", ROOT / "scripts" / "make_fixture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def pg_conn():
    from testcontainers.community.postgres import PostgresContainer

    with PostgresContainer(
        "pgvector/pgvector:pg16", username="kgrag", password="kgrag", dbname="kgrag"
    ) as pg:
        dsn = (
            f"postgresql://kgrag:kgrag@{pg.get_container_host_ip()}:"
            f"{pg.get_exposed_port(5432)}/kgrag"
        )
        conn = connect(dsn)
        apply_migrations(conn)
        yield conn
        conn.close()


@pytest.fixture(scope="session")
def neo4j_driver():
    from neo4j import GraphDatabase
    from testcontainers.community.neo4j import Neo4jContainer

    with Neo4jContainer("neo4j:5.26") as neo:
        driver = GraphDatabase.driver(neo.get_connection_url(), auth=("neo4j", neo.password))
        driver.verify_connectivity()
        yield driver
        driver.close()


@pytest.fixture(scope="session")
def slice_conn(pg_conn):
    mf = make_fixture_module()
    info = pg_conn.info
    admin = f"postgresql://{info.user}:{info.password}@{info.host}:{info.port}/{info.dbname}"
    recorded = json.loads((FIXTURES / "recorded_extractions.json").read_text())
    with mf.fresh_database(admin, "kgrag_graph_fixture") as conn:
        mf.load_slice(conn, FIXTURES / "dev_slice.mbox", recorded)
        save_resolution(conn, "exact", resolve_all(
            ExactMatchResolver(), load_mentions(conn), load_anchor_hints(conn)))
        conn.commit()
        yield conn


@pytest.fixture(scope="session")
def fixture_kg(pg_conn):
    """KnowledgeGraph over the dev slice (exact resolver), in its own database and Neo4j container.

    Uses RecordedEmbeddings when tests/fixtures/recorded_embeddings.json exists and was recorded
    with models.embed, otherwise the deterministic HashEmbeddings (`offline_embeddings`).
    """
    from testcontainers.community.neo4j import Neo4jContainer

    from kgrag.graph.build import build_graph
    from kgrag.graph.kg import KnowledgeGraph
    from kgrag.retrieve.embeddings import offline_embeddings, set_embeddings_for_tests
    from kgrag.retrieve.index import build_index, open_stores

    set_embeddings_for_tests(offline_embeddings(FIXTURES / "recorded_embeddings.json"))
    mf = make_fixture_module()
    i = pg_conn.info
    admin = f"postgresql://{i.user}:{i.password}@{i.host}:{i.port}/{i.dbname}"
    recorded = json.loads((FIXTURES / "recorded_extractions.json").read_text())
    try:
        with mf.fresh_database(admin, "kgrag_retrieve_fixture") as conn, \
                Neo4jContainer("neo4j:5.26") as neo:
            from neo4j import GraphDatabase

            mf.load_slice(conn, FIXTURES / "dev_slice.mbox", recorded)
            save_resolution(conn, "exact", resolve_all(
                ExactMatchResolver(), load_mentions(conn), load_anchor_hints(conn)))
            conn.commit()
            driver = GraphDatabase.driver(neo.get_connection_url(), auth=("neo4j", neo.password))
            dsn = f"postgresql://{i.user}:{i.password}@{i.host}:{i.port}/{conn.info.dbname}"
            try:
                build_graph(conn, driver, "exact")
                build_index(conn, driver, connection=dsn)
                with open_stores(dsn) as (chunks, nodes):
                    yield KnowledgeGraph(driver, conn, chunks, nodes)
            finally:
                driver.close()
    finally:
        set_embeddings_for_tests(None)


def pytest_collection_modifyitems(items):
    for item in items:
        if {"pg_conn", "neo4j_driver", "fixture_kg"} & set(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.integration)

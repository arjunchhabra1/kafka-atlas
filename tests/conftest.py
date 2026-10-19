import pytest

from kgrag.store.db import apply_migrations, connect


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

    with Neo4jContainer("neo4j:5") as neo:
        driver = GraphDatabase.driver(neo.get_connection_url(), auth=("neo4j", neo.password))
        driver.verify_connectivity()
        yield driver
        driver.close()


def pytest_collection_modifyitems(items):
    for item in items:
        if {"pg_conn", "neo4j_driver"} & set(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.integration)

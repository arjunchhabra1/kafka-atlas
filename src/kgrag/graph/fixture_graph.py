"""Build the dev-fixture graph in a throwaway Neo4j container (never touches the real graph)."""

import importlib.util
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from neo4j import Driver, GraphDatabase

from kgrag.config import get_settings
from kgrag.graph.build import build_graph
from kgrag.resolve.exact_match import ExactMatchResolver
from kgrag.resolve.interface import resolve_all
from kgrag.store.repo import load_anchor_hints, load_mentions, save_resolution

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests" / "fixtures"


def _make_fixture_module():
    spec = importlib.util.spec_from_file_location("make_fixture", ROOT / "scripts" / "make_fixture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def fixture_graph(resolver_name: str) -> Iterator[Driver]:
    """Yield a driver on a fresh Neo4j container holding the dev-slice graph for `resolver_name`.

    The slice is ingested into a temporary Postgres database on the configured server.
    """
    from testcontainers.community.neo4j import Neo4jContainer

    from kgrag.resolve.resolver import UserResolver

    resolvers = {"exact": ExactMatchResolver, "user": UserResolver}
    mf = _make_fixture_module()
    recorded = json.loads((FIXTURES / "recorded_extractions.json").read_text())
    with mf.fresh_database(get_settings().pg_dsn, "kgrag_graph_check") as conn:
        mf.load_slice(conn, FIXTURES / "dev_slice.mbox", recorded)
        resolution = resolve_all(resolvers[resolver_name](), load_mentions(conn),
                                 load_anchor_hints(conn))
        save_resolution(conn, resolver_name, resolution)
        conn.commit()
        with Neo4jContainer("neo4j:5.26") as neo:
            driver = GraphDatabase.driver(neo.get_connection_url(), auth=("neo4j", neo.password))
            try:
                build_graph(conn, driver, resolver_name)
                yield driver
            finally:
                driver.close()

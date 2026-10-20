"""A Postgres loaded with the dev slice and its `exact` resolution, shared by graph tests."""

import importlib.util
import json
from pathlib import Path

import pytest

from kgrag.resolve.exact_match import ExactMatchResolver
from kgrag.resolve.interface import resolve_all
from kgrag.store.repo import load_anchor_hints, load_mentions, save_resolution

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"


def make_fixture_module():
    spec = importlib.util.spec_from_file_location("make_fixture", ROOT / "scripts" / "make_fixture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

"""Fixture mentions + hints, built once per session from the hand-labelled dev slice."""

import importlib.util
import json
from pathlib import Path

import pytest

from kgrag.resolve.resolver import UserResolver
from kgrag.store.repo import load_anchor_hints, load_mentions

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"


def _make_fixture_module():
    spec = importlib.util.spec_from_file_location("make_fixture", ROOT / "scripts" / "make_fixture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def fixture_data(pg_conn):
    """(mentions, hints) for the whole dev slice."""
    mf = _make_fixture_module()
    info = pg_conn.info
    admin = f"postgresql://{info.user}:{info.password}@{info.host}:{info.port}/{info.dbname}"
    recorded = json.loads((FIXTURES / "recorded_extractions.json").read_text())
    with mf.fresh_database(admin, "kgrag_resolve_fixture") as conn:
        mf.load_slice(conn, FIXTURES / "dev_slice.mbox", recorded)
        return load_mentions(conn), load_anchor_hints(conn)


@pytest.fixture(scope="session")
def fixture_mentions(fixture_data):
    return fixture_data[0]


@pytest.fixture(scope="session")
def fixture_hints(fixture_data):
    return fixture_data[1]


@pytest.fixture(scope="session")
def gold_clusters():
    return json.loads((FIXTURES / "gold_entities.json").read_text())["clusters"]


@pytest.fixture(scope="session")
def gold(gold_clusters):
    """mention_id -> gold entity name."""
    return {mid: f"{c['type']}:{c['entity']}" for c in gold_clusters for mid in c["mention_ids"]}


@pytest.fixture(scope="session")
def user_resolver():
    return UserResolver()

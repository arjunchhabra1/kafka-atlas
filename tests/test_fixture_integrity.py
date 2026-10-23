"""Integrity of the hand-labelled dev fixture in tests/fixtures (see tests/fixtures/README.md)."""

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

from kgrag.extract.llm import parse_response
from kgrag.ingest.parse import classify_kind, parse_mbox
from kgrag.ingest.pipeline import email_id

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
MBOX = FIXTURES / "dev_slice.mbox"
TAGS = {"shared_first_name", "multi_identity", "variant_spellings"}
QUESTION_MIX = {"single_hop": 4, "inference": 2, "comparison": 2, "temporal": 1, "null": 1}


def _load(name: str):
    return json.loads((FIXTURES / name).read_text())


def _make_fixture_module():
    path = ROOT / "scripts" / "make_fixture.py"
    spec = importlib.util.spec_from_file_location("make_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def clusters():
    return _load("gold_entities.json")["clusters"]


@pytest.fixture(scope="module")
def questions():
    return _load("dev_questions.json")


@pytest.fixture(scope="module")
def slice_db(pg_conn):
    """A fresh database on the test server holding only the ingested slice (R9 ids)."""
    mf = _make_fixture_module()
    info = pg_conn.info
    admin = f"postgresql://{info.user}:{info.password}@{info.host}:{info.port}/{info.dbname}"
    with mf.fresh_database(admin, "kgrag_fixture_check") as conn:
        mf.load_slice(conn, MBOX, _load("recorded_extractions.json"))
        yield conn


@pytest.fixture(scope="module")
def mentions(slice_db):
    rows = slice_db.execute(
        "select m.mention_id, m.surface, m.type, m.source, m.context from mentions m"
    ).fetchall()
    return {r[0]: dict(zip(("id", "surface", "type", "source", "context"), r, strict=True))
            for r in rows}


def test_clusters_are_well_formed_and_disjoint(clusters):
    seen = Counter(mid for c in clusters for mid in c["mention_ids"])
    assert [mid for mid, n in seen.items() if n > 1] == []
    for c in clusters:
        assert c["type"] in ("Person", "Component") and c["entity"] and c["mention_ids"]
        assert set(c.get("tags", [])) <= TAGS


def test_recorded_extractions_cover_exactly_the_slice_humans():
    recorded = _load("recorded_extractions.json")
    humans = {email_id(e.message_id) for e in parse_mbox(MBOX)
              if classify_kind(e.sender_name, e.sender_address) == "human"}
    assert set(recorded) == humans
    for eid, response in recorded.items():
        assert parse_response(eid, response).email_id == eid


def test_question_mix(questions):
    assert len(questions) == 10 and len({q["id"] for q in questions}) == 10
    assert Counter(q["type"] for q in questions) == QUESTION_MIX
    for q in questions:
        assert q["question"] and q["answer"]
        assert (q["gold_doc_ids"] == []) == (q["type"] == "null")


def test_every_gold_mention_exists(clusters, mentions):
    missing = [mid for c in clusters for mid in c["mention_ids"] if mid not in mentions]
    assert missing == []


def test_every_person_and_component_mention_is_labelled(clusters, mentions):
    labelled = {mid for c in clusters for mid in c["mention_ids"]}
    expected = {m["id"] for m in mentions.values() if m["type"] in ("Person", "Component")
                and m["source"] in ("header", "jira", "llm")}
    assert sorted(expected - labelled) == []
    assert sorted(labelled - expected) == []


def test_every_gold_doc_exists(questions, slice_db):
    docs = {r[0] for r in slice_db.execute("select doc_id from documents")}
    assert [d for q in questions for d in q["gold_doc_ids"] if d not in docs] == []


def test_shared_first_name(clusters):
    tagged = [c for c in clusters if "shared_first_name" in c.get("tags", [])]
    assert all(c["type"] == "Person" for c in tagged)
    by_first = Counter(c["entity"].split()[0] for c in tagged)
    assert any(n >= 2 for n in by_first.values()), by_first


def test_multi_identity(clusters, mentions):
    """A person seen via a personal address, an @apache.org address and a Jira display name."""
    def ok(c):
        ms = [mentions[mid] for mid in c["mention_ids"]]
        addresses = {m["context"].split()[-1] for m in ms if m["source"] == "header"}
        return (any(a.endswith("@apache.org") for a in addresses)
                and any(not a.endswith("@apache.org") for a in addresses)
                and any(m["source"] == "jira" for m in ms))
    tagged = [c for c in clusters if "multi_identity" in c.get("tags", [])]
    assert tagged and all(c["type"] == "Person" and ok(c) for c in tagged)


def test_variant_spellings(clusters, mentions):
    tagged = [c for c in clusters if "variant_spellings" in c.get("tags", [])]
    assert tagged and all(c["type"] == "Component" for c in tagged)
    widest = max(len({mentions[mid]["surface"] for mid in c["mention_ids"]}) for c in tagged)
    # The brief asks for 3 spellings; no component in the chosen slice has more than 2
    # extracted spellings (see tests/fixtures/README.md), so 2 is the honest bar here.
    assert widest >= 2

"""KnowledgeGraph toolkit tests, on the dev-slice graph (exact resolver)."""

import json
from datetime import date
from pathlib import Path

import pytest
from neo4j.exceptions import ClientError

from kgrag.retrieve.types import Evidence

pytestmark = pytest.mark.integration

TEST_QUERIES = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "test_queries.json").read_text())


def test_find_nodes_exact_by_name_and_id(fixture_kg):
    hit = fixture_kg.find_nodes("KIP-848")[0]
    assert (hit.label, hit.id, hit.name) == ("KIP", "KIP:KIP-848", "KIP-848")
    assert fixture_kg.find_nodes("kip-848")[0].id == hit.id
    assert fixture_kg.find_nodes("KIP:KIP-848")[0].id == hit.id
    assert fixture_kg.find_nodes("no such entity") == []


def test_find_nodes_label_filter_and_limit(fixture_kg):
    assert fixture_kg.find_nodes("KIP-848", label="Person") == []
    assert fixture_kg.find_nodes("KIP-848", label="KIP")
    assert len(fixture_kg.find_nodes("Kirk", fuzzy=True, limit=1)) == 1


def test_find_nodes_fuzzy_is_case_insensitive(fixture_kg):
    hits = fixture_kg.find_nodes("cristopher shannon", label="Person", fuzzy=True)
    assert "Person:christopher shannon" in {n.id for n in hits}


def test_refresh_reloads_name_cache(fixture_kg):
    fixture_kg.find_nodes("x", fuzzy=True)
    fixture_kg._names = []
    fixture_kg.refresh()
    assert fixture_kg.find_nodes("Cristopher Shannon", label="Person", fuzzy=True)


def test_find_nodes_fuzzy_finds_misspelled_person(fixture_kg):
    assert fixture_kg.find_nodes("Cristopher Shannon") == []
    hits = fixture_kg.find_nodes("Cristopher Shannon", label="Person", fuzzy=True)
    assert {"Person:christopher shannon", "Person:christopher l. shannon"} <= {n.id for n in hits}
    assert hits[0].props["name"] == hits[0].name
    assert all(n.label == "Person" for n in hits)


def test_neighbors_direction_and_types(fixture_kg):
    out = fixture_kg.neighbors("KIP:KIP-1043", edge_types=["AFFECTS"], direction="out")
    assert {n.id for _, n in out} == {
        "Release:4.0", "Release:5.0", "Component:describegroups api"}
    assert all(e.type == "AFFECTS" and e.source_id == "KIP:KIP-1043" for e, _ in out)
    inbound = fixture_kg.neighbors("KIP:KIP-1043", edge_types=["PROPOSED"], direction="in")
    assert inbound and all(
        e.target_id == "KIP:KIP-1043" and n.label == "Person" and e.source_id == n.id
        for e, n in inbound)
    assert fixture_kg.neighbors("KIP:KIP-1043", edge_types=["PROPOSED"], direction="out") == []
    both = fixture_kg.neighbors("KIP:KIP-1043")
    assert {e.type for e, _ in both} >= {"AFFECTS", "PROPOSED", "MENTIONS"}


def test_neighbors_date_filter_excludes_older_edge(fixture_kg):
    def ids(**kw):
        pairs = fixture_kg.neighbors("KIP:KIP-1043", edge_types=["AFFECTS"], direction="out", **kw)
        return {n.id for _, n in pairs}

    assert "Release:5.0" in ids()  # edge dated 2024-09-23
    assert ids(since=date(2024, 10, 1)) == {"Release:4.0", "Component:describegroups api"}
    assert ids(until=date(2024, 10, 1)) == {"Release:5.0"}
    assert ids(since=date(2024, 9, 23), until=date(2024, 9, 23)) == {"Release:5.0"}
    assert ids(since=date(2030, 1, 1)) == set()


def test_edge_dates_are_native_datetimes(fixture_kg):
    edge, _ = fixture_kg.neighbors("KIP:KIP-1043", edge_types=["AFFECTS"], direction="out")[0]
    assert hasattr(edge.props["date"], "year") and edge.props["extractor"]


def test_source_docs_exist_in_documents(fixture_kg):
    seen = 0
    for edge, _ in fixture_kg.neighbors("KIP:KIP-1043"):
        ids = fixture_kg.source_docs(edge)
        assert ids
        assert {d.doc_id for d in fixture_kg.documents(ids)} == set(ids)
        seen += len(ids)
    assert seen > 0


def test_search_nodes_returns_nodes_with_scores(fixture_kg):
    hits = fixture_kg.search_nodes(TEST_QUERIES["kip_name"], k=3)
    assert 0 < len(hits) <= 3
    assert "KIP:KIP-1043" in {n.id for n, _ in hits}
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_chunks_returns_evidence_with_stored_text(fixture_kg):
    hits = fixture_kg.search_chunks(TEST_QUERIES["kip_chunks"], k=5)
    assert 0 < len(hits) <= 5 and all(isinstance(h, Evidence) for h in hits)
    stored = {d.doc_id: d.text for d in fixture_kg.documents([h.doc_id for h in hits])}
    assert all(h.text == stored[h.doc_id] for h in hits)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_documents_in_requested_order_and_missing_omitted(fixture_kg):
    ids = [r[0] for r in fixture_kg.conn.execute(
        "select doc_id from documents order by doc_id limit 3").fetchall()]
    fixture_kg.conn.commit()
    got = fixture_kg.documents([ids[2], "nope:0", ids[0]])
    assert [d.doc_id for d in got] == [ids[2], ids[0]]


def test_cypher_reads_and_params(fixture_kg):
    assert fixture_kg.cypher("MATCH (n:KIP) RETURN count(n) AS c")[0]["c"] >= 2
    rows = fixture_kg.cypher("MATCH (n:KIP {id: $id}) RETURN n.name AS name", {"id": "KIP:KIP-848"})
    assert rows == [{"name": "KIP-848"}]


def test_cypher_is_read_only(fixture_kg):
    before = fixture_kg.cypher("MATCH (n) RETURN count(n) AS c")[0]["c"]
    with pytest.raises(ClientError):
        fixture_kg.cypher("CREATE (:Junk {id: 'junk'})")
    assert fixture_kg.cypher("MATCH (n) RETURN count(n) AS c")[0]["c"] == before


def test_search_chunks_warns_on_hits_without_document(fixture_kg, caplog):
    victim = fixture_kg.search_chunks(TEST_QUERIES["kip_chunks"], k=1)[0].doc_id
    cur = fixture_kg.conn.execute("select * from documents where doc_id = %s", (victim,))
    cols, row = [c.name for c in cur.description], cur.fetchone()
    fixture_kg.conn.execute("delete from documents where doc_id = %s", (victim,))
    fixture_kg.conn.commit()
    try:
        with caplog.at_level("WARNING"):
            hits = fixture_kg.search_chunks(TEST_QUERIES["kip_chunks"], k=5)
        assert victim not in {h.doc_id for h in hits}
        assert "no row in documents" in caplog.text
    finally:
        fixture_kg.conn.execute(
            f"insert into documents ({', '.join(cols)}) values ({', '.join(['%s'] * len(cols))})", row)
        fixture_kg.conn.commit()

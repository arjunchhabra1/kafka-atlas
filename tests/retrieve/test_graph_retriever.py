"""Graph retriever on the dev-slice graph: entity linking and the trace (contract is elsewhere)."""

import pytest

from kgrag.retrieve import graph_retriever
from kgrag.retrieve.graph_linking import link_entities

pytestmark = pytest.mark.integration


def test_link_entities_ids_aliases_and_fuzzy_names(fixture_kg):
    linking = link_entities("Did Cristopher Shannon vote on KIP 1032 for release 4.0 or KIP-99999?",
                            fixture_kg)
    how = {s.id: s.how for s in linking.seeds}
    assert how["KIP:KIP-1032"] == "id"
    assert how["Release:4.0"] == how["Release:4.0.0"] == "id"   # x.y links its x.y.0 twin
    assert how["Person:christopher shannon"] == "fuzzy_name"
    assert {"surface": "KIP-99999", "reason": "id not in graph"} in linking.dropped


def test_link_entities_prefers_full_name_and_ignores_unparseable_ids(fixture_kg):
    linking = link_entities("Which KIP did Andrew Schofield write?", fixture_kg)
    strong = [s for s in linking.seeds if s.strong]
    assert [(s.id, s.how) for s in strong] == [("Person:andrew schofield", "exact_name")]
    assert not any(":?" in s.id for s in linking.seeds)


def test_trace_explains_a_multi_seed_question(graph_retrieve):
    res = graph_retrieve("Which KIP received more binding votes, KIP-1032 or KIP-1043?", 4000)
    t = res.trace
    assert t["strategy"] == "graph"
    assert {s["id"] for s in t["linked"]} >= {"KIP:KIP-1032", "KIP:KIP-1043"}
    assert "VOTED" in t["cues"]
    graph_rows = [r for r in t["selected"] if r["source"] == "graph"]
    assert graph_rows and all(r["paths"] and r["seeds"] for r in graph_rows)
    assert [r["doc_id"] for r in t["selected"]] == [e.doc_id for e in res.evidence]
    # diversification: both KIPs are supported within the first few docs
    first = {s for r in graph_rows[:4] for s in r["seeds"]}
    assert {"KIP:KIP-1032", "KIP:KIP-1043"} <= first


def test_no_entity_question_falls_back_to_vector(graph_retrieve, fixture_kg):
    res = graph_retrieve("What's a good pizza recipe?", 4000)
    assert res.trace["strategy"] == "vector_fallback"
    assert res.trace["fallback_reason"] == "no entity linked"
    assert [e.doc_id for e in res.evidence] == [
        e.doc_id for e in graph_retriever.pack_to_budget(fixture_kg.search_chunks(
            "What's a good pizza recipe?", k=graph_retriever.VECTOR_K), 4000)]


def test_neighbors_many_matches_kg_neighbors(fixture_kg):
    ids = ["KIP:KIP-1043", "Person:chia-ping tsai", "Release:4.0.0"]

    def rows(pairs):
        return [(e.type, e.source_id, e.target_id, e.props.get("source_doc_ids"),
                 e.props.get("confidence"), n.id, n.label, n.name) for e, n in pairs]

    for types in (None, graph_retriever.FACT_TYPES):
        many = graph_retriever.neighbors_many(fixture_kg, ids, types)
        for i in ids:
            assert rows(many[i]) == rows(fixture_kg.neighbors(i, edge_types=types)), (i, types)

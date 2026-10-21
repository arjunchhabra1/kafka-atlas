"""Graph retriever on the dev-slice graph: entity linking, the walk query and the trace
(the contract is in test_graph_contract.py)."""

import pytest

from kgrag.retrieve import graph_retriever
from kgrag.retrieve.graph_config import DEFAULT_CONFIG, GRAPH_ONLY_CONFIG
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


def test_single_word_name_is_a_weak_seed(fixture_kg):
    linking = link_entities("What did Kirk say?", fixture_kg)
    assert [(s.id, s.how, s.weight) for s in linking.seeds] == [
        ("Person:kirk", "single_name", DEFAULT_CONFIG.weight_single_name)]
    assert not linking.seeds[0].strong


def test_trace_explains_a_multi_seed_question(graph_retrieve):
    res = graph_retrieve("Which KIP received more binding votes, KIP-1032 or KIP-1043?", 4000)
    t = res.trace
    assert t["strategy"] == "graph" and t["config"]["semantic_rerank"]
    assert {s["id"] for s in t["linked"]} >= {"KIP:KIP-1032", "KIP:KIP-1043"}
    assert "VOTED" in t["cues"]
    graph_rows = [r for r in t["selected"] if r["source"] == "graph"]
    assert graph_rows and all(r["paths"] and r["seeds"] for r in graph_rows)
    assert [r["doc_id"] for r in t["selected"]] == [e.doc_id for e in res.evidence]
    # diversification: both KIPs are supported within the first few docs
    first = {s for r in graph_rows[:4] for s in r["seeds"]}
    assert {"KIP:KIP-1032", "KIP:KIP-1043"} <= first


def test_question_is_embedded_once(fixture_kg, monkeypatch):
    calls = []
    real = fixture_kg.embed_query
    monkeypatch.setattr(fixture_kg, "embed_query", lambda text: calls.append(text) or real(text))
    q = "Who proposed KIP-1043 and which release does it affect?"
    res = graph_retriever.retrieve(q, fixture_kg, 4000)
    assert res.trace["strategy"] == "graph" and calls == [q] and res.trace["embeddings"] == 1
    calls.clear()
    graph_retriever.retrieve("What's a good pizza recipe?", fixture_kg, 4000)
    assert len(calls) == 1  # node-search fallback and vector fallback share one embedding


def _forbid_embedding(kg, monkeypatch):
    monkeypatch.setattr(kg, "embed_query", lambda text: pytest.fail("embedded"))
    monkeypatch.setattr(kg.chunk_store.embeddings, "embed_query",
                        lambda text: pytest.fail("embedded via the store"))


def test_graph_only_config_uses_no_embeddings(fixture_kg, monkeypatch):
    _forbid_embedding(fixture_kg, monkeypatch)
    assert not GRAPH_ONLY_CONFIG.uses_embeddings and DEFAULT_CONFIG.uses_embeddings
    res = graph_retriever.retrieve("Which KIP received more binding votes, KIP-1032 or KIP-1043?",
                                   fixture_kg, 4000, GRAPH_ONLY_CONFIG)
    assert res.evidence and res.trace["embeddings"] == 0 and res.trace["backfilled"] == 0
    assert all(r["semantic"] == 0.0 for r in res.trace["selected"])
    assert res.trace["unlinked"] is False and res.trace["fallback"] is False


def test_graph_only_unlinkable_question_is_empty_and_never_embeds(fixture_kg, monkeypatch):
    _forbid_embedding(fixture_kg, monkeypatch)
    # lowercase, no ids: only the node-name embedding search could link it
    res = graph_retriever.retrieve("what's a good pizza recipe?", fixture_kg, 4000,
                                   GRAPH_ONLY_CONFIG)
    assert res.evidence == []
    assert res.trace["unlinked"] is True and res.trace["fallback"] is False
    assert res.trace["strategy"] == "empty" and res.trace["embeddings"] == 0


def test_no_entity_question_falls_back_to_vector(graph_retrieve, fixture_kg):
    res = graph_retrieve("What's a good pizza recipe?", 4000)
    assert res.trace["strategy"] == "vector_fallback"
    assert res.trace["unlinked"] is True and res.trace["fallback"] is True
    assert res.trace["fallback_reason"] == "no entity linked"
    assert [e.doc_id for e in res.evidence] == [
        e.doc_id for e in graph_retriever.pack_to_budget(fixture_kg.search_chunks(
            "What's a good pizza recipe?", k=graph_retriever.VECTOR_K), 4000)]


def _rows(pairs):
    return [(e.type, e.source_id, e.target_id, e.props.get("source_doc_ids"),
             e.props.get("confidence"), n.id, n.label, n.name) for e, n in pairs]


def test_neighbors_many_matches_kg_neighbors(fixture_kg):
    ids = ["KIP:KIP-1043", "Person:chia-ping tsai", "Release:4.0.0"]
    for types in (None, graph_retriever.fact_types()):
        many, totals = graph_retriever.neighbors_many(fixture_kg, ids, types)
        for i in ids:
            expected = fixture_kg.neighbors(i, edge_types=types)
            assert _rows(many[i]) == _rows(expected), (i, types)
            assert totals[i] == len(expected)


def test_neighbors_many_caps_each_node_to_its_strongest_edges_in_cypher(fixture_kg):
    node = "KIP:KIP-1043"
    full = fixture_kg.neighbors(node)
    capped, totals = graph_retriever.neighbors_many(fixture_kg, [node], None, limit=3)
    assert totals[node] == len(full) > 3 and len(capped[node]) == 3

    def strength(e):
        return DEFAULT_CONFIG.weight(e.type) * (e.props.get("confidence") or 1.0)

    kept = sorted(strength(e) for e, _ in capped[node])
    dropped = [strength(e) for e, _ in full if (e.type, e.source_id, e.target_id) not in
               {(k.type, k.source_id, k.target_id) for k, _ in capped[node]}]
    assert min(kept) >= max(dropped)
    # a cue raises its edge type's priority: for some cap, more AFFECTS edges survive it

    def affects(limit, cues):
        pairs, _ = graph_retriever.neighbors_many(fixture_kg, [node], None, limit, cues)
        return sum(e.type == "AFFECTS" for e, _ in pairs[node])

    assert any(affects(k, frozenset({"AFFECTS"})) > affects(k, frozenset()) for k in range(1, len(full)))

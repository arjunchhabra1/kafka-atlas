import pytest

from kgrag.graph.check import (
    connectivity,
    doc_reached,
    find_seeds,
    reachability,
    reachable,
)

pytestmark = pytest.mark.integration

# Chain: Alice -SENT-> email:e1 -MENTIONS-> KIP-7 -DEPENDS_ON-> KIP-8 -AFFECTS-> Streams
#        Bob (isolated, degree 0 apart from nothing)
GRAPH = """
CREATE (a:Person {id:'Person:alice', name:'Alice Smith'}),
       (e:Email {id:'email:e1', name:'s'}),
       (k7:KIP {id:'KIP:KIP-7', name:'KIP-7'}),
       (k8:KIP {id:'KIP:KIP-8', name:'KIP-8'}),
       (c:Component {id:'Component:streams', name:'Streams'}),
       (b:Person {id:'Person:bob', name:'Bob'}),
       (a)-[:SENT {source_doc_ids:['email:e1:0']}]->(e),
       (e)-[:MENTIONS {source_doc_ids:['email:e1:0']}]->(k7),
       (k7)-[:DEPENDS_ON {source_doc_ids:['email:e2:0']}]->(k8),
       (k8)-[:AFFECTS {source_doc_ids:['email:e3:1', 'jira:9']}]->(c)
"""


@pytest.fixture
def graph(neo4j_driver):
    with neo4j_driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n").consume()
        s.run(GRAPH).consume()
    return neo4j_driver


def test_seed_matching_name_and_id_case_insensitive(graph):
    assert find_seeds(graph, "what did alice smith say about kip-7?") == ["KIP:KIP-7", "Person:alice"]


def test_seed_matching_respects_token_boundaries(graph):
    assert find_seeds(graph, "KIP-78 and Bobby") == []


def test_bfs_respects_hop_limit(graph):
    nodes1, _ = reachable(graph, ["KIP:KIP-7"], 1)
    assert nodes1 == {"KIP:KIP-7", "email:e1", "KIP:KIP-8"}
    nodes2, _ = reachable(graph, ["KIP:KIP-7"], 2)
    assert "Component:streams" in nodes2 and "Person:alice" in nodes2
    assert "Person:bob" not in nodes2


def test_doc_reached_via_provenance_and_email_node(graph):
    nodes, docs = reachable(graph, ["KIP:KIP-8"], 1)
    assert doc_reached("jira:9", nodes, docs)          # cited by a traversed relationship
    assert doc_reached("email:e2:0", nodes, docs)      # cited by DEPENDS_ON
    assert not doc_reached("email:e1:0", nodes, docs)  # two hops away
    nodes, docs = reachable(graph, ["KIP:KIP-8"], 2)
    assert doc_reached("email:e1:0", nodes, docs)      # the Email node itself is reached


def test_reachability_report_counts_multi_hop_only(graph):
    qs = [
        {"id": "a", "type": "inference", "question": "Does KIP-7 matter?",
         "gold_doc_ids": ["email:e1:0", "jira:9"]},
        {"id": "b", "type": "single_hop", "question": "KIP-7?", "gold_doc_ids": ["email:e1:0"]},
        {"id": "c", "type": "temporal", "question": "nothing here", "gold_doc_ids": ["jira:9"]},
    ]
    r = reachability(graph, qs, hops=1)
    assert [q.id for q in r.questions] == ["a", "c"]
    assert r.questions[0].reached == ["email:e1:0"]  # jira:9 is two hops out
    assert r.questions[1].seeds == []
    assert r.share == pytest.approx(1 / 3)


def test_connectivity_stats(graph):
    c = connectivity(graph)
    assert c.nodes_by_label["Person"] == 2 and c.rels_by_type["SENT"] == 1
    assert c.largest_component_share == pytest.approx(5 / 6)
    assert c.degree["KIP"] == (2.0, 2.0)
    assert c.person_degree_one == 1  # alice; bob has degree 0

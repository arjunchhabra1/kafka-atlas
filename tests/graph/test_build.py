import pytest

from kgrag.graph.build import build_graph, read_graph_meta

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def built(slice_conn, neo4j_driver):
    stats = build_graph(slice_conn, neo4j_driver, "exact")
    return stats


def _scalar(driver, query, **kw):
    with driver.session() as s:
        return s.run(query, **kw).single()[0]


def test_node_counts_by_label(built, slice_conn, neo4j_driver):
    humans = slice_conn.execute("select count(*) from emails where kind = 'human'").fetchone()[0]
    assert humans > 0
    assert built.nodes["Email"] == humans
    assert _scalar(neo4j_driver, "MATCH (n:Email) RETURN count(n)") == humans
    for label in ("Person", "KIP", "JiraTicket", "Release", "Component"):
        assert _scalar(neo4j_driver, f"MATCH (n:{label}) RETURN count(n)") == built.nodes[label] > 0


def test_sent_and_replied_to(built, slice_conn, neo4j_driver):
    humans = slice_conn.execute("select count(*) from emails where kind = 'human'").fetchone()[0]
    # Every human email has exactly one sender edge.
    assert _scalar(neo4j_driver, "MATCH (:Person)-[r:SENT]->(e:Email) RETURN count(DISTINCT e)") == humans
    assert _scalar(neo4j_driver, "MATCH (e:Email) WHERE NOT ()-[:SENT]->(e) RETURN count(e)") == 0
    expected = slice_conn.execute(
        "select count(*) from emails c join emails p on p.message_id = c.parent_message_id "
        "where c.kind = 'human' and p.kind = 'human'").fetchone()[0]
    assert expected > 0
    assert _scalar(neo4j_driver, "MATCH (:Email)-[r:REPLIED_TO]->(:Email) RETURN count(r)") == expected


def test_jira_edges(built, slice_conn, neo4j_driver):
    ev = slice_conn.execute(
        "select id, ticket, fix_version from jira_events where action = 'resolved' "
        "and fix_version is not null order by id").fetchone()
    assert ev, "fixture needs a resolved event with a fix version"
    event_id, ticket, fix = ev
    assert _scalar(neo4j_driver,
                   "MATCH (:Person)-[r:RESOLVED]->(t:JiraTicket {id: $id}) "
                   "WHERE $doc IN r.source_doc_ids RETURN count(r)",
                   id=f"JiraTicket:{ticket}", doc=f"jira:{event_id}") == 1
    assert _scalar(neo4j_driver,
                   "MATCH (:JiraTicket {id: $id})-[r:FIX_VERSION]->(:Release) RETURN count(r)",
                   id=f"JiraTicket:{ticket}") >= 1


def test_edge_provenance_present(built, slice_conn, neo4j_driver):
    known = {r[0] for r in slice_conn.execute("select doc_id from documents")}
    with neo4j_driver.session() as s:
        rows = s.run("MATCH ()-[r]->() RETURN type(r) AS t, r.source_doc_ids AS d, "
                     "r.extractor AS x, r.confidence AS c, r.date AS date").data()
    assert rows
    for r in rows:
        assert r["d"], r
        assert set(r["d"]) <= known, r
        assert r["x"] in {"header", "regex", "jira", "llm"}
        assert 0.0 <= r["c"] <= 1.0 and r["date"] is not None


def test_parallel_facts_merged(built, neo4j_driver):
    assert _scalar(neo4j_driver,
                   "MATCH (a)-[r]->(b) WITH a, b, type(r) AS t, count(r) AS c "
                   "WHERE c > 1 RETURN count(*)") == 0


def test_rebuild_idempotent(slice_conn, neo4j_driver):
    a = build_graph(slice_conn, neo4j_driver, "exact")
    b = build_graph(slice_conn, neo4j_driver, "exact")
    assert (a.nodes, a.rels) == (b.nodes, b.rels)
    assert _scalar(neo4j_driver, "MATCH (n) WHERE NOT n:GraphMeta RETURN count(n)") == sum(b.nodes.values())
    assert _scalar(neo4j_driver, "MATCH ()-[r]->() RETURN count(r)") == sum(b.rels.values())


def test_graph_meta_written(slice_conn, neo4j_driver):
    stats = build_graph(slice_conn, neo4j_driver, "exact")
    meta = read_graph_meta(neo4j_driver)
    assert meta.resolver == "exact"
    assert meta.node_count == sum(stats.nodes.values())
    assert meta.rel_count == sum(stats.rels.values())
    assert len(meta.config_hash) == 12 and meta.built_at.tzinfo is not None
    assert _scalar(neo4j_driver, "MATCH (m:GraphMeta) RETURN count(m)") == 1


def test_missing_resolution_raises(slice_conn, neo4j_driver):
    with pytest.raises(ValueError, match="no stored resolution"):
        build_graph(slice_conn, neo4j_driver, "nope")

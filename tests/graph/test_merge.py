"""Pure-Python tests of fact merging, endpoint rules, chunk selection and the wipe guard."""

from datetime import UTC, datetime

import pytest
import typer

from kgrag.graph.build import _chunks_containing, _Edge, _Graph

D1, D2, D3 = (datetime(2024, m, 1, tzinfo=UTC) for m in (1, 2, 3))


def edge(t="PROPOSED", src="Person:a", dst="KIP:KIP-1", sl="Person", dl="KIP", docs=("d1",),
         date=D2, ex="llm", conf=0.5, vote=None, binding=None):
    return _Edge(t, src, dst, sl, dl, list(docs), date, ex, conf, vote, binding,
                 date if t == "VOTED" else None)


def test_merge_accumulates_docs_dedupes_earliest_date_max_confidence():
    g = _Graph()
    g.edge(edge(docs=("d1", "d2"), date=D2, conf=0.4))
    g.edge(edge(docs=("d2", "d3"), date=D1, conf=0.9))
    g.edge(edge(docs=("d4",), date=D3, conf=0.1))
    (e,) = g.edges.values()
    assert e.docs == ["d1", "d2", "d3", "d4"]
    assert e.date == D1 and e.confidence == 0.9


def test_merge_extractor_prefers_deterministic_source():
    g = _Graph()
    g.edge(edge("MENTIONS", "email:e", "KIP:KIP-1", "Email", "KIP", ex="llm"))
    g.edge(edge("MENTIONS", "email:e", "KIP:KIP-1", "Email", "KIP", ex="regex", conf=1.0))
    (e,) = g.edges.values()
    assert e.extractor == "regex" and e.confidence == 1.0


def test_voted_keeps_value_and_binding_of_latest_vote():
    g = _Graph()
    g.edge(edge("VOTED", date=D2, vote=1, binding=True))
    g.edge(edge("VOTED", date=D3, vote=-1, binding=False))
    g.edge(edge("VOTED", date=D1, vote=0, binding=None))  # older, arrives last
    (e,) = g.edges.values()
    assert (e.vote_value, e.binding, e.date) == (-1, False, D1)


def test_caller_doc_lists_are_never_aliased_or_mutated():
    shared = ["jira:1"]
    g = _Graph()
    g.edge(_Edge("FILED", "Person:bob", "JiraTicket:T1", "Person", "JiraTicket", shared, D1, "jira", 1.0))
    g.edge(_Edge("FIX_VERSION", "JiraTicket:T1", "Release:1", "JiraTicket", "Release", shared, D1, "jira", 1.0))
    g.edge(_Edge("RESOLVED", "Person:alice", "JiraTicket:T1", "Person", "JiraTicket", ["jira:2"], D2, "jira", 1.0))
    g.edge(_Edge("FIX_VERSION", "JiraTicket:T1", "Release:1", "JiraTicket", "Release", ["jira:2"], D2, "jira", 1.0))
    by = {e.type: e.docs for e in g.edges.values()}
    assert by["FILED"] == ["jira:1"] and by["RESOLVED"] == ["jira:2"]
    assert by["FIX_VERSION"] == ["jira:1", "jira:2"]
    assert shared == ["jira:1"]


@pytest.mark.parametrize("etype,src,dst", [
    ("PROPOSED", "Component", "KIP"), ("SENT", "Component", "Email"),
    ("FILED", "KIP", "JiraTicket"), ("DEPENDS_ON", "KIP", "Release"),
])
def test_endpoint_rule_violations_are_skipped_and_counted(etype, src, dst):
    g = _Graph()
    g.edge(edge(etype, "x", "y", src, dst))
    assert not g.edges and g.skipped == {f"type_violation:{etype}": 1}


@pytest.mark.parametrize("etype,dst", [("VOTED", "Release"), ("AFFECTS", "Release")])
def test_extended_rules_allow_release_targets(etype, dst):
    g = _Graph()
    src = "KIP" if etype == "AFFECTS" else "Person"
    g.edge(edge(etype, "x", "y", src, dst))
    assert len(g.edges) == 1


CHUNKS = [("c0", "intro text"), ("c1", "we discuss KIP-1043 and Bob"), ("c2", "KIP-10 only"),
          ("c3", "bob and kip-1043 together")]


def test_chunk_selection_prefers_chunks_with_all_surfaces():
    assert _chunks_containing(CHUNKS, "KIP-1043", "Bob") == ["c1", "c3"]
    assert _chunks_containing(CHUNKS, "KIP-1043", "Alice") == ["c1", "c3"]  # either, no chunk has both


def test_chunk_selection_token_bounded_case_insensitive_with_chunk0_fallback():
    assert _chunks_containing(CHUNKS, "kip-10") == ["c2"]  # not inside KIP-1043
    assert _chunks_containing(CHUNKS, "nothing here") == ["c0"]


class FakeDriver:
    def __init__(self, nodes, meta):
        self.nodes, self.meta = nodes, meta

    def session(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, q, **kw):
        outer = self

        class R:
            def single(self):
                if "GraphMeta" in q:
                    return {"m": outer.meta} if outer.meta else None
                return {"c": outer.nodes}
        return R()


def test_wipe_guard(monkeypatch):
    from kgrag.cli import _guard_wipe

    unknown = FakeDriver(5, None)
    with pytest.raises(typer.Exit) as ex:
        _guard_wipe(unknown, "bolt://localhost:7687", force=False, yes=False)
    assert ex.value.exit_code == 2
    _guard_wipe(unknown, "bolt://localhost:7687", force=True, yes=False)
    _guard_wipe(FakeDriver(0, None), "bolt://127.0.0.1:7687", force=False, yes=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
    with pytest.raises(typer.Exit):
        _guard_wipe(FakeDriver(0, None), "neo4j+s://x.databases.neo4j.io", force=False, yes=False)
    _guard_wipe(FakeDriver(0, None), "neo4j+s://x.databases.neo4j.io", force=False, yes=True)

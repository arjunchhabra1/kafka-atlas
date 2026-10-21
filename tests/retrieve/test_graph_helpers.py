"""Unit tests for the graph retriever's pure helpers (no Docker, tiny hand-built inputs)."""

import pytest

from kgrag.retrieve.graph_ablation import evaluate, format_table, recall_by_type
from kgrag.retrieve.graph_config import DEFAULT_CONFIG, GRAPH_ONLY_CONFIG, GraphRetrieverConfig
from kgrag.retrieve.graph_linking import (
    NameSpan,
    Seed,
    candidate_ngrams,
    id_seeds,
    name_tier,
    rank_same_name,
    release_aliases,
    select_spans,
)
from kgrag.retrieve.graph_retriever import (
    Ranked,
    Support,
    add_support,
    anchor,
    diversify,
    doc_share,
    edge_score,
    minmax,
    off_focus,
    rank_docs,
    relation_cues,
)


def _texts(spans):
    return [s.text for s in spans]


# --- linking ------------------------------------------------------------------------------

def test_candidate_ngrams_longest_first_and_trims_stopwords():
    q = "After Jane Doe simplified KIP-123, which tool did it add?"
    texts = _texts(candidate_ngrams(q))
    assert texts[0] == "Jane Doe"  # "After" is a stopword, the 2-gram comes first
    assert set(texts) == {"Jane Doe", "Jane", "Doe"}
    assert "KIP-123" not in texts and "After" not in texts and "Which" not in texts


@pytest.mark.parametrize("verb", ["Name", "List", "Compare", "Which", "Who", "Summarize", "Describe"])
def test_sentence_initial_imperatives_are_not_names(verb):
    assert _texts(candidate_ngrams(f"{verb} the reviewers")) == []


def test_candidate_ngrams_handles_initials_possessives_and_hyphens():
    texts = _texts(candidate_ngrams("Did Mary J. Major reply to Jon Smith's and Li-Na Wu's notes?"))
    assert "Mary J. Major" in texts
    assert "Jon Smith" in texts and "Smith's" not in " ".join(texts)
    assert "Li-Na Wu" in texts


def test_candidate_ngrams_camel_case_and_numbered_names():
    texts = _texts(candidate_ngrams("does the connect DescribeGroups API change MirrorMaker 2"))
    assert "DescribeGroups API" in texts and "MirrorMaker 2" in texts
    assert "connect" not in texts  # lowercase words are left to the node-name search


def test_candidate_ngrams_skips_short_single_tokens():
    assert _texts(candidate_ngrams("Is JS or Al relevant?")) == []


def test_select_spans_prefers_longest_and_never_overlaps():
    spans = candidate_ngrams("Did Jane Doe and Jane agree?")
    chosen = select_spans(spans, {"jane doe", "jane", "doe"})
    assert [(s.text, s.start) for s in chosen] == [("Jane Doe", 4), ("Jane", 17)]


def test_single_word_names_are_a_weak_tier():
    assert name_tier(NameSpan("Jane Doe", 0, 8, 2)) == "exact_name"
    assert name_tier(NameSpan("Jane", 0, 4, 1)) == "single_name"
    assert not Seed("Person:j", "Person", "Jane", "single_name", "Jane", 0.5).strong
    assert Seed("Person:jd", "Person", "Jane Doe", "exact_name", "Jane Doe", 1.0).strong


def test_rank_same_name_by_degree_then_id():
    rows = [{"id": "P:b", "degree": 3}, {"id": "P:a", "degree": 3}, {"id": "P:c", "degree": 40}]
    assert [r["id"] for r in rank_same_name(rows)] == ["P:c", "P:a", "P:b"]


def test_release_aliases():
    assert release_aliases("4.0") == ["4.0", "4.0.0"]
    assert release_aliases("4.0.0") == ["4.0.0", "4.0"]
    assert release_aliases("3.9.1") == ["3.9.1"]


def test_id_seeds_from_question():
    assert id_seeds("Did KIP 848 ship in release 4.0 or fix KAFKA-16891?") == [
        ("KIP", "KIP-848", "KIP 848"), ("Release", "4.0", "4.0"),
        ("JiraTicket", "KAFKA-16891", "KAFKA-16891")]


# --- cues ---------------------------------------------------------------------------------

@pytest.mark.parametrize("question, cue", [
    ("Who voted on it?", "VOTED"),
    ("How many binding votes did it get?", "VOTED"),
    ("Who proposed it?", "PROPOSED"),
    ("When was the discussion thread started?", "PROPOSED"),
    ("Who resolved the ticket?", "RESOLVED"),
    ("Who created the Jira ticket KAFKA-1?", "FILED"),
    ("Who opened an issue for it?", "FILED"),
    ("Who filed KAFKA-1?", "FILED"),
    ("What concerns did she raise?", "REVIEWED"),
    ("Which tickets blocked the release?", "BLOCKS"),
    ("Which KIPs does it depend on?", "DEPENDS_ON"),
    ("Which release did it ship in?", "FIX_VERSION"),
    ("Which components does it affect?", "AFFECTS"),
    ("Does it supersede KIP-1?", "SUPERSEDES"),
])
def test_relation_cues_positives(question, cue):
    assert cue in relation_cues(question)


@pytest.mark.parametrize("question, not_cue", [
    ("Which open-source project uses it?", "FILED"),
    ("He agreed to open the vote", "FILED"),
    ("Which versions should the KIP target?", "AFFECTS"),
    ("Who leads the effort?", "PROPOSED"),
    ("Is this a requirement?", "DEPENDS_ON"),
])
def test_relation_cues_negatives(question, not_cue):
    assert not_cue not in relation_cues(question)


def test_relation_cues_none():
    assert relation_cues("What is a pizza?") == frozenset()


# --- scoring ------------------------------------------------------------------------------

def test_edge_score_orders_semantic_over_structural_and_decays_by_hop():
    cues = frozenset()
    assert edge_score("VOTED", 1.0, 1, 1.0, cues) > edge_score("MENTIONS", 1.0, 1, 1.0, cues)
    assert edge_score("VOTED", 1.0, 2, 1.0, cues) < edge_score("VOTED", 1.0, 1, 1.0, cues)
    assert edge_score("VOTED", 0.5, 1, 1.0, cues) < edge_score("VOTED", 1.0, 1, 1.0, cues)
    assert edge_score("VOTED", 1.0, 1, 1.0, frozenset({"VOTED"})) > edge_score("VOTED", 1.0, 1, 1.0, cues)
    assert edge_score("VOTED", 1.0, 1, 1.0, cues, bridge=True) > edge_score("VOTED", 1.0, 1, 1.0, cues)
    assert edge_score("VOTED", None, 1, 0.5, cues) == pytest.approx(edge_score("VOTED", 1.0, 1, 0.5, cues))
    assert edge_score("NEW_TYPE", 1.0, 1, 1.0, cues) == DEFAULT_CONFIG.unknown_edge_weight


def test_edge_score_switches():
    off = DEFAULT_CONFIG.with_(cues=False, bridge_boost=False)
    cued = frozenset({"VOTED"})
    assert edge_score("VOTED", 1.0, 1, 1.0, cued, bridge=True, config=off) == \
        edge_score("VOTED", 1.0, 1, 1.0, frozenset(), config=off)


def test_doc_share_favours_specific_edges():
    assert doc_share(1) == 1.0 and doc_share(0) == 1.0
    assert doc_share(16) == pytest.approx(0.5)
    assert doc_share(16, DEFAULT_CONFIG.with_(doc_share=False)) == 1.0


def test_graph_only_config_has_no_embedding_terms():
    assert not GRAPH_ONLY_CONFIG.semantic_rerank and not GRAPH_ONLY_CONFIG.vector_backfill
    assert not GRAPH_ONLY_CONFIG.vector_fallback and not GRAPH_ONLY_CONFIG.node_search
    assert not GRAPH_ONLY_CONFIG.uses_embeddings
    assert GRAPH_ONLY_CONFIG.cues and GraphRetrieverConfig() == DEFAULT_CONFIG


def test_graph_config_registry_and_fingerprint():
    from kgrag.retrieve.graph_config import UnknownGraphConfigError, get_graph_config

    assert get_graph_config("hybrid") is DEFAULT_CONFIG
    assert get_graph_config("graph-only") is GRAPH_ONLY_CONFIG
    with pytest.raises(UnknownGraphConfigError, match="unknown graph config 'nope'.*hybrid"):
        get_graph_config("nope")
    fp = DEFAULT_CONFIG.fingerprint()
    assert len(fp) == 12 and fp == GraphRetrieverConfig().fingerprint()
    assert fp != GRAPH_ONLY_CONFIG.fingerprint()
    assert fp != DEFAULT_CONFIG.with_(decay=0.4).fingerprint()


def _seed(node_id, name, how="id"):
    return Seed(node_id, node_id.split(":")[0], name, how, name, 1.0)


def test_anchor_adds_support_for_seeds_named_in_text_without_mutating_input():
    supports = {"jira:4": {"KIP:KIP-2": Support(0.2, "far")}, "e:1": {"KIP:KIP-2": Support(0.9, "x")}}
    before = {d: dict(s) for d, s in supports.items()}
    texts = {"jira:4": "resolved KAFKA-7 (KIP-1) as Fixed", "e:1": "about KIP-10 and KIP-2",
             "not-a-candidate": "KIP-1"}
    seeds = [_seed("KIP:KIP-1", "KIP-1"), _seed("KIP:KIP-2", "KIP-2"),
             _seed("Person:jane doe", "Jane Doe", how="single_name")]
    out, added = anchor(supports, texts, seeds)
    assert added == 1 and supports == before and "not-a-candidate" not in out
    assert out["jira:4"]["KIP:KIP-1"].path == "text names KIP-1"
    assert out["jira:4"]["KIP:KIP-2"].score == 0.2   # not named in jira:4
    assert "KIP:KIP-1" not in out["e:1"]             # KIP-10 is not KIP-1
    assert out["e:1"]["KIP:KIP-2"].path == "x"       # already stronger than an anchor


def test_anchor_skips_single_word_names_and_weak_seeds():
    supports = {"d": {"KIP:KIP-1": Support(0.5, "p")}}
    jane = Seed("Person:jane", "Person", "Jane", "exact_name", "Jane", 1.0)
    assert anchor(supports, {"d": "Jane wrote this"}, [jane])[1] == 0


def test_off_focus_only_for_other_nodes_of_a_named_type():
    kip, jane = _seed("KIP:KIP-9", "KIP-9"), _seed("Person:jane", "Jane Doe", "exact_name")
    seeds = [kip, jane]
    assert off_focus("KIP", "KIP:KIP-5", jane, seeds)          # Jane's vote on another KIP
    assert not off_focus("KIP", "KIP:KIP-9", jane, seeds)      # the named KIP itself
    assert not off_focus("KIP", "KIP:KIP-5", kip, seeds)       # KIP -DEPENDS_ON-> KIP: same type
    assert not off_focus("Release", "Release:3.9", jane, seeds)  # no Release is named
    comp = Seed("Component:x", "Component", "x", "node_search", "q", 0.3)
    assert not off_focus("Component", "Component:y", kip, [kip, comp])  # weak seeds set no focus


def test_add_support_keeps_best_path_per_seed_and_breaks_ties_on_path():
    supports: dict[str, dict[str, Support]] = {}
    add_support(supports, "d1", "A", 0.3, "A -MENTIONS- e")
    add_support(supports, "d1", "A", 0.9, "A -VOTED- p")
    add_support(supports, "d1", "A", 0.5, "A -SENT- e")
    add_support(supports, "d1", "B", 0.2, "z")
    add_support(supports, "d1", "B", 0.2, "a")
    add_support(supports, "d1", "B", 0.2, "m")
    assert supports["d1"]["A"] == Support(0.9, "A -VOTED- p")
    assert supports["d1"]["B"] == Support(0.2, "a")


def test_minmax():
    assert minmax({"a": 0.2, "b": 0.6, "c": 0.4}) == pytest.approx({"a": 0.0, "b": 1.0, "c": 0.5})
    assert minmax({"a": 0.3, "b": 0.3}) == {"a": 1.0, "b": 1.0}
    assert minmax({}) == {}


def test_rank_docs_rewards_bridging_docs_and_breaks_ties_by_doc_id():
    supports = {
        "d-one-seed": {"A": Support(1.0, "p")},
        "d-bridge": {"A": Support(0.6, "p"), "B": Support(0.6, "q")},
        "c-tie": {"A": Support(1.0, "p")},
    }
    ranked = rank_docs(supports, {})
    assert [r.doc_id for r in ranked] == ["d-bridge", "c-tie", "d-one-seed"]
    assert ranked[0].seeds == ("A", "B") and ranked[0].via == ("p", "q")


def test_rank_docs_semantic_term_reorders():
    supports = {"d1": {"A": Support(1.0, "p")}, "d2": {"A": Support(0.9, "p")}}
    ranked = rank_docs(supports, {"d1": 0.0, "d2": 1.0}, DEFAULT_CONFIG.with_(sem_weight=0.5))
    assert [r.doc_id for r in ranked] == ["d2", "d1"]
    assert ranked[0].score == pytest.approx(1.4)


def _r(doc, score, *seeds):
    return Ranked(doc_id=doc, score=score, graph=score, semantic=0.0, seeds=tuple(seeds))


def test_diversify_round_robins_each_seed_before_filling():
    ranked = [_r("a1", 5, "A"), _r("a2", 4, "A"), _r("a3", 3, "A"), _r("b1", 1, "B"), _r("b2", 0.5, "B")]
    out = diversify(ranked, ["A", "B"], rounds=2)
    assert [r.doc_id for r in out] == ["a1", "b1", "a2", "b2", "a3"]


def test_diversify_counts_a_bridging_doc_for_every_seed_it_supports():
    ranked = [_r("ab", 5, "A", "B"), _r("a1", 4, "A"), _r("b1", 1, "B")]
    out = diversify(ranked, ["A", "B"], rounds=1)
    assert [r.doc_id for r in out] == ["ab", "a1", "b1"]


def test_diversify_single_seed_or_zero_rounds_is_plain_ranking():
    ranked = [_r("x", 3, "A"), _r("y", 2, "A"), _r("z", 1, "C")]
    assert diversify(ranked, ["A"], rounds=3) == ranked
    two = [_r("a1", 5, "A"), _r("a2", 4, "A"), _r("b1", 1, "B")]
    assert diversify(two, ["A", "B"], rounds=0) == two


# --- ablation helper ----------------------------------------------------------------------

def test_recall_by_type_groups_and_skips_null_questions():
    from kgrag.retrieve.types import Evidence, RetrievalResult

    questions = [
        {"question": "q1", "type": "single_hop", "gold_doc_ids": ["a"]},
        {"question": "q2", "type": "comparison", "gold_doc_ids": ["b", "c"]},
        {"question": "q3", "type": "temporal", "gold_doc_ids": ["d"]},
        {"question": "q4", "type": "null", "gold_doc_ids": []},
    ]
    found = {"q1": ["a"], "q2": ["b", "x"], "q3": [], "q4": ["a"]}

    def fake(q, budget):
        return RetrievalResult(evidence=[Evidence(doc_id=d, text="t", score=1.0) for d in found[q]])

    r = recall_by_type(questions, fake, 100)
    assert r["single-hop"] == 1.0 and r["comparison"] == 0.5 and r["temporal"] == 0.0
    assert r["multi-hop"] == pytest.approx(0.25) and r["all"] == pytest.approx(0.5)
    assert "null" not in r
    table = format_table({"sys": evaluate(questions, fake, 100)})
    assert "single-hop" in table and "0.25" in table


def test_evaluate_reports_unlinked_and_fallback_rates():
    from kgrag.retrieve.types import Evidence, RetrievalResult

    questions = [{"question": f"q{i}", "type": "single_hop", "gold_doc_ids": ["a"]}
                 for i in range(4)]
    traces = {"q0": {"unlinked": True, "fallback": True}, "q1": {"unlinked": True, "fallback": False},
              "q2": {"unlinked": False, "fallback": False}, "q3": {"unlinked": False}}

    def graph(q, budget):
        ev = [] if traces[q]["unlinked"] and not traces[q].get("fallback") else \
            [Evidence(doc_id="a", text="t", score=1.0)]
        return RetrievalResult(evidence=ev, trace=traces[q])

    def vector(q, budget):
        return RetrievalResult(evidence=[Evidence(doc_id="a", text="t", score=1.0)])

    g, v = evaluate(questions, graph, 100), evaluate(questions, vector, 100)
    assert g["unlinked_rate"] == 0.5 and g["fallback_rate"] == 0.25
    assert g["recall"]["single_hop"] == 0.75 and g["n"]["single_hop"] == 4
    assert v["unlinked_rate"] is None and v["fallback_rate"] is None
    table = format_table({"graph-only": g, "vector": v})
    unlinked = next(line for line in table.splitlines() if line.startswith("unlinked"))
    assert "50%" in unlinked and unlinked.rstrip().endswith("-")
    assert "vector fallback" in table and "25%" in table

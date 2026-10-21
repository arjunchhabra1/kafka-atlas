"""Unit tests for the graph retriever's pure helpers (no Docker, tiny hand-built inputs)."""

import pytest

from kgrag.retrieve.graph_linking import (
    Seed,
    candidate_ngrams,
    id_seeds,
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
    q = "After Andrew Schofield simplified KIP-1043, which tool did it add?"
    texts = _texts(candidate_ngrams(q))
    assert texts[0] == "Andrew Schofield"  # "After" is trimmed, the 2-gram comes first
    assert set(texts) == {"Andrew Schofield", "Andrew", "Schofield"}
    assert "KIP-1043" not in texts and "After" not in texts and "Which" not in texts


def test_candidate_ngrams_handles_initials_possessives_and_hyphens():
    texts = _texts(candidate_ngrams("Did Christopher L. Shannon reply to David Jacot's and Chia-Ping Tsai's notes?"))
    assert "Christopher L. Shannon" in texts
    assert "David Jacot" in texts and "Jacot's" not in " ".join(texts)
    assert "Chia-Ping Tsai" in texts


def test_candidate_ngrams_camel_case_and_numbered_names():
    texts = _texts(candidate_ngrams("does the connect DescribeGroups API change MirrorMaker 2"))
    assert "DescribeGroups API" in texts and "MirrorMaker 2" in texts
    assert "connect" not in texts  # lowercase words are left to the node-name search


def test_candidate_ngrams_skips_short_single_tokens():
    assert _texts(candidate_ngrams("Is JS or Al relevant?")) == []


def test_select_spans_prefers_longest_and_never_overlaps():
    spans = candidate_ngrams("Did Andrew Schofield and Andrew agree?")
    chosen = select_spans(spans, {"andrew schofield", "andrew", "schofield"})
    assert [(s.text, s.start) for s in chosen] == [("Andrew Schofield", 4), ("Andrew", 25)]


def test_release_aliases():
    assert release_aliases("4.0") == ["4.0", "4.0.0"]
    assert release_aliases("4.0.0") == ["4.0.0", "4.0"]
    assert release_aliases("3.9.1") == ["3.9.1"]


def test_id_seeds_from_question():
    assert id_seeds("Did KIP 848 ship in release 4.0 or fix KAFKA-16891?") == [
        ("KIP", "KIP-848", "KIP 848"), ("Release", "4.0", "4.0"),
        ("JiraTicket", "KAFKA-16891", "KAFKA-16891")]


# --- scoring ------------------------------------------------------------------------------

def test_relation_cues():
    assert relation_cues("Who voted on it and who proposed it?") >= {"VOTED", "PROPOSED"}
    assert "RESOLVED" in relation_cues("Who resolved the ticket?")
    assert relation_cues("What is a pizza?") == frozenset()


def test_edge_score_orders_semantic_over_structural_and_decays_by_hop():
    cues = frozenset()
    assert edge_score("VOTED", 1.0, 1, 1.0, cues) > edge_score("MENTIONS", 1.0, 1, 1.0, cues)
    assert edge_score("VOTED", 1.0, 2, 1.0, cues) < edge_score("VOTED", 1.0, 1, 1.0, cues)
    assert edge_score("VOTED", 0.5, 1, 1.0, cues) < edge_score("VOTED", 1.0, 1, 1.0, cues)
    assert edge_score("VOTED", 1.0, 1, 1.0, frozenset({"VOTED"})) > edge_score("VOTED", 1.0, 1, 1.0, cues)
    assert edge_score("VOTED", 1.0, 1, 1.0, cues, bridge=True) > edge_score("VOTED", 1.0, 1, 1.0, cues)
    assert edge_score("VOTED", None, 1, 0.5, cues) == pytest.approx(edge_score("VOTED", 1.0, 1, 0.5, cues))


def test_doc_share_favours_specific_edges():
    assert doc_share(1) == 1.0 and doc_share(0) == 1.0
    assert doc_share(16) == pytest.approx(0.5)


def _seed(node_id, name, how="id"):
    return Seed(node_id, node_id.split(":")[0], name, how, name, 1.0)


def test_anchor_adds_support_for_seeds_named_in_text():
    supports = {"jira:4": {"KIP:KIP-1043": Support(0.2, "far")}, "e:1": {"KIP:KIP-1043": Support(0.9, "x")}}
    texts = {"jira:4": "resolved KAFKA-16437 (KIP-1032) as Fixed", "e:1": "about KIP-10320 and KIP-1043"}
    seeds = [_seed("KIP:KIP-1032", "KIP-1032"), _seed("KIP:KIP-1043", "KIP-1043"),
             _seed("Person:andrew", "Andrew", how="exact_name")]
    assert anchor(supports, texts, seeds) == 1
    assert supports["jira:4"]["KIP:KIP-1032"].path == "text names KIP-1032"
    assert supports["jira:4"]["KIP:KIP-1043"].score == 0.2   # not named in jira:4
    assert "KIP:KIP-1032" not in supports["e:1"]             # KIP-10320 is not KIP-1032
    assert supports["e:1"]["KIP:KIP-1043"].path == "x"       # already stronger than an anchor


def test_anchor_skips_single_word_names():
    supports = {"d": {"KIP:KIP-1": Support(0.5, "p")}}
    assert anchor(supports, {"d": "Andrew wrote this"}, [_seed("Person:andrew", "Andrew", "exact_name")]) == 0


def test_off_focus_only_for_other_nodes_of_a_named_type():
    kip, jun = _seed("KIP:KIP-966", "KIP-966"), _seed("Person:jun", "Jun Rao", "exact_name")
    seeds = [kip, jun]
    assert off_focus("KIP", "KIP:KIP-1109", jun, seeds)          # Jun Rao's vote on another KIP
    assert not off_focus("KIP", "KIP:KIP-966", jun, seeds)       # the named KIP itself
    assert not off_focus("KIP", "KIP:KIP-853", kip, seeds)       # KIP -DEPENDS_ON-> KIP: same type
    assert not off_focus("Release", "Release:3.9", jun, seeds)   # no Release is named
    comp = Seed("Component:x", "Component", "x", "node_search", "q", 0.3)
    assert not off_focus("Component", "Component:y", kip, [kip, comp])  # weak seeds set no focus


def test_add_support_keeps_best_path_per_seed():
    supports: dict[str, dict[str, Support]] = {}
    add_support(supports, "d1", "A", 0.3, "A -MENTIONS- e")
    add_support(supports, "d1", "A", 0.9, "A -VOTED- p")
    add_support(supports, "d1", "A", 0.5, "A -SENT- e")
    add_support(supports, "d1", "B", 0.2, "B -MENTIONS- e")
    assert supports["d1"]["A"] == Support(0.9, "A -VOTED- p")
    assert set(supports["d1"]) == {"A", "B"}


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
    ranked = rank_docs(supports, {}, sem_weight=1.0)
    assert [r.doc_id for r in ranked] == ["d-bridge", "c-tie", "d-one-seed"]
    assert ranked[0].seeds == ("A", "B")


def test_rank_docs_semantic_term_reorders():
    supports = {"d1": {"A": Support(1.0, "p")}, "d2": {"A": Support(0.9, "p")}}
    ranked = rank_docs(supports, {"d1": 0.0, "d2": 1.0}, sem_weight=0.5)
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


def test_diversify_single_seed_is_plain_ranking():
    ranked = [_r("x", 3, "A"), _r("y", 2, "A"), _r("z", 1, "C")]
    assert diversify(ranked, ["A"], rounds=3) == ranked

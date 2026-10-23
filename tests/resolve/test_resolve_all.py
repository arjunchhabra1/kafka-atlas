import pytest

from kgrag.eval.bcubed import bcubed
from kgrag.extract.schema import AnchorHints, Mention
from kgrag.resolve.exact_match import ExactMatchResolver
from kgrag.resolve.interface import resolve_all

HINTS = AnchorHints(sender_pairs=[], jira_actors=[])


def _m(mid, surface, mtype):
    return Mention(mention_id=mid, surface=surface, type=mtype, email_id="e", context="")


class Spy:
    name = "spy"

    def __init__(self):
        self.seen = []

    def resolve(self, mentions, hints):
        self.seen = mentions
        return ExactMatchResolver().resolve(mentions, hints)


def test_resolve_all_normalizes_ids_and_filters_resolver_input():
    ms = [
        _m("e:m1", "KIP-848", "KIP"), _m("e:m2", "kip 848", "KIP"),
        _m("e:m3", "KAFKA-1234", "JiraTicket"), _m("e:m4", "Apache Kafka 3.4.0", "Release"),
        _m("e:m5", "3.4.0", "Release"), _m("e:m6", "gibberish", "KIP"),
        _m("e:m7", "Ann", "Person"), _m("e:m8", "ann ", "Person"),
    ]
    spy = Spy()
    res = resolve_all(spy, ms, HINTS)
    assert {m.type for m in spy.seen} == {"Person"}
    a = res.assignments
    assert a["e:m1"] == a["e:m2"] == "KIP:KIP-848"
    assert a["e:m3"] == "JiraTicket:KAFKA-1234"
    assert a["e:m4"] == a["e:m5"] == "Release:3.4.0"
    assert a["e:m6"] == "KIP:?e:m6" and res.canonical_names["KIP:?e:m6"] == "gibberish"
    assert a["e:m7"] == a["e:m8"] == "Person:ann"
    assert res.entity_types["Person:ann"] == "Person"


def test_resolve_all_prefixes_resolver_ids():
    class Bare:
        name = "bare"

        def resolve(self, mentions, hints):
            from kgrag.resolve.interface import Resolution
            return Resolution(assignments={m.mention_id: "x1" for m in mentions},
                              canonical_names={"x1": "X"}, entity_types={"x1": "Component"})

    res = resolve_all(Bare(), [_m("e:m1", "broker", "Component")], HINTS)
    assert res.assignments["e:m1"] == "Component:x1" and res.canonical_names["Component:x1"] == "X"


@pytest.mark.integration
def test_exact_match_bcubed_on_fixture(fixture_mentions, fixture_hints, gold):
    pred = resolve_all(ExactMatchResolver(), fixture_mentions, fixture_hints).assignments
    p, r, f = bcubed(pred, gold)
    print(f"\nexact-match B3 precision={p:.4f} recall={r:.4f} F1={f:.4f}")
    assert 0.0 < f < 1.0

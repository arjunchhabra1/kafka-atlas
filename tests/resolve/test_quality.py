"""Entity-resolution quality bars for UserResolver, driven by the gold tags.

These xfail (NotImplementedError) until UserResolver is implemented, then run for real.
"""

import pytest

from kgrag.eval.bcubed import bcubed
from kgrag.resolve.interface import resolve_all

pytestmark = pytest.mark.quality


@pytest.fixture(scope="module")
def predicted(user_resolver, fixture_mentions, fixture_hints, gold):
    return resolve_all(user_resolver, fixture_mentions, fixture_hints).assignments


def _tagged(gold_clusters, tag):
    return [c for c in gold_clusters if tag in c.get("tags", [])]


def test_bcubed_at_least_0_85(predicted, gold):
    p, r, f = bcubed(predicted, gold)
    print(f"\nB3 precision={p:.3f} recall={r:.3f} F1={f:.3f}")
    assert f >= 0.85


def test_shared_first_name_not_merged(predicted, gold_clusters):
    tagged = _tagged(gold_clusters, "shared_first_name")
    assert len(tagged) >= 2
    entity_of = {c["entity"]: {predicted[m] for m in c["mention_ids"]} for c in tagged}
    print(f"\nshared_first_name entities: {entity_of}")
    names = list(entity_of)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not entity_of[a] & entity_of[b], f"{a} and {b} share a predicted entity"


def test_multi_identity_merged(predicted, gold_clusters):
    tagged = _tagged(gold_clusters, "multi_identity")
    assert tagged
    for c in tagged:
        ents = {predicted[m] for m in c["mention_ids"]}
        print(f"\n{c['entity']}: {len(ents)} predicted entities")
        assert len(ents) == 1, f"{c['entity']} split across {sorted(ents)}"


def test_variant_spellings_merged(predicted, gold_clusters):
    tagged = _tagged(gold_clusters, "variant_spellings")
    assert tagged
    for c in tagged:
        ents = {predicted[m] for m in c["mention_ids"]}
        print(f"\n{c['entity']}: {len(ents)} predicted entities")
        assert len(ents) == 1, f"{c['entity']} split across {sorted(ents)}"

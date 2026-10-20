"""Contract every resolver must satisfy, on the full fixture mention set."""

import pytest

from kgrag.resolve.exact_match import ExactMatchResolver
from kgrag.resolve.interface import resolve_all


@pytest.fixture(params=["exact", "user"])
def resolver(request, user_resolver_lazy):
    return ExactMatchResolver() if request.param == "exact" else user_resolver_lazy()


@pytest.fixture
def resolution(resolver, fixture_mentions, fixture_hints):
    return resolve_all(resolver, fixture_mentions, fixture_hints)


def test_every_mention_assigned(resolution, fixture_mentions):
    assert {m.mention_id for m in fixture_mentions} == set(resolution.assignments)


def test_deterministic(resolver, fixture_mentions, fixture_hints):
    a = resolve_all(resolver, fixture_mentions, fixture_hints)
    b = resolve_all(resolver, list(reversed(fixture_mentions)), fixture_hints)
    assert a.assignments == b.assignments
    assert a.canonical_names == b.canonical_names


def test_no_cross_type_merge(resolution, fixture_mentions):
    types: dict[str, set[str]] = {}
    for m in fixture_mentions:
        types.setdefault(resolution.assignments[m.mention_id], set()).add(m.type)
    assert {e: t for e, t in types.items() if len(t) > 1} == {}
    for e, t in types.items():
        assert resolution.entity_types[e] == next(iter(t))


def test_canonical_name_for_every_entity(resolution):
    entities = set(resolution.assignments.values())
    assert entities <= set(resolution.canonical_names)
    assert entities <= set(resolution.entity_types)
    assert all(resolution.canonical_names[e].strip() for e in entities)


def test_id_mentions_normalized(resolution, fixture_mentions):
    for m in fixture_mentions:
        if m.type == "KIP":
            assert resolution.assignments[m.mention_id].startswith("KIP:")

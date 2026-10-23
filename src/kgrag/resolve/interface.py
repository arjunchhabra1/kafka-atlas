"""The entity-resolution contract: `Resolver`, `Resolution`, and `resolve_all`."""

from collections import defaultdict
from typing import Protocol

from pydantic import BaseModel

from kgrag.extract.regex_ids import extract_ids
from kgrag.extract.schema import AnchorHints, Mention

ID_TYPES = ("KIP", "JiraTicket", "Release")
RESOLVER_TYPES = ("Person", "Component")


class Resolution(BaseModel):
    assignments: dict[str, str]  # mention_id -> entity_id
    canonical_names: dict[str, str]  # entity_id -> canonical name
    entity_types: dict[str, str]  # entity_id -> mention type


class Resolver(Protocol):
    name: str

    def resolve(self, mentions: list[Mention], hints: AnchorHints) -> Resolution: ...


def normalize_id(m: Mention) -> str | None:
    """Canonical id (e.g. `KIP-892`, `KAFKA-1234`, `3.4.0`) for an ID-type mention, if any."""
    candidates = [m.surface]
    if m.type == "Release":
        candidates.append(f"version {m.surface}")  # a bare "3.4.0" is only a release in context
    for text in candidates:
        for found in extract_ids(text):
            if found.kind == m.type:
                return found.id
    return None


def _id_entities(mentions: list[Mention]) -> Resolution:
    out = Resolution(assignments={}, canonical_names={}, entity_types={})
    for m in mentions:
        canonical = normalize_id(m)
        if canonical is None:  # unparseable: never drop it, never merge it
            entity_id, name = f"{m.type}:?{m.mention_id}", m.surface
        else:
            entity_id, name = f"{m.type}:{canonical}", canonical
        out.assignments[m.mention_id] = entity_id
        out.canonical_names[entity_id] = name
        out.entity_types[entity_id] = m.type
    return out


def resolve_all(resolver: Resolver, mentions: list[Mention], hints: AnchorHints) -> Resolution:
    """Resolve every mention: ID types deterministically, Person/Component via `resolver`.

    The resolver sees only Person and Component mentions, sorted by mention_id.
    Its entity ids are prefixed with the entity type (`Person:...`) if it didn't.
    """
    ordered = sorted(mentions, key=lambda m: m.mention_id)
    merged = _id_entities([m for m in ordered if m.type in ID_TYPES])
    candidates = [m for m in ordered if m.type in RESOLVER_TYPES]
    res = resolver.resolve(candidates, hints)

    members: dict[str, list[Mention]] = defaultdict(list)
    by_id = {m.mention_id: m for m in candidates}
    for mid, entity in res.assignments.items():
        if mid in by_id:
            members[entity].append(by_id[mid])
    for entity, ms in members.items():
        etype = res.entity_types.get(entity) or ms[0].type
        final = entity if entity.startswith(f"{etype}:") else f"{etype}:{entity}"
        for m in ms:
            merged.assignments[m.mention_id] = final
        merged.entity_types[final] = etype
        if entity in res.canonical_names:
            merged.canonical_names[final] = res.canonical_names[entity]
    return merged

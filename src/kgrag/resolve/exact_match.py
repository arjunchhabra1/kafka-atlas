"""Baseline resolver: mentions with the same type and case-folded surface are one entity."""

from collections import Counter, defaultdict

from kgrag.extract.schema import AnchorHints, Mention
from kgrag.resolve.interface import Resolution


class ExactMatchResolver:
    name = "exact"

    def resolve(self, mentions: list[Mention], hints: AnchorHints) -> Resolution:
        groups: dict[tuple[str, str], list[Mention]] = defaultdict(list)
        for m in mentions:
            groups[(m.type, m.surface.strip().casefold())].append(m)
        out = Resolution(assignments={}, canonical_names={}, entity_types={})
        for (mtype, key), ms in groups.items():
            entity_id = f"{mtype}:{key}"
            counts = Counter(m.surface.strip() for m in ms)
            # Most common spelling; ties broken alphabetically so the result is stable.
            out.canonical_names[entity_id] = min(counts, key=lambda s: (-counts[s], s))
            out.entity_types[entity_id] = mtype
            for m in ms:
                out.assignments[m.mention_id] = entity_id
        return out

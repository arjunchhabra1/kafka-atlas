from kgrag.retrieve.types import Evidence
from kgrag.tokens import count_tokens


def pack_to_budget(candidates: list[Evidence], budget_tokens: int) -> list[Evidence]:
    """Greedily keep candidates in order while they fit; skip oversized ones, dedupe by doc_id."""
    remaining = budget_tokens
    seen: set[str] = set()
    packed: list[Evidence] = []
    for cand in candidates:
        if cand.doc_id in seen:
            continue
        cost = count_tokens(cand.text)
        if cost > remaining:
            continue
        seen.add(cand.doc_id)
        packed.append(cand)
        remaining -= cost
        if remaining <= 0:
            break
    return packed

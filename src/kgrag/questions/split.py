import random
from collections import defaultdict
from typing import Literal

import psycopg

from kgrag.store.repo import QuestionRow

KEPT = ("kept", "edited")


class AlreadySplitError(RuntimeError):
    """Kept questions already carry a dev/test split."""


def assign_splits(
    questions: list[QuestionRow], dev_frac: float = 0.2, seed: int = 13
) -> dict[str, Literal["dev", "test"]]:
    """Stratified dev/test assignment of kept/edited questions, deterministic in (ids, seed)."""
    by_type: dict[str, list[str]] = defaultdict(list)
    for q in questions:
        if q.status in KEPT:
            by_type[q.type].append(q.id)
    out: dict[str, Literal["dev", "test"]] = {}
    for qtype, ids in by_type.items():
        ids.sort()
        random.Random(f"{seed}:{qtype}").shuffle(ids)
        n_dev = round(len(ids) * dev_frac)
        for i, qid in enumerate(ids):
            out[qid] = "dev" if i < n_dev else "test"
    return out


def persist_splits(
    conn: psycopg.Connection, *, force: bool = False, reshuffle: bool = False,
    dev_frac: float = 0.2, seed: int = 13,
) -> dict[str, dict[str, int]]:
    """Write splits for kept/edited questions; return per-type dev/test counts of all of them.

    Default: refuses when any kept/edited question already has a split. `force`: keeps every
    existing split and assigns only the kept/edited questions without one (stratified among
    themselves), so no question ever moves between dev and test. `reshuffle`: reassigns all of
    them; a question that was tuned on in dev may land in test (leakage), so use it only before
    any dev work. Non-kept rows get their split cleared so a dropped question can never linger.
    """
    rows = [QuestionRow(**dict(zip(QuestionRow.model_fields, r, strict=True))) for r in conn.execute(
        f"select {', '.join(QuestionRow.model_fields)} from questions").fetchall()]
    if not (force or reshuffle) and any(q.split and q.status in KEPT for q in rows):
        raise AlreadySplitError("kept questions already have a split")
    kept_split = {} if reshuffle else {q.id: q.split for q in rows if q.status in KEPT and q.split}
    splits = assign_splits([q for q in rows if q.id not in kept_split], dev_frac, seed)
    with conn.transaction():
        conn.execute("update questions set split = null where status not in ('kept', 'edited')")
        for qid, split in splits.items():
            conn.execute("update questions set split = %s where id = %s", (split, qid))
    conn.commit()
    final = {**kept_split, **splits}
    counts: dict[str, dict[str, int]] = {}
    for q in rows:
        if q.id in final:
            c = counts.setdefault(q.type, {"dev": 0, "test": 0, "new": 0})
            c[final[q.id]] += 1
            c["new"] += q.id in splits
    return counts

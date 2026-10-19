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
    conn: psycopg.Connection, *, force: bool = False, dev_frac: float = 0.2, seed: int = 13
) -> dict[str, dict[str, int]]:
    """Write splits for kept/edited questions; return per-type dev/test counts.

    Refuses when any kept/edited question already has a split, unless force. Non-kept rows
    get their split cleared so a dropped question can never linger in dev or test.
    """
    rows = [QuestionRow(**dict(zip(QuestionRow.model_fields, r, strict=True))) for r in conn.execute(
        f"select {', '.join(QuestionRow.model_fields)} from questions").fetchall()]
    if not force and any(q.split and q.status in KEPT for q in rows):
        raise AlreadySplitError("kept questions already have a split")
    splits = assign_splits(rows, dev_frac, seed)
    with conn.transaction():
        conn.execute("update questions set split = null where status not in ('kept', 'edited')")
        for qid, split in splits.items():
            conn.execute("update questions set split = %s where id = %s", (split, qid))
    conn.commit()
    counts: dict[str, dict[str, int]] = {}
    for q in rows:
        if q.id in splits:
            c = counts.setdefault(q.type, {"dev": 0, "test": 0})
            c[splits[q.id]] += 1
    return counts

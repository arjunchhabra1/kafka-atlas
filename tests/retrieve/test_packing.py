from kgrag.retrieve.packing import pack_to_budget
from kgrag.retrieve.types import Evidence
from kgrag.tokens import count_tokens


def ev(doc_id: str, n_tokens: int) -> Evidence:
    text = " ".join(["word"] * n_tokens)
    assert count_tokens(text) == n_tokens
    return Evidence(doc_id=doc_id, text=text, score=1.0)


def total(items: list[Evidence]) -> int:
    return sum(count_tokens(e.text) for e in items)


def test_respects_budget():
    out = pack_to_budget([ev("a", 60), ev("b", 60), ev("c", 30)], 100)
    assert total(out) <= 100
    assert [e.doc_id for e in out] == ["a", "c"]


def test_dedupes():
    out = pack_to_budget([ev("a", 10), ev("a", 10), ev("b", 10)], 100)
    assert [e.doc_id for e in out] == ["a", "b"]


def test_skips_oversized_then_continues():
    out = pack_to_budget([ev("big", 150), ev("x", 40), ev("y", 40)], 100)
    assert [e.doc_id for e in out] == ["x", "y"]


def test_empty_and_zero_budget():
    assert pack_to_budget([], 100) == []
    assert pack_to_budget([ev("a", 10)], 0) == []

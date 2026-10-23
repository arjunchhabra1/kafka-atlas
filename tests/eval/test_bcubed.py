import pytest

from kgrag.eval.bcubed import bcubed, bcubed_f1


def test_bcubed_perfect():
    gold = {"a": "x", "b": "x", "c": "y"}
    assert bcubed_f1({"a": "1", "b": "1", "c": "2"}, gold) == 1.0


def test_bcubed_all_singletons():
    gold = dict.fromkeys("abcd", "x")
    pred = {m: m for m in "abcd"}
    p, r, f = bcubed(pred, gold)
    assert (p, r) == (1.0, 0.25)
    assert f == pytest.approx(0.4)
    assert bcubed_f1(pred, gold) == pytest.approx(0.4)


def test_bcubed_all_merged():
    gold = {"a": "x", "b": "x", "c": "y", "d": "y"}
    p, r, f = bcubed(dict.fromkeys("abcd", "z"), gold)
    assert (p, r) == (0.5, 1.0)
    assert f == pytest.approx(2 / 3)


def test_bcubed_ignores_mentions_outside_gold():
    gold = {"a": "x", "b": "x"}
    pred = {"a": "1", "b": "1", "zzz": "1", "yyy": "1"}
    assert bcubed_f1(pred, gold) == 1.0


def test_bcubed_missing_prediction_counts_as_singleton():
    gold = {"a": "x", "b": "x"}
    p, r, _ = bcubed({"a": "1"}, gold)
    assert (p, r) == (1.0, 0.5)


def test_bcubed_empty_gold():
    assert bcubed_f1({}, {}) == 0.0

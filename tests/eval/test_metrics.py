import pytest

from kgrag.eval.metrics import (
    bootstrap_ci,
    full_evidence,
    paired_diff_ci,
    source_recall,
)


def test_source_recall_partial():
    assert source_recall(["a", "b"], ["a", "c"]) == 0.5


def test_source_recall_empty_gold_is_none():
    assert source_recall(["a"], []) is None


def test_source_recall_ignores_duplicates():
    assert source_recall(["a", "a", "a"], ["a", "c"]) == 0.5
    assert source_recall(["a"], ["a", "a"]) == 1.0


def test_full_evidence():
    assert full_evidence(["a", "c", "x"], ["a", "c"]) is True
    assert full_evidence(["a", "x"], ["a", "c"]) is False
    assert full_evidence(["a"], []) is None


def test_bootstrap_constant():
    assert bootstrap_ci([1.0] * 50) == (1.0, 1.0, 1.0)


def test_bootstrap_deterministic():
    vals = [i / 7 for i in range(50)]  # continuous-ish, so seeds yield distinct CIs
    assert bootstrap_ci(vals, seed=3) == bootstrap_ci(vals, seed=3)
    assert bootstrap_ci(vals, seed=3) != bootstrap_ci(vals, seed=4)


def test_bootstrap_empty_raises():
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_bootstrap_known_variance():
    import numpy as np

    vals = list(np.random.default_rng(42).integers(0, 2, 1000).astype(float))
    mean, lo, hi = bootstrap_ci(vals)
    assert mean == pytest.approx(sum(vals) / 1000)
    assert lo < mean < hi
    assert hi - lo == pytest.approx(0.062, abs=0.02)


def test_paired_diff():
    a = [1, 1, 1, 0] * 25
    b = [0, 1, 1, 0] * 25
    mean, lo, hi = paired_diff_ci(a, b)
    assert mean == 0.25
    assert lo > 0 and hi > lo
    assert paired_diff_ci(a, b, seed=1) == paired_diff_ci(a, b, seed=1)


def test_paired_diff_is_paired():
    # identical-per-question differences of exactly 1 -> zero-width interval,
    # which only happens if indices are resampled jointly.
    a = [0.0, 1.0, 0.0, 1.0] * 10
    b = [x - 1 for x in a]
    assert paired_diff_ci(a, b) == (1.0, 1.0, 1.0)


def test_paired_diff_length_mismatch():
    with pytest.raises(ValueError):
        paired_diff_ci([1.0, 0.0], [1.0])


def test_paired_diff_empty():
    with pytest.raises(ValueError):
        paired_diff_ci([], [])

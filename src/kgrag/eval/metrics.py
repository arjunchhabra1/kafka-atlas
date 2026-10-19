"""Retrieval metrics and bootstrap confidence intervals."""

import numpy as np


def source_recall(retrieved: list[str], gold: list[str]) -> float | None:
    """Fraction of distinct gold sources found among retrieved; None if no gold."""
    gold_set = set(gold)
    if not gold_set:
        return None
    return len(gold_set & set(retrieved)) / len(gold_set)


def full_evidence(retrieved: list[str], gold: list[str]) -> bool | None:
    """True when every gold source was retrieved; None if no gold."""
    gold_set = set(gold)
    if not gold_set:
        return None
    return gold_set <= set(retrieved)


def _percentile_ci(
    values: np.ndarray, n: int, seed: int, alpha: float
) -> tuple[float, float, float]:
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def bootstrap_ci(
    values: list[float], n: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Percentile bootstrap CI of the mean: (mean, lower, upper).

    `values` must contain no None; callers filter those out first.
    """
    return _percentile_ci(np.asarray(values, dtype=float), n, seed, alpha)


def paired_diff_ci(
    a: list[float],
    b: list[float],
    n: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Bootstrap CI of mean(a - b), resampling question indices jointly."""
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return _percentile_ci(diff, n, seed, alpha)

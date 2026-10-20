"""B-cubed precision/recall/F1 for clustering evaluation."""

from collections import defaultdict


def bcubed(pred: dict[str, str], gold: dict[str, str]) -> tuple[float, float, float]:
    """(precision, recall, F1) averaged over the mentions in `gold`.

    Predictions for mentions outside gold are ignored; a gold mention with no prediction
    counts as its own singleton cluster.
    """
    if not gold:
        return 0.0, 0.0, 0.0
    pred = {m: pred.get(m, f"\0singleton:{m}") for m in gold}
    pred_clusters: dict[str, set[str]] = defaultdict(set)
    gold_clusters: dict[str, set[str]] = defaultdict(set)
    for m in gold:
        pred_clusters[pred[m]].add(m)
        gold_clusters[gold[m]].add(m)
    p = r = 0.0
    for m in gold:
        both = len(pred_clusters[pred[m]] & gold_clusters[gold[m]])
        p += both / len(pred_clusters[pred[m]])
        r += both / len(gold_clusters[gold[m]])
    p, r = p / len(gold), r / len(gold)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def bcubed_f1(pred: dict[str, str], gold: dict[str, str]) -> float:
    return bcubed(pred, gold)[2]

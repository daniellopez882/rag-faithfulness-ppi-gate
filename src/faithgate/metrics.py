"""Classification and agreement metrics: precision/recall/F1, AUROC, Cohen's kappa and
Krippendorff's alpha."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel
from scipy.stats import rankdata


class PRF(BaseModel):
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int


def precision_recall_f1(
    y_true: Sequence[int] | np.ndarray, y_pred: Sequence[int] | np.ndarray
) -> PRF:
    """Scores for the positive (hallucinated) class; an undefined ratio is reported as 0."""
    t = np.asarray(y_true, dtype=bool)
    p = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(t & p))
    fp = int(np.sum(~t & p))
    fn = int(np.sum(t & ~p))
    tn = int(np.sum(~t & ~p))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PRF(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)


def auroc(y_true: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray) -> float:
    """Area under the ROC curve as the Mann-Whitney U statistic, with ties counted as one half."""
    t = np.asarray(y_true, dtype=bool)
    n_pos = int(t.sum())
    n_neg = t.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC needs both classes")
    ranks = rankdata(np.asarray(scores, dtype=float), method="average")
    return float((ranks[t].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def best_f1_threshold(
    y_true: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray
) -> float:
    """Threshold t maximising F1 of the rule ``score >= t``; ties go to the larger threshold."""
    t = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s, kind="stable")
    s_sorted, t_sorted = s[order], t[order]
    tp = np.cumsum(t_sorted)
    fp = np.cumsum(1 - t_sorted)
    # Only the last position of each run of equal scores is a reachable cut.
    last = np.append(s_sorted[1:] != s_sorted[:-1], True)
    tp, fp, cuts = tp[last], fp[last], s_sorted[last]
    f1 = 2 * tp / (tp + fp + t.sum())
    return float(cuts[int(np.argmax(f1))])


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("kappa needs two equally long, non-empty rating lists")
    cats = sorted(set(a) | set(b), key=str)
    index = {c: i for i, c in enumerate(cats)}
    table = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b, strict=True):
        table[index[x], index[y]] += 1
    n = table.sum()
    observed = np.trace(table) / n
    expected = float(table.sum(axis=1) @ table.sum(axis=0)) / n**2
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return float((observed - expected) / (1 - expected))


def krippendorff_alpha(
    ratings: Sequence[Sequence[float | None]],
    level: Literal["nominal", "interval"] = "nominal",
) -> float:
    """Krippendorff's alpha for a raters x units matrix; ``None`` or NaN marks a missing rating.

    Units with fewer than two ratings are not pairable and are ignored, as in Krippendorff (2011).
    """
    matrix = np.array([[np.nan if v is None else float(v) for v in row] for row in ratings])
    values = np.unique(matrix[~np.isnan(matrix)])
    if values.size == 0:
        raise ValueError("no ratings")
    index = {v: i for i, v in enumerate(values.tolist())}
    coincidence = np.zeros((values.size, values.size))
    for unit in matrix.T:
        present = unit[~np.isnan(unit)]
        m = present.size
        if m < 2:
            continue
        counts = np.zeros(values.size)
        for v in present.tolist():
            counts[index[v]] += 1
        pairs = np.outer(counts, counts) - np.diag(counts)
        coincidence += pairs / (m - 1)
    n_c = coincidence.sum(axis=1)
    n = n_c.sum()
    if level == "nominal":
        delta = 1.0 - np.eye(values.size)
    else:
        delta = np.subtract.outer(values, values) ** 2
    observed = float((coincidence * delta).sum())
    expected = float((np.outer(n_c, n_c) * delta).sum()) / (n - 1)
    if expected == 0.0:
        return 1.0
    return 1.0 - observed / expected

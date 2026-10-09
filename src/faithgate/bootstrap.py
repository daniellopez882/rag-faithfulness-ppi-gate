"""Stratified percentile bootstrap: resample within each stratum so every replicate keeps the
stratum sizes of the original sample (here, the task-type mix of RAGTruth's test split)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from pydantic import BaseModel


class BootstrapCI(BaseModel):
    estimate: float
    lower: float
    upper: float
    level: float
    n_resamples: int


def stratified_indices(strata: Sequence[str] | np.ndarray, rng: np.random.Generator) -> np.ndarray:
    labels = np.asarray(strata)
    parts = []
    for value in np.unique(labels):
        members = np.flatnonzero(labels == value)
        parts.append(rng.choice(members, size=members.size, replace=True))
    return np.concatenate(parts)


def stratified_bootstrap(
    statistic: Callable[[np.ndarray], float],
    n: int,
    strata: Sequence[str] | np.ndarray,
    n_resamples: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile CI of ``statistic(indices)`` where ``indices`` select rows of the data.

    Replicates on which the statistic is undefined (e.g. AUROC on a single-class resample)
    raise ``ValueError`` and are skipped.
    """
    if len(strata) != n:
        raise ValueError("one stratum label per row is required")
    rng = np.random.default_rng(seed)
    estimate = statistic(np.arange(n))
    values = []
    for _ in range(n_resamples):
        try:
            values.append(statistic(stratified_indices(strata, rng)))
        except ValueError:
            continue
    tail = (1 - level) / 2
    lower, upper = np.quantile(values, [tail, 1 - tail])
    return BootstrapCI(
        estimate=float(estimate),
        lower=float(lower),
        upper=float(upper),
        level=level,
        n_resamples=len(values),
    )

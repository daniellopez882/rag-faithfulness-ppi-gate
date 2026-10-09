from __future__ import annotations

import collections

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from faithgate.bootstrap import stratified_bootstrap, stratified_indices
from faithgate.metrics import auroc


@given(
    st.lists(st.sampled_from(["QA", "Summary", "Data2txt"]), min_size=1, max_size=80),
    st.integers(0, 10_000),
)
@settings(max_examples=150, deadline=None)
def test_resample_keeps_stratum_sizes(strata: list[str], seed: int) -> None:
    idx = stratified_indices(strata, np.random.default_rng(seed))
    labels = np.asarray(strata)
    assert collections.Counter(labels[idx].tolist()) == collections.Counter(strata)


def test_percentile_interval_covers_the_mean() -> None:
    rng = np.random.default_rng(0)
    hits = 0
    for trial in range(200):
        x = rng.normal(1.0, 1.0, 120)
        strata = np.repeat(["a", "b", "c"], 40)
        ci = stratified_bootstrap(
            lambda i, x=x: float(x[i].mean()), x.size, strata, n_resamples=300, seed=trial
        )
        hits += ci.lower <= 1.0 <= ci.upper
        assert ci.lower <= ci.estimate <= ci.upper
    assert hits / 200 == pytest.approx(0.95, abs=0.04)


def test_undefined_replicates_are_skipped() -> None:
    y = np.array([0] * 30 + [1])
    s = np.linspace(0, 1, 31)
    strata = np.array(["x"] * 31)
    ci = stratified_bootstrap(lambda i: auroc(y[i], s[i]), 31, strata, n_resamples=200, seed=1)
    assert 0 < ci.n_resamples < 200
    with pytest.raises(ValueError, match="stratum"):
        stratified_bootstrap(lambda i: 0.0, 3, ["a"], n_resamples=5)

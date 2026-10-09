from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from faithgate.ppi import (
    classical_mean,
    estimate_rate,
    interval,
    judge_only_mean,
    optimal_lambda,
    ppi_mean,
    ppi_plusplus_mean,
)


def biased_judge(
    n: int, rng: np.random.Generator, theta: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """Labels with rate theta; a judge that is informative but systematically over-flags."""
    y = (rng.uniform(size=n) < theta).astype(float)
    f = np.clip(0.25 + 0.5 * y + rng.normal(0, 0.15, n), 0, 1)
    return y, f


@given(
    st.lists(st.integers(0, 1), min_size=3, max_size=40),
    st.lists(st.floats(0, 1), min_size=3, max_size=40),
)
@settings(max_examples=100, deadline=None)
def test_lambda_zero_is_classical(y: list[int], f_unlab: list[float]) -> None:
    f_lab = np.linspace(0, 1, len(y))
    zero = ppi_mean(y, f_lab, f_unlab, lam=0.0)
    classic = classical_mean(y)
    assert zero.estimate == pytest.approx(classic.estimate)
    assert zero.se == pytest.approx(classic.se)
    assert zero.se >= 0


def test_perfect_judge_gives_unlabelled_mean_and_tiny_error() -> None:
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=200) < 0.4).astype(float)
    y_unlab = (rng.uniform(size=5000) < 0.4).astype(float)
    est = ppi_mean(y, y, y_unlab)
    assert est.estimate == pytest.approx(y_unlab.mean())
    assert est.se < classical_mean(y).se / 3


def test_optimal_lambda_is_regression_slope_over_inflation() -> None:
    rng = np.random.default_rng(1)
    y, f = biased_judge(400, rng)
    _, f_unlab = biased_judge(4000, rng)
    lam = optimal_lambda(y, f, f_unlab)
    slope = np.cov(y, f)[0, 1] / np.var(np.concatenate([f, f_unlab]), ddof=1)
    assert lam == pytest.approx(slope / (1 + 400 / 4000))
    assert optimal_lambda(y, np.full(400, 0.5), np.full(10, 0.5)) == 0.0


@pytest.mark.parametrize("method", ["classical", "ppi", "ppi++"])
def test_two_sided_coverage_is_near_nominal_with_a_biased_judge(method: str) -> None:
    rng = np.random.default_rng(2024)
    theta, alpha, reps = 0.3, 0.1, 2000
    hits = 0
    for _ in range(reps):
        y, f_lab = biased_judge(150, rng, theta)
        _, f_unlab = biased_judge(1500, rng, theta)
        iv = interval(estimate_rate(method, y, f_lab, f_unlab), alpha)
        hits += iv.lower <= theta <= iv.upper  # type: ignore[operator]
    coverage = hits / reps
    # Monte Carlo standard error at 2000 reps is about 0.0067; allow three of them.
    assert abs(coverage - (1 - alpha)) < 0.02


def test_upper_bound_coverage_and_judge_only_failure() -> None:
    rng = np.random.default_rng(7)
    theta, alpha, reps = 0.3, 0.05, 1500
    upper_hits = naive_hits = 0
    for _ in range(reps):
        y, f_lab = biased_judge(150, rng, theta)
        _, f_unlab = biased_judge(1500, rng, theta)
        bound = interval(ppi_plusplus_mean(y, f_lab, f_unlab), alpha, alternative="upper")
        upper_hits += theta <= bound.upper
        naive = interval(judge_only_mean(np.concatenate([f_lab, f_unlab])), alpha)
        naive_hits += naive.lower <= theta <= naive.upper  # type: ignore[operator]
    assert upper_hits / reps == pytest.approx(1 - alpha, abs=0.02)
    # The judge over-flags, so its own mean is biased upward and its interval almost never covers.
    assert naive_hits / reps < 0.05


def test_good_judge_shrinks_the_interval() -> None:
    rng = np.random.default_rng(3)
    widths = {"classical": [], "ppi": [], "ppi++": []}
    for _ in range(300):
        y, f_lab = biased_judge(100, rng)
        _, f_unlab = biased_judge(2000, rng)
        for method, store in widths.items():
            store.append(interval(estimate_rate(method, y, f_lab, f_unlab), 0.1).width)
    assert np.mean(widths["ppi++"]) < 0.8 * np.mean(widths["classical"])
    assert np.mean(widths["ppi++"]) <= np.mean(widths["ppi"]) + 1e-9


def test_useless_judge_ppi_plusplus_falls_back_towards_classical() -> None:
    rng = np.random.default_rng(5)
    ratios = []
    for _ in range(200):
        y = (rng.uniform(size=100) < 0.3).astype(float)
        noise_lab, noise_unlab = rng.uniform(size=100), rng.uniform(size=2000)
        pp = interval(ppi_plusplus_mean(y, noise_lab, noise_unlab), 0.1).width
        ratios.append(pp / interval(classical_mean(y), 0.1).width)
    assert np.mean(ratios) == pytest.approx(1.0, abs=0.05)


def test_clipped_lambda_stays_in_unit_interval() -> None:
    y = np.array([0, 1, 0, 1, 1, 0], dtype=float)
    f = 0.5 + 0.01 * (y - 0.5)  # nearly constant judge, so the raw slope is huge
    f_unlab = 0.5 + 0.001 * np.sin(np.arange(50))
    assert optimal_lambda(y, f, f_unlab) > 1.0
    est = ppi_plusplus_mean(y, f, f_unlab, clip=True)
    assert est.lam == 1.0


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        ppi_mean([1, 0], [0.5], [0.2, 0.3])
    with pytest.raises(ValueError, match="unlabelled"):
        ppi_mean([1, 0], [0.5, 0.5], [])
    with pytest.raises(ValueError, match="alpha"):
        interval(classical_mean([0, 1]), alpha=1.5)
    with pytest.raises(ValueError, match="unknown estimator"):
        estimate_rate("bayes", [0, 1], [0.1, 0.2], [0.3, 0.4])

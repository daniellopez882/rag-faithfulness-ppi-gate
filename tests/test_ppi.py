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


@pytest.mark.parametrize("kind", ["score", "wald"])
@pytest.mark.parametrize("method", ["classical", "ppi", "ppi++"])
def test_two_sided_coverage_is_near_nominal_with_a_biased_judge(method: str, kind: str) -> None:
    rng = np.random.default_rng(2024)
    theta, alpha, reps = 0.3, 0.1, 2000
    hits = 0
    for _ in range(reps):
        y, f_lab = biased_judge(150, rng, theta)
        _, f_unlab = biased_judge(1500, rng, theta)
        iv = interval(estimate_rate(method, y, f_lab, f_unlab), alpha, kind=kind)  # type: ignore[arg-type]
        hits += iv.lower <= theta <= iv.upper  # type: ignore[operator]
    coverage = hits / reps
    # Monte Carlo standard error at 2000 reps is about 0.0067; allow three of them. With a judge
    # this strong the score interval may run conservative, never liberal.
    assert coverage > (1 - alpha) - 0.02
    assert coverage < (1 - alpha) + (0.02 if kind == "wald" else 0.04)


def test_upper_bound_coverage_and_judge_only_failure() -> None:
    rng = np.random.default_rng(7)
    theta, alpha, reps = 0.3, 0.05, 1500
    score_hits = wald_hits = naive_hits = 0
    for _ in range(reps):
        y, f_lab = biased_judge(150, rng, theta)
        _, f_unlab = biased_judge(1500, rng, theta)
        est = ppi_plusplus_mean(y, f_lab, f_unlab)
        score_hits += theta <= interval(est, alpha, alternative="upper").upper
        wald_hits += theta <= interval(est, alpha, alternative="upper", kind="wald").upper
        naive = interval(judge_only_mean(np.concatenate([f_lab, f_unlab])), alpha)
        naive_hits += naive.lower <= theta <= naive.upper  # type: ignore[operator]
    assert wald_hits / reps == pytest.approx(1 - alpha, abs=0.02)
    assert 1 - alpha - 0.02 < score_hits / reps < 1 - alpha + 0.035
    # The judge over-flags, so its own mean is biased upward and its interval almost never covers.
    assert naive_hits / reps < 0.05


def calibrated_judge(
    y: np.ndarray, rng: np.random.Generator, separation: float = 1.2, prior: float = 0.3
) -> np.ndarray:
    """P(hallucinated | judge signal) for a signal N(separation * y, 1), computed with a prior
    above the true rate: a judge calibrated on a set with more hallucinations (AUROC 0.80)."""
    z = rng.normal(separation * y, 1.0)
    logit = separation * z - separation**2 / 2 + np.log(prior / (1 - prior))
    return 1 / (1 + np.exp(-logit))


@pytest.mark.parametrize("n", [30, 50, 100])
@pytest.mark.parametrize("theta", [0.02, 0.05, 0.1])
def test_upper_bound_holds_at_low_rates_and_small_n(theta: float, n: int) -> None:
    """The regime gate budgets live in: a few per cent hallucinated, a few dozen labels."""
    rng = np.random.default_rng(int(theta * 1000) * 1000 + n)
    alpha, reps = 0.05, 1500
    hits = {"classical": 0, "ppi++": 0}
    for _ in range(reps):
        y = (rng.uniform(size=n) < theta).astype(float)
        f_lab = calibrated_judge(y, rng)
        f_unlab = calibrated_judge((rng.uniform(size=1000) < theta).astype(float), rng)
        for method in hits:
            bound = interval(estimate_rate(method, y, f_lab, f_unlab), alpha, alternative="upper")
            hits[method] += theta <= bound.upper
    # Monte Carlo standard error at 1500 reps is about 0.0056.
    for method, count in hits.items():
        assert count / reps > 1 - alpha - 0.02, method


def test_wald_bound_fails_where_the_score_bound_holds() -> None:
    rng = np.random.default_rng(31)
    theta, n, reps = 0.05, 30, 1000
    wald_hits = 0
    for _ in range(reps):
        y = (rng.uniform(size=n) < theta).astype(float)
        f_lab = calibrated_judge(y, rng)
        f_unlab = calibrated_judge((rng.uniform(size=1000) < theta).astype(float), rng)
        est = ppi_plusplus_mean(y, f_lab, f_unlab)
        wald_hits += theta <= interval(est, 0.05, alternative="upper", kind="wald").upper
    # A fifth of these samples hold no hallucinated response, and Wald puts their bound at 0.
    assert wald_hits / reps < 0.85


def test_no_positive_label_still_gives_a_positive_bound() -> None:
    n, z = 30, 1.6448536269514722
    f_lab, f_unlab = np.linspace(0.05, 0.3, n), np.linspace(0.0, 0.4, 500)
    wilson = z * z / (n - 1 + z * z)
    for est in (classical_mean(np.zeros(n)), ppi_plusplus_mean(np.zeros(n), f_lab, f_unlab)):
        assert interval(est, 0.05, alternative="upper", kind="wald").upper == 0.0
        assert interval(est, 0.05, alternative="upper").upper == pytest.approx(wilson)
    assert ppi_plusplus_mean(np.zeros(n), f_lab, f_unlab).lam == 0.0


@given(
    st.lists(st.integers(0, 1), min_size=2, max_size=60),
    st.floats(0.5, 0.995),
)
@settings(max_examples=150, deadline=None)
def test_score_interval_contains_the_wald_interval(labels: list[int], level: float) -> None:
    y = np.asarray(labels, dtype=float)
    f_lab = np.linspace(0, 1, y.size)
    f_unlab = np.linspace(0.2, 0.6, 40)
    for est in (
        classical_mean(y),
        ppi_mean(y, f_lab, f_unlab),
        ppi_plusplus_mean(y, f_lab, f_unlab),
    ):
        score = interval(est, 1 - level)
        wald = interval(est, 1 - level, kind="wald")
        assert score.lower <= wald.lower + 1e-12  # type: ignore[operator]
        assert score.upper >= wald.upper - 1e-12
        assert score.lower <= est.estimate <= score.upper  # type: ignore[operator]


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
    assert ppi_plusplus_mean(y, f, f_unlab).lam == 1.0
    assert ppi_plusplus_mean(y, f, f_unlab, clip=False).lam == optimal_lambda(y, f, f_unlab)
    assert ppi_plusplus_mean(y, 1 - f, f_unlab).lam == 0.0


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        ppi_mean([1, 0], [0.5], [0.2, 0.3])
    with pytest.raises(ValueError, match="unlabelled"):
        ppi_mean([1, 0], [0.5, 0.5], [])
    with pytest.raises(ValueError, match="unlabelled"):
        ppi_plusplus_mean([0, 1, 1], [0.1, 0.9, 0.8], [])
    with pytest.raises(ValueError, match="unlabelled"):
        estimate_rate("ppi++", [0, 1, 1], [0.1, 0.9, 0.8], [])
    with pytest.raises(ValueError, match="differ in length"):
        optimal_lambda([1, 0, 1], [0.5], [0.2, 0.3])
    with pytest.raises(ValueError, match="two labels"):
        interval(classical_mean([1]), alpha=0.05)
    with pytest.raises(ValueError, match="alpha"):
        interval(classical_mean([0, 1]), alpha=1.5)
    with pytest.raises(ValueError, match="unknown estimator"):
        estimate_rate("bayes", [0, 1], [0.1, 0.2], [0.3, 0.4])

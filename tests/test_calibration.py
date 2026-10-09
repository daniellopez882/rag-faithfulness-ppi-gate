from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from faithgate.calibration import (
    METHODS,
    HistogramCalibrator,
    IsotonicCalibrator,
    Method,
    PlattCalibrator,
    TemperatureCalibrator,
    brier_score,
    calibrator_from_params,
    expected_calibration_error,
    logit,
    make_calibrator,
    maximum_calibration_error,
    pava,
    reliability_bins,
    sigmoid,
)

# Scores on a 1e-4 grid: scikit-learn merges x values closer than float resolution (1e-15),
# which only matters for denormal-sized gaps that no judge produces.
scores_and_labels = st.integers(min_value=2, max_value=60).flatmap(
    lambda n: st.tuples(
        st.lists(st.integers(0, 10_000).map(lambda k: k / 10_000), min_size=n, max_size=n),
        st.lists(st.integers(0, 1), min_size=n, max_size=n),
    )
)


@given(scores_and_labels)
@settings(max_examples=150, deadline=None)
def test_isotonic_matches_sklearn(data: tuple[list[float], list[int]]) -> None:
    s, y = np.array(data[0]), np.array(data[1])
    ours = IsotonicCalibrator().fit(s, y)
    ref = IsotonicRegression(out_of_bounds="clip").fit(s, y)
    grid = np.concatenate([s, np.linspace(-0.2, 1.2, 41)])
    np.testing.assert_allclose(ours.predict(grid), ref.predict(grid), atol=1e-9)


@given(scores_and_labels)
@settings(max_examples=150, deadline=None)
def test_isotonic_output_is_monotone(data: tuple[list[float], list[int]]) -> None:
    s, y = np.array(data[0]), np.array(data[1])
    grid = np.linspace(-0.5, 1.5, 201)
    pred = IsotonicCalibrator().fit(s, y).predict(grid)
    assert np.all(np.diff(pred) >= -1e-12)
    assert pred.min() >= 0
    assert pred.max() <= 1


def test_pava_pools_violators() -> None:
    np.testing.assert_allclose(
        pava(np.array([1.0, 3.0, 2.0, 4.0]), np.ones(4)), [1.0, 2.5, 2.5, 4.0]
    )
    np.testing.assert_allclose(pava(np.array([3.0, 1.0]), np.array([1.0, 3.0])), [1.5, 1.5])


def _miscalibrated(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """True probability p; the judge reports p ** 3, so it under-states every risk."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < p).astype(int)
    return p**3, y


@pytest.mark.parametrize("method", ["platt", "isotonic", "histogram"])
def test_calibration_reduces_out_of_sample_ece(method: Method) -> None:
    s_fit, y_fit = _miscalibrated(4000, 1)
    s_test, y_test = _miscalibrated(4000, 2)
    calibrator = make_calibrator(method).fit(s_fit, y_fit)
    before = expected_calibration_error(s_test, y_test)
    after = expected_calibration_error(calibrator.predict(s_test), y_test)
    assert before > 0.15
    assert after < 0.05


def test_temperature_recovers_known_temperature() -> None:
    rng = np.random.default_rng(3)
    z = rng.normal(0, 2, 20000)
    y = (rng.uniform(0, 1, z.size) < sigmoid(z)).astype(int)
    reported = sigmoid(2.5 * z)  # over-confident by a factor of 2.5
    temp = TemperatureCalibrator().fit(reported, y)
    assert temp.temperature == pytest.approx(2.5, rel=0.05)


def test_platt_recovers_slope_and_intercept() -> None:
    rng = np.random.default_rng(4)
    s = rng.uniform(0.01, 0.99, 20000)
    y = (rng.uniform(0, 1, s.size) < sigmoid(0.5 * logit(s) - 1.0)).astype(int)
    platt = PlattCalibrator().fit(s, y)
    assert platt.a == pytest.approx(0.5, abs=0.05)
    assert platt.b == pytest.approx(-1.0, abs=0.08)


def test_platt_stays_finite_on_separable_data() -> None:
    platt = PlattCalibrator().fit([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    probs = platt.predict([0.1, 0.9])
    assert np.isfinite([platt.a, platt.b]).all()
    assert 0 < probs[0] < 0.5 < probs[1] < 1


def test_histogram_bins_are_label_means() -> None:
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    y = np.array([0, 0, 0, 1, 1, 0, 1, 1])
    hist = HistogramCalibrator(n_bins=2).fit(s, y)
    np.testing.assert_allclose(hist.predict([0.0, 0.25, 0.65, 1.0]), [0.25, 0.25, 0.75, 0.75])


@pytest.mark.parametrize("method", METHODS)
def test_params_round_trip(method: Method) -> None:
    s, y = _miscalibrated(300, 5)
    fitted = make_calibrator(method).fit(s, y)
    restored = calibrator_from_params(fitted.params())
    grid = np.linspace(0, 1, 37)
    np.testing.assert_allclose(restored.predict(grid), fitted.predict(grid))


@given(scores_and_labels)
@settings(max_examples=100, deadline=None)
def test_reliability_bins_match_sklearn_calibration_curve(
    data: tuple[list[float], list[int]],
) -> None:
    p, y = np.array(data[0]), np.array(data[1])
    if len(set(y.tolist())) < 2:
        return
    bins = reliability_bins(p, y, n_bins=10)
    prob_true, prob_pred = calibration_curve(y, p, n_bins=10, strategy="uniform")
    np.testing.assert_allclose([b.fraction_positive for b in bins], prob_true, atol=1e-12)
    np.testing.assert_allclose([b.mean_predicted for b in bins], prob_pred, atol=1e-12)


@given(scores_and_labels)
@settings(max_examples=100, deadline=None)
def test_ece_mce_brier_ranges_and_brier_parity(data: tuple[list[float], list[int]]) -> None:
    p, y = np.array(data[0]), np.array(data[1])
    ece = expected_calibration_error(p, y)
    mce = maximum_calibration_error(p, y)
    assert 0 <= ece <= mce + 1e-12 <= 1 + 1e-12
    assert brier_score(p, y) == pytest.approx(brier_score_loss(y, p, pos_label=1), abs=1e-12)


def test_perfectly_calibrated_bins_have_zero_ece() -> None:
    p = np.array([0.25] * 4 + [0.75] * 4)
    y = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    assert expected_calibration_error(p, y, n_bins=4) == pytest.approx(0.0)

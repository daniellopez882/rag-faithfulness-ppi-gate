from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from faithgate.artifacts import fit_artifact
from faithgate.calibration import METHODS
from faithgate.conformal import CRCThreshold, crc_threshold, flag, realised_risk
from faithgate.records import Scored


def draw(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A judge with continuous scores, higher on hallucinated responses; base rate 0.35."""
    y = (rng.uniform(size=n) < 0.35).astype(int)
    s = 1 / (1 + np.exp(-(rng.normal(0, 1.2, n) + 1.5 * y - 0.5)))
    return s, y


@pytest.mark.parametrize(
    ("risk", "alpha"), [("slip", 0.05), ("slip", 0.1), ("miss", 0.1), ("miss", 0.2)]
)
def test_expected_risk_is_controlled(risk: str, alpha: float) -> None:
    rng = np.random.default_rng(11)
    n_cal, reps = 200, 1500
    risks = []
    for _ in range(reps):
        s_cal, y_cal = draw(n_cal, rng)
        t = crc_threshold(s_cal, y_cal, alpha, risk=risk).threshold  # type: ignore[arg-type]
        s_new, y_new = draw(400, rng)
        risks.append(realised_risk(s_new, y_new, t, risk))  # type: ignore[arg-type]
    mean_risk = float(np.mean(risks))
    mc_error = float(np.std(risks) / math.sqrt(reps))
    assert mean_risk <= alpha + 3 * mc_error
    # CRC is tight for continuous scores: E[risk] >= alpha - 2B/(n+1), with n positives for "miss".
    n_eff = n_cal if risk == "slip" else 0.35 * n_cal
    assert mean_risk >= alpha - 2 / (n_eff + 1) - 3 * mc_error


@given(
    st.lists(st.floats(0, 1, allow_nan=False), min_size=5, max_size=60),
    st.floats(0.01, 0.5),
    st.floats(0.01, 0.5),
)
@settings(max_examples=150, deadline=None)
def test_threshold_is_monotone_in_alpha(scores: list[float], a1: float, a2: float) -> None:
    labels = [int(i % 3 == 0) for i in range(len(scores))]
    lo, hi = sorted((a1, a2))
    assert (
        crc_threshold(scores, labels, lo).threshold <= crc_threshold(scores, labels, hi).threshold
    )


def test_threshold_satisfies_the_inflated_bound() -> None:
    rng = np.random.default_rng(3)
    s, y = draw(300, rng)
    for alpha in (0.02, 0.05, 0.1, 0.3):
        result = crc_threshold(s, y, alpha)
        n = result.n_calibration
        assert (n * result.calibration_risk + 1) / (n + 1) <= alpha + 1e-12
        # One step higher (the next hallucinated score) would break the bound.
        positives = np.sort(s[y == 1])
        above = positives[positives > result.threshold]
        if above.size:
            worse = realised_risk(s, y, float(above[0]) + 1e-12, "slip")
            assert (n * worse + 1) / (n + 1) > alpha


def test_edge_cases() -> None:
    # alpha below 1/(n+1): only flagging everything is safe.
    tight = crc_threshold([0.2, 0.9], [0, 1], alpha=0.1)
    assert tight.threshold == -math.inf
    assert tight.flag_rate == 1.0
    # No hallucinations in calibration: nothing needs flagging.
    loose = crc_threshold([0.1, 0.2, 0.3] * 10, [0] * 30, alpha=0.1)
    assert loose.threshold == math.inf
    assert flag([0.5], loose.threshold).tolist() == [False]
    with pytest.raises(ValueError, match="alpha"):
        crc_threshold([0.1], [1], alpha=0.0)


def _rows(scores: np.ndarray, labels: np.ndarray) -> list[Scored]:
    return [
        Scored(id=str(i), judge="j", score=float(s), label=int(y))
        for i, (s, y) in enumerate(zip(scores, labels, strict=True))
    ]


def test_threshold_does_not_depend_on_the_calibrator() -> None:
    s, y = draw(120, np.random.default_rng(5))
    raw = crc_threshold(s, y, alpha=0.1).threshold
    for method in METHODS:
        artifact = fit_artifact(_rows(s, y), method=method, alpha=0.1)
        assert artifact.crc.threshold == raw
        assert artifact.flags(s).tolist() == (s >= raw).tolist()


@pytest.mark.parametrize("method", ["platt", "histogram"])
def test_risk_is_controlled_through_fit_artifact(method: str) -> None:
    """The calibrator is fitted on the same labels the threshold is chosen on. With a judge that
    carries no signal Platt takes its direction from noise, and a threshold picked on its output
    let 0.217 slip through at alpha 0.2 with this seed; histogram bins need not be monotone
    either. The artifact flags on the raw score, which was fixed before the labels were seen."""
    rng = np.random.default_rng(17)
    alpha, theta, n, reps = 0.2, 0.4, 30, 600
    fresh = np.linspace(0.0, 1.0, 2001)  # new scores are U(0, 1), independent of the label
    risks: list[float] = []
    while len(risks) < reps:
        s, y = rng.uniform(size=n), (rng.uniform(size=n) < theta).astype(int)
        if y.min() == y.max():
            continue
        artifact = fit_artifact(_rows(s, y), method=method, alpha=alpha)  # type: ignore[arg-type]
        risks.append(theta * float(np.mean(~artifact.flags(fresh))))
    mean_risk = float(np.mean(risks))
    assert mean_risk <= alpha + 3 * float(np.std(risks)) / math.sqrt(reps)


def test_infinite_threshold_survives_json() -> None:
    result = crc_threshold([0.2, 0.9], [0, 1], alpha=0.1)
    restored = CRCThreshold.model_validate_json(result.model_dump_json())
    assert restored.threshold == -math.inf

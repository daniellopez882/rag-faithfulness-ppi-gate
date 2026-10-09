"""Conformal risk control (Angelopoulos, Bates, Fisch, Lei and Schuster, 2022) for the flag
threshold of a hallucination judge.

A response is flagged when its score is at least ``t``. For a loss that is bounded by B and grows
with ``t``, CRC picks

    t_hat = sup { t : (n * R_n(t) + B) / (n + 1) <= alpha }

on n calibration responses, which guarantees E[L_{n+1}(t_hat)] <= alpha for a new response that
is exchangeable with them. Two losses are offered:

* ``slip``: 1 if the response is hallucinated and not flagged. The guarantee bounds the expected
  share of all responses that are hallucinated and pass unflagged.
* ``miss``: the same loss computed over hallucinated responses only, which bounds the expected
  miss rate (false-negative rate) among hallucinated responses.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

Risk = Literal["slip", "miss"]


class CRCThreshold(BaseModel):
    # The threshold can be +/-inf; keep it round-trippable through JSON.
    model_config = ConfigDict(ser_json_inf_nan="strings")

    risk: Risk
    alpha: float
    threshold: float
    n_calibration: int
    calibration_risk: float
    flag_rate: float


def crc_threshold(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    alpha: float,
    risk: Risk = "slip",
) -> CRCThreshold:
    """Largest threshold whose inflated empirical risk stays within ``alpha``.

    ``threshold`` may be ``-inf`` (flag everything, which always has zero risk) or ``+inf``
    (flag nothing) when the calibration set allows it.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    if s.size != y.size or s.size == 0:
        raise ValueError("scores and labels must be equally long and non-empty")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    pool = s if risk == "slip" else s[y == 1]
    n = pool.size
    positives = np.sort(s[y == 1])
    # With t at the k-th smallest hallucinated score, k - 1 hallucinated responses are missed
    # (ties at t are flagged). Allowed misses m satisfy (m + 1) / (n + 1) <= alpha.
    allowed = math.floor(alpha * (n + 1) - 1 + 1e-12)
    if n == 0 or allowed < 0:
        threshold = -math.inf
    elif allowed >= positives.size:
        threshold = math.inf
    else:
        threshold = float(positives[allowed])
    missed = int(np.sum(positives < threshold))
    return CRCThreshold(
        risk=risk,
        alpha=alpha,
        threshold=threshold,
        n_calibration=n,
        calibration_risk=missed / n if n else 0.0,
        flag_rate=float(np.mean(s >= threshold)),
    )


def flag(scores: Sequence[float] | np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(scores, dtype=float) >= threshold


def realised_risk(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    threshold: float,
    risk: Risk,
) -> float:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    missed = (y == 1) & (s < threshold)
    if risk == "slip":
        return float(missed.mean())
    positives = y == 1
    return float(missed[positives].mean()) if positives.any() else 0.0

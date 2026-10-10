"""Mean estimation with prediction-powered inference.

Setting: n responses carry a human label Y and a judge prediction f(X); N further responses carry
only f(X). All n + N are an i.i.d. sample from the population whose hallucination rate
theta = E[Y] we want, and the labelled ones are a uniformly random subset of them.

* classical: the human labels alone.
* PPI (Angelopoulos et al., 2023): judge mean on the unlabelled set, debiased by the mean
  judge error measured on the labelled set.
* PPI++ (Angelopoulos, Duchi and Zrnic, 2023): the same with the judge term scaled by a
  plug-in lambda that minimises the variance; lambda = 0 is classical and lambda = 1 is PPI.

Two interval constructions are offered. ``wald`` is the textbook CLT interval, estimate +/- z se.
It is asymptotically valid whatever the judge's bias, but at small n and low rates it is badly
optimistic: with no hallucinated response among the labels the label variance is 0, lambda is 0
and the upper bound is exactly 0. ``score`` (the default) inverts the same normal approximation
with the label variance taken at the hypothesised rate t, t(1 - t), instead of at the sample
mean, as the Wilson interval does for a proportion. The part of the label variance that the judge
term removes is estimated from the sample and carried over to t in two ways, as the same amount
and as the same fraction; the interval uses whichever gives the larger variance, and never ends
inside the Wald interval. With lambda = 0 it is the Wilson interval (with n - 1 in place of n,
matching the sample variance), so 30 clean labels give a 95% upper bound of about 0.085, not 0.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel
from scipy.stats import norm

Alternative = Literal["two-sided", "upper"]
Kind = Literal["score", "wald"]
ArrayLike = Sequence[float] | np.ndarray


class MeanEstimate(BaseModel):
    """A point estimate with the per-observation variances its intervals are built from.

    ``var_labels`` is the sample variance of Y, ``var_rectified`` that of Y - lambda f on the
    labelled responses and ``var_judge_term`` that of lambda f on the unlabelled ones, so
    ``se**2 = var_rectified / n_labelled + var_judge_term / n_unlabelled``.
    """

    method: str
    estimate: float
    se: float
    n_labelled: int
    n_unlabelled: int
    lam: float | None = None
    var_labels: float = 0.0
    var_rectified: float = 0.0
    var_judge_term: float = 0.0


class Interval(BaseModel):
    method: str
    estimate: float
    lower: float | None
    upper: float
    alpha: float
    alternative: Alternative
    kind: Kind = "score"
    lam: float | None = None

    @property
    def width(self) -> float:
        if self.lower is None:
            raise ValueError("a one-sided interval has no width")
        return self.upper - self.lower


def _arrays(*arrays: ArrayLike) -> list[np.ndarray]:
    return [np.asarray(a, dtype=float) for a in arrays]


def _var(x: np.ndarray) -> float:
    return float(np.var(x, ddof=1)) if x.size > 1 else 0.0


def classical_mean(y: ArrayLike) -> MeanEstimate:
    (yy,) = _arrays(y)
    if yy.size == 0:
        raise ValueError("the classical estimate needs at least one label")
    var_y = _var(yy)
    return MeanEstimate(
        method="classical",
        estimate=float(yy.mean()),
        se=float(np.sqrt(var_y / yy.size)),
        n_labelled=yy.size,
        n_unlabelled=0,
        var_labels=var_y,
        var_rectified=var_y,
    )


def _check_ppi_inputs(yy: np.ndarray, fl: np.ndarray, fu: np.ndarray) -> None:
    if yy.size != fl.size:
        raise ValueError("labels and labelled predictions differ in length")
    if yy.size == 0:
        raise ValueError("PPI needs labelled responses")
    if fu.size == 0:
        raise ValueError("PPI needs unlabelled predictions")


def ppi_mean(
    y: ArrayLike, f_labelled: ArrayLike, f_unlabelled: ArrayLike, lam: float = 1.0
) -> MeanEstimate:
    """theta_hat = mean(Y - lam f) + lam mean(f_unlabelled), for a fixed lambda (1 = plain PPI)."""
    yy, fl, fu = _arrays(y, f_labelled, f_unlabelled)
    _check_ppi_inputs(yy, fl, fu)
    rectified = yy - lam * fl
    estimate = float(rectified.mean() + lam * fu.mean())
    var_rectified = _var(rectified)
    var_judge_term = lam**2 * _var(fu)
    variance = var_rectified / yy.size + var_judge_term / fu.size
    return MeanEstimate(
        method="ppi" if lam == 1.0 else "ppi-lambda",
        estimate=estimate,
        se=float(np.sqrt(variance)),
        n_labelled=yy.size,
        n_unlabelled=fu.size,
        lam=lam,
        var_labels=_var(yy),
        var_rectified=var_rectified,
        var_judge_term=var_judge_term,
    )


def optimal_lambda(y: ArrayLike, f_labelled: ArrayLike, f_unlabelled: ArrayLike) -> float:
    """Plug-in lambda* = Cov(Y, f) / ((1 + n/N) Var(f)), Var(f) pooled over both sets."""
    yy, fl, fu = _arrays(y, f_labelled, f_unlabelled)
    _check_ppi_inputs(yy, fl, fu)
    var_f = _var(np.concatenate([fl, fu]))
    if var_f == 0.0 or yy.size < 2:
        return 0.0
    cov = float(np.cov(yy, fl, ddof=1)[0, 1])
    return cov / ((1.0 + yy.size / fu.size) * var_f)


def ppi_plusplus_mean(
    y: ArrayLike, f_labelled: ArrayLike, f_unlabelled: ArrayLike, clip: bool = True
) -> MeanEstimate:
    """PPI with the plug-in lambda. ``clip`` keeps lambda in [0, 1], between the classical
    estimate and PPI: from a few dozen labels with few positives the plug-in lambda is noisy,
    and values outside that range buy little and cost coverage."""
    lam = optimal_lambda(y, f_labelled, f_unlabelled)
    if clip:
        lam = min(max(lam, 0.0), 1.0)
    estimate = ppi_mean(y, f_labelled, f_unlabelled, lam=lam)
    return estimate.model_copy(update={"method": "ppi++"})


def judge_only_mean(f_all: ArrayLike) -> MeanEstimate:
    """Mean judge prediction with its sampling error only. Biased whenever the judge is; it is
    reported to show what skipping the human labels costs, not as an estimator to use."""
    (ff,) = _arrays(f_all)
    return MeanEstimate(
        method="judge-only",
        estimate=float(ff.mean()),
        se=float(np.sqrt(_var(ff) / ff.size)),
        n_labelled=0,
        n_unlabelled=ff.size,
    )


def _score_limits(est: MeanEstimate, z: float) -> tuple[float, float]:
    """Smallest and largest t with (t - theta_hat)^2 <= z^2 V(t), where V(t) is the variance of
    theta_hat with the label variance taken at t (see the module docstring)."""
    n, theta = est.n_labelled, est.estimate
    bernoulli = n / (n - 1)  # puts t(1 - t) on the scale of a ddof=1 sample variance
    judge_term = est.var_judge_term / est.n_unlabelled if est.n_unlabelled else 0.0
    # Each form is V(t) = (scale * bernoulli * t(1 - t) - shift) / n + judge_term.
    forms = [(1.0, est.var_labels - est.var_rectified)]
    if est.var_labels > 0:
        forms.append((est.var_rectified / est.var_labels, 0.0))
    lower, upper = theta - z * est.se, theta + z * est.se
    for scale, shift in forms:
        a = z * z * bernoulli * scale / n
        d = z * z * (judge_term - shift / n)
        # (t - theta)^2 <= a t (1 - t) + d, i.e. (1 + a) t^2 - (2 theta + a) t + theta^2 - d <= 0
        b = 2 * theta + a
        disc = b * b - 4 * (1 + a) * (theta * theta - d)
        if disc >= 0:
            root = math.sqrt(disc)
            lower = min(lower, (b - root) / (2 * (1 + a)))
            upper = max(upper, (b + root) / (2 * (1 + a)))
    return lower, upper


def interval(
    estimate: MeanEstimate,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    kind: Kind = "score",
) -> Interval:
    """Confidence interval, or with ``alternative="upper"`` a one-sided upper bound.

    The judge-only estimate has no labels to take a score variance from, so it always gets the
    Wald interval; ``Interval.kind`` records which construction was used.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if estimate.n_labelled == 0:
        kind = "wald"
    elif kind == "score" and estimate.n_labelled < 2:
        raise ValueError("a score interval needs at least two labels")
    z = float(norm.ppf(1 - alpha / 2 if alternative == "two-sided" else 1 - alpha))
    if kind == "score":
        low, high = _score_limits(estimate, z)
    else:
        low, high = estimate.estimate - z * estimate.se, estimate.estimate + z * estimate.se
    return Interval(
        method=estimate.method,
        estimate=estimate.estimate,
        lower=low if alternative == "two-sided" else None,
        upper=high,
        alpha=alpha,
        alternative=alternative,
        kind=kind,
        lam=estimate.lam,
    )


def estimate_rate(
    method: str,
    y: ArrayLike,
    f_labelled: ArrayLike,
    f_unlabelled: ArrayLike,
) -> MeanEstimate:
    """Dispatch by name: 'classical', 'ppi' or 'ppi++'."""
    if method == "classical":
        return classical_mean(y)
    if method == "ppi":
        return ppi_mean(y, f_labelled, f_unlabelled)
    if method == "ppi++":
        return ppi_plusplus_mean(y, f_labelled, f_unlabelled)
    raise ValueError(f"unknown estimator {method!r}; expected classical, ppi or ppi++")

"""Mean estimation with prediction-powered inference.

Setting: n responses carry a human label Y and a judge prediction f(X); N further responses carry
only f(X). All n + N are an i.i.d. sample from the population whose hallucination rate
theta = E[Y] we want, and the labelled ones are a uniformly random subset of them.

* classical: the human labels alone.
* PPI (Angelopoulos et al., 2023): judge mean on the unlabelled set, debiased by the mean
  judge error measured on the labelled set.
* PPI++ (Angelopoulos, Duchi and Zrnic, 2023): the same with the judge term scaled by a
  plug-in lambda that minimises the variance; lambda = 0 is classical and lambda = 1 is PPI.

The intervals are CLT intervals: asymptotically valid whatever the judge's bias, which only
changes their width.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel
from scipy.stats import norm

Alternative = Literal["two-sided", "upper"]
ArrayLike = Sequence[float] | np.ndarray


class MeanEstimate(BaseModel):
    method: str
    estimate: float
    se: float
    n_labelled: int
    n_unlabelled: int
    lam: float | None = None


class Interval(BaseModel):
    method: str
    estimate: float
    lower: float | None
    upper: float
    alpha: float
    alternative: Alternative
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
    return MeanEstimate(
        method="classical",
        estimate=float(yy.mean()),
        se=float(np.sqrt(_var(yy) / yy.size)),
        n_labelled=yy.size,
        n_unlabelled=0,
    )


def ppi_mean(
    y: ArrayLike, f_labelled: ArrayLike, f_unlabelled: ArrayLike, lam: float = 1.0
) -> MeanEstimate:
    """theta_hat = mean(Y - lam f) + lam mean(f_unlabelled), for a fixed lambda (1 = plain PPI)."""
    yy, fl, fu = _arrays(y, f_labelled, f_unlabelled)
    if yy.size != fl.size:
        raise ValueError("labels and labelled predictions differ in length")
    if fu.size == 0:
        raise ValueError("PPI needs unlabelled predictions")
    rectified = yy - lam * fl
    estimate = float(rectified.mean() + lam * fu.mean())
    variance = _var(rectified) / yy.size + lam**2 * _var(fu) / fu.size
    return MeanEstimate(
        method="ppi" if lam == 1.0 else "ppi-lambda",
        estimate=estimate,
        se=float(np.sqrt(variance)),
        n_labelled=yy.size,
        n_unlabelled=fu.size,
        lam=lam,
    )


def optimal_lambda(y: ArrayLike, f_labelled: ArrayLike, f_unlabelled: ArrayLike) -> float:
    """Plug-in lambda* = Cov(Y, f) / ((1 + n/N) Var(f)), Var(f) pooled over both sets."""
    yy, fl, fu = _arrays(y, f_labelled, f_unlabelled)
    var_f = _var(np.concatenate([fl, fu]))
    if var_f == 0.0 or yy.size < 2:
        return 0.0
    cov = float(np.cov(yy, fl, ddof=1)[0, 1])
    return cov / ((1.0 + yy.size / fu.size) * var_f)


def ppi_plusplus_mean(
    y: ArrayLike, f_labelled: ArrayLike, f_unlabelled: ArrayLike, clip: bool = False
) -> MeanEstimate:
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


def interval(
    estimate: MeanEstimate, alpha: float = 0.05, alternative: Alternative = "two-sided"
) -> Interval:
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if alternative == "two-sided":
        z = float(norm.ppf(1 - alpha / 2))
        lower: float | None = estimate.estimate - z * estimate.se
    else:
        z = float(norm.ppf(1 - alpha))
        lower = None
    return Interval(
        method=estimate.method,
        estimate=estimate.estimate,
        lower=lower,
        upper=estimate.estimate + z * estimate.se,
        alpha=alpha,
        alternative=alternative,
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

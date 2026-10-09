"""Post-hoc calibration of judge scores (Platt, isotonic, temperature, histogram binning) and the
calibration metrics used to compare them (ECE, MCE, Brier, reliability bins)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol

import numpy as np
from pydantic import BaseModel
from scipy.optimize import minimize_scalar

Method = Literal["none", "platt", "isotonic", "temperature", "histogram"]
METHODS: tuple[Method, ...] = ("none", "platt", "isotonic", "temperature", "histogram")
_EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(q) - np.log1p(-q)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=float)))


class CalibratorParams(BaseModel):
    method: Method
    a: float = 1.0
    b: float = 0.0
    temperature: float = 1.0
    x: list[float] = []
    y: list[float] = []
    edges: list[float] = []
    values: list[float] = []


class Calibrator(Protocol):
    method: Method

    def fit(
        self, scores: Sequence[float] | np.ndarray, labels: Sequence[int] | np.ndarray
    ) -> Calibrator: ...

    def predict(self, scores: Sequence[float] | np.ndarray) -> np.ndarray: ...

    def params(self) -> CalibratorParams: ...


class IdentityCalibrator:
    method: Method = "none"

    def fit(
        self, scores: Sequence[float] | np.ndarray, labels: Sequence[int] | np.ndarray
    ) -> IdentityCalibrator:
        return self

    def predict(self, scores: Sequence[float] | np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)

    def params(self) -> CalibratorParams:
        return CalibratorParams(method="none")


class PlattCalibrator:
    """p = sigmoid(a * logit(s) + b), fitted by Newton's method on Platt's smoothed targets.

    The targets (N+ + 1)/(N+ + 2) and 1/(N- + 2) keep the fit finite when the scores separate
    the classes perfectly, which happens on small calibration sets.
    """

    method: Method = "platt"

    def __init__(self) -> None:
        self.a, self.b = 1.0, 0.0

    def fit(
        self, scores: Sequence[float] | np.ndarray, labels: Sequence[int] | np.ndarray
    ) -> PlattCalibrator:
        x = logit(np.asarray(scores, dtype=float))
        y = np.asarray(labels, dtype=float)
        n_pos, n_neg = y.sum(), y.size - y.sum()
        t = np.where(y == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))
        theta = np.array([0.0, float(np.log((n_pos + 1) / (n_neg + 1)))])
        design = np.column_stack([x, np.ones_like(x)])

        def nll(params: np.ndarray) -> float:
            z = design @ params
            return float(np.sum(np.logaddexp(0.0, z) - t * z))

        current = nll(theta)
        for _ in range(100):
            p = sigmoid(design @ theta)
            grad = design.T @ (p - t)
            if np.max(np.abs(grad)) < 1e-8:
                break
            hess = design.T @ (design * (p * (1 - p))[:, None]) + 1e-12 * np.eye(2)
            step = np.linalg.solve(hess, grad)
            rate = 1.0
            while rate > 1e-10:
                candidate = theta - rate * step
                value = nll(candidate)
                if value <= current - 1e-4 * rate * float(grad @ step):
                    break
                rate /= 2
            else:
                break
            theta, current = candidate, value
        self.a, self.b = float(theta[0]), float(theta[1])
        return self

    def predict(self, scores: Sequence[float] | np.ndarray) -> np.ndarray:
        return sigmoid(self.a * logit(np.asarray(scores, dtype=float)) + self.b)

    def params(self) -> CalibratorParams:
        return CalibratorParams(method="platt", a=self.a, b=self.b)


class TemperatureCalibrator:
    """p = sigmoid(logit(s) / T). One parameter: it rescales confidence but cannot move the
    decision boundary, so it cannot fix a judge whose errors are mostly a shifted base rate."""

    method: Method = "temperature"

    def __init__(self) -> None:
        self.temperature = 1.0

    def fit(
        self, scores: Sequence[float] | np.ndarray, labels: Sequence[int] | np.ndarray
    ) -> TemperatureCalibrator:
        z = logit(np.asarray(scores, dtype=float))
        y = np.asarray(labels, dtype=float)

        def nll(log_t: float) -> float:
            u = z / np.exp(log_t)
            return float(np.mean(np.logaddexp(0.0, u) - y * u))

        result = minimize_scalar(nll, bounds=(-5.0, 5.0), method="bounded")
        self.temperature = float(np.exp(result.x))
        return self

    def predict(self, scores: Sequence[float] | np.ndarray) -> np.ndarray:
        return sigmoid(logit(np.asarray(scores, dtype=float)) / self.temperature)

    def params(self) -> CalibratorParams:
        return CalibratorParams(method="temperature", temperature=self.temperature)


def pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: the non-decreasing sequence closest to ``y`` in weighted L2."""
    values: list[float] = []
    weights: list[float] = []
    counts: list[int] = []
    for yi, wi in zip(y.tolist(), w.tolist(), strict=True):
        values.append(yi)
        weights.append(wi)
        counts.append(1)
        while len(values) > 1 and values[-2] > values[-1]:
            total = weights[-2] + weights[-1]
            merged = (values[-2] * weights[-2] + values[-1] * weights[-1]) / total
            count = counts[-2] + counts[-1]
            del values[-1], weights[-1], counts[-1]
            values[-1], weights[-1], counts[-1] = merged, total, count
    return np.repeat(values, counts)


class IsotonicCalibrator:
    """Isotonic regression of labels on scores, predicting by linear interpolation between the
    ends of each constant block and clipping outside the fitted range."""

    method: Method = "isotonic"

    def __init__(self) -> None:
        self.x = np.array([0.0, 1.0])
        self.y = np.array([0.0, 1.0])

    def fit(
        self, scores: Sequence[float] | np.ndarray, labels: Sequence[int] | np.ndarray
    ) -> IsotonicCalibrator:
        s = np.asarray(scores, dtype=float)
        lab = np.asarray(labels, dtype=float)
        uniq, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
        means = np.bincount(inverse, weights=lab) / counts
        fitted = pava(means, counts.astype(float))
        # Only the first and last point of each constant run matter for interpolation.
        keep = np.ones(fitted.size, dtype=bool)
        if fitted.size > 2:
            keep[1:-1] = (fitted[1:-1] != fitted[:-2]) | (fitted[1:-1] != fitted[2:])
        self.x, self.y = uniq[keep], fitted[keep]
        return self

    def predict(self, scores: Sequence[float] | np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(scores, dtype=float), self.x, self.y)

    def params(self) -> CalibratorParams:
        return CalibratorParams(method="isotonic", x=self.x.tolist(), y=self.y.tolist())


class HistogramCalibrator:
    """Equal-mass histogram binning: each score maps to the label frequency of its bin."""

    method: Method = "histogram"

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins
        self.edges = np.array([])
        self.values = np.array([0.5])

    def fit(
        self, scores: Sequence[float] | np.ndarray, labels: Sequence[int] | np.ndarray
    ) -> HistogramCalibrator:
        s = np.asarray(scores, dtype=float)
        lab = np.asarray(labels, dtype=float)
        inner = np.quantile(s, np.linspace(0, 1, self.n_bins + 1)[1:-1])
        self.edges = np.unique(inner)
        ids = np.searchsorted(self.edges, s, side="right")
        sums = np.bincount(ids, weights=lab, minlength=self.edges.size + 1)
        counts = np.bincount(ids, minlength=self.edges.size + 1)
        overall = lab.mean() if lab.size else 0.5
        self.values = np.where(counts > 0, sums / np.maximum(counts, 1), overall)
        return self

    def predict(self, scores: Sequence[float] | np.ndarray) -> np.ndarray:
        ids = np.searchsorted(self.edges, np.asarray(scores, dtype=float), side="right")
        return self.values[ids]

    def params(self) -> CalibratorParams:
        return CalibratorParams(
            method="histogram", edges=self.edges.tolist(), values=self.values.tolist()
        )


def make_calibrator(method: Method) -> Calibrator:
    factories: dict[str, type] = {
        "none": IdentityCalibrator,
        "platt": PlattCalibrator,
        "isotonic": IsotonicCalibrator,
        "temperature": TemperatureCalibrator,
        "histogram": HistogramCalibrator,
    }
    if method not in factories:
        raise ValueError(f"unknown calibration method {method!r}; choose from {METHODS}")
    return factories[method]()


def calibrator_from_params(params: CalibratorParams) -> Calibrator:
    if params.method == "platt":
        platt = PlattCalibrator()
        platt.a, platt.b = params.a, params.b
        return platt
    if params.method == "temperature":
        temp = TemperatureCalibrator()
        temp.temperature = params.temperature
        return temp
    if params.method == "isotonic":
        iso = IsotonicCalibrator()
        iso.x, iso.y = np.asarray(params.x), np.asarray(params.y)
        return iso
    if params.method == "histogram":
        hist = HistogramCalibrator(n_bins=len(params.values))
        hist.edges, hist.values = np.asarray(params.edges), np.asarray(params.values)
        return hist
    return IdentityCalibrator()


class ReliabilityBin(BaseModel):
    lower: float
    upper: float
    count: int
    mean_predicted: float
    fraction_positive: float


def reliability_bins(
    probs: Sequence[float] | np.ndarray, labels: Sequence[int] | np.ndarray, n_bins: int = 15
) -> list[ReliabilityBin]:
    """Equal-width bins over [0, 1]; a value on an inner edge falls into the lower bin."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.searchsorted(edges[1:-1], p)
    out = []
    for b in range(n_bins):
        mask = ids == b
        if mask.any():
            out.append(
                ReliabilityBin(
                    lower=float(edges[b]),
                    upper=float(edges[b + 1]),
                    count=int(mask.sum()),
                    mean_predicted=float(p[mask].mean()),
                    fraction_positive=float(y[mask].mean()),
                )
            )
    return out


def expected_calibration_error(
    probs: Sequence[float] | np.ndarray, labels: Sequence[int] | np.ndarray, n_bins: int = 15
) -> float:
    bins = reliability_bins(probs, labels, n_bins)
    total = sum(b.count for b in bins)
    return float(sum(b.count / total * abs(b.fraction_positive - b.mean_predicted) for b in bins))


def maximum_calibration_error(
    probs: Sequence[float] | np.ndarray, labels: Sequence[int] | np.ndarray, n_bins: int = 15
) -> float:
    bins = reliability_bins(probs, labels, n_bins)
    return float(max(abs(b.fraction_positive - b.mean_predicted) for b in bins))


def brier_score(probs: Sequence[float] | np.ndarray, labels: Sequence[int] | np.ndarray) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    return float(np.mean((p - y) ** 2))

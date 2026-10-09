"""SummaC-style scoring: an NLI matrix of response units against source chunks, aggregated either
zero-shot (SummaC-ZS) or with a learned histogram convolution (SummaC-Conv)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel
from scipy.optimize import minimize

from faithgate.nli import EntailmentModel
from faithgate.text import Segment, chunk_source, split_units

Aggregation = Literal["mean", "min"]
ConvMode = Literal["mean", "max"]


@dataclass(frozen=True)
class ChunkConfig:
    """How the response and the source are cut before NLI."""

    unit: str = "sentence"
    chunk_words: int = 100
    overlap: int = 1

    def key(self) -> str:
        return f"{self.unit}-w{self.chunk_words}-o{self.overlap}"


@dataclass
class NLIMatrix:
    """``probs[j, i]`` is P(chunk i entails unit j)."""

    units: list[Segment]
    chunks: list[Segment]
    probs: np.ndarray

    def support(self) -> np.ndarray:
        """Best entailment probability of each unit over all chunks (the SummaC-ZS row max)."""
        if self.probs.size == 0:
            return np.zeros(len(self.units))
        return self.probs.max(axis=1)

    def evidence(self) -> list[Segment | None]:
        if self.probs.size == 0:
            return [None] * len(self.units)
        return [self.chunks[i] for i in self.probs.argmax(axis=1)]


def layout(source: str, response: str, config: ChunkConfig) -> tuple[list[Segment], list[Segment]]:
    """The response units (rows) and source chunks (columns) of the NLI matrix."""
    return split_units(response, config.unit), chunk_source(
        source, config.chunk_words, config.overlap
    )


def build_matrices(
    model: EntailmentModel, pairs: Sequence[tuple[str, str]], config: ChunkConfig
) -> list[NLIMatrix]:
    """NLI matrices for many (source, response) pairs, scored in one batched call."""
    layouts = [layout(source, response, config) for source, response in pairs]
    flat = [
        (chunk.text, unit.text) for units, chunks in layouts for unit in units for chunk in chunks
    ]
    probs = model.entail_probs(flat)
    matrices: list[NLIMatrix] = []
    offset = 0
    for units, chunks in layouts:
        size = len(units) * len(chunks)
        block = probs[offset : offset + size].reshape(len(units), len(chunks))
        matrices.append(NLIMatrix(units, chunks, block))
        offset += size
    return matrices


def zs_hallucination(matrix: NLIMatrix, aggregation: Aggregation = "mean") -> float:
    """1 - SummaC-ZS consistency.

    With a binary NLI head the SummaC term P(entail) - P(contradict) becomes 2 P(entail) - 1, a
    monotone map, so the row max is taken on P(entail) directly. ``mean`` is SummaC's original
    aggregation; ``min`` treats one unsupported unit as enough to call the response hallucinated.
    A response with no checkable units asserts nothing and scores 0.
    """
    support = matrix.support()
    if support.size == 0:
        return 0.0
    agg = support.mean() if aggregation == "mean" else support.min()
    return float(1.0 - agg)


class ConvParams(BaseModel):
    n_bins: int
    mode: ConvMode
    l2: float
    weights: list[float]
    bias: float


def _logsumexp_groups(values: np.ndarray, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-group log-sum-exp of a flat array whose groups begin at ``starts``; also the softmax."""
    peak = np.maximum.reduceat(values, starts)
    sizes = np.diff(np.append(starts, values.size))
    shifted = np.exp(values - np.repeat(peak, sizes))
    totals = np.add.reduceat(shifted, starts)
    return peak + np.log(totals), shifted / np.repeat(totals, sizes)


class ConvAggregator:
    """SummaC-Conv: each unit's row of entailment probabilities becomes a histogram, a learned
    linear filter over the bins (a 1-D convolution whose kernel spans all bins) maps it to a unit
    logit, and unit logits are pooled into a response logit.

    ``mode='mean'`` pools by averaging, as in SummaC. ``mode='max'`` pools with log-sum-exp, a
    smooth maximum, so one badly supported unit can carry the response.
    """

    def __init__(self, n_bins: int = 20, mode: ConvMode = "mean", l2: float = 1e-3) -> None:
        self.n_bins = n_bins
        self.mode: ConvMode = mode
        self.l2 = l2
        self.weights = np.zeros(n_bins)
        self.bias = 0.0

    def features(self, matrix: NLIMatrix) -> np.ndarray:
        if matrix.probs.size == 0:
            return np.zeros((len(matrix.units), self.n_bins))
        bins = np.minimum((matrix.probs * self.n_bins).astype(int), self.n_bins - 1)
        # Raw counts, as in SummaC: "one chunk entails this unit" stays the same signal however
        # many chunks the source has.
        hist = np.zeros((matrix.probs.shape[0], self.n_bins))
        for b in range(self.n_bins):
            hist[:, b] = (bins == b).sum(axis=1)
        return hist

    def stack(self, matrices: Sequence[NLIMatrix]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unit features of all responses with at least one unit, the row where each response
        starts, and the indices of those responses."""
        kept = [i for i, m in enumerate(matrices) if len(m.units) > 0]
        feats = [self.features(matrices[i]) for i in kept]
        sizes = np.array([f.shape[0] for f in feats], dtype=int)
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(int)
        return np.vstack(feats), starts, np.array(kept, dtype=int)

    def _response_logits(self, unit_logits: np.ndarray, starts: np.ndarray) -> np.ndarray:
        if self.mode == "mean":
            sizes = np.diff(np.append(starts, unit_logits.size))
            return np.add.reduceat(unit_logits, starts) / sizes
        return _logsumexp_groups(unit_logits, starts)[0]

    def objective(
        self, theta: np.ndarray, feats: np.ndarray, starts: np.ndarray, y: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Mean binary cross-entropy plus L2 on the filter, and its gradient in (weights, bias)."""
        w, b = theta[:-1], theta[-1]
        sizes = np.diff(np.append(starts, feats.shape[0]))
        group_of_unit = np.repeat(np.arange(starts.size), sizes)
        unit_logits = feats @ w + b
        if self.mode == "mean":
            z = np.add.reduceat(unit_logits, starts) / sizes
            unit_weight = 1.0 / sizes[group_of_unit]
        else:
            z, unit_weight = _logsumexp_groups(unit_logits, starts)
        nll = float(np.mean(np.logaddexp(0.0, z) - y * z))
        p = 1.0 / (1.0 + np.exp(-z))
        d_unit = ((p - y) / y.size)[group_of_unit] * unit_weight
        grad = np.append(feats.T @ d_unit + 2 * self.l2 * w, d_unit.sum())
        return nll + self.l2 * float(w @ w), grad

    def fit(self, matrices: Sequence[NLIMatrix], labels: Sequence[int]) -> ConvAggregator:
        feats, starts, kept = self.stack(matrices)
        y = np.asarray(labels, dtype=float)[kept]
        result = minimize(
            self.objective,
            np.zeros(self.n_bins + 1),
            args=(feats, starts, y),
            jac=True,
            method="L-BFGS-B",
        )
        self.weights, self.bias = result.x[:-1], float(result.x[-1])
        return self

    def unit_probs(self, matrix: NLIMatrix) -> np.ndarray:
        logits = self.features(matrix) @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-logits))

    def predict(self, matrices: Sequence[NLIMatrix]) -> np.ndarray:
        """Hallucination probability per response (0 for responses with no checkable unit)."""
        out = np.zeros(len(matrices))
        if not any(len(m.units) for m in matrices):
            return out
        feats, starts, kept = self.stack(matrices)
        z = self._response_logits(feats @ self.weights + self.bias, starts)
        out[kept] = 1.0 / (1.0 + np.exp(-z))
        return out

    def to_params(self) -> ConvParams:
        return ConvParams(
            n_bins=self.n_bins,
            mode=self.mode,
            l2=self.l2,
            weights=[float(w) for w in self.weights],
            bias=self.bias,
        )

    @classmethod
    def from_params(cls, params: ConvParams) -> ConvAggregator:
        conv = cls(params.n_bins, params.mode, params.l2)
        conv.weights = np.asarray(params.weights, dtype=float)
        conv.bias = params.bias
        return conv

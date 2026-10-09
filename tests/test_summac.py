from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import check_grad

from faithgate.metrics import auroc
from faithgate.summac import (
    ChunkConfig,
    ConvAggregator,
    ConvParams,
    NLIMatrix,
    build_matrices,
    zs_hallucination,
)
from faithgate.text import Segment
from tests.fakes import LexicalEntailment

SOURCE = "The museum opens at ten. Tickets cost fifteen dollars. Children enter free on Sundays."


def test_matrix_shapes_and_single_batched_call(lexical_nli: LexicalEntailment) -> None:
    pairs = [
        (SOURCE, "The museum opens at ten. Tickets cost fifteen dollars."),
        (SOURCE, ""),
        ("", "Hello there."),
    ]
    mats = build_matrices(lexical_nli, pairs, ChunkConfig(chunk_words=5, overlap=0))
    assert lexical_nli.calls == 1
    assert mats[0].probs.shape == (2, 3)
    assert mats[1].probs.shape == (0, 3)
    assert mats[2].probs.shape == (1, 0)


def test_zero_shot_aggregations(lexical_nli: LexicalEntailment) -> None:
    supported = "The museum opens at ten."
    invented = "The aquarium closes at midnight."
    [mat] = build_matrices(
        lexical_nli, [(SOURCE, f"{supported} {invented}")], ChunkConfig(chunk_words=6)
    )
    support = mat.support()
    assert support[0] == pytest.approx(1.0)
    assert support[1] < 0.2
    assert zs_hallucination(mat, "min") == pytest.approx(1 - support.min())
    assert zs_hallucination(mat, "mean") == pytest.approx(1 - support.mean())
    assert zs_hallucination(NLIMatrix([], [], np.zeros((0, 0)))) == 0.0
    assert mat.evidence()[0] is not None


def _synthetic(n: int, seed: int) -> tuple[list[NLIMatrix], np.ndarray]:
    """Faithful responses: every unit has a high-entailment chunk. Hallucinated ones: one unit
    has none. Unit and chunk counts vary per response."""
    rng = np.random.default_rng(seed)
    mats, labels = [], []
    for _ in range(n):
        units, chunks = int(rng.integers(1, 7)), int(rng.integers(2, 9))
        probs = rng.uniform(0, 0.3, (units, chunks))
        probs[np.arange(units), rng.integers(0, chunks, units)] = rng.uniform(0.7, 1.0, units)
        y = int(rng.uniform() < 0.4)
        if y:
            probs[rng.integers(0, units)] = rng.uniform(0, 0.3, chunks)
        segs = [Segment(i, i + 1, "u") for i in range(units)]
        mats.append(NLIMatrix(segs, [Segment(0, 0, "")] * chunks, probs))
        labels.append(y)
    return mats, np.array(labels)


@pytest.mark.parametrize("mode", ["mean", "max"])
def test_conv_learns_unsupported_unit_pattern(mode: str) -> None:
    train_m, train_y = _synthetic(400, 1)
    test_m, test_y = _synthetic(400, 2)
    conv = ConvAggregator(n_bins=10, mode=mode).fit(train_m, train_y)  # type: ignore[arg-type]
    pred = conv.predict(test_m)
    assert pred.min() >= 0
    assert pred.max() <= 1
    assert auroc(test_y, pred) > 0.95


@pytest.mark.parametrize("mode", ["mean", "max"])
def test_conv_gradient_matches_finite_differences(mode: str) -> None:
    mats, y = _synthetic(60, 3)
    conv = ConvAggregator(n_bins=6, mode=mode)  # type: ignore[arg-type]
    feats, starts, kept = conv.stack(mats)
    labels = y[kept].astype(float)
    theta0 = np.random.default_rng(0).normal(0, 1, 7)
    error = check_grad(
        lambda t: conv.objective(t, feats, starts, labels)[0],
        lambda t: conv.objective(t, feats, starts, labels)[1],
        theta0,
    )
    assert error < 1e-5


def test_conv_params_round_trip() -> None:
    mats, y = _synthetic(80, 4)
    conv = ConvAggregator(n_bins=8, mode="max").fit(mats, y)
    restored = ConvAggregator.from_params(
        ConvParams.model_validate_json(conv.to_params().model_dump_json())
    )
    np.testing.assert_allclose(restored.predict(mats), conv.predict(mats))
    empty = NLIMatrix([], [], np.zeros((0, 0)))
    assert conv.predict([empty]).tolist() == [0.0]

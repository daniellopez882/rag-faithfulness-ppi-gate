from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from faithgate.metrics import (
    auroc,
    best_f1_threshold,
    cohen_kappa,
    krippendorff_alpha,
    matched_auroc,
    precision_recall_f1,
)

labels_and_preds = st.integers(min_value=1, max_value=80).flatmap(
    lambda n: st.tuples(
        st.lists(st.integers(0, 1), min_size=n, max_size=n),
        st.lists(st.integers(0, 1), min_size=n, max_size=n),
    )
)

labels_and_tied_scores = st.integers(min_value=2, max_value=80).flatmap(
    lambda n: st.tuples(
        st.lists(st.integers(0, 1), min_size=n, max_size=n),
        st.lists(st.sampled_from([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]), min_size=n, max_size=n),
    )
)


@given(labels_and_preds)
@settings(max_examples=200, deadline=None)
def test_prf_matches_sklearn(data: tuple[list[int], list[int]]) -> None:
    y, p = data
    ours = precision_recall_f1(y, p)
    assert ours.precision == pytest.approx(precision_score(y, p, zero_division=0))
    assert ours.recall == pytest.approx(recall_score(y, p, zero_division=0))
    assert ours.f1 == pytest.approx(f1_score(y, p, zero_division=0))
    assert ours.tp + ours.fp + ours.fn + ours.tn == len(y)


@given(labels_and_tied_scores)
@settings(max_examples=200, deadline=None)
def test_auroc_matches_sklearn_with_ties(data: tuple[list[int], list[float]]) -> None:
    y, s = data
    if len(set(y)) < 2:
        with pytest.raises(ValueError, match="both classes"):
            auroc(y, s)
        return
    assert auroc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert auroc(y, [-v for v in s]) == pytest.approx(1 - auroc(y, s))


@pytest.mark.filterwarnings("ignore::UserWarning")  # scikit-learn warns on single-label inputs
@given(labels_and_preds)
@settings(max_examples=200, deadline=None)
def test_kappa_matches_sklearn(data: tuple[list[int], list[int]]) -> None:
    a, b = data
    expected = cohen_kappa_score(a, b)
    if np.isnan(expected):
        return
    assert cohen_kappa(a, b) == pytest.approx(expected)


# Reliability data from Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability":
# four observers, twelve units, None for a missing value.
KRIPPENDORFF_2011: list[list[float | None]] = [
    [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None],
    [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3],
    [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None],
    [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None],
]


def test_krippendorff_alpha_reproduces_published_values() -> None:
    assert krippendorff_alpha(KRIPPENDORFF_2011, "nominal") == pytest.approx(0.743, abs=5e-4)
    assert krippendorff_alpha(KRIPPENDORFF_2011, "interval") == pytest.approx(0.849, abs=5e-4)


def test_krippendorff_alpha_extremes() -> None:
    assert krippendorff_alpha([[0, 1, 1, 0], [0, 1, 1, 0]]) == pytest.approx(1.0)
    # Systematic disagreement is worse than chance.
    assert krippendorff_alpha([[0, 1, 0, 1], [1, 0, 1, 0]]) < 0


@given(labels_and_tied_scores)
@settings(max_examples=150, deadline=None)
def test_best_f1_threshold_is_optimal(data: tuple[list[int], list[float]]) -> None:
    y, s = np.array(data[0]), np.array(data[1])
    if y.sum() == 0:
        return
    chosen = best_f1_threshold(y, s)
    best = max(precision_recall_f1(y, s >= t).f1 for t in np.unique(s))
    assert precision_recall_f1(y, s >= chosen).f1 == pytest.approx(best)


@given(
    st.lists(
        st.tuples(st.integers(0, 1), st.sampled_from([0.0, 0.3, 0.5, 0.9]), st.sampled_from("ab")),
        min_size=2,
        max_size=40,
    )
)
@settings(max_examples=200, deadline=None)
def test_matched_auroc_counts_only_pairs_within_a_stratum(
    rows: list[tuple[int, float, str]],
) -> None:
    pairs = [(p[1], n[1]) for p in rows for n in rows if p[0] == 1 and n[0] == 0 and p[2] == n[2]]
    if not pairs:
        with pytest.raises(ValueError, match="no stratum"):
            matched_auroc(*zip(*rows, strict=True))
        return
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a, b in pairs)
    value, count = matched_auroc(*zip(*rows, strict=True))
    assert count == len(pairs)
    assert value == pytest.approx(wins / len(pairs))


def test_a_score_that_only_tracks_the_stratum_is_at_chance() -> None:
    rng = np.random.default_rng(0)
    length = rng.integers(1, 6, size=500)
    y = (rng.uniform(size=500) < length / 6).astype(int)
    assert auroc(y, length) > 0.6
    assert matched_auroc(y, length, length.astype(str))[0] == pytest.approx(0.5)

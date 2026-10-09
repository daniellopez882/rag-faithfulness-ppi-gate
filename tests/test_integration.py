"""Checks against the real RAGTruth files and NLI model; skipped unless they were fetched."""

from __future__ import annotations

import collections

import pytest

from faithgate.fetch import locate_nli_model
from faithgate.nli import OnnxNLI
from faithgate.paths import ragtruth_dir

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def nli() -> OnnxNLI:
    folder = locate_nli_model()
    if folder is None:
        pytest.skip("NLI model not fetched (faithgate fetch model)")
    return OnnxNLI(folder)


def test_ragtruth_split_sizes() -> None:
    if not (ragtruth_dir() / "response.jsonl").exists():
        pytest.skip("RAGTruth not fetched (faithgate fetch ragtruth)")
    from faithgate.ragtruth import load_ragtruth

    test = load_ragtruth("test")
    assert len(test) == 2700
    assert collections.Counter(r.task_type for r in test) == {
        "QA": 900,
        "Summary": 900,
        "Data2txt": 900,
    }
    assert sum(r.label or 0 for r in test) == 943
    for record in test:
        for start, end in record.spans:
            assert 0 <= start < end <= len(record.response)


def test_nli_model_separates_entailment_from_contradiction(nli: OnnxNLI) -> None:
    probs = nli.entail_probs(
        [
            ("The cafe opens at 9 am and serves breakfast.", "The cafe serves breakfast."),
            ("The cafe opens at 9 am and serves breakfast.", "The cafe serves dinner only."),
            ("Anne Frank died of typhus at the age of 15.", "Anne Frank died at 15."),
            ("Anne Frank died of typhus at the age of 15.", "Anne Frank died at 40 of cholera."),
        ]
    )
    assert probs[0] > 0.8
    assert probs[2] > 0.5
    assert probs[1] < 0.2
    assert probs[3] < 0.2


def test_nli_batching_does_not_change_results(nli: OnnxNLI) -> None:
    pairs = [
        ("A short premise.", "A short premise."),
        ("A much longer premise " * 20, "Something else."),
    ] * 5
    one_by_one = [float(nli.entail_probs([p])[0]) for p in pairs]
    batched = nli.entail_probs(pairs).tolist()
    assert batched == pytest.approx(one_by_one, abs=1e-4)

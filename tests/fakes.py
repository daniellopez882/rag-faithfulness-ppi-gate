"""Deterministic stand-ins used across the test suite."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from faithgate.text import vocabulary


class LexicalEntailment:
    """Deterministic stand-in for the NLI model: P(entail) grows with the share of the
    hypothesis' content words found in the premise."""

    name = "lexical-fake"

    def __init__(self) -> None:
        self.calls = 0

    def entail_probs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        self.calls += 1
        out = []
        for premise, hypothesis in pairs:
            hyp = vocabulary(hypothesis)
            share = len(hyp & vocabulary(premise)) / len(hyp) if hyp else 1.0
            out.append(share**3)
        return np.asarray(out, dtype=float)

"""Span localisation from unit-level judge scores, and character-level span P/R/F1."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from collections.abc import Set as AbstractSet

from pydantic import BaseModel

from faithgate.records import Span, UnitScore
from faithgate.text import content_tokens


def merge_spans(spans: Iterable[Span]) -> list[Span]:
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def unsupported_runs(text: str, start: int, end: int, vocab: AbstractSet[str]) -> list[Span]:
    """Runs of consecutive content words in ``text[start:end]`` that never occur in the source.

    Stopwords between two unsupported words do not break a run, so "free valet parking" stays
    one span when all three content words are missing from the source.
    """
    runs: list[list[int]] = []
    open_run = False
    for tok_start, tok_end, form in content_tokens(text, start, end):
        if form in vocab:
            open_run = False
            continue
        if open_run:
            runs[-1][1] = tok_end
        else:
            runs.append([tok_start, tok_end])
            open_run = True
    return [(a, b) for a, b in runs]


def localise(
    response: str,
    units: Sequence[UnitScore],
    threshold: float,
    source_vocab: AbstractSet[str] | None = None,
) -> list[Span]:
    """Spans for the units scored at or above ``threshold``.

    Without ``source_vocab`` a flagged unit is returned whole. With it, a flagged unit is narrowed
    to its words that the source never uses; a unit whose words all appear in the source (a
    contradiction built from source words, say) is still returned whole.
    """
    spans: list[Span] = []
    for unit in units:
        if unit.score < threshold:
            continue
        if source_vocab is not None:
            runs = unsupported_runs(response, unit.start, unit.end, source_vocab)
            spans.extend(runs or [(unit.start, unit.end)])
        else:
            spans.append((unit.start, unit.end))
    return merge_spans(spans)


def _length(spans: Sequence[Span]) -> int:
    return sum(b - a for a, b in spans)


def overlap_chars(pred: Sequence[Span], gold: Sequence[Span]) -> int:
    p, g = merge_spans(pred), merge_spans(gold)
    total, i, j = 0, 0, 0
    while i < len(p) and j < len(g):
        lo, hi = max(p[i][0], g[j][0]), min(p[i][1], g[j][1])
        total += max(0, hi - lo)
        if p[i][1] < g[j][1]:
            i += 1
        else:
            j += 1
    return total


class SpanScores(BaseModel):
    precision: float
    recall: float
    f1: float
    predicted_chars: int
    gold_chars: int
    overlap_chars: int


def span_scores(pairs: Iterable[tuple[Sequence[Span], Sequence[Span]]]) -> SpanScores:
    """Micro-averaged character-level P/R/F1 over (predicted, gold) span lists."""
    pred_total = gold_total = hit = 0
    for pred, gold in pairs:
        pred_total += _length(merge_spans(pred))
        gold_total += _length(merge_spans(gold))
        hit += overlap_chars(pred, gold)
    precision = hit / pred_total if pred_total else 0.0
    recall = hit / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return SpanScores(
        precision=precision,
        recall=recall,
        f1=f1,
        predicted_chars=pred_total,
        gold_chars=gold_total,
        overlap_chars=hit,
    )

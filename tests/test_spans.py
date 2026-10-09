from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from faithgate.records import UnitScore
from faithgate.spans import localise, merge_spans, overlap_chars, span_scores, unsupported_runs
from faithgate.text import vocabulary

spans_strategy = st.lists(
    st.tuples(st.integers(0, 50), st.integers(0, 50)).map(lambda t: (min(t), max(t))), max_size=8
)


@given(spans_strategy, spans_strategy)
@settings(max_examples=200, deadline=None)
def test_overlap_matches_character_sets(
    pred: list[tuple[int, int]], gold: list[tuple[int, int]]
) -> None:
    def chars(spans: list[tuple[int, int]]) -> set[int]:
        return {i for a, b in spans for i in range(a, b)}

    assert overlap_chars(pred, gold) == len(chars(pred) & chars(gold))
    merged = merge_spans(pred)
    assert chars(merged) == chars(pred)
    assert all(merged[i][1] < merged[i + 1][0] for i in range(len(merged) - 1))


def test_span_scores_micro_average() -> None:
    scores = span_scores([([(0, 10)], [(5, 15)]), ([], [(0, 4)]), ([(0, 2)], [])])
    assert scores.overlap_chars == 5
    assert scores.predicted_chars == 12
    assert scores.gold_chars == 14
    assert scores.precision == 5 / 12
    assert scores.recall == 5 / 14


def test_unsupported_runs_bridge_stopwords() -> None:
    text = "It has free valet parking and a pool."
    vocab = vocabulary("It has a pool.")
    assert [text[a:b] for a, b in unsupported_runs(text, 0, len(text), vocab)] == [
        "free valet parking"
    ]


def test_localise_whole_unit_when_every_word_is_in_the_source() -> None:
    text = "Tickets cost ten dollars. The museum opens daily."
    units = [UnitScore(start=0, end=25, score=0.9), UnitScore(start=26, end=len(text), score=0.1)]
    vocab = vocabulary("The museum tickets cost fifteen dollars, not ten.")
    assert localise(text, units, 0.5, vocab) == [(0, 25)]
    assert localise(text, units, 0.5) == [(0, 25)]
    assert localise(text, units, 0.95) == []

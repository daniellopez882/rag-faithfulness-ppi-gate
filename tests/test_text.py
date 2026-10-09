from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from faithgate.text import (
    chunk_source,
    is_checkable,
    normalise_token,
    split_claims,
    split_sentences,
    split_units,
)

PIECES = st.sampled_from(
    [
        "The cat sat.",
        "Dr. Smith left",
        "e.g. this",
        "It costs 3.5 dollars",
        "\n",
        " ",
        "1. First item",
        "Why?",
        "Wow!",
        "U.S. troops",
        "J. K. Rowling wrote it.",
        "and it closes at 5",
        ", but then",
        ";",
        '"Quoted."',
        "(Note.)",
        "- bullet point",
        "lower case. next",
    ]
)


@given(st.lists(PIECES, max_size=12).map(" ".join))
@settings(max_examples=200, deadline=None)
def test_segments_are_exact_ordered_slices(text: str) -> None:
    for splitter in (split_sentences, split_claims):
        segments = splitter(text)
        last_end = 0
        for seg in segments:
            assert text[seg.start : seg.end] == seg.text
            assert seg.text == seg.text.strip()
            assert seg.start >= last_end
            last_end = seg.end


def test_abbreviations_initials_and_decimals_do_not_split() -> None:
    text = (
        "Dr. Smith met J. K. Rowling in the U.S. on Monday. The bill was 3.50 dollars, e.g. cheap."
    )
    assert [s.text for s in split_sentences(text)] == [
        "Dr. Smith met J. K. Rowling in the U.S. on Monday.",
        "The bill was 3.50 dollars, e.g. cheap.",
    ]


def test_list_markers_stay_with_their_item() -> None:
    text = "Steps:\n1. Preheat the oven.\n2. Season the duck."
    assert [s.text for s in split_sentences(text)] == [
        "Steps:",
        "Preheat the oven.",
        "Season the duck.",
    ]


def test_claims_split_clauses_but_not_lists() -> None:
    text = (
        "The shop sells bread, cakes, and pies. "
        "It opens at nine every day, but it closes early on Sundays."
    )
    claims = [c.text for c in split_claims(text)]
    assert claims == [
        "The shop sells bread, cakes, and pies.",
        "It opens at nine every day",
        "but it closes early on Sundays.",
    ]


def test_units_drop_questions_lead_ins_and_refusals() -> None:
    text = (
        "Here is the answer:\nWhat is it? It is blue. Unable to answer based on given passages. 1."
    )
    assert [u.text for u in split_units(text, "sentence")] == ["It is blue."]
    assert not is_checkable(split_sentences("Is it?")[0])


def test_chunks_respect_budget_and_cover_every_sentence() -> None:
    text = " ".join(f"Sentence number {i} has five words." for i in range(1, 11))
    chunks = chunk_source(text, max_words=12, overlap=1)
    for chunk in chunks:
        assert text[chunk.start : chunk.end] == chunk.text
        assert len(chunk.text.split()) <= 12
    covered = " ".join(c.text for c in chunks)
    for i in range(1, 11):
        assert f"number {i} " in covered
    # Overlap of one sentence: consecutive chunks share a sentence.
    assert chunks[0].end > chunks[1].start


def test_long_sentence_is_windowed() -> None:
    text = " ".join(f"w{i}" for i in range(30)) + "."
    chunks = chunk_source(text, max_words=10, overlap=0)
    assert len(chunks) == 3
    assert chunks[0].text.startswith("w0")
    assert chunks[-1].text.endswith("w29.")


def test_normalise_token() -> None:
    assert normalise_token("1,000") == "1000"
    assert normalise_token("Restaurants") == normalise_token("restaurant")
    assert normalise_token("Anne's") == "anne"
    assert normalise_token("2019") == "2019"

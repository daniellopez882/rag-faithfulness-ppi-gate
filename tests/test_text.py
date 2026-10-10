from __future__ import annotations

import re

import pytest
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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The answer is no. It was closed on Monday.", 2),
        ("He moved to the U.S. He liked it.", 2),
        ("We sat in the sun. Then we left.", 2),
        ("Born in Washington, D.C. She studied law.", 2),
        ("The U.S. Army said so.", 1),
        ("See No. 5 and Fig. 2 for details.", 1),
        ("Smith et al. (2019) found it.", 1),
        ("It opens at 5 p.m. The bar closes late.", 2),
    ],
)
def test_word_like_abbreviations_and_acronyms_end_sentences_when_they_should(
    text: str, expected: int
) -> None:
    assert len(split_sentences(text)) == expected


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
    assert normalise_token("Anne's") == normalise_token("Anne")
    assert normalise_token("2019") == "2019"
    assert normalise_token("U.S") == "u.s"


@pytest.mark.parametrize(
    "forms",
    [
        ("run", "runs", "running"),
        ("address", "addresses"),
        ("close", "closes", "closed", "closing"),
        ("service", "services"),
        ("passage", "passages"),
        ("note", "notes", "noted"),
        ("study", "studies", "studied", "studying"),
        ("need", "needs", "needed"),
        ("agree", "agrees", "agreed", "agreeing"),
        ("use", "uses", "used", "using"),
        ("hundred", "hundreds"),
    ],
)
def test_inflections_share_a_key(forms: tuple[str, ...]) -> None:
    assert len({normalise_token(f) for f in forms}) == 1


def test_words_that_only_look_inflected_are_kept() -> None:
    assert normalise_token("bring") == "bring"
    assert normalise_token("red") == "red"
    assert normalise_token("bus") == normalise_token("buses") == "bus"


_ONSETS = st.sampled_from(["b", "c", "d", "f", "g", "h", "l", "m", "n", "p", "r", "s", "t",
                           "w", "br", "cl", "pr", "st", "tr", "sh", "ch", "sp"])  # fmt: skip
_VOWELS = st.sampled_from(["a", "i", "o", "u"])
_CODAS = st.sampled_from(["t", "n", "m", "p", "k", "d", "l", "r", "st", "nd", "rk", "mp", "sh",
                          "ch", "x", "ss", "ll"])  # fmt: skip
_SINGLE = st.sampled_from(["t", "n", "m", "p", "k", "d", "l", "r", "s", "c", "g", "v", "z"])


@st.composite
def _lemmas(draw: st.DrawFn) -> str:
    """Regular English-shaped verbs and nouns: CVC(C), CVCe ("close") and CVCy ("study")."""
    head = draw(_ONSETS) + draw(_VOWELS)
    if draw(st.booleans()):
        head += draw(_SINGLE) + draw(_VOWELS)
    shape = draw(st.sampled_from(["closed", "e", "y"]))
    if shape == "closed":
        return head + draw(_CODAS)
    return head + draw(_SINGLE) + shape


def _single_syllable_cvc(word: str) -> bool:
    vowel_groups = re.findall(r"[aeiou]+", word)
    return (
        len(vowel_groups) == 1
        and len(vowel_groups[0]) == 1
        and re.search(r"[^aeiou][aeiou][^aeiouwxy]$", word) is not None
    )


def _inflect(lemma: str) -> tuple[str, str, str]:
    """Plural or third person, past and -ing, by the regular spelling rules."""
    consonant_y = lemma.endswith("y") and lemma[-2] not in "aeiou"
    if lemma.endswith(("s", "x", "z", "sh", "ch")):
        plural = lemma + "es"
    elif consonant_y:
        plural = lemma[:-1] + "ies"
    else:
        plural = lemma + "s"
    if lemma.endswith("e"):
        past, ing = lemma + "d", lemma[:-1] + "ing"
    elif consonant_y:
        past, ing = lemma[:-1] + "ied", lemma + "ing"
    elif _single_syllable_cvc(lemma):
        past, ing = lemma + lemma[-1] + "ed", lemma + lemma[-1] + "ing"
    else:
        past, ing = lemma + "ed", lemma + "ing"
    return plural, past, ing


@given(_lemmas())
@settings(max_examples=400, deadline=None)
def test_plural_past_and_ing_forms_normalise_to_the_lemma_key(lemma: str) -> None:
    key = normalise_token(lemma)
    for form in _inflect(lemma):
        assert normalise_token(form) == key, (lemma, form)

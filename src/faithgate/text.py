"""Rule-based sentence and claim splitting with exact character offsets, source chunking and the
light word normalisation used by the lexical judge and the span localiser."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    text: str


# Tokens that end with a period without ending a sentence. Lower-cased, without the final period.
ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "ft", "vs", "etc", "inc", "ltd",
        "corp", "dept", "univ", "gov", "gen", "col", "lt", "sgt", "capt", "rev", "hon", "approx",
        "jan", "feb", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "mon",
        "tue", "thu", "fri", "ave", "blvd", "rd", "hwy",
    }
)  # fmt: skip
# Abbreviations that are also ordinary words ("the answer is no.", "we sat in the sun."). They only
# hold the sentence open before a number or a bracket ("No. 5", "Mar. 3", "et al. (2019)").
AMBIGUOUS_ABBREVIATIONS = frozenset(
    {"no", "nos", "co", "est", "al", "fig", "vol", "mar", "sat", "sun", "wed"}
)
# After a dotted acronym ("U.S.", "D.C.", "p.m.") a capitalised word is usually a name ("U.S.
# Army"); one of these openers means the acronym ended the sentence.
_SENTENCE_OPENERS = frozenset(
    "the a an he she it they we i you this that these those there his her its their our my "
    "in on at after before when while if as but however then also many some most each".split()
)

_BOUNDARY = re.compile(r"[.!?]+[\"')\]]*(?=\s)")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d{1,2}[.)])\s+")
_WORD = re.compile(r"[A-Za-z0-9]+(?:[.,'’][A-Za-z0-9]+)*")
_CLAUSE_JOINERS = re.compile(r";\s+|,\s+(but|while|whereas|and|although|though)\s+", re.I)
# After ", and" a new clause normally opens with a subject; anything else is usually a list item.
_SUBJECT_STARTERS = frozenset(
    "it they he she we i you this that these those there its their his her our many some most "
    "the a an each all both".split()
)

STOPWORDS = frozenset(
    """a an the and or but if then else of to in on at by for with from into onto over under
    about as is are was were be been being am do does did doing have has had having it its it's
    this that these those there here which who whom whose what when where why how not no nor so
    than too very can could should would may might must will shall i me my we our ours you your
    he him his she her they them their theirs also just only such some any each every both all
    most more other own same up down out off again further once s t don now one""".split()
)


def _strip_segment(text: str, start: int, end: int) -> Segment | None:
    piece = text[start:end]
    bullet = _BULLET.match(piece)
    if bullet:
        start += bullet.end()
        piece = text[start:end]
    lead = len(piece) - len(piece.lstrip())
    trail = len(piece) - len(piece.rstrip())
    start, end = start + lead, end - trail
    if start >= end or not any(ch.isalnum() for ch in text[start:end]):
        return None
    return Segment(start, end, text[start:end])


def _is_false_boundary(text: str, dot_pos: int, line_start: int) -> bool:
    """True when the punctuation at ``dot_pos`` belongs to an abbreviation or an initial."""
    if text[dot_pos] != ".":
        return False
    left = text[line_start:dot_pos]
    token = left.split()[-1] if left.split() else ""
    token = token.lstrip("(\"'")
    low = token.lower()
    nxt = text[dot_pos + 1 :].lstrip()
    if low in ("e.g", "i.e") or low in ABBREVIATIONS:
        return True
    if low in AMBIGUOUS_ABBREVIATIONS and nxt[:1] and (nxt[0].isdigit() or nxt[0] in "(["):
        return True
    # "1. Preheat the oven" - a list marker at the start of a line, not a sentence.
    if re.fullmatch(r"\d{1,2}", token) and left.strip() == token:
        return True
    # Single capital initials ("J. K. Rowling").
    if re.fullmatch(r"[A-Z]", token):
        return True
    if re.fullmatch(r"(?:[A-Za-z]\.)+[A-Za-z]", token):
        opener = re.match(r"[A-Za-z]+", nxt)
        return not (
            opener and opener.group()[0].isupper() and opener.group().lower() in _SENTENCE_OPENERS
        )
    # A lower-case continuation almost never starts a new sentence in edited text.
    return bool(nxt) and nxt[0].islower()


def split_sentences(text: str) -> list[Segment]:
    """Split ``text`` into sentences; every segment satisfies ``text[s.start:s.end] == s.text``."""
    segments: list[Segment] = []
    for line in re.finditer(r"[^\n]+", text):
        line_start, line_end = line.start(), line.end()
        cursor = line_start
        for match in _BOUNDARY.finditer(text, line_start, line_end):
            first_mark = match.start()
            if _is_false_boundary(text, first_mark, cursor):
                continue
            seg = _strip_segment(text, cursor, match.end())
            if seg:
                segments.append(seg)
            cursor = match.end()
        seg = _strip_segment(text, cursor, line_end)
        if seg:
            segments.append(seg)
    return segments


def _word_count(text: str) -> int:
    return len(_WORD.findall(text))


def _opens_clause(joiner: re.Match[str], right: str) -> bool:
    if joiner.group(1) is None or joiner.group(1).lower() != "and":
        return True
    first = right.split(maxsplit=1)[0] if right.strip() else ""
    return first.lower() in _SUBJECT_STARTERS or first[:1].isupper()


_REFUSAL = re.compile(
    r"\b(?:unable to answer|cannot answer|can't answer|not able to answer|i'm sorry|i am sorry)\b",
    re.I,
)


def is_checkable(segment: Segment) -> bool:
    """False for units that assert nothing about the world: questions, lead-ins ending in a colon
    ("Here is the answer:"), refusals, and fragments without a single letter."""
    tail = segment.text.rstrip("\"')] ")
    if tail.endswith(("?", ":")) or not any(ch.isalpha() for ch in segment.text):
        return False
    return not _REFUSAL.search(segment.text)


def split_claims(text: str, min_words: int = 5) -> list[Segment]:
    """Split sentences further at semicolons and clause-joining conjunctions.

    A sentence is only cut where both sides keep at least ``min_words`` words and, for "and",
    where the right side opens with a subject. Lists such as "fast, cheap, and friendly service"
    stay whole while "it opens at 9, and it closes at 5 on weekdays" is split.
    """
    claims: list[Segment] = []
    for sentence in split_sentences(text):
        cursor = sentence.start
        for joiner in _CLAUSE_JOINERS.finditer(text, sentence.start, sentence.end):
            left = text[cursor : joiner.start()]
            right = text[joiner.end() : sentence.end]
            long_enough = _word_count(left) >= min_words and _word_count(right) >= min_words
            if long_enough and _opens_clause(joiner, right):
                seg = _strip_segment(text, cursor, joiner.start())
                if seg:
                    claims.append(seg)
                # Keep the conjunction with the right-hand claim so it still reads as a clause.
                cursor = joiner.start() + (1 if text[joiner.start()] in ";," else 0)
        seg = _strip_segment(text, cursor, sentence.end)
        if seg:
            claims.append(seg)
    return claims


def split_units(text: str, unit: str) -> list[Segment]:
    """Checkable units of a response: sentences or claims, minus questions and lead-ins."""
    if unit == "sentence":
        units = split_sentences(text)
    elif unit == "claim":
        units = split_claims(text)
    else:
        raise ValueError(f"unknown unit {unit!r}; expected 'sentence' or 'claim'")
    return [u for u in units if is_checkable(u)]


def chunk_source(text: str, max_words: int, overlap: int = 1) -> list[Segment]:
    """Pack consecutive source sentences into chunks of at most ``max_words`` words.

    Chunks overlap by ``overlap`` sentences so evidence that straddles a boundary is seen whole by
    at least one chunk. A sentence longer than the budget is cut into word windows.
    """
    if max_words < 1:
        raise ValueError("max_words must be positive")
    pieces: list[Segment] = []
    for sentence in split_sentences(text):
        words = list(_WORD.finditer(text, sentence.start, sentence.end))
        if len(words) <= max_words:
            pieces.append(sentence)
            continue
        for i in range(0, len(words), max_words):
            window = words[i : i + max_words]
            start = sentence.start if i == 0 else window[0].start()
            end = sentence.end if i + max_words >= len(words) else window[-1].end()
            pieces.append(Segment(start, end, text[start:end]))
    if not pieces:
        return []
    counts = [_word_count(p.text) for p in pieces]
    chunks: list[Segment] = []
    i = 0
    while i < len(pieces):
        j, total = i, 0
        while j < len(pieces) and (j == i or total + counts[j] <= max_words):
            total += counts[j]
            j += 1
        chunks.append(
            Segment(pieces[i].start, pieces[j - 1].end, text[pieces[i].start : pieces[j - 1].end])
        )
        if j >= len(pieces):
            break
        i = max(i + 1, j - overlap)
    return chunks


def _is_vowel(word: str, i: int) -> bool:
    if word[i] in "aeiou":
        return True
    return word[i] == "y" and i > 0 and not _is_vowel(word, i - 1)


def _measure(stem: str) -> int:
    """Porter's m: how many vowel-consonant sequences ``stem`` contains."""
    shape = "".join("v" if _is_vowel(stem, i) else "c" for i in range(len(stem)))
    return len(re.findall(r"v+c+", shape))


def _strip_plural(word: str) -> str:
    if word.endswith("ies") and len(word) >= 4:
        return word[:-2]
    if (
        len(word) >= 5
        and word.endswith("es")
        and (word[-3] in "sxz" or word[-4:-2] in ("ch", "sh"))
    ):
        return word[:-2]
    if len(word) >= 4 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _strip_ed_ing(word: str) -> str:
    # "need" and "agreed" both end in -eed; _canonical tells them apart by the stem's measure.
    if word.endswith("eed"):
        return word
    for suffix in ("ing", "ed"):
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            # A stem without a vowel means the suffix was part of the word: "bring", "red".
            if len(stem) >= 2 and any(_is_vowel(stem, i) for i in range(len(stem))):
                return stem
            return word
    return word


def _canonical(stem: str) -> str:
    """Applied to every word, inflected or not, so both sides of a match end up in one form:
    final e dropped ("close", "clos-ed"), doubled consonant undone ("run", "runn-ing"), final
    consonant-y as i ("study", "studi-es")."""
    if stem.endswith("eed") and _measure(stem[:-3]) > 0:
        stem = stem[:-1]
    if stem.endswith("e") and len(stem) >= 3:
        stem = stem[:-1]
    if len(stem) >= 3 and stem[-1] == stem[-2] and not _is_vowel(stem, len(stem) - 1):
        stem = stem[:-1]
    if len(stem) >= 2 and stem.endswith("y") and not _is_vowel(stem, len(stem) - 2):
        stem = stem[:-1] + "i"
    return stem


def normalise_token(token: str) -> str:
    """Lower-case, drop thousands separators and possessives, and reduce the regular inflections
    (plural or third-person -s, -ed, -ing) to one key: "closes", "closed", "closing" and "close"
    all normalise to the same form."""
    tok = token.lower().replace("’", "'")
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", tok):
        return tok.replace(",", "")
    if tok.endswith("'s"):
        tok = tok[:-2]
    if not tok.isalpha():
        return tok
    return _canonical(_strip_ed_ing(_strip_plural(tok)))


def content_tokens(text: str, start: int = 0, end: int | None = None) -> list[tuple[int, int, str]]:
    """Content words of ``text[start:end]`` as (start, end, normalised form), stopwords removed."""
    stop = len(text) if end is None else end
    out = []
    for match in _WORD.finditer(text, start, stop):
        raw = match.group()
        if raw.lower() in STOPWORDS:
            continue
        out.append((match.start(), match.end(), normalise_token(raw)))
    return out


def vocabulary(text: str) -> set[str]:
    return {form for _, _, form in content_tokens(text)}

"""Faithfulness judges. Each maps a batch of records to :class:`Scored` rows whose ``score`` is a
raw hallucination score in [0, 1] (higher = more likely hallucinated)."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Protocol

from faithgate.llm import ChatProvider, Message
from faithgate.nli import EntailmentModel
from faithgate.records import Record, Scored, Span, UnitScore
from faithgate.spans import localise, merge_spans
from faithgate.summac import (
    ChunkConfig,
    ConvAggregator,
    NLIMatrix,
    build_matrices,
    zs_hallucination,
)
from faithgate.text import content_tokens, split_units, vocabulary


class Judge(Protocol):
    name: str

    def score(self, records: Sequence[Record]) -> list[Scored]: ...


class AlwaysHallucinatedJudge:
    """Baseline that flags every response in full."""

    name = "always-hallucinated"

    def score(self, records: Sequence[Record]) -> list[Scored]:
        return [
            Scored(
                id=r.id,
                judge=self.name,
                score=1.0,
                spans=[(0, len(r.response))] if r.response else [],
                task_type=r.task_type,
                label=r.label,
            )
            for r in records
        ]


class OverlapJudge:
    """Unigram-overlap baseline: the share of the response's content words absent from the source.

    Unit scores are the same share per sentence; spans are the runs of unsupported words in the
    sentences scored at or above ``span_threshold``.
    """

    name = "overlap"

    # 0.6 maximised span F1 on the RAGTruth train slice (bench/results/full.json, "spans").
    def __init__(self, span_threshold: float = 0.6) -> None:
        self.span_threshold = span_threshold

    def score(self, records: Sequence[Record]) -> list[Scored]:
        out = []
        for record in records:
            vocab = vocabulary(record.source)
            units: list[UnitScore] = []
            missing = total = 0
            for unit in split_units(record.response, "sentence"):
                tokens = content_tokens(record.response, unit.start, unit.end)
                absent = sum(1 for _, _, form in tokens if form not in vocab)
                missing += absent
                total += len(tokens)
                units.append(
                    UnitScore(
                        start=unit.start,
                        end=unit.end,
                        score=absent / len(tokens) if tokens else 0.0,
                    )
                )
            out.append(
                Scored(
                    id=record.id,
                    judge=self.name,
                    score=missing / total if total else 0.0,
                    units=units,
                    spans=localise(record.response, units, self.span_threshold, vocab),
                    task_type=record.task_type,
                    label=record.label,
                )
            )
        return out


class NLIJudge:
    """SummaC over an NLI model, aggregated zero-shot ('zs-mean', 'zs-min') or by a trained
    :class:`ConvAggregator`.

    Spans: units whose hallucination score is at least ``span_threshold``, narrowed to source-absent
    words when ``refine_spans`` is set. The default 0.6 is the value tuned on RAGTruth train for
    ``zs-mean`` with narrowing; other aggregations put unit scores on other scales.
    """

    def __init__(
        self,
        model: EntailmentModel,
        config: ChunkConfig,
        aggregation: str | ConvAggregator = "zs-min",
        *,
        span_threshold: float = 0.6,
        refine_spans: bool = True,
        keep_matrix: bool = False,
        batch_records: int = 32,
    ) -> None:
        if isinstance(aggregation, str) and aggregation not in ("zs-mean", "zs-min"):
            raise ValueError("aggregation must be 'zs-mean', 'zs-min' or a ConvAggregator")
        self.model = model
        self.config = config
        self.aggregation = aggregation
        self.span_threshold = span_threshold
        self.refine_spans = refine_spans
        self.keep_matrix = keep_matrix
        self.batch_records = batch_records
        label = aggregation if isinstance(aggregation, str) else f"conv-{aggregation.mode}"
        self.name = f"nli-{label}"

    def matrices(self, records: Sequence[Record]) -> list[NLIMatrix]:
        out: list[NLIMatrix] = []
        for start in range(0, len(records), self.batch_records):
            batch = records[start : start + self.batch_records]
            out.extend(
                build_matrices(self.model, [(r.source, r.response) for r in batch], self.config)
            )
        return out

    def score(self, records: Sequence[Record]) -> list[Scored]:
        return self.from_matrices(records, self.matrices(records))

    def from_matrices(
        self, records: Sequence[Record], matrices: Sequence[NLIMatrix]
    ) -> list[Scored]:
        if isinstance(self.aggregation, ConvAggregator):
            response_scores = self.aggregation.predict(matrices).tolist()
        else:
            agg = "mean" if self.aggregation == "zs-mean" else "min"
            response_scores = [zs_hallucination(m, agg) for m in matrices]
        out = []
        for record, matrix, response_score in zip(records, matrices, response_scores, strict=True):
            if isinstance(self.aggregation, ConvAggregator):
                unit_scores = self.aggregation.unit_probs(matrix).tolist()
            else:
                unit_scores = (1.0 - matrix.support()).tolist()
            units = [
                UnitScore(
                    start=u.start,
                    end=u.end,
                    score=float(s),
                    evidence=(ev.start, ev.end) if ev is not None else None,
                )
                for u, s, ev in zip(matrix.units, unit_scores, matrix.evidence(), strict=True)
            ]
            vocab = vocabulary(record.source) if self.refine_spans else None
            out.append(
                Scored(
                    id=record.id,
                    judge=self.name,
                    score=float(response_score),
                    units=units,
                    spans=localise(record.response, units, self.span_threshold, vocab),
                    task_type=record.task_type,
                    label=record.label,
                    unit_probs=matrix.probs.tolist() if self.keep_matrix else None,
                )
            )
        return out


_SYSTEM_PROMPT = (
    "You check whether a RESPONSE is fully supported by a SOURCE. Treat any statement that the "
    "SOURCE does not state or directly imply as unsupported, even if it is true in the world. "
    'Reply with JSON only: {"verdict": "supported" or "unsupported", "unsupported_quotes": '
    "[exact phrases copied from the RESPONSE that the SOURCE does not support]}."
)


def _block(title: str, text: str) -> str:
    return f"{title}:\n<<<\n{text}\n>>>"


def build_prompts(record: Record) -> tuple[list[Message], list[Message]]:
    """The same question with the SOURCE first and with the RESPONSE first."""
    question = "Is every statement in the RESPONSE supported by the SOURCE?"
    source, response = _block("SOURCE", record.source), _block("RESPONSE", record.response)
    forward: list[Message] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"{source}\n\n{response}\n\n{question}"},
    ]
    swapped: list[Message] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"{response}\n\n{source}\n\n{question}"},
    ]
    return forward, swapped


def parse_verdict(reply: str) -> tuple[float | None, list[str]]:
    """(1.0 unsupported | 0.0 supported | None unreadable, quoted phrases)."""
    match = re.search(r"\{.*\}", reply, flags=re.S)
    if not match:
        return None, []
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None, []
    verdict = str(data.get("verdict", "")).strip().lower()
    quotes = [str(q) for q in data.get("unsupported_quotes", []) if str(q).strip()]
    if verdict == "unsupported":
        return 1.0, quotes
    if verdict == "supported":
        return 0.0, quotes
    return None, quotes


def _locate(response: str, quotes: Sequence[str]) -> list[Span]:
    low = response.lower()
    spans: list[Span] = []
    for quote in quotes:
        at = low.find(quote.strip().lower())
        if at >= 0:
            spans.append((at, at + len(quote.strip())))
    return spans


class LLMJudge:
    """Asks a chat model twice, with the SOURCE and RESPONSE blocks in both orders.

    Agreeing verdicts give a score of 0 or 1; disagreeing or unreadable ones give 0.5 and mark the
    row ``consistent=False``, so position bias shows up as uncertainty instead of a silent flip.
    """

    def __init__(self, provider: ChatProvider, *, max_tokens: int = 400) -> None:
        self.provider = provider
        self.max_tokens = max_tokens
        self.name = f"llm:{provider.name}"

    def score(self, records: Sequence[Record]) -> list[Scored]:
        out = []
        for record in records:
            verdicts: list[float | None] = []
            quotes: list[str] = []
            for messages in build_prompts(record):
                reply = self.provider.complete(
                    messages, temperature=0.0, max_tokens=self.max_tokens
                )
                verdict, found = parse_verdict(reply)
                verdicts.append(verdict)
                quotes.extend(found)
            known = [v for v in verdicts if v is not None]
            consistent = len(known) == 2 and known[0] == known[1]
            score = known[0] if consistent else 0.5
            out.append(
                Scored(
                    id=record.id,
                    judge=self.name,
                    score=score,
                    spans=merge_spans(_locate(record.response, quotes)) if score > 0 else [],
                    task_type=record.task_type,
                    label=record.label,
                    consistent=consistent,
                )
            )
        return out


def consistency_rate(rows: Sequence[Scored]) -> float:
    """Share of rows whose two position-swapped verdicts agreed."""
    flags = [r.consistent for r in rows if r.consistent is not None]
    return sum(flags) / len(flags) if flags else 0.0

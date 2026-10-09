"""Record types shared by the CLI, the benchmark and the pytest gate, plus JSONL helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from pydantic import BaseModel, Field

Span = tuple[int, int]


class Record(BaseModel):
    """One RAG output: the context it was generated from and the response.

    ``label`` is the human verdict when one exists (1 = contains a hallucination). ``spans`` are
    character offsets into ``response`` that annotators marked as hallucinated.
    """

    id: str
    source: str
    response: str
    task_type: str | None = None
    label: int | None = Field(default=None, ge=0, le=1)
    spans: list[Span] = Field(default_factory=list)
    meta: dict[str, str] = Field(default_factory=dict)


class UnitScore(BaseModel):
    """A sentence or claim of the response and how likely the judge thinks it is unsupported."""

    start: int
    end: int
    score: float
    evidence: Span | None = None


class Scored(BaseModel):
    """Judge output for one record. ``score`` is the raw hallucination score in [0, 1]."""

    id: str
    judge: str
    score: float
    units: list[UnitScore] = Field(default_factory=list)
    spans: list[Span] = Field(default_factory=list)
    task_type: str | None = None
    label: int | None = Field(default=None, ge=0, le=1)
    consistent: bool | None = None
    unit_probs: list[list[float]] | None = None
    response: str | None = None


def read_jsonl[T: BaseModel](path: Path, model: type[T]) -> list[T]:
    with path.open(encoding="utf-8") as handle:
        return [model.model_validate_json(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[BaseModel]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(row.model_dump_json(exclude_none=True))
            handle.write("\n")
            count += 1
    return count


def iter_json_lines(path: Path) -> Iterator[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)

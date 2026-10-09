"""The CI gate: a hallucination-rate budget checked against a PPI upper confidence bound, and
per-response flags from the conformal-risk-control threshold."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from faithgate.artifacts import CalibrationArtifact
from faithgate.conformal import flag
from faithgate.ppi import estimate_rate, interval
from faithgate.records import Scored, read_jsonl


class GateConfig(BaseModel):
    """The ``[rag_gate]`` table of a gate TOML file. Relative paths resolve against the file."""

    scores: Path
    calibration: Path | None = None
    labels: Path | None = None
    max_rate: float = Field(gt=0, lt=1)
    confidence: float = Field(default=0.95, gt=0.5, lt=1)
    estimator: Literal["ppi++", "ppi", "classical"] = "ppi++"
    min_labelled: int = Field(default=30, ge=2)
    per_response: bool = True


class LabelRow(BaseModel):
    id: str
    label: int = Field(ge=0, le=1)


class RateCheck(BaseModel):
    estimator: str
    estimate: float
    upper: float
    max_rate: float
    confidence: float
    n_labelled: int
    n_unlabelled: int
    lam: float | None
    passed: bool


class ResponseCheck(BaseModel):
    id: str
    probability: float
    threshold: float
    flagged: bool
    unit_text: str | None
    unit_score: float | None


class GateOutcome(BaseModel):
    config: GateConfig
    rate: RateCheck | None
    rate_skipped: str | None
    responses: list[ResponseCheck]


def load_config(path: Path) -> GateConfig:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    table = data.get("rag_gate") or data.get("tool", {}).get("rag_gate")
    if table is None:
        raise ValueError(f"{path} has no [rag_gate] table")
    config = GateConfig.model_validate(table)
    base = path.parent
    updates: dict[str, Path] = {}
    for field in ("scores", "calibration", "labels"):
        value = getattr(config, field)
        if value is not None and not value.is_absolute():
            updates[field] = (base / value).resolve()
    return config.model_copy(update=updates)


def _attach_labels(rows: list[Scored], labels_path: Path) -> list[Scored]:
    labels = {row.id: row.label for row in read_jsonl(labels_path, LabelRow)}
    return [r.model_copy(update={"label": labels.get(r.id, r.label)}) for r in rows]


def evaluate(config: GateConfig) -> GateOutcome:
    rows = read_jsonl(config.scores, Scored)
    if not rows:
        raise ValueError(f"{config.scores} is empty")
    if config.labels is not None:
        rows = _attach_labels(rows, config.labels)
    artifact = CalibrationArtifact.load(config.calibration) if config.calibration else None
    scores = np.array([r.score for r in rows], dtype=float)
    probs = artifact.probabilities(scores) if artifact else np.clip(scores, 0.0, 1.0)

    has_label = np.array([r.label is not None for r in rows])
    y = np.array([r.label for r in rows if r.label is not None], dtype=float)
    rate: RateCheck | None = None
    skipped: str | None = None
    if y.size < config.min_labelled:
        skipped = (
            f"{y.size} labelled responses, fewer than min_labelled={config.min_labelled}; "
            "a rate bound from so few labels would not be trustworthy"
        )
    else:
        method = config.estimator if (~has_label).sum() >= 2 else "classical"
        estimate = estimate_rate(method, y, probs[has_label], probs[~has_label])
        bound = interval(estimate, alpha=1 - config.confidence, alternative="upper")
        rate = RateCheck(
            estimator=method,
            estimate=estimate.estimate,
            upper=bound.upper,
            max_rate=config.max_rate,
            confidence=config.confidence,
            n_labelled=int(y.size),
            n_unlabelled=int((~has_label).sum()),
            lam=estimate.lam,
            passed=bound.upper <= config.max_rate,
        )

    responses: list[ResponseCheck] = []
    if config.per_response:
        threshold = artifact.crc.threshold if artifact else 0.5
        flagged = flag(probs, threshold)
        for row, prob, is_flagged in zip(rows, probs, flagged, strict=True):
            worst = max(row.units, key=lambda u: u.score) if row.units else None
            text = row.response[worst.start : worst.end] if worst and row.response else None
            responses.append(
                ResponseCheck(
                    id=row.id,
                    probability=float(prob),
                    threshold=threshold,
                    flagged=bool(is_flagged),
                    unit_text=text,
                    unit_score=worst.score if worst else None,
                )
            )
    return GateOutcome(config=config, rate=rate, rate_skipped=skipped, responses=responses)

"""The calibration artifact written by ``faithgate calibrate`` and read by ``report`` and the gate:
a fitted calibrator plus a conformal-risk-control flag threshold on the calibrated scale."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from faithgate.calibration import (
    CalibratorParams,
    Method,
    brier_score,
    calibrator_from_params,
    expected_calibration_error,
    make_calibrator,
)
from faithgate.conformal import CRCThreshold, Risk, crc_threshold
from faithgate.metrics import auroc
from faithgate.records import Scored


class FitSummary(BaseModel):
    n: int
    positives: int
    auroc: float | None
    ece_raw: float
    ece_calibrated: float
    brier_raw: float
    brier_calibrated: float


class CalibrationArtifact(BaseModel):
    judge: str
    calibrator: CalibratorParams
    crc: CRCThreshold
    summary: FitSummary
    created: str

    def probabilities(self, scores: Sequence[float] | np.ndarray) -> np.ndarray:
        return calibrator_from_params(self.calibrator).predict(scores)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CalibrationArtifact:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def labelled(rows: Sequence[Scored]) -> tuple[np.ndarray, np.ndarray]:
    kept = [r for r in rows if r.label is not None]
    return np.array([r.score for r in kept], dtype=float), np.array(
        [r.label for r in kept], dtype=int
    )


def fit_artifact(
    rows: Sequence[Scored],
    method: Method = "isotonic",
    alpha: float = 0.1,
    risk: Risk = "slip",
) -> CalibrationArtifact:
    """Fit the calibrator and the CRC threshold on the labelled rows of ``rows``.

    The same labelled rows serve both fits; the CRC guarantee then holds for the calibrated score
    as a fixed function only if these rows are not reused to estimate the rate later.
    """
    judges = {r.judge for r in rows}
    if len(judges) != 1:
        raise ValueError(f"expected rows from one judge, got {sorted(judges)}")
    scores, labels = labelled(rows)
    if labels.size < 10 or labels.min() == labels.max():
        raise ValueError("calibration needs at least 10 labelled rows covering both classes")
    calibrator = make_calibrator(method).fit(scores, labels)
    probs = calibrator.predict(scores)
    summary = FitSummary(
        n=int(labels.size),
        positives=int(labels.sum()),
        auroc=auroc(labels, scores),
        ece_raw=expected_calibration_error(np.clip(scores, 0, 1), labels),
        ece_calibrated=expected_calibration_error(probs, labels),
        brier_raw=brier_score(np.clip(scores, 0, 1), labels),
        brier_calibrated=brier_score(probs, labels),
    )
    return CalibrationArtifact(
        judge=judges.pop(),
        calibrator=calibrator.params(),
        crc=crc_threshold(probs, labels, alpha=alpha, risk=risk),
        summary=summary,
        created=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    )

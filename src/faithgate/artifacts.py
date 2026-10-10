"""The calibration artifact written by ``faithgate calibrate`` and read by ``report`` and the gate:
a fitted calibrator plus a conformal-risk-control flag threshold on the judge's raw score."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ValidationError

from faithgate.calibration import (
    CalibratorParams,
    Method,
    brier_score,
    calibrator_from_params,
    expected_calibration_error,
    make_calibrator,
)
from faithgate.conformal import CRCThreshold, Risk, crc_threshold, flag
from faithgate.metrics import auroc
from faithgate.records import Scored


class FitSummary(BaseModel):
    """Fit diagnostics. The in-sample figures come from the rows the calibrator was fitted on,
    where isotonic regression reaches an ECE near zero by construction; the cross-validated ones
    are what to expect on new responses."""

    n: int
    positives: int
    auroc: float | None
    ece_raw: float
    ece_calibrated_in_sample: float
    ece_calibrated_cv: float | None
    brier_raw: float
    brier_calibrated_in_sample: float
    brier_calibrated_cv: float | None
    cv_folds: int | None


class CalibrationArtifact(BaseModel):
    judge: str
    calibrator: CalibratorParams
    crc: CRCThreshold
    # The CRC threshold applies to the judge's raw score, not to the calibrated probability.
    # Artifacts from before this field put it on the calibrated scale and are refused on load.
    crc_scale: Literal["raw score"]
    summary: FitSummary
    created: str

    def probabilities(self, scores: Sequence[float] | np.ndarray) -> np.ndarray:
        return calibrator_from_params(self.calibrator).predict(scores)

    def flags(self, scores: Sequence[float] | np.ndarray) -> np.ndarray:
        """Responses at or above the CRC threshold, compared on the raw score."""
        return flag(scores, self.crc.threshold)

    def check_judge(self, judge: str) -> None:
        """Refuse scores from a judge other than the one this artifact was fitted for: the
        calibrator and the flag threshold mean nothing on another judge's scale."""
        if judge != self.judge:
            raise ValueError(
                f"the calibration was fitted on {self.judge!r} scores but the scores come from "
                f"{judge!r}; refit it with `faithgate calibrate` on {judge!r} scores"
            )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CalibrationArtifact:
        try:
            return cls.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError as exc:
            if any(e["loc"] == ("crc_scale",) for e in exc.errors()):
                raise ValueError(
                    f"{path} was written by an older faithgate whose flag threshold is on the "
                    "calibrated scale; refit it with `faithgate calibrate`"
                ) from exc
            raise


def judge_of(rows: Sequence[Scored]) -> str:
    """The single judge that produced ``rows``."""
    judges = {r.judge for r in rows}
    if len(judges) != 1:
        raise ValueError(f"expected rows from one judge, got {sorted(judges)}")
    return judges.pop()


def out_of_fold_probabilities(
    method: Method, scores: np.ndarray, labels: np.ndarray, folds: int = 5, seed: int = 0
) -> tuple[np.ndarray, int] | None:
    """Each row's calibrated probability from a calibrator fitted without that row's fold.

    Folds are stratified by label so every training part sees both classes. Returns None when
    the smaller class has fewer than two rows.
    """
    k = min(folds, int(labels.sum()), int(labels.size - labels.sum()))
    if k < 2:
        return None
    rng = np.random.default_rng(seed)
    fold = np.empty(labels.size, dtype=int)
    for cls in (0, 1):
        members = rng.permutation(np.flatnonzero(labels == cls))
        fold[members] = np.arange(members.size) % k
    out = np.empty(labels.size)
    for f in range(k):
        held = fold == f
        out[held] = make_calibrator(method).fit(scores[~held], labels[~held]).predict(scores[held])
    return out, k


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

    CRC needs a score that was fixed before the calibration labels were seen. The calibrator is
    fitted on these same labels (Platt even picks its direction from them, and histogram bins
    need not be monotone), so the threshold is chosen on the raw judge score instead, and the
    calibrator only turns scores into probabilities for the rate estimate and the report. The
    rows must still not be reused as the labelled sample of a later rate estimate.
    """
    judge = judge_of(rows)
    scores, labels = labelled(rows)
    if labels.size < 10 or labels.min() == labels.max():
        raise ValueError("calibration needs at least 10 labelled rows covering both classes")
    calibrator = make_calibrator(method).fit(scores, labels)
    probs = calibrator.predict(scores)
    crc = crc_threshold(scores, labels, alpha=alpha, risk=risk)
    oof = out_of_fold_probabilities(method, scores, labels)
    summary = FitSummary(
        n=int(labels.size),
        positives=int(labels.sum()),
        auroc=auroc(labels, scores),
        ece_raw=expected_calibration_error(np.clip(scores, 0, 1), labels),
        ece_calibrated_in_sample=expected_calibration_error(probs, labels),
        ece_calibrated_cv=expected_calibration_error(oof[0], labels) if oof else None,
        brier_raw=brier_score(np.clip(scores, 0, 1), labels),
        brier_calibrated_in_sample=brier_score(probs, labels),
        brier_calibrated_cv=brier_score(oof[0], labels) if oof else None,
        cv_folds=oof[1] if oof else None,
    )
    return CalibrationArtifact(
        judge=judge,
        calibrator=calibrator.params(),
        crc=crc,
        crc_scale="raw score",
        summary=summary,
        created=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    )

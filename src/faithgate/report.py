"""JSON and HTML reports for a scored batch: rate intervals, meta-evaluation against whatever
human labels exist, reliability diagrams and the responses flagged by the CRC threshold."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from jinja2 import Environment, PackageLoader, select_autoescape
from pydantic import BaseModel

from faithgate.artifacts import CalibrationArtifact, judge_of
from faithgate.calibration import (
    ReliabilityBin,
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_bins,
)
from faithgate.conformal import CRCThreshold, flag
from faithgate.judges import consistency_rate
from faithgate.metrics import PRF, auroc, cohen_kappa, precision_recall_f1
from faithgate.ppi import (
    Interval,
    classical_mean,
    interval,
    judge_only_mean,
    ppi_mean,
    ppi_plusplus_mean,
)
from faithgate.records import Scored


class RateBlock(BaseModel):
    confidence: float
    n_labelled: int
    n_unlabelled: int
    intervals: list[Interval]
    note: str


class LabelledMetrics(BaseModel):
    n: int
    positives: int
    auroc: float | None
    decision_threshold: float
    prf: PRF
    kappa: float
    ece_raw: float
    ece_calibrated: float
    mce_raw: float
    mce_calibrated: float
    brier_raw: float
    brier_calibrated: float


class FlaggedRow(BaseModel):
    id: str
    score: float
    probability: float
    unit_text: str | None
    unit_score: float | None


class Report(BaseModel):
    judge: str
    created: str
    n: int
    n_labelled: int
    calibration_method: str
    rate: RateBlock
    metrics: LabelledMetrics | None
    reliability_raw: list[ReliabilityBin]
    reliability_calibrated: list[ReliabilityBin]
    crc: CRCThreshold | None
    flagged: list[FlaggedRow]
    position_swap_consistency: float | None = None


def rate_intervals(
    labels: np.ndarray, f_labelled: np.ndarray, f_unlabelled: np.ndarray, confidence: float
) -> RateBlock:
    alpha = 1 - confidence
    found: list[Interval] = []
    if labels.size >= 2:
        found.append(interval(classical_mean(labels), alpha))
    if labels.size >= 2 and f_unlabelled.size >= 2:
        found.append(interval(ppi_mean(labels, f_labelled, f_unlabelled), alpha))
        found.append(interval(ppi_plusplus_mean(labels, f_labelled, f_unlabelled), alpha))
    f_all = np.concatenate([f_labelled, f_unlabelled])
    if f_all.size >= 2:
        found.append(interval(judge_only_mean(f_all), alpha))
    if labels.size < 2:
        note = (
            "No human labels: only the judge-only estimate exists, and it carries the judge's bias."
        )
    elif f_unlabelled.size < 2:
        note = "Every response is labelled, so the classical interval is the one to use."
    else:
        note = (
            "classical, PPI and PPI++ cover the true rate at the stated level when the labelled "
            "responses are a uniform random subset of the batch; judge-only does not."
        )
    # A rate lives in [0, 1]; clipping the displayed interval cannot change whether it covers.
    clipped = [
        iv.model_copy(update={"lower": max(0.0, iv.lower or 0.0), "upper": min(1.0, iv.upper)})
        for iv in found
    ]
    return RateBlock(
        confidence=confidence,
        n_labelled=int(labels.size),
        n_unlabelled=int(f_unlabelled.size),
        intervals=clipped,
        note=note,
    )


def _top_unit(row: Scored) -> tuple[str | None, float | None]:
    if not row.units:
        return None, None
    worst = max(row.units, key=lambda u: u.score)
    text = row.response[worst.start : worst.end] if row.response else None
    return text, worst.score


def build_report(
    rows: Sequence[Scored],
    artifact: CalibrationArtifact | None = None,
    confidence: float = 0.95,
) -> Report:
    if not rows:
        raise ValueError("nothing to report")
    judge = judge_of(rows)
    if artifact is not None:
        artifact.check_judge(judge)
    scores = np.array([r.score for r in rows], dtype=float)
    probs = artifact.probabilities(scores) if artifact else np.clip(scores, 0.0, 1.0)
    has_label = np.array([r.label is not None for r in rows])
    labels = np.array([r.label for r in rows if r.label is not None], dtype=int)
    rate = rate_intervals(labels, probs[has_label], probs[~has_label], confidence)

    # The CRC threshold is on the raw score; without an artifact, 0.5 on the raw score.
    threshold = artifact.crc.threshold if artifact else 0.5
    flagged_mask = flag(scores, threshold)
    metrics = None
    raw_bins: list[ReliabilityBin] = []
    cal_bins: list[ReliabilityBin] = []
    if labels.size >= 2:
        raw = np.clip(scores[has_label], 0.0, 1.0)
        cal = probs[has_label]
        decided = flagged_mask[has_label]
        both_classes = labels.min() != labels.max()
        metrics = LabelledMetrics(
            n=int(labels.size),
            positives=int(labels.sum()),
            auroc=auroc(labels, scores[has_label]) if both_classes else None,
            decision_threshold=threshold,
            prf=precision_recall_f1(labels, decided),
            kappa=cohen_kappa(labels.tolist(), decided.astype(int).tolist()),
            ece_raw=expected_calibration_error(raw, labels),
            ece_calibrated=expected_calibration_error(cal, labels),
            mce_raw=maximum_calibration_error(raw, labels),
            mce_calibrated=maximum_calibration_error(cal, labels),
            brier_raw=brier_score(raw, labels),
            brier_calibrated=brier_score(cal, labels),
        )
        raw_bins = reliability_bins(raw, labels, n_bins=10)
        cal_bins = reliability_bins(cal, labels, n_bins=10)

    flagged = []
    for row, score, prob, is_flagged in zip(rows, scores, probs, flagged_mask, strict=True):
        if is_flagged:
            text, unit_score = _top_unit(row)
            flagged.append(
                FlaggedRow(
                    id=row.id,
                    score=float(score),
                    probability=float(prob),
                    unit_text=text,
                    unit_score=unit_score,
                )
            )
    flagged.sort(key=lambda f: (-f.probability, -f.score))
    return Report(
        judge=judge,
        created=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        n=len(rows),
        n_labelled=int(labels.size),
        calibration_method=artifact.calibrator.method if artifact else "none",
        rate=rate,
        metrics=metrics,
        reliability_raw=raw_bins,
        reliability_calibrated=cal_bins,
        crc=artifact.crc if artifact else None,
        flagged=flagged,
        position_swap_consistency=(
            consistency_rate(rows) if any(r.consistent is not None for r in rows) else None
        ),
    )


def reliability_svg(
    series: Sequence[tuple[str, Sequence[ReliabilityBin], str]], size: int = 280
) -> str:
    """A reliability diagram: mean predicted probability against observed hallucination rate,
    one marker per non-empty bin, marker area proportional to the bin's share of responses."""
    pad = 36
    span = size - 2 * pad

    def xy(p: float, q: float) -> tuple[float, float]:
        return pad + p * span, size - pad - q * span

    parts = [
        f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" role="img" '
        'aria-label="reliability diagram" xmlns="http://www.w3.org/2000/svg">',
        f'<rect x="{pad}" y="{pad}" width="{span}" height="{span}" class="frame"/>',
    ]
    x0, y0 = xy(0, 0)
    x1, y1 = xy(1, 1)
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" class="diag"/>')
    for tick in (0.0, 0.5, 1.0):
        tx, ty = xy(tick, 0)
        parts.append(
            f'<text x="{tx}" y="{ty + 16}" class="tick" text-anchor="middle">{tick:g}</text>'
        )
        lx, ly = xy(0, tick)
        parts.append(
            f'<text x="{lx - 6}" y="{ly + 4}" class="tick" text-anchor="end">{tick:g}</text>'
        )
    parts.append(
        f'<text x="{size / 2}" y="{size - 4}" class="axis" text-anchor="middle">'
        "predicted probability</text>"
    )
    parts.append(
        f'<text x="12" y="{size / 2}" class="axis" text-anchor="middle" '
        f'transform="rotate(-90 12 {size / 2})">observed rate</text>'
    )
    for _name, bins, css in series:
        total = sum(b.count for b in bins) or 1
        points = [xy(b.mean_predicted, b.fraction_positive) for b in bins]
        if len(points) > 1:
            path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            parts.append(f'<polyline points="{path}" class="line {css}"/>')
        for b, (x, y) in zip(bins, points, strict=True):
            radius = 2.5 + 9 * (b.count / total) ** 0.5
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" class="dot {css}">'
                f"<title>{b.count} responses, predicted {b.mean_predicted:.2f}, "
                f"observed {b.fraction_positive:.2f}</title></circle>"
            )
    parts.append("</svg>")
    return "".join(parts)


def render_html(report: Report) -> str:
    env = Environment(
        loader=PackageLoader("faithgate", "templates"),
        autoescape=select_autoescape(["html", "j2"]),
    )
    series = []
    if report.reliability_raw:
        series.append(("raw score", report.reliability_raw, "raw"))
    if report.reliability_calibrated and report.calibration_method != "none":
        series.append(
            (f"calibrated ({report.calibration_method})", report.reliability_calibrated, "cal")
        )
    svg = reliability_svg(series) if series else ""
    return env.get_template("report.html.j2").render(report=report, svg=svg, series=series)


def write_report(report: Report, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, html_path = out_dir / "report.json", out_dir / "report.html"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")
    return json_path, html_path

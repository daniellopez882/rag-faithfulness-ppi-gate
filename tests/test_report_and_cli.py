from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from faithgate.artifacts import CalibrationArtifact, fit_artifact
from faithgate.cli import main
from faithgate.gate import evaluate, load_config
from faithgate.records import Scored, read_jsonl
from faithgate.report import build_report, render_html
from tests.conftest import EXAMPLES


def _scored(n: int, labelled: int, seed: int = 0) -> list[Scored]:
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=n) < 0.3).astype(int)
    s = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.15, n), 0, 1)
    return [
        Scored(id=f"x{i}", judge="j", score=float(s[i]), label=int(y[i]) if i < labelled else None)
        for i in range(n)
    ]


def test_report_with_partial_labels_has_all_intervals() -> None:
    artifact = fit_artifact(_scored(300, 300, seed=1), method="platt", alpha=0.1)
    report = build_report(_scored(500, 100), artifact, confidence=0.9)
    methods = [iv.method for iv in report.rate.intervals]
    assert methods == ["classical", "ppi", "ppi++", "judge-only"]
    assert all(
        0.0 <= (iv.lower or 0.0) <= iv.estimate <= iv.upper <= 1.0 for iv in report.rate.intervals
    )
    assert report.metrics is not None
    assert report.metrics.n == 100
    assert report.reliability_calibrated
    assert all(f.probability >= artifact.crc.threshold for f in report.flagged)
    html = render_html(report)
    assert "<svg" in html
    assert "ppi++" in html


def test_report_without_labels_or_artifact() -> None:
    report = build_report(_scored(50, 0))
    assert [iv.method for iv in report.rate.intervals] == ["judge-only"]
    assert report.metrics is None
    assert "No human labels" in report.rate.note
    assert "<svg" not in render_html(report)


def test_fit_artifact_validation() -> None:
    with pytest.raises(ValueError, match="at least 10"):
        fit_artifact(_scored(5, 5))
    mixed = [*_scored(20, 20), Scored(id="z", judge="other", score=0.1, label=0)]
    with pytest.raises(ValueError, match="one judge"):
        fit_artifact(mixed)


def _run(*args: object) -> int:
    return main([str(a) for a in args])


def _score_examples(out: Path) -> None:
    assert (
        _run("score", EXAMPLES / "outputs.jsonl", "--judge", "overlap", "-o", out / "scores.jsonl")
        == 0
    )
    assert (
        _run("score", EXAMPLES / "calibration.jsonl", "--judge", "overlap", "-o", out / "cal.jsonl")
        == 0
    )
    calibration = out / "calibration.json"
    assert (
        _run(
            "calibrate", out / "cal.jsonl", "--method", "platt", "--alpha", "0.2", "-o", calibration
        )
        == 0
    )


def test_cli_pipeline_on_examples(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _score_examples(tmp_path)
    report_dir = tmp_path / "rep"
    calibration = tmp_path / "calibration.json"
    scores = tmp_path / "scores.jsonl"
    assert _run("report", scores, "--calibration", calibration, "--out-dir", report_dir) == 0
    rows = read_jsonl(scores, Scored)
    assert len(rows) == 16
    assert rows[0].response is not None
    assert CalibrationArtifact.load(calibration).calibrator.method == "platt"
    data = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    assert data["n"] == 16
    assert data["n_labelled"] == 8
    assert (report_dir / "report.html").exists()
    assert "ppi++" in capsys.readouterr().out


def test_example_gate_config_evaluates(tmp_path: Path) -> None:
    _score_examples(tmp_path / "out")
    gate = tmp_path / "rag-gate.toml"
    gate.write_text((EXAMPLES / "rag-gate.toml").read_text(encoding="utf-8"), encoding="utf-8")
    outcome = evaluate(load_config(gate))
    assert outcome.rate is not None
    assert outcome.rate.passed
    assert outcome.rate.n_labelled == 8


def test_cli_rejects_incomplete_judge_options(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="llm-base-url"):
        _run("score", EXAMPLES / "outputs.jsonl", "--judge", "llm", "-o", tmp_path / "s.jsonl")

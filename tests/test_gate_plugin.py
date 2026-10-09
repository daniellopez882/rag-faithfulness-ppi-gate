from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from faithgate.artifacts import fit_artifact
from faithgate.records import Scored, UnitScore, write_jsonl


def _rows(n: int, n_labelled: int, rate: float, seed: int) -> list[Scored]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        y = int(rng.uniform() < rate)
        score = float(np.clip(0.2 + 0.6 * y + rng.normal(0, 0.12), 0, 1))
        response = f"Response {i} says something. It adds a detail."
        rows.append(
            Scored(
                id=f"r{i}",
                judge="overlap",
                score=score,
                units=[
                    UnitScore(start=0, end=27, score=score / 2),
                    UnitScore(start=28, end=46, score=score),
                ],
                label=y if i < n_labelled else None,
                response=response,
            )
        )
    return rows


def _write_gate(folder: Path, body: str, rate: float = 0.1, n_labelled: int = 80) -> None:
    write_jsonl(folder / "scores.jsonl", _rows(400, n_labelled, rate, seed=1))
    fit_artifact(_rows(300, 300, 0.3, seed=2), method="isotonic", alpha=0.1).save(
        folder / "calibration.json"
    )
    (folder / "gate.toml").write_text(
        '[rag_gate]\nscores = "scores.jsonl"\ncalibration = "calibration.json"\n' + body,
        encoding="utf-8",
    )


def test_rate_within_budget_passes(pytester: pytest.Pytester) -> None:
    _write_gate(pytester.path, "max_rate = 0.3\nper_response = false\n")
    result = pytester.runpytest("--rag-gate", "gate.toml")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(
        ["*rag-gate*", "rate (ppi++, lambda *): estimate *upper bound*budget 0.300 -> pass*"]
    )


def test_rate_over_budget_fails_with_reason(pytester: pytest.Pytester) -> None:
    _write_gate(pytester.path, "max_rate = 0.05\nper_response = false\n", rate=0.2)
    result = pytester.runpytest("--rag-gate", "gate.toml")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*exceeds the budget 0.050*"])
    assert "Traceback" not in result.stdout.str()


def test_per_response_items_fail_above_the_crc_threshold(pytester: pytest.Pytester) -> None:
    _write_gate(pytester.path, "max_rate = 0.5\nper_response = true\n", rate=0.2)
    result = pytester.runpytest("--rag-gate", "gate.toml", "-q")
    outcomes = result.parseoutcomes()
    assert outcomes["passed"] > 1
    assert outcomes["failed"] >= 1
    assert outcomes["passed"] + outcomes["failed"] == 401
    result.stdout.fnmatch_lines(
        ["*flagged: calibrated hallucination probability*least supported unit*"]
    )


def test_too_few_labels_skips_the_rate_item(pytester: pytest.Pytester) -> None:
    _write_gate(
        pytester.path, "max_rate = 0.3\nper_response = false\nmin_labelled = 100\n", n_labelled=20
    )
    result = pytester.runpytest("--rag-gate", "gate.toml", "-rs")
    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(["*fewer than min_labelled=100*"])


def test_keyword_selection_and_coexistence_with_tests(pytester: pytest.Pytester) -> None:
    _write_gate(pytester.path, "max_rate = 0.3\nper_response = true\n")
    pytester.makepyfile(test_unit="def test_ok():\n    assert True\n")
    result = pytester.runpytest("--rag-gate", "gate.toml", "-k", "rag_gate_rate or test_ok")
    result.assert_outcomes(passed=2, deselected=400)


def test_bad_config_is_a_usage_error(pytester: pytest.Pytester) -> None:
    (pytester.path / "gate.toml").write_text(
        '[rag_gate]\nscores = "missing.jsonl"\n', encoding="utf-8"
    )
    result = pytester.runpytest("--rag-gate", "gate.toml")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*--rag-gate gate.toml*GateConfig*", "max_rate"])


def test_without_the_option_nothing_is_added(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_unit="def test_ok():\n    assert True\n")
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)
    assert "rag-gate" not in result.stdout.str()

"""Print the README's results tables from a results JSON written by ``bench.run``:

uv run python -m bench.tables bench/results/full.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

TASKS = ("QA", "Summary", "Data2txt")


def _f(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def measured_on(r: dict[str, Any]) -> str:
    m = r["machine"]
    return (
        f"Measured on {m['cpu']}, {m['logical_cores']} cores, {m['ram_gb']} GB RAM, {m['os']}, "
        f"Python {m['python']}, onnxruntime {m['packages']['onnxruntime']}, CPU only; "
        f"{r['date'][:10]}, commit {r['git_sha'][:7]}, profile {r['profile']}."
    )


def judges_table(r: dict[str, Any]) -> str:
    rows = []
    for name, m in r["meta_eval"].items():
        ci = m["auroc_ci95"]
        iso = m["calibrated"].get("isotonic")
        rows.append(
            [
                name,
                f"{_f(m['auroc'])} [{_f(ci['lower'])}, {_f(ci['upper'])}]",
                _f(m["uncalibrated_at_0.5"]["f1"]),
                _f(m["at_tuned_threshold"]["precision"]),
                _f(m["at_tuned_threshold"]["recall"]),
                _f(m["at_tuned_threshold"]["f1"]),
                _f(m["ece_raw"]),
                _f(iso["ece"]) if iso else "n/a",
                f"{_f(m['brier_raw'])} -> {_f(iso['brier'])}" if iso else _f(m["brier_raw"]),
            ]
        )
    header = [
        "judge",
        "AUROC [95% CI]",
        "F1 raw@0.5",
        "P tuned",
        "R tuned",
        "F1 tuned",
        "ECE raw",
        "ECE isotonic",
        "Brier raw -> isotonic",
    ]
    return _table(header, rows)


def calibration_table(r: dict[str, Any], judge: str) -> str:
    m = r["meta_eval"][judge]
    rows = [
        [
            "none",
            _f(m["ece_raw"]),
            _f(m["mce_raw"]),
            _f(m["brier_raw"]),
            _f(m["uncalibrated_at_0.5"]["f1"]),
        ]
    ]
    for method, c in m["calibrated"].items():
        rows.append([method, _f(c["ece"]), _f(c["mce"]), _f(c["brier"]), _f(c["at_0.5"]["f1"])])
    per_task = m["isotonic_per_task"]
    rows.append(
        [
            "isotonic, one per task type",
            _f(per_task["ece"]),
            "",
            _f(per_task["brier"]),
            _f(per_task["at_0.5"]["f1"]),
        ]
    )
    return _table(["calibration", "ECE", "MCE", "Brier", "F1 at 0.5"], rows)


def per_task_table(r: dict[str, Any], judge: str) -> str:
    m = r["meta_eval"][judge]
    overlap = r["meta_eval"]["overlap"]
    rows = []
    for task in TASKS:
        t = m["per_task"].get(task)
        if t is None:
            continue
        rows.append(
            [
                task,
                str(t["n"]),
                _f(t["positive_rate"]),
                _f(t["auroc"]),
                _f(overlap["per_task"][task]["auroc"]),
                _f(t["f1_at_tuned_threshold"]),
                _f(t["ece_raw"]),
                _f(t["ece_isotonic_global"]),
                _f(m["isotonic_per_task"]["per_task"][task]["ece_isotonic_per_task"]),
            ]
        )
    header = [
        "task",
        "n",
        "hallucinated share",
        f"AUROC {judge}",
        "AUROC overlap",
        "F1 tuned",
        "ECE raw",
        "ECE isotonic (global)",
        "ECE isotonic (per task)",
    ]
    return _table(header, rows)


def spans_table(r: dict[str, Any]) -> str:
    rows = []
    for name, s in r["spans"].items():
        rows.append(
            [name, _f(s.get("unit_threshold"), 2), _f(s["precision"]), _f(s["recall"]), _f(s["f1"])]
        )
    return _table(
        ["span predictor", "unit threshold", "char precision", "char recall", "char F1"], rows
    )


def ppi_table(r: dict[str, Any], judge: str) -> str:
    ppi = r["ppi"]
    rows = []
    for n, by_method in ppi["by_judge"][judge].items():
        for method in ("classical", "ppi", "ppi++", "judge-only"):
            s = by_method[method]
            rows.append(
                [
                    n,
                    method,
                    _f(s["coverage"]),
                    _f(s["mean_width"]),
                    _f(s["upper_coverage"]),
                    _f(s.get("mean_lambda"), 2) if "mean_lambda" in s else "",
                ]
            )
    nominal = 1 - ppi["alpha_two_sided"]
    upper = 1 - ppi["alpha_upper"]
    header = [
        "labelled n",
        "estimator",
        f"coverage (nominal {nominal:.2f})",
        "mean CI width",
        f"upper-bound coverage (nominal {upper:.3f})",
        "mean lambda",
    ]
    return _table(header, rows)


def crc_table(r: dict[str, Any]) -> str:
    rows = []
    for row in r["crc"]["rows"]:
        fixed, resampled = row["calibrated_on_train"], row["resampled_within_test"]
        rows.append(
            [
                row["risk"],
                _f(row["alpha"], 2),
                _f(fixed["threshold"]),
                _f(fixed["test_risk"]),
                _f(fixed["test_flag_rate"]),
                _f(resampled["mean_risk"]),
                _f(resampled["mean_flag_rate"]),
            ]
        )
    header = [
        "risk",
        "alpha",
        "threshold (train calibration)",
        "test risk",
        "test flag rate",
        "mean risk, resampled",
        "mean flag rate, resampled",
    ]
    return _table(header, rows)


def throughput_table(r: dict[str, Any]) -> str:
    t = r["throughput"]
    rows = [
        [
            f"NLI ({t['nli']['config']})",
            str(t["nli"]["responses"]),
            _f(t["nli"]["responses_per_second"], 2),
            _f(t["nli"]["nli_pairs_per_second"], 1),
        ],
        [
            "overlap",
            str(t["overlap"]["responses"]),
            _f(t["overlap"]["responses_per_second"], 1),
            "",
        ],
    ]
    return _table(["judge", "responses timed", "responses/s", "NLI pairs/s"], rows)


def chunk_table(r: dict[str, Any]) -> str:
    rows = [
        [g["config"], _f(g["auroc_zs-mean"]), _f(g["auroc_zs-min"]), _f(g["pairs_per_response"], 1)]
        for g in r["chunk_selection"]["grid"]
    ]
    return _table(
        ["config", "dev AUROC zs-mean", "dev AUROC zs-min", "NLI pairs per response"], rows
    )


def main(argv: list[str]) -> int:
    results = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    primary = results["primary_judge"]
    sections = [
        ("Measured on", measured_on(results)),
        ("Chunking chosen on dev", chunk_table(results)),
        ("Judges on RAGTruth test", judges_table(results)),
        (f"Calibration of {primary}", calibration_table(results, primary)),
        (f"Per task type, {primary}", per_task_table(results, primary)),
        ("Span localisation", spans_table(results)),
        (f"PPI coverage, {primary} (isotonic)", ppi_table(results, f"{primary} (isotonic)")),
        ("PPI coverage, overlap (isotonic)", ppi_table(results, "overlap (isotonic)")),
        (f"Conformal risk control, {primary} (isotonic)", crc_table(results)),
        ("Throughput", throughput_table(results)),
    ]
    for title, body in sections:
        print(f"### {title}\n\n{body}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

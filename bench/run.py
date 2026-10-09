"""Benchmark harness. One command runs everything and writes one results JSON:

    uv run python -m bench.run --profile full

Stages: chunk-size selection on a dev slice of RAGTruth train, NLI scoring of a calibration slice of
train and of the full test split, judge meta-evaluation with calibration, span F1, agreement, the
PPI coverage experiment, the conformal-risk-control check and judge throughput. NLI rows are cached
under ``$FAITHGATE_DATA_DIR/nli-cache`` so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from bench.machine import git_state, machine_info
from faithgate.assets import NLI_LICENSE, NLI_REPO, NLI_REVISION, RAGTRUTH_COMMIT, RAGTRUTH_LICENSE
from faithgate.bootstrap import stratified_bootstrap
from faithgate.calibration import (
    METHODS,
    Calibrator,
    brier_score,
    expected_calibration_error,
    make_calibrator,
    maximum_calibration_error,
    reliability_bins,
)
from faithgate.conformal import Risk, crc_threshold, realised_risk
from faithgate.judges import AlwaysHallucinatedJudge, NLIJudge, OverlapJudge
from faithgate.metrics import (
    auroc,
    best_f1_threshold,
    cohen_kappa,
    krippendorff_alpha,
    precision_recall_f1,
)
from faithgate.nli import EntailmentModel, load_default_nli
from faithgate.paths import nli_cache_dir
from faithgate.ppi import classical_mean, interval, judge_only_mean, ppi_mean, ppi_plusplus_mean
from faithgate.ragtruth import TASK_TYPES, load_ragtruth, stratified_sample
from faithgate.records import Record, Scored
from faithgate.report import reliability_svg
from faithgate.spans import SpanScores, localise, span_scores
from faithgate.summac import ChunkConfig, ConvAggregator, ConvMode, NLIMatrix, layout
from faithgate.text import vocabulary

_START = time.perf_counter()


def log(message: str) -> None:
    print(f"[{time.perf_counter() - _START:7.1f}s] {message}", file=sys.stderr, flush=True)


@dataclass(frozen=True)
class Profile:
    name: str
    dev_per_task: int
    calib_per_task: int
    test_per_task: int | None
    grid: tuple[ChunkConfig, ...]
    ppi_ns: tuple[int, ...]
    ppi_reps: int
    crc_reps: int
    crc_n: int
    bootstrap: int
    throughput_n: int
    seed: int = 7


PROFILES = {
    "full": Profile(
        name="full",
        dev_per_task=50,
        calib_per_task=400,
        test_per_task=None,
        grid=(
            ChunkConfig("sentence", 25),
            ChunkConfig("sentence", 60),
            ChunkConfig("sentence", 150),
            ChunkConfig("claim", 25),
        ),
        ppi_ns=(50, 100, 200, 500),
        ppi_reps=1000,
        crc_reps=500,
        crc_n=500,
        bootstrap=1000,
        throughput_n=150,
    ),
    "smoke": Profile(
        name="smoke",
        dev_per_task=4,
        calib_per_task=12,
        test_per_task=12,
        grid=(ChunkConfig("sentence", 25), ChunkConfig("sentence", 60)),
        ppi_ns=(12,),
        ppi_reps=100,
        crc_reps=50,
        crc_n=18,
        bootstrap=50,
        throughput_n=6,
    ),
}

NLI_JUDGES = ("nli-zs-mean", "nli-zs-min", "nli-conv-mean", "nli-conv-max")


class MatrixCache:
    """NLI rows per response, appended to a JSONL file keyed by id and a digest of the texts."""

    def __init__(self, model: EntailmentModel, config: ChunkConfig) -> None:
        self.config = config
        self.judge = NLIJudge(model, config, "zs-min")
        self.path = nli_cache_dir() / NLI_REVISION[:12] / f"{config.key()}.jsonl"
        self.entries: dict[str, tuple[str, list[list[float]]]] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    self.entries[row["id"]] = (row["digest"], row["probs"])

    @staticmethod
    def digest(record: Record) -> str:
        return hashlib.sha1(f"{record.source}\0{record.response}".encode()).hexdigest()

    def matrices(self, records: Sequence[Record], label: str) -> list[NLIMatrix]:
        missing = [r for r in records if self.entries.get(r.id, ("",))[0] != self.digest(r)]
        if missing:
            log(f"NLI {self.config.key()} on {label}: {len(missing)} of {len(records)} uncached")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            with self.path.open("a", encoding="utf-8") as handle:
                for start in range(0, len(missing), 64):
                    batch = missing[start : start + 64]
                    for record, matrix in zip(batch, self.judge.matrices(batch), strict=True):
                        probs = np.round(matrix.probs, 6).tolist()
                        self.entries[record.id] = (self.digest(record), probs)
                        handle.write(
                            json.dumps(
                                {"id": record.id, "digest": self.digest(record), "probs": probs}
                            )
                            + "\n"
                        )
                    handle.flush()
                    done = start + len(batch)
                    rate = done / (time.perf_counter() - started)
                    log(f"  {done}/{len(missing)} responses, {rate:.2f}/s")
        out = []
        for record in records:
            units, chunks = layout(record.source, record.response, self.config)
            probs = np.asarray(self.entries[record.id][1], dtype=float).reshape(
                len(units), len(chunks)
            )
            out.append(NLIMatrix(units, chunks, probs))
        return out


def _labels(records: Sequence[Record]) -> np.ndarray:
    return np.array([r.label for r in records], dtype=int)


def _prf(y: np.ndarray, decided: np.ndarray) -> dict[str, float]:
    m = precision_recall_f1(y, decided)
    return {"precision": m.precision, "recall": m.recall, "f1": m.f1}


def select_chunking(
    dev: list[Record], model: EntailmentModel, profile: Profile
) -> tuple[ChunkConfig, dict[str, object]]:
    y = _labels(dev)
    table = []
    for config in profile.grid:
        mats = MatrixCache(model, config).matrices(dev, "dev")
        row: dict[str, object] = {"config": config.key()}
        for aggregation in ("zs-mean", "zs-min"):
            scores = [
                s.score for s in NLIJudge(model, config, aggregation).from_matrices(dev, mats)
            ]
            row[f"auroc_{aggregation}"] = auroc(y, scores)
        row["pairs_per_response"] = float(np.mean([m.probs.size for m in mats]))
        table.append(row)
        log(f"dev {config.key()}: {row}")
    best_auroc = [
        max(float(str(row["auroc_zs-mean"])), float(str(row["auroc_zs-min"]))) for row in table
    ]
    chosen = profile.grid[int(np.argmax(best_auroc))]
    return chosen, {
        "dev_n": len(dev),
        "grid": table,
        "chosen": chosen.key(),
        "criterion": "max dev AUROC",
    }


def nli_outputs(
    model: EntailmentModel,
    config: ChunkConfig,
    calib: list[Record],
    test: list[Record],
    dev: list[Record],
    calib_m: list[NLIMatrix],
    test_m: list[NLIMatrix],
    dev_m: list[NLIMatrix],
) -> tuple[dict[str, dict[str, list[Scored]]], dict[str, object]]:
    """Scored rows of every NLI judge on calib, test and dev. Conv is trained on calib only."""
    y_calib = _labels(calib)
    out: dict[str, dict[str, list[Scored]]] = {}
    conv_info: dict[str, object] = {}
    for name in NLI_JUDGES:
        if name.startswith("nli-conv"):
            mode: ConvMode = "max" if name.endswith("max") else "mean"
            conv = ConvAggregator(n_bins=20, mode=mode).fit(calib_m, y_calib)
            conv_info[name] = {
                "n_train": len(calib),
                "weights": conv.to_params().weights,
                "bias": conv.bias,
            }
            judge = NLIJudge(model, config, conv)
        else:
            judge = NLIJudge(model, config, name.removeprefix("nli-"))
        out[name] = {
            "calib": judge.from_matrices(calib, calib_m),
            "test": judge.from_matrices(test, test_m),
            "dev": judge.from_matrices(dev, dev_m),
        }
    return out, conv_info


def evaluate_judge(
    cal_s: np.ndarray,
    cal_y: np.ndarray,
    test_s: np.ndarray,
    test_y: np.ndarray,
    tasks: np.ndarray,
    profile: Profile,
    calibrate: bool = True,
) -> dict[str, object]:
    raw = np.clip(test_s, 0.0, 1.0)
    threshold = best_f1_threshold(cal_y, cal_s)
    tuned = test_s >= threshold
    result: dict[str, object] = {
        "auroc": auroc(test_y, test_s),
        "auroc_ci95": stratified_bootstrap(
            lambda i: auroc(test_y[i], test_s[i]), test_y.size, tasks, profile.bootstrap, seed=1
        ).model_dump(),
        "uncalibrated_at_0.5": _prf(test_y, raw >= 0.5),
        "tuned_threshold": threshold,
        "at_tuned_threshold": _prf(test_y, tuned),
        "f1_at_tuned_threshold_ci95": stratified_bootstrap(
            lambda i: precision_recall_f1(test_y[i], tuned[i]).f1,
            test_y.size,
            tasks,
            profile.bootstrap,
            seed=2,
        ).model_dump(),
        "ece_raw": expected_calibration_error(raw, test_y),
        "mce_raw": maximum_calibration_error(raw, test_y),
        "brier_raw": brier_score(raw, test_y),
    }
    calibrated: dict[str, dict[str, object]] = {}
    fitted: dict[str, Calibrator] = {}
    if calibrate:
        for method in METHODS:
            if method == "none":
                continue
            calibrator = make_calibrator(method).fit(cal_s, cal_y)
            fitted[method] = calibrator
            probs = calibrator.predict(test_s)
            calibrated[method] = {
                "ece": expected_calibration_error(probs, test_y),
                "mce": maximum_calibration_error(probs, test_y),
                "brier": brier_score(probs, test_y),
                "at_0.5": _prf(test_y, probs >= 0.5),
            }
    result["calibrated"] = calibrated
    per_task: dict[str, dict[str, object]] = {}
    for task in TASK_TYPES:
        mask = tasks == task
        if not mask.any():
            continue
        y_t, s_t = test_y[mask], test_s[mask]
        entry: dict[str, object] = {
            "n": int(mask.sum()),
            "positive_rate": float(y_t.mean()),
            "auroc": auroc(y_t, s_t) if 0 < y_t.sum() < y_t.size else None,
            "f1_at_tuned_threshold": precision_recall_f1(y_t, tuned[mask]).f1,
            "ece_raw": expected_calibration_error(raw[mask], y_t),
        }
        if calibrate:
            entry["ece_isotonic_global"] = expected_calibration_error(
                fitted["isotonic"].predict(s_t), y_t
            )
            entry["f1_isotonic_global_at_0.5"] = precision_recall_f1(
                y_t, fitted["isotonic"].predict(s_t) >= 0.5
            ).f1
        per_task[task] = entry
    result["per_task"] = per_task
    return result


def per_task_calibration(
    cal_s: np.ndarray,
    cal_y: np.ndarray,
    cal_tasks: np.ndarray,
    test_s: np.ndarray,
    test_y: np.ndarray,
    tasks: np.ndarray,
) -> dict[str, object]:
    """Isotonic calibrators fitted separately per task type: base rates differ a lot by task."""
    probs = np.zeros_like(test_s)
    per_task = {}
    for task in TASK_TYPES:
        cal_mask, test_mask = cal_tasks == task, tasks == task
        if not cal_mask.any() or not test_mask.any():
            continue
        calibrator = make_calibrator("isotonic").fit(cal_s[cal_mask], cal_y[cal_mask])
        probs[test_mask] = calibrator.predict(test_s[test_mask])
        per_task[task] = {
            "ece_isotonic_per_task": expected_calibration_error(probs[test_mask], test_y[test_mask])
        }
    return {
        "ece": expected_calibration_error(probs, test_y),
        "brier": brier_score(probs, test_y),
        "at_0.5": _prf(test_y, probs >= 0.5),
        "per_task": per_task,
    }


def tune_span_threshold(
    rows: Sequence[Scored], records: Sequence[Record], refine: bool
) -> tuple[float, float]:
    best = (0.5, -1.0)
    for tau in np.round(np.linspace(0.05, 0.95, 19), 2):
        f1 = _span_eval(rows, records, float(tau), refine).f1
        if f1 > best[1]:
            best = (float(tau), f1)
    return best


@functools.lru_cache(maxsize=8192)
def _source_vocab(source: str) -> frozenset[str]:
    return frozenset(vocabulary(source))


def _span_eval(
    rows: Sequence[Scored], records: Sequence[Record], tau: float, refine: bool
) -> SpanScores:
    pairs = []
    for row, record in zip(rows, records, strict=True):
        vocab = _source_vocab(record.source) if refine else None
        pairs.append((localise(record.response, row.units, tau, vocab), record.spans))
    return span_scores(pairs)


def span_experiment(
    judged: dict[str, dict[str, list[Scored]]],
    calib: list[Record],
    test: list[Record],
    primary: str,
) -> dict[str, object]:
    out: dict[str, object] = {}
    variants = [
        (f"{primary} (whole unit)", primary, False),
        (f"{primary} (source-absent words in flagged units)", primary, True),
        ("overlap (unsupported word runs)", "overlap", True),
    ]
    for label, judge, refine in variants:
        tau, cal_f1 = tune_span_threshold(judged[judge]["calib"], calib, refine)
        scores = _span_eval(judged[judge]["test"], test, tau, refine)
        per_task = {}
        for task in TASK_TYPES:
            idx = [i for i, r in enumerate(test) if r.task_type == task]
            if idx:
                per_task[task] = _span_eval(
                    [judged[judge]["test"][i] for i in idx], [test[i] for i in idx], tau, refine
                ).f1
        out[label] = {
            "unit_threshold": tau,
            "calib_f1": cal_f1,
            **scores.model_dump(),
            "per_task_f1": per_task,
        }
    always = span_scores(([(0, len(r.response))], r.spans) for r in test)
    out["always-hallucinated (whole response)"] = always.model_dump()
    return out


def ppi_experiment(
    y_all: np.ndarray, judges: dict[str, np.ndarray], profile: Profile, alpha: float = 0.1
) -> dict[str, object]:
    """Repeatedly label a random n of the test pool and treat the rest as unlabelled."""
    rng = np.random.default_rng(profile.seed)
    theta = float(y_all.mean())
    pool = y_all.size
    out: dict[str, object] = {
        "pool_n": pool,
        "true_rate": theta,
        "alpha_two_sided": alpha,
        "alpha_upper": alpha / 2,
        "nominal_coverage_two_sided": 1 - alpha,
        "nominal_coverage_upper": 1 - alpha / 2,
        "reps": profile.ppi_reps,
        "by_judge": {},
    }
    stats: dict[str, dict[int, dict[str, dict[str, list[float]]]]] = {
        j: {n: {} for n in profile.ppi_ns} for j in judges
    }
    for n in profile.ppi_ns:
        for _ in range(profile.ppi_reps):
            perm = rng.permutation(pool)
            lab, unlab = perm[:n], perm[n:]
            y = y_all[lab]
            classical = classical_mean(y)
            for judge, f in judges.items():
                estimates = {
                    "classical": classical,
                    "ppi": ppi_mean(y, f[lab], f[unlab]),
                    "ppi++": ppi_plusplus_mean(y, f[lab], f[unlab]),
                    "judge-only": judge_only_mean(f),
                }
                for method, est in estimates.items():
                    two = interval(est, alpha)
                    upper = interval(est, alpha / 2, alternative="upper")
                    bucket = stats[judge][n].setdefault(
                        method,
                        {"cover": [], "width": [], "upper_cover": [], "upper": [], "lam": []},
                    )
                    bucket["cover"].append(
                        float(two.lower is not None and two.lower <= theta <= two.upper)
                    )
                    bucket["width"].append(two.width)
                    bucket["upper_cover"].append(float(theta <= upper.upper))
                    bucket["upper"].append(upper.upper)
                    if est.lam is not None:
                        bucket["lam"].append(est.lam)
        log(f"PPI n={n} done")
    by_judge: dict[str, object] = {}
    for judge in judges:
        rows = {}
        for n in profile.ppi_ns:
            rows[str(n)] = {
                method: {
                    "coverage": float(np.mean(b["cover"])),
                    "mean_width": float(np.mean(b["width"])),
                    "upper_coverage": float(np.mean(b["upper_cover"])),
                    "mean_upper_bound": float(np.mean(b["upper"])),
                    **({"mean_lambda": float(np.mean(b["lam"]))} if b["lam"] else {}),
                }
                for method, b in stats[judge][n].items()
            }
            classical_w = rows[str(n)]["classical"]["mean_width"]
            ppp_w = rows[str(n)]["ppi++"]["mean_width"]
            rows[str(n)]["classical_equivalent_labels_ppi++"] = float(
                n * (classical_w / ppp_w) ** 2
            )
        by_judge[judge] = rows
    out["by_judge"] = by_judge
    return out


def crc_experiment(
    cal_p: np.ndarray, cal_y: np.ndarray, test_p: np.ndarray, test_y: np.ndarray, profile: Profile
) -> dict[str, object]:
    rng = np.random.default_rng(profile.seed + 1)
    rows = []
    risks_to_check: tuple[Risk, ...] = ("slip", "miss")
    for risk in risks_to_check:
        for alpha in (0.05, 0.1, 0.2):
            fixed = crc_threshold(cal_p, cal_y, alpha, risk=risk)
            risks, flags = [], []
            for _ in range(profile.crc_reps):
                perm = rng.permutation(test_y.size)
                c, e = perm[: profile.crc_n], perm[profile.crc_n :]
                t = crc_threshold(test_p[c], test_y[c], alpha, risk=risk).threshold
                risks.append(realised_risk(test_p[e], test_y[e], t, risk))
                flags.append(float(np.mean(test_p[e] >= t)))
            rows.append(
                {
                    "risk": risk,
                    "alpha": alpha,
                    "calibrated_on_train": {
                        "threshold": fixed.threshold,
                        "test_risk": realised_risk(test_p, test_y, fixed.threshold, risk),
                        "test_flag_rate": float(np.mean(test_p >= fixed.threshold)),
                    },
                    "resampled_within_test": {
                        "n_calibration": profile.crc_n,
                        "n_evaluation": int(test_y.size - profile.crc_n),
                        "reps": profile.crc_reps,
                        "mean_risk": float(np.mean(risks)),
                        "sd_risk": float(np.std(risks)),
                        "mean_flag_rate": float(np.mean(flags)),
                    },
                }
            )
    return {"rows": rows}


def throughput(
    test: list[Record], model: EntailmentModel, config: ChunkConfig, profile: Profile
) -> dict[str, object]:
    sample = stratified_sample(test, max(1, profile.throughput_n // 3), seed=profile.seed + 2)
    judge = NLIJudge(model, config, "zs-min")
    judge.score(sample[:2])  # warm-up: session and tokenizer caches
    started = time.perf_counter()
    judge.score(sample)
    nli_seconds = time.perf_counter() - started
    pairs = sum(len(u) * len(c) for u, c in (layout(r.source, r.response, config) for r in sample))
    started = time.perf_counter()
    OverlapJudge().score(test)
    overlap_seconds = time.perf_counter() - started
    return {
        "nli": {
            "responses": len(sample),
            "seconds": nli_seconds,
            "responses_per_second": len(sample) / nli_seconds,
            "nli_pairs_per_second": pairs / nli_seconds,
            "mean_pairs_per_response": pairs / len(sample),
            "config": config.key(),
        },
        "overlap": {
            "responses": len(test),
            "seconds": overlap_seconds,
            "responses_per_second": len(test) / overlap_seconds,
        },
    }


def _reliability_figure(raw: np.ndarray, calibrated: np.ndarray, y: np.ndarray, path: Path) -> None:
    svg = reliability_svg(
        [
            ("raw", reliability_bins(raw, y, n_bins=10), "raw"),
            ("isotonic", reliability_bins(calibrated, y, n_bins=10), "cal"),
        ]
    )
    style = (
        "<style>.frame{fill:none;stroke:#bbb}.diag{stroke:#888;stroke-dasharray:4 4}"
        ".tick,.axis{fill:#666;font:11px sans-serif}.line{fill:none;stroke-width:1.5}"
        ".line.raw{stroke:#999}.line.cal{stroke:#2f6f9f}.dot.raw{fill:#999}.dot.cal{fill:#2f6f9f}</style>"
    )
    path.write_text(svg.replace(">", ">" + style, 1), encoding="utf-8")


def run(profile: Profile, out_path: Path, argv: Sequence[str]) -> dict[str, object]:
    timings: dict[str, float] = {}
    stage = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal stage
        timings[name] = round(time.perf_counter() - stage, 1)
        stage = time.perf_counter()

    train = load_ragtruth("train")
    test_all = load_ragtruth("test")
    dev = stratified_sample(train, profile.dev_per_task, seed=profile.seed)
    calib = stratified_sample(
        train, profile.calib_per_task, seed=profile.seed + 10, exclude={r.id for r in dev}
    )
    test = (
        test_all
        if profile.test_per_task is None
        else stratified_sample(test_all, profile.test_per_task, seed=profile.seed + 20)
    )
    log(f"dev {len(dev)}, calib {len(calib)}, test {len(test)}")
    model = load_default_nli()
    lap("load")

    config, chunk_selection = select_chunking(dev, model, profile)
    lap("chunk_selection")
    cache = MatrixCache(model, config)
    calib_m = cache.matrices(calib, "calib")
    test_m = cache.matrices(test, "test")
    dev_m = cache.matrices(dev, "dev")
    lap("nli_scoring")

    judged, conv_info = nli_outputs(model, config, calib, test, dev, calib_m, test_m, dev_m)
    for name, judge in (
        ("overlap", OverlapJudge()),
        ("always-hallucinated", AlwaysHallucinatedJudge()),
    ):
        judged[name] = {
            "calib": judge.score(calib),
            "test": judge.score(test),
            "dev": judge.score(dev),
        }
    y_dev, y_cal, y_test = _labels(dev), _labels(calib), _labels(test)
    dev_auroc = {name: auroc(y_dev, [s.score for s in judged[name]["dev"]]) for name in NLI_JUDGES}
    primary = max(dev_auroc, key=lambda k: dev_auroc[k])
    log(f"dev AUROC {dev_auroc}; primary judge {primary}")

    tasks = np.array([r.task_type for r in test])
    cal_tasks = np.array([r.task_type for r in calib])
    scores = {
        name: {split: np.array([s.score for s in rows]) for split, rows in d.items()}
        for name, d in judged.items()
    }
    meta: dict[str, dict[str, object]] = {}
    thresholds: dict[str, float] = {}
    for name in judged:
        thresholds[name] = best_f1_threshold(y_cal, scores[name]["calib"])
        meta[name] = evaluate_judge(
            scores[name]["calib"],
            y_cal,
            scores[name]["test"],
            y_test,
            tasks,
            profile,
            calibrate=name != "always-hallucinated",
        )
        if name != "always-hallucinated":
            meta[name]["isotonic_per_task"] = per_task_calibration(
                scores[name]["calib"], y_cal, cal_tasks, scores[name]["test"], y_test, tasks
            )
    lap("meta_eval")

    spans = span_experiment(judged, calib, test, primary)
    lap("spans")

    iso = {
        n: make_calibrator("isotonic").fit(scores[n]["calib"], y_cal) for n in (primary, "overlap")
    }
    decisions = {
        name: (scores[name]["test"] >= thresholds[name]).astype(int)
        for name in ("overlap", *NLI_JUDGES)
    }
    agreement = {
        "decision_rule": "score >= F1-optimal threshold chosen on the calibration split",
        "cohen_kappa_vs_human": {
            n: cohen_kappa(y_test.tolist(), d.tolist()) for n, d in decisions.items()
        },
        "krippendorff_alpha_human_and_judges": krippendorff_alpha(
            [y_test.tolist(), *(d.tolist() for d in decisions.values())]
        ),
        "krippendorff_alpha_judges_only": krippendorff_alpha(
            [d.tolist() for d in decisions.values()]
        ),
    }
    lap("agreement")

    ppi_judges = {
        f"{primary} (isotonic)": iso[primary].predict(scores[primary]["test"]),
        f"{primary} (raw score)": np.clip(scores[primary]["test"], 0, 1),
        "overlap (isotonic)": iso["overlap"].predict(scores["overlap"]["test"]),
    }
    ppi = ppi_experiment(y_test, ppi_judges, profile)
    lap("ppi")

    crc = crc_experiment(
        iso[primary].predict(scores[primary]["calib"]),
        y_cal,
        iso[primary].predict(scores[primary]["test"]),
        y_test,
        profile,
    )
    crc["judge"] = f"{primary} (isotonic)"
    lap("crc")

    speed = throughput(test, model, config, profile)
    lap("throughput")

    timings["total"] = round(time.perf_counter() - _START, 1)
    figure = out_path.with_name(out_path.stem + "-reliability.svg")
    _reliability_figure(
        np.clip(scores[primary]["test"], 0, 1),
        iso[primary].predict(scores[primary]["test"]),
        y_test,
        figure,
    )

    return {
        "profile": profile.name,
        "profile_settings": {k: v for k, v in asdict(profile).items() if k != "grid"},
        "date": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "command": "uv run python -m bench.run " + " ".join(argv),
        **git_state(),
        "machine": machine_info(),
        "dataset": {
            "name": "RAGTruth",
            "commit": RAGTRUTH_COMMIT,
            "licence": RAGTRUTH_LICENSE,
            "dev_n": len(dev),
            "calib_n": len(calib),
            "calib_positive_rate": float(y_cal.mean()),
            "test_n": len(test),
            "test_positive_rate": float(y_test.mean()),
        },
        "model": {"repo": NLI_REPO, "revision": NLI_REVISION, "licence": NLI_LICENSE},
        "timings_seconds": timings,
        "chunk_selection": chunk_selection,
        "dev_auroc": dev_auroc,
        "primary_judge": primary,
        "conv": conv_info,
        "meta_eval": meta,
        "spans": spans,
        "agreement": agreement,
        "ppi": ppi,
        "crc": crc,
        "throughput": speed,
        "llm_judge": {
            "status": "not run",
            "reason": "needs an OpenAI-compatible endpoint; the adapter and the position-swap "
            "check are covered by tests with a scripted provider only",
        },
        "reliability_figure": figure.name,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), default="full")
    parser.add_argument(
        "--out", type=Path, default=None, help="results JSON (default bench/results/<profile>.json)"
    )
    args = parser.parse_args(args_list)
    profile = PROFILES[args.profile]
    out_path = args.out or Path(__file__).resolve().parent / "results" / f"{profile.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = run(profile, out_path, args_list)
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

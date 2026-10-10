"""Command line: fetch, export-ragtruth, score, train-conv, calibrate, report."""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from faithgate import __version__
from faithgate.artifacts import CalibrationArtifact, fit_artifact
from faithgate.calibration import METHODS
from faithgate.judges import AlwaysHallucinatedJudge, Judge, LLMJudge, NLIJudge, OverlapJudge
from faithgate.records import Record, Scored, read_jsonl, write_jsonl
from faithgate.report import build_report, write_report
from faithgate.summac import ChunkConfig, ConvAggregator, ConvParams, NLIMatrix
from faithgate.text import Segment

JUDGES = ("nli-zs-min", "nli-zs-mean", "nli-conv", "overlap", "always-hallucinated", "llm")


def _cmd_fetch(args: argparse.Namespace) -> int:
    from faithgate.fetch import fetch_nli_model, fetch_ragtruth

    if args.what in ("ragtruth", "all"):
        fetch_ragtruth()
    if args.what in ("model", "all"):
        fetch_nli_model()
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from faithgate.ragtruth import load_ragtruth, stratified_sample

    records = load_ragtruth(args.split)
    if args.per_task:
        records = stratified_sample(records, args.per_task, seed=args.seed)
    if args.keep_labels is not None:
        rng = random.Random(args.seed)
        keep = set(rng.sample(range(len(records)), min(args.keep_labels, len(records))))
        records = [
            r if i in keep else r.model_copy(update={"label": None, "spans": []})
            for i, r in enumerate(records)
        ]
    count = write_jsonl(args.output, records)
    print(f"wrote {count} records to {args.output}")
    return 0


def _make_judge(args: argparse.Namespace) -> Judge:
    if args.judge == "overlap":
        return OverlapJudge(
            span_threshold=args.span_threshold if args.span_threshold is not None else 0.25
        )
    if args.judge == "always-hallucinated":
        return AlwaysHallucinatedJudge()
    if args.judge == "llm":
        from faithgate.llm import OpenAICompatibleProvider

        if not args.llm_base_url or not args.llm_model:
            raise SystemExit("--judge llm needs --llm-base-url and --llm-model")
        return LLMJudge(OpenAICompatibleProvider(args.llm_base_url, args.llm_model))
    from faithgate.nli import load_default_nli

    config = ChunkConfig(unit=args.unit, chunk_words=args.chunk_words, overlap=args.overlap)
    aggregation: str | ConvAggregator
    if args.judge == "nli-conv":
        if not args.conv:
            raise SystemExit("--judge nli-conv needs --conv (see `faithgate train-conv`)")
        aggregation = ConvAggregator.from_params(
            ConvParams.model_validate_json(args.conv.read_text())
        )
    else:
        aggregation = args.judge.removeprefix("nli-")
    return NLIJudge(
        load_default_nli(threads=args.threads),
        config,
        aggregation,
        span_threshold=args.span_threshold if args.span_threshold is not None else 0.5,
        keep_matrix=args.keep_matrix,
    )


def _cmd_score(args: argparse.Namespace) -> int:
    records = read_jsonl(args.input, Record)
    judge = _make_judge(args)
    rows = judge.score(records)
    rows = [
        row.model_copy(update={"response": rec.response})
        for row, rec in zip(rows, records, strict=True)
    ]
    write_jsonl(args.output, rows)
    print(f"scored {len(rows)} responses with {judge.name} -> {args.output}")
    return 0


def _matrix_from_row(row: Scored) -> NLIMatrix:
    if row.unit_probs is None:
        raise SystemExit(f"{row.id} has no unit_probs; score with an NLI judge and --keep-matrix")
    probs = np.array(row.unit_probs, dtype=float).reshape(len(row.units), -1)
    units = [Segment(u.start, u.end, "") for u in row.units]
    return NLIMatrix(units, [Segment(0, 0, "")] * probs.shape[1], probs)


def _cmd_train_conv(args: argparse.Namespace) -> int:
    rows = [r for r in read_jsonl(args.scores, Scored) if r.label is not None]
    if not rows:
        raise SystemExit("no labelled rows to train on")
    conv = ConvAggregator(n_bins=args.bins, mode=args.mode, l2=args.l2)
    conv.fit([_matrix_from_row(r) for r in rows], [int(r.label or 0) for r in rows])
    args.output.write_text(conv.to_params().model_dump_json(indent=2), encoding="utf-8")
    print(f"trained SummaC-Conv ({args.mode}) on {len(rows)} responses -> {args.output}")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.scores, Scored)
    artifact = fit_artifact(rows, method=args.method, alpha=args.alpha, risk=args.risk)
    artifact.save(args.output)
    s = artifact.summary
    held_out = (
        f"{s.ece_calibrated_cv:.3f} with {s.cv_folds}-fold cross-validation"
        if s.ece_calibrated_cv is not None
        else "not cross-validated (too few rows of one class)"
    )
    print(
        f"{artifact.calibrator.method} on {s.n} labelled responses: ECE raw {s.ece_raw:.3f}, "
        f"calibrated {held_out}, {s.ece_calibrated_in_sample:.3f} in-sample; "
        f"CRC {artifact.crc.risk} threshold {artifact.crc.threshold:.3f} "
        f"(alpha {artifact.crc.alpha}) -> {args.output}"
    )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.scores, Scored)
    artifact = CalibrationArtifact.load(args.calibration) if args.calibration else None
    report = build_report(rows, artifact, confidence=args.confidence)
    json_path, html_path = write_report(report, args.out_dir)
    for iv in report.rate.intervals:
        print(f"{iv.method:>11}: {iv.estimate:.3f} [{iv.lower:.3f}, {iv.upper:.3f}]")
    print(f"wrote {json_path} and {html_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="faithgate", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download RAGTruth and/or the NLI model, verifying sha256")
    p.add_argument("what", choices=("ragtruth", "model", "all"))
    p.set_defaults(func=_cmd_fetch)

    p = sub.add_parser("export-ragtruth", help="write a RAGTruth split as input JSONL")
    p.add_argument("--split", choices=("train", "test"), default="test")
    p.add_argument("--per-task", type=int, default=None, help="sample this many per task type")
    p.add_argument(
        "--keep-labels", type=int, default=None, help="keep labels on a random subset only"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.set_defaults(func=_cmd_export)

    p = sub.add_parser("score", help="score responses with a judge")
    p.add_argument("input", type=Path, help="JSONL with id, source, response[, label, task_type]")
    p.add_argument("-o", "--output", type=Path, required=True)
    # Defaults follow the benchmark's dev slice (bench/results/full.json, chunk_selection):
    # claim units against 25-word chunks, and zs-mean as the stronger zero-shot aggregation there.
    p.add_argument("--judge", choices=JUDGES, default="nli-zs-mean")
    p.add_argument("--unit", choices=("sentence", "claim"), default="claim")
    p.add_argument("--chunk-words", type=int, default=25)
    p.add_argument("--overlap", type=int, default=1)
    p.add_argument("--conv", type=Path, default=None, help="trained SummaC-Conv parameters")
    p.add_argument("--span-threshold", type=float, default=None)
    p.add_argument("--keep-matrix", action="store_true", help="store NLI rows for train-conv")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument(
        "--llm-base-url",
        default=None,
        help="OpenAI-compatible endpoint, e.g. http://localhost:11434/v1",
    )
    p.add_argument("--llm-model", default=None)
    p.set_defaults(func=_cmd_score)

    p = sub.add_parser("train-conv", help="fit SummaC-Conv on labelled NLI scores")
    p.add_argument("scores", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--mode", choices=("mean", "max"), default="max")
    p.add_argument("--bins", type=int, default=20)
    p.add_argument("--l2", type=float, default=1e-3)
    p.set_defaults(func=_cmd_train_conv)

    p = sub.add_parser("calibrate", help="fit a calibrator and a CRC flag threshold")
    p.add_argument("scores", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--method", choices=METHODS, default="isotonic")
    p.add_argument("--alpha", type=float, default=0.1, help="CRC risk level")
    p.add_argument("--risk", choices=("slip", "miss"), default="slip")
    p.set_defaults(func=_cmd_calibrate)

    p = sub.add_parser("report", help="write report.json and report.html")
    p.add_argument("scores", type=Path)
    p.add_argument("--calibration", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("report"))
    p.add_argument("--confidence", type=float, default=0.95)
    p.set_defaults(func=_cmd_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ValueError as exc:
        raise SystemExit(f"faithgate {args.command}: {exc}") from exc


if __name__ == "__main__":
    sys.exit(main())

"""Load RAGTruth (response.jsonl + source_info.jsonl) as :class:`Record` objects."""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from faithgate.paths import ragtruth_dir
from faithgate.records import Record

TASK_TYPES = ("QA", "Summary", "Data2txt")


def _humanise_key(key: str) -> str:
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", key).replace("_", " ")
    return spaced.strip()


def _humanise_value(value: Any) -> str | None:
    if value is None or value == "None":
        return None
    if value is True or value == "True":
        return "yes"
    if value is False or value == "False":
        return "no"
    if isinstance(value, str):
        # Yelp stores some strings as Python literals, e.g. "u'free'".
        return re.sub(r"^u?'(.*)'$", r"\1", value)
    return str(value)


def _clock(value: str) -> str:
    """'16:30' -> '4:30 pm', '9:0' -> '9 am' (Yelp drops the trailing zero of minutes)."""
    hour_text, _, minute_text = value.partition(":")
    hour, minute = int(hour_text), int(minute_text or 0)
    suffix = "am" if hour % 24 < 12 else "pm"
    shown = hour % 12 or 12
    return f"{shown}:{minute:02d} {suffix}" if minute else f"{shown} {suffix}"


def _opening_hours(hours: dict[str, Any]) -> str:
    parts = []
    for day, span in hours.items():
        opens, _, closes = str(span).partition("-")
        try:
            parts.append(f"{day} {_clock(opens)} to {_clock(closes)}")
        except ValueError:
            parts.append(f"{day} {span}")
    return "Opening hours: " + "; ".join(parts) + "."


def linearise_business(data: dict[str, Any]) -> str:
    """Turn a Yelp business record into short sentences a small NLI model can read.

    The head fields become "<name> is located at <address>, <city>, <state>.", times use a
    12-hour clock, and each attribute gets its own sentence. Attributes with a null value are
    dropped rather than rendered, so a response that asserts them is not entailed by the source.
    """
    name = str(data.get("name") or "The business")
    place = ", ".join(str(data[k]) for k in ("address", "city", "state") if data.get(k))
    lines: list[str] = [f"{name} is located at {place}." if place else f"{name}."]
    for key, value in data.items():
        if key in ("name", "address", "city", "state", "review_info"):
            continue
        if key == "hours" and isinstance(value, dict):
            lines.append(_opening_hours(value))
        elif key == "categories" and value:
            lines.append(f"{name} is listed under: {value}.")
        elif key == "business_stars" and value is not None:
            lines.append(f"{name} has an average rating of {value} stars.")
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    inner = [
                        f"{_humanise_key(k)} {v}"
                        for k, raw in sub_value.items()
                        if (v := _humanise_value(raw)) is not None
                    ]
                    if inner:
                        lines.append(f"{_humanise_key(sub_key)}: {', '.join(inner)}.")
                elif (text := _humanise_value(sub_value)) is not None:
                    lines.append(f"{_humanise_key(sub_key)}: {text}.")
        elif (text := _humanise_value(value)) is not None:
            lines.append(f"{_humanise_key(key)}: {text}.")
    for i, review in enumerate(data.get("review_info") or [], start=1):
        stars = review.get("review_stars")
        date = review.get("review_date")
        body = " ".join(str(review.get("review_text", "")).split())
        lines.append(f"Review {i} ({stars} stars, {date}): {body}")
    return "\n".join(lines)


def source_text(entry: dict[str, Any]) -> str:
    info = entry["source_info"]
    task = entry["task_type"]
    if task == "Summary":
        return str(info)
    if task == "QA":
        return f"Question: {info['question']}\n{info['passages'].strip()}"
    if task == "Data2txt":
        return linearise_business(info if isinstance(info, dict) else json.loads(info))
    raise ValueError(f"unknown RAGTruth task type {task!r}")


def load_ragtruth(split: str | None = None, root: Path | None = None) -> list[Record]:
    """All RAGTruth responses of ``split`` ('train', 'test' or None for both).

    A response counts as hallucinated when annotators marked at least one span, including spans
    flagged ``implicit_true`` or ``due_to_null``; RAGTruth's own response-level labels do the same.
    """
    folder = root or ragtruth_dir()
    response_path, source_path = folder / "response.jsonl", folder / "source_info.jsonl"
    if not response_path.exists() or not source_path.exists():
        raise FileNotFoundError(f"RAGTruth not found in {folder}; run `faithgate fetch ragtruth`")
    sources: dict[str, dict[str, Any]] = {}
    with source_path.open(encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            sources[entry["source_id"]] = entry
    records: list[Record] = []
    with response_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if split is not None and row["split"] != split:
                continue
            entry = sources[row["source_id"]]
            spans = sorted((int(lab["start"]), int(lab["end"])) for lab in row["labels"])
            records.append(
                Record(
                    id=f"ragtruth-{row['id']}",
                    source=source_text(entry),
                    response=row["response"],
                    task_type=entry["task_type"],
                    label=int(bool(row["labels"])),
                    spans=spans,
                    meta={
                        "model": row["model"],
                        "source_id": row["source_id"],
                        "split": row["split"],
                        "quality": row["quality"],
                    },
                )
            )
    return records


def stratified_sample(
    records: Sequence[Record], per_task: int, seed: int, exclude: set[str] | None = None
) -> list[Record]:
    """``per_task`` records from each task type, drawn uniformly without replacement."""
    rng = random.Random(seed)
    by_task: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        if exclude and record.id in exclude:
            continue
        by_task[record.task_type or ""].append(record)
    sample: list[Record] = []
    for task in sorted(by_task):
        pool = by_task[task]
        sample.extend(rng.sample(pool, min(per_task, len(pool))))
    return sample

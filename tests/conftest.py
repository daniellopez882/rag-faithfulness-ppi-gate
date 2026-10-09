from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.fakes import LexicalEntailment

pytest_plugins = ["pytester"]

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def lexical_nli() -> LexicalEntailment:
    return LexicalEntailment()


@pytest.fixture
def ragtruth_fixture(tmp_path: Path) -> Path:
    """A three-response RAGTruth folder in the original file layout."""
    folder = tmp_path / "ragtruth"
    folder.mkdir()
    sources = [
        {
            "source_id": "s1",
            "task_type": "QA",
            "source": "MARCO",
            "source_info": {
                "question": "when does it open",
                "passages": "passage 1: It opens at 9 am.\n\n",
            },
            "prompt": "...",
        },
        {
            "source_id": "s2",
            "task_type": "Data2txt",
            "source": "Yelp",
            "source_info": {
                "name": "Cafe Uno",
                "address": "1 Main St",
                "city": "Springfield",
                "state": "IL",
                "hours": {"Monday": "16:30-21:0"},
                "attributes": {
                    "WiFi": "u'free'",
                    "BusinessParking": None,
                    "Ambience": {"casual": True, "romantic": None},
                },
                "business_stars": 4.5,
                "review_info": [
                    {
                        "review_stars": 5.0,
                        "review_date": "2022-01-01",
                        "review_text": "Great  coffee.",
                    }
                ],
            },
            "prompt": "...",
        },
    ]
    responses = [
        {
            "id": "0",
            "source_id": "s1",
            "model": "m",
            "temperature": 0.7,
            "labels": [],
            "split": "test",
            "quality": "good",
            "response": "It opens at 9 am.",
        },
        {
            "id": "1",
            "source_id": "s1",
            "model": "m",
            "temperature": 0.7,
            "split": "test",
            "quality": "good",
            "labels": [
                {
                    "start": 12,
                    "end": 17,
                    "text": "10 am",
                    "meta": "",
                    "label_type": "Evident Conflict",
                    "implicit_true": False,
                    "due_to_null": False,
                }
            ],
            "response": "It opens at 10 am.",
        },
        {
            "id": "2",
            "source_id": "s2",
            "model": "m",
            "temperature": 0.7,
            "labels": [],
            "split": "train",
            "quality": "good",
            "response": "Cafe Uno offers free Wi-Fi.",
        },
    ]
    (folder / "source_info.jsonl").write_text(
        "\n".join(json.dumps(s) for s in sources) + "\n", encoding="utf-8"
    )
    (folder / "response.jsonl").write_text(
        "\n".join(json.dumps(r) for r in responses) + "\n", encoding="utf-8"
    )
    return folder

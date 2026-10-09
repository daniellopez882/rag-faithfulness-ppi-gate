from __future__ import annotations

from pathlib import Path

import pytest

from faithgate.ragtruth import linearise_business, load_ragtruth, stratified_sample


def test_loader_builds_records_with_labels(ragtruth_fixture: Path) -> None:
    test = load_ragtruth("test", root=ragtruth_fixture)
    assert [r.id for r in test] == ["ragtruth-0", "ragtruth-1"]
    assert [r.label for r in test] == [0, 1]
    assert test[1].spans == [(12, 17)]
    assert test[1].response[12:17] == "10 am"
    assert test[0].source.startswith("Question: when does it open\npassage 1:")
    assert test[0].meta["split"] == "test"
    assert len(load_ragtruth(None, root=ragtruth_fixture)) == 3


def test_missing_folder_explains_how_to_fetch(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="faithgate fetch ragtruth"):
        load_ragtruth("test", root=tmp_path)


def test_business_linearisation(ragtruth_fixture: Path) -> None:
    [record] = load_ragtruth("train", root=ragtruth_fixture)
    text = record.source
    assert text.splitlines()[0] == "Cafe Uno is located at 1 Main St, Springfield, IL."
    assert "Opening hours: Monday 4:30 pm to 9 pm." in text
    assert "Wi Fi: free." in text
    assert "Ambience: casual yes." in text
    assert "Parking" not in text  # null attributes are dropped, not rendered
    assert "Review 1 (5.0 stars, 2022-01-01): Great coffee." in text


def test_linearisation_handles_midnight_and_noon() -> None:
    text = linearise_business({"name": "X", "hours": {"Friday": "0:0-12:0"}})
    assert "Friday 12 am to 12 pm" in text


def test_stratified_sample_is_balanced_and_seeded(ragtruth_fixture: Path) -> None:
    records = load_ragtruth(None, root=ragtruth_fixture)
    sample = stratified_sample(records, per_task=1, seed=3)
    assert sorted(r.task_type or "" for r in sample) == ["Data2txt", "QA"]
    assert sample == stratified_sample(records, per_task=1, seed=3)
    rest = stratified_sample(records, per_task=5, seed=3, exclude={r.id for r in sample})
    assert {r.id for r in rest}.isdisjoint({r.id for r in sample})

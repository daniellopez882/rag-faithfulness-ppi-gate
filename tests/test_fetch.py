from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from faithgate import fetch
from faithgate.assets import NLI_FILES, RAGTRUTH_FILES, DatasetFile
from faithgate.paths import DATA_DIR_ENV, data_dir


def test_pinned_hashes_are_well_formed() -> None:
    for item in (*RAGTRUTH_FILES, *NLI_FILES):
        assert len(item.sha256) == 64
        int(item.sha256, 16)
        assert item.size > 0


def test_data_dir_comes_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    assert data_dir() == tmp_path
    monkeypatch.delenv(DATA_DIR_ENV)
    assert data_dir().name == "rag-faithfulness-ppi-gate"


def test_download_verifies_checksum(tmp_path: Path) -> None:
    payload = tmp_path / "payload.txt"
    payload.write_bytes(b"hello")
    good = hashlib.sha256(b"hello").hexdigest()
    dest = tmp_path / "out" / "copy.txt"
    fetch._download(payload.as_uri(), dest, good)
    assert dest.read_bytes() == b"hello"

    bad_dest = tmp_path / "out" / "bad.txt"
    with pytest.raises(fetch.ChecksumError):
        fetch._download(payload.as_uri(), bad_dest, "0" * 64)
    assert not bad_dest.exists()
    assert list((tmp_path / "out").glob("*.part")) == []


def test_fetch_ragtruth_skips_verified_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "remote.jsonl"
    source.write_bytes(b'{"a": 1}\n')
    item = DatasetFile(
        name="response.jsonl",
        url=source.as_uri(),
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        size=source.stat().st_size,
    )
    monkeypatch.setattr(fetch, "RAGTRUTH_FILES", (item,))
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "data"))
    logs: list[str] = []
    fetch.fetch_ragtruth(log=logs.append)
    fetch.fetch_ragtruth(log=logs.append)
    assert logs[0].startswith("download")
    assert logs[-1].startswith("ok")

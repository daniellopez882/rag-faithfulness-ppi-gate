"""Download RAGTruth and the NLI model; anything whose sha256 differs from the pin is refused."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path

from faithgate.assets import NLI_FILES, NLI_REPO, NLI_REVISION, RAGTRUTH_FILES, DatasetFile
from faithgate.paths import ragtruth_dir


class ChecksumError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, dest: Path, expected_sha256: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=dest.name, suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=120) as resp:
            shutil.copyfileobj(resp, out, length=1 << 20)
        got = sha256_of(tmp)
        if got != expected_sha256:
            raise ChecksumError(f"{url}: sha256 {got} does not match pinned {expected_sha256}")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def fetch_ragtruth(log: Callable[[str], None] = print) -> Path:
    """Fetch RAGTruth's two JSONL files into ``$FAITHGATE_DATA_DIR/ragtruth``."""
    target = ragtruth_dir()
    for item in RAGTRUTH_FILES:
        dest = target / item.name
        if dest.exists() and _matches(dest, item):
            log(f"ok        {dest}")
            continue
        log(f"download  {item.url}")
        _download(item.url, dest, item.sha256)
        log(f"verified  {dest}")
    return target


def _matches(path: Path, item: DatasetFile) -> bool:
    return path.stat().st_size == item.size and sha256_of(path) == item.sha256


def fetch_nli_model(log: Callable[[str], None] = print) -> Path:
    """Fetch the pinned ONNX NLI model into the Hugging Face cache (``$HF_HOME``)."""
    from huggingface_hub import hf_hub_download

    for item in NLI_FILES:
        local = Path(hf_hub_download(NLI_REPO, item.filename, revision=NLI_REVISION))
        got = sha256_of(local)
        if got != item.sha256:
            raise ChecksumError(f"{NLI_REPO}/{item.filename}: sha256 {got} != pinned {item.sha256}")
        log(f"verified  {local}")
    folder = locate_nli_model()
    if folder is None:
        raise RuntimeError(f"{NLI_REPO} was downloaded but is missing from the Hugging Face cache")
    return folder


def locate_nli_model() -> Path | None:
    """The cached snapshot folder of the pinned NLI model, or None if it was never fetched."""
    from huggingface_hub import try_to_load_from_cache

    found = [
        try_to_load_from_cache(NLI_REPO, item.filename, revision=NLI_REVISION) for item in NLI_FILES
    ]
    if not all(isinstance(path, str) for path in found):
        return None
    config = next(
        Path(p) for p, f in zip(found, NLI_FILES, strict=True) if f.filename == "config.json"
    )
    return config.parent

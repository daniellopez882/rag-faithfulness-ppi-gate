"""Where downloaded data lives. The location comes from the environment, never from the repo."""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR_ENV = "FAITHGATE_DATA_DIR"


def data_dir() -> Path:
    """Root for downloaded datasets and benchmark caches (``$FAITHGATE_DATA_DIR``)."""
    configured = os.environ.get(DATA_DIR_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "rag-faithfulness-ppi-gate"


def ragtruth_dir() -> Path:
    return data_dir() / "ragtruth"


def nli_cache_dir() -> Path:
    return data_dir() / "nli-cache"

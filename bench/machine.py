"""What the benchmark ran on: CPU, cores, RAM, OS, Python and package versions, git state."""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

PACKAGES = ("numpy", "scipy", "onnxruntime", "tokenizers", "pydantic", "jinja2", "huggingface-hub")


def _cpu_model() -> str:
    if sys.platform == "win32":
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _ram_bytes() -> int:
    if sys.platform == "win32":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong),
                ("avail_phys", ctypes.c_ulonglong),
                ("total_page", ctypes.c_ulonglong),
                ("avail_page", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("avail_virtual", ctypes.c_ulonglong),
                ("avail_extended", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined]
        return int(status.total_phys)
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def _git(*args: str) -> str:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    return result.stdout.strip()


def machine_info() -> dict[str, object]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {
        "cpu": _cpu_model(),
        "logical_cores": os.cpu_count(),
        "ram_gb": round(_ram_bytes() / 2**30, 1),
        "os": platform.platform(),
        "python": platform.python_version(),
        "accelerator": "none (CPU only)",
        "packages": versions,
    }


def git_state() -> dict[str, object]:
    return {"git_sha": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain"))}

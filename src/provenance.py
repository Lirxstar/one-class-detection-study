"""Identify published source used for the experiment records.

Prefer the Git revision of this repository, or hash source/configs when no
Git working tree is present. Internal report generation was removed.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from config import ROOT


def source_files(root: Path = ROOT):
    files = [*root.glob("src/*.py"), *root.glob("configs/*.yaml"), root / "requirements.lock", root / "pyproject.toml"]
    return sorted(p for p in files if p.is_file())


def source_hash(root: Path = ROOT) -> str:
    h = hashlib.sha256()
    for p in source_files(root):
        h.update(str(p.relative_to(root)).encode() + b"\0")
        h.update(p.read_bytes())
    return h.hexdigest()


def code_revision(root: Path = ROOT) -> str:
    if (root / ".git").exists():
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        except Exception:  # noqa: BLE001
            pass
    return "source-sha256:" + source_hash(root)

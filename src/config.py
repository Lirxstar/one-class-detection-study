"""Load the protocol configs and compute the config hash (PROTOCOL.md Section 10)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
CONFIG_FILES = ("conditions.yaml", "core.yaml", "e2_postprocessing.yaml", "e3_encoding.yaml")
E0_PENDING = "E0_PENDING"


def load(name: str) -> dict:
    """Load one config by stem, e.g. load("core")."""
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_all() -> dict[str, dict]:
    return {Path(f).stem: load(Path(f).stem) for f in CONFIG_FILES}


def config_hash() -> str:
    """sha256 of the four config files concatenated in sorted path order."""
    h = hashlib.sha256()
    for f in sorted(CONFIG_FILES):
        h.update((CONFIG_DIR / f).read_bytes())
    return h.hexdigest()


def pending_values(obj, path: str = "") -> list[str]:
    """Paths of all values still marked E0_PENDING."""
    out: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += pending_values(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += pending_values(v, f"{path}[{i}]")
    elif obj == E0_PENDING:
        out.append(path)
    return out

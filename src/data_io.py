"""Stored-byte access and decoding (PROTOCOL.md Sections 2.1, 2.3, 3).

Rows are addressed by `row_id = <split>/<parquet stem>/<row index in file>`.
Decoding: `Image.open(BytesIO(b))`, full load, then RGB. RGBA is accepted only
if alpha is 255 at every pixel (asserted per image); any other mode raises.
`ImageOps.exif_transpose` is never applied.
"""
from __future__ import annotations

import io
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from config import ROOT

DATA_DIR = ROOT / "data" / "tiny_genimage" / "data"


class AlphaNotOpaque(ValueError):
    """An RGBA image whose alpha channel is not 255 everywhere."""


def parse_row_id(row_id: str) -> tuple[str, str, int]:
    split, stem, idx = row_id.split("/")
    return split, stem, int(idx)


def read_bytes(row_ids, data_dir: Path = DATA_DIR) -> dict[str, bytes]:
    """Stored image bytes for each row_id, read file by file with pyarrow."""
    by_file: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for rid in row_ids:
        _, stem, idx = parse_row_id(rid)
        by_file[stem].append((idx, rid))
    out: dict[str, bytes] = {}
    for stem, items in sorted(by_file.items()):
        items.sort()
        col = pq.read_table(data_dir / f"{stem}.parquet", columns=["image"]).column("image")
        taken = col.take([i for i, _ in items]).to_pylist()
        for (_, rid), rec in zip(items, taken):
            out[rid] = rec["bytes"]
    return out


def alpha_min(b: bytes) -> int | None:
    """Minimum alpha value of an RGBA image, None for other modes."""
    im = Image.open(io.BytesIO(b))
    if im.mode != "RGBA":
        return None
    im.load()
    return int(np.asarray(im)[..., 3].min())


def decode_rgb(b: bytes) -> np.ndarray:
    """Decode stored bytes to a uint8 (H, W, 3) array per PROTOCOL.md Section 3."""
    im = Image.open(io.BytesIO(b))
    im.load()
    if im.mode == "RGB":
        arr = np.asarray(im)
    elif im.mode == "RGBA":
        a = np.asarray(im)
        if int(a[..., 3].min()) != 255:
            raise AlphaNotOpaque("RGBA image with alpha < 255")
        arr = np.asarray(im.convert("RGB"))
        if not np.array_equal(arr, a[..., :3]):  # convert must only drop alpha
            raise AssertionError("RGBA -> RGB conversion changed colour channels")
    else:
        raise ValueError(f"mode {im.mode} is not eligible")
    return np.ascontiguousarray(arr, dtype=np.uint8)

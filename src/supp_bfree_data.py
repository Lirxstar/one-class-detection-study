"""Supplementary experiment S-BF (PROTOCOL.md Section 7): file access for B-Free extended Synthbuster.

A sample is addressed by `row_id = bfree/<CLASS>/<source_id>`, CLASS in REAL, SD35, FLUX.
The source id is the file stem shared by the three folders (one RAISE-1k photograph and the
two images generated from its description). Decoding reuses `data_io.decode_rgb`, the core
decoder (Pillow, full load, RGB, no EXIF transpose); `decode_srgb` then converts ICC-tagged
real images to standard sRGB with ImageCms (colour rule, PROTOCOL.md Section 7).
"""
from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

import config
from config import ROOT
from data_io import decode_rgb

CLASSES = ["REAL", "SD35", "FLUX"]
LABEL = {"REAL": 0, "SD35": 1, "FLUX": 1}


def cfg() -> dict:
    return config.load("supp_bfree")


def supp_config_hash() -> str:
    """sha256 over the core four-file config hash and the S-BF config file."""
    h = hashlib.sha256(config.config_hash().encode())
    h.update((config.CONFIG_DIR / "supp_bfree.yaml").read_bytes())
    return h.hexdigest()


def class_dir(cls: str) -> Path:
    d = cfg()["data"]
    return ROOT / d["root"] / d["dirs"][cls]


def listing(cls: str) -> tuple[list[str], list[str]]:
    """(source ids matching the pattern, other entries) of one class folder, both sorted."""
    pat = re.compile(cfg()["data"]["id_pattern"])
    names = sorted(p.name for p in class_dir(cls).iterdir())
    ids = [n[:-4] for n in names if pat.match(n)]
    return ids, [n for n in names if not pat.match(n)]


def path(cls: str, sid: str) -> Path:
    return class_dir(cls) / f"{sid}.png"


def row_id(cls: str, sid: str) -> str:
    return f"bfree/{cls}/{sid}"


def parse(rid: str) -> tuple[str, str]:
    _, cls, sid = rid.split("/")
    return cls, sid


def decode(cls: str, sid: str) -> np.ndarray:
    """Stored pixel values, no colour management (the core decoder)."""
    return decode_rgb(path(cls, sid).read_bytes())


def icc_profile(cls: str, sid: str) -> bytes | None:
    return Image.open(path(cls, sid)).info.get("icc_profile") or None


def to_srgb(arr: np.ndarray, icc: bytes) -> np.ndarray:
    """ImageCms conversion from the embedded profile to standard sRGB (S-BF colour rule)."""
    c = cfg()["colour"]["primary"]
    if c["rendering_intent"] != "relative_colorimetric" or int(c["flags"]) != 0:
        raise ValueError("unsupported colour rule")
    src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
    out = ImageCms.profileToProfile(Image.fromarray(arr, "RGB"), src, ImageCms.createProfile("sRGB"),
                                    renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC, outputMode="RGB", flags=0)
    return np.ascontiguousarray(np.asarray(out, dtype=np.uint8))


def decode_srgb(cls: str, sid: str) -> np.ndarray:
    """Pixels used by every S-BF analysis: real images with an ICC profile converted to sRGB;
    untagged real images (assumed sRGB) and synthetic images unchanged."""
    arr = decode(cls, sid)
    if cls != "REAL":
        if icc_profile(cls, sid) is not None:
            raise ValueError(f"synthetic image with an ICC profile: {cls}/{sid}")
        return arr
    icc = icc_profile(cls, sid)
    return arr if icc is None else to_srgb(arr, icc)


def icc_description(icc: bytes | None) -> str:
    if not icc:
        return ""
    return ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(icc))).strip()

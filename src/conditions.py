"""Data conditions, post-processing and encodings on uint8 RGB arrays
(PROTOCOL.md Sections 3, 7 E2 and E3C).

All functions take and return uint8 (H, W, 3) arrays. JPEG encodes use Pillow
with the standard IJG tables, `optimize=False`, `progressive=False`, and the
modal real chroma subsampling unless stated otherwise.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

import config

VIEW = 224
SUBSAMPLING_CODE = {"4:4:4": 0, "4:2:2": 1, "4:2:0": 2}


def _cond_cfg() -> dict:
    return config.load("conditions")


def modal_subsampling() -> str:
    return str(_cond_cfg()["modal_real_subsampling"])


def offset_multiple() -> int:
    c = _cond_cfg()["conditions"]["d_ctrl"]
    m = int(c["offset_multiple"])
    rule = c["offset_multiple_rule"]
    assert m == int(rule.get(modal_subsampling(), rule["default"])), "offset multiple inconsistent with rule"
    return m


def jpeg_roundtrip(arr: np.ndarray, quality: int, subsampling: str | None = None) -> np.ndarray:
    """Encode once as JPEG with Pillow and decode back to RGB."""
    sub = modal_subsampling() if subsampling is None else subsampling
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(arr, dtype=np.uint8), "RGB").save(
        buf, format="JPEG", quality=int(quality), subsampling=SUBSAMPLING_CODE[sub], optimize=False, progressive=False)
    im = Image.open(io.BytesIO(buf.getvalue()))
    im.load()
    return np.asarray(im.convert("RGB"), dtype=np.uint8)


def jpeg_bytes(arr: np.ndarray, quality: int, subsampling: str | None = None) -> bytes:
    sub = modal_subsampling() if subsampling is None else subsampling
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(arr, dtype=np.uint8), "RGB").save(
        buf, format="JPEG", quality=int(quality), subsampling=SUBSAMPLING_CODE[sub], optimize=False, progressive=False)
    return buf.getvalue()


def png_roundtrip(arr: np.ndarray) -> np.ndarray:
    level = int(config.load("e3_encoding")["E3C"]["png"]["compress_level"])
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(arr, dtype=np.uint8), "RGB").save(buf, format="PNG", compress_level=level)
    im = Image.open(io.BytesIO(buf.getvalue()))
    im.load()
    return np.asarray(im.convert("RGB"), dtype=np.uint8)


def resize_short_256(arr: np.ndarray) -> np.ndarray:
    """Shorter side to 256, longer side floor(256 * long / short), Pillow bicubic."""
    h, w = arr.shape[:2]
    if w <= h:
        size = (256, (256 * h) // w)
    else:
        size = ((256 * w) // h, 256)
    im = Image.fromarray(arr, "RGB")
    if im.size == size:
        return arr.copy()  # Pillow returns a copy for an identical size; made explicit
    return np.asarray(im.resize(size, Image.Resampling.BICUBIC, reducing_gap=None), dtype=np.uint8)


def crop_offsets(h: int, w: int, multiple: int = 1, size: int = VIEW) -> tuple[int, int]:
    if h < size or w < size:
        raise ValueError(f"image {w}x{h} smaller than {size}")
    top = ((h - size) // 2) // multiple * multiple
    left = ((w - size) // 2) // multiple * multiple
    return top, left


def crop(arr: np.ndarray, top: int, left: int, size: int = VIEW) -> np.ndarray:
    out = arr[top:top + size, left:left + size]
    assert out.shape == (size, size, 3)
    return np.ascontiguousarray(out)


def harmonize(arr: np.ndarray, is_real: bool) -> np.ndarray:
    """D-jpeg/D-ctrl step 1: real unchanged; synthetic encoded once at Q96 at native size."""
    q = int(_cond_cfg()["conditions"]["d_jpeg"]["jpeg_harmonize"]["quality"])
    return arr if is_real else jpeg_roundtrip(arr, q)


def make_views(decoded: np.ndarray, is_real: bool) -> dict[str, np.ndarray]:
    """All three condition views of one decoded image, plus crop metadata."""
    h, w = decoded.shape[:2]
    raw_r = resize_short_256(decoded)
    t, l = crop_offsets(*raw_r.shape[:2])
    d_raw = crop(raw_r, t, l)
    harm = harmonize(decoded, is_real)
    jr = resize_short_256(harm)
    d_jpeg = crop(jr, *crop_offsets(*jr.shape[:2]))
    m = offset_multiple()
    ct, cl = crop_offsets(h, w, m)
    d_ctrl = crop(harm, ct, cl)
    return {"d_raw": d_raw, "d_jpeg": d_jpeg, "d_ctrl": d_ctrl, "ctrl_top": ct, "ctrl_left": cl,
            "height": h, "width": w}


def gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    y = gaussian_filter(arr.astype(np.float64), sigma=(sigma, sigma, 0), mode="reflect", truncate=4.0)
    return np.clip(np.rint(y), 0, 255).astype(np.uint8)


def e2_transform(arr: np.ndarray, name: str) -> np.ndarray:
    """E2 transformations by name: clean, jpeg95, jpeg75, jpeg50, blur0.5, blur1.0."""
    if name == "clean":
        return arr
    if name.startswith("jpeg"):
        return jpeg_roundtrip(arr, int(name[4:]))
    if name.startswith("blur"):
        return gaussian_blur(arr, float(name[4:]))
    raise ValueError(name)


E2_NAMES = ["clean", "jpeg95", "jpeg75", "jpeg50", "blur0.5", "blur1.0"]


def e2_names_from_config() -> list[str]:
    t = config.load("e2_postprocessing")["transforms"]
    names = ["clean"] + [f"jpeg{q}" for q in t["jpeg"]["quality"]] + [f"blur{s}" for s in t["gaussian_blur"]["sigma"]]
    assert names == E2_NAMES
    return names


def e3c_encode(arr: np.ndarray, enc: str) -> np.ndarray:
    """E3C encodings: 'png' or 'jpeg_q75' applied to a D-ctrl view."""
    if enc == "png":
        return png_roundtrip(arr)
    if enc == "jpeg_q75":
        q = int(config.load("e3_encoding")["E3C"]["jpeg"]["quality"])
        return jpeg_roundtrip(arr, q)
    raise ValueError(enc)


def native_view(decoded: np.ndarray) -> np.ndarray:
    """POST-HOC exploratory condition D-native (added after the final runs, PROTOCOL.md v0.6.4):
    the D-ctrl crop (no resizing, offset multiple m) applied to the decoded stored image
    WITHOUT the synthetic JPEG harmonization. Real views equal D-ctrl views exactly."""
    h, w = decoded.shape[:2]
    return crop(decoded, *crop_offsets(h, w, offset_multiple()))

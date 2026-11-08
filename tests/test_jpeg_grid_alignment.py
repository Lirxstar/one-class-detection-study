"""D-ctrl crop offsets align with the JPEG block grid; harmonizing JPEG matches the real tables."""
import io

import numpy as np
import pytest
from PIL import Image

import conditions as C
from audit_dataset import estimate_jpeg_quality, reference_qtables, SUBSAMPLING
from PIL import JpegImagePlugin


def test_offset_multiple_from_rule():
    assert C.modal_subsampling() == "4:4:4" and C.offset_multiple() == 8


@pytest.mark.parametrize("h,w", [(224, 224), (256, 256), (375, 500), (500, 333), (1024, 1024), (241, 999)])
def test_ctrl_offsets_are_aligned_and_centred(h, w):
    m = C.offset_multiple()
    t, l = C.crop_offsets(h, w, m)
    assert t % m == 0 and l % m == 0
    assert 0 <= (h - 224) // 2 - t < m and 0 <= (w - 224) // 2 - l < m


def test_ctrl_offsets_in_cache(view_index):
    m = C.offset_multiple()
    assert (view_index["ctrl_top"] % m == 0).all() and (view_index["ctrl_left"] % m == 0).all()
    assert ((view_index["height"] - 224) // 2 - view_index["ctrl_top"]).between(0, m - 1).all()


def test_harmonizing_jpeg_uses_exact_ijg_q96_444():
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    im = Image.open(io.BytesIO(C.jpeg_bytes(arr, 96)))
    q, err, exact, _ = estimate_jpeg_quality(im.quantization, reference_qtables())
    assert (q, exact) == (96, True)
    assert SUBSAMPLING[JpegImagePlugin.get_sampling(im)] == "4:4:4"


def _smooth_image(seed=0, size=512):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(size // 8, size // 8, 3))
    img = np.kron(x, np.ones((8, 8, 1)))
    from scipy.ndimage import gaussian_filter
    img = gaussian_filter(img, (3, 3, 0))
    img = (img - img.min()) / (img.max() - img.min()) * 255
    return img.astype(np.uint8)


def test_aligned_crop_recompresses_closer_than_misaligned():
    """A crop at a multiple of 8 keeps the original block grid: re-encoding at the
    same quality changes it less than a crop shifted by 4 pixels."""
    once = C.jpeg_roundtrip(_smooth_image(), 75)
    errs = {}
    for name, off in (("aligned", 8 * 5), ("shifted", 8 * 5 + 4)):
        v = C.crop(once, off, off)
        errs[name] = np.abs(C.jpeg_roundtrip(v, 75).astype(int) - v).mean()
    assert errs["aligned"] < 0.75 * errs["shifted"], errs

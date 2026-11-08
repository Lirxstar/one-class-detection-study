"""Resize, crop, blur, decode rules (PROTOCOL.md Sections 3 and 7)."""
import io

import numpy as np
import pytest
from PIL import Image

import conditions as C
from data_io import AlphaNotOpaque, decode_rgb


def _png(arr, mode):
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.parametrize("h,w,expected", [(375, 500, (256, 341)), (500, 333, (384, 256)), (256, 256, (256, 256)),
                                          (1024, 1024, (256, 256)), (300, 301, (256, 256))])
def test_resize_rule(h, w, expected):
    arr = np.zeros((h, w, 3), np.uint8)
    assert C.resize_short_256(arr).shape[:2] == expected


def test_no_resampling_at_256():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    assert np.array_equal(C.resize_short_256(arr), arr)


def test_center_crop_d_raw():
    t, l = C.crop_offsets(256, 341)
    assert (t, l) == (16, 58)


def test_blur_matches_spec():
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    for s in (0.5, 1.0):
        want = np.clip(np.rint(gaussian_filter(arr.astype(np.float64), (s, s, 0), mode="reflect", truncate=4.0)), 0, 255)
        out = C.gaussian_blur(arr, s)
        assert out.dtype == np.uint8 and np.array_equal(out, want.astype(np.uint8))
        assert not np.array_equal(out, arr)


def test_e2_names_match_config():
    assert C.e2_names_from_config() == ["clean", "jpeg95", "jpeg75", "jpeg50", "blur0.5", "blur1.0"]
    arr = np.full((224, 224, 3), 7, np.uint8)
    assert C.e2_transform(arr, "clean") is arr


def test_decode_rgba_requires_opaque_alpha():
    rng = np.random.default_rng(0)
    rgba = rng.integers(0, 256, (32, 32, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    assert np.array_equal(decode_rgb(_png(rgba, "RGBA")), rgba[..., :3])
    rgba[0, 0, 3] = 254
    with pytest.raises(AlphaNotOpaque):
        decode_rgb(_png(rgba, "RGBA"))


def test_decode_rejects_other_modes():
    with pytest.raises(ValueError):
        decode_rgb(_png(np.zeros((8, 8), np.uint8), "L"))


def test_synthetic_harmonized_real_untouched():
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)
    assert C.harmonize(arr, True) is arr
    assert not np.array_equal(C.harmonize(arr, False), arr)
    v = C.make_views(arr, True)
    t, l = C.crop_offsets(300, 400, 8)
    assert np.array_equal(v["d_ctrl"], arr[t:t + 224, l:l + 224])


def test_posthoc_native_view_is_ctrl_crop_without_harmonization():
    rng = np.random.default_rng(4)
    arr = rng.integers(0, 256, (300, 420, 3), dtype=np.uint8)
    t, l = C.crop_offsets(300, 420, C.offset_multiple())
    assert np.array_equal(C.native_view(arr), arr[t:t + 224, l:l + 224])
    assert np.array_equal(C.native_view(arr), C.make_views(arr, True)["d_ctrl"])      # real: identical to D-ctrl
    assert not np.array_equal(C.native_view(arr), C.make_views(arr, False)["d_ctrl"])  # synthetic: no JPEG

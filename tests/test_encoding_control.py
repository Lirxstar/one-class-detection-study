"""E3C encodings: C0 PNG is lossless on D-ctrl views; C1 JPEG Q75 changes them."""
import numpy as np

import conditions as C
import config
from config import ROOT


def test_e3c_config():
    e = config.load("e3_encoding")["E3C"]
    assert e["jpeg"]["quality"] == 75 and e["C0"]["reuse_d_ctrl_features_if_bitwise_equal"] is True
    assert [v["name"] for v in e["C1"]["test_variants"]] == ["biased", "matched", "inverted"]


def test_png_roundtrip_bitwise_equal_on_ctrl_views(view_index):
    v = np.load(ROOT / "artifacts" / "cache" / "views" / "d_ctrl.npy", mmap_mode="r")
    for i in range(0, len(view_index), 41):
        assert np.array_equal(C.e3c_encode(np.asarray(v[i]), "png"), v[i])


def test_jpeg75_changes_views(view_index):
    v = np.load(ROOT / "artifacts" / "cache" / "views" / "d_ctrl.npy", mmap_mode="r")
    a = np.asarray(v[0])
    assert not np.array_equal(C.e3c_encode(a, "jpeg_q75"), a)

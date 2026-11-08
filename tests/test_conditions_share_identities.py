"""All conditions use identical row identities (PROTOCOL.md Section 3)."""
import numpy as np

from config import ROOT
from conftest import flat

VIEW_DIR = ROOT / "artifacts" / "cache" / "views"


def test_view_index_covers_exactly_the_partitions(view_index, splits):
    want = sorted(r for v in splits["partitions"].values() for r in flat(v))
    assert view_index["row_id"].tolist() == want
    assert (view_index["pos"] == np.arange(len(view_index))).all()


def test_all_conditions_same_shape_and_rows(view_index):
    shapes = {c: np.load(VIEW_DIR / f"{c}.npy", mmap_mode="r").shape for c in ("d_raw", "d_jpeg", "d_ctrl")}
    assert set(shapes.values()) == {(len(view_index), 224, 224, 3)}


def test_views_are_filled(view_index):
    rng = np.random.default_rng(0)
    for c in ("d_raw", "d_jpeg", "d_ctrl"):
        v = np.load(VIEW_DIR / f"{c}.npy", mmap_mode="r")
        for i in rng.choice(len(view_index), 50, replace=False):
            assert v[i].std() > 0, (c, i)


def test_real_views_identical_in_raw_and_jpeg(view_index):
    """Real images are not re-encoded: D-raw and D-jpeg views coincide for real rows."""
    raw = np.load(VIEW_DIR / "d_raw.npy", mmap_mode="r")
    jpg = np.load(VIEW_DIR / "d_jpeg.npy", mmap_mode="r")
    real = view_index.index[view_index["label"] == 0][::97]
    synth = view_index.index[view_index["label"] == 1][::53]
    assert all(np.array_equal(raw[i], jpg[i]) for i in real)
    assert not all(np.array_equal(raw[i], jpg[i]) for i in synth)

"""Cached features: dimensions, finiteness, provenance, and bitwise reproducibility."""
import json

import numpy as np
import pandas as pd
import pytest

from config import ROOT

FEAT = ROOT / "artifacts" / "cache" / "features"


def _files():
    return sorted(FEAT.glob("*/*/*.npz")) if FEAT.exists() else []


def test_cached_features_finite_and_dims():
    files = _files()
    if not files:
        pytest.skip("feature cache not built")
    for f in files:
        z = np.load(f)
        meta = json.loads(f.with_suffix(".json").read_text())
        dim = 384 if "primary" in f.parts else 768
        assert z["h_pre"].shape == z["h_post"].shape == (meta["n"], dim)
        assert np.isfinite(z["h_pre"]).all() and np.isfinite(z["h_post"]).all()
        assert list(z["row_ids"]) == sorted(z["row_ids"])


def test_reextraction_is_bitwise_identical():
    f = FEAT / "primary" / "d_ctrl" / "clean.npz"
    if not f.exists():
        pytest.skip("feature cache not built")
    import encoder
    z = np.load(f)
    idx = pd.read_csv(ROOT / "artifacts" / "cache" / "views" / "index.csv").set_index("row_id")
    ids = [r for r in z["row_ids"][:200] if idx.loc[r, "partition"] == "R_train"][:40]
    views = np.load(ROOT / "artifacts" / "cache" / "views" / "d_ctrl.npy", mmap_mode="r")[idx.loc[ids, "pos"].to_numpy()]
    h_pre, h_post = encoder.extract(encoder.load_backbone("primary"), views)
    pos = {r: i for i, r in enumerate(z["row_ids"])}
    sel = [pos[r] for r in ids]
    assert np.array_equal(h_pre, z["h_pre"][sel]) and np.array_equal(h_post, z["h_post"][sel])

#!/usr/bin/env python3
"""E3C C0 check (PROTOCOL.md 7, E3C): every D-ctrl view used by E3C is bitwise
unchanged by a PNG save/decode, so C0 reuses the D-ctrl clean features.
Writes artifacts/audit/c0_png_equality.json."""
import json

import numpy as np
import pandas as pd

import conditions as C
from config import ROOT

idx = pd.read_csv(ROOT / "artifacts" / "cache" / "views" / "index.csv")
v = np.load(ROOT / "artifacts" / "cache" / "views" / "d_ctrl.npy", mmap_mode="r")
bad = [rid for rid, p in zip(idx["row_id"], idx["pos"]) if not np.array_equal(C.e3c_encode(np.asarray(v[p]), "png"), v[p])]
res = {"n_views_checked": int(len(idx)), "n_different": len(bad), "different_row_ids": bad[:50],
       "all_bitwise_equal": not bad}
(ROOT / "artifacts" / "audit" / "c0_png_equality.json").write_text(json.dumps(res, indent=1) + "\n")
print(res)

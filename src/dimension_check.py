#!/usr/bin/env python3
"""Dimension-dominance check on R_train h_pre (PROTOCOL.md Section 4), train data only.

The M0-x dimension set is fixed from R_train in the primary condition D-ctrl
and applied unchanged in every condition; the other conditions are reported
for information. Writes artifacts/audit/dimension_check_<backbone>.json.
"""
import json
import sys

import config
import detectors as D
from config import ROOT
from pipeline import Features, part_ids

backbone = sys.argv[1] if len(sys.argv) > 1 else "primary"
factor = float(config.load("core")["encoder"]["dimension_dominance"]["dominant_if_gt_median_times"])
conds = ["d_ctrl", "d_raw", "d_jpeg"] if backbone == "primary" else ["d_ctrl"]
res = {"backbone": backbone, "rule": f"mean |h_pre| on R_train > {factor} x median", "per_condition": {}}
for c in conds:
    h_pre, _ = Features(backbone, c, "clean").take(part_ids("R_train"))
    res["per_condition"][c] = D.dominant_dims(h_pre, factor)
p = res["per_condition"]["d_ctrl"]
res["dominant_dims"] = p["dominant_dims"]
res["ablation_triggered"] = bool(p["dominant_dims"])
res["note"] = "M0-x excludes dominant_dims (from D-ctrl R_train) in all conditions; primary M0 unchanged"
out = ROOT / "artifacts" / "audit" / f"dimension_check_{backbone}.json"
out.write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps({k: v for k, v in res.items() if k != "per_condition"}, indent=1))
for c, v in res["per_condition"].items():
    print(c, "dominant:", v["dominant_dims"], "share:", round(v["dominant_share"], 3), "max/median:", round(v["max_over_median"], 1))

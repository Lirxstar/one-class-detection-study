#!/usr/bin/env python3
"""Feature cache: both class-token stages from one forward pass (PROTOCOL.md Section 4).

Output per job: `artifacts/cache/features/<backbone>/<condition>/<variant>.npz`
with `row_ids`, `h_pre`, `h_post` (float32) and a JSON sidecar with provenance.
Variants: `clean`; E2 transforms (`jpeg95`, `jpeg75`, `jpeg50`, `blur0.5`,
`blur1.0`) applied to the 224 view; E3C `c1_jpeg75` (JPEG Q75 of the D-ctrl
view). Rows are taken from the view cache in sorted row_id order.

Usage: python src/extract_features.py --plan all
       python src/extract_features.py --backbone primary --condition d_ctrl --variant clean
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time

import numpy as np
import pandas as pd
import torch

import conditions as C
import config
import encoder
from config import ROOT

VIEW_DIR = ROOT / "artifacts" / "cache" / "views"
FEAT_DIR = ROOT / "artifacts" / "cache" / "features"

ALL = ["R_train", "F_seen_train", "R_cal", "R_test", "F_seen_test", "F_unseen_test", "P_cal", "P_test", "P_fake"]
TEST = ["R_test", "F_seen_test", "F_unseen_test", "P_test", "P_fake"]
E3C = ["R_train", "R_cal", "R_test", "F_seen_test", "F_unseen_test", "P_cal", "P_test", "P_fake"]
STUDY = ["R_train", "F_seen_train", "R_cal", "R_test", "F_seen_test", "F_unseen_test"]

PLAN = (
    [("primary", c, "clean", ALL) for c in ("d_ctrl", "d_raw", "d_jpeg")]
    + [("primary", c, v, TEST) for c in ("d_ctrl", "d_raw") for v in C.E2_NAMES[1:]]
    + [("primary", "d_ctrl", "c1_jpeg75", E3C)]
    + [("secondary", "d_ctrl", "clean", STUDY)]
)


def git_commit() -> str:
    from provenance import code_revision
    return code_revision()


def transform(view: np.ndarray, variant: str) -> np.ndarray:
    if variant == "clean":
        return view
    if variant in C.E2_NAMES:
        return C.e2_transform(view, variant)
    if variant == "c1_jpeg75":
        return C.e3c_encode(view, "jpeg_q75")
    raise ValueError(variant)


def feature_path(backbone: str, condition: str, variant: str):
    return FEAT_DIR / backbone / condition / f"{variant}.npz"


def load_features(backbone: str, condition: str, variant: str) -> dict:
    z = np.load(feature_path(backbone, condition, variant), allow_pickle=False)
    return {"row_ids": z["row_ids"].astype(str), "h_pre": z["h_pre"], "h_post": z["h_post"]}


def run_job(model, backbone: str, condition: str, variant: str, partitions: list[str]) -> dict:
    idx = pd.read_csv(VIEW_DIR / "index.csv")
    sel = idx[idx["partition"].isin(partitions)].sort_values("row_id")
    views = np.load(VIEW_DIR / f"{condition}.npy", mmap_mode="r")
    t = time.time()
    x = np.stack([transform(np.asarray(views[p]), variant) for p in sel["pos"]])
    t_views = time.time() - t
    h_pre, h_post = encoder.extract(model, x)
    if not (np.isfinite(h_pre).all() and np.isfinite(h_post).all()):
        raise RuntimeError("non-finite features")
    out = feature_path(backbone, condition, variant)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, row_ids=sel["row_id"].to_numpy().astype("U64"), h_pre=h_pre, h_post=h_post)
    spec = config.load("core")["encoder"][backbone]
    meta = {
        "backbone": backbone, "model": spec["name"], "weights_sha256": spec["weights_sha256"],
        "encoder_revision": spec["repo_commit"], "condition": condition, "variant": variant,
        "partitions": partitions, "n": int(len(sel)), "dim": int(h_pre.shape[1]),
        "batch_size": int(config.load("core")["encoder"]["batch_size"]), "torch": torch.__version__,
        "threads": torch.get_num_threads(), "git_commit": git_commit(), "config_hash": config.config_hash(),
        "seconds_views": round(t_views, 1), "seconds_total": round(time.time() - t, 1),
        "sha256_h_pre": hashlib.sha256(h_pre.tobytes()).hexdigest(),
        "sha256_h_post": hashlib.sha256(h_post.tobytes()).hexdigest(),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{backbone}/{condition}/{variant}: n={meta['n']} {meta['seconds_total']}s", flush=True)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=["all"], default=None)
    ap.add_argument("--backbone", default="primary")
    ap.add_argument("--condition")
    ap.add_argument("--variant", default="clean")
    ap.add_argument("--partitions", nargs="+", default=ALL)
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    jobs = PLAN if a.plan == "all" else [(a.backbone, a.condition, a.variant, a.partitions)]
    models = {}
    for backbone, cond, var, parts in jobs:
        if a.skip_existing and feature_path(backbone, cond, var).exists():
            continue
        if backbone not in models:
            models.clear()
            models[backbone] = encoder.load_backbone(backbone)
        run_job(models[backbone], backbone, cond, var, parts)


if __name__ == "__main__":
    main()

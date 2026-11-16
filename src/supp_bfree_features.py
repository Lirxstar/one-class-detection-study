#!/usr/bin/env python3
"""S-BF views and frozen DINOv2 features (PROTOCOL.md Section 7).

Views of every row in an S-BF partition (never the `unused` rows): `native` (core D-native crop) and
`resize` (core D-raw view), both from `supp_bfree_data.decode_srgb` pixels. Features per cell
(backbone, condition) with `encoder.extract`, unchanged. Cache: artifacts/cache/supp_bfree/ (not
committed; outside artifacts/cache/features/ so core feature tests never read it).

Checks (all must pass before the cache is written): (a) when the main-study caches exist, the encoder
reproduces 32 cached D-ctrl R_train features bitwise for each backbone; (b) re-extracting 32 S-BF views
reproduces their features bitwise; (c) for every sRGB-tagged real row, `decode_srgb` equals the stored
pixels bitwise.

Usage: python src/supp_bfree_features.py --workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

import conditions as C
import config
import encoder
import extract_features as X
import supp_bfree_data as S
from config import ROOT
from provenance import code_revision

CACHE = ROOT / "artifacts" / "cache" / "supp_bfree"
VIEW_DIR = CACHE / "views"
FEAT_DIR = CACHE / "features"
SPLITS = ROOT / "artifacts" / "supp_bfree" / "splits.json"
MANIFEST = ROOT / "artifacts" / "supp_bfree" / "manifest.csv"
CONDITIONS = ["native", "resize"]
CELLS = [("primary", "native"), ("primary", "resize"), ("secondary", "native")]
N_CHECK = 32


def manifest() -> pd.DataFrame:
    return pd.read_csv(MANIFEST, keep_default_na=False, dtype=str).set_index("row_id")


def rows_needed() -> list[str]:
    parts = json.loads(SPLITS.read_text())["partitions"]
    return sorted(set().union(*parts.values()))


def views_of(arr: np.ndarray) -> dict[str, np.ndarray]:
    native = C.native_view(arr)
    r = C.resize_short_256(arr)
    return {"native": native, "resize": C.crop(r, *C.crop_offsets(*r.shape[:2]))}


def _one(args: tuple[str, str]) -> tuple[str, dict[str, np.ndarray], bool]:
    rid, icc = args
    cls, sid = S.parse(rid)
    arr = S.decode_srgb(cls, sid)
    identity = None
    if cls == "REAL" and icc == S.cfg()["colour"]["sensitivity"]["real_icc_description"]:
        identity = bool(np.array_equal(arr, S.decode(cls, sid)))
    return rid, views_of(arr), identity


def build_views(ids: list[str], workers: int) -> tuple[dict[str, np.ndarray], dict]:
    m = manifest()
    out = {c: np.zeros((len(ids), 224, 224, 3), np.uint8) for c in CONDITIONS}
    pos = {r: i for i, r in enumerate(ids)}
    n_identity, failures = 0, []
    with ProcessPoolExecutor(workers) as ex:
        for rid, v, identity in ex.map(_one, [(r, m.loc[r, "icc_description"]) for r in ids], chunksize=8):
            for c in CONDITIONS:
                out[c][pos[rid]] = v[c]
            if identity is not None:
                n_identity += identity
                if not identity:
                    failures.append(rid)
    if failures:
        raise AssertionError(f"sRGB conversion is not the identity for {len(failures)} sRGB-tagged rows: {failures[:5]}")
    return out, {"srgb_identity_rows": n_identity}


def core_reproduction(model, backbone: str) -> int:
    """(a): cached core D-ctrl R_train features are reproduced bitwise by this environment.
    Returns 0 (check not run) when the main-study view and feature caches are absent."""
    if not (X.feature_path(backbone, "d_ctrl", "clean").exists()
            and (ROOT / "artifacts" / "cache" / "views" / "d_ctrl.npy").exists()):
        return 0
    core = X.load_features(backbone, "d_ctrl", "clean")
    idx = pd.read_csv(ROOT / "artifacts" / "cache" / "views" / "index.csv").set_index("row_id")
    ids = [r for r in core["row_ids"] if idx.loc[r, "partition"] == "R_train"][:N_CHECK]
    views = np.load(ROOT / "artifacts" / "cache" / "views" / "d_ctrl.npy", mmap_mode="r")[idx.loc[ids, "pos"].to_numpy()]
    pre, post = encoder.extract(model, np.asarray(views))
    p = {r: i for i, r in enumerate(core["row_ids"])}
    rows = [p[r] for r in ids]
    if not (np.array_equal(pre, core["h_pre"][rows]) and np.array_equal(post, core["h_post"][rows])):
        raise AssertionError(f"{backbone}: core features not reproduced bitwise")
    return len(ids)


def feature_path(backbone: str, condition: str):
    return FEAT_DIR / backbone / f"{condition}.npz"


def load(backbone: str, condition: str) -> dict:
    z = np.load(feature_path(backbone, condition), allow_pickle=False)
    return {"row_ids": z["row_ids"].astype(str), "h_pre": z["h_pre"], "h_post": z["h_post"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    commit = code_revision()
    ids = rows_needed()
    t = time.time()
    views, vdiag = build_views(ids, a.workers)
    VIEW_DIR.mkdir(parents=True, exist_ok=True)
    for c in CONDITIONS:
        np.save(VIEW_DIR / f"{c}.npy", views[c])
    pd.DataFrame({"pos": range(len(ids)), "row_id": ids}).to_csv(VIEW_DIR / "index.csv", index=False)
    print(f"views: {len(ids)} rows, {time.time() - t:.0f}s, {vdiag}", flush=True)
    summary = {"git_commit": commit, "n_rows": len(ids), **vdiag, "cells": {}}
    for backbone in ("primary", "secondary"):
        model = encoder.load_backbone(backbone)
        n_core = core_reproduction(model, backbone)
        for cb, cond in CELLS:
            if cb != backbone:
                continue
            t = time.time()
            h_pre, h_post = encoder.extract(model, views[cond])
            if not (np.isfinite(h_pre).all() and np.isfinite(h_post).all()):
                raise RuntimeError("non-finite features")
            pre2, post2 = encoder.extract(model, views[cond][:N_CHECK])
            if not (np.array_equal(pre2, h_pre[:N_CHECK]) and np.array_equal(post2, h_post[:N_CHECK])):
                raise AssertionError("re-extraction not bitwise identical")
            out = feature_path(backbone, cond)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out, row_ids=np.array(ids).astype("U64"), h_pre=h_pre, h_post=h_post)
            spec = config.load("core")["encoder"][backbone]
            meta = {"backbone": backbone, "model": spec["name"], "weights_sha256": spec["weights_sha256"],
                    "encoder_revision": spec["repo_commit"], "condition": cond, "n": len(ids), "dim": int(h_pre.shape[1]),
                    "batch_size": int(config.load("core")["encoder"]["batch_size"]), "torch": torch.__version__,
                    "threads": torch.get_num_threads(), "git_commit": commit, "supp_config_hash": S.supp_config_hash(),
                    "core_reproduction_rows": n_core, "redo_check_rows": N_CHECK, "seconds": round(time.time() - t, 1),
                    "sha256_h_pre": hashlib.sha256(h_pre.tobytes()).hexdigest(),
                    "sha256_h_post": hashlib.sha256(h_post.tobytes()).hexdigest()}
            out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
            summary["cells"][f"{backbone}/{cond}"] = meta
            print(f"{backbone}/{cond}: n={len(ids)} {meta['seconds']}s", flush=True)
    (CACHE / "feature_cache_manifest.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""POST-HOC, OPTIONAL: reference sampling sensitivity (study proposal Section 6.3; PROTOCOL v0.6.6).

Run once from tag `posthoc-reseed` in the original repository, after all E1 to E3 results were seen
(requested by the author). The public script refuses to overwrite existing outputs instead.

R_train is redrawn with seeds 7, 13 and 29 by the rule that drew it with seed 42
(`make_splits.permuted` over the eligible `train_real` pool, first 2,000 rows), holding
F_seen_train, R_cal, R_test and every test partition fixed. B1, O1 and O2 are refitted (B1's C
again by grouped 5-fold CV inside the training data) and every method is recalibrated on the
fixed R_cal. M0 does not depend on R_train. D-ctrl, ViT-S/14 only. Seeds are not independent
datasets.

Redrawn images outside the frozen partitions have no cached view or feature. Their D-ctrl views
(`conditions.make_views`, real branch) and ViT-S/14 features are computed here into new cache
files; existing caches are never modified. Before any reseeded fit the script asserts that
(1) seed 42 reproduces R_train, (2) the view code reproduces cached D-ctrl views of R_train
images bitwise, (3) their features are reproduced bitwise, and (4) the reseeding path with seed
42 reproduces the saved pilot (train-only) and E1 D-ctrl predictions exactly. The pilot records
are not distributed in the public release; without them only the E1 comparison runs, and
`--dev-check` (which needs them) is unavailable.
"""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import conditions as C
import config
import detectors as D
import encoder
import extract_features as X
import pipeline as P
from config import ROOT
from data_io import DATA_DIR, decode_rgb, parse_row_id
from make_splits import permuted

SEEDS = [7, 13, 29]
FROZEN_SEED = 42
VIEW_DIR = ROOT / "artifacts" / "cache" / "views"
VIEWS_OUT = VIEW_DIR / "reseed_d_ctrl.npy"
INDEX_OUT = VIEW_DIR / "reseed_index.csv"
FEAT_OUT = X.feature_path("primary", "d_ctrl", "reseed_real")
OUT = P.PRED_DIR / "posthoc"
PRED_OUT = OUT / "posthoc_reseed_primary_d_ctrl.csv.gz"
DIAG_OUT = OUT / "posthoc_reseed_diagnostics.json"
N_CHECK = 64


def train_real_pool() -> pd.DataFrame:
    m = pd.read_csv(ROOT / "artifacts" / "manifest.csv", usecols=["row_id", "split", "label", "role", "eligible"])
    return m[m["eligible"].astype(str).str.lower().eq("true") & (m["role"] == "train_real")]


def draw(seed: int) -> list[str]:
    """R_train for `seed`, by the frozen rule; seed 42 is the frozen R_train."""
    return sorted(permuted(train_real_pool(), seed)[:len(P.part_ids("R_train"))])


def _views(args):
    stem, items = args
    col = pq.read_table(DATA_DIR / f"{stem}.parquet", columns=["image"]).column("image")
    recs = col.take([i for _, i in items]).to_pylist()
    return [(rid, C.make_views(decode_rgb(rec["bytes"]), True)["d_ctrl"]) for (rid, _), rec in zip(items, recs)]


def build_views(ids: list[str], workers: int = 8) -> np.ndarray:
    """D-ctrl views of real images, in the order of `ids`."""
    jobs = {}
    for rid in ids:
        _, stem, i = parse_row_id(rid)
        jobs.setdefault(stem, []).append((rid, i))
    got = {}
    with ProcessPoolExecutor(workers) as ex:
        for part in ex.map(_views, sorted(jobs.items())):
            got.update(part)
    return np.stack([got[r] for r in ids])


class Union:
    """Row-id addressable union of feature sets (each row id at most once)."""

    def __init__(self, parts: list[dict]):
        ids = np.concatenate([p["row_ids"] for p in parts])
        if len(set(ids)) != len(ids):
            raise ValueError("feature sets overlap")
        self.pos = {r: i for i, r in enumerate(ids)}
        self.h_pre = np.concatenate([p["h_pre"] for p in parts])
        self.h_post = np.concatenate([p["h_post"] for p in parts])

    take = P.Features.take


def training_inputs(feats, r_ids: list[str]) -> dict:
    """As pipeline.training_inputs, with an explicit real reference set (train split, real only)."""
    m = P.meta()
    assert (m.loc[r_ids, "split"] == "train").all() and (m.loc[r_ids, "label"] == 0).all()
    f_ids = P.part_ids("F_seen_train")
    _, r_post = feats.take(r_ids)
    _, f_post = feats.take(f_ids)
    return {"R_train": {"h_post": r_post, "row_ids": np.array(r_ids), "groups": m.loc[r_ids, "dup_group"].to_numpy()},
            "F_seen_train": {"h_post": f_post, "groups": m.loc[f_ids, "dup_group"].to_numpy()}}


def run_seed(feats, seed: int, role: str, run_id: str) -> tuple[pd.DataFrame, dict]:
    r_ids = draw(seed)
    fit = D.fit_methods(training_inputs(feats, r_ids), P.m0x_dims("primary"))
    cal_ids = P.part_ids(P.ROLES[role]["cal"])
    if set(cal_ids) & set(r_ids):
        raise AssertionError("reference and calibration rows overlap")
    taus = P.thresholds(fit, feats, cal_ids)
    real, fake = P.eval_ids(role)
    out = []
    for ids in (cal_ids, real + fake):
        h_pre, h_post = feats.take(ids)
        out.append(P.records(D.score_all(fit, h_pre, h_post, ids), taus, ids, experiment="POSTHOC-reseed",
                             run_id=run_id, backbone="primary", condition="d_ctrl", perturbation="clean",
                             encoding="none", fit=fit))
    df = pd.concat(out, ignore_index=True)
    df.insert(df.columns.get_loc("method") + 1, "rtrain_seed", seed)
    frozen = set(P.part_ids("R_train"))
    diag = {"seed": seed, "role": role, "n_reference": len(r_ids), "overlap_with_frozen_R_train": len(frozen & set(r_ids)),
            "thresholds": taus, **fit.diagnostics()}
    return df, diag


def assert_same(new: pd.DataFrame, saved_path, what: str) -> int:
    keys = ["row_id", "method", "partition"]
    old = pd.read_csv(saved_path).set_index(keys).sort_index()
    b = new.set_index(keys).sort_index()
    if not old.index.equals(b.index):
        raise AssertionError(f"{what}: identities differ")
    for col in ("score", "threshold", "predicted_label"):
        if not np.array_equal(old[col].to_numpy(), b[col].to_numpy()):
            raise AssertionError(f"{what}: {col} differs (max {np.max(np.abs(old[col].to_numpy() - b[col].to_numpy()))})")
    return len(b)


def save_features(ids: list[str], h_pre: np.ndarray, h_post: np.ndarray, seconds: float, commit: str) -> None:
    np.savez(FEAT_OUT, row_ids=np.array(ids).astype("U64"), h_pre=h_pre, h_post=h_post)
    spec = config.load("core")["encoder"]["primary"]
    meta = {"backbone": "primary", "model": spec["name"], "weights_sha256": spec["weights_sha256"],
            "encoder_revision": spec["repo_commit"], "condition": "d_ctrl", "variant": "reseed_real",
            "partitions": ["reseeded R_train rows outside the frozen partitions (post hoc, PROTOCOL v0.6.6)"],
            "n": int(len(ids)), "dim": int(h_pre.shape[1]),
            "batch_size": int(config.load("core")["encoder"]["batch_size"]), "git_commit": commit,
            "config_hash": config.config_hash(), "seconds_total": round(seconds, 1),
            "sha256_h_pre": hashlib.sha256(h_pre.tobytes()).hexdigest(),
            "sha256_h_post": hashlib.sha256(h_post.tobytes()).hexdigest()}
    FEAT_OUT.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")


def checks(model, clean: dict) -> list[str]:
    """(1) to (3); train-split data only. Returns the checked R_train ids."""
    frozen = P.part_ids("R_train")
    if draw(FROZEN_SEED) != frozen:
        raise AssertionError("seed 42 does not reproduce R_train")
    idx = pd.read_csv(VIEW_DIR / "index.csv").set_index("row_id")
    check = frozen[:N_CHECK]
    v_new = build_views(check)
    v_old = np.load(VIEW_DIR / "d_ctrl.npy", mmap_mode="r")[idx.loc[check, "pos"].to_numpy()]
    if not np.array_equal(v_new, v_old):
        raise AssertionError("view code does not reproduce cached D-ctrl views")
    pre_c, post_c = encoder.extract(model, v_new)
    pos = {r: i for i, r in enumerate(clean["row_ids"])}
    rows = [pos[r] for r in check]
    if not (np.array_equal(pre_c, clean["h_pre"][rows]) and np.array_equal(post_c, clean["h_post"][rows])):
        raise AssertionError("feature extraction does not reproduce cached features")
    return check


def dev_check() -> None:
    """Debugging on train-split data only (integrity rule 3): checks (1) to (3) and the pilot half of (4).
    Writes nothing."""
    clean = X.load_features("primary", "d_ctrl", "clean")
    check = checks(encoder.load_backbone("primary"), clean)
    pilot42, _ = run_seed(Union([clean]), FROZEN_SEED, "pilot", "dev-check")
    n = assert_same(pilot42, P.PRED_DIR / "pilot" / "pilot_clean_d_ctrl.csv.gz", "pilot")
    print(f"dev check passed: {len(check)} views/features bitwise, {n} pilot rows identical")


def main(guard: bool = True):
    outs = [PRED_OUT, DIAG_OUT, FEAT_OUT, VIEWS_OUT]
    commit = P.assert_new_run_allowed(outs) if guard else "reproduction"
    clean = X.load_features("primary", "d_ctrl", "clean")
    new_ids = sorted(set().union(*(draw(s) for s in SEEDS)) - set(clean["row_ids"]))
    model = encoder.load_backbone("primary")
    check = checks(model, clean)
    # new views and features
    t = time.time()
    views = build_views(new_ids)
    np.save(VIEWS_OUT, views)
    pd.DataFrame({"pos": range(len(new_ids)), "row_id": new_ids}).to_csv(INDEX_OUT, index=False)
    h_pre, h_post = encoder.extract(model, views)
    if not (np.isfinite(h_pre).all() and np.isfinite(h_post).all()):
        raise RuntimeError("non-finite features")
    save_features(new_ids, h_pre, h_post, time.time() - t, commit)
    fit_and_save(clean, commit, {"n_new_features": len(new_ids), "n_view_feature_check": len(check)})


def fit_and_save(clean: dict, commit: str, extra: dict) -> None:
    """Check (4), then the reseeded fits; uses the cached clean and reseed_real features."""
    feats = Union([clean, X.load_features("primary", "d_ctrl", "reseed_real")])
    run_id = P.new_run_id("posthoc-reseed")
    diag = {"run_id": run_id, "git_commit": commit, "label": "post hoc, optional (proposal 6.3)", **extra}
    # (4): seed 42 through this path reproduces the saved pilot and E1 predictions
    pilot_saved = P.PRED_DIR / "pilot" / "pilot_clean_d_ctrl.csv.gz"
    if pilot_saved.exists():  # not distributed in the public release
        pilot42, _ = run_seed(feats, FROZEN_SEED, "pilot", run_id)
        diag["check_seed42_pilot_rows"] = assert_same(pilot42, pilot_saved, "pilot")
    final42, _ = run_seed(feats, FROZEN_SEED, "final", run_id)
    diag["check_seed42_e1_rows"] = assert_same(final42, P.PRED_DIR / "e1" / "e1_primary_d_ctrl.csv.gz", "E1")
    preds, diag["seeds"] = [], {}
    for s in SEEDS:
        df, d = run_seed(feats, s, "final", run_id)
        preds.append(df)
        diag["seeds"][str(s)] = d
    P.save(pd.concat(preds, ignore_index=True), PRED_OUT)
    DIAG_OUT.write_text(json.dumps(diag, indent=1, default=float) + "\n")
    print(f"saved {PRED_OUT.name}: seeds {SEEDS}; new features {extra.get('n_new_features', 0)}; checks passed")


if __name__ == "__main__":
    import sys
    dev_check() if "--dev-check" in sys.argv else main()

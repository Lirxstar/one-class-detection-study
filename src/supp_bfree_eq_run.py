#!/usr/bin/env python3
"""S-BF-EQ: B1 refitted on half of the S-BF training ids (PROTOCOL.md Section 8).

S-BF and main-study modules are imported unchanged. For each of five fixed seeds, 300 of the 600 S-BF train
source ids are drawn (stratified by split stratum); B1 (`detectors.B1.fit`: grouped 5-fold CV for C) is fitted
on the real and SD3.5 image of each drawn id (300 + 300 = 600 images, the size of the O1/O2 training set) and
calibrated on the S-BF real calibration images. O1, O2 and B1-full are not refitted: the analysis reads them
from the S-BF final predictions.

Roles
  pilot: development on the S-BF pilot partitions (train ids only; pool P_fit_real 360 ids, subsample 180;
         SD3.5 as fake test). Also checks that this code, refitted on the full pilot pool, reproduces the S-BF
         pilot B1 scores bitwise. Needs the S-BF pilot records from `supp_bfree_run.py --pilot` (not
         distributed). Output artifacts/predictions/supp_bfree_eq/pilot/.
  final: S-BF R_train pool, R_cal, R_test, F_unseen_test (FLUX), F_seen_test (SD3.5). Ran once in the original
         repository; this script refuses to overwrite existing outputs.
Needs the S-BF feature cache from `supp_bfree_features.py` (artifacts/cache/supp_bfree/features/).

Usage: python src/supp_bfree_eq_run.py --pilot | python src/supp_bfree_eq_run.py
       python src/supp_bfree_eq_run.py --write-subsamples   (only if artifacts/supp_bfree_eq/subsamples.json is absent)
"""
from __future__ import annotations

import hashlib
import json
import sys
from math import floor

import numpy as np
import pandas as pd

import calibrate
import config
import detectors as D
import pipeline as P
import supp_bfree_data as S
import supp_bfree_features as F
import supp_bfree_run as R
from config import ROOT

ANALYSIS = "all"
OUT = P.PRED_DIR / "supp_bfree_eq"
ART = ROOT / "artifacts" / "supp_bfree_eq"
SUBSAMPLES = ART / "subsamples.json"
METHOD = "B1-eq"


def cfg() -> dict:
    return config.load("supp_bfree_eq")


def eq_config_hash() -> str:
    """sha256 over the S-BF supp_config_hash and this experiment's config file."""
    h = hashlib.sha256(S.supp_config_hash().encode())
    h.update((config.CONFIG_DIR / "supp_bfree_eq.yaml").read_bytes())
    return h.hexdigest()


def seeds() -> list[int]:
    return [int(s) for s in cfg()["train_subsample"]["seeds"]]


# --------------------------------------------------------------------------- subsampling
def allocate(counts: dict[str, int], fraction: float) -> dict[str, int]:
    """floor(n_h * fraction) per stratum; leftover ids (to reach floor(total * fraction + 0.5)) go to the
    largest fractional parts, ties broken by stratum name."""
    total = sum(counts.values())
    target = floor(total * fraction + 0.5)
    k = {h: floor(n * fraction) for h, n in counts.items()}
    order = sorted(counts, key=lambda h: (-(counts[h] * fraction - k[h]), h))
    for h in order[: target - sum(k.values())]:
        k[h] += 1
    return k


def pool(role: str) -> pd.DataFrame:
    """Source ids (index) and strata of the S-BF training pool of `role`."""
    m = R.manifest()
    name = cfg()["train_subsample"]["pool"][role]
    rows = R.part_ids(role, name, ANALYSIS)
    sub = m.loc[rows]
    if not ((sub["class"] == "REAL").all() and (sub["split_role"] == "train").all()):
        raise AssertionError("pool must be real rows of train ids")
    return sub.set_index("source_id")[["stratum", "id_group"]].sort_index()


def draw(role: str, seed: int, fraction: float | None = None) -> list[str]:
    """Sorted source ids drawn for one seed (PROTOCOL.md Section 8)."""
    if fraction is None:
        fraction = float(cfg()["train_subsample"]["fraction"])
    p = pool(role)
    strata = sorted(p["stratum"].unique())
    k = allocate({h: int((p["stratum"] == h).sum()) for h in strata}, fraction)
    rng = np.random.default_rng(seed)
    out: list[str] = []
    for h in strata:
        ids = sorted(p.index[p["stratum"] == h])
        out += [str(x) for x in rng.choice(np.array(ids), size=k[h], replace=False)]
    return sorted(out)


def ids_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def build_subsamples() -> dict:
    out = {"protocol": cfg()["protocol_version"], "eq_config_hash": eq_config_hash(),
           "fraction": float(cfg()["train_subsample"]["fraction"]), "seeds": seeds(), "roles": {}}
    for role in ("final", "pilot"):
        p = pool(role)
        rr = {"pool_ids": len(p), "pool_per_stratum": {h: int(n) for h, n in p["stratum"].value_counts().sort_index().items()},
              "draws": {}}
        for s in seeds():
            ids = draw(role, s)
            rr["draws"][str(s)] = {"n_ids": len(ids), "per_stratum": {h: int(n) for h, n in p.loc[ids, "stratum"].value_counts().sort_index().items()},
                                   "sha256": ids_sha(ids), "source_ids": ids}
        out["roles"][role] = rr
    return out


def committed_subsamples() -> dict:
    return json.loads(SUBSAMPLES.read_text())


# --------------------------------------------------------------------------- fitting
def training_rows(role: str, ids: list[str]) -> tuple[list[str], list[str]]:
    """REAL and SD3.5 rows of the drawn source ids; both must lie in the S-BF training partitions of `role`."""
    r_rows = [S.row_id("REAL", s) for s in ids]
    f_rows = [S.row_id("SD35", s) for s in ids]
    real_pool = set(R.part_ids(role, R.ROLES[role]["fit_real"], ANALYSIS))
    fake_pool = set(R.part_ids(role, R.ROLES[role]["fit_fake"], ANALYSIS))
    if not (set(r_rows) <= real_pool and set(f_rows) <= fake_pool):
        raise AssertionError("drawn ids outside the S-BF training partitions")
    m = R.manifest()
    if not ((m.loc[r_rows, "class"] == "REAL").all() and (m.loc[f_rows, "class"] == "SD35").all()):
        raise AssertionError("training rows must be REAL (label 0) and SD35 (label 1)")
    return r_rows, f_rows


def fit_b1(feats, role: str, ids: list[str]) -> D.B1:
    m = R.manifest()
    r_rows, f_rows = training_rows(role, ids)
    _, zr = feats.take(r_rows)
    _, zf = feats.take(f_rows)
    return D.B1.fit(zr, zf, m.loc[r_rows, "id_group"].to_numpy(), m.loc[f_rows, "id_group"].to_numpy())


def eval_rows(role: str) -> tuple[list[str], list[str], list[str]]:
    r = R.ROLES[role]
    cal = R.part_ids(role, r["cal"], ANALYSIS)
    real = R.part_ids(role, r["real_test"], ANALYSIS)
    fake = sorted(i for p in r["fake_test"] for i in R.part_ids(role, p, ANALYSIS))
    return cal, real, fake


def records(scores: np.ndarray, tau: float, ids: list[str], *, role: str, backbone: str, condition: str, run_id: str,
            b1: D.B1, seed: int, train_ids: list[str], commit: str) -> pd.DataFrame:
    core = config.load("core")
    m = R.manifest().loc[ids]
    s = np.asarray(scores, np.float64)
    return pd.DataFrame({
        "run_id": run_id, "experiment": "S-BF-EQ" if role == "final" else "S-BF-EQ-pilot", "analysis": ANALYSIS,
        "config_hash": config.config_hash(), "supp_config_hash": S.supp_config_hash(), "eq_config_hash": eq_config_hash(),
        "dataset_revision": "bfree-local", "source_split": m["split_role"].to_numpy(), "row_id": ids,
        "source_id": m["source_id"].to_numpy(), "id_group": m["id_group"].to_numpy(), "stratum": m["stratum"].to_numpy(),
        "image_hash": m["rgb_sha1"].to_numpy(), "partition": m[R.ROLES[role]["key"]].to_numpy(),
        "authenticity_label": m["label"].astype(int).to_numpy(), "generator": m["class"].to_numpy(),
        "data_condition": condition, "feature_stage": D.FEATURE_STAGE["B1"], "perturbation": "clean",
        "encoding_condition": "none", "method": METHOD, "backbone": backbone, "score": s, "threshold": tau,
        "predicted_label": (s > tau).astype(int), "selected_C": b1.C, "subsample_seed": seed,
        "n_train_ids": len(train_ids), "n_train_images": 2 * len(train_ids), "train_ids_sha256": ids_sha(train_ids),
        "encoder_revision": core["encoder"][backbone]["repo_commit"], "git_commit": commit,
    })


def run_cell(feats, role: str, backbone: str, condition: str, seed: int, ids: list[str], run_id: str,
             commit: str) -> tuple[pd.DataFrame, dict]:
    m = R.manifest()
    b1 = fit_b1(feats, role, ids)
    cal, real, fake = eval_rows(role)
    if not (m.loc[cal, "class"] == "REAL").all():
        raise AssertionError("calibration rows must be real")
    train_sids = set(ids)
    for rows in (cal, real, fake):
        if train_sids & set(m.loc[rows, "source_id"]):
            raise AssertionError("a drawn training id appears in calibration or test")
    tau = calibrate.threshold(b1.score(feats.take(cal)[1]))
    out = [records(b1.score(feats.take(rows)[1]), tau, rows, role=role, backbone=backbone, condition=condition,
                   run_id=run_id, b1=b1, seed=seed, train_ids=ids, commit=commit) for rows in (cal, real + fake)]
    diag = {"role": role, "backbone": backbone, "condition": condition, "seed": seed, "n_train_ids": len(ids),
            "train_ids_sha256": ids_sha(ids), "threshold": tau, "n_cal": len(cal), "n_real_test": len(real),
            "n_fake_test": len(fake), "B1_selected_C": b1.C, "B1_cv_table": b1.cv_table, "B1_n_iter": b1.n_iter}
    return pd.concat(out, ignore_index=True), diag


def pilot_code_path_check(feats, backbone: str, condition: str) -> dict:
    """This code on the full pilot pool must reproduce the S-BF pilot B1 scores bitwise. Scores are computed in
    the batches of `run_cell` and of S-BF (calibration rows, then test rows): BLAS results depend on the batch
    composition at the 1e-15 level."""
    ids = draw("pilot", 0, fraction=1.0)
    if len(ids) != len(pool("pilot")):
        raise AssertionError("fraction 1.0 must take the whole pool")
    b1 = fit_b1(feats, "pilot", ids)
    saved = pd.read_csv(R.OUT / "pilot" / f"pilot_{backbone}_{condition}.csv.gz", usecols=["analysis", "method", "row_id", "score"])
    saved = saved[(saved["analysis"] == ANALYSIS) & (saved["method"] == "B1")].set_index("row_id")["score"]
    cal, real, fake = eval_rows("pilot")
    batches = (cal, real + fake)
    if sorted(cal + real + fake) != sorted(saved.index):
        raise AssertionError("pilot evaluation rows differ from S-BF")
    equal = all(np.array_equal(b1.score(feats.take(rows)[1]), saved.loc[rows].to_numpy(np.float64)) for rows in batches)
    if not equal:
        raise AssertionError(f"{backbone}/{condition}: refit on the full pilot pool does not reproduce S-BF pilot B1")
    return {"cell": f"{backbone}/{condition}", "rows": len(saved), "bitwise_equal": equal, "C": b1.C}


def run(role: str, run_id: str, commit: str, outdir) -> None:
    sub = committed_subsamples()["roles"][role]["draws"]
    diags = {"run_id": run_id, "git_commit": commit, "role": role, "eq_config_hash": eq_config_hash(),
             "seeds": seeds(), "cells": [], "code_path_check": []}
    for backbone, cond in F.CELLS:
        feats = R.Feats(backbone, cond)
        if role == "pilot":
            diags["code_path_check"].append(pilot_code_path_check(feats, backbone, cond))
        preds = []
        for seed in seeds():
            ids = draw(role, seed)
            if ids != sub[str(seed)]["source_ids"]:
                raise AssertionError(f"seed {seed}: draw differs from artifacts/supp_bfree_eq/subsamples.json")
            df, d = run_cell(feats, role, backbone, cond, seed, ids, run_id, commit)
            preds.append(df)
            diags["cells"].append(d)
            print(f"{role} {backbone}/{cond} seed {seed}: C={d['B1_selected_C']}", flush=True)
        P.save(pd.concat(preds, ignore_index=True), outdir / f"{role}_{backbone}_{cond}.csv.gz")
    (outdir / f"{role}_diagnostics.json").write_text(json.dumps(diags, indent=1, default=float) + "\n")


def outputs(role: str, outdir) -> list:
    return [outdir / f"{role}_{b}_{c}.csv.gz" for b, c in F.CELLS] + [outdir / f"{role}_diagnostics.json"]


def main() -> None:
    if "--write-subsamples" in sys.argv:
        P.assert_new_run_allowed([SUBSAMPLES])
        ART.mkdir(parents=True, exist_ok=True)
        SUBSAMPLES.write_text(json.dumps(build_subsamples(), indent=1) + "\n")
        print(f"wrote {SUBSAMPLES.relative_to(ROOT)}")
        return
    role = "pilot" if "--pilot" in sys.argv else "final"
    outdir = OUT / "pilot" if role == "pilot" else OUT
    if role == "pilot" and not (R.OUT / "pilot").exists():
        raise FileNotFoundError("S-BF pilot records missing: run `python src/supp_bfree_run.py --pilot` first")
    commit = P.assert_new_run_allowed(outputs(role, outdir))
    outdir.mkdir(parents=True, exist_ok=True)
    run(role, P.new_run_id(f"sbf-eq-{role}"), commit, outdir)


if __name__ == "__main__":
    main()

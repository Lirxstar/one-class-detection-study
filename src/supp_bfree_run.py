#!/usr/bin/env python3
"""S-BF Steps 2 and 3: fit, calibrate, score (PROTOCOL.md Section 7).

Fitting uses `detectors.fit_methods` unchanged: its training dict holds exactly `R_train` (S-BF real
train rows) and `F_seen_train` (S-BF SD3.5 train rows); O1 and O2 receive only the real arrays, B1
both, M0 nothing. Thresholds: `calibrate.threshold` on real calibration rows only.

Roles
  pilot: development on train ids only (P_fit_real, P_fit_fake, P_cal, P_test, P_fake = SD3.5);
         output artifacts/predictions/supp_bfree/pilot/ (not distributed).
  final: R_train, F_seen_train, R_cal, R_test, F_unseen_test (FLUX), F_seen_test (SD3.5). Ran once in
         the original repository; this script refuses to overwrite existing outputs.
Analyses: `all` (all eligible ids, colour-converted reals) and `srgb` (ids whose real image is tagged
Nikon sRGB only: refit, recalibrate, test). Cells: ViT-S/14 native (primary), ViT-S/14 resize, ViT-B/14 native.

Usage: python src/supp_bfree_run.py --pilot      |      python src/supp_bfree_run.py
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

import calibrate
import config
import detectors as D
import pipeline as P
import supp_bfree_data as S
import supp_bfree_features as F
from config import ROOT

OUT = P.PRED_DIR / "supp_bfree"
ANALYSES = ["all", "srgb"]
ROLES = {
    "final": {"fit_real": "R_train", "fit_fake": "F_seen_train", "cal": "R_cal", "real_test": "R_test",
              "fake_test": ["F_unseen_test", "F_seen_test"], "key": "partition"},
    "pilot": {"fit_real": "P_fit_real", "fit_fake": "P_fit_fake", "cal": "P_cal", "real_test": "P_test",
              "fake_test": ["P_fake"], "key": "pilot_partition"},
}


@lru_cache(maxsize=1)
def manifest() -> pd.DataFrame:
    return F.manifest()


def part_ids(role: str, name: str, analysis: str) -> list[str]:
    m = manifest()
    sel = m[m[ROLES[role]["key"]] == name]
    if analysis == "srgb":
        sel = sel[sel["srgb_tagged_id"] == "True"]
    elif analysis != "all":
        raise ValueError(analysis)
    return sorted(sel.index)


class Feats:
    def __init__(self, backbone: str, condition: str):
        f = F.load(backbone, condition)
        self.pos = {r: i for i, r in enumerate(f["row_ids"])}
        self.h_pre, self.h_post = f["h_pre"], f["h_post"]

    def take(self, ids):
        idx = [self.pos[r] for r in ids]
        return self.h_pre[idx], self.h_post[idx]


def training_inputs(feats, role: str, analysis: str) -> dict:
    """Exactly the real and SD3.5 training rows of train ids; never FLUX, never a calibration or test id."""
    m = manifest()
    r_ids = part_ids(role, ROLES[role]["fit_real"], analysis)
    f_ids = part_ids(role, ROLES[role]["fit_fake"], analysis)
    if not ((m.loc[r_ids, "class"] == "REAL").all() and (m.loc[f_ids, "class"] == "SD35").all()):
        raise AssertionError("training rows must be REAL (label 0) and SD35 (label 1)")
    if not (m.loc[r_ids + f_ids, "split_role"] == "train").all():
        raise AssertionError("training rows must come from train ids")
    _, r_post = feats.take(r_ids)
    _, f_post = feats.take(f_ids)
    return {"R_train": {"h_post": r_post, "row_ids": np.array(r_ids), "groups": m.loc[r_ids, "id_group"].to_numpy()},
            "F_seen_train": {"h_post": f_post, "groups": m.loc[f_ids, "id_group"].to_numpy()}}


def correspondence(feats, real_ids: list[str], fake_ids: list[str]) -> dict:
    """Rank (1 = most similar) of each synthetic image's own real counterpart among all given reals,
    by cosine similarity of L2-normalized post-LN tokens. Diagnostic only."""
    m = manifest()
    _, r = feats.take(real_ids)
    _, f = feats.take(fake_ids)
    sim = D.l2n(f) @ D.l2n(r).T
    col = {m.loc[x, "source_id"]: i for i, x in enumerate(real_ids)}
    own = np.array([col[m.loc[x, "source_id"]] for x in fake_ids])
    own_sim = sim[np.arange(len(fake_ids)), own]
    rank = 1 + (sim > own_sim[:, None]).sum(axis=1)
    return {"n_fake": len(fake_ids), "n_real": len(real_ids), "top1": float((rank == 1).mean()),
            "top5": float((rank <= 5).mean()), "median_rank": float(np.median(rank)),
            "chance_top1": 1.0 / len(real_ids), "chance_median_rank": (len(real_ids) + 1) / 2}


def records(scores: dict, taus: dict, ids: list[str], *, role: str, analysis: str, backbone: str, condition: str,
            run_id: str, fit: D.Fitted) -> pd.DataFrame:
    core = config.load("core")
    m = manifest().loc[ids]
    key = ROLES[role]["key"]
    rows = []
    for method, s in scores.items():
        tau = taus[method]
        sel_C = fit.b1.C if method == "B1" else (fit.b1_c1.C if method == "B1-C1" else np.nan)
        rows.append(pd.DataFrame({
            "run_id": run_id, "experiment": "S-BF" if role == "final" else "S-BF-pilot", "analysis": analysis,
            "config_hash": config.config_hash(), "supp_config_hash": S.supp_config_hash(), "dataset_revision": "bfree-local",
            "source_split": m["split_role"].to_numpy(), "row_id": ids, "source_id": m["source_id"].to_numpy(),
            "id_group": m["id_group"].to_numpy(), "stratum": m["stratum"].to_numpy(), "image_hash": m["rgb_sha1"].to_numpy(),
            "partition": m[key].to_numpy(), "authenticity_label": m["label"].astype(int).to_numpy(),
            "generator": m["class"].to_numpy(), "data_condition": condition, "feature_stage": D.FEATURE_STAGE[method],
            "perturbation": "clean", "encoding_condition": "none", "method": method, "backbone": backbone,
            "score": np.asarray(s, np.float64), "threshold": tau, "predicted_label": (np.asarray(s) > tau).astype(int),
            "selected_C": sel_C, "encoder_revision": core["encoder"][backbone]["repo_commit"],
            "random_seed": int(S.cfg()["splits"]["seed"]), "git_commit": P.git_commit(),
        }))
    return pd.concat(rows, ignore_index=True)


def run_cell(feats, role: str, analysis: str, backbone: str, condition: str, run_id: str) -> tuple[pd.DataFrame, dict]:
    r = ROLES[role]
    m = manifest()
    fit = D.fit_methods(training_inputs(feats, role, analysis), [])
    cal = part_ids(role, r["cal"], analysis)
    if not (m.loc[cal, "class"] == "REAL").all():
        raise AssertionError("calibration rows must be real")
    h_pre, h_post = feats.take(cal)
    taus = {k: calibrate.threshold(s) for k, s in D.score_all(fit, h_pre, h_post, cal).items()}
    real = part_ids(role, r["real_test"], analysis)
    fake = sorted(i for p in r["fake_test"] for i in part_ids(role, p, analysis))
    train_ids = set(m.loc[part_ids(role, r["fit_real"], analysis), "source_id"])
    for ids in (cal, real, fake):
        if train_ids & set(m.loc[ids, "source_id"]):
            raise AssertionError("a source id is shared between training and calibration or test")
    if set(m.loc[cal, "source_id"]) & set(m.loc[real + fake, "source_id"]):
        raise AssertionError("a source id is shared between calibration and test")
    out = []
    for ids in (cal, real + fake):
        hp, hq = feats.take(ids)
        out.append(records(D.score_all(fit, hp, hq, ids), taus, ids, role=role, analysis=analysis, backbone=backbone,
                           condition=condition, run_id=run_id, fit=fit))
    corr = {}
    for p in r["fake_test"]:
        cls = m.loc[part_ids(role, p, analysis)[0], "class"]
        corr[f"{p}:{cls}"] = correspondence(feats, real, part_ids(role, p, analysis))
    diag = {"role": role, "analysis": analysis, "backbone": backbone, "condition": condition, "thresholds": taus,
            "n_fit_real": len(fit.o1.ref_ids), "n_cal": len(cal), "n_real_test": len(real), "n_fake_test": len(fake),
            "correspondence": corr, **fit.diagnostics()}
    return pd.concat(out, ignore_index=True), diag


def run(role: str, run_id: str, commit: str, outdir) -> None:
    diags = {"run_id": run_id, "git_commit": commit, "role": role, "supp_config_hash": S.supp_config_hash(), "cells": []}
    for backbone, cond in F.CELLS:
        feats = Feats(backbone, cond)
        preds = []
        for analysis in ANALYSES:
            df, d = run_cell(feats, role, analysis, backbone, cond, run_id)
            preds.append(df)
            diags["cells"].append(d)
        P.save(pd.concat(preds, ignore_index=True), outdir / f"{role}_{backbone}_{cond}.csv.gz")
        print(f"{role} {backbone}/{cond}: saved", flush=True)
    (outdir / f"{role}_diagnostics.json").write_text(json.dumps(diags, indent=1, default=float) + "\n")


def outputs(role: str, outdir) -> list:
    return [outdir / f"{role}_{b}_{c}.csv.gz" for b, c in F.CELLS] + [outdir / f"{role}_diagnostics.json"]


def main() -> None:
    role = "pilot" if "--pilot" in sys.argv else "final"
    outdir = OUT / "pilot" if role == "pilot" else OUT
    commit = P.assert_new_run_allowed(outputs(role, outdir))
    outdir.mkdir(parents=True, exist_ok=True)
    run(role, P.new_run_id(f"sbf-{role}"), commit, outdir)


if __name__ == "__main__":
    main()

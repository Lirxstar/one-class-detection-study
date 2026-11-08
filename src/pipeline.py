"""Shared experiment pipeline: inputs by partition, fitting, real-only calibration,
scoring, and prediction records (PROTOCOL.md Sections 5, 6, 10).

Roles map partitions to their use. The final role uses validation data; the
pilot role uses train-only data (PROTOCOL.md 9.3):
  final: calibrate on R_cal; evaluate R_test vs F_seen_test and F_unseen_test
  pilot: calibrate on P_cal; evaluate P_test vs P_fake (VQDM, Wukong)
Fitting always uses R_train (O1, O2, B1 label 0) and F_seen_train (B1 label 1).
"""
from __future__ import annotations

import json
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import calibrate
import config
import detectors as D
from config import ROOT
from extract_features import load_features

PRED_DIR = ROOT / "artifacts" / "predictions"

ROLES = {
    "final": {"cal": "R_cal", "real_test": "R_test", "fake_test": ["F_seen_test", "F_unseen_test"]},
    "pilot": {"cal": "P_cal", "real_test": "P_test", "fake_test": ["P_fake"]},
}


@lru_cache(maxsize=1)
def splits() -> dict:
    return json.loads((ROOT / "artifacts" / "splits.json").read_text())


@lru_cache(maxsize=1)
def meta() -> pd.DataFrame:
    m = pd.read_csv(ROOT / "artifacts" / "manifest.csv",
                    usecols=["row_id", "split", "label", "generator", "rgb_sha1", "dup_group", "partition"])
    return m.set_index("row_id")


def part_ids(name: str) -> list[str]:
    v = splits()["partitions"][name]
    return sorted(r for lst in v.values() for r in lst) if isinstance(v, dict) else sorted(v)


class Features:
    """Row-id addressable view of one feature file."""

    def __init__(self, backbone: str, condition: str, variant: str):
        f = load_features(backbone, condition, variant)
        self.backbone, self.condition, self.variant = backbone, condition, variant
        self.pos = {r: i for i, r in enumerate(f["row_ids"])}
        self.h_pre, self.h_post = f["h_pre"], f["h_post"]

    def take(self, ids) -> tuple[np.ndarray, np.ndarray]:
        idx = [self.pos[r] for r in ids]
        return self.h_pre[idx], self.h_post[idx]


def training_inputs(feats_real: Features, feats_fake: Features | None = None) -> dict:
    """Exactly R_train and F_seen_train; labels come from the partition, never from a generator field."""
    feats_fake = feats_fake or feats_real
    groups = meta()["dup_group"]
    r_ids, f_ids = part_ids("R_train"), part_ids("F_seen_train")
    _, r_post = feats_real.take(r_ids)
    _, f_post = feats_fake.take(f_ids)
    return {"R_train": {"h_post": r_post, "row_ids": np.array(r_ids), "groups": groups.loc[r_ids].to_numpy()},
            "F_seen_train": {"h_post": f_post, "groups": groups.loc[f_ids].to_numpy()}}


def m0x_dims(backbone: str) -> list[int]:
    p = ROOT / "artifacts" / "audit" / f"dimension_check_{backbone}.json"
    d = json.loads(p.read_text())
    return d["dominant_dims"] if d["ablation_triggered"] else []


def thresholds(fit: D.Fitted, cal: Features, cal_ids: list[str]) -> dict[str, float]:
    """Real-only thresholds from clean calibration rows of the given role."""
    assert (meta().loc[cal_ids, "label"] == 0).all(), "calibration rows must be real"
    h_pre, h_post = cal.take(cal_ids)
    return {m: calibrate.threshold(s) for m, s in D.score_all(fit, h_pre, h_post, cal_ids).items()}


def git_commit() -> str:
    # A copied project must not claim the enclosing original repository's HEAD.
    from provenance import code_revision
    return code_revision()


def records(scores: dict[str, np.ndarray], taus: dict[str, float], ids: list[str], *, experiment: str, run_id: str,
            backbone: str, condition: str, perturbation: str, encoding: str, fit: D.Fitted) -> pd.DataFrame:
    core = config.load("core")
    m = meta().loc[ids]
    enc_rev = core["encoder"][backbone]["repo_commit"]
    rows = []
    for method, s in scores.items():
        tau = taus[method]
        sel_C = fit.b1.C if method == "B1" else (fit.b1_c1.C if method == "B1-C1" else np.nan)
        rows.append(pd.DataFrame({
            "run_id": run_id, "experiment": experiment, "config_hash": config.config_hash(),
            "dataset_revision": core["data"]["revision"], "source_split": m["split"].to_numpy(), "row_id": ids,
            "image_hash": m["rgb_sha1"].to_numpy(), "partition": m["partition"].to_numpy(),
            "authenticity_label": m["label"].to_numpy(), "generator": m["generator"].to_numpy(),
            "data_condition": condition, "feature_stage": D.FEATURE_STAGE[method], "perturbation": perturbation,
            "encoding_condition": encoding, "method": method, "backbone": backbone, "score": np.asarray(s, np.float64),
            "threshold": tau, "predicted_label": (np.asarray(s) > tau).astype(int), "selected_C": sel_C,
            "encoder_revision": enc_rev, "random_seed": int(core["seeds"]["split"]), "git_commit": git_commit(),
        }))
    return pd.concat(rows, ignore_index=True)


def eval_ids(role: str) -> tuple[list[str], list[str]]:
    r = ROLES[role]
    real = part_ids(r["real_test"])
    fake = sorted(i for p in r["fake_test"] for i in part_ids(p))
    return real, fake


def run_clean(role: str, backbone: str, condition: str, experiment: str, run_id: str):
    """E1 protocol for one condition: fit, calibrate, score cal + test rows. Returns (preds, fit, taus, diag)."""
    feats = Features(backbone, condition, "clean")
    fit = D.fit_methods(training_inputs(feats), m0x_dims(backbone))
    cal_ids = part_ids(ROLES[role]["cal"])
    taus = thresholds(fit, feats, cal_ids)
    real, fake = eval_ids(role)
    out = []
    for ids in (cal_ids, real + fake):
        h_pre, h_post = feats.take(ids)
        out.append(records(D.score_all(fit, h_pre, h_post, ids), taus, ids, experiment=experiment, run_id=run_id,
                           backbone=backbone, condition=condition, perturbation="clean", encoding="none", fit=fit))
    diag = {"role": role, "backbone": backbone, "condition": condition, "thresholds": taus, **fit.diagnostics()}
    return pd.concat(out, ignore_index=True), fit, taus, diag


def run_e2(role: str, condition: str, fit: D.Fitted, taus: dict, experiment: str, run_id: str, transforms) -> pd.DataFrame:
    """E2: perturbed test views scored with the clean models and thresholds of the same condition."""
    real, fake = eval_ids(role)
    ids = real + fake
    out = []
    for t in transforms:
        feats = Features("primary", condition, t)
        h_pre, h_post = feats.take(ids)
        out.append(records(D.score_all(fit, h_pre, h_post, ids), taus, ids, experiment=experiment, run_id=run_id,
                           backbone="primary", condition=condition, perturbation=t, encoding="none", fit=fit))
    return pd.concat(out, ignore_index=True)


def run_e3c(role: str, experiment: str, run_id: str, c0_png_equal: bool) -> tuple[pd.DataFrame, dict]:
    """E3C on D-ctrl views. C0: PNG (bitwise equal to clean, so clean features are reused).
    C1: real train/cal JPEG Q75, SD15 train PNG; variants biased, matched, inverted reuse the C1 fit and thresholds."""
    if not c0_png_equal:
        raise RuntimeError("C0 PNG views differ from D-ctrl views; separate C0 features required by protocol")
    clean = Features("primary", "d_ctrl", "clean")
    jpeg = Features("primary", "d_ctrl", "c1_jpeg75")
    cal_ids = part_ids(ROLES[role]["cal"])
    real, fake = eval_ids(role)
    dims = m0x_dims("primary")
    out, diag = [], {}

    def score_mixed(fit, taus, real_src, fake_src, enc):
        rows = []
        for ids, src in ((real, real_src), (fake, fake_src)):
            h_pre, h_post = src.take(ids)
            rows.append(records(D.score_all(fit, h_pre, h_post, ids), taus, ids, experiment=experiment,
                                run_id=run_id, backbone="primary", condition="d_ctrl", perturbation="clean",
                                encoding=enc, fit=fit))
        return rows

    # C0: common PNG encoding
    fit0 = D.fit_methods(training_inputs(clean, clean), dims)
    tau0 = thresholds(fit0, clean, cal_ids)
    h_pre, h_post = clean.take(cal_ids)
    out.append(records(D.score_all(fit0, h_pre, h_post, cal_ids), tau0, cal_ids, experiment=experiment, run_id=run_id,
                       backbone="primary", condition="d_ctrl", perturbation="clean", encoding="C0_cal", fit=fit0))
    out += score_mixed(fit0, tau0, clean, clean, "C0_common")
    diag["C0"] = {"thresholds": tau0, **fit0.diagnostics()}
    # C1: injected JPEG cue on real training and calibration views
    fit1 = D.fit_methods(training_inputs(jpeg, clean), dims)
    tau1 = thresholds(fit1, jpeg, cal_ids)
    h_pre, h_post = jpeg.take(cal_ids)
    out.append(records(D.score_all(fit1, h_pre, h_post, cal_ids), tau1, cal_ids, experiment=experiment, run_id=run_id,
                       backbone="primary", condition="d_ctrl", perturbation="clean", encoding="C1_cal", fit=fit1))
    out += score_mixed(fit1, tau1, jpeg, clean, "C1_biased")
    out += score_mixed(fit1, tau1, jpeg, jpeg, "C1_matched")
    out += score_mixed(fit1, tau1, clean, jpeg, "C1_inverted")
    diag["C1"] = {"thresholds": tau1, **fit1.diagnostics()}
    return pd.concat(out, ignore_index=True), diag


def new_run_id(experiment: str) -> str:
    return f"{experiment}-{uuid.uuid4().hex[:8]}"


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip", lineterminator="\n")


# --------------------------------------------------------------------------- public-run safety guard
def assert_new_run_allowed(outputs: list[Path]) -> str:
    """Preserve bundled reference predictions by refusing to overwrite output files."""
    existing = [str(p) for p in outputs if p.exists()]
    if existing:
        raise RuntimeError("Refusing to overwrite existing experiment outputs: " + str(existing) +
                           ". Use a disposable working copy with reference predictions moved aside.")
    return git_commit()

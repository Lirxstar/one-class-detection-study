"""Metadata-only baseline on stored-file properties (PROTOCOL.md 5.5, E3A).

Features: one-hot format (JPEG, PNG, other), width, height, aspect ratio w/h,
bytes per pixel, estimated JPEG quality (0 when not JPEG), is_jpeg.
StandardScaler (training statistics) then logistic regression with C chosen by
the B1 rule (grouped 5-fold CV inside the training rows).
Primary: R_train vs F_seen_train (SD15). Supplementary: all decodable,
non-excluded train rows of every generator (manifest column `supp_meta`).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import calibrate
import config
import detectors as D
from config import ROOT

COLUMNS = ["fmt_jpeg", "fmt_png", "fmt_other", "width", "height", "aspect_ratio", "bytes_per_pixel",
           "jpeg_quality_or_0", "is_jpeg"]


def load_manifest() -> pd.DataFrame:
    return pd.read_csv(ROOT / "artifacts" / "manifest.csv", keep_default_na=False, na_values=[""]).set_index("row_id")


def design(m: pd.DataFrame) -> np.ndarray:
    fmt = m["format"].astype(str)
    is_jpeg = (fmt == "JPEG").astype(float)
    X = pd.DataFrame({
        "fmt_jpeg": is_jpeg, "fmt_png": (fmt == "PNG").astype(float),
        "fmt_other": (~fmt.isin(["JPEG", "PNG"])).astype(float),
        "width": m["width"].astype(float), "height": m["height"].astype(float),
        "aspect_ratio": m["width"].astype(float) / m["height"].astype(float),
        "bytes_per_pixel": m["bytes_per_pixel"].astype(float),
        "jpeg_quality_or_0": np.where(is_jpeg == 1, pd.to_numeric(m["jpeg_q"], errors="coerce").fillna(0), 0.0),
        "is_jpeg": is_jpeg,
    }, index=m.index)[COLUMNS]
    return X.to_numpy(np.float64)


def select_metadata_C(X: np.ndarray, y: np.ndarray, groups: np.ndarray, C_grid=None):
    """Select C with a fresh scaler fitted on each CV training fold only.

    Post-review correction: fitting one scaler before splitting exposes the
    held-out fold's feature distribution to preprocessing. No validation-split
    row is involved in either the original implementation or this correction.
    """
    if C_grid is None:
        C_grid = config.load("core")["methods"]["B1"]["C_grid"]
    folds = D.b1_cv_folds(y, groups)
    best_C, best, table = None, -np.inf, []
    for C in sorted(float(c) for c in C_grid):
        aucs, iters = [], []
        for tr, te in folds:
            scaler = StandardScaler().fit(X[tr])
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model = D._lr(C).fit(scaler.transform(X[tr]), y[tr])
            aucs.append(float(roc_auc_score(y[te], model.decision_function(scaler.transform(X[te])))))
            iters.append(int(model.n_iter_[0]))
        mean = float(np.mean(aucs))
        table.append({"C": C, "mean_auroc": mean, "fold_auroc": aucs, "max_n_iter": max(iters)})
        if mean > best:
            best_C, best = C, mean
    return best_C, table


class MetaBaseline:
    def __init__(self, m: pd.DataFrame, train_ids: list[str]):
        rows = m.loc[train_ids]
        if not rows["split"].eq("train").all():
            raise ValueError("metadata fitting and CV accept training-split rows only")
        y = (rows["label"].to_numpy() == 1).astype(int)
        X = design(rows)
        self.C, self.cv_table = select_metadata_C(X, y, rows["dup_group"].to_numpy())
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            self.model = D._lr(self.C).fit(Xs, y)
        self.n_train = {"real": int((y == 0).sum()), "synthetic": int((y == 1).sum())}

    def score(self, m: pd.DataFrame, ids) -> np.ndarray:
        return self.model.decision_function(self.scaler.transform(design(m.loc[list(ids)]))).astype(np.float64)

    def diagnostics(self) -> dict:
        return {"selected_C": self.C, "cv_table": self.cv_table, "n_train": self.n_train,
                "preprocessing_cv": "StandardScaler fitted independently within each training fold",
                "coef_standardized": dict(zip(COLUMNS, map(float, self.model.coef_[0]))),
                "intercept": float(self.model.intercept_[0])}


def train_ids(m: pd.DataFrame, which: str) -> list[str]:
    from pipeline import part_ids
    if which == "primary":
        return part_ids("R_train") + part_ids("F_seen_train")
    if which == "supplementary":
        s = m[(m["split"] == "train") & m["supp_meta"].astype(str).str.lower().eq("true")]
        return sorted(s.index)
    raise ValueError(which)


def records(m: pd.DataFrame, ids: list[str], scores: np.ndarray, tau: float, *, method: str, experiment: str,
            run_id: str, sel_C: float, git_commit: str, partition_override: str | None = None) -> pd.DataFrame:
    core = config.load("core")
    r = m.loc[ids]
    return pd.DataFrame({
        "run_id": run_id, "experiment": experiment, "config_hash": config.config_hash(),
        "dataset_revision": core["data"]["revision"], "source_split": r["split"].to_numpy(), "row_id": ids,
        "image_hash": r["rgb_sha1"].to_numpy(),
        "partition": (partition_override if partition_override else r["partition"].fillna("").to_numpy()),
        "authenticity_label": r["label"].to_numpy(), "generator": r["generator"].to_numpy(),
        "data_condition": "stored_file", "feature_stage": "file_metadata", "perturbation": "clean",
        "encoding_condition": "none", "method": method, "backbone": "none", "score": scores, "threshold": tau,
        "predicted_label": (scores > tau).astype(int), "selected_C": sel_C, "encoder_revision": "none",
        "random_seed": int(core["seeds"]["split"]), "git_commit": git_commit,
    })


def run(role: str, experiment: str, run_id: str, git_commit: str, include_supplementary: bool) -> tuple[pd.DataFrame, dict]:
    from pipeline import ROLES, eval_ids, part_ids
    m = load_manifest()
    cal_ids = part_ids(ROLES[role]["cal"])
    real, fake = eval_ids(role)
    out, diag = [], {}
    variants = ["primary"] + (["supplementary"] if include_supplementary else [])
    for which in variants:
        mb = MetaBaseline(m, train_ids(m, which))
        name = f"META-{which}"
        tau = calibrate.threshold(mb.score(m, cal_ids))
        for ids in (cal_ids, real + fake):
            out.append(records(m, ids, mb.score(m, ids), tau, method=name, experiment=experiment, run_id=run_id,
                               sel_C=mb.C, git_commit=git_commit))
        if which == "supplementary" and role == "final":
            v = m[(m["split"] == "validation") & m["supp_meta"].astype(str).str.lower().eq("true")]
            ids = sorted(v.index)
            out.append(records(m, ids, mb.score(m, ids), tau, method=name, experiment=experiment, run_id=run_id,
                               sel_C=mb.C, git_commit=git_commit, partition_override="ALL_VALIDATION"))
        diag[name] = {"threshold": tau, **mb.diagnostics()}
    return pd.concat(out, ignore_index=True), diag

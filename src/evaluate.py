"""Metrics from prediction records only (PROTOCOL.md Section 8).

Every function takes a predictions DataFrame (as saved in artifacts/predictions)
and recomputes metrics from scores and stored thresholds; nothing is read from
the backbone or the feature cache.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

UNSEEN = ["ADM", "GLIDE", "Midjourney"]
SEEN = "SD15"
REAL_TEST = {"final": "R_test", "pilot": "P_test"}
KEYS = ["experiment", "backbone", "data_condition", "perturbation", "encoding_condition", "method"]


def auroc(real: np.ndarray, fake: np.ndarray) -> float:
    """Mann-Whitney AUROC, synthetic = positive, ties count one half."""
    real, fake = np.asarray(real, np.float64), np.asarray(fake, np.float64)
    r = rankdata(np.r_[real, fake])
    n0, n1 = len(real), len(fake)
    return float((r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def average_precision(real: np.ndarray, fake: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    y = np.r_[np.zeros(len(real)), np.ones(len(fake))]
    return float(average_precision_score(y, np.r_[real, fake]))


def split_eval(df: pd.DataFrame, role: str = "final") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Real test rows and synthetic test rows of one prediction block (calibration rows dropped)."""
    real = df[(df["partition"] == REAL_TEST[role]) & (df["authenticity_label"] == 0)]
    fake = df[df["authenticity_label"] == 1]
    return real, fake


def per_generator(df: pd.DataFrame, role: str = "final") -> pd.DataFrame:
    """One row per (KEYS, generator): AUROC vs the shared real test set, TPR, real FPR, counts."""
    out = []
    for key, g in df.groupby(KEYS, sort=False):
        real, fake = split_eval(g, role)
        r = real["score"].to_numpy()
        fpr_k, n_r = int(real["predicted_label"].sum()), len(real)
        tau = float(g["threshold"].iloc[0])
        for gen, f in fake.groupby("generator", sort=True):
            s = f["score"].to_numpy()
            out.append({**dict(zip(KEYS, key)), "generator": gen, "auroc": auroc(r, s), "ap": average_precision(r, s),
                        "prevalence": len(s) / (len(s) + n_r), "tpr": f["predicted_label"].mean(),
                        "tp": int(f["predicted_label"].sum()), "n_fake": len(f), "fpr": fpr_k / n_r, "fp": fpr_k,
                        "n_real": n_r, "threshold": tau})
    return pd.DataFrame(out)


def summary(pg: pd.DataFrame, unseen=UNSEEN, seen=SEEN) -> pd.DataFrame:
    """Mean unseen AUROC (with components), mean unseen TPR, real FPR, Delta_B1."""
    rows = []
    for key, g in pg.groupby(KEYS, sort=False):
        a = g.set_index("generator")
        u = [x for x in unseen if x in a.index]
        row = {**dict(zip(KEYS, key)), "real_fpr": a["fpr"].iloc[0], "fp": int(a["fp"].iloc[0]),
               "n_real": int(a["n_real"].iloc[0])}
        for x in a.index:
            row[f"auroc_{x}"] = a.loc[x, "auroc"]
            row[f"tpr_{x}"] = a.loc[x, "tpr"]
        if u:
            row["mean_unseen_auroc"] = float(a.loc[u, "auroc"].mean())
            row["mean_unseen_tpr"] = float(a.loc[u, "tpr"].mean())
        if seen in a.index and u:
            row["delta_seen_minus_unseen"] = float(a.loc[seen, "auroc"] - row["mean_unseen_auroc"])
        rows.append(row)
    return pd.DataFrame(rows)


def jaccard_flags(df: pd.DataFrame, methods: list[str], role: str = "final") -> pd.DataFrame:
    """Jaccard index of flagged real test images between each pair of methods."""
    real = df[(df["partition"] == REAL_TEST[role]) & (df["authenticity_label"] == 0)]
    flagged = {m: set(real[(real["method"] == m) & (real["predicted_label"] == 1)]["row_id"]) for m in methods}
    rows = []
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            inter, union = len(flagged[a] & flagged[b]), len(flagged[a] | flagged[b])
            rows.append({"method_a": a, "method_b": b, "n_a": len(flagged[a]), "n_b": len(flagged[b]),
                         "intersection": inter, "union": union, "jaccard": inter / union if union else np.nan})
    return pd.DataFrame(rows)


def calibration_check(df: pd.DataFrame, cal_partition: str = "R_cal") -> pd.DataFrame:
    """Recompute every stored threshold from the stored calibration scores (consistency check)."""
    import calibrate
    out = []
    for key, g in df[df["partition"] == cal_partition].groupby(KEYS, sort=False):
        tau = calibrate.threshold(g["score"].to_numpy())
        out.append({**dict(zip(KEYS, key)), "n_cal": len(g), "tau_recomputed": tau,
                    "tau_stored": float(g["threshold"].iloc[0]), "cal_fpr": float((g["score"] > tau).mean())})
    return pd.DataFrame(out)


def jaccard_chance(n_a: int, n_b: int, observed: int, n: int) -> dict:
    """Jaccard of two independent random subsets of sizes n_a, n_b drawn from n items (hypergeometric
    number of shared items K): mean, 2.5/97.5 percentiles, and tail probabilities of the observed K."""
    from scipy.stats import hypergeom
    k = np.arange(0, min(n_a, n_b) + 1)
    p = hypergeom.pmf(k, n, n_a, n_b)
    with np.errstate(invalid="ignore", divide="ignore"):
        j = np.where(n_a + n_b - k > 0, k / (n_a + n_b - k), 0.0)
    cdf = np.cumsum(p)
    return {"chance_mean": float((p * j).sum()), "chance_lo": float(j[np.searchsorted(cdf, 0.025)]),
            "chance_hi": float(j[min(np.searchsorted(cdf, 0.975), len(j) - 1)]),
            "expected_intersection": float(n_a * n_b / n),
            "p_upper": float(hypergeom.sf(observed - 1, n, n_a, n_b)), "p_lower": float(hypergeom.cdf(observed, n, n_a, n_b))}

#!/usr/bin/env python3
"""All tables and statistics from artifacts/predictions only (PROTOCOL.md Sections 7, 8; W6 to W9).

Writes CSV and Markdown tables to artifacts/tables/ and a flat `note_values.json`
used to fill the manuscript. Every interval is a paired percentile bootstrap
over test identities (src/bootstrap.py). Run after any final experiment; it
builds whatever the available predictions allow.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import bootstrap as BS
import evaluate as E
from config import ROOT

PRED = ROOT / "artifacts" / "predictions"
TAB = ROOT / "artifacts" / "tables"
CORE = ["B1", "O1", "O2", "M0"]
EXPLORATORY = ["B1-C1", "O1-k1", "O1-k10", "M0-L2", "M0-ratio", "M0-x"]
GENS = ["SD15", "ADM", "GLIDE", "Midjourney"]
UNSEEN = ["ADM", "GLIDE", "Midjourney"]
USES_LABELS = {"B1": "Yes (SD15 train)", "O1": "No", "O2": "No", "M0": "No", "B1-C1": "Yes (SD15 train)",
               "O1-k1": "No", "O1-k10": "No", "M0-L2": "No", "M0-ratio": "No", "M0-x": "No",
               "META-primary": "Yes (SD15 train)", "META-supplementary": "Yes (all train generators)"}
COND_NAME = {"d_ctrl": "D-ctrl", "d_raw": "D-raw", "d_jpeg": "D-jpeg"}
E2_ORDER = ["clean", "jpeg95", "jpeg75", "jpeg50", "blur0.5", "blur1.0"]
E2_NAME = {"clean": "Clean", "jpeg95": "JPEG 95", "jpeg75": "JPEG 75", "jpeg50": "JPEG 50", "blur0.5": "Blur 0.5",
           "blur1.0": "Blur 1.0"}
VALUES: dict[str, object] = {}


def load(pattern: str) -> pd.DataFrame:
    files = sorted(PRED.glob(pattern))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def f3(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def fci(x, lo, hi) -> str:
    return f"{x:.3f} [{lo:.3f}, {hi:.3f}]"


def pct(k: int, n: int) -> str:
    return f"{100 * k / n:.1f}% ({k}/{n})"


def rate_ci(k: int, n: int, boot: np.ndarray) -> str:
    lo, hi = BS.ci(boot)
    return f"{pct(k, n)} [{100 * lo:.1f}, {100 * hi:.1f}]"


def generator_rows(p: dict, b: dict, **labels) -> list[dict]:
    """Machine-readable per-generator metrics, counts, and conditional test CIs."""
    rows = []
    for gen in GENS:
        if f"auroc_{gen}" not in p:
            continue
        row = {**labels, "generator": gen, "n_real": p["n_real"], "fp": p["fp"],
               "n_fake": p[f"n_{gen}"], "tp": p[f"tp_{gen}"]}
        for name, key in (("auroc", f"auroc_{gen}"), ("tpr", f"tpr_{gen}"), ("fpr", "fpr")):
            row[name] = p[key]
            row[name + "_ci_lo"], row[name + "_ci_hi"] = BS.ci(b[key])
        rows.append(row)
    return rows


def paired_metric_rows(pa: dict, ba: dict, pb: dict, bb: dict, **labels) -> list[dict]:
    """A minus B using shared identity draws; rates and rate changes are fractions."""
    metrics = [f"{metric}_{g}" for g in GENS for metric in ("auroc", "tpr")]
    metrics += ["mean_unseen_auroc", "mean_unseen_tpr", "fpr"]
    rows = []
    for key in metrics:
        if key not in pa or key not in pb:
            continue
        lo, hi = BS.ci(ba[key] - bb[key])
        rows.append({**labels, "metric": key, "estimate": pa[key] - pb[key], "ci_lo": lo, "ci_hi": hi})
    return rows


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out) + "\n"


def write(name: str, df: pd.DataFrame, title: str, note: str = "") -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=False)
    (TAB / f"{name}.md").write_text(f"### {title}\n\n{md(df)}\n{note}\n" if note else f"### {title}\n\n{md(df)}")


# --------------------------------------------------------------------------- bootstrap helpers
_DRAWS: BS.Draws | None = None


def draws(df: pd.DataFrame) -> BS.Draws:
    """One draw set for the final role: shared R_test plus each generator's test identities."""
    global _DRAWS
    if _DRAWS is None:
        real = sorted(df.loc[df["partition"] == "R_test", "row_id"].unique())
        ids = {"REAL": real}
        for g in GENS:
            ids[g] = sorted(df.loc[(df["authenticity_label"] == 1) & (df["generator"] == g), "row_id"].unique())
        _DRAWS = BS.Draws(ids)
    return _DRAWS


def block_vectors(g: pd.DataFrame, dr: BS.Draws, col: str = "score") -> dict[str, np.ndarray]:
    """Score (or flag) vectors aligned with the draw identities for one prediction block."""
    real = g[g["partition"] == "R_test"].set_index("row_id")
    fake = g[g["authenticity_label"] == 1].set_index("row_id")
    out = {"REAL": dr.align("REAL", real, col)}
    for gen in GENS:
        if gen in set(fake["generator"]):
            out[gen] = dr.align(gen, fake, col)
    return out


def boot_block(g: pd.DataFrame, dr: BS.Draws) -> dict[str, np.ndarray]:
    """Bootstrap distributions for one block: auroc_<gen>, tpr_<gen>, fpr, mean_unseen_auroc, mean_unseen_tpr."""
    s = block_vectors(g, dr, "score")
    f = block_vectors(g, dr, "predicted_label")
    cr = dr.counts["REAL"]
    out = {"fpr": BS.boot_rate(f["REAL"], cr)}
    for gen in GENS:
        if gen in s:
            out[f"auroc_{gen}"] = BS.boot_auroc(s["REAL"], s[gen], cr, dr.counts[gen])
            out[f"tpr_{gen}"] = BS.boot_rate(f[gen], dr.counts[gen])
    if all(f"auroc_{u}" in out for u in UNSEEN):
        out["mean_unseen_auroc"] = np.mean([out[f"auroc_{u}"] for u in UNSEEN], axis=0)
        out["mean_unseen_tpr"] = np.mean([out[f"tpr_{u}"] for u in UNSEEN], axis=0)
    if "auroc_SD15" in out and "mean_unseen_auroc" in out:
        out["delta_seen_minus_unseen"] = out["auroc_SD15"] - out["mean_unseen_auroc"]
    return out


def point_block(g: pd.DataFrame) -> dict[str, float]:
    real = g[g["partition"] == "R_test"]
    fake = g[g["authenticity_label"] == 1]
    out = {"fpr": real["predicted_label"].mean(), "fp": int(real["predicted_label"].sum()), "n_real": len(real)}
    for gen, f in fake.groupby("generator"):
        out[f"auroc_{gen}"] = E.auroc(real["score"].to_numpy(), f["score"].to_numpy())
        out[f"tpr_{gen}"] = f["predicted_label"].mean()
        out[f"tp_{gen}"] = int(f["predicted_label"].sum())
        out[f"n_{gen}"] = len(f)
    if all(f"auroc_{u}" in out for u in UNSEEN):
        out["mean_unseen_auroc"] = float(np.mean([out[f"auroc_{u}"] for u in UNSEEN]))
        out["mean_unseen_tpr"] = float(np.mean([out[f"tpr_{u}"] for u in UNSEEN]))
    if "auroc_SD15" in out and "mean_unseen_auroc" in out:
        out["delta_seen_minus_unseen"] = out["auroc_SD15"] - out["mean_unseen_auroc"]
    return out


def long_table(blocks: dict[tuple, pd.DataFrame], dr: BS.Draws) -> pd.DataFrame:
    """Point estimates with bootstrap intervals for every metric of every block."""
    rows = []
    for key, g in blocks.items():
        p, b = point_block(g), boot_block(g, dr)
        for metric, v in p.items():
            if metric.startswith(("fp", "n_", "tp_")) and metric not in ("fpr",):
                continue
            lo, hi = BS.ci(b[metric]) if metric in b else (np.nan, np.nan)
            rows.append({**dict(zip(["backbone", "condition", "perturbation", "encoding", "method"], key)),
                         "metric": metric, "value": v, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def blocks_of(df: pd.DataFrame, methods=None) -> dict[tuple, pd.DataFrame]:
    keys = ["backbone", "data_condition", "perturbation", "encoding_condition", "method"]
    d = df if methods is None else df[df["method"].isin(methods)]
    return {k: g for k, g in d.groupby(keys, sort=False)}


def put(key: str, v) -> None:
    """Store a note value; setting an existing key to a different value is an error (no silent overwrite)."""
    if key in VALUES and json.dumps(VALUES[key], default=float) != json.dumps(v, default=float):
        raise KeyError(f"note value {key} set twice with different values")
    VALUES[key] = v


# --------------------------------------------------------------------------- E1 / E3B
def table1(e1: pd.DataFrame, backbone: str, cond: str, methods: list[str], name: str, title: str) -> pd.DataFrame:
    d = e1[(e1["backbone"] == backbone) & (e1["data_condition"] == cond)]
    dr = draws(e1)
    rows, rows_tpr = [], []
    for m in methods:
        g = d[d["method"] == m]
        if g.empty:
            continue
        p, b = point_block(g), boot_block(g, dr)
        row = {"Method": m, "Uses synthetic labels": USES_LABELS[m]}
        rt = {"Method": m}
        for gen in GENS:
            row[f"{gen}{' (seen by B1)' if gen == 'SD15' else ''}"] = fci(p[f"auroc_{gen}"], *BS.ci(b[f"auroc_{gen}"]))
            rt[gen] = rate_ci(p[f"tp_{gen}"], p[f"n_{gen}"], b[f"tpr_{gen}"])
            for metric in ("auroc", "tpr"):
                put(f"{name}.{m}.{metric}_{gen}", p[f"{metric}_{gen}"])
                put(f"{name}.{m}.{metric}_{gen}.ci", BS.ci(b[f"{metric}_{gen}"]))
        row["Mean unseen AUROC"] = fci(p["mean_unseen_auroc"], *BS.ci(b["mean_unseen_auroc"]))
        row["Real FPR at 5% target"] = pct(p["fp"], p["n_real"])
        rt["Mean unseen TPR"] = f"{100 * p['mean_unseen_tpr']:.1f}%"
        rt["Real FPR"] = rate_ci(p["fp"], p["n_real"], b["fpr"])
        rt["Seen minus unseen AUROC"] = fci(p["delta_seen_minus_unseen"], *BS.ci(b["delta_seen_minus_unseen"]))
        for k in ("mean_unseen_auroc", "mean_unseen_tpr", "fpr", "delta_seen_minus_unseen"):
            put(f"{name}.{m}.{k}", p[k])
            put(f"{name}.{m}.{k}.ci", BS.ci(b[k]))
        put(f"{name}.{m}.fp", p["fp"])
        rows.append(row)
        rows_tpr.append(rt)
    t = pd.DataFrame(rows)
    n_r = int(d[d["partition"] == "R_test"]["row_id"].nunique())
    n_f = {g: int(d[(d["authenticity_label"] == 1) & (d["generator"] == g)]["row_id"].nunique()) for g in GENS}
    note = (f"AUROC with paired percentile bootstrap 95% intervals (2,000 resamples of test identities). "
            f"Shared real test set R_test (n = {n_r}); synthetic test sets: "
            f"{', '.join(f'{g} n = {n}' for g, n in n_f.items())}. Real FPR at the threshold "
            f"calibrated on R_cal (k-th smallest, k = ceil(0.95(n+1))). B1 is the only method that uses synthetic "
            f"labels (SD15 train); M0 uses the pre-LayerNorm class token, the others the L2-normalized post-LN token. "
            f"Source: artifacts/predictions/{'e1' if cond != 'd_jpeg' else 'e3'}/.")
    write(name, t, title, note)
    write(name + "_tpr", pd.DataFrame(rows_tpr), title.replace("AUROC", "TPR") + " (operating point)",
          "TPR = synthetic images flagged at the clean real-calibrated threshold, with counts and 95% intervals in percent.")
    return t


def jaccard(e1: pd.DataFrame, cond: str, backbone: str = "primary") -> None:
    d = e1[(e1["backbone"] == backbone) & (e1["data_condition"] == cond) & (e1["perturbation"] == "clean")]
    j = E.jaccard_flags(d, CORE)
    n_real = int(d.loc[d["partition"] == "R_test", "row_id"].nunique())
    rows = []
    for r in j.itertuples():
        ch = E.jaccard_chance(r.n_a, r.n_b, r.intersection, n_real)
        key = f"jaccard.{backbone}.{cond}.{r.method_a}-{r.method_b}"
        put(key, r.jaccard)
        for k, v in ch.items():
            put(f"{key}.{k}", v)
        put(f"{key}.intersection", int(r.intersection))
        rows.append({"A": r.method_a, "B": r.method_b, "flagged A": r.n_a, "flagged B": r.n_b, "shared": r.intersection,
                     "expected shared (chance)": f"{ch['expected_intersection']:.2f}", "Jaccard": f3(r.jaccard),
                     "chance Jaccard mean [95% range]": f"{ch['chance_mean']:.3f} [{ch['chance_lo']:.3f}, {ch['chance_hi']:.3f}]",
                     "P(shared >= obs)": f"{ch['p_upper']:.3f}", "P(shared <= obs)": f"{ch['p_lower']:.3f}"})
    write(f"jaccard_{backbone}_{cond}", pd.DataFrame(rows), f"Overlap of flagged real test images, {COND_NAME[cond]} ({backbone})",
          f"Jaccard index of the sets of R_test images (n = {n_real}) flagged at each method's clean threshold. Chance "
          "reference (post hoc, PROTOCOL v0.6.4): two independent random sets of the same sizes (hypergeometric); "
          "tail probabilities of the observed number of shared images.")


def method_differences(e1: pd.DataFrame, cond: str, backbone: str = "primary") -> None:
    """Paired AUROC differences between core methods, per generator and mean unseen."""
    d = e1[(e1["backbone"] == backbone) & (e1["data_condition"] == cond)]
    dr = draws(e1)
    boots = {m: boot_block(d[d["method"] == m], dr) for m in CORE}
    pts = {m: point_block(d[d["method"] == m]) for m in CORE}
    rows = []
    for i, a in enumerate(CORE):
        for bm in CORE[i + 1:]:
            for metric in [f"auroc_{g}" for g in GENS] + ["mean_unseen_auroc"]:
                diff = pts[a][metric] - pts[bm][metric]
                lo, hi = BS.ci(boots[a][metric] - boots[bm][metric])
                rows.append({"A": a, "B": bm, "metric": metric, "A minus B": f3(diff), "95% CI": f"[{lo:.3f}, {hi:.3f}]",
                             "excludes 0": bool(lo > 0 or hi < 0)})
                put(f"diff.{backbone}.{cond}.{a}-{bm}.{metric}", diff)
                put(f"diff.{backbone}.{cond}.{a}-{bm}.{metric}.ci", (lo, hi))
    write(f"method_differences_{backbone}_{cond}", pd.DataFrame(rows),
          f"Paired AUROC differences between core methods, {COND_NAME[cond]} ({backbone})",
          "Same bootstrap draws for both methods (paired).")


def ladder(e1: pd.DataFrame, e3b: pd.DataFrame, meta: pd.DataFrame | None) -> None:
    allp = pd.concat([e1[e1["backbone"] == "primary"], e3b], ignore_index=True)
    dr = draws(allp)
    conds = ["d_raw", "d_jpeg", "d_ctrl"]
    pts, boots = {}, {}
    for c in conds:
        for m in CORE + EXPLORATORY:
            g = allp[(allp["data_condition"] == c) & (allp["method"] == m)]
            if not g.empty:
                pts[(c, m)], boots[(c, m)] = point_block(g), boot_block(g, dr)
    ranks = {c: pd.Series({m: pts[(c, m)]["mean_unseen_auroc"] for m in CORE}).rank(ascending=False, method="min")
             for c in conds}
    rows = []
    for m in CORE:
        row = {"Method": m}
        for c in conds:
            row[f"Mean unseen AUROC {COND_NAME[c]}"] = fci(pts[(c, m)]["mean_unseen_auroc"], *BS.ci(boots[(c, m)]["mean_unseen_auroc"]))
            put(f"ladder.{c}.{m}.mean_unseen_auroc", pts[(c, m)]["mean_unseen_auroc"])
            put(f"ladder.{c}.{m}.mean_unseen_auroc.ci", BS.ci(boots[(c, m)]["mean_unseen_auroc"]))
            put(f"ladder.{c}.{m}.fpr", pts[(c, m)]["fpr"])
        for c in conds:
            row[f"Rank {COND_NAME[c]}"] = int(ranks[c][m])
            put(f"ladder.{c}.{m}.rank", int(ranks[c][m]))
        rows.append(row)
    if meta is not None and not meta.empty:
        mp = meta[(meta["method"] == "META-primary") & (meta["partition"] != "ALL_VALIDATION")]
        p, b = point_block(mp), boot_block(mp, dr)
        rows.append({"Method": "Metadata only (stored file)",
                     "Mean unseen AUROC D-raw": fci(p["mean_unseen_auroc"], *BS.ci(b["mean_unseen_auroc"])),
                     "Mean unseen AUROC D-jpeg": "n/a", "Mean unseen AUROC D-ctrl": "n/a",
                     "Rank D-raw": "n/a", "Rank D-jpeg": "n/a", "Rank D-ctrl": "n/a"})
    write("table3_ladder", pd.DataFrame(rows), "Table 3. Mean unseen AUROC along the ladder (ViT-S/14)",
          "Identical identities in all conditions; ranks among the four core methods (1 = highest mean unseen AUROC). "
          "The metadata baseline uses stored-file properties (the D-raw inputs) and has no D-jpeg/D-ctrl version. "
          "Sources: artifacts/predictions/e1/ (D-raw, D-ctrl), e3/ (D-jpeg, metadata).")
    # per generator, with paired condition differences
    rows, drows = [], []
    for m in CORE:
        for gen in GENS:
            row = {"Method": m, "Generator": gen}
            for c in conds:
                row[COND_NAME[c]] = fci(pts[(c, m)][f"auroc_{gen}"], *BS.ci(boots[(c, m)][f"auroc_{gen}"]))
                put(f"ladder.{c}.{m}.auroc_{gen}", pts[(c, m)][f"auroc_{gen}"])
            rows.append(row)
            for a, bcond in (("d_jpeg", "d_raw"), ("d_ctrl", "d_jpeg"), ("d_ctrl", "d_raw")):
                diff = pts[(a, m)][f"auroc_{gen}"] - pts[(bcond, m)][f"auroc_{gen}"]
                lo, hi = BS.ci(boots[(a, m)][f"auroc_{gen}"] - boots[(bcond, m)][f"auroc_{gen}"])
                drows.append({"Method": m, "Generator": gen, "Step": f"{COND_NAME[bcond]} to {COND_NAME[a]}",
                              "AUROC change": f3(diff), "95% CI": f"[{lo:.3f}, {hi:.3f}]"})
                put(f"ladderdiff.{bcond}-{a}.{m}.{gen}", diff)
                put(f"ladderdiff.{bcond}-{a}.{m}.{gen}.ci", (lo, hi))
        for a, bcond in (("d_jpeg", "d_raw"), ("d_ctrl", "d_jpeg"), ("d_ctrl", "d_raw")):
            diff = pts[(a, m)]["mean_unseen_auroc"] - pts[(bcond, m)]["mean_unseen_auroc"]
            lo, hi = BS.ci(boots[(a, m)]["mean_unseen_auroc"] - boots[(bcond, m)]["mean_unseen_auroc"])
            drows.append({"Method": m, "Generator": "mean unseen", "Step": f"{COND_NAME[bcond]} to {COND_NAME[a]}",
                          "AUROC change": f3(diff), "95% CI": f"[{lo:.3f}, {hi:.3f}]"})
            put(f"ladderdiff.{bcond}-{a}.{m}.mean_unseen", diff)
            put(f"ladderdiff.{bcond}-{a}.{m}.mean_unseen.ci", (lo, hi))
    write("table3_ladder_per_generator", pd.DataFrame(rows), "Per-generator AUROC along the ladder (ViT-S/14)",
          "Paired bootstrap intervals; same identities in every condition.")
    write("table3_ladder_differences", pd.DataFrame(drows), "Paired AUROC changes between conditions (ViT-S/14)",
          "Same bootstrap draws across conditions (paired).")
    pd.DataFrame([{"condition": c, "method": m, **pts[(c, m)]} for (c, m) in pts]).to_csv(TAB / "ladder_points_all_methods.csv", index=False)


# --------------------------------------------------------------------------- E2
def e2_tables(e1: pd.DataFrame, e2: pd.DataFrame, methods: list[str], prefix: str = "") -> None:
    for cond in ("d_ctrl", "d_raw"):
        clean = e1[(e1["backbone"] == "primary") & (e1["data_condition"] == cond)]
        d = pd.concat([clean, e2[e2["data_condition"] == cond]], ignore_index=True)
        dr = draws(d)
        fpr_rows, tpr_rows, auc_rows, shift_rows = [], [], [], []
        per_gen, contrasts, image_shifts = [], [], []
        for m in methods:
            dm = d[d["method"] == m]
            if dm.empty:
                continue
            rf, rt, ra = {"Method": m}, {"Method": m}, {"Method": m}
            cb = boot_block(dm[dm["perturbation"] == "clean"], dr)
            cp = point_block(dm[dm["perturbation"] == "clean"])
            creal = dm[(dm["perturbation"] == "clean") & (dm["partition"] == "R_test")].set_index("row_id")
            for t in E2_ORDER:
                g = dm[dm["perturbation"] == t]
                p, b = point_block(g), boot_block(g, dr)
                per_gen += generator_rows(p, b, condition=cond, method=m, transform=t)
                rf[E2_NAME[t]] = f"{pct(p['fp'], p['n_real'])} [{100 * BS.ci(b['fpr'])[0]:.1f}, {100 * BS.ci(b['fpr'])[1]:.1f}]"
                rt[E2_NAME[t]] = f"{100 * p['mean_unseen_tpr']:.1f}% [{100 * BS.ci(b['mean_unseen_tpr'])[0]:.1f}, {100 * BS.ci(b['mean_unseen_tpr'])[1]:.1f}]"
                ra[E2_NAME[t]] = fci(p["mean_unseen_auroc"], *BS.ci(b["mean_unseen_auroc"]))
                for k in ("fpr", "mean_unseen_tpr", "mean_unseen_auroc",
                          *[f"{metric}_{gen}" for gen in GENS for metric in ("auroc", "tpr")]):
                    put(f"e2.{cond}.{m}.{t}.{k}", p[k])
                    put(f"e2.{cond}.{m}.{t}.{k}.ci", BS.ci(b[k]))
                if t != "clean":
                    contrasts += paired_metric_rows(p, b, cp, cb, condition=cond, method=m,
                                                     transform=t, reference="clean")
                    dlo, dhi = BS.ci(b["fpr"] - cb["fpr"])
                    put(f"e2.{cond}.{m}.{t}.delta_fpr", p["fpr"] - cp["fpr"])
                    put(f"e2.{cond}.{m}.{t}.delta_fpr.ci", (dlo, dhi))
                    alo, ahi = BS.ci(b["mean_unseen_auroc"] - cb["mean_unseen_auroc"])
                    put(f"e2.{cond}.{m}.{t}.delta_mean_unseen_auroc", p["mean_unseen_auroc"] - cp["mean_unseen_auroc"])
                    put(f"e2.{cond}.{m}.{t}.delta_mean_unseen_auroc.ci", (alo, ahi))
                    tlo, thi = BS.ci(b["mean_unseen_tpr"] - cb["mean_unseen_tpr"])
                    put(f"e2.{cond}.{m}.{t}.delta_mean_unseen_tpr", p["mean_unseen_tpr"] - cp["mean_unseen_tpr"])
                    put(f"e2.{cond}.{m}.{t}.delta_mean_unseen_tpr.ci", (tlo, thi))
                    real = g[g["partition"] == "R_test"].set_index("row_id")
                    ds = (real["score"] - creal.loc[real.index, "score"]).to_numpy()
                    sd = creal["score"].std()
                    for rid, delta in zip(real.index, ds):
                        image_shifts.append({"condition": cond, "method": m, "transform": t, "row_id": rid,
                                             "clean_score": creal.loc[rid, "score"], "score": real.loc[rid, "score"],
                                             "delta_score": delta, "delta_in_clean_real_sd": delta / sd,
                                             "threshold": real.loc[rid, "threshold"],
                                             "clean_flag": creal.loc[rid, "predicted_label"],
                                             "flag": real.loc[rid, "predicted_label"]})
                    # bootstrap the mean paired shift with the same real draws
                    vec = dr.align("REAL", pd.DataFrame({"x": ds}, index=real.index), "x")
                    bm = dr.counts["REAL"] @ vec / len(vec)
                    shift_rows.append({"Method": m, "Transform": E2_NAME[t], "mean delta s": f"{ds.mean():.4g}",
                                       "95% CI": f"[{BS.ci(bm)[0]:.4g}, {BS.ci(bm)[1]:.4g}]",
                                       "mean delta s / SD(clean real s)": f"{ds.mean() / sd:.3f}",
                                       "share of real scores up": f"{(ds > 0).mean():.3f}",
                                       "delta FPR (pp)": f"{100 * (p['fpr'] - cp['fpr']):+.1f} [{100 * dlo:+.1f}, {100 * dhi:+.1f}]"})
                    put(f"e2.{cond}.{m}.{t}.std_shift", ds.mean() / sd)
            fpr_rows.append(rf); tpr_rows.append(rt); auc_rows.append(ra)
        n = COND_NAME[cond]
        src = "Source: artifacts/predictions/e1/ (clean) and e2/. Thresholds and models frozen from E1 clean."
        tag = "Exploratory variants: " if prefix else ""
        write(f"{prefix}table2_e2_fpr_{cond}", pd.DataFrame(fpr_rows), f"{tag}Real FPR at frozen thresholds, {n}", src)
        write(f"{prefix}table2_e2_tpr_{cond}", pd.DataFrame(tpr_rows), f"{tag}Mean unseen TPR at frozen thresholds, {n}", src)
        write(f"{prefix}table2_e2_auroc_{cond}", pd.DataFrame(auc_rows), f"{tag}Mean unseen AUROC under post-processing, {n}", src)
        write(f"{prefix}e2_score_shift_{cond}", pd.DataFrame(shift_rows), f"{tag}Paired score changes on real test images, {n}",
              "delta s = s(T(r)) - s(r) on the same R_test images; delta FPR with paired bootstrap interval.")
        write(f"{prefix}e2_per_generator_{cond}", pd.DataFrame(per_gen), f"E2 per-generator metrics, {n}",
              "Rates and confidence bounds are fractions; counts accompany every operating point. "
              "Intervals condition on the fitted model and calibration threshold. Shared real identities across generators.")
        write(f"{prefix}e2_paired_changes_{cond}", pd.DataFrame(contrasts), f"E2 paired changes from clean, {n}",
              "Transformed minus clean; shared identity draws. Rate changes are fractions, not percentage points.")
        pd.DataFrame(image_shifts).to_csv(TAB / f"{prefix}e2_real_score_changes_{cond}.csv", index=False)


# --------------------------------------------------------------------------- E3A, E3C
def e3a_tables(meta: pd.DataFrame) -> None:
    dr = draws(meta[meta["partition"] != "ALL_VALIDATION"])
    rows = []
    for m in ("META-primary", "META-supplementary"):
        g = meta[(meta["method"] == m) & (meta["partition"] != "ALL_VALIDATION")]
        if g.empty:
            continue
        p, b = point_block(g), boot_block(g, dr)
        row = {"Model": m, "Trained on": USES_LABELS[m], "Evaluation set": "study partitions (R_test vs test sets)"}
        for gen in GENS:
            row[gen] = fci(p[f"auroc_{gen}"], *BS.ci(b[f"auroc_{gen}"]))
            put(f"e3a.{m}.auroc_{gen}", p[f"auroc_{gen}"])
        row["Mean unseen"] = f3(p["mean_unseen_auroc"])
        put(f"e3a.{m}.mean_unseen_auroc", p["mean_unseen_auroc"])
        rows.append(row)
        av = meta[(meta["method"] == m) & (meta["partition"] == "ALL_VALIDATION")]
        if not av.empty:
            real = av[av["authenticity_label"] == 0]["score"].to_numpy()
            row = {"Model": m, "Trained on": USES_LABELS[m], "Evaluation set": f"all validation rows (real n = {len(real)})"}
            for gen, f in av[av["authenticity_label"] == 1].groupby("generator"):
                row[gen] = f"{E.auroc(real, f['score'].to_numpy()):.3f} (n = {len(f)})"
                put(f"e3a.{m}.allval.auroc_{gen}", E.auroc(real, f["score"].to_numpy()))
            rows.append(row)
    t = pd.DataFrame(rows).fillna("")
    write("e3a_metadata", t, "E3A. Metadata-only baseline (stored-file properties)",
          "Logistic regression on format, width, height, aspect ratio, bytes per pixel and JPEG quality. ADM is stored "
          "as RGBA PNG (alpha 255), which raises its bytes per pixel. Observational: shows how much label information "
          "the stored files carry, not what the visual models use. Source: artifacts/predictions/e3/e3a_metadata.csv.gz.")


def e3c_tables(e3c: pd.DataFrame) -> None:
    ev = e3c[~e3c["encoding_condition"].isin(["C0_cal", "C1_cal"])]
    dr = draws(ev)
    variants = ["C0_common", "C1_biased", "C1_matched", "C1_inverted"]
    names = {"C0_common": "C0 both PNG", "C1_biased": "C1(a) real JPEG75, synth PNG",
             "C1_matched": "C1(b) both JPEG75", "C1_inverted": "C1(c) real PNG, synth JPEG75"}
    rows, drows, per_gen, contrasts, real_shifts, shift_summary = [], [], [], [], [], []
    for m in CORE:
        base = None
        points, boots = {}, {}
        for v in variants:
            g = ev[(ev["method"] == m) & (ev["encoding_condition"] == v)]
            p, b = point_block(g), boot_block(g, dr)
            points[v], boots[v] = p, b
            per_gen += generator_rows(p, b, method=m, variant=v)
            if v == "C0_common":
                base = (p, b)
            rows.append({"Method": m, "Variant": names[v],
                         "Mean unseen AUROC": fci(p["mean_unseen_auroc"], *BS.ci(b["mean_unseen_auroc"])),
                         "SD15 AUROC": fci(p["auroc_SD15"], *BS.ci(b["auroc_SD15"])),
                         "Real FPR": rate_ci(p["fp"], p["n_real"], b["fpr"]),
                         "Mean unseen TPR": f"{100 * p['mean_unseen_tpr']:.1f}% [{100 * BS.ci(b['mean_unseen_tpr'])[0]:.1f}, {100 * BS.ci(b['mean_unseen_tpr'])[1]:.1f}]",
                         "SD15 TPR": rate_ci(p["tp_SD15"], p["n_SD15"], b["tpr_SD15"])})
            for k in ("mean_unseen_auroc", "auroc_SD15", "fpr", "mean_unseen_tpr", "tpr_SD15"):
                put(f"e3c.{m}.{v}.{k}", p[k])
                put(f"e3c.{m}.{v}.{k}.ci", BS.ci(b[k]))
            if v != "C0_common":
                for k in ("mean_unseen_auroc", "fpr"):
                    lo, hi = BS.ci(b[k] - base[1][k])
                    drows.append({"Method": m, "Variant": names[v], "metric": k, "minus C0": f3(p[k] - base[0][k]),
                                  "95% CI": f"[{lo:.3f}, {hi:.3f}]"})
                    put(f"e3c.{m}.{v}.{k}.minus_c0", p[k] - base[0][k])
                    put(f"e3c.{m}.{v}.{k}.minus_c0.ci", (lo, hi))
        for ia, va in enumerate(variants):
            for vb in variants[ia + 1:]:
                rs = paired_metric_rows(points[va], boots[va], points[vb], boots[vb],
                                        method=m, variant_a=va, variant_b=vb)
                contrasts += rs
                for row in rs:
                    key = f"e3c_pair.{m}.{va}-{vb}.{row['metric']}"
                    put(key, row["estimate"])
                    put(key + ".ci", (row["ci_lo"], row["ci_hi"]))
        dm = ev[(ev["method"] == m) & (ev["partition"] == "R_test")]
        jpg = dm[dm["encoding_condition"] == "C1_biased"].set_index("row_id").loc[dr.ids["REAL"]]
        png = dm[dm["encoding_condition"] == "C1_inverted"].set_index("row_id").loc[dr.ids["REAL"]]
        delta = jpg["score"].to_numpy() - png["score"].to_numpy()
        scale = float(png["score"].std())
        mean_boot = dr.counts["REAL"] @ delta / len(delta)
        lo, hi = BS.ci(mean_boot)
        shift_summary.append({"method": m, "n_real": len(delta), "mean_jpeg75_minus_png": float(delta.mean()),
                              "ci_lo": lo, "ci_hi": hi, "png_real_score_sd": scale,
                              "mean_change_in_png_sd": float(delta.mean()) / scale,
                              "standardized_ci_lo": lo / scale, "standardized_ci_hi": hi / scale})
        for i, rid in enumerate(dr.ids["REAL"]):
            real_shifts.append({"method": m, "row_id": rid, "score_jpeg75": jpg.loc[rid, "score"],
                                "score_png": png.loc[rid, "score"], "delta_score": delta[i],
                                "delta_in_png_real_sd": delta[i] / scale, "threshold": jpg.loc[rid, "threshold"]})
    write("e3c_injected_cue", pd.DataFrame(rows), "E3C. Injected JPEG cue on fixed D-ctrl identities (ViT-S/14)",
          "C0 and C1 fitted and calibrated separately; C1 variants share the C1 model and threshold. "
          "Source: artifacts/predictions/e3/e3c_primary_d_ctrl.csv.gz.")
    write("e3c_differences", pd.DataFrame(drows), "E3C. Paired changes relative to C0", "Same identity draws (paired).")
    write("e3c_per_generator", pd.DataFrame(per_gen), "E3C. Per-generator AUROC and operating points",
          "Rates and bounds are fractions; counts and 95% intervals use the same identity draws.")
    write("e3c_paired_contrasts", pd.DataFrame(contrasts), "E3C. All paired contrasts",
          "variant_a minus variant_b. C1 comparisons hold the model and threshold fixed; C0 comparisons include refitting and recalibration.")
    within = pd.DataFrame(contrasts)
    within = within[within["variant_a"].str.startswith("C1") & within["variant_b"].str.startswith("C1")]
    write("e3c_within_c1", within, "E3C. Paired contrasts within C1",
          "The same fitted C1 model, threshold and image identities are used on both sides. Rates are fractions.")
    write("e3c_real_score_shift", pd.DataFrame(shift_summary), "E3C. Paired real-score changes under the fixed C1 model",
          "JPEG75 minus PNG on each R_test identity. Standardization uses SD of C1 PNG real scores; "
          "no subtraction of scores from separately fitted C0 and C1 models.")
    pd.DataFrame(real_shifts).to_csv(TAB / "e3c_real_score_changes.csv", index=False)


def e3c_decomposition(e3c: pd.DataFrame) -> None:
    """POST HOC (PROTOCOL v0.6.4): within C1 (one model, one threshold) form all four combinations of
    real test encoding x synthetic test encoding from the stored C1 predictions."""
    ev = e3c[~e3c["encoding_condition"].isin(["C0_cal", "C1_cal"])]
    dr = draws(ev)
    src = {"real_png": ("C1_inverted", 0), "real_jpeg": ("C1_biased", 0), "synth_png": ("C1_biased", 1),
           "synth_jpeg": ("C1_matched", 1)}
    cells = {"C1 real PNG, synth PNG": ("real_png", "synth_png"), "C1 real JPEG75, synth PNG = (a)": ("real_jpeg", "synth_png"),
             "C1 real PNG, synth JPEG75 = (c)": ("real_png", "synth_jpeg"), "C1 real JPEG75, synth JPEG75 = (b)": ("real_jpeg", "synth_jpeg")}
    short = {"C1 real PNG, synth PNG": "C1_rpng_spng", "C1 real JPEG75, synth PNG = (a)": "C1_rjpg_spng",
             "C1 real PNG, synth JPEG75 = (c)": "C1_rpng_sjpg", "C1 real JPEG75, synth JPEG75 = (b)": "C1_rjpg_sjpg"}
    rows, drows = [], []
    for m in CORE:
        dm = ev[ev["method"] == m]

        def part(key):
            enc, lab = src[key]
            g = dm[dm["encoding_condition"] == enc]
            return g[(g["partition"] == "R_test")] if lab == 0 else g[g["authenticity_label"] == 1]

        res = {"C0 both PNG": (point_block(dm[dm["encoding_condition"] == "C0_common"]),
                               boot_block(dm[dm["encoding_condition"] == "C0_common"], dr))}
        for name, (rk, sk) in cells.items():
            g = pd.concat([part(rk), part(sk)], ignore_index=True)
            res[name] = (point_block(g), boot_block(g, dr))
        for name, (p, b) in res.items():
            rows.append({"Method": m, "Model / test encoding": name,
                         "SD15 AUROC": fci(p["auroc_SD15"], *BS.ci(b["auroc_SD15"])),
                         "Mean unseen AUROC": fci(p["mean_unseen_auroc"], *BS.ci(b["mean_unseen_auroc"]))})
            key = "C0" if name.startswith("C0") else short[name]
            for k in ("auroc_SD15", "mean_unseen_auroc"):
                put(f"e3cdec.{m}.{key}.{k}", p[k])
        effects = {"training on JPEG75 reals (C1 vs C0, both test sets PNG)": ("C1 real PNG, synth PNG", "C0 both PNG", "train"),
                   "JPEG75 on real test images (synth PNG)": ("C1 real JPEG75, synth PNG = (a)", "C1 real PNG, synth PNG", "real_jpeg"),
                   "JPEG75 on synthetic test images (real PNG)": ("C1 real PNG, synth JPEG75 = (c)", "C1 real PNG, synth PNG", "synth_jpeg")}
        for label, (x, y, key) in effects.items():
            for k in ("auroc_SD15", "mean_unseen_auroc"):
                diff = res[x][0][k] - res[y][0][k]
                lo, hi = BS.ci(res[x][1][k] - res[y][1][k])
                drows.append({"Method": m, "Effect": label, "metric": k, "change": f3(diff), "95% CI": f"[{lo:.3f}, {hi:.3f}]"})
                put(f"e3cdec.{m}.effect_{key}.{k}", diff)
                put(f"e3cdec.{m}.effect_{key}.{k}.ci", (lo, hi))
    note = ("Post hoc (PROTOCOL v0.6.4). Within C1 every cell uses the single C1 model and threshold; real test scores "
            "come from the C1 variants with that real encoding and synthetic scores from the variants with that synthetic "
            "encoding. Paired bootstrap intervals. Source: artifacts/predictions/e3/e3c_primary_d_ctrl.csv.gz.")
    write("e3c_decomposition", pd.DataFrame(rows), "E3C decomposition: model and test encoding (post hoc)", note)
    write("e3c_decomposition_effects", pd.DataFrame(drows), "E3C decomposition: paired effects (post hoc)", note)


def posthoc_native(e1: pd.DataFrame, e3b: pd.DataFrame, ph: pd.DataFrame) -> None:
    """POST HOC (PROTOCOL v0.6.4): D-native = native crop without JPEG harmonization."""
    allp = pd.concat([e1[e1["backbone"] == "primary"], e3b, ph], ignore_index=True)
    dr = draws(allp)
    conds = ["d_raw", "d_jpeg", "d_native", "d_ctrl"]
    names = {**COND_NAME, "d_native": "D-native (post hoc)"}
    pts, boots = {}, {}
    for c in conds:
        for m in CORE:
            g = allp[(allp["data_condition"] == c) & (allp["method"] == m)]
            pts[(c, m)], boots[(c, m)] = point_block(g), boot_block(g, dr)
    rows = []
    for m in CORE:
        row = {"Method": m}
        for c in conds:
            row[names[c]] = fci(pts[(c, m)]["mean_unseen_auroc"], *BS.ci(boots[(c, m)]["mean_unseen_auroc"]))
            if c == "d_native":
                for k in ["mean_unseen_auroc", "fpr", "mean_unseen_tpr"] + [f"auroc_{g}" for g in GENS]:
                    put(f"posthoc.d_native.{m}.{k}", pts[(c, m)][k])
        rows.append(row)
    write("posthoc_native_ladder", pd.DataFrame(rows), "Post hoc: mean unseen AUROC with the D-native cell (ViT-S/14)",
          "D-native: native-resolution crop as D-ctrl but without JPEG harmonization of synthetic images (post hoc, "
          "PROTOCOL v0.6.4, tag posthoc-d-native). Same identities; paired bootstrap intervals. Sources: "
          "artifacts/predictions/e1/, e3/, posthoc/.")
    steps = {"JPEG harmonization at native resolution (D-native to D-ctrl)": ("d_ctrl", "d_native", "harm_native"),
             "JPEG harmonization with downsampling (D-raw to D-jpeg)": ("d_jpeg", "d_raw", "harm_resized"),
             "Resizing and field of view, no harmonization (D-raw to D-native)": ("d_native", "d_raw", "resize_noharm"),
             "Resizing and field of view, harmonized (D-jpeg to D-ctrl)": ("d_ctrl", "d_jpeg", "resize_harm")}
    drows = []
    for m in CORE:
        for label, (x, y, key) in steps.items():
            for k in ["mean_unseen_auroc"] + [f"auroc_{g}" for g in GENS]:
                diff = pts[(x, m)][k] - pts[(y, m)][k]
                lo, hi = BS.ci(boots[(x, m)][k] - boots[(y, m)][k])
                if k == "mean_unseen_auroc" or m in ("B1", "M0"):
                    drows.append({"Method": m, "Step": label, "metric": k, "change": f3(diff), "95% CI": f"[{lo:.3f}, {hi:.3f}]"})
                put(f"posthocdiff.{key}.{m}.{k}", diff)
                put(f"posthocdiff.{key}.{m}.{k}.ci", (lo, hi))
    write("posthoc_native_differences", pd.DataFrame(drows), "Post hoc: paired AUROC changes for each ladder step (ViT-S/14)",
          "Same bootstrap draws in every condition. Per-generator rows shown for B1 and M0; all values in note_values.json.")
    for key in ("harm_native", "harm_resized"):
        put(f"agg.posthoc.{key}.realonly.max_abs", max(abs(VALUES[f"posthocdiff.{key}.{m}.mean_unseen_auroc"]) for m in ("O1", "O2", "M0")))
    put("agg.posthoc.resize_noharm.min_all", min(VALUES[f"posthocdiff.resize_noharm.{m}.mean_unseen_auroc"] for m in CORE))


def posthoc_reseed(e1: pd.DataFrame, rs: pd.DataFrame) -> None:
    """POST HOC (PROTOCOL v0.6.6, proposal 6.3): R_train redrawn with seeds 7, 13, 29; seed 42 is E1."""
    base = e1[(e1["backbone"] == "primary") & (e1["data_condition"] == "d_ctrl")].assign(rtrain_seed=42)
    allp = pd.concat([base, rs], ignore_index=True)
    dr = draws(allp)
    seeds = [42] + sorted(int(s) for s in rs["rtrain_seed"].unique())
    pts, boots = {}, {}
    for s in seeds:
        for m in CORE:
            g = allp[(allp["rtrain_seed"] == s) & (allp["method"] == m)]
            pts[(s, m)], boots[(s, m)] = point_block(g), boot_block(g, dr)
            for k in ["mean_unseen_auroc", "mean_unseen_tpr", "fpr"] + [f"auroc_{x}" for x in GENS]:
                put(f"posthoc.reseed.{s}.{m}.{k}", pts[(s, m)][k])
                put(f"posthoc.reseed.{s}.{m}.{k}.ci", BS.ci(boots[(s, m)][k]))
    label = {s: f"seed {s}" + (" (frozen R_train)" if s == 42 else "") for s in seeds}
    rows, ops, drows = [], [], []
    for m in CORE:
        xs = [pts[(s, m)]["mean_unseen_auroc"] for s in seeds]
        rows.append({"Method": m, **{label[s]: fci(pts[(s, m)]["mean_unseen_auroc"], *BS.ci(boots[(s, m)]["mean_unseen_auroc"]))
                                     for s in seeds}, "Range over draws": f3(max(xs) - min(xs))})
        put(f"agg.posthoc.reseed.{m}.mean_unseen_auroc_range", max(xs) - min(xs))
        put(f"agg.posthoc.reseed.{m}.mean_unseen_auroc_min", min(xs))
        put(f"agg.posthoc.reseed.{m}.mean_unseen_auroc_max", max(xs))
        fp = [pts[(s, m)]["fpr"] for s in seeds]
        put(f"agg.posthoc.reseed.{m}.fpr_min", min(fp))
        put(f"agg.posthoc.reseed.{m}.fpr_max", max(fp))
        ops.append({"Method": m, **{label[s]: f"{100 * pts[(s, m)]['mean_unseen_tpr']:.1f}% / {pct(pts[(s, m)]['fp'], pts[(s, m)]['n_real'])}"
                                    for s in seeds}})
    los = []
    for s in seeds:
        for m in ("O1", "O2", "M0"):
            diff = pts[(s, "B1")]["mean_unseen_auroc"] - pts[(s, m)]["mean_unseen_auroc"]
            lo, hi = BS.ci(boots[(s, "B1")]["mean_unseen_auroc"] - boots[(s, m)]["mean_unseen_auroc"])
            los.append(lo)
            drows.append({"R_train draw": label[s], "Contrast": f"B1 minus {m}", "Mean unseen AUROC difference": f3(diff),
                          "95% CI": f"[{lo:.3f}, {hi:.3f}]"})
            put(f"posthoc.reseed.{s}.diff.B1-{m}.mean_unseen_auroc", diff)
            put(f"posthoc.reseed.{s}.diff.B1-{m}.mean_unseen_auroc.ci", (lo, hi))
    put("agg.posthoc.reseed.min_b1_minus_realonly_ci_lo", min(los))
    put("agg.posthoc.reseed.min_b1_minus_realonly", min(VALUES[f"posthoc.reseed.{s}.diff.B1-{m}.mean_unseen_auroc"]
                                                       for s in seeds for m in ("O1", "O2", "M0")))
    ro = [pts[(s, m)]["mean_unseen_auroc"] for s in seeds for m in ("O1", "O2", "M0")]
    put("agg.posthoc.reseed.realonly.mean_unseen_auroc_max", max(ro))
    put("agg.posthoc.reseed.realonly.mean_unseen_auroc_min", min(ro))
    put("agg.posthoc.reseed.max_range_reference_based",
        max(VALUES[f"agg.posthoc.reseed.{m}.mean_unseen_auroc_range"] for m in ("B1", "O1", "O2")))
    note = ("Post hoc, optional check of proposal Section 6.3 (PROTOCOL v0.6.6, tag posthoc-reseed). R_train redrawn by the "
            "frozen rule with seeds 7, 13, 29; F_seen_train, R_cal, R_test and test images fixed; B1, O1, O2 refitted and all "
            "methods recalibrated; M0 does not use R_train. Seeds are not independent datasets. D-ctrl, ViT-S/14; paired "
            "bootstrap intervals with the shared draws. Sources: artifacts/predictions/e1/, posthoc/.")
    write("posthoc_reseed_auroc", pd.DataFrame(rows), "Post hoc: mean unseen AUROC by R_train draw (D-ctrl, ViT-S/14)", note)
    write("posthoc_reseed_operating_points", pd.DataFrame(ops),
          "Post hoc: mean unseen TPR / real FPR by R_train draw (D-ctrl, ViT-S/14)", note)
    write("posthoc_reseed_differences", pd.DataFrame(drows), "Post hoc: B1 minus real-only scores by R_train draw", note)


def design_values() -> None:
    """Dimension check and duplicate exclusions (sources: artifacts/audit/, artifacts/manifest.csv)."""
    import make_splits
    d = json.loads((ROOT / "artifacts" / "audit" / "dimension_check_primary.json").read_text())
    put("dim.primary.max_over_median", d["per_condition"]["d_ctrl"]["max_over_median"])
    put("dim.primary.n_dominant", len(d["dominant_dims"]))
    m = pd.read_csv(ROOT / "artifacts" / "manifest.csv", keep_default_na=False, na_values=[""])
    ex = make_splits.duplicate_exclusions(m)
    for r in ex.itertuples():
        put(f"dupexcl.{r.split}.{r.generator}", int(r.excluded_by_duplicates))
    put("dupexcl.total", int(ex["excluded_by_duplicates"].sum()))
    grp = m.groupby("dup_group").size()
    put("dup.max_group_size", int(grp.max()))
    # D-raw resize factor 256 / min side over the evaluated test partitions
    t = m[m["partition"].isin(["R_test", "F_seen_test", "F_unseen_test"])]
    f = 256.0 / t[["width", "height"]].min(axis=1)
    for g, x in f.groupby(t["generator"]):
        put(f"scale.{g}.median", float(x.median()))
        put(f"scale.{g}.p05", float(x.quantile(0.05)))
        put(f"scale.{g}.p95", float(x.quantile(0.95)))


# --------------------------------------------------------------------------- failure cases
def failure_cases(e1: pd.DataFrame) -> None:
    d = e1[(e1["backbone"] == "primary") & (e1["data_condition"] == "d_ctrl") & e1["method"].isin(CORE)]
    rows = []
    for m in CORE:
        dm = d[d["method"] == m]
        real = dm[dm["partition"] == "R_test"].sort_values(["score", "row_id"], ascending=[False, True]).head(5)
        rows += [{"method": m, "kind": "highest-scoring real", "generator": "Real", "rank": i + 1, "row_id": r.row_id,
                  "score": r.score, "threshold": r.threshold, "flagged": r.predicted_label} for i, r in enumerate(real.itertuples())]
        for gen in GENS:
            f = dm[(dm["authenticity_label"] == 1) & (dm["generator"] == gen)].sort_values(["score", "row_id"]).head(5)
            rows += [{"method": m, "kind": "lowest-scoring synthetic", "generator": gen, "rank": i + 1, "row_id": r.row_id,
                      "score": r.score, "threshold": r.threshold, "flagged": r.predicted_label} for i, r in enumerate(f.itertuples())]
    pd.DataFrame(rows).to_csv(TAB / "failure_cases_d_ctrl.csv", index=False)


# --------------------------------------------------------------------------- consistency
def consistency(df: pd.DataFrame, name: str) -> None:
    c = E.calibration_check(df, "R_cal")
    c = c[c["n_cal"] > 0]
    ok = bool(np.allclose(c["tau_recomputed"], c["tau_stored"], rtol=0, atol=0))
    put(f"check.{name}.thresholds_recomputed_equal", ok)
    put(f"check.{name}.max_cal_fpr", float(c["cal_fpr"].max()))
    c.to_csv(TAB / f"calibration_check_{name}.csv", index=False)
    if not ok:
        raise AssertionError(f"stored thresholds differ from recomputed ones in {name}")


def aggregates() -> None:
    """Ranges over the real-only methods (O1, O2, M0), derived from values computed above."""
    ro = ["O1", "O2", "M0"]
    V = VALUES
    if "table1_d_ctrl.O1.mean_unseen_auroc" in V:
        for k in ("mean_unseen_auroc", "mean_unseen_tpr", "fpr"):
            xs = [V[f"table1_d_ctrl.{m}.{k}"] for m in ro]
            put(f"agg.e1.d_ctrl.realonly.{k}_min", min(xs))
            put(f"agg.e1.d_ctrl.realonly.{k}_max", max(xs))
        put("agg.e1.d_ctrl.realonly.max_auroc_any_generator",
            max(V[f"table1_d_ctrl.{m}.auroc_{g}"] for m in ro for g in GENS))
        hw = [(V[f"table1_d_ctrl.{m}.auroc_{g}.ci"][1] - V[f"table1_d_ctrl.{m}.auroc_{g}.ci"][0]) / 2
              for m in CORE for g in GENS]
        put("agg.e1.d_ctrl.auroc_ci_halfwidth_min", min(hw))
        put("agg.e1.d_ctrl.auroc_ci_halfwidth_max", max(hw))
    if "e2.d_ctrl.B1.jpeg50.fpr" in V:
        put("agg.e2.d_ctrl.B1.jpeg50_over_clean_fpr", V["e2.d_ctrl.B1.jpeg50.fpr"] / V["e2.d_ctrl.B1.clean.fpr"])
    if "e2.d_ctrl.O1.jpeg75.fpr" in V:
        for cond in ("d_ctrl", "d_raw"):
            f = [V[f"e2.{cond}.{m}.{t}.fpr"] for m in ro for t in E2_ORDER]
            put(f"agg.e2.{cond}.realonly.fpr_min", min(f))
            put(f"agg.e2.{cond}.realonly.fpr_max", max(f))
            d = [abs(V[f"e2.{cond}.{m}.{t}.delta_mean_unseen_auroc"]) for m in ro for t in E2_ORDER[1:]]
            put(f"agg.e2.{cond}.realonly.max_abs_delta_auroc", max(d))
    if "ladderdiff.d_raw-d_jpeg.O1.mean_unseen" in V:
        put("agg.ladder.raw_to_jpeg.realonly.max_abs", max(abs(V[f"ladderdiff.d_raw-d_jpeg.{m}.mean_unseen"]) for m in ro))
    if "e3c.O1.C1_biased.mean_unseen_auroc.minus_c0" in V:
        vs = ["C1_biased", "C1_matched", "C1_inverted"]
        put("agg.e3c.realonly.max_abs_auroc_change", max(abs(V[f"e3c.{m}.{v}.mean_unseen_auroc.minus_c0"]) for m in ro for v in vs))
        put("agg.e3c.all.max_abs_fpr_change", max(abs(V[f"e3c.{m}.{v}.fpr.minus_c0"]) for m in CORE for v in vs))


def note_tables() -> None:
    """Compact main-text tables; full generator/count/interval tables remain available."""
    if (TAB / "table1_d_ctrl.csv").exists():
        t = pd.read_csv(TAB / "table1_d_ctrl.csv").set_index("Method")
        rows = []
        for m in CORE:
            k = f"table1_d_ctrl.{m}.mean_unseen_tpr"
            lo, hi = VALUES[k + ".ci"]
            rows.append({"Method": m, "Unseen AUROC [95% CI]": t.loc[m, "Mean unseen AUROC"],
                         "Unseen TPR [95% CI]": f"{100 * VALUES[k]:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]",
                         "Real FPR (count)": t.loc[m, "Real FPR at 5% target"]})
        write("note_table1", pd.DataFrame(rows), "Operating points, D-ctrl",
              "Shared R_test; equal-weight average over ADM, GLIDE and Midjourney. Full generator counts and intervals: table1_d_ctrl_tpr.")
    if (TAB / "table2_e2_fpr_d_ctrl.csv").exists():
        t = pd.read_csv(TAB / "table2_e2_fpr_d_ctrl.csv")
        write("note_table2", t[t["Method"].isin(CORE)], "Real FPR, D-ctrl",
              "Percent, count, and 95% interval; models and thresholds frozen. Full per-generator TPR/AUROC: e2_per_generator_d_ctrl.")
    if (TAB / "table3_ladder.csv").exists():
        t = pd.read_csv(TAB / "table3_ladder.csv")
        t = t[t["Method"].isin(CORE)].iloc[:, :4]
        t.columns = ["Method", "D-raw", "D-jpeg", "D-ctrl"]
        write("note_table3", t, "Mean unseen AUROC along the ladder", "95% intervals; paired condition contrasts and per-generator components are in table3_ladder_differences.")


def main() -> None:
    e1 = load("e1/*.csv.gz")
    e2 = load("e2/*.csv.gz")
    meta = load("e3/e3a_*.csv.gz")
    e3b = load("e3/e3b_*.csv.gz")
    e3c = load("e3/e3c_*.csv.gz")
    TAB.mkdir(parents=True, exist_ok=True)
    sp_path = ROOT / "artifacts" / "splits.json"
    if sp_path.exists():  # design counts (source: artifacts/splits.json, artifacts/audit/data_audit.json)
        sp = json.loads(sp_path.read_text())
        for k, v in sp["counts"].items():
            if isinstance(v, dict):
                for g, n in v.items():
                    put(f"count.{k}.{g}", n)
            else:
                put(f"count.{k}", v)
        for k, v in sp["eligible_pool_sizes"].items():
            put(f"pool.{k}", v)
        audit = json.loads((ROOT / "artifacts" / "audit" / "data_audit.json").read_text())
        put("data.n_rows", audit["n_rows"])
        put("data.real_share_exact_q96", audit["g0_signals"]["real_share_exact_ijg_q96"])
        put("data.adm_rgba", audit["mode_by_generator"]["ADM"].get("RGBA", 0))
        put("data.sd14_rows", sum(audit["counts_split_generator"].get("SD14", {}).values()))
    if not e1.empty:
        import calibrate
        n_cal = int(e1.loc[(e1["partition"] == "R_cal") & (e1["method"] == "B1") & (e1["backbone"] == "primary")
                           & (e1["data_condition"] == "d_ctrl"), "row_id"].nunique())
        put("calib.n", n_cal)
        put("calib.k", calibrate.order_statistic_index(n_cal))
        consistency(e1, "e1")
        table1(e1, "primary", "d_ctrl", CORE, "table1_d_ctrl", "Table 1. AUROC by generator, D-ctrl (ViT-S/14)")
        table1(e1, "primary", "d_raw", CORE, "table1_d_raw", "AUROC by generator, D-raw (ViT-S/14)")
        table1(e1, "secondary", "d_ctrl", CORE, "table1_vitb_d_ctrl", "AUROC by generator, D-ctrl (ViT-B/14, secondary)")
        table1(e1, "primary", "d_ctrl", EXPLORATORY, "table1_exploratory_d_ctrl", "Exploratory variants, D-ctrl (ViT-S/14)")
        put("check.b1_c1_identical_d_ctrl", bool(np.array_equal(
            e1[(e1["backbone"] == "primary") & (e1["data_condition"] == "d_ctrl") & (e1["method"] == "B1")]["score"].to_numpy(),
            e1[(e1["backbone"] == "primary") & (e1["data_condition"] == "d_ctrl") & (e1["method"] == "B1-C1")]["score"].to_numpy())))
        for c in ("d_ctrl", "d_raw"):
            jaccard(e1, c)
            method_differences(e1, c)
        jaccard(e1, "d_ctrl", "secondary")
        failure_cases(e1)
        E.per_generator(e1).to_csv(TAB / "e1_per_generator_all.csv", index=False)
        diag = json.loads((PRED / "e1" / "e1_diagnostics.json").read_text())
        for k, v in diag.items():
            if isinstance(v, dict) and "B1_selected_C" in v:
                put(f"diag.{k}.B1_C", v["B1_selected_C"])
                put(f"diag.{k}.O2_shrinkage", v["O2_shrinkage"])
                put(f"diag.{k}.O2_condition_number", v["O2_condition_number"])
    if not e2.empty:
        e2_tables(e1, e2, CORE)
        e2_tables(e1, e2, [m for m in EXPLORATORY if m in set(e2["method"])], prefix="appendix_")
    if not e3b.empty:
        consistency(e3b, "e3b")
        table1(e3b, "primary", "d_jpeg", CORE, "table1_d_jpeg", "AUROC by generator, D-jpeg (ViT-S/14)")
        ladder(e1, e3b, meta)
    if not meta.empty:
        e3a_tables(meta)
    if not e3c.empty:
        e3c_tables(e3c)
        e3c_decomposition(e3c)
    ph = load("posthoc/posthoc_primary_d_native.csv.gz")
    if not ph.empty and not e3b.empty:
        posthoc_native(e1, e3b, ph)
    rs = load("posthoc/posthoc_reseed_*.csv.gz")
    if not rs.empty and not e1.empty:
        posthoc_reseed(e1, rs)
    if sp_path.exists():
        design_values()
    aggregates()
    note_tables()
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "note_values.json").write_text(json.dumps(VALUES, indent=1, default=float) + "\n")
    print(f"wrote {len(VALUES)} values and tables to {TAB}")


if __name__ == "__main__":
    main()

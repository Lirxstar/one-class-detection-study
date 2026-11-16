#!/usr/bin/env python3
"""S-BF Step 4: metrics, bootstrap intervals, tables and figures from artifacts/predictions/supp_bfree only
(PROTOCOL.md Section 7).

Bootstrap: 2,000 resamples of test source-id groups with `default_rng(2026)`; the real, FLUX and SD3.5
images of a drawn id move together. One draw set per analysis, reused for every method, cell, generator
and metric (paired contrasts). Thresholds tau are fixed; the matched-FPR threshold is recomputed per draw.
Core numbers in the comparison table are read from the existing core tables (not recomputed).

Outputs: artifacts/tables/supp_bfree/*.csv|md; report/figures/supp_bfree/*.svg (png ignored).
Run src/analysis.py first: the comparison table reads three main-study tables it writes.
Usage: python src/supp_bfree_analysis.py [--pilot]
"""
from __future__ import annotations

import json
import sys
from math import floor

import numpy as np
import pandas as pd

import config
import pipeline as P
from config import ROOT
from evaluate import auroc

PRED = P.PRED_DIR / "supp_bfree"
TABLES = ROOT / "artifacts" / "tables" / "supp_bfree"
CORE_TABLES = ROOT / "artifacts" / "tables"
FIG = ROOT / "report" / "figures" / "supp_bfree"
METHODS = ["B1", "O1", "O2", "M0"]
REAL_ONLY = ["O1", "O2", "M0"]
CELLS = [("primary", "native"), ("primary", "resize"), ("secondary", "native")]
CELL_NAME = {("primary", "native"): "ViT-S/14 native (primary)", ("primary", "resize"): "ViT-S/14 resize",
             ("secondary", "native"): "ViT-B/14 native"}
CELL_KEY = {("primary", "native"): "s_native", ("primary", "resize"): "s_resize", ("secondary", "native"): "b_native"}
ANALYSIS_NAME = {"all": "all ids, real images converted to sRGB", "srgb": "sRGB-tagged ids only (sensitivity)"}
GEN_NAME = {"FLUX": "FLUX (unseen)", "SD35": "SD3.5 (seen by B1)"}
REAL_TEST = {"final": "R_test", "pilot": "P_test"}
VALUES: dict[str, object] = {}


# --------------------------------------------------------------------------- metrics
def matched_threshold(real: np.ndarray, alpha: float) -> float:
    """Largest real test score t such that mean(real > t) <= alpha: the (m + 1)-th largest, m = floor(alpha n)."""
    m = floor(alpha * len(real) + 1e-9)
    return float(np.sort(real)[::-1][m])


def tpr_at_matched_fpr(real: np.ndarray, fake: np.ndarray, alpha: float) -> tuple[float, float]:
    t = matched_threshold(real, alpha)
    return float((fake > t).mean()), float((real > t).mean())


def pair_rate(real: np.ndarray, fake: np.ndarray) -> float:
    """Share of ids with s(fake_i) > s(real_i), ties 1/2 (arrays aligned by source id)."""
    return float(((fake > real) + 0.5 * (fake == real)).mean())


def metrics(real: np.ndarray, fake: np.ndarray, tau: float, alpha: float) -> dict:
    tpr_m, fpr_m = tpr_at_matched_fpr(real, fake, alpha)
    return {"auroc": auroc(real, fake), "tpr_matched": tpr_m, "fpr_matched": fpr_m, "pair": pair_rate(real, fake),
            "fpr_cal": float((real > tau).mean()), "tpr_cal": float((fake > tau).mean())}


# --------------------------------------------------------------------------- bootstrap
def make_draws(groups: pd.Series, n_boot: int, seed: int) -> list[np.ndarray]:
    """groups: id_group per test source id, index sorted by source id. Returns, per draw, positions into
    that index; whole groups are drawn with replacement."""
    ids = list(groups.index)
    assert ids == sorted(ids)
    members = {g: np.array(sorted(groups.index.get_indexer(v))) for g, v in groups.groupby(groups).groups.items()}
    order = sorted(members)
    gi = np.random.default_rng(seed).integers(0, len(order), size=(n_boot, len(order)))
    if all(len(members[g]) == 1 for g in order):
        flat = np.array([members[g][0] for g in order])
        return [flat[row] for row in gi]
    return [np.concatenate([members[order[j]] for j in row]) for row in gi]


def boot(real: np.ndarray, fake: np.ndarray, tau: float, alpha: float, draws: list[np.ndarray]) -> dict[str, np.ndarray]:
    out = {k: np.empty(len(draws)) for k in ("auroc", "tpr_matched", "fpr_matched", "pair", "fpr_cal", "tpr_cal")}
    for b, idx in enumerate(draws):
        for k, v in metrics(real[idx], fake[idx], tau, alpha).items():
            out[k][b] = v
    return out


def ci(x: np.ndarray) -> list[float]:
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


# --------------------------------------------------------------------------- data
def load(role: str) -> pd.DataFrame:
    d = PRED if role == "final" else PRED / "pilot"
    files = sorted(d.glob(f"{role}_*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"no {role} predictions in {d}")
    return pd.concat([pd.read_csv(f, dtype={"source_id": str, "id_group": str}) for f in files], ignore_index=True)


def wide(block: pd.DataFrame, role: str) -> pd.DataFrame:
    """Test rows of one (analysis, backbone, condition, method) block, one row per source id."""
    t = block[(block["partition"] == REAL_TEST[role]) | (block["authenticity_label"] == 1)]
    w = t.pivot(index="source_id", columns="generator", values="score").sort_index()
    w["id_group"] = t.drop_duplicates("source_id").set_index("source_id").loc[w.index, "id_group"]
    if w.drop(columns="id_group").isna().any().any():
        raise AssertionError("test ids without a complete real/synthetic set")
    return w


def compute(df: pd.DataFrame, role: str) -> tuple[pd.DataFrame, dict]:
    sb = config.load("supp_bfree")
    alpha, B, seed = float(sb["metrics"]["matched_fpr"]), int(sb["bootstrap"]["n"]), int(sb["bootstrap"]["seed"])
    rows, boots = [], {}
    for analysis in ("all", "srgb"):
        a = df[df["analysis"] == analysis]
        if a.empty:
            continue
        ref = wide(a[(a["backbone"] == "primary") & (a["data_condition"] == "native") & (a["method"] == "B1")], role)
        draws = make_draws(ref["id_group"], B, seed)
        for (backbone, cond) in CELLS:
            for method in METHODS:
                blk = a[(a["backbone"] == backbone) & (a["data_condition"] == cond) & (a["method"] == method)]
                if blk.empty:
                    continue
                w = wide(blk, role)
                if list(w.index) != list(ref.index):
                    raise AssertionError("test ids differ between cells")
                tau = float(blk["threshold"].iloc[0])
                real = w["REAL"].to_numpy(np.float64)
                for gen in [g for g in ("FLUX", "SD35") if g in w.columns]:
                    fake = w[gen].to_numpy(np.float64)
                    pt = metrics(real, fake, tau, alpha)
                    bt = boot(real, fake, tau, alpha, draws)
                    boots[(analysis, backbone, cond, method, gen)] = bt
                    rows.append({"analysis": analysis, "backbone": backbone, "condition": cond, "method": method,
                                 "generator": gen, "n_real": len(real), "n_fake": len(fake), "threshold": tau,
                                 "fp_cal": int((real > tau).sum()), "tp_cal": int((fake > tau).sum()),
                                 "matched_threshold": matched_threshold(real, alpha), "alpha": alpha,
                                 **pt, **{f"{k}_ci": ci(v) for k, v in bt.items()}})
    return pd.DataFrame(rows), boots


# --------------------------------------------------------------------------- tables
def f3(x: float, c) -> str:
    return f"{x:.3f} [{c[0]:.3f}, {c[1]:.3f}]"


def p1(x: float, c) -> str:
    return f"{100 * x:.1f}% [{100 * c[0]:.1f}, {100 * c[1]:.1f}]"


def d3(x: np.ndarray, point: float) -> str:
    c = ci(x)
    flag = "" if c[0] <= 0 <= c[1] else " *"
    return f"{point:+.3f} [{c[0]:+.3f}, {c[1]:+.3f}]{flag}"


def write(name: str, df: pd.DataFrame, title: str, foot: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{name}.csv", index=False, lineterminator="\n")
    cols = list(df.columns)
    md = [f"### {title}", "", "| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    (TABLES / f"{name}.md").write_text("\n".join(md + ["", foot, ""]))


def put(key: str, v) -> None:
    if key in VALUES and VALUES[key] != v:
        raise AssertionError(f"conflicting value for {key}")
    VALUES[key] = v


def tables(res: pd.DataFrame, boots: dict, role: str) -> None:
    gens = [g for g in ("FLUX", "SD35") if g in set(res["generator"])]
    for analysis in sorted(set(res["analysis"])):
        a = res[res["analysis"] == analysis]
        main = []
        for (backbone, cond) in CELLS:
            for method in METHODS:
                r = a[(a["backbone"] == backbone) & (a["condition"] == cond) & (a["method"] == method)].set_index("generator")
                if r.empty:
                    continue
                row = {"Cell": CELL_NAME[(backbone, cond)], "Method": method}
                for g in gens:
                    x = r.loc[g]
                    row[f"AUROC {GEN_NAME[g]}"] = f3(x["auroc"], x["auroc_ci"])
                    row[f"TPR at 5% FPR {g}"] = p1(x["tpr_matched"], x["tpr_matched_ci"])
                    row[f"Matched-pair rate {g}"] = f3(x["pair"], x["pair_ci"])
                    k = f"{analysis}.{CELL_KEY[(backbone, cond)]}.{method}.{g}"
                    for m in ("auroc", "tpr_matched", "fpr_matched", "pair", "fpr_cal", "tpr_cal"):
                        put(f"{k}.{m}", float(x[m]))
                        put(f"{k}.{m}.ci", x[f"{m}_ci"])
                    put(f"{k}.tp_cal", int(x["tp_cal"]))
                x0 = r.iloc[0]
                row["Calibrated real FPR"] = f"{100 * x0['fpr_cal']:.1f}% ({x0['fp_cal']}/{x0['n_real']})"
                g0 = gens[0]
                row[f"Calibrated TPR {g0}"] = f"{100 * r.loc[g0, 'tpr_cal']:.1f}% ({r.loc[g0, 'tp_cal']}/{r.loc[g0, 'n_fake']})"
                put(f"{analysis}.{CELL_KEY[(backbone, cond)]}.{method}.fp_cal", int(x0["fp_cal"]))
                put(f"{analysis}.n_real", int(x0["n_real"]))
                main.append(row)
        n = int(a["n_real"].iloc[0])
        write(f"{role}_main_{analysis}", pd.DataFrame(main),
              f"S-BF {role}: detection by method and cell ({ANALYSIS_NAME[analysis]})",
              f"Test: {n} real images and their content-matched synthetic images (same source ids). Point estimate "
              "[percentile 95% interval], 2,000 resamples of test source ids (real and synthetic of an id drawn together). "
              "TPR at 5% FPR: threshold from the test real scores so that every method has real FPR <= 5% (oracle "
              "operating point). Matched-pair rate: share of ids whose synthetic image scores above its own real image. "
              "Calibrated: threshold from real calibration images only (k = ceil(0.95(n+1))). B1 is trained on real vs "
              f"SD3.5; O1, O2, M0 never see a synthetic image. Source: artifacts/predictions/supp_bfree/{'pilot/' if role == 'pilot' else ''}.")
        # contrasts: real-only minus B1, seen minus unseen
        g_main = "FLUX" if "FLUX" in gens else gens[0]
        con = []
        for (backbone, cond) in CELLS:
            if (analysis, backbone, cond, "B1", g_main) not in boots:
                continue
            b1 = boots[(analysis, backbone, cond, "B1", g_main)]
            p_b1 = a[(a["backbone"] == backbone) & (a["condition"] == cond) & (a["method"] == "B1") & (a["generator"] == g_main)].iloc[0]
            for method in REAL_ONLY:
                bm = boots[(analysis, backbone, cond, method, g_main)]
                pm = a[(a["backbone"] == backbone) & (a["condition"] == cond) & (a["method"] == method) & (a["generator"] == g_main)].iloc[0]
                row = {"Cell": CELL_NAME[(backbone, cond)], "Contrast": f"{method} minus B1"}
                for m, lab in (("auroc", f"AUROC {g_main}"), ("tpr_matched", f"TPR at 5% FPR {g_main}"), ("pair", f"Matched-pair rate {g_main}")):
                    dd = bm[m] - b1[m]
                    pt = float(pm[m] - p_b1[m])
                    row[lab] = d3(dd, pt)
                    k = f"{analysis}.{CELL_KEY[(backbone, cond)]}.{method}_minus_B1.{g_main}.{m}"
                    put(k, pt)
                    put(k + ".ci", ci(dd))
                con.append(row)
        write(f"{role}_contrasts_{analysis}", pd.DataFrame(con),
              f"S-BF {role}: real-only scores minus B1 on {g_main} ({ANALYSIS_NAME[analysis]})",
              "Paired bootstrap differences (same resampled source ids for both methods). * : the 95% interval excludes 0. "
              f"Source: artifacts/predictions/supp_bfree/{'pilot/' if role == 'pilot' else ''}.")
        if {"FLUX", "SD35"} <= set(gens):
            gap = []
            for (backbone, cond) in CELLS:
                for method in METHODS:
                    s, u = boots[(analysis, backbone, cond, method, "SD35")], boots[(analysis, backbone, cond, method, "FLUX")]
                    ps = a[(a["backbone"] == backbone) & (a["condition"] == cond) & (a["method"] == method)].set_index("generator")
                    pt = float(ps.loc["SD35", "auroc"] - ps.loc["FLUX", "auroc"])
                    k = f"{analysis}.{CELL_KEY[(backbone, cond)]}.{method}.seen_minus_unseen"
                    put(k, pt)
                    put(k + ".ci", ci(s["auroc"] - u["auroc"]))
                    gap.append({"Cell": CELL_NAME[(backbone, cond)], "Method": method,
                                "AUROC SD3.5 minus FLUX": d3(s["auroc"] - u["auroc"], pt)})
            write(f"{role}_seen_unseen_{analysis}", pd.DataFrame(gap),
                  f"S-BF {role}: AUROC on SD3.5 (seen by B1) minus FLUX (unseen) ({ANALYSIS_NAME[analysis]})",
                  "Paired over the same test source ids. * : the 95% interval excludes 0. Only B1 has seen SD3.5; for "
                  "O1, O2, M0 the difference measures how hard each generator is for that score. "
                  "Source: artifacts/predictions/supp_bfree/.")
            cells = []
            for method in METHODS:
                base = boots[(analysis, "primary", "native", method, "FLUX")]
                pb = a[(a["backbone"] == "primary") & (a["condition"] == "native") & (a["method"] == method) & (a["generator"] == "FLUX")].iloc[0]
                for (backbone, cond), lab in ((("primary", "resize"), "ViT-S/14 resize minus native"),
                                             (("secondary", "native"), "ViT-B/14 minus ViT-S/14 (native)")):
                    o = boots[(analysis, backbone, cond, method, "FLUX")]
                    po = a[(a["backbone"] == backbone) & (a["condition"] == cond) & (a["method"] == method) & (a["generator"] == "FLUX")].iloc[0]
                    pt = float(po["auroc"] - pb["auroc"])
                    put(f"{analysis}.{CELL_KEY[(backbone, cond)]}_minus_s_native.{method}.FLUX.auroc", pt)
                    put(f"{analysis}.{CELL_KEY[(backbone, cond)]}_minus_s_native.{method}.FLUX.auroc.ci", ci(o["auroc"] - base["auroc"]))
                    cells.append({"Method": method, "Contrast": lab, "AUROC FLUX": d3(o["auroc"] - base["auroc"], pt),
                                  "TPR at 5% FPR FLUX": d3(o["tpr_matched"] - base["tpr_matched"],
                                                           float(po["tpr_matched"] - pb["tpr_matched"]))})
            write(f"{role}_cells_{analysis}", pd.DataFrame(cells),
                  f"S-BF {role}: secondary cells minus the primary cell, FLUX ({ANALYSIS_NAME[analysis]})",
                  "Paired over the same test source ids. * : the 95% interval excludes 0. Source: artifacts/predictions/supp_bfree/.")


def core_comparison(res: pd.DataFrame) -> None:
    """S-BF FLUX beside the core unseen-generator results (descriptive; values read from core tables)."""
    t_ctrl = pd.read_csv(CORE_TABLES / "table1_d_ctrl.csv").set_index("Method")
    t_raw = pd.read_csv(CORE_TABLES / "table1_d_raw.csv").set_index("Method")
    t_nat = pd.read_csv(CORE_TABLES / "posthoc_native_ladder.csv").set_index("Method")
    rows = []
    for m in METHODS:
        for key, t, col in (("d_ctrl.mean_unseen", t_ctrl, "Mean unseen AUROC"), ("d_raw.mean_unseen", t_raw, "Mean unseen AUROC"),
                            ("d_native.mean_unseen", t_nat, "D-native (post hoc)"), ("d_ctrl.sd15", t_ctrl, "SD15 (seen by B1)")):
            s = str(t.loc[m, col])
            put(f"core.{key}.{m}", float(s.split(" [")[0]))
            put(f"core.{key}.{m}.ci", [float(x) for x in s.split("[")[1].rstrip("]").split(", ")])

        def sb(analysis, backbone, cond, gen="FLUX"):
            x = res[(res["analysis"] == analysis) & (res["backbone"] == backbone) & (res["condition"] == cond)
                    & (res["method"] == m) & (res["generator"] == gen)].iloc[0]
            return f3(x["auroc"], x["auroc_ci"])
        rows.append({"Method": m,
                     "Core D-ctrl, mean unseen (ADM, GLIDE, MJ)": t_ctrl.loc[m, "Mean unseen AUROC"],
                     "Core D-native (post hoc), mean unseen": t_nat.loc[m, "D-native (post hoc)"],
                     "S-BF native, FLUX (all ids)": sb("all", "primary", "native"),
                     "S-BF native, FLUX (sRGB ids)": sb("srgb", "primary", "native"),
                     "Core D-raw, mean unseen": t_raw.loc[m, "Mean unseen AUROC"],
                     "S-BF resize, FLUX (all ids)": sb("all", "primary", "resize"),
                     "Core seen SD15 (D-ctrl)": t_ctrl.loc[m, "SD15 (seen by B1)"],
                     "S-BF seen SD3.5 (native, all ids)": sb("all", "primary", "native", "SD35")})
    write("final_core_comparison", pd.DataFrame(rows),
          "S-BF beside the core study (ViT-S/14 AUROC; descriptive, not a paired test)",
          "Core columns copied from artifacts/tables/table1_d_ctrl.csv, table1_d_raw.csv and posthoc_native_ladder.csv "
          "(Tiny GenImage: real JPEG, synthetic PNG; B1 trained on SD15; 500 real and 300 per generator). S-BF columns from "
          "artifacts/predictions/supp_bfree/ (B-Free extended Synthbuster: both classes lossless PNG, content-matched; B1 "
          "trained on SD3.5). Native = no resizing (core D-native crop); resize = core D-raw view. Different data, generators "
          "and sizes: compare directions, not values.")


def diagnostics(role: str) -> None:
    d = json.loads(((PRED if role == "final" else PRED / "pilot") / f"{role}_diagnostics.json").read_text())
    rows = []
    for c in d["cells"]:
        row = {"Analysis": c["analysis"], "Cell": CELL_NAME[(c["backbone"], c["condition"])],
               "n fit real": c["n_fit_real"], "n cal": c["n_cal"], "n real test": c["n_real_test"],
               "B1 C": c["B1_selected_C"], "O2 shrinkage": round(c["O2_shrinkage"], 3)}
        for k, v in c["correspondence"].items():
            g = k.split(":")[1]
            row[f"{g} own real top-1"] = f"{100 * v['top1']:.1f}%"
            row[f"{g} own real median rank"] = f"{v['median_rank']:.0f} of {v['n_real']}"
            put(f"{c['analysis']}.{CELL_KEY[(c['backbone'], c['condition'])]}.corr.{g}.top1", float(v["top1"]))
            put(f"{c['analysis']}.{CELL_KEY[(c['backbone'], c['condition'])]}.corr.{g}.median_rank", float(v["median_rank"]))
        rows.append(row)
    write(f"{role}_diagnostics", pd.DataFrame(rows), f"S-BF {role}: fitting and correspondence diagnostics",
          "B1 C by grouped 5-fold CV inside the training ids; O2 Ledoit-Wolf shrinkage. Correspondence: rank of each "
          "synthetic test image's own real counterpart among all real test images by cosine similarity of post-LN class "
          "tokens (1 = most similar; chance top-1 = 1/n). Source: diagnostics JSON in artifacts/predictions/supp_bfree/.")


# --------------------------------------------------------------------------- figures
def figures(df: pd.DataFrame, res: pd.DataFrame) -> None:
    import figures as FG  # core style (colours, markers); imported here so pilot runs need no matplotlib
    plt = FG.plt
    FIG.mkdir(parents=True, exist_ok=True)
    labels = {"B1": "B1 probe (SD3.5 labels)", "O1": "O1 kNN", "O2": "O2 Mahalanobis", "M0": "M0 -L1 magnitude"}
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))
    for ax, analysis in zip(axes, ("all", "srgb")):
        a = df[(df["analysis"] == analysis) & (df["backbone"] == "primary") & (df["data_condition"] == "native")]
        for m in METHODS:
            w = wide(a[a["method"] == m], "final")
            r, f = w["REAL"].to_numpy(), w["FLUX"].to_numpy()
            th = np.r_[np.inf, np.unique(np.r_[r, f])[::-1]]
            fpr = [(r > t).mean() for t in th] + [1.0]
            tpr = [(f > t).mean() for t in th] + [1.0]
            au = float(res[(res["analysis"] == analysis) & (res["backbone"] == "primary") & (res["condition"] == "native")
                           & (res["method"] == m) & (res["generator"] == "FLUX")]["auroc"].iloc[0])
            ax.step(fpr, tpr, where="post", color=FG.COLORS[m], lw=1.6, label=f"{labels[m]} ({au:.3f})")
        ax.plot([0, 1], [0, 1], color=FG.GRID, lw=1.0, ls="--")
        ax.axvline(0.05, color=FG.INK2, lw=0.8, ls=":")
        ax.set_xlim(0, 1), ax.set_ylim(0, 1.01)
        ax.set_xlabel("Real test FPR"), ax.set_ylabel("FLUX TPR")
        ax.set_title("All ids (sRGB-converted)" if analysis == "all" else "sRGB-tagged ids (sensitivity)", fontsize=9)
        ax.legend(loc="lower right", fontsize=7)
    fig.suptitle("S-BF: FLUX vs content-matched real, ViT-S/14 native crop (AUROC in legend; dotted line = 5% FPR)", fontsize=9)
    _save(fig, "sbf_roc_flux")

    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    groups = [("all", "primary", "native"), ("srgb", "primary", "native"), ("all", "primary", "resize"), ("all", "secondary", "native")]
    names = ["ViT-S native\nall ids", "ViT-S native\nsRGB ids", "ViT-S resize\nall ids", "ViT-B native\nall ids"]
    for gi, (analysis, backbone, cond) in enumerate(groups):
        for mi, m in enumerate(METHODS):
            x = res[(res["analysis"] == analysis) & (res["backbone"] == backbone) & (res["condition"] == cond)
                    & (res["method"] == m) & (res["generator"] == "FLUX")].iloc[0]
            xp = gi + (mi - 1.5) * 0.17
            ax.errorbar([xp], [x["auroc"]], yerr=[[x["auroc"] - x["auroc_ci"][0]], [x["auroc_ci"][1] - x["auroc"]]],
                        fmt=FG.MARKERS[m], color=FG.COLORS[m], ms=5, lw=1.4, capsize=0, mec=FG.SURF, mew=0.8,
                        label=labels[m] if gi == 0 else None)
    ax.axhline(0.5, color=FG.INK2, lw=0.8, ls=":")
    ax.set_xticks(range(len(groups)), names)
    ax.set_ylabel("AUROC, FLUX vs real")
    ax.set_ylim(0.25, 1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7, ncol=4)
    ax.set_title("S-BF: AUROC on the unseen generator with 95% bootstrap intervals", fontsize=9)
    _save(fig, "sbf_auroc_cells")


def _save(fig, name: str) -> None:
    import figures as FG
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=200)
    FG.plt.close(fig)


def main() -> None:
    role = "pilot" if "--pilot" in sys.argv else "final"
    global TABLES
    if role == "pilot":
        TABLES = TABLES / "pilot"
    df = load(role)
    res, boots = compute(df, role)
    TABLES.mkdir(parents=True, exist_ok=True)
    flat = res.copy()
    for c in [c for c in flat.columns if c.endswith("_ci")]:
        flat[c + "_lo"] = flat[c].map(lambda v: v[0])
        flat[c + "_hi"] = flat[c].map(lambda v: v[1])
        flat = flat.drop(columns=c)
    flat.to_csv(TABLES / f"{role}_metrics_long.csv", index=False, lineterminator="\n")
    tables(res, boots, role)
    diagnostics(role)
    if role == "final":
        core_comparison(res)
        figures(df, res)
    print(f"{role}: {len(res)} metric rows, tables in {TABLES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

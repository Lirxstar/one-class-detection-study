#!/usr/bin/env python3
"""S-BF-EQ: metrics, paired bootstrap intervals, tables and figure (PROTOCOL.md Section 8).

Reads only artifacts/predictions/supp_bfree_eq/ (B1-eq) and the S-BF predictions (O1, O2, B1-full; analysis
`all`). Metric and bootstrap functions are those of `supp_bfree_analysis`, imported unchanged; the draws are
built exactly as there (same sorted test ids and id groups, 2,000 resamples, seed 2026), so every method, seed
and cell is evaluated on the same resampled test ids.

Outputs: artifacts/tables/supp_bfree_eq/*.csv|md; report/figures/supp_bfree_eq/*.svg (png ignored).
Usage: python src/supp_bfree_eq_analysis.py [--pilot]
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import config
import pipeline as P
import supp_bfree_analysis as A
from config import ROOT

PRED = P.PRED_DIR / "supp_bfree_eq"
TABLES = ROOT / "artifacts" / "tables" / "supp_bfree_eq"
FIG = ROOT / "report" / "figures" / "supp_bfree_eq"
CELLS = A.CELLS
CELL_NAME, CELL_KEY = A.CELL_NAME, A.CELL_KEY
REFERENCE = ["B1", "O1", "O2"]          # from S-BF predictions; B1 there is B1-full
VALUES: dict[str, object] = {}


def cfg() -> dict:
    return config.load("supp_bfree_eq")


def seeds() -> list[int]:
    return [int(s) for s in cfg()["train_subsample"]["seeds"]]


def put(key: str, v) -> None:
    if key in VALUES and VALUES[key] != v:
        raise AssertionError(f"conflicting value for {key}")
    VALUES[key] = v


# --------------------------------------------------------------------------- data
def load_eq(role: str) -> pd.DataFrame:
    d = PRED if role == "final" else PRED / "pilot"
    files = sorted(d.glob(f"{role}_*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"no {role} predictions in {d}")
    return pd.concat([pd.read_csv(f, dtype={"source_id": str, "id_group": str}) for f in files], ignore_index=True)


def load_sbf(role: str) -> pd.DataFrame:
    df = A.load(role)
    return df[(df["analysis"] == "all") & (df["method"].isin(REFERENCE))]


def gens_of(df: pd.DataFrame) -> list[str]:
    return [g for g in ("FLUX", "SD35") if g in set(df["generator"])]


def main_gen(gens: list[str]) -> str:
    return "FLUX" if "FLUX" in gens else gens[0]


# --------------------------------------------------------------------------- compute
def compute(eq: pd.DataFrame, sbf: pd.DataFrame, role: str):
    sb = config.load("supp_bfree")
    alpha, B, bseed = float(sb["metrics"]["matched_fpr"]), int(sb["bootstrap"]["n"]), int(sb["bootstrap"]["seed"])
    ref = A.wide(sbf[(sbf["backbone"] == "primary") & (sbf["data_condition"] == "native") & (sbf["method"] == "B1")], role)
    draws = A.make_draws(ref["id_group"], B, bseed)
    gens = gens_of(sbf)
    rows, boots = [], {}

    def add(label, block, backbone, cond, extra):
        w = A.wide(block, role)
        if list(w.index) != list(ref.index):
            raise AssertionError(f"test ids differ ({label}, {backbone}/{cond})")
        tau = float(block["threshold"].iloc[0])
        real = w["REAL"].to_numpy(np.float64)
        for g in gens:
            fake = w[g].to_numpy(np.float64)
            boots[(backbone, cond, label, g)] = A.boot(real, fake, tau, alpha, draws)
            rows.append({"backbone": backbone, "condition": cond, "label": label, "generator": g, "n_real": len(real),
                         "n_fake": len(fake), "threshold": tau, "fp_cal": int((real > tau).sum()),
                         "tp_cal": int((fake > tau).sum()), **extra, **A.metrics(real, fake, tau, alpha),
                         **{f"{k}_ci": A.ci(v) for k, v in boots[(backbone, cond, label, g)].items()}})

    for backbone, cond in CELLS:
        c_sbf = sbf[(sbf["backbone"] == backbone) & (sbf["data_condition"] == cond)]
        c_eq = eq[(eq["backbone"] == backbone) & (eq["data_condition"] == cond)]
        for m in REFERENCE:
            blk = c_sbf[c_sbf["method"] == m]
            add("B1-full" if m == "B1" else m, blk, backbone, cond,
                {"seed": np.nan, "selected_C": float(blk["selected_C"].iloc[0]) if m == "B1" else np.nan,
                 "n_train_images": 1200 if m == "B1" and role == "final" else np.nan})
        if sorted(c_eq["subsample_seed"].unique()) != seeds():
            raise AssertionError(f"{backbone}/{cond}: seeds in predictions differ from the config")
        for s in seeds():
            blk = c_eq[c_eq["subsample_seed"] == s]
            add(f"B1-eq s{s}", blk, backbone, cond,
                {"seed": s, "selected_C": float(blk["selected_C"].iloc[0]), "n_train_images": int(blk["n_train_images"].iloc[0])})
    return pd.DataFrame(rows), boots, gens


# --------------------------------------------------------------------------- tables
def write(name: str, df: pd.DataFrame, title: str, foot: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{name}.csv", index=False, lineterminator="\n")
    cols = list(df.columns)
    md = [f"### {title}", "", "| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    (TABLES / f"{name}.md").write_text("\n".join(md + ["", foot, ""]))


def excl0(c) -> bool:
    return not (c[0] <= 0 <= c[1])


def tables(res: pd.DataFrame, boots: dict, gens: list[str], role: str) -> None:
    g = main_gen(gens)
    src = f"artifacts/predictions/supp_bfree_eq/{'pilot/' if role == 'pilot' else ''} and artifacts/predictions/supp_bfree/{'pilot/' if role == 'pilot' else ''}"
    eq_labels = [f"B1-eq s{s}" for s in seeds()]
    n_real = int(res["n_real"].iloc[0])

    def pt(backbone, cond, label, gen, metric="auroc"):
        r = res[(res["backbone"] == backbone) & (res["condition"] == cond) & (res["label"] == label) & (res["generator"] == gen)]
        return float(r[metric].iloc[0])

    def eq_mean_boot(backbone, cond, gen, metric="auroc"):
        return np.mean([boots[(backbone, cond, l, gen)][metric] for l in eq_labels], axis=0)

    auroc_rows, con_rows, summ_rows, op_rows = [], [], [], []
    for backbone, cond in CELLS:
        ck = CELL_KEY[(backbone, cond)]
        cell = CELL_NAME[(backbone, cond)]
        # AUROC by method / seed
        for label in ["B1-full"] + eq_labels + ["B1-eq mean of 5", "O1", "O2"]:
            row = {"Cell": cell, "Method": label}
            if label == "B1-eq mean of 5":
                for gen in gens:
                    b = eq_mean_boot(backbone, cond, gen)
                    p = float(np.mean([pt(backbone, cond, l, gen) for l in eq_labels]))
                    row[f"AUROC {A.GEN_NAME[gen]}"] = A.f3(p, A.ci(b))
                    put(f"{ck}.B1eq.mean.{gen}.auroc", p)
                    put(f"{ck}.B1eq.mean.{gen}.auroc.ci", A.ci(b))
                row["Training images"] = "600 (each seed)" if role == "final" else "360 (each seed)"
                row["B1 C"] = ""
            else:
                r = res[(res["backbone"] == backbone) & (res["condition"] == cond) & (res["label"] == label)].set_index("generator")
                key = {"B1-full": "B1full", "O1": "O1", "O2": "O2"}.get(label, label.replace("B1-eq ", "B1eq."))
                for gen in gens:
                    row[f"AUROC {A.GEN_NAME[gen]}"] = A.f3(r.loc[gen, "auroc"], r.loc[gen, "auroc_ci"])
                    put(f"{ck}.{key}.{gen}.auroc", float(r.loc[gen, "auroc"]))
                    put(f"{ck}.{key}.{gen}.auroc.ci", r.loc[gen, "auroc_ci"])
                if role == "final":
                    row["Training images"] = {"B1-full": "1,200 (600 real + 600 SD3.5)", "O1": "600 real", "O2": "600 real"}.get(
                        label, "600 (300 real + 300 SD3.5)")
                else:
                    row["Training images"] = {"B1-full": "720 (360 + 360)", "O1": "360 real", "O2": "360 real"}.get(label, "360 (180 + 180)")
                c = r["selected_C"].iloc[0]
                row["B1 C"] = "" if pd.isna(c) else f"{c:g}"
                if not pd.isna(c):
                    put(f"{ck}.{key}.C", float(c))
            auroc_rows.append(row)
        # paired contrasts on the main generator
        # reference row: the S-BF contrasts against B1-full, recomputed with the same draws
        row = {"Cell": cell, "B1": "B1-full (S-BF)"}
        b_full = boots[(backbone, cond, "B1-full", g)]["auroc"]
        for other in ("O1", "O2"):
            d = boots[(backbone, cond, other, g)]["auroc"] - b_full
            p = pt(backbone, cond, other, g) - pt(backbone, cond, "B1-full", g)
            row[f"{other} minus B1, AUROC {g}"] = A.d3(d, p)
            put(f"{ck}.{other}_minus_B1full.{g}.auroc", p)
            put(f"{ck}.{other}_minus_B1full.{g}.auroc.ci", A.ci(d))
        row[f"B1-eq minus B1-full, AUROC {g}"] = ""
        con_rows.append(row)
        for label in eq_labels + ["mean of 5"]:
            row = {"Cell": cell, "B1": "B1-eq, mean of 5" if label == "mean of 5" else f"B1-eq, seed {label.removeprefix('B1-eq s')}"}
            if label == "mean of 5":
                b_eq = eq_mean_boot(backbone, cond, g)
                p_eq = float(np.mean([pt(backbone, cond, l, g) for l in eq_labels]))
                key = "mean"
            else:
                b_eq = boots[(backbone, cond, label, g)]["auroc"]
                p_eq = pt(backbone, cond, label, g)
                key = label.replace("B1-eq ", "")
            for other in ("O1", "O2"):
                d = boots[(backbone, cond, other, g)]["auroc"] - b_eq
                p = pt(backbone, cond, other, g) - p_eq
                row[f"{other} minus B1, AUROC {g}"] = A.d3(d, p)
                put(f"{ck}.{other}_minus_B1eq.{key}.{g}.auroc", p)
                put(f"{ck}.{other}_minus_B1eq.{key}.{g}.auroc.ci", A.ci(d))
            d = b_eq - boots[(backbone, cond, "B1-full", g)]["auroc"]
            p = p_eq - pt(backbone, cond, "B1-full", g)
            row[f"B1-eq minus B1-full, AUROC {g}"] = A.d3(d, p)
            put(f"{ck}.B1eq_minus_B1full.{key}.{g}.auroc", p)
            put(f"{ck}.B1eq_minus_B1full.{key}.{g}.auroc.ci", A.ci(d))
            con_rows.append(row)
        # seed spread and summary statement
        eq_pts = np.array([pt(backbone, cond, l, g) for l in eq_labels])
        put(f"{ck}.B1eq.{g}.auroc.sd", float(eq_pts.std(ddof=1)))
        put(f"{ck}.B1eq.{g}.auroc.min", float(eq_pts.min()))
        put(f"{ck}.B1eq.{g}.auroc.max", float(eq_pts.max()))
        summ_rows.append({"Cell": cell, "Quantity": f"B1-eq AUROC {g}", "Mean of 5 seeds": f"{eq_pts.mean():.3f}",
                          "SD": f"{eq_pts.std(ddof=1):.3f}", "Min": f"{eq_pts.min():.3f}", "Max": f"{eq_pts.max():.3f}",
                          "Seeds with interval excluding 0": "", "Seeds with negative point estimate": "", "Statement": ""})
        for other in ("O1", "O2"):
            diffs = np.array([pt(backbone, cond, other, g) - pt(backbone, cond, l, g) for l in eq_labels])
            cis = [A.ci(boots[(backbone, cond, other, g)]["auroc"] - boots[(backbone, cond, l, g)]["auroc"]) for l in eq_labels]
            n_ex = sum(excl0(c) for c in cis)
            n_neg = int((diffs < 0).sum())
            mean_ci = VALUES[f"{ck}.{other}_minus_B1eq.mean.{g}.auroc.ci"]
            statement = ("B1-eq above" if mean_ci[1] < 0 else "real-only score above" if mean_ci[0] > 0
                         else "no detectable difference")
            k = f"{ck}.{other}_minus_B1eq.{g}.auroc"
            put(f"{k}.sd", float(diffs.std(ddof=1)))
            put(f"{k}.min", float(diffs.min()))
            put(f"{k}.max", float(diffs.max()))
            put(f"{k}.n_excl0", int(n_ex))
            put(f"{k}.n_neg", n_neg)
            put(f"{k}.statement", statement)
            summ_rows.append({"Cell": cell, "Quantity": f"{other} minus B1-eq, AUROC {g}", "Mean of 5 seeds": f"{diffs.mean():+.3f}",
                              "SD": f"{diffs.std(ddof=1):.3f}", "Min": f"{diffs.min():+.3f}", "Max": f"{diffs.max():+.3f}",
                              "Seeds with interval excluding 0": f"{n_ex} of {len(eq_labels)}",
                              "Seeds with negative point estimate": f"{n_neg} of {len(eq_labels)}", "Statement": statement})
        # operating points
        for label in ["B1-full"] + eq_labels + ["O1", "O2"]:
            r = res[(res["backbone"] == backbone) & (res["condition"] == cond) & (res["label"] == label) & (res["generator"] == g)].iloc[0]
            op_rows.append({"Cell": cell, "Method": label, f"TPR at 5% FPR {g}": A.p1(r["tpr_matched"], r["tpr_matched_ci"]),
                            "Calibrated real FPR": f"{100 * r['fpr_cal']:.1f}% ({r['fp_cal']}/{r['n_real']})",
                            f"Calibrated TPR {g}": f"{100 * r['tpr_cal']:.1f}% ({r['tp_cal']}/{r['n_fake']})"})
    put("n_real_test", n_real)
    put("n_seeds", len(eq_labels))
    boot_note = (f"Test: {n_real} real images and their content-matched synthetic images. Point estimate [percentile 95% "
                 "interval], 2,000 resamples of test source-id groups (S-BF draws, shared by every method, seed and cell). ")
    write(f"{role}_auroc_by_seed", pd.DataFrame(auroc_rows), f"S-BF-EQ {role}: AUROC of B1 at equal total training size",
          boot_note + "B1-full, O1, O2: S-BF predictions (not refitted). B1-eq: B1 refitted on the real and SD3.5 images of "
          "half of the training ids, one draw per seed; C by grouped 5-fold CV inside those rows. Mean of 5: per resample, "
          f"the mean over seeds. Source: {src}.")
    write(f"{role}_contrasts", pd.DataFrame(con_rows), f"S-BF-EQ {role}: paired AUROC differences on {g}",
          "Paired bootstrap differences (same resampled test ids). * : the 95% interval excludes 0. B1-full row: the S-BF "
          "contrasts (B1 on 1,200 images), same draws. Mean of 5: per "
          "resample, the difference to the mean B1-eq AUROC over seeds; it reflects test-sample uncertainty for these "
          f"subsamples, not between-seed variability (see seed summary). Source: {src}.")
    write(f"{role}_seed_summary", pd.DataFrame(summ_rows), f"S-BF-EQ {role}: variability over the 5 training subsamples ({g})",
          "Mean, SD (ddof 1), min and max of the 5 per-seed point estimates. Statement (fixed in PROTOCOL.md "
          "Section 8) from the interval of the seed-mean contrast: below 0 = B1-eq above; contains 0 = no detectable "
          f"difference; above 0 = real-only score above. Source: {src}.")
    write(f"{role}_operating_points", pd.DataFrame(op_rows), f"S-BF-EQ {role}: operating points on {g} (secondary)",
          "TPR at 5% FPR: threshold from the test real scores (oracle, equal real FPR for every method). Calibrated: "
          f"threshold from real calibration images only (k = ceil(0.95(n+1))). Source: {src}.")


def diagnostics(role: str) -> None:
    d = json.loads(((PRED if role == "final" else PRED / "pilot") / f"{role}_diagnostics.json").read_text())
    sub = json.loads((ROOT / "artifacts" / "supp_bfree_eq" / "subsamples.json").read_text())["roles"][role]["draws"]
    rows = []
    for c in d["cells"]:
        best = [r for r in c["B1_cv_table"] if r["C"] == c["B1_selected_C"]][0]
        ps = sub[str(c["seed"])]["per_stratum"]
        if c["train_ids_sha256"] != sub[str(c["seed"])]["sha256"]:
            raise AssertionError("predictions were fitted on ids other than the committed subsample")
        rows.append({"Cell": CELL_NAME[(c["backbone"], c["condition"])], "Seed": c["seed"],
                     "Train ids (srgb / other)": f"{c['n_train_ids']} ({ps.get('srgb', 0)} / {ps.get('other', 0)})",
                     "B1 C": f"{c['B1_selected_C']:g}", "CV mean AUROC at C": f"{best['mean_auroc']:.4f}",
                     "Max LR iterations": best["max_n_iter"], "Threshold": f"{c['threshold']:.4f}",
                     "n cal": c["n_cal"], "n real test": c["n_real_test"]})
    write(f"{role}_diagnostics", pd.DataFrame(rows), f"S-BF-EQ {role}: B1-eq fitting diagnostics",
          "C selected by grouped 5-fold CV inside the subsample rows (S-BF grid). Threshold from the real calibration "
          "images. Train ids per stratum from artifacts/supp_bfree_eq/subsamples.json.")
    if role == "pilot":
        for chk in d["code_path_check"]:
            put(f"pilot.code_path.{chk['cell']}", bool(chk["bitwise_equal"]))


# --------------------------------------------------------------------------- figure
def figure(res: pd.DataFrame, gens: list[str]) -> None:
    import figures as FG
    plt = FG.plt
    g = main_gen(gens)
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    eq_labels = [f"B1-eq s{s}" for s in seeds()]
    order = [("B1-full", "B1", "B1-full (1,200 images)")] + [(l, "B1", None) for l in eq_labels] + \
            [("O1", "O1", "O1 kNN (600 real)"), ("O2", "O2", "O2 Mahalanobis (600 real)")]
    width = 0.09
    for gi, (backbone, cond) in enumerate(CELLS):
        for j, (label, m, leg) in enumerate(order):
            r = res[(res["backbone"] == backbone) & (res["condition"] == cond) & (res["label"] == label) & (res["generator"] == g)].iloc[0]
            xp = gi + (j - (len(order) - 1) / 2) * width
            is_eq = label.startswith("B1-eq")
            ax.errorbar([xp], [r["auroc"]], yerr=[[r["auroc"] - r["auroc_ci"][0]], [r["auroc_ci"][1] - r["auroc"]]],
                        fmt=FG.MARKERS[m], color=FG.COLORS[m], ms=4.5 if is_eq else 5.5, lw=1.2, capsize=0,
                        mfc=FG.SURF if is_eq else FG.COLORS[m], mec=FG.COLORS[m], mew=1.0,
                        label=(leg if gi == 0 and leg else ("B1-eq, 5 seeds (600 images each)" if gi == 0 and label == eq_labels[0] else None)))
    ax.axhline(0.5, color=FG.INK2, lw=0.8, ls=":")
    ax.set_xticks(range(len(CELLS)), [CELL_NAME[c] for c in CELLS])
    ax.set_ylabel(f"AUROC, {g} vs real")
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=7, ncol=4)
    ax.set_title(f"S-BF-EQ: B1 at equal total training size vs real-only scores ({g}, 95% bootstrap intervals)", fontsize=9)
    fig.savefig(FIG / "eq_auroc_flux.svg", bbox_inches="tight", metadata={"Date": None})
    fig.savefig(FIG / "eq_auroc_flux.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    role = "pilot" if "--pilot" in sys.argv else "final"
    global TABLES
    if role == "pilot":
        TABLES = TABLES / "pilot"
    eq, sbf = load_eq(role), load_sbf(role)
    res, boots, gens = compute(eq, sbf, role)
    TABLES.mkdir(parents=True, exist_ok=True)
    flat = res.copy()
    for c in [c for c in flat.columns if c.endswith("_ci")]:
        flat[c + "_lo"] = flat[c].map(lambda v: v[0])
        flat[c + "_hi"] = flat[c].map(lambda v: v[1])
        flat = flat.drop(columns=c)
    flat.to_csv(TABLES / f"{role}_metrics_long.csv", index=False, lineterminator="\n")
    tables(res, boots, gens, role)
    diagnostics(role)
    if role == "final":
        figure(res, gens)
    print(f"{role}: {len(res)} metric rows, tables in {TABLES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

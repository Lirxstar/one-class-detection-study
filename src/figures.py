#!/usr/bin/env python3
"""Figures F1 to F4 from artifacts/predictions only (PROTOCOL.md 7; proposal Section 10).

SVG (committed) and PNG (ignored) in report/figures/. Method colours: validated
categorical slots 1 to 4 (blue, orange, aqua, yellow; CVD check passed on the
adjacent pairlist), always with distinct markers and a legend, so identity is
never colour alone. Intervals: paired percentile bootstrap (src/bootstrap.py).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis as A
import bootstrap as BS
from config import ROOT

OUT = ROOT / "report" / "figures"
COLORS = {"B1": "#2a78d6", "O1": "#eb6834", "O2": "#1baf7a", "M0": "#eda100"}
MARKERS = {"B1": "o", "O1": "s", "O2": "^", "M0": "D"}
LABELS = {"B1": "B1 probe (SD15 labels)", "O1": "O1 kNN", "O2": "O2 Mahalanobis", "M0": "M0 -L1 magnitude"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "legend.frameon": False, "svg.fonttype": "none", "svg.hashsalt": "ocsa",
})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def stats(g: pd.DataFrame, dr: BS.Draws):
    return A.point_block(g), A.boot_block(g, dr)


def errbar(ax, x, p, b, key, m, scale=1.0, label=None):
    lo, hi = BS.ci(b[key])
    ax.errorbar([x], [scale * p[key]], yerr=[[scale * (p[key] - lo)], [scale * (hi - p[key])]], fmt=MARKERS[m],
                color=COLORS[m], ms=6, lw=1.5, capsize=0, mec=SURF, mew=1.0, label=label)


def f1(e1: pd.DataFrame) -> None:
    d = e1[(e1["backbone"] == "primary") & (e1["data_condition"] == "d_ctrl")]
    dr = A.draws(e1)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), gridspec_kw={"width_ratios": [1, 1]})
    off = {m: (i - 1.5) * 0.17 for i, m in enumerate(A.CORE)}
    fprs = []
    for m in A.CORE:
        p, b = stats(d[d["method"] == m], dr)
        for i, gen in enumerate(A.GENS):
            errbar(axes[0], i + off[m], p, b, f"auroc_{gen}", m, label=LABELS[m] if i == 0 else None)
            errbar(axes[1], i + off[m], p, b, f"tpr_{gen}", m, scale=100)
        fprs.append(f"{m} {100 * p['fpr']:.1f}%")
    ticks = ["SD15\n(seen by B1)", "ADM", "GLIDE", "Midjourney"]
    for ax in axes:
        ax.set_xticks(range(4), ticks)
        ax.set_xlim(-0.5, 3.5)
    axes[0].axhline(0.5, color=INK2, lw=0.8, ls=":")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("AUROC vs shared R_test (95% CI)")
    axes[0].set_title("A. Per-generator AUROC, D-ctrl", loc="left", fontsize=9)
    axes[1].axhline(5, color=INK2, lw=0.8, ls=":")
    axes[1].set_ylim(0, 102)
    axes[1].set_ylabel("TPR at real-calibrated threshold (%)")
    axes[1].set_title("B. TPR at the real-calibrated threshold (5% FPR target)", loc="left", fontsize=9)
    axes[1].text(0.99, 0.97, "Real FPR on R_test: " + ", ".join(fprs), transform=axes[1].transAxes, ha="right",
                 va="top", fontsize=7.5, color=INK2)
    fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.13), fontsize=8)
    save(fig, "F1_e1_d_ctrl")


def f2(e1: pd.DataFrame, e2: pd.DataFrame) -> None:
    d = pd.concat([e1[(e1["backbone"] == "primary") & (e1["data_condition"] == "d_ctrl")],
                   e2[e2["data_condition"] == "d_ctrl"]], ignore_index=True)
    dr = A.draws(e1)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    xs = np.arange(len(A.E2_ORDER))
    for m in A.CORE:
        res = [stats(d[(d["method"] == m) & (d["perturbation"] == t)], dr) for t in A.E2_ORDER]
        for ax, key in ((axes[0], "fpr"), (axes[1], "mean_unseen_tpr")):
            y = [100 * p[key] for p, _ in res]
            lo = [100 * BS.ci(b[key])[0] for _, b in res]
            hi = [100 * BS.ci(b[key])[1] for _, b in res]
            for seg in (slice(0, 4), slice(4, 6)):  # JPEG family and blur family drawn separately
                ax.fill_between(xs[seg], lo[seg], hi[seg], color=COLORS[m], alpha=0.12, lw=0)
                ax.plot(xs[seg], y[seg], marker=MARKERS[m], color=COLORS[m], lw=2, ms=6, mec=SURF, mew=1.0,
                        label=LABELS[m] if (ax is axes[0] and seg.start == 0) else None)
    for ax in axes:
        ax.set_xticks(xs, [A.E2_NAME[t] for t in A.E2_ORDER])
        ax.axvline(3.5, color=GRID, lw=1.2)
    axes[0].axhline(5, color=INK2, lw=0.8, ls=":")
    axes[0].set_ylabel("Real FPR on R_test (%)")
    axes[0].set_title("A. Real false positives, thresholds frozen", loc="left", fontsize=9)
    axes[1].set_ylabel("Mean unseen TPR (%)")
    axes[1].set_title("B. Mean unseen TPR (ADM, GLIDE, Midjourney)", loc="left", fontsize=9)
    axes[0].set_ylim(0, None)
    axes[1].set_ylim(0, 102)
    fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.13), fontsize=8)
    save(fig, "F2_e2_d_ctrl")


def f3(e1: pd.DataFrame, e3b: pd.DataFrame) -> None:
    allp = pd.concat([e1[e1["backbone"] == "primary"], e3b], ignore_index=True)
    dr = A.draws(e1)
    conds = ["d_raw", "d_jpeg", "d_ctrl"]
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.0), sharey=True)
    for m in A.CORE:
        res = {c: stats(allp[(allp["data_condition"] == c) & (allp["method"] == m)], dr) for c in conds}
        for ax, gen in zip(axes, A.GENS):
            k = f"auroc_{gen}"
            y = [res[c][0][k] for c in conds]
            lo = [BS.ci(res[c][1][k])[0] for c in conds]
            hi = [BS.ci(res[c][1][k])[1] for c in conds]
            ax.errorbar(range(3), y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], marker=MARKERS[m], color=COLORS[m],
                        lw=2, ms=5, capsize=0, mec=SURF, mew=1.0, label=LABELS[m] if gen == "SD15" else None)
    for ax, gen in zip(axes, A.GENS):
        ax.set_xticks(range(3), ["D-raw", "D-jpeg", "D-ctrl"])
        ax.set_title(gen + (" (seen by B1)" if gen == "SD15" else ""), fontsize=9, loc="left")
        ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
        ax.set_xlim(-0.3, 2.3)
    axes[0].set_ylabel("AUROC (95% CI)")
    axes[0].set_ylim(0, 1.02)
    fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1), fontsize=8)
    save(fig, "F3_ladder")


def f4(e3c: pd.DataFrame) -> None:
    ev = e3c[~e3c["encoding_condition"].isin(["C0_cal", "C1_cal"])]
    dr = A.draws(ev)
    variants = ["C0_common", "C1_biased", "C1_matched", "C1_inverted"]
    ticks = ["C0\nboth PNG", "C1(a)\nreal JPEG75\nsynth PNG", "C1(b)\nboth JPEG75", "C1(c)\nreal PNG\nsynth JPEG75"]
    fig, grid = plt.subplots(2, 2, figsize=(10, 7.0))
    axes = grid.ravel()
    off = {m: (i - 1.5) * 0.15 for i, m in enumerate(A.CORE)}
    for m in A.CORE:
        for i, v in enumerate(variants):
            p, b = stats(ev[(ev["method"] == m) & (ev["encoding_condition"] == v)], dr)
            errbar(axes[0], i + off[m], p, b, "mean_unseen_auroc", m, label=LABELS[m] if i == 0 else None)
            errbar(axes[1], i + off[m], p, b, "fpr", m, scale=100)
            errbar(axes[3], i + off[m], p, b, "mean_unseen_tpr", m, scale=100)
        # Paired JPEG75 versus PNG real scores, holding the C1 model fixed.
        dm = ev[(ev["method"] == m) & (ev["partition"] == "R_test")]
        jpg = dm[dm["encoding_condition"] == "C1_biased"].set_index("row_id").loc[dr.ids["REAL"], "score"]
        png = dm[dm["encoding_condition"] == "C1_inverted"].set_index("row_id").loc[dr.ids["REAL"], "score"]
        delta = (jpg.to_numpy() - png.to_numpy()) / png.std()
        bm = dr.counts["REAL"] @ delta / len(delta)
        x = A.CORE.index(m)
        jitter = np.random.default_rng(42).uniform(-0.16, 0.16, len(delta))
        axes[2].scatter(x + jitter, delta, color=COLORS[m], s=3, alpha=0.18, rasterized=True)
        lo, hi = BS.ci(bm)
        axes[2].errorbar(x, delta.mean(), yerr=[[delta.mean() - lo], [hi - delta.mean()]],
                        fmt=MARKERS[m], color=INK, ms=6, capsize=4, lw=1.5)
    for ax in (axes[0], axes[1], axes[3]):
        ax.set_xticks(range(4), ticks, fontsize=7.5)
        ax.axvline(0.5, color=GRID, lw=1.2)
    axes[0].axhline(0.5, color=INK2, lw=0.8, ls=":")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("Mean unseen AUROC (95% CI)")
    axes[0].set_title("A. Discrimination on fixed identities", loc="left", fontsize=9)
    axes[1].axhline(5, color=INK2, lw=0.8, ls=":")
    axes[1].set_ylim(0, None)
    axes[1].set_ylabel("Real FPR on R_test (%)")
    axes[1].set_title("B. Real false positives at each condition's threshold", loc="left", fontsize=9)
    axes[2].set_xticks(range(4), A.CORE)
    axes[2].axhline(0, color=INK2, ls=":", lw=0.8)
    axes[2].set_ylabel("JPEG75 - PNG score / SD(PNG real score)")
    axes[2].set_title("C. Paired real-score shifts; C1 model fixed", loc="left", fontsize=9)
    axes[3].set_ylabel("Mean unseen TPR (%)")
    axes[3].set_ylim(0, 102)
    axes[3].set_title("D. Sensitivity at each condition's threshold", loc="left", fontsize=9)
    fig.subplots_adjust(hspace=0.65, wspace=0.3)
    fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=8)
    save(fig, "F4_e3c_injected_cue")


def score_distributions(e1: pd.DataFrame, condition: str = "d_ctrl") -> None:
    d = e1[(e1["backbone"] == "primary") & (e1["data_condition"] == condition) & e1["method"].isin(A.CORE)]
    d = d[d["partition"] != "R_cal"]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.6))
    groups = [("Real", d["authenticity_label"] == 0)] + [(g, d["generator"] == g) for g in A.GENS]
    shades = ["#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    for ax, m in zip(axes, A.CORE):
        dm = d["method"] == m
        lo, hi = np.percentile(d.loc[dm, "score"], [0.5, 99.5])
        bins = np.linspace(lo, hi, 40)
        for (name, mask), c in zip(groups, shades):
            ax.hist(d.loc[dm & mask, "score"], bins=bins, histtype="step", lw=1.5, color=c, density=True,
                    label=name if m == "B1" else None)
        ax.axvline(d.loc[dm, "threshold"].iloc[0], color=INK, lw=0.8, ls=":")
        ax.set_title(LABELS[m], fontsize=8.5, loc="left")
        ax.set_yticks([])
    fig.legend(loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1), fontsize=8)
    fig.subplots_adjust(top=0.80)  # room for the figure title above the panel titles
    fig.suptitle(f"Score distributions, {A.COND_NAME[condition]}; dotted line: real-calibrated threshold", fontsize=9)
    save(fig, f"S1_score_distributions_{condition}")


def real_response_curves(e1: pd.DataFrame, e2: pd.DataFrame, condition: str) -> None:
    """All 500 R_test identities, with each curve anchored to its own clean score."""
    clean = e1[(e1["backbone"] == "primary") & (e1["data_condition"] == condition)]
    d = pd.concat([clean, e2[e2["data_condition"] == condition]], ignore_index=True)
    d = d[d["partition"] == "R_test"]
    fig, axes = plt.subplots(4, 2, figsize=(9.6, 9.2), sharex="col")
    for row, m in enumerate(A.CORE):
        wide = d[d["method"] == m].pivot(index="row_id", columns="perturbation", values="score").sort_index()
        scale = wide["clean"].std()
        for col, names in enumerate((["clean", "jpeg95", "jpeg75", "jpeg50"], ["clean", "blur0.5", "blur1.0"])):
            ax = axes[row, col]
            ys = wide[names].sub(wide["clean"], axis=0).to_numpy() / scale
            xs = np.arange(len(names))
            ax.plot(xs, ys.T, color=COLORS[m], alpha=0.05, lw=0.45, rasterized=True)
            ax.plot(xs, ys.mean(axis=0), color=INK, marker=MARKERS[m], lw=1.6, label="Mean paired change")
            ax.axhline(0, color=INK2, ls=":", lw=0.6)
            ax.set_xticks(xs, [A.E2_NAME[n] for n in names])
            ax.set_ylabel(f"{m}: change / SD(clean real)")
            if row == 0:
                ax.set_title("JPEG recompression" if col == 0 else "Gaussian blur", loc="left")
    fig.suptitle(f"{A.COND_NAME[condition]}: one thin curve per R_test image; black = mean", fontsize=10)
    fig.subplots_adjust(hspace=0.2, wspace=0.3)
    save(fig, f"S2_real_response_curves_{condition}")


def main() -> None:
    e1 = A.load("e1/*.csv.gz")
    e2 = A.load("e2/*.csv.gz")
    e3b = A.load("e3/e3b_*.csv.gz")
    e3c = A.load("e3/e3c_*.csv.gz")
    if not e1.empty:
        f1(e1)
        score_distributions(e1)
    if not e2.empty:
        f2(e1, e2)
    if not e3b.empty:
        f3(e1, e3b)
        score_distributions(e3b, "d_jpeg")
    if not e1.empty:
        score_distributions(e1, "d_raw")
    if not e2.empty:
        for c in ("d_ctrl", "d_raw"):
            real_response_curves(e1, e2, c)
    if not e3c.empty:
        f4(e3c)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render DATA_CARD.md from artifacts/manifest.csv, artifacts/splits.json and
artifacts/audit/data_audit.json. Every number in the card is computed here."""
from __future__ import annotations

import json

import pandas as pd

import config
from config import ROOT

GEN_ORDER = ["Real", "SD15", "ADM", "GLIDE", "Midjourney", "VQDM", "Wukong", "BigGAN", "SD14"]


def md_table(df: pd.DataFrame, index_name: str | None = None) -> str:
    df = df.copy()
    if index_name is not None:
        df = df.reset_index().rename(columns={"index": index_name})
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.tolist()) + " |")
    return "\n".join(lines)


def main() -> None:
    core = config.load("core")
    d = core["data"]
    m = pd.read_csv(ROOT / "artifacts" / "manifest.csv", keep_default_na=False, na_values=[""])
    sp = json.loads((ROOT / "artifacts" / "splits.json").read_text())
    audit = json.loads((ROOT / "artifacts" / "audit" / "data_audit.json").read_text())
    gens = [g for g in GEN_ORDER if g in set(m["generator"])]
    present = set(m["generator"])

    counts = pd.crosstab(m["generator"], m["split"]).reindex(gens)
    counts.loc["SD14 (listed on card)"] = [0, 0]
    fmt = pd.crosstab(m["generator"], [m["format"], m["mode"]]).reindex(gens)
    fmt.columns = [f"{a} {b}" for a, b in fmt.columns]
    size_rows = []
    for g in gens:
        s = m[m["generator"] == g]
        top = (s["width"].astype(str) + "x" + s["height"].astype(str)).value_counts()
        ms = s[["width", "height"]].min(axis=1)
        size_rows.append({"generator": g, "distinct sizes": int(top.size),
                          "top sizes": ", ".join(f"{k} ({v})" for k, v in top.head(3).items()),
                          "min side min / median / max": f"{int(ms.min())} / {int(ms.median())} / {int(ms.max())}",
                          "share min side < 224": f"{(ms < 224).mean():.3f}",
                          "median bytes per pixel": f"{s['bytes_per_pixel'].median():.3f}"})
    size_tab = pd.DataFrame(size_rows)

    real = m[m["generator"] == "Real"]
    q = real["jpeg_q"].round().astype("Int64").astype(str)
    qtab = pd.crosstab(q, real["split"])
    qtab = qtab.loc[qtab.sum(axis=1).sort_values(ascending=False).index[:8]]
    qtab.index.name = "estimated quality"
    exact = real["jpeg_q_exact"].astype(str).str.lower().eq("true")
    sub = pd.crosstab(real["jpeg_subsampling"].fillna("unknown"), real["split"])
    q96 = real[exact & (real["jpeg_q"] == 96)]
    elig_real = real[real["eligible"].astype(str).str.lower().eq("true")]
    sub96 = pd.crosstab(q96["jpeg_subsampling"].fillna("unknown"), q96["split"])
    paths = {
        "train real named n<synset>_<id>.JPEG": int(real[real["split"] == "train"]["orig_path"].str.fullmatch(r"n\d{8}_\d+\.JPEG").sum()),
        "validation real named ILSVRC2012_val_<id>.JPEG": int(real[real["split"] == "validation"]["orig_path"].str.fullmatch(r"ILSVRC2012_val_\d+\.JPEG").sum()),
        "distinct synsets in train real": int(real[real["split"] == "train"]["orig_path"].str.slice(0, 9).nunique()),
    }
    path_examples = {g: m[m["generator"] == g]["orig_path"].head(2).tolist() for g in gens}
    rgba = m[m["mode"] == "RGBA"]
    alpha = rgba["alpha_min"].value_counts().to_dict()

    grp = m.groupby("dup_group").agg(size=("row_id", "size"), xs=("group_crosses_split", "first"),
                                     xl=("group_crosses_label", "first"))
    multi = grp[grp["size"] > 1]
    dup = {
        "groups with more than one row": int(len(multi)),
        "rows in such groups": int(multi["size"].sum()),
        "largest group (rows)": int(multi["size"].max()) if len(multi) else 0,
        "groups crossing train/validation": int(multi["xs"].sum()),
        "groups crossing real/synthetic": int(multi["xl"].sum()),
        "rows in groups crossing split or label": int((m["group_crosses_split"] | m["group_crosses_label"]).sum()),
        "exact-hash groups with more than one row": int(audit["exact_duplicates_rgb"]["groups_with_duplicates"]),
    }

    import make_splits
    dupex = make_splits.duplicate_exclusions(m)
    dupex = dupex[dupex["passing_other_rules"] > 0]
    reasons = pd.crosstab(m["exclusion_reason"].fillna("eligible").replace("", "eligible"),
                          [m["split"], m["generator"]])
    reasons.columns = [f"{a}/{b}" for a, b in reasons.columns]
    reasons = reasons.loc[:, (reasons.sum() > 0)]

    part = m[m["partition"].notna() & (m["partition"] != "")]
    ptab = pd.crosstab(part["partition"], part["generator"])
    order = ["R_train", "F_seen_train", "R_cal", "R_test", "F_seen_test", "F_unseen_test", "P_cal", "P_test", "P_fake"]
    ptab = ptab.reindex([p for p in order if p in ptab.index])

    def kv(dct):
        return "\n".join(f"- {k}: {v}" for k, v in dct.items())

    text = f"""# DATA CARD

Generated by `src/data_card.py` from `artifacts/audit/data_audit.json`,
`artifacts/manifest.csv` and `artifacts/splits.json` (manifest sha256
`{sp['manifest_sha256']}`). Do not edit by hand.

## Provenance

- Repository `{d['repository']}`, revision `{d['revision']}`; upstream named on the card: `yangsangtai/tiny-genimage` (Kaggle); license CC BY-NC-SA 4.0.
- {audit['n_rows']} rows in {len(audit['files'])} Parquet shards; Pillow {audit['pillow_version']} used for the audit. Shard sha256 values: `artifacts/provenance/dataset_files.sha256`.
- Fields: `image` (struct of stored `bytes` and original file name `path`), `label` (0 real, 1 fake), `generator` (ClassLabel with 9 names).

## Rows per split and generator

{md_table(counts, 'generator')}

**SD14 is absent.** The card's ClassLabel lists `SD14` (index 5), but no row has generator 5 in either split. `SD15` is therefore the seen generator (PROTOCOL.md Section 2). Labels are consistent with generators in every row: {audit['g0_signals']['labels_consistent_with_generator']}. Decode failures: {audit['g0_signals']['decode_failures']}.

## Stored format and colour mode

{md_table(fmt, 'generator')}

**ADM is stored as RGBA PNG.** All {len(rgba)} RGBA rows are ADM; minimum alpha per image (value: count): {alpha}. The RGB conversion therefore only drops a constant opaque channel; it is asserted per image at every decode. The extra channel raises ADM's stored bytes per pixel, which matters for the metadata baseline (E3A). Real images in mode `L` are not eligible.

## Sizes and bytes per pixel

{md_table(size_tab)}

## JPEG properties of real images

All real rows are JPEG; share whose quantization tables match a scaled IJG table exactly: {exact.mean():.4f}. Estimated quality (8 most frequent):

{md_table(qtab, "estimated quality")}

Chroma subsampling, all real rows:

{md_table(sub, 'subsampling')}

Chroma subsampling, real rows with exact IJG quality 96:

{md_table(sub96, 'subsampling')}

Modal subsampling among eligible real rows: `{elig_real['jpeg_subsampling'].mode().iloc[0]}` (D-ctrl offset multiple m = 8). EXIF orientation values among real rows (value: count): {real['exif_orientation'].value_counts(dropna=False).to_dict()}.

## Upstream names and semantic labels

{kv(paths)}

Example `image.path` values: {json.dumps(path_examples)}

Train real images look like ImageNet train files and validation real images like ImageNet (ILSVRC2012) validation files; the latter carry no class label, so `R_cal`/`R_test` are not stratified. Synthetic names start with a number that may be an ImageNet class index; it is not used.

## Duplicates (exact RGB hash or dHash Hamming <= 4, union-find over all rows)

{kv(dup)}

Chaining through low-texture images produces a few large groups. Groups that cross the train/validation boundary or the real/synthetic labels are excluded entirely; other groups contribute one representative.

Rows excluded by the duplicate rules although they pass every other rule (per split and generator; total {int(dupex['excluded_by_duplicates'].sum())}; largest group {int(grp['size'].max())} rows):

{md_table(dupex)}

## Eligibility (first failed rule per row)

{md_table(reasons, 'reason')}

Eligible pool sizes: {json.dumps(sp['eligible_pool_sizes'])}

## Partitions

{md_table(ptab, 'partition')}

`F_seen_train` holds all eligible train SD15 rows (shortfall from the 2,000 target). `P_*` partitions are train-only pilot data.

## G0 decision

Passed: the derivative keeps original-looking stored bytes (real JPEG with mixed qualities and {audit['g0_signals']['real_distinct_sizes']} distinct sizes; synthetic PNG at the native GenImage sizes), so D-jpeg and D-ctrl can be built. SD14 is substituted by SD15; ADM is kept with an alpha assertion; real images are restricted to 4:4:4.

## Limitations

- A PNG container does not prove the absence of an earlier JPEG history.
- Real images come from two ImageNet subsets (train real from ImageNet train, validation real from ILSVRC2012 val) with different quality mixes; within each split the eligibility rule fixes the stored JPEG history to exact IJG Q96 4:4:4.
- No content matching between real and synthetic images; semantic differences remain.
- Near-duplicate chaining removes some low-texture images from every partition.
- SD15 replaces SD14, so results are not directly comparable with "train on SD 1.4" GenImage results.
"""
    (ROOT / "DATA_CARD.md").write_text(text, encoding="utf-8")
    print("wrote DATA_CARD.md")


if __name__ == "__main__":
    main()

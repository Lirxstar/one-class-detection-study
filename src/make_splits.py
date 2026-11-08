#!/usr/bin/env python3
"""Eligibility, duplicate handling, and partitions (PROTOCOL.md Sections 2.3 to 2.5).

Input: `artifacts/audit/audit_rows.csv` from `src/audit_dataset.py`.
Output: `artifacts/manifest.csv` (one row per dataset row) and
`artifacts/splits.json` (partition -> sorted row_ids, counts, manifest sha256).

Usage: python src/make_splits.py [--counts-only]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from config import ROOT
from data_io import alpha_min, read_bytes
from duplicates import duplicate_groups

AUDIT_ROWS = ROOT / "artifacts" / "audit" / "audit_rows.csv"
MANIFEST = ROOT / "artifacts" / "manifest.csv"
SPLITS = ROOT / "artifacts" / "splits.json"

MANIFEST_COLUMNS = [
    "row_id", "split", "file", "row_in_file", "label", "generator", "orig_path", "format", "width", "height",
    "mode", "n_bytes", "bytes_per_pixel", "jpeg_q", "jpeg_q_exact", "jpeg_subsampling", "jpeg_layers",
    "exif_orientation", "alpha_min", "rgb_sha1", "dhash", "dup_group", "dup_group_size", "group_crosses_split",
    "group_crosses_label", "role", "row_rules_ok", "eligible", "exclusion_reason", "partition", "supp_meta",
]


def _true(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().eq("true")


def roles(df: pd.DataFrame, d: dict) -> pd.Series:
    """Which pool a row may enter (PROTOCOL.md 2.2); 'none' means unused."""
    seen, unseen, dev = d["generator_seen"], d["generators_unseen"], d["generators_dev_only"]
    r = pd.Series("none", index=df.index, dtype=object)
    tr, va = df["split"] == "train", df["split"] == "validation"
    real = df["generator"] == d["real_generator"]
    r[tr & real] = "train_real"
    r[va & real] = "validation_real"
    r[tr & (df["generator"] == seen)] = "train_seen"
    r[va & (df["generator"] == seen)] = "validation_seen"
    r[va & df["generator"].isin(unseen)] = "validation_unseen"
    r[tr & df["generator"].isin(dev)] = "train_dev"
    return r


def row_rules(df: pd.DataFrame, elig: dict) -> tuple[pd.Series, pd.Series]:
    """Per-row eligibility rules and the first failed rule as reason."""
    reason = pd.Series("", index=df.index, dtype=object)

    def fail(mask, why):
        m = mask & reason.eq("")
        reason[m] = why

    is_real = df["label"] == 0
    re_, sy = elig["real"], elig["synthetic"]
    fail(~df["decode_ok"].astype(bool), "decode_failed")
    fail(is_real != (df["generator"] == "Real"), "label_generator_inconsistent")
    fail(df["role"] == "none", "no_role_for_generator_in_split")
    fail(df["group_crosses_split"], "dup_group_crosses_split")
    fail(df["group_crosses_label"], "dup_group_crosses_label")
    # real rows
    fail(is_real & (df["format"] != re_["format"]), "real_format")
    fail(is_real & (df["mode"] != re_["mode"]), "real_mode")
    fail(is_real & (df["jpeg_layers"] != re_["jpeg_components"]), "real_jpeg_components")
    fail(is_real & ~(_true(df["jpeg_q_exact"]) & (df["jpeg_q"] == re_["jpeg_quality_exact_ijg"])), "real_not_exact_ijg_q96")
    allowed = [v for v in re_["exif_orientation_allowed"] if v is not None]
    ori_ok = df["exif_orientation"].isna() | df["exif_orientation"].isin(allowed)
    fail(is_real & ~ori_ok, "real_exif_orientation")
    fail(is_real & (df["jpeg_subsampling"] != re_["chroma_subsampling"]), "real_chroma_subsampling")
    fail(is_real & (df["min_side"] < re_["min_side"]), "min_side")
    # synthetic rows
    fail(~is_real & (df["format"] != sy["format"]), "synthetic_format")
    mode_ok = df["mode"].eq("RGB") | (df["mode"].eq("RGBA") & df["alpha_min"].eq(255))
    fail(~is_real & df["mode"].eq("RGBA") & ~df["alpha_min"].eq(255), "synthetic_alpha_not_255")
    fail(~is_real & ~mode_ok, "synthetic_mode")
    fail(~is_real & (df["min_side"] < sy["min_side"]), "min_side")
    return reason.eq(""), reason


def duplicate_exclusions(m: pd.DataFrame) -> pd.DataFrame:
    """Rows of a usable pool that pass every other rule but are excluded by the duplicate rules
    (group crosses split, group crosses label, or not the group representative), per split and generator."""
    d = m.copy()
    d["decode_ok"] = d["format"].notna()
    d["min_side"] = d[["width", "height"]].min(axis=1)
    d["alpha_min"] = pd.to_numeric(d["alpha_min"], errors="coerce")
    d["group_crosses_split"] = False
    d["group_crosses_label"] = False
    ok_nodup, _ = row_rules(d, config.load("core")["data"]["eligibility"])
    elig = d["eligible"].astype(str).str.lower().eq("true")
    assert not (elig & ~ok_nodup).any()
    ex = ok_nodup & ~elig
    out = d[ok_nodup].groupby(["split", "generator"]).size().rename("passing_other_rules").to_frame()
    out["excluded_by_duplicates"] = d[ex].groupby(["split", "generator"]).size()
    out["crosses_split"] = d[ex & m["group_crosses_split"].astype(str).str.lower().eq("true")].groupby(["split", "generator"]).size()
    out["crosses_label"] = d[ex & m["group_crosses_label"].astype(str).str.lower().eq("true")].groupby(["split", "generator"]).size()
    return out.fillna(0).astype(int).reset_index()


def permuted(pool: pd.DataFrame, seed: int) -> list[str]:
    ids = sorted(pool["row_id"])
    perm = np.random.default_rng(seed).permutation(len(ids))
    return [ids[i] for i in perm]


def build(counts_only: bool = False) -> dict:
    core = config.load("core")
    d = core["data"]
    seed = int(core["seeds"]["split"])
    t = d["targets"]
    df = pd.read_csv(AUDIT_ROWS)
    df = df.sort_values("row_id").reset_index(drop=True)
    df = df.join(duplicate_groups(df, d["duplicates"]["near"]["max_hamming"]))

    rgba = df.loc[df["mode"].eq("RGBA"), "row_id"].tolist()
    amin = {rid: alpha_min(b) for rid, b in read_bytes(rgba).items()} if rgba else {}
    df["alpha_min"] = df["row_id"].map(amin).astype("Int64")

    df["role"] = roles(df, d)
    ok, reason = row_rules(df, d["eligibility"])
    df["row_rules_ok"] = ok
    # one representative per duplicate group among rows passing the row rules
    rep = df[ok].drop_duplicates("dup_group", keep="first").index  # df is sorted by row_id
    df["eligible"] = False
    df.loc[rep, "eligible"] = True
    reason[ok & ~df["eligible"]] = "duplicate_not_representative"
    df["exclusion_reason"] = reason

    # supplementary metadata baseline members: decodable, not in an excluded group, one per group
    base = df["decode_ok"].astype(bool) & ~df["group_crosses_split"] & ~df["group_crosses_label"]
    supp = df[base].drop_duplicates("dup_group", keep="first").index
    df["supp_meta"] = False
    df.loc[supp, "supp_meta"] = True

    pools = {
        "train_real": df[df["eligible"] & (df["role"] == "train_real")],
        "train_seen": df[df["eligible"] & (df["role"] == "train_seen")],
        "validation_real": df[df["eligible"] & (df["role"] == "validation_real")],
    }
    for g in d["generators_unseen"]:
        pools[f"validation_{g}"] = df[df["eligible"] & (df["generator"] == g) & (df["split"] == "validation")]
    pools["validation_seen"] = df[df["eligible"] & (df["role"] == "validation_seen")]
    for g in d["generators_dev_only"]:
        pools[f"train_{g}"] = df[df["eligible"] & (df["generator"] == g) & (df["split"] == "train")]
    pool_sizes = {k: int(len(v)) for k, v in pools.items()}

    # shortfall rule (PROTOCOL.md 2.4), applied from eligible counts only
    n_rtrain = min(t["R_train"], pool_sizes["train_real"])
    n_fseen = min(t["F_seen_train_max"], pool_sizes["train_seen"])
    n_rcal = n_rtest = min(t["R_cal"], t["R_test"], pool_sizes["validation_real"] // 2)
    gens_test = [d["generator_seen"], *d["generators_unseen"]]
    n_ftest = min(t["fake_test_per_generator"],
                  pool_sizes["validation_seen"], *[pool_sizes[f"validation_{g}"] for g in d["generators_unseen"]])
    dev_t = d["dev_targets"]
    surplus = pool_sizes["train_real"] - n_rtrain
    n_pcal = min(dev_t["P_cal"], surplus // 2)
    n_ptest = min(dev_t["P_test"], surplus - n_pcal)
    sizes = {"R_train": n_rtrain, "F_seen_train": n_fseen, "R_cal": n_rcal, "R_test": n_rtest,
             "F_test_per_generator": n_ftest, "P_cal": n_pcal, "P_test": n_ptest}
    if counts_only:
        return {"pool_sizes": pool_sizes, "sizes": sizes}

    parts: dict[str, object] = {}
    tr = permuted(pools["train_real"], seed)
    parts["R_train"] = tr[:n_rtrain]
    parts["P_cal"] = tr[n_rtrain:n_rtrain + n_pcal]
    parts["P_test"] = tr[n_rtrain + n_pcal:n_rtrain + n_pcal + n_ptest]
    parts["F_seen_train"] = permuted(pools["train_seen"], seed)[:n_fseen]
    va = permuted(pools["validation_real"], seed)
    parts["R_cal"] = va[:n_rcal]
    parts["R_test"] = va[n_rcal:n_rcal + n_rtest]
    parts["F_seen_test"] = permuted(pools["validation_seen"], seed)[:n_ftest]
    parts["F_unseen_test"] = {g: permuted(pools[f"validation_{g}"], seed)[:n_ftest] for g in d["generators_unseen"]}
    p_fake = {}
    for g in d["generators_dev_only"]:
        p_fake[g] = permuted(pools[f"train_{g}"], seed)[:min(dev_t["P_fake_per_generator"], pool_sizes[f"train_{g}"])]
    parts["P_fake"] = p_fake

    df["partition"] = ""
    for name, v in parts.items():
        ids = [i for lst in v.values() for i in lst] if isinstance(v, dict) else v
        m = df["row_id"].isin(ids)
        assert (df.loc[m, "partition"] == "").all(), f"row in two partitions ({name})"
        df.loc[m, "partition"] = name

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df[MANIFEST_COLUMNS].to_csv(MANIFEST, index=False, lineterminator="\n")
    sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()

    def srt(v):
        return {k: sorted(x) for k, x in v.items()} if isinstance(v, dict) else sorted(v)

    counts = {k: ({g: len(x) for g, x in v.items()} if isinstance(v, dict) else len(v)) for k, v in parts.items()}
    out = {
        "protocol_version": core["protocol_version"],
        "dataset_revision": d["revision"],
        "seed_split": seed,
        "generator_seen": d["generator_seen"],
        "generators_unseen": d["generators_unseen"],
        "generators_dev_only": d["generators_dev_only"],
        "manifest_sha256": sha,
        "eligible_pool_sizes": pool_sizes,
        "counts": counts,
        "exclusion_reasons": {k: int(v) for k, v in df["exclusion_reason"].replace("", "eligible").value_counts().items()},
        "partitions": {k: srt(v) for k, v in parts.items()},
    }
    SPLITS.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts-only", action="store_true")
    a = ap.parse_args()
    out = build(a.counts_only)
    print(json.dumps({k: v for k, v in out.items() if k != "partitions"}, indent=1))


if __name__ == "__main__":
    main()

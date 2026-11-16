#!/usr/bin/env python3
"""S-BF partitions by source id (PROTOCOL.md Section 7). No feature or score is read.

1. Eligible source ids from the Step 1 audit (all three files pass).
2. Source-id groups: ids are linked when any of their images share an exact RGB hash or a dHash
   within Hamming distance 4 (core rule, `duplicates.duplicate_groups` over all eligible files),
   so a duplicate never straddles partitions.
3. Strata: `srgb` (groups whose real images are all tagged Nikon sRGB) and `other`. In each stratum
   the groups (sorted by smallest id) are permuted with a fresh `default_rng(42)` and cut in order
   into train, calibration and test at 60/20/20 of the stratum's ids (a group goes to the partition
   where its first id falls). The sensitivity subset is every id whose real image is tagged Nikon
   sRGB (`srgb_tagged_id`), including the few in mixed groups.
4. Row partitions: R_train (REAL) and F_seen_train (SD35) of train ids; R_cal (REAL) of calibration
   ids; R_test (REAL), F_unseen_test (FLUX) and F_seen_test (SD35) of test ids. FLUX of train and
   calibration ids and SD35 of calibration ids are `unused` and never read.
5. Development pilot, inside train ids only, same procedure: P_fit (REAL, SD35), P_cal (REAL),
   P_test (REAL) and P_fake (SD35). No FLUX image and no calibration or test id enters the pilot.

Outputs: artifacts/supp_bfree/manifest.csv and artifacts/supp_bfree/splits.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import supp_bfree_data as S
from config import ROOT
from duplicates import UnionFind, duplicate_groups

AUDIT = ROOT / "artifacts" / "supp_bfree" / "audit"
OUT = ROOT / "artifacts" / "supp_bfree"
ORDER = ["train", "cal", "test"]


def id_groups(files: pd.DataFrame) -> pd.Series:
    """Source id -> group id (smallest source id in the group)."""
    df = files.assign(split="all", label=files["class"].map(S.LABEL)).reset_index(drop=True)
    g = duplicate_groups(df[["rgb_sha1", "dhash", "split", "label"]], max_hamming=int(S.cfg()["duplicates"]["near"]["max_hamming"]))
    ids = sorted(df["source_id"].unique())
    pos = {s: i for i, s in enumerate(ids)}
    uf = UnionFind(len(ids))
    for _, members in df.groupby(g["dup_group"])["source_id"]:
        m = sorted(set(members))
        for s in m[1:]:
            uf.union(pos[m[0]], pos[s])
    root = {s: ids[uf.find(pos[s])] for s in ids}
    return pd.Series(root, name="id_group")


def allocate(groups: pd.DataFrame, props: dict, names: list[str], seed: int) -> pd.Series:
    """groups: one row per source id with columns id_group, stratum. Returns source id -> partition name."""
    out = {}
    keys = list(props)
    for stratum, g in groups.groupby("stratum", sort=True):
        members = g.groupby("id_group").groups
        order = sorted(members)
        perm = np.random.default_rng(seed).permutation(len(order))
        n = len(g)
        n_cal, n_test = int(round(props[keys[1]] * n)), int(round(props[keys[2]] * n))
        cut = [n - n_cal - n_test, n - n_test]
        pos = 0
        for i in perm:
            ids = sorted(members[order[i]])
            part = names[0] if pos < cut[0] else names[1] if pos < cut[1] else names[2]
            for s in ids:
                out[s] = part
            pos += len(ids)
    return pd.Series(out)


def build() -> tuple[pd.DataFrame, dict]:
    c = S.cfg()
    audit = json.loads((AUDIT / "audit.json").read_text())
    if not audit["gate"]["passed"]:
        raise RuntimeError("gate G0-BF failed; see DATA_CARD_SUPP_BFREE.md")
    files = pd.read_csv(AUDIT / "files.csv", keep_default_na=False,
                        dtype={"dhash": str, "rgb_sha1": str, "source_id": str, "sha256": str})
    per_id = files.groupby("source_id").agg(n=("class", "size"), ok=("file_ok", lambda s: s.astype(str).eq("True").all()),
                                            icc=("icc_ok", lambda s: s.astype(str).eq("True").all()),
                                            srgb=("srgb_tagged", lambda s: s.astype(str).eq("True").any()))
    eligible = sorted(per_id.index[(per_id["n"] == 3) & per_id["ok"] & per_id["icc"]])
    files = files[files["source_id"].isin(eligible)].copy()
    grp = id_groups(files)
    ids = pd.DataFrame({"id_group": grp.loc[eligible], "srgb": per_id.loc[eligible, "srgb"]})
    ids["stratum"] = np.where(ids.groupby("id_group")["srgb"].transform("all"), "srgb", "other")
    seed = int(c["splits"]["seed"])
    ids["split_role"] = allocate(ids, c["splits"]["proportions"], ORDER, seed)
    tr = ids[ids["split_role"] == "train"]
    ids["pilot_role"] = allocate(tr, c["splits"]["pilot"], ["fit", "cal", "test"], seed).reindex(ids.index).fillna("")

    files["label"] = files["class"].map(S.LABEL)
    ids["srgb_tagged_id"] = ids["srgb"].map({True: "True", False: "False"})
    files = files.join(ids[["id_group", "stratum", "srgb_tagged_id", "split_role", "pilot_role"]], on="source_id")
    part = {("train", "REAL"): "R_train", ("train", "SD35"): "F_seen_train", ("cal", "REAL"): "R_cal",
            ("test", "REAL"): "R_test", ("test", "FLUX"): "F_unseen_test", ("test", "SD35"): "F_seen_test"}
    files["partition"] = [part.get((r, k), "unused") for r, k in zip(files["split_role"], files["class"])]
    pil = {("fit", "REAL"): "P_fit_real", ("fit", "SD35"): "P_fit_fake", ("cal", "REAL"): "P_cal",
           ("test", "REAL"): "P_test", ("test", "SD35"): "P_fake"}
    files["pilot_partition"] = [pil.get((r, k), "") for r, k in zip(files["pilot_role"], files["class"])]
    cols = ["row_id", "class", "label", "source_id", "id_group", "stratum", "srgb_tagged_id", "split_role", "partition",
            "pilot_role",
            "pilot_partition", "sha256", "rgb_sha1", "dhash", "icc_description", "width", "height"]
    manifest = files[cols].sort_values("row_id").reset_index(drop=True)

    def lists(col):
        return {p: sorted(g["row_id"]) for p, g in manifest[manifest[col] != ""].groupby(col) if p != "unused"}

    parts, pilot = lists("partition"), lists("pilot_partition")
    srgb_rows = set(manifest.loc[manifest["srgb_tagged_id"] == "True", "row_id"])
    gsize = ids.groupby("id_group").size()
    splits = {
        "protocol": c["protocol_version"], "seed": seed, "supp_config_hash": S.supp_config_hash(),
        "eligible_ids": len(eligible), "id_groups": int(len(gsize)), "id_groups_with_more_than_one_id": int((gsize > 1).sum()),
        "largest_id_group": int(gsize.max()),
        "ids_per_role": ids["split_role"].value_counts().reindex(ORDER).to_dict(),
        "ids_per_role_and_stratum": {f"{r}/{s}": int(n) for (r, s), n in ids.groupby(["split_role", "stratum"]).size().items()},
        "pilot_ids_per_role": ids.loc[ids["pilot_role"] != "", "pilot_role"].value_counts().to_dict(),
        "sizes": {p: len(v) for p, v in parts.items()},
        "sizes_sensitivity_srgb": {p: len(set(v) & srgb_rows) for p, v in parts.items()},
        "partitions": parts, "pilot_partitions": pilot,
        "sensitivity_srgb_row_ids": sorted(srgb_rows & set().union(*parts.values())),
    }
    return manifest, splits


def main() -> None:
    manifest, splits = build()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUT / "manifest.csv", index=False, lineterminator="\n")
    (OUT / "splits.json").write_text(json.dumps(splits, indent=1) + "\n")
    show = {k: v for k, v in splits.items() if k not in ("partitions", "pilot_partitions", "sensitivity_srgb_row_ids")}
    print(json.dumps(show, indent=1))


if __name__ == "__main__":
    main()

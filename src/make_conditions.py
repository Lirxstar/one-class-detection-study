#!/usr/bin/env python3
"""Build D-raw, D-jpeg and D-ctrl views for every partition row (PROTOCOL.md Section 3).

Writes uint8 memmaps `artifacts/cache/views/{d_raw,d_jpeg,d_ctrl}.npy` of shape
(N, 224, 224, 3) and `artifacts/cache/views/index.csv` (row order, partition,
generator, label, native size, D-ctrl crop offsets). Rows are in sorted
row_id order; every condition uses the same rows (shared identities).

Usage: python src/make_conditions.py [--workers 8] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from conditions import make_views
from config import ROOT
from data_io import DATA_DIR, decode_rgb, parse_row_id

VIEW_DIR = ROOT / "artifacts" / "cache" / "views"
CONDS = ("d_raw", "d_jpeg", "d_ctrl")


def partition_rows() -> pd.DataFrame:
    sp = json.loads((ROOT / "artifacts" / "splits.json").read_text())
    man = pd.read_csv(ROOT / "artifacts" / "manifest.csv", usecols=["row_id", "label", "generator", "rgb_sha1"])
    rows = []
    for part, v in sp["partitions"].items():
        items = [(g, rid) for g, lst in v.items() for rid in lst] if isinstance(v, dict) else [(None, rid) for rid in v]
        rows += [{"row_id": rid, "partition": part} for _, rid in items]
    df = pd.DataFrame(rows).merge(man, on="row_id", how="left", validate="one_to_one")
    return df.sort_values("row_id").reset_index(drop=True)


def _work(args):
    stem, items, n_total = args  # items: list of (position, row_index, is_real)
    col = pq.read_table(DATA_DIR / f"{stem}.parquet", columns=["image"]).column("image")
    recs = col.take([i for _, i, _ in items]).to_pylist()
    mm = {c: np.load(VIEW_DIR / f"{c}.npy", mmap_mode="r+") for c in CONDS}
    meta = []
    for (pos, _, is_real), rec in zip(items, recs):
        v = make_views(decode_rgb(rec["bytes"]), bool(is_real))
        for c in CONDS:
            mm[c][pos] = v[c]
        meta.append((pos, v["height"], v["width"], v["ctrl_top"], v["ctrl_left"]))
    for c in CONDS:
        mm[c].flush()
    return meta


def build(workers: int = 8, limit: int | None = None) -> pd.DataFrame:
    df = partition_rows()
    if limit:
        df = df.iloc[:limit].copy()
    VIEW_DIR.mkdir(parents=True, exist_ok=True)
    n = len(df)
    for c in CONDS:
        np.lib.format.open_memmap(VIEW_DIR / f"{c}.npy", mode="w+", dtype=np.uint8, shape=(n, 224, 224, 3)).flush()
    jobs = {}
    for pos, (rid, lab) in enumerate(zip(df["row_id"], df["label"])):
        _, stem, idx = parse_row_id(rid)
        jobs.setdefault(stem, []).append((pos, idx, lab == 0))
    t = time.time()
    meta = []
    with ProcessPoolExecutor(workers) as ex:
        for out in ex.map(_work, [(s, it, n) for s, it in sorted(jobs.items())]):
            meta += out
            print(f"  {len(meta)}/{n} views, {time.time() - t:.0f}s", flush=True)
    m = pd.DataFrame(meta, columns=["pos", "height", "width", "ctrl_top", "ctrl_left"]).sort_values("pos")
    df[["height", "width", "ctrl_top", "ctrl_left"]] = m[["height", "width", "ctrl_top", "ctrl_left"]].to_numpy()
    df.insert(0, "pos", np.arange(n))
    df.to_csv(VIEW_DIR / "index.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    df = build(a.workers, a.limit)
    print(df.groupby(["partition", "generator"]).size())


if __name__ == "__main__":
    main()

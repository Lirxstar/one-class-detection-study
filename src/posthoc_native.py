#!/usr/bin/env python3
"""POST-HOC, EXPLORATORY (added after E1 to E3 results were seen, at the author's request;
PROTOCOL.md v0.6.4).

The missing ladder cell: native-resolution crop (as D-ctrl) WITHOUT JPEG harmonization of
synthetic images ("D-native"). D-native vs D-ctrl isolates JPEG harmonization at native
resolution; D-raw vs D-native isolates resizing plus field of view without harmonization.
Same identities, methods, CV rule, calibration rule and bootstrap as E1. Ran once from tag
`posthoc-d-native` in the original repository; the public script refuses to overwrite outputs.

Steps: build views (artifacts/cache/views/d_native.npy, same row order as index.csv), assert real
views equal D-ctrl views, extract ViT-S/14 features for the study partitions, run the E1 protocol.
"""
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import conditions as C
import encoder
import pipeline as P
from config import ROOT
from data_io import DATA_DIR, decode_rgb, parse_row_id
from extract_features import STUDY, run_job

VIEW_DIR = ROOT / "artifacts" / "cache" / "views"
OUT = P.PRED_DIR / "posthoc"


def _work(args):
    stem, items = args
    col = pq.read_table(DATA_DIR / f"{stem}.parquet", columns=["image"]).column("image")
    recs = col.take([i for _, i in items]).to_pylist()
    mm = np.load(VIEW_DIR / "d_native.npy", mmap_mode="r+")
    for (pos, _), rec in zip(items, recs):
        mm[pos] = C.native_view(decode_rgb(rec["bytes"]))
    mm.flush()
    return len(items)


def build_views() -> None:
    idx = pd.read_csv(VIEW_DIR / "index.csv")
    np.lib.format.open_memmap(VIEW_DIR / "d_native.npy", mode="w+", dtype=np.uint8, shape=(len(idx), 224, 224, 3)).flush()
    jobs = {}
    for pos, rid in zip(idx["pos"], idx["row_id"]):
        _, stem, i = parse_row_id(rid)
        jobs.setdefault(stem, []).append((pos, i))
    with ProcessPoolExecutor(8) as ex:
        n = sum(ex.map(_work, sorted(jobs.items())))
    nat = np.load(VIEW_DIR / "d_native.npy", mmap_mode="r")
    ctrl = np.load(VIEW_DIR / "d_ctrl.npy", mmap_mode="r")
    real = idx.loc[idx["label"] == 0, "pos"].to_numpy()
    assert all(np.array_equal(nat[p], ctrl[p]) for p in real), "real D-native views must equal D-ctrl views"
    synth = idx.loc[idx["label"] == 1, "pos"].to_numpy()
    n_diff = int(sum(not np.array_equal(nat[p], ctrl[p]) for p in synth))
    (ROOT / "artifacts" / "audit" / "posthoc_d_native_views.json").write_text(json.dumps(
        {"n_views": n, "real_equal_to_d_ctrl": int(len(real)), "synthetic_differing_from_d_ctrl": n_diff,
         "n_synthetic": int(len(synth))}, indent=1) + "\n")
    print(f"built {n} D-native views; synthetic differing from D-ctrl: {n_diff}/{len(synth)}")


def main():
    outs = [OUT / "posthoc_primary_d_native.csv.gz", OUT / "posthoc_diagnostics.json"]
    commit = P.assert_new_run_allowed(outs)
    build_views()
    run_job(encoder.load_backbone("primary"), "primary", "d_native", "clean", STUDY)
    run_id = P.new_run_id("posthoc-d-native")
    preds, _, _, diag = P.run_clean("final", "primary", "d_native", "POSTHOC-D-native", run_id)
    P.save(preds, outs[0])
    outs[1].write_text(json.dumps({"run_id": run_id, "git_commit": commit, "label": "post hoc, exploratory",
                                   "d_native": diag}, indent=1, default=float) + "\n")
    print(f"saved {outs[0].name}: {len(preds)} rows")


if __name__ == "__main__":
    main()

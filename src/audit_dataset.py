#!/usr/bin/env python3
"""E0 dataset audit for Tiny GenImage (proposal v0.6, Sections 3.7 and E0).

Reads the Parquet shards directly with pyarrow so that the *stored* image bytes
are inspected. (Loading through `datasets` decodes images to PIL first and
hides the original container, JPEG tables and file size.)

For every row it records: split, file, row index, label, generator, original
`path` field (if any), stored format, width, height, color mode, byte size,
JPEG quality estimated from the quantization tables (exact match against the
IJG standard tables scaled to quality 1..100), chroma subsampling, EXIF
orientation, decode errors, an exact hash of the decoded RGB pixels, and a
64-bit dHash for near-duplicate screening.

Outputs (in --out):
  audit_rows.csv     one row per sample, no image bytes
  data_audit.json    distributions by split/class/generator, duplicates,
                     eligibility counts for D-ctrl, and G0 signals

Usage:
  # quick look on one or two shards
  python src/audit_dataset.py --data-dir data/tiny_genimage --max-rows-per-file 1500 --out artifacts/audit_quick
  # full audit
  python src/audit_dataset.py --data-dir data/tiny_genimage --revision <sha> --workers 4 --out artifacts/audit

The G0 decision itself is yours; this script only collects the evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import PIL
import pyarrow.parquet as pq
from PIL import Image, JpegImagePlugin

SUBSAMPLING = {-1: "unknown", 0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}
EXPECTED_SYNTH_SIZE = {"SD14": "512x512", "ADM": "256x256", "GLIDE": "256x256", "Midjourney": "1024x1024"}
TARGETS = {"R_train": 2000, "F_seen_train_max": 2000, "R_cal_plus_R_test": 1000, "fake_test_per_generator": 300}


# --------------------------------------------------------------------------- JPEG
def reference_qtables() -> dict[int, dict[int, np.ndarray]]:
    """IJG standard tables scaled to quality 1..100, produced by Pillow's own
    libjpeg so that table order and format match `Image.quantization` exactly."""
    ref = {}
    dummy = Image.new("RGB", (16, 16), (128, 128, 128))
    for q in range(1, 101):
        buf = io.BytesIO()
        dummy.save(buf, "JPEG", quality=q, subsampling=0)
        tabs = Image.open(io.BytesIO(buf.getvalue())).quantization
        ref[q] = {k: np.asarray(v, dtype=np.int32) for k, v in tabs.items()}
    return ref


def estimate_jpeg_quality(qt: dict, ref: dict) -> tuple[int | None, float, bool, int]:
    """Return (best_quality, mean_abs_table_error, exact_ijg_match, n_tables)."""
    if not qt or 0 not in qt:
        return None, float("nan"), False, len(qt or {})
    luma = np.asarray(qt[0], dtype=np.int32)
    chroma = np.asarray(qt[1], dtype=np.int32) if 1 in qt else None
    best_q, best_err = None, np.inf
    for q, tabs in ref.items():  # ascending q; strict "<" keeps the lowest q on ties
        err = np.abs(luma - tabs[0]).mean()
        if chroma is not None and 1 in tabs:
            err = 0.5 * (err + np.abs(chroma - tabs[1]).mean())
        if err < best_err:
            best_q, best_err = q, err
    return best_q, float(best_err), bool(best_err == 0), len(qt)


# --------------------------------------------------------------------------- hashes
def dhash_hex(img: Image.Image, size: int = 8) -> str:
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):0{size * size // 4}x}"


def rgb_sha1(img: Image.Image) -> str:
    arr = np.asarray(img.convert("RGB"))
    h = hashlib.sha1(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- per row
def inspect_bytes(b: bytes, ref: dict) -> dict:
    rec: dict = {"n_bytes": len(b) if b is not None else 0}
    if not b:
        rec.update(decode_ok=False, error="empty bytes")
        return rec
    try:
        im = Image.open(io.BytesIO(b))
        rec["format"] = im.format
        rec["width"], rec["height"] = im.size
        rec["mode"] = im.mode
        rec["info_keys"] = ";".join(sorted(map(str, im.info.keys())))
        try:
            rec["exif_orientation"] = im.getexif().get(0x0112)
        except Exception:
            rec["exif_orientation"] = None
        if im.format == "JPEG":
            q, err, exact, n_tab = estimate_jpeg_quality(getattr(im, "quantization", {}), ref)
            rec.update(jpeg_q=q, jpeg_q_err=err, jpeg_q_exact=exact, jpeg_n_qtables=n_tab,
                       jpeg_subsampling=SUBSAMPLING.get(JpegImagePlugin.get_sampling(im), "other"),
                       jpeg_layers=getattr(im, "layers", None),
                       jpeg_progressive=bool(im.info.get("progressive") or im.info.get("progression")))
        im.load()  # full decode; catches truncated files
        rec["rgb_sha1"] = rgb_sha1(im)
        rec["dhash"] = dhash_hex(im)
        rec["decode_ok"] = True
    except Exception as e:  # noqa: BLE001
        rec.update(decode_ok=False, error=repr(e)[:300])
    return rec


def class_names_from_parquet(path: Path) -> dict[str, list[str]]:
    md = pq.read_schema(path).metadata or {}
    raw = md.get(b"huggingface")
    if not raw:
        return {}
    feats = json.loads(raw).get("info", {}).get("features", {})
    return {c: s["names"] for c, s in feats.items() if isinstance(s, dict) and s.get("_type") == "ClassLabel"}


def class_names_from_readme(data_dir: Path) -> dict[str, list[str]]:
    readme = data_dir / "README.md"
    if not readme.exists():
        return {}
    try:
        import yaml
        text = readme.read_text(encoding="utf-8")
        front = yaml.safe_load(text.split("---")[1])
        out = {}
        for f in front["dataset_info"]["features"]:
            cl = f.get("dtype", {}).get("class_label") if isinstance(f.get("dtype"), dict) else None
            if cl:
                names = cl["names"]
                out[f["name"]] = [names[k] for k in sorted(names, key=lambda x: int(x))]
        return out
    except Exception:  # noqa: BLE001
        return {}


def split_of(path: Path) -> str:
    n = path.name.lower()
    for s in ("train", "validation", "test"):
        if n.startswith(s):
            return s
    return "unknown"


def audit_file(args: tuple[str, int | None]) -> list[dict]:
    path, max_rows = Path(args[0]), args[1]
    ref = reference_qtables()
    pf = pq.ParquetFile(path)
    out, i = [], 0
    for batch in pf.iter_batches(batch_size=64):
        for row in batch.to_pylist():
            if max_rows is not None and i >= max_rows:
                return out
            img = row.get("image") or {}
            b = img.get("bytes") if isinstance(img, dict) else img
            rec = {"split": split_of(path), "file": path.name, "row_in_file": i,
                   "row_id": f"{split_of(path)}/{path.stem}/{i}",
                   "label_raw": row.get("label"), "generator_raw": row.get("generator"),
                   "orig_path": img.get("path") if isinstance(img, dict) else None}
            rec.update(inspect_bytes(b, ref))
            out.append(rec)
            i += 1
    return out


# --------------------------------------------------------------------------- summary helpers
def nested_counts(df: pd.DataFrame, rows: str, cols: str) -> dict:
    t = pd.crosstab(df[rows], df[cols].fillna("NA"))
    return {str(r): {str(c): int(v) for c, v in t.loc[r].items() if v} for r in t.index}


def top_values(df: pd.DataFrame, by: str, col: str, k: int = 5) -> dict:
    return {str(g): {str(v): int(n) for v, n in s[col].fillna("NA").astype(str).value_counts().head(k).items()}
            for g, s in df.groupby(by)}


def dup_report(df: pd.DataFrame, key: str) -> dict:
    d = df[df[key].notna()]
    grp = d.groupby(key)
    sizes = grp.size()
    multi = sizes[sizes > 1].index
    sub = d[d[key].isin(multi)]
    cross_split = sub.groupby(key)["split"].nunique().gt(1).sum()
    cross_label = sub.groupby(key)["label"].nunique().gt(1).sum()
    return {"groups_with_duplicates": int(len(multi)), "rows_in_such_groups": int(len(sub)),
            "groups_crossing_split": int(cross_split), "groups_crossing_label": int(cross_label)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path, help="local snapshot (contains data/*.parquet)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--revision", default="UNPINNED", help="dataset commit sha, recorded in the report")
    ap.add_argument("--max-rows-per-file", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--real-name", default="Real")
    ap.add_argument("--seen", default="SD14")
    ap.add_argument("--unseen", nargs="+", default=["ADM", "GLIDE", "Midjourney"])
    ap.add_argument("--real-quality", type=int, default=96)
    ap.add_argument("--min-side", type=int, default=224)
    a = ap.parse_args()

    files = sorted(a.data_dir.rglob("*.parquet"))
    if not files:
        sys.exit(f"no parquet files under {a.data_dir}")
    a.out.mkdir(parents=True, exist_ok=True)

    names = class_names_from_parquet(files[0]) or class_names_from_readme(a.data_dir)
    gen_names, lab_names = names.get("generator"), names.get("label")
    if not gen_names:
        print("WARNING: generator class names not found; raw integers kept. Check README.md.", file=sys.stderr)

    jobs = [(str(f), a.max_rows_per_file) for f in files]
    records: list[dict] = []
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as ex:
            for f, recs in zip(files, ex.map(audit_file, jobs)):
                print(f"  {f.name}: {len(recs)} rows", flush=True)
                records += recs
    else:
        for f, job in zip(files, jobs):
            recs = audit_file(job)
            print(f"  {f.name}: {len(recs)} rows", flush=True)
            records += recs

    df = pd.DataFrame(records)
    map_gen = (lambda v: gen_names[v] if gen_names and isinstance(v, (int, np.integer)) else v)
    map_lab = (lambda v: lab_names[v] if lab_names and isinstance(v, (int, np.integer)) else v)
    df["generator"] = df["generator_raw"].map(map_gen).astype(str)
    df["label"] = df["label_raw"]
    df["label_name"] = df["label_raw"].map(map_lab)
    df["is_real"] = df["label_raw"] == 0
    df["min_side"] = df[["width", "height"]].min(axis=1)
    df["size"] = df["width"].astype("Int64").astype(str) + "x" + df["height"].astype("Int64").astype(str)
    df["bytes_per_pixel"] = df["n_bytes"] / (df["width"] * df["height"])

    # cross-split exact duplicates are excluded from every partition
    cross = df.groupby("rgb_sha1")["split"].transform("nunique") > 1
    df["cross_split_duplicate"] = cross & df["rgb_sha1"].notna()

    for c in ("jpeg_q", "jpeg_q_exact", "jpeg_subsampling", "jpeg_layers"):
        if c not in df:
            df[c] = np.nan

    real_ok = (df["is_real"] & (df["format"] == "JPEG") & (df["jpeg_q_exact"] == True)  # noqa: E712
               & (df["jpeg_q"] == a.real_quality) & (df["min_side"] >= a.min_side))
    real_strict = real_ok & (df["mode"] == "RGB") & (df["jpeg_layers"] == 3) & df["exif_orientation"].isin([None, 1, np.nan])
    selected = [a.seen] + a.unseen
    fake_ok = (~df["is_real"]) & df["generator"].isin(selected) & (df["min_side"] >= a.min_side)
    base = df["decode_ok"] & ~df["cross_split_duplicate"]
    df["eligible_v06"] = base & (real_ok | fake_ok)
    df["eligible_strict"] = base & (real_strict | (fake_ok & (df["format"] == "PNG")))

    df.drop(columns=["generator_raw", "label_raw"]).to_csv(a.out / "audit_rows.csv", index=False)

    real = df[df["is_real"]]
    fake_sel = df[(~df["is_real"]) & df["generator"].isin(selected)]
    elig = df[df["eligible_v06"]]
    elig_s = df[df["eligible_strict"]]

    def n(d, split, gen=None, real_only=False):
        m = d["split"] == split
        m &= d["is_real"] if real_only else (d["generator"] == gen)
        return int(m.sum())

    def counts_vs_targets(d):
        return {
            "train_real (R_train needs 2000)": n(d, "train", real_only=True),
            f"train_{a.seen} (F_seen_train up to 2000)": n(d, "train", a.seen),
            "validation_real (R_cal + R_test need 1000)": n(d, "validation", real_only=True),
            **{f"validation_{g} (needs 300)": n(d, "validation", g) for g in selected},
        }

    synth_modal = {g: (fake_sel[fake_sel.generator == g]["size"].mode().iloc[0]
                       if (fake_sel.generator == g).any() else None) for g in selected}
    signals = {
        "labels_consistent_with_generator": bool(((df["generator"] == a.real_name) == df["is_real"]).all()),
        "real_share_jpeg": round(float((real["format"] == "JPEG").mean()), 4) if len(real) else None,
        f"real_share_exact_ijg_q{a.real_quality}": round(float(((real["jpeg_q"] == a.real_quality) & (real["jpeg_q_exact"] == True)).mean()), 4) if len(real) else None,  # noqa: E712
        "real_distinct_sizes": int(real["size"].nunique()),
        "synthetic_share_png_by_generator": {g: round(float((fake_sel[fake_sel.generator == g]["format"] == "PNG").mean()), 4)
                                             for g in selected if (fake_sel.generator == g).any()},
        "synthetic_modal_size_vs_expected": {g: {"observed": synth_modal[g], "expected_original_genimage": EXPECTED_SYNTH_SIZE.get(g)}
                                             for g in selected},
        "uniform_resize_suspect": bool(df["size"].nunique() <= 2),
        "uniform_reencode_suspect": bool(df["format"].nunique() == 1 and df["jpeg_q"].nunique() <= 1),
        "decode_failures": int((~df["decode_ok"]).sum()),
        "cross_split_exact_duplicate_rows": int(df["cross_split_duplicate"].sum()),
        "generators_seen_in_this_audit": sorted(df["generator"].unique().tolist()),
        "partial_audit": a.max_rows_per_file is not None,
    }

    summary = {
        "dataset_revision": a.revision,
        "pillow_version": PIL.__version__,
        "files": [f.name for f in files],
        "n_rows": int(len(df)),
        "class_names": names,
        "counts_split_generator": nested_counts(df, "generator", "split"),
        "format_by_generator": nested_counts(df, "generator", "format"),
        "mode_by_generator": nested_counts(df, "generator", "mode"),
        "jpeg_quality_top5_by_generator": top_values(df[df["format"] == "JPEG"], "generator", "jpeg_q"),
        "jpeg_exact_ijg_share_by_generator": {str(g): round(float((s["jpeg_q_exact"] == True).mean()), 4)  # noqa: E712
                                              for g, s in df[df["format"] == "JPEG"].groupby("generator")},
        "jpeg_subsampling_by_generator": nested_counts(df[df["format"] == "JPEG"], "generator", "jpeg_subsampling")
        if (df["format"] == "JPEG").any() else {},
        "size_top5_by_generator": top_values(df, "generator", "size"),
        "min_side_by_generator": {str(g): {"min": int(s.min()), "median": float(s.median()), "max": int(s.max())}
                                  for g, s in df.groupby("generator")["min_side"]},
        "bytes_per_pixel_median_by_generator": {str(g): round(float(s.median()), 4)
                                                for g, s in df.groupby("generator")["bytes_per_pixel"]},
        "nontrivial_exif_orientation_by_generator": {str(g): int(s.isin([2, 3, 4, 5, 6, 7, 8]).sum())
                                                     for g, s in df.groupby("generator")["exif_orientation"]},
        "orig_path_examples_by_generator": {str(g): s["orig_path"].dropna().head(3).tolist() for g, s in df.groupby("generator")},
        "info_keys_top3_by_generator": top_values(df, "generator", "info_keys", 3),
        "exact_duplicates_rgb": dup_report(df, "rgb_sha1"),
        "identical_dhash": dup_report(df, "dhash"),
        "eligible_v06_vs_targets": counts_vs_targets(elig),
        "eligible_strict_vs_targets": counts_vs_targets(elig_s),
        "strict_rule_note": "strict adds: real mode RGB with 3 JPEG components and no EXIF rotation; synthetic stored as PNG",
        "g0_signals": signals,
    }
    (a.out / "data_audit.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n=== G0 signals ===")
    for k, v in signals.items():
        print(f"  {k}: {v}")
    print("\n=== eligible (v0.6 rule) vs targets ===")
    for k, v in summary["eligible_v06_vs_targets"].items():
        print(f"  {k}: {v}")
    print(f"\nwrote {a.out / 'audit_rows.csv'} and {a.out / 'data_audit.json'}")


if __name__ == "__main__":
    main()

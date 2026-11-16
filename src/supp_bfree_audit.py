#!/usr/bin/env python3
"""S-BF Step 1: byte, encoding-history and correspondence audit of B-Free extended Synthbuster
(PROTOCOL.md Section 7). No DINOv2 feature, score or detection metric is computed here.

Per file: sha256 and size, PNG header (size, bit depth, colour type, interlace), ancillary chunks,
ICC profile description, decoded mode and shape, exact RGB hash and dHash (the core E0 functions),
an 8x8 JPEG quantization-lattice statistic at all 64 grid offsets, and for ICC-tagged real images
the pixel change a colour-managed conversion to standard sRGB would cause.

JPEG lattice statistic. Luma Y (JFIF weights) of a central 256 x 256 region is cut into 8 x 8 blocks
at each grid offset (dy, dx) in 0..7; orthonormal 2-D DCT (= the JPEG FDCT). For each AC frequency
and step q in 2..10, E = |mean exp(2 pi i c / q)| over coefficients with |c| >= q / 2 (frequencies
with fewer than 50 such coefficients count 0); L(dy, dx) = mean over AC frequencies of max over q.
A JPEG encoded at this resolution leaves coefficients near multiples of its quantization steps on
the aligned grid (0, 0) only; content-driven structure is shared by all offsets and cancels. An image
is flagged if `L(0, 0) - median L > 0.1`; the max-over-offsets `grid_score` is kept as a descriptive
column. Positive controls: 50 images per class
re-encoded once with Pillow JPEG Q96 and Q90, 4:4:4 (`conditions.jpeg_roundtrip`).

Outputs: artifacts/supp_bfree/audit/{files.csv, positive_controls.csv, audit.json} and DATA_CARD_SUPP_BFREE.md.
Usage: python src/supp_bfree_audit.py --workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image, ImageCms  # noqa: F401 (ImageCms.core for the LittleCMS version)
from scipy.fft import dctn

import conditions as C
import supp_bfree_data as S
from audit_dataset import dhash_hex, rgb_sha1
from config import ROOT

OUT = ROOT / "artifacts" / "supp_bfree" / "audit"
CARD = ROOT / "DATA_CARD_SUPP_BFREE.md"
PNG_SIG = b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------- PNG container
def png_chunks(b: bytes) -> list[tuple[str, bytes]]:
    if b[:8] != PNG_SIG:
        raise ValueError("not a PNG file")
    i, out = 8, []
    while i < len(b):
        n = struct.unpack(">I", b[i:i + 4])[0]
        out.append((b[i + 4:i + 8].decode("latin1"), b[i + 8:i + 8 + n]))
        i += 12 + n
    return out


icc_description = S.icc_description


def icc_to_srgb_change(arr: np.ndarray, icc: bytes) -> tuple[int, float, float, bool]:
    """(max, mean, share > 1, identical) of |S-BF sRGB conversion - stored values|."""
    d = np.abs(S.to_srgb(arr, icc).astype(np.int16) - arr.astype(np.int16))
    return int(d.max()), float(d.mean()), float((d > 1).mean()), bool(d.max() == 0)


# --------------------------------------------------------------------------- JPEG lattice
def lattice(arr: np.ndarray, lc: dict) -> np.ndarray:
    """L(dy, dx) for the 64 grid offsets, shape (8, 8)."""
    region, (q0, q1), nmin = int(lc["region"]), lc["q_range"], int(lc["min_coefficients"])
    y = arr.astype(np.float64) @ np.asarray(lc["luma"], np.float64) - 128.0
    h, w = y.shape
    t0, l0 = (h - region - 8) // 2 // 8 * 8, (w - region - 8) // 2 // 8 * 8
    nb = region // 8
    blocks = np.stack([y[t0 + dy:t0 + dy + region, l0 + dx:l0 + dx + region]
                       .reshape(nb, 8, nb, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
                       for dy in range(8) for dx in range(8)])
    c = dctn(blocks, axes=(2, 3), norm="ortho").reshape(64, nb * nb, 64)[:, :, 1:]
    best = np.zeros((64, 63))
    for q in range(int(q0), int(q1) + 1):
        m = np.abs(c) >= q / 2
        n = m.sum(axis=1)
        e = np.abs((np.exp(2j * np.pi * c / q) * m).sum(axis=1)) / np.maximum(n, 1)
        best = np.maximum(best, np.where(n >= nmin, e, 0.0))
    return best.mean(axis=1).reshape(8, 8)


def lattice_summary(arr: np.ndarray, lc: dict) -> dict:
    L = lattice(arr, lc)
    flat = L.ravel()
    k = int(flat.argmax())
    return {"lattice_aligned": float(L[0, 0]), "lattice_median": float(np.median(flat)), "lattice_max": float(flat[k]),
            "lattice_argmax": f"{k // 8},{k % 8}", "grid_score": float(flat[k] - np.median(flat))}


# --------------------------------------------------------------------------- per file
def audit_one(args: tuple[str, str]) -> dict:
    cls, sid = args
    lc = S.cfg()["jpeg_lattice"]
    p = S.path(cls, sid)
    b = p.read_bytes()
    ch = png_chunks(b)
    w, h, depth, ctype, _, _, interlace = struct.unpack(">IIBBBBB", ch[0][1][:13])
    rec = {"row_id": S.row_id(cls, sid), "class": cls, "source_id": sid, "sha256": hashlib.sha256(b).hexdigest(),
           "bytes": len(b), "width": w, "height": h, "bit_depth": depth, "png_color_type": ctype,
           "interlace": interlace, "chunks": "+".join(sorted({t for t, _ in ch})),
           "text_chunks": sum(t in ("tEXt", "iTXt", "zTXt") for t, _ in ch), "has_exif": any(t == "eXIf" for t, _ in ch)}
    im = Image.open(io.BytesIO(b))
    icc = im.info.get("icc_profile")
    rec["icc_description"] = icc_description(icc)
    rec["pil_mode"] = im.mode
    try:
        arr = S.decode(cls, sid)
        rec["decode_ok"], rec["decode_error"] = True, ""
    except Exception as e:  # noqa: BLE001 (recorded, makes the id ineligible)
        rec["decode_ok"], rec["decode_error"] = False, repr(e)
        return rec
    rec["decoded_h"], rec["decoded_w"] = arr.shape[:2]
    pil = Image.fromarray(arr, "RGB")
    rec["rgb_sha1"], rec["dhash"] = rgb_sha1(pil), dhash_hex(pil)
    rec["bytes_per_pixel"] = len(b) / (w * h)
    rec.update(lattice_summary(arr, lc))
    if icc:
        (rec["icc_srgb_max_change"], rec["icc_srgb_mean_change"], rec["icc_srgb_share_gt1"],
         rec["icc_srgb_identical"]) = icc_to_srgb_change(arr, icc)
    return rec


def control_one(args: tuple[str, str, int]) -> dict:
    cls, sid, q = args
    arr = C.jpeg_roundtrip(S.decode(cls, sid), q, "4:4:4")
    return {"row_id": S.row_id(cls, sid), "class": cls, "quality": q, **lattice_summary(arr, S.cfg()["jpeg_lattice"])}


# --------------------------------------------------------------------------- summary
def _b(s: pd.Series) -> pd.Series:
    """Boolean column from in-memory bools or CSV strings ("True"/"False"/"")."""
    return s.astype(str).str.lower().eq("true")


def read_saved() -> tuple[pd.DataFrame, pd.DataFrame]:
    str_cols = {k: str for k in ("row_id", "class", "source_id", "sha256", "chunks", "icc_description", "pil_mode",
                                  "decode_error", "rgb_sha1", "dhash", "lattice_argmax")}
    files = pd.read_csv(OUT / "files.csv", dtype=str_cols)
    for k in str_cols:
        files[k] = files[k].fillna("")
    controls = pd.read_csv(OUT / "positive_controls.csv", dtype={"lattice_argmax": str})
    drop = [k for k in ("file_ok", "icc_ok", "srgb_tagged", "jpeg_flag", "aligned_minus_median") if k in files]
    return files.drop(columns=drop), controls


def q(s: pd.Series) -> dict:
    s = s.dropna()
    return {"n": int(len(s)), "min": float(s.min()), "p50": float(s.median()), "p99": float(s.quantile(0.99)),
            "max": float(s.max())} if len(s) else {"n": 0}


def summarize(files: pd.DataFrame, controls: pd.DataFrame, ids: dict, other: dict) -> dict:
    c = S.cfg()
    el = c["eligibility"]
    out = {"dataset": c["data"]["name"], "root": c["data"]["root"], "dirs": c["data"]["dirs"],
           "ids_per_class": {k: len(v) for k, v in ids.items()},
           "ids_identical_across_classes": all(ids[k] == ids["REAL"] for k in S.CLASSES),
           "non_id_entries": other, "per_class": {}}
    for cls, g in files.groupby("class"):
        sizes = g["width"].astype(str) + "x" + g["height"].astype(str)
        out["per_class"][cls] = {
            "n": int(len(g)), "decode_ok": int(_b(g["decode_ok"]).sum()), "pil_mode": g["pil_mode"].value_counts().to_dict(),
            "bit_depth": g["bit_depth"].value_counts().to_dict(), "png_color_type": g["png_color_type"].value_counts().to_dict(),
            "interlaced": int((g["interlace"] != 0).sum()), "sizes": sizes.value_counts().to_dict(),
            "pixels": q(g["width"] * g["height"]), "min_side": int(np.minimum(g["width"], g["height"]).min()),
            "chunks": g["chunks"].value_counts().to_dict(), "text_chunks": int(g["text_chunks"].sum()),
            "exif": int(_b(g["has_exif"]).sum()), "icc": g["icc_description"].replace("", "none").value_counts().to_dict(),
            "bytes_per_pixel": q(g["bytes_per_pixel"]), "grid_score": q(g["grid_score"]),
            "lattice_aligned_minus_median": q(g["lattice_aligned"] - g["lattice_median"]),
        }
    real = files[files["class"] == "REAL"]
    for desc, g in real.groupby("icc_description"):
        if desc:
            out["per_class"]["REAL"][f"icc_to_srgb::{desc}"] = {
                "n": int(len(g)), "identical": int(_b(g["icc_srgb_identical"]).sum()),
                "max_change": q(g["icc_srgb_max_change"]), "mean_change": q(g["icc_srgb_mean_change"]),
                "share_gt1": q(g["icc_srgb_share_gt1"])}
    out["littlecms_version"] = getattr(ImageCms.core, "littlecms_version", "unknown")
    lc = c["jpeg_lattice"]
    thr, gt = float(lc["flag"]["threshold"]), lc["gate"]
    files = files.assign(aligned_minus_median=files["lattice_aligned"] - files["lattice_median"])
    controls = controls.assign(aligned_minus_median=controls["lattice_aligned"] - controls["lattice_median"])
    files["jpeg_flag"] = files["aligned_minus_median"] > thr
    det = float((controls["aligned_minus_median"] > thr).mean())
    argmax_share = {k: float((g["lattice_argmax"] == "0,0").mean()) for k, g in files.groupby("class")}
    out["jpeg_lattice"] = {
        "flag_rule": f"lattice_aligned - lattice_median > {thr}",
        "stored": {k: {"aligned_minus_median": q(g["aligned_minus_median"]), "flagged": int(g["jpeg_flag"].sum()),
                       "aligned_is_argmax_share": argmax_share[k], "grid_score_max_over_offsets": q(g["grid_score"])}
                   for k, g in files.groupby("class")},
        "controls": {f"Q{k}": {"aligned_minus_median": q(g["aligned_minus_median"]),
                               "detected": float((g["aligned_minus_median"] > thr).mean())}
                     for k, g in controls.groupby("quality")},
        "control_detection_rate": det, "chance_argmax_share": 1 / 64,
        "aligned_is_argmax_share_controls": float((controls["lattice_argmax"] == "0,0").mean()),
    }
    jpeg_ok = (int(files["jpeg_flag"].sum()) <= int(gt["max_flagged_stored"]) and det >= float(gt["min_control_detection"])
               and max(argmax_share.values()) <= float(gt["max_aligned_argmax_share_stored"]))
    out["jpeg_lattice"]["no_jpeg_history"] = bool(jpeg_ok)
    # eligibility per source id (all three files must pass)
    ok = _b(files["decode_ok"]) & (files["bit_depth"] == el["bit_depth"]) & (files["png_color_type"] == el["png_color_type"]) \
        & (np.minimum(files["width"], files["height"]) >= el["min_side"]) & (files["pil_mode"] == "RGB") \
        & ~files["jpeg_flag"]
    allowed = {("" if a is None else a) for a in el["real_icc_allowed"]}
    icc_ok = np.where(files["class"] == "REAL", files["icc_description"].isin(allowed), files["icc_description"] == "")
    srgb = c["colour"]["sensitivity"]["real_icc_description"]
    files = files.assign(file_ok=ok, icc_ok=icc_ok, srgb_tagged=(files["class"] == "REAL") & (files["icc_description"] == srgb))
    per_id = files.groupby("source_id").agg(file_ok=("file_ok", "all"), icc_ok=("icc_ok", "all"), n=("class", "size"),
                                            srgb_tagged=("srgb_tagged", "any"))
    per_id["eligible"] = per_id["file_ok"] & per_id["icc_ok"] & (per_id["n"] == 3)
    out["eligibility"] = {"ids_total": int(len(per_id)), "ids_failing_file_rules": int((~per_id["file_ok"]).sum()),
                          "ids_failing_icc_rule": int((per_id["file_ok"] & ~per_id["icc_ok"]).sum()),
                          "ids_eligible": int(per_id["eligible"].sum()),
                          "ids_eligible_srgb_tagged": int((per_id["eligible"] & per_id["srgb_tagged"]).sum())}
    srgb_rows = files[files["srgb_tagged"]]
    out["sensitivity_conversion_identity"] = bool(_b(srgb_rows["icc_srgb_identical"]).all())
    # file-level separability (what a metadata classifier could use; the visual pipeline never sees it)
    sz = files["width"].astype(str) + "x" + files["height"].astype(str)
    real_sizes, syn_sizes = set(sz[files["class"] == "REAL"]), set(sz[files["class"] != "REAL"])
    out["metadata_separability"] = {"size_sets_disjoint": not (real_sizes & syn_sizes),
                                    "iccp_only_in_real": bool(files.loc[files["class"] != "REAL", "icc_description"].eq("").all())}
    n_needed = 600  # at least 120 calibration and 120 test ids at the pre-set proportions
    out["gate"] = {
        "ids_identical": out["ids_identical_across_classes"],
        "all_png_8bit_rgb_decodable": bool(_b(files["decode_ok"]).all() and (files["bit_depth"] == 8).all()
                                           and (files["png_color_type"] == 2).all()),
        "no_jpeg_lattice_in_any_class": out["jpeg_lattice"]["no_jpeg_history"],
        "enough_eligible_ids": out["eligibility"]["ids_eligible"] >= n_needed,
        "min_eligible_ids_required": n_needed,
        "srgb_conversion_identity_for_sensitivity_ids": out["sensitivity_conversion_identity"],
    }
    out["gate"]["passed"] = all(v for k, v in out["gate"].items() if k != "min_eligible_ids_required")
    return out, files


def write_card(s: dict) -> None:
    pc = s["per_class"]
    lines = ["# DATA CARD: B-Free extended Synthbuster (supplementary experiment S-BF)", "",
             "Generated by `src/supp_bfree_audit.py` from `artifacts/supp_bfree/audit/audit.json`; do not edit by hand.",
             "Protocol: `PROTOCOL.md` Section 7. No DINOv2 feature or detection score was used for this card.", "",
             "## Source", "",
             f"- Local copy `{s['root']}`; folders {', '.join(f'`{k}`: `{v}`' for k, v in s['dirs'].items())}.",
             "- Real: 1,000 RAISE-1k photographs (Dang-Nguyen et al., MMSys 2015; non-commercial license, `RAISE_License.pdf`).",
             "- Synthetic: one image per real photograph and generator, generated from a caption of that photograph "
             "(Synthbuster construction; extended by B-Free, Guillaro et al., CVPR 2025, with FLUX and Stable Diffusion "
             "3.5). The SD3.5 folder is named `sd3_large`; the files carry no generator metadata, so the model variant is "
             "as described by the dataset provider.",
             "- No revision id: every file's sha256 is in `artifacts/supp_bfree/audit/files.csv`.", "",
             "## Correspondence", "",
             f"- Source ids per class: {s['ids_per_class']}; identical id sets: **{s['ids_identical_across_classes']}**.",
             f"- Entries ignored (not matching the id pattern): {s['non_id_entries']}.",
             "- Correspondence is semantic (same scene description), not pixel alignment: compositions, viewpoints and "
             "styles differ; the real images are 3:2 photographs, the synthetic ones 1:1. A visual spot check of random "
             "triplets confirmed the same subject per id.", "",
             "## Stored files", "", "| Property | REAL | SD35 | FLUX |", "| --- | --- | --- | --- |"]

    def row(name, f):
        lines.append(f"| {name} | " + " | ".join(str(f(pc[c])) for c in S.CLASSES) + " |")

    row("files", lambda p: p["n"])
    row("decodable", lambda p: p["decode_ok"])
    row("bit depth / colour type", lambda p: f"{p['bit_depth']} / {p['png_color_type']}")
    row("sizes (w x h: n)", lambda p: "; ".join(f"{k}: {v}" for k, v in p["sizes"].items()))
    row("pixels (median)", lambda p: f"{p['pixels']['p50']:,.0f}")
    row("ICC profile", lambda p: "; ".join(f"{k}: {v}" for k, v in p["icc"].items()))
    row("text / EXIF chunks", lambda p: f"{p['text_chunks']} / {p['exif']}")
    row("bytes per pixel (median)", lambda p: f"{p['bytes_per_pixel']['p50']:.3f}")
    j = s["jpeg_lattice"]
    row("JPEG lattice, aligned minus median (median / max)",
        lambda p: "")  # placeholder, replaced below
    lines[-1] = "| JPEG lattice, aligned minus median (median / max) | " + " | ".join(
        f"{j['stored'][c]['aligned_minus_median']['p50']:.4f} / {j['stored'][c]['aligned_minus_median']['max']:.4f}"
        for c in S.CLASSES) + " |"
    lines.append("| best grid offset is (0, 0) | " + " | ".join(
        f"{100 * j['stored'][c]['aligned_is_argmax_share']:.1f}%" for c in S.CLASSES) + " |")
    lines += ["", "## Encoding history", "",
              f"- JPEG test: an image is flagged if its lattice on the aligned grid exceeds the median over the 64 grid "
              f"offsets by more than {j['flag_rule'].split('> ')[1]}. Stored images flagged: " + ", ".join(
                  f"{c} {j['stored'][c]['flagged']}" for c in S.CLASSES) + ". Positive controls (the same images "
              f"re-encoded once, 4:4:4) detected: " + "; ".join(
                  f"{k} {100 * v['detected']:.1f}%" for k, v in j["controls"].items()) +
              f". The best of the 64 offsets is the aligned grid for {100 * j['aligned_is_argmax_share_controls']:.1f}% "
              "of controls but for " + ", ".join(f"{100 * j['stored'][c]['aligned_is_argmax_share']:.1f}% ({c})"
                                                 for c in S.CLASSES) +
              f" of stored images (chance {100 * j['chance_argmax_share']:.1f}%). "
              f"No JPEG history at the stored resolution: **{j['no_jpeg_history']}**. The test misses a few controls "
              "whose content gives a high lattice level at every offset, so it bounds rather than excludes a JPEG "
              "history in single images; it cannot see a JPEG applied before the real images were resized.",
              "- Real images: RAISE photographs (developed from camera RAW) downscaled to about 1024^2 pixels "
              "(1258 x 833 or 1256 x 835, i.e. the camera aspect ratio at the pixel count of 1024 x 1024), then stored "
              "as PNG. They carry a resampling history that the synthetic images (native 1024 x 1024 generator output) do not.",
              "- All files are lossless 8-bit RGB PNG; the JPEG-versus-PNG asymmetry of the core study is absent."]
    lines += ["", "## Colour space", "",
              "- Rule (fixed before feature extraction): every ICC-tagged real image is converted to standard "
              "sRGB with ImageCms (relative colorimetric, LittleCMS " + str(s.get("littlecms_version")) + "); the untagged real "
              "image is assumed sRGB and left unchanged; synthetic images have no ICC chunk and are unchanged. All 1,000 "
              "source ids are kept (primary analysis); the sRGB-tagged ids form the sensitivity analysis."]
    for k, v in pc["REAL"].items():
        if k.startswith("icc_to_srgb::"):
            lines.append(f"- `{k.split('::')[1]}` ({v['n']} images): conversion leaves {v['identical']} images bitwise "
                         f"unchanged; largest change {v['max_change']['max']:.0f} levels, median per-image mean change "
                         f"{v['mean_change']['p50']:.3f}.")
    lines += ["- Without conversion, Adobe RGB values read as sRGB are desaturated, a real-only pixel difference. The "
              "conversion removes it but is itself a processing step applied to real images only (identity for the "
              f"sRGB-tagged ones: {s['sensitivity_conversion_identity']}), hence the sensitivity analysis without converted images.",
              f"- File-level cues remain: real and synthetic size sets disjoint: {s['metadata_separability']['size_sets_disjoint']}; "
              f"ICC chunk only in real files: {s['metadata_separability']['iccp_only_in_real']}. The visual methods see only "
              "decoded 224 x 224 pixels, never the file size or chunks; any metadata classifier would separate the classes perfectly."]
    e = s["eligibility"]
    g = s["gate"]
    lines += ["", "## Eligibility and gate", "",
              f"- Source ids: {e['ids_total']}; failing file rules: {e['ids_failing_file_rules']}; failing the ICC rule: "
              f"{e['ids_failing_icc_rule']}; eligible: **{e['ids_eligible']}**, of which sRGB-tagged (sensitivity): "
              f"**{e['ids_eligible_srgb_tagged']}**.",
              f"- Gate G0-BF: ids identical {g['ids_identical']}; all PNG 8-bit RGB decodable {g['all_png_8bit_rgb_decodable']}; "
              f"no JPEG lattice {g['no_jpeg_lattice_in_any_class']}; at least {g['min_eligible_ids_required']} eligible ids "
              f"{g['enough_eligible_ids']}; sRGB conversion is the identity on the sensitivity ids "
              f"{g['srgb_conversion_identity_for_sensitivity_ids']}. **Passed: {g['passed']}.**", "",
              "## Remaining differences (limitations)", "",
              "- Resampling history (real downscaled, synthetic native), aspect ratio (3:2 versus 1:1), and the semantic, "
              "not pixel-level, content match (synthetic images are more stylized). The native-crop condition removes "
              "resizing by the pipeline but not the real images' earlier downscaling.", ""]
    CARD.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--summarize-only", action="store_true", help="re-summarize the saved per-file statistics")
    a = ap.parse_args()
    ids, other = {}, {}
    for cls in S.CLASSES:
        ids[cls], other[cls] = S.listing(cls)
    if a.summarize_only:
        files, controls = read_saved()
        if sorted(files["row_id"]) != sorted(S.row_id(c, i) for c in S.CLASSES for i in ids[c]):
            raise RuntimeError("saved per-file statistics do not match the folders")
    else:
        jobs = [(cls, sid) for cls in S.CLASSES for sid in ids[cls]]
        with ProcessPoolExecutor(a.workers) as ex:
            files = pd.DataFrame(list(ex.map(audit_one, jobs, chunksize=8)))
            pc = S.cfg()["jpeg_lattice"]["positive_control"]
            common = sorted(set(ids["REAL"]) & set(ids["SD35"]) & set(ids["FLUX"]))
            pick = [common[i] for i in np.random.default_rng(42).permutation(len(common))[:int(pc["per_class"])]]
            cjobs = [(cls, sid, int(qq)) for cls in S.CLASSES for sid in pick for qq in pc["quality"]]
            controls = pd.DataFrame(list(ex.map(control_one, cjobs, chunksize=4)))
    summary, files = summarize(files, controls, ids, other)
    OUT.mkdir(parents=True, exist_ok=True)
    files.sort_values("row_id").to_csv(OUT / "files.csv", index=False, lineterminator="\n")
    controls.sort_values(["row_id", "quality"]).to_csv(OUT / "positive_controls.csv", index=False, lineterminator="\n")
    (OUT / "audit.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    write_card(summary)
    print(json.dumps({"gate": summary["gate"], "eligibility": summary["eligibility"], "jpeg": summary["jpeg_lattice"]}, indent=1))


if __name__ == "__main__":
    main()

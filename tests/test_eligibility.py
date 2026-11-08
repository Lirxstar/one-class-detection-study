"""Every partition row satisfies the v0.6.2 eligibility rules (PROTOCOL.md 2.3)."""
import pandas as pd


def test_partition_rows_are_eligible(manifest, parts):
    m = manifest.set_index("row_id")
    used = [r for ids in parts.values() for r in ids]
    assert m.loc[used, "eligible"].astype(str).str.lower().eq("true").all()


def test_real_rules(manifest, parts):
    m = manifest.set_index("row_id")
    real = m.loc[parts["R_train"] + parts["R_cal"] + parts["R_test"] + parts["P_cal"] + parts["P_test"]]
    assert (real["format"] == "JPEG").all() and (real["mode"] == "RGB").all()
    assert (real["jpeg_q"] == 96).all() and real["jpeg_q_exact"].astype(str).str.lower().eq("true").all()
    assert (real["jpeg_layers"] == 3).all() and (real["jpeg_subsampling"] == "4:4:4").all()
    assert (real["exif_orientation"].isna() | (real["exif_orientation"] == 1)).all()
    assert (real[["width", "height"]].min(axis=1) >= 224).all()


def test_synthetic_rules(manifest, parts):
    m = manifest.set_index("row_id")
    syn = m.loc[parts["F_seen_train"] + parts["F_seen_test"] + [r for v in ("F_unseen_test", "P_fake") for r in parts[v]]]
    assert (syn["format"] == "PNG").all()
    rgba = syn["mode"] == "RGBA"
    assert (syn["mode"].isin(["RGB", "RGBA"])).all()
    assert (pd.to_numeric(syn.loc[rgba, "alpha_min"]) == 255).all()
    assert set(syn.loc[rgba, "generator"]) <= {"ADM"}
    assert (syn[["width", "height"]].min(axis=1) >= 224).all()


def test_exclusion_reason_iff_not_eligible(manifest):
    elig = manifest["eligible"].astype(str).str.lower().eq("true")
    reason = manifest["exclusion_reason"].fillna("")
    assert ((reason == "") == elig).all()

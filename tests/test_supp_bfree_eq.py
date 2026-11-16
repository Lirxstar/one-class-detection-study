"""S-BF-EQ (PROTOCOL.md Section 8): subsampling, label rules, reuse of S-BF outputs, saved results.

Tests on saved artifacts skip when those artifacts do not exist (splits, feature cache, predictions).
"""
import json

import numpy as np
import pandas as pd
import pytest

import config
from config import ROOT

SB = ROOT / "artifacts" / "supp_bfree"
EQ = ROOT / "artifacts" / "supp_bfree_eq"


def _manifest():
    p = SB / "manifest.csv"
    if not p.exists():
        pytest.skip("S-BF splits not built")
    return pd.read_csv(p, keep_default_na=False, dtype=str).set_index("row_id")


def _subsamples():
    p = EQ / "subsamples.json"
    if not p.exists():
        pytest.skip("S-BF-EQ subsamples not written")
    return json.loads(p.read_text())


# --------------------------------------------------------------------------- design fixed in the config
def test_config_fixed():
    c = config.load("supp_bfree_eq")
    assert c["protocol_version"] == "S-BF-EQ 1.0" and c["base_protocol"] == "S-BF 1.0"
    assert c["train_subsample"]["seeds"] == [101, 102, 103, 104, 105]
    assert c["train_subsample"]["fraction"] == 0.5 and c["analysis"] == "all"
    assert config.CONFIG_FILES == ("conditions.yaml", "core.yaml", "e2_postprocessing.yaml", "e3_encoding.yaml")


def test_outputs_in_own_directories():
    import supp_bfree_eq_analysis as A
    import supp_bfree_eq_run as R
    for p in (R.OUT, R.ART, A.PRED, A.TABLES, A.FIG):
        assert "supp_bfree_eq" in p.parts, p


# --------------------------------------------------------------------------- subsampling
def test_allocate_largest_remainder():
    import supp_bfree_eq_run as R
    assert R.allocate({"other": 170, "srgb": 430}, 0.5) == {"other": 85, "srgb": 215}
    assert R.allocate({"a": 3, "b": 3}, 0.5) == {"a": 2, "b": 1}           # 3 of 6; tie broken by name
    assert R.allocate({"a": 5, "b": 2}, 0.5) == {"a": 3, "b": 1}           # floor(3.5 + 0.5) = 4
    assert R.allocate({"a": 7, "b": 9}, 1.0) == {"a": 7, "b": 9}


@pytest.mark.parametrize("role,n_pool,n_draw", [("final", 600, 300), ("pilot", 360, 180)])
def test_draws_from_training_pool_only(role, n_pool, n_draw):
    m = _manifest()
    import supp_bfree_eq_run as R
    p = R.pool(role)
    assert len(p) == n_pool
    draws = [R.draw(role, s) for s in R.seeds()]
    for ids in draws:
        assert len(ids) == len(set(ids)) == n_draw and set(ids) <= set(p.index)
        real = m.loc[[f"bfree/REAL/{s}" for s in ids]]
        assert (real["split_role"] == "train").all()
        if role == "pilot":
            assert set(real["pilot_role"]) == {"fit"}
        counts = real["stratum"].value_counts()
        full = p["stratum"].value_counts()
        assert all(abs(counts[h] - full[h] / 2) <= 0.5 for h in full.index)
    assert len({tuple(d) for d in draws}) == len(draws), "two seeds drew the same ids"


def test_subsamples_reproducible():
    sub = _subsamples()
    _manifest()
    import supp_bfree_eq_run as R
    assert sub["seeds"] == R.seeds()
    for role in ("final", "pilot"):
        for s in R.seeds():
            d = sub["roles"][role]["draws"][str(s)]
            ids = R.draw(role, s)
            assert d["source_ids"] == ids and d["sha256"] == R.ids_sha(ids)
    assert sub["roles"]["final"]["draws"]["101"]["per_stratum"] == {"other": 85, "srgb": 215}


# --------------------------------------------------------------------------- label rules
class _FakeFeats:
    def __init__(self, ids, dim=8, seed=0):
        rng = np.random.default_rng(seed)
        self.pos = {r: i for i, r in enumerate(ids)}
        self.h_pre = rng.normal(size=(len(ids), dim)).astype(np.float32)
        self.h_post = rng.normal(size=(len(ids), dim)).astype(np.float32)

    def take(self, ids):
        idx = [self.pos[r] for r in ids]
        return self.h_pre[idx], self.h_post[idx]


@pytest.mark.parametrize("role", ["final", "pilot"])
def test_b1_eq_receives_only_drawn_train_rows(role, monkeypatch):
    """B1-eq gets the REAL (label 0) and SD3.5 (label 1) rows of the drawn ids only: no FLUX, no calibration or
    test id; O1 and O2 are never fitted."""
    m = _manifest()
    import detectors as D
    import supp_bfree_eq_run as R
    seen = {}
    real_fit = D.B1.fit

    def spy(z_real, z_fake, g_real, g_fake, C=None):
        seen["n"] = (len(z_real), len(z_fake))
        seen["groups"] = (list(g_real), list(g_fake))
        return real_fit(z_real, z_fake, g_real, g_fake, C=C)

    def forbidden(*a, **k):
        raise AssertionError("O1/O2 must not be refitted in S-BF-EQ")

    monkeypatch.setattr(D.B1, "fit", staticmethod(spy))
    monkeypatch.setattr(D.O1, "fit", staticmethod(forbidden))
    monkeypatch.setattr(D.O2, "fit", staticmethod(forbidden))
    ids = R.draw(role, R.seeds()[0])
    r_rows, f_rows = R.training_rows(role, ids)
    assert (m.loc[r_rows, "class"] == "REAL").all() and (m.loc[f_rows, "class"] == "SD35").all()
    assert (m.loc[r_rows + f_rows, "split_role"] == "train").all()
    assert list(m.loc[r_rows, "source_id"]) == list(m.loc[f_rows, "source_id"]) == ids
    feats = _FakeFeats(sorted(m.index))
    df, diag = R.run_cell(feats, role, "primary", "native", R.seeds()[0], ids, "test", "test")
    assert seen["n"] == (len(ids), len(ids))
    assert seen["groups"][0] == list(m.loc[r_rows, "id_group"]) == seen["groups"][1]
    assert not set(df["source_id"]) & set(ids)
    cal = df[df["partition"] == R.R.ROLES[role]["cal"]]
    assert (cal["generator"] == "REAL").all() and len(cal)
    import calibrate
    assert calibrate.threshold(cal["score"].to_numpy()) == df["threshold"].iloc[0]
    if role == "pilot":
        assert "FLUX" not in set(df["generator"])
        assert set(df["source_split"]) == {"train"}


def test_drawn_ids_outside_pool_are_refused():
    m = _manifest()
    import supp_bfree_eq_run as R
    test_id = m[m["partition"] == "R_test"]["source_id"].iloc[0]
    with pytest.raises(AssertionError):
        R.training_rows("final", [test_id])
    final_only = sorted(set(R.pool("final").index) - set(R.pool("pilot").index))[0]
    with pytest.raises(AssertionError):
        R.training_rows("pilot", [final_only])


def test_code_path_reproduces_sbf_pilot_b1():
    """This code refitted on the full pilot pool gives the S-BF pilot B1 scores bitwise (train ids only)."""
    import supp_bfree_eq_run as R
    import supp_bfree_features as F
    if not F.feature_path("primary", "native").exists():
        pytest.skip("S-BF feature cache not available")
    if not (R.R.OUT / "pilot" / "pilot_primary_native.csv.gz").exists():
        pytest.skip("S-BF pilot records not available")
    chk = R.pilot_code_path_check(R.R.Feats("primary", "native"), "primary", "native")
    assert chk["bitwise_equal"] and chk["rows"] == 360


# --------------------------------------------------------------------------- saved predictions and tables
def _final():
    files = sorted((ROOT / "artifacts/predictions/supp_bfree_eq").glob("final_*.csv.gz"))
    if not files:
        pytest.skip("S-BF-EQ final predictions not available")
    return pd.concat([pd.read_csv(f, dtype={"source_id": str}) for f in files], ignore_index=True)


def test_saved_predictions_rules():
    import calibrate
    d = _final()
    sub = _subsamples()["roles"]["final"]["draws"]
    assert set(d["method"]) == {"B1-eq"} and set(d["analysis"]) == {"all"}
    assert sorted(d["subsample_seed"].unique()) == [101, 102, 103, 104, 105]
    assert set(d.loc[d["generator"] == "FLUX", "partition"]) == {"F_unseen_test"}
    assert set(d["source_split"]) == {"cal", "test"}
    for (s, b, c), g in d.groupby(["subsample_seed", "backbone", "data_condition"]):
        assert g["train_ids_sha256"].iloc[0] == sub[str(s)]["sha256"] and g["n_train_images"].iloc[0] == 600
        assert not set(g["source_id"]) & set(sub[str(s)]["source_ids"])
        cal = g[g["partition"] == "R_cal"]
        assert (cal["generator"] == "REAL").all() and len(cal) == 200
        assert calibrate.threshold(cal["score"].to_numpy()) == g["threshold"].iloc[0]
    assert (d["predicted_label"] == (d["score"] > d["threshold"]).astype(int)).all()


def test_eval_rows_match_sbf_final():
    d = _final()
    files = sorted((ROOT / "artifacts/predictions/supp_bfree").glob("final_*.csv.gz"))
    if not files:
        pytest.skip("S-BF final predictions not available")
    s = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    s = s[(s["analysis"] == "all") & (s["method"] == "B1")]
    for (b, c), g in d.groupby(["backbone", "data_condition"]):
        ref = s[(s["backbone"] == b) & (s["data_condition"] == c)]
        for seed, gs in g.groupby("subsample_seed"):
            assert sorted(gs["row_id"]) == sorted(ref["row_id"])


def test_reused_reference_values_equal_sbf_tables():
    """B1-full, O1 and O2 values and their contrasts in the S-BF-EQ tables equal the S-BF tables (same draws)."""
    t = ROOT / "artifacts/tables"
    need = [t / "supp_bfree_eq/final_auroc_by_seed.csv", t / "supp_bfree_eq/final_contrasts.csv",
            t / "supp_bfree/final_main_all.csv", t / "supp_bfree/final_contrasts_all.csv"]
    if not all(p.exists() for p in need):
        pytest.skip("S-BF or S-BF-EQ tables not available")
    eq, con, main, sb_con = (pd.read_csv(p) for p in need)
    for label, m in (("B1-full", "B1"), ("O1", "O1"), ("O2", "O2")):
        a = eq[eq["Method"] == label].set_index("Cell")
        b = main[main["Method"] == m].set_index("Cell")
        for col in ("AUROC FLUX (unseen)", "AUROC SD3.5 (seen by B1)"):
            assert (a[col] == b.loc[a.index, col]).all(), (label, col)
    ref = con[con["B1"] == "B1-full (S-BF)"].set_index("Cell")
    for m in ("O1", "O2"):
        s = sb_con[sb_con["Contrast"] == f"{m} minus B1"].set_index("Cell")
        assert (ref[f"{m} minus B1, AUROC FLUX"] == s.loc[ref.index, "AUROC FLUX"]).all(), m

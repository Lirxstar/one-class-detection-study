"""S-BF supplementary experiment (PROTOCOL.md Section 7): integrity, leakage and metric tests.

Tests on saved artifacts skip when those artifacts do not exist yet (audit, splits, features, predictions).
"""
import json

import numpy as np
import pandas as pd
import pytest

import config
from config import ROOT

SB = ROOT / "artifacts" / "supp_bfree"


def _manifest():
    p = SB / "manifest.csv"
    if not p.exists():
        pytest.skip("S-BF splits not built")
    return pd.read_csv(p, keep_default_na=False, dtype=str)


# --------------------------------------------------------------------------- core untouched
def test_core_configs_unchanged():
    """The S-BF config is not part of the core config hash; the core hash equals that of the last core run."""
    assert config.CONFIG_FILES == ("conditions.yaml", "core.yaml", "e2_postprocessing.yaml", "e3_encoding.yaml")
    p = ROOT / "artifacts/predictions/posthoc/posthoc_reseed_primary_d_ctrl.csv.gz"
    if not p.exists():
        pytest.skip("core predictions not available")
    saved = pd.read_csv(p, usecols=["config_hash"], nrows=1)["config_hash"].iloc[0]
    assert saved == config.config_hash()


def test_outputs_outside_core_directories():
    import supp_bfree_analysis as A
    import supp_bfree_audit as AU
    import supp_bfree_features as F
    import supp_bfree_run as R
    import supp_bfree_splits as SP
    for p in (A.PRED, A.TABLES, A.FIG, AU.OUT, F.CACHE, R.OUT, SP.OUT):
        assert "supp_bfree" in p.parts, p
    assert F.FEAT_DIR != ROOT / "artifacts" / "cache" / "features"


# --------------------------------------------------------------------------- splits and leakage
def test_partitions_disjoint_by_source_id():
    m = _manifest()
    roles = m.groupby("source_id")["split_role"].nunique()
    assert (roles == 1).all(), "a source id has rows in two roles"
    grp = m.groupby("id_group")["split_role"].nunique()
    assert (grp == 1).all(), "a duplicate group straddles roles"
    ids = {r: set(g["source_id"]) for r, g in m.groupby("split_role")}
    assert not (ids["train"] & ids["cal"]) and not (ids["train"] & ids["test"]) and not (ids["cal"] & ids["test"])


def test_partition_classes_and_flux_only_in_test():
    m = _manifest()
    allowed = {"R_train": "REAL", "F_seen_train": "SD35", "R_cal": "REAL", "R_test": "REAL", "F_unseen_test": "FLUX",
               "F_seen_test": "SD35"}
    for p, cls in allowed.items():
        assert (m.loc[m["partition"] == p, "class"] == cls).all(), p
    flux = m[m["class"] == "FLUX"]
    assert set(flux["partition"]) <= {"F_unseen_test", "unused"}
    assert (flux.loc[flux["partition"] == "F_unseen_test", "split_role"] == "test").all()
    used = m[m["partition"] != "unused"]
    assert set(used["partition"]) == set(allowed)


def test_every_test_id_is_a_complete_triplet():
    m = _manifest()
    t = m[m["split_role"] == "test"]
    per = t[t["partition"] != "unused"].groupby("source_id")["class"].apply(lambda s: tuple(sorted(s)))
    assert (per == ("FLUX", "REAL", "SD35")).all()


def test_pilot_uses_train_ids_only_and_no_flux():
    m = _manifest()
    p = m[m["pilot_partition"] != ""]
    assert (p["split_role"] == "train").all()
    assert not (p["class"] == "FLUX").any()
    roles = p.groupby("source_id")["pilot_role"].nunique()
    assert (roles == 1).all()


def test_split_sizes_and_strata():
    s = json.loads((SB / "splits.json").read_text()) if (SB / "splits.json").exists() else pytest.skip("no splits")
    n = s["eligible_ids"]
    for role, frac in (("train", 0.6), ("cal", 0.2), ("test", 0.2)):
        assert abs(s["ids_per_role"][role] - frac * n) <= max(3, s["largest_id_group"] + 1)
    m = _manifest()
    tag = config.load("supp_bfree")["colour"]["sensitivity"]["real_icc_description"]
    real = m[m["class"] == "REAL"]
    assert (real.loc[real["stratum"] == "srgb", "icc_description"] == tag).all()
    assert ((real["srgb_tagged_id"] == "True") == (real["icc_description"] == tag)).all()
    audit = json.loads((SB / "audit" / "audit.json").read_text())
    assert (real["srgb_tagged_id"] == "True").sum() == audit["eligibility"]["ids_eligible_srgb_tagged"]


def test_splits_reproducible():
    if not (SB / "audit" / "files.csv").exists() or not (SB / "manifest.csv").exists():
        pytest.skip("S-BF audit or splits not built")
    import supp_bfree_splits as SP
    manifest, splits = SP.build()
    saved = pd.read_csv(SB / "manifest.csv", keep_default_na=False, dtype=str)
    assert manifest.astype(str).reset_index(drop=True).equals(saved.reset_index(drop=True))
    assert splits["partitions"] == json.loads((SB / "splits.json").read_text())["partitions"]


# --------------------------------------------------------------------------- label rules in fitting
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
@pytest.mark.parametrize("analysis", ["all", "srgb"])
def test_training_inputs_rows(role, analysis):
    m = _manifest().set_index("row_id")
    import supp_bfree_run as R
    feats = _FakeFeats(sorted(m.index))
    d = R.training_inputs(feats, role, analysis)
    assert set(d) == {"R_train", "F_seen_train"}
    r_ids = list(d["R_train"]["row_ids"])
    assert (m.loc[r_ids, "class"] == "REAL").all() and (m.loc[r_ids, "split_role"] == "train").all()
    fake_rows = R.part_ids(role, R.ROLES[role]["fit_fake"], analysis)
    assert (m.loc[fake_rows, "class"] == "SD35").all() and (m.loc[fake_rows, "split_role"] == "train").all()
    assert len(d["F_seen_train"]["h_post"]) == len(fake_rows)
    if role == "pilot":
        assert set(m.loc[r_ids, "pilot_role"]) == {"fit"}
    if analysis == "srgb":
        assert (m.loc[r_ids, "srgb_tagged_id"] == "True").all()


def test_real_only_methods_never_receive_synthetic_rows(monkeypatch):
    """O1 and O2 are fitted on the R_train arrays only; the arrays passed for R_train are real rows."""
    m = _manifest().set_index("row_id")
    import detectors as D
    import supp_bfree_run as R
    seen = {}
    real_fit_o1, real_fit_o2 = D.O1.fit, D.O2.fit

    def o1(h, ids):
        seen["o1"] = list(ids)
        return real_fit_o1(h, ids)

    def o2(h):
        seen["o2"] = h.shape
        return real_fit_o2(h)

    monkeypatch.setattr(D.O1, "fit", staticmethod(o1))
    monkeypatch.setattr(D.O2, "fit", staticmethod(o2))
    feats = _FakeFeats(sorted(m.index))
    d = R.training_inputs(feats, "pilot", "all")
    D.fit_methods(d, [])
    assert (m.loc[seen["o1"], "class"] == "REAL").all()
    assert seen["o2"][0] == len(seen["o1"])


# --------------------------------------------------------------------------- metrics and bootstrap
def test_matched_fpr_threshold_bruteforce():
    import supp_bfree_analysis as A
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = int(rng.integers(20, 300))
        real = np.round(rng.normal(size=n), int(rng.integers(0, 3)))
        fake = np.round(rng.normal(0.7, 1, size=int(rng.integers(20, 300))), 2)
        tpr, fpr = A.tpr_at_matched_fpr(real, fake, 0.05)
        assert fpr <= 0.05 + 1e-12
        best = max((fake > t).mean() for t in np.r_[real, np.inf] if (real > t).mean() <= 0.05)
        assert tpr == pytest.approx(best)


def test_pair_rate_and_auroc_orientation():
    import supp_bfree_analysis as A
    real = np.array([0.0, 1.0, 2.0])
    assert A.pair_rate(real, real + 1) == 1.0 and A.pair_rate(real, real) == 0.5
    assert A.metrics(real, real + 10, tau=1.0, alpha=0.05)["auroc"] == 1.0


def test_cluster_draws_keep_groups_together():
    import supp_bfree_analysis as A
    ids = [f"r{i:03d}" for i in range(10)]
    groups = pd.Series(["g0", "g0", "g1", "g2", "g3", "g3", "g3", "g4", "g5", "g6"], index=ids)
    draws = A.make_draws(groups, 50, 2026)
    for d in draws:
        for g, members in ((0, [0, 1]), (4, [4, 5, 6])):
            c = [int((d == j).sum()) for j in members]
            assert len(set(c)) == 1  # members of a group are drawn equally often
    singles = A.make_draws(pd.Series([f"g{i}" for i in range(10)], index=ids), 5, 2026)
    assert all(len(d) == 10 for d in singles)


# --------------------------------------------------------------------------- decoding and colour rule
def test_srgb_conversion_identity_and_adobe_change():
    files = SB / "audit" / "files.csv"
    if not files.exists():
        pytest.skip("S-BF audit not built")
    import supp_bfree_data as S
    if not S.class_dir("REAL").exists():
        pytest.skip("B-Free images not available")
    f = pd.read_csv(files, keep_default_na=False, dtype=str)
    real = f[f["class"] == "REAL"]
    srgb = real[real["icc_description"] == "Nikon sRGB 4.0.0.3002"]["source_id"].head(3)
    adobe = real[real["icc_description"] == "Nikon Adobe RGB 4.0.0.3001"]["source_id"].head(2)
    for sid in srgb:
        assert np.array_equal(S.decode_srgb("REAL", sid), S.decode("REAL", sid))
    for sid in adobe:
        assert not np.array_equal(S.decode_srgb("REAL", sid), S.decode("REAL", sid))
    sid = f[f["class"] == "FLUX"]["source_id"].iloc[0]
    assert np.array_equal(S.decode_srgb("FLUX", sid), S.decode("FLUX", sid))


# --------------------------------------------------------------------------- saved predictions
def _final():
    files = sorted((ROOT / "artifacts/predictions/supp_bfree").glob("final_*.csv.gz"))
    if not files:
        pytest.skip("S-BF final predictions not available")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def test_saved_thresholds_recompute_from_real_calibration_rows():
    import calibrate
    d = _final()
    cal = d[d["partition"] == "R_cal"]
    assert (cal["generator"] == "REAL").all()
    for _, g in cal.groupby(["analysis", "backbone", "data_condition", "method"]):
        assert calibrate.threshold(g["score"].to_numpy()) == g["threshold"].iloc[0]
    assert (d["predicted_label"] == (d["score"] > d["threshold"]).astype(int)).all()


def test_saved_predictions_share_no_source_id_across_roles():
    d = _final()
    roles = d.groupby("source_id")["source_split"].nunique()
    assert (roles == 1).all()
    assert set(d.loc[d["generator"] == "FLUX", "partition"]) == {"F_unseen_test"}


def test_m0_orientation_and_independence_from_training():
    d = _final()
    m0 = d[d["method"] == "M0"].pivot_table(index=["row_id", "backbone", "data_condition"], columns="analysis",
                                            values="score")
    both = m0.dropna()
    assert len(both) and np.array_equal(both["all"].to_numpy(), both["srgb"].to_numpy())
    assert (d.loc[d["method"] == "M0", "score"] < 0).all()

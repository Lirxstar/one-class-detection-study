"""PROTOCOL.md and configs/*.yaml must agree on every frozen value."""
import re

import pytest

import config
from config import ROOT

ALLOWED_PENDING: dict[str, set[str]] = {}  # v0.6.2: every E0 value is resolved


@pytest.fixture(scope="module")
def cfg():
    return config.load_all()


@pytest.fixture(scope="module")
def protocol_text():
    return (ROOT / "PROTOCOL.md").read_text(encoding="utf-8")


def test_all_configs_load(cfg):
    assert set(cfg) == {"core", "conditions", "e2_postprocessing", "e3_encoding"}


def test_protocol_version_matches(cfg, protocol_text):
    m = re.search(r"\*\*Protocol version:\*\* v(\S+)", protocol_text)
    assert m and m.group(1) == cfg["core"]["protocol_version"]


def test_only_declared_values_pending(cfg):
    for name, c in cfg.items():
        assert set(config.pending_values(c)) <= ALLOWED_PENDING.get(name, set()), name


def test_dataset_pinned(cfg, protocol_text):
    rev = cfg["core"]["data"]["revision"]
    assert re.fullmatch(r"[0-9a-f]{40}", rev)
    assert rev in protocol_text
    prov = (ROOT / "artifacts/provenance/dataset_revision.txt").read_text()
    assert f"revision: {rev}" in prov


def test_encoder_pinned(cfg, protocol_text):
    enc = cfg["core"]["encoder"]
    prov_w = (ROOT / "artifacts/provenance/weights.sha256").read_text()
    prov_c = (ROOT / "artifacts/provenance/dinov2_commit.txt").read_text()
    for role in ("primary", "secondary"):
        e = enc[role]
        assert e["registers"] is False
        assert f"{e['weights_sha256']}  {e['weights']}" in prov_w
        assert f"commit: {e['repo_commit']}" in prov_c
        assert e["weights_sha256"] in protocol_text
    assert enc["primary"]["name"] == "dinov2_vits14" and enc["primary"]["dim"] == 384
    assert enc["device"] == "cpu" and enc["dtype"] == "float32"


def test_generator_roles_disjoint(cfg):
    d = cfg["core"]["data"]
    seen, unseen, dev = {d["generator_seen"]}, set(d["generators_unseen"]), set(d["generators_dev_only"])
    assert seen == {"SD15"} and unseen == {"ADM", "GLIDE", "Midjourney"}
    assert dev == {"VQDM", "Wukong", "BigGAN"} and d["generators_absent"] == ["SD14"]
    assert not (seen & unseen) and not ((seen | unseen) & dev)
    assert set(d["eligibility"]["synthetic"]["generators"]) == seen | unseen
    assert dev == set(cfg["core"]["integrity"]["dev_only_train_generators"])


def test_targets(cfg):
    t = cfg["core"]["data"]["targets"]
    assert t == {"R_train": 2000, "F_seen_train_max": 2000, "R_cal": 500, "R_test": 500,
                 "fake_test_per_generator": 300}


def test_real_eligibility(cfg):
    r = cfg["core"]["data"]["eligibility"]["real"]
    assert r["format"] == "JPEG" and r["jpeg_quality_exact_ijg"] == 96 and r["mode"] == "RGB"
    assert r["jpeg_components"] == 3 and r["exif_orientation_allowed"] == [None, 1] and r["min_side"] == 224
    assert r["chroma_subsampling"] == "4:4:4"
    assert cfg["core"]["data"]["eligibility"]["exif_transpose"] == "never"
    s = cfg["core"]["data"]["eligibility"]["synthetic"]
    assert s["format"] == "PNG" and s["mode"] == ["RGB", "RGBA_alpha_255"] and s["min_side"] == 224


def test_near_duplicate_rule(cfg):
    near = cfg["core"]["data"]["duplicates"]["near"]
    assert near["hash"] == "dhash64" and near["max_hamming"] == 4


def test_methods_frozen(cfg):
    m = cfg["core"]["methods"]
    assert m["B1"]["C_grid"] == [0.01, 0.1, 1.0, 10.0, 100.0]
    assert m["B1"]["selection"]["n_splits"] == 5 and m["B1"]["selection"]["data"] == "train_split_only"
    assert m["B1"]["selection"]["tie_break"] == "smaller_C"
    assert m["B1"]["sensitivity_C"] == 1.0 and m["B1"]["max_iter"] == 5000 and m["B1"]["class_weight"] is None
    assert set(m["B1"]["labels_allowed"]) == {"R_train", "F_seen_train"}
    assert m["O1"]["k"] == 5 and m["O1"]["metric"] == "cosine"
    assert m["O2"]["covariance"] == "ledoit_wolf"
    assert m["M0"]["feature"] == "pre_ln_cls" and m["M0"]["flip"] == "never"
    assert m["M0"]["orientation"] == "fixed_negative"
    for name in ("O1", "O2", "M0"):
        assert "labels_allowed" not in m[name]


def test_calibration_rule(cfg):
    c = cfg["core"]["calibration"]
    assert c["target_fpr"] == 0.05 and c["source"] == "clean_authentic_R_cal_only"
    assert c["quantile_rule"]["flag"] == "score_strictly_greater"


def test_conditions(cfg):
    c = cfg["conditions"]
    assert c["primary"] == "d_ctrl" and c["ladder"] == ["d_raw", "d_jpeg", "d_ctrl"]
    assert c["conditions"]["d_ctrl"]["resize"] == "none"
    assert c["conditions"]["d_ctrl"]["offset_multiple_rule"] == {"4:2:0": 16, "default": 8}
    for k in ("d_jpeg", "d_ctrl"):
        assert c["conditions"][k]["jpeg_harmonize"]["quality"] == 96
        assert c["conditions"][k]["jpeg_harmonize"]["apply_to"] == "synthetic"
    assert c["conditions"]["d_raw"]["jpeg_harmonize"] is False
    assert c["modal_real_subsampling"] == "4:4:4"
    rule = c["conditions"]["d_ctrl"]["offset_multiple_rule"]
    assert c["conditions"]["d_ctrl"]["offset_multiple"] == rule.get(c["modal_real_subsampling"], rule["default"]) == 8


def test_e2_and_e3(cfg):
    e2, e3 = cfg["e2_postprocessing"], cfg["e3_encoding"]
    assert e2["transforms"]["jpeg"]["quality"] == [95, 75, 50]
    assert e2["transforms"]["gaussian_blur"]["sigma"] == [0.5, 1.0]
    assert e2["reuse_clean_threshold"] is True and e2["refit"] is False
    assert [v["name"] for v in e3["E3C"]["C1"]["test_variants"]] == ["biased", "matched", "inverted"]
    assert e3["E3C"]["jpeg"]["quality"] == 75
    assert e3["E3C"]["C0"]["reuse_d_ctrl_features_if_bitwise_equal"] is True


def test_config_hash_is_stable():
    h1, h2 = config.config_hash(), config.config_hash()
    assert h1 == h2 and re.fullmatch(r"[0-9a-f]{64}", h1)

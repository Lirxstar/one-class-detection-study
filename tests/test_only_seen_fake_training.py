"""O1, O2, M0 never read synthetic labels; B1 reads only R_train/F_seen_train (PROTOCOL.md 9.4)."""
import numpy as np
import pytest

import detectors as D
import pipeline as P


def _toy(seed=0, n=60, d=8):
    rng = np.random.default_rng(seed)
    return {"R_train": {"h_post": rng.normal(size=(n, d)), "row_ids": np.array([f"r{i}" for i in range(n)]),
                        "groups": np.arange(n)},
            "F_seen_train": {"h_post": rng.normal(1.0, 1.0, size=(n, d)), "groups": np.arange(n, 2 * n)}}


def test_fit_methods_rejects_other_partitions():
    t = _toy()
    with pytest.raises(ValueError):
        D.fit_methods({**t, "F_seen_test": t["F_seen_train"]})
    with pytest.raises(ValueError):
        D.fit_methods({"R_train": t["R_train"]})


def test_reference_methods_ignore_synthetic_training_data():
    a, b = _toy(0), _toy(0)
    b["F_seen_train"]["h_post"] = np.random.default_rng(9).normal(size=b["F_seen_train"]["h_post"].shape)
    fa, fb = D.fit_methods(a), D.fit_methods(b)
    q = np.random.default_rng(5).normal(size=(10, 8))
    assert np.array_equal(fa.o1.score(q), fb.o1.score(q))
    assert np.array_equal(fa.o2.score(q), fb.o2.score(q))
    assert not np.array_equal(fa.b1.score(q), fb.b1.score(q))


def test_training_inputs_are_exactly_rtrain_and_fseentrain():
    feats = pytest.importorskip("pipeline").Features
    try:
        f = feats("primary", "d_ctrl", "clean")
    except FileNotFoundError:
        pytest.skip("feature cache not built")
    t = P.training_inputs(f)
    assert set(t) == {"R_train", "F_seen_train"}
    m = P.meta()
    r = m.loc[list(t["R_train"]["row_ids"])]
    assert (r["partition"] == "R_train").all() and (r["label"] == 0).all() and (r["split"] == "train").all()
    assert len(t["F_seen_train"]["h_post"]) == len(P.part_ids("F_seen_train"))
    fs = m.loc[P.part_ids("F_seen_train")]
    assert set(fs["generator"]) == {"SD15"} and set(fs["split"]) == {"train"}


def test_fit_and_thresholds_invariant_to_validation_synthetic_labels_and_features(monkeypatch):
    try:
        f = P.Features("primary", "d_ctrl", "clean")
    except FileNotFoundError:
        pytest.skip("feature cache not built")
    fit = D.fit_methods(P.training_inputs(f))
    cal = P.part_ids("R_cal")
    taus = P.thresholds(fit, f, cal)
    # scramble labels/generators of every validation synthetic row and their features
    m = P.meta().copy()
    vs = m.index[(m["split"] == "validation") & (m["label"] == 1)]
    m.loc[vs, "generator"] = "SCRAMBLED"
    monkeypatch.setattr(P, "meta", lambda: m)
    for r in vs:
        if r in f.pos:
            f.h_post[f.pos[r]] = 0.0
            f.h_pre[f.pos[r]] = 0.0
    fit2 = D.fit_methods(P.training_inputs(f))
    assert fit2.b1.C == fit.b1.C
    assert P.thresholds(fit2, f, cal) == taus


def test_calibration_rejects_synthetic_rows():
    try:
        f = P.Features("primary", "d_ctrl", "clean")
    except FileNotFoundError:
        pytest.skip("feature cache not built")
    fit = D.fit_methods(P.training_inputs(f))
    with pytest.raises(AssertionError):
        P.thresholds(fit, f, P.part_ids("R_cal")[:10] + P.part_ids("F_seen_test")[:1])

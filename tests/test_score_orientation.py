"""Larger score = more synthetic; M0 = -||h_pre||_1, never flipped (PROTOCOL.md 5)."""
import numpy as np

import detectors as D


def test_m0_is_negative_l1():
    h = np.array([[1.0, -2.0, 3.0], [0.5, 0.5, 0.0]])
    assert np.array_equal(D.m0(h), np.array([-6.0, -1.0]))
    assert D.m0(2 * h)[0] < D.m0(h)[0]  # larger magnitude, less synthetic


def test_m0_variants_fixed_orientation():
    h = np.array([[3.0, 4.0]])
    assert D.m0_l2(h)[0] == -5.0 and D.m0_ratio(h)[0] == -(7.0 / 5.0)
    assert np.array_equal(D.m0_x(np.array([[1.0, 10.0, 2.0]]), [1]), np.array([-3.0]))


def test_score_all_uses_fixed_m0():
    rng = np.random.default_rng(0)
    t = {"R_train": {"h_post": rng.normal(size=(30, 5)), "row_ids": np.arange(30).astype(str), "groups": np.arange(30)},
         "F_seen_train": {"h_post": rng.normal(2, 1, size=(30, 5)), "groups": np.arange(30, 60)}}
    fit = D.fit_methods(t)
    h_pre = rng.normal(size=(7, 5))
    s = D.score_all(fit, h_pre, rng.normal(size=(7, 5)), None)
    assert np.array_equal(s["M0"], -np.abs(h_pre).sum(axis=1))


def test_reference_scores_grow_with_distance():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(200, 6)) + np.array([5, 0, 0, 0, 0, 0])
    o1, o2 = D.O1.fit(ref, np.arange(200).astype(str)), D.O2.fit(ref)
    near = np.array([[5.0, 0, 0, 0, 0, 0]])
    far = np.array([[-5.0, 0, 0, 0, 0, 0]])
    assert o1.score(far)[0] > o1.score(near)[0]
    assert o2.score(far)[0] > o2.score(near)[0]
    assert (o2.score(ref) >= 0).all()  # quadratic form with a positive-definite covariance


def test_o1_excludes_self():
    ref = np.eye(6)
    o1 = D.O1.fit(ref, np.array(list("abcdef")))
    assert o1.score(ref[:1], ids=["a"], k=1)[0] == 1.0  # nearest other point is orthogonal
    assert o1.score(ref[:1], k=1)[0] == 0.0


def test_b1_positive_class_scores_higher():
    rng = np.random.default_rng(2)
    t = {"R_train": {"h_post": rng.normal(-1, 1, (80, 4)), "row_ids": np.arange(80).astype(str), "groups": np.arange(80)},
         "F_seen_train": {"h_post": rng.normal(1, 1, (80, 4)), "groups": np.arange(80, 160)}}
    fit = D.fit_methods(t)
    assert fit.b1.score(np.ones((1, 4)))[0] > fit.b1.score(-np.ones((1, 4)))[0]

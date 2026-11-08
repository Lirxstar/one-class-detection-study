"""Threshold rule of PROTOCOL.md Section 6."""
import math

import numpy as np
import pytest

from calibrate import flag, order_statistic_index, threshold


@pytest.mark.parametrize("n", [1, 5, 19, 20, 99, 100, 499, 500, 501, 1000])
def test_k_matches_ceiling(n):
    assert order_statistic_index(n) == math.ceil(0.95 * (n + 1) - 1e-12)


def test_k_for_n_500():
    assert order_statistic_index(500) == 476


def test_threshold_is_kth_smallest():
    s = np.arange(500, dtype=float)[::-1]  # unsorted input
    assert threshold(s) == 475.0           # 476th smallest of 0..499
    assert flag(s, threshold(s)).sum() == 24


def test_ties_not_flagged():
    s = np.r_[np.zeros(480), np.ones(20)]
    tau = threshold(s)
    assert tau == 0.0
    assert not flag([0.0], tau)[0] and flag([1e-9], tau)[0]


def test_small_n_gives_infinite_threshold():
    assert threshold(np.arange(10.0)) == float("inf")   # k = 11 > n = 10
    assert not flag([1e300], threshold(np.arange(10.0)))[0]


def test_rejects_nonfinite_and_empty():
    with pytest.raises(ValueError):
        threshold([0.0, np.nan])
    with pytest.raises(ValueError):
        threshold([])


def test_marginal_fpr_at_most_target_under_exchangeability():
    rng = np.random.default_rng(0)
    fpr = []
    for _ in range(400):
        x = rng.standard_normal(500 + 2000)
        tau = threshold(x[:500])
        fpr.append(flag(x[500:], tau).mean())
    assert abs(np.mean(fpr) - 25 / 501) < 0.004

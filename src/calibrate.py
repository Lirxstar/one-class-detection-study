"""Authentic-only threshold calibration (PROTOCOL.md Section 6).

k = ceil(0.95 (n + 1)) in integer arithmetic; tau = k-th smallest calibration
score (1-indexed), +inf if k > n; flag iff score > tau (ties not flagged).
"""
from __future__ import annotations

import numpy as np

TARGET_FPR_PERCENT = 5  # integer percent avoids float error in the ceiling


def order_statistic_index(n: int, target_fpr_percent: int = TARGET_FPR_PERCENT) -> int:
    """1-indexed k = ceil((100 - target) / 100 * (n + 1)), exact integer arithmetic."""
    if n < 1:
        raise ValueError("calibration set is empty")
    keep = 100 - target_fpr_percent
    return -(-keep * (n + 1) // 100)


def threshold(cal_scores, target_fpr_percent: int = TARGET_FPR_PERCENT) -> float:
    s = np.asarray(cal_scores, dtype=np.float64).ravel()
    if not np.all(np.isfinite(s)):
        raise ValueError("calibration scores must be finite")
    k = order_statistic_index(s.size, target_fpr_percent)
    if k > s.size:
        return float("inf")
    return float(np.sort(s, kind="stable")[k - 1])


def flag(scores, tau: float) -> np.ndarray:
    """Predicted synthetic iff score > tau (strict)."""
    return np.asarray(scores, dtype=np.float64) > tau

"""Paired percentile bootstrap over test identities (PROTOCOL.md Section 8).

Draws are generated once per identity set (the shared real test set and each
generator's test set, identities sorted by row_id) from
`default_rng(seed)` in a fixed order, and reused for every method, condition,
perturbation, and encoding variant, so all contrasts are paired. Thresholds are
not resampled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

ORDER = ["REAL", "SD15", "ADM", "GLIDE", "Midjourney", "VQDM", "Wukong"]


class Draws:
    """Resampling counts: counts[name] is (B, n) with how often each identity is drawn."""

    def __init__(self, ids: dict[str, list[str]], n_boot: int | None = None, seed: int | None = None):
        core = config.load("core")
        self.B = int(n_boot or core["statistics"]["n_bootstrap"])
        rng = np.random.default_rng(int(core["seeds"]["bootstrap"] if seed is None else seed))
        self.ids, self.counts = {}, {}
        for name in [n for n in ORDER if n in ids] + sorted(set(ids) - set(ORDER)):
            s = sorted(ids[name])
            n = len(s)
            idx = rng.integers(0, n, size=(self.B, n))
            c = np.zeros((self.B, n), np.float64)
            np.add.at(c, (np.repeat(np.arange(self.B), n), idx.ravel()), 1.0)
            self.ids[name], self.counts[name] = s, c

    def align(self, name: str, df: pd.DataFrame, col: str = "score") -> np.ndarray:
        """Vector of `col` ordered like the identities of `name` (df indexed by row_id)."""
        return df.loc[self.ids[name], col].to_numpy(np.float64)


def boot_auroc(real: np.ndarray, fake: np.ndarray, cr: np.ndarray, cf: np.ndarray) -> np.ndarray:
    """AUROC per draw from count matrices: sum_ij cf_j M_ji cr_i / (n_f n_r)."""
    M = (fake[:, None] > real[None, :]).astype(np.float64) + 0.5 * (fake[:, None] == real[None, :])
    return np.einsum("bi,bi->b", cf @ M, cr) / (cf.shape[1] * cr.shape[1])


def boot_rate(flags: np.ndarray, c: np.ndarray) -> np.ndarray:
    return c @ flags.astype(np.float64) / c.shape[1]


def ci(x: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)

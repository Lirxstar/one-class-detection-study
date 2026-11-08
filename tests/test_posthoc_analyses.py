"""Post-hoc analyses (PROTOCOL v0.6.4): Jaccard chance reference and duplicate exclusion counts."""
import numpy as np
import pandas as pd

import evaluate as E
import make_splits


def test_jaccard_chance_matches_simulation():
    rng = np.random.default_rng(0)
    n, a, b = 500, 24, 30
    sims = []
    for _ in range(20000):
        A = set(rng.choice(n, a, replace=False))
        B = set(rng.choice(n, b, replace=False))
        sims.append(len(A & B) / len(A | B))
    ch = E.jaccard_chance(a, b, observed=2, n=n)
    assert abs(ch["chance_mean"] - np.mean(sims)) < 0.002
    assert abs(ch["expected_intersection"] - a * b / n) < 1e-12
    assert 0 < ch["p_upper"] < 1 and 0 < ch["p_lower"] < 1


def test_duplicate_exclusions_consistent_with_manifest(manifest):
    ex = make_splits.duplicate_exclusions(manifest)
    elig = manifest["eligible"].astype(str).str.lower().eq("true")
    pools = ex.set_index(["split", "generator"])
    got = manifest[elig].groupby(["split", "generator"]).size()
    for key, n_elig in got.items():
        assert pools.loc[key, "passing_other_rules"] - pools.loc[key, "excluded_by_duplicates"] == n_elig

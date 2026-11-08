"""Post hoc R_train reseeding (PROTOCOL v0.6.6, proposal 6.3): draw rule, label isolation, feature union."""
import numpy as np
import pytest

from config import ROOT

pytestmark = pytest.mark.skipif(not (ROOT / "artifacts" / "splits.json").exists(), reason="partitions not built")


def test_seed42_reproduces_frozen_rtrain():
    import pipeline as P
    import posthoc_reseed as RS
    assert RS.draw(42) == P.part_ids("R_train")


def test_reseeded_references_are_eligible_train_reals_disjoint_from_validation():
    import pipeline as P
    import posthoc_reseed as RS
    m = P.meta()
    val = {r for name in ("R_cal", "R_test", "F_seen_test", "F_unseen_test") for r in P.part_ids(name)}
    pool = set(RS.train_real_pool()["row_id"])
    frozen = set(P.part_ids("R_train"))
    for s in RS.SEEDS:
        ids = RS.draw(s)
        assert len(ids) == len(set(ids)) == len(frozen)
        assert set(ids) <= pool
        assert (m.loc[ids, "split"] == "train").all() and (m.loc[ids, "label"] == 0).all()
        assert not set(ids) & val
        assert set(ids) != frozen


def test_feature_union_rejects_overlap_and_addresses_rows():
    import posthoc_reseed as RS
    a = {"row_ids": np.array(["x", "y"]), "h_pre": np.zeros((2, 3)), "h_post": np.zeros((2, 3))}
    b = {"row_ids": np.array(["z"]), "h_pre": np.ones((1, 3)), "h_post": np.ones((1, 3))}
    with pytest.raises(ValueError):
        RS.Union([a, a])
    pre, post = RS.Union([a, b]).take(["z", "x"])
    assert pre[0].sum() == 3 and pre[1].sum() == 0 and np.array_equal(pre, post)


def test_reference_rows_must_be_train_split_reals():
    import pipeline as P
    import posthoc_reseed as RS
    rid = P.part_ids("R_test")[:1]
    feats = RS.Union([{"row_ids": np.array(rid), "h_pre": np.zeros((1, 3)), "h_post": np.zeros((1, 3))}])
    with pytest.raises(AssertionError):
        RS.training_inputs(feats, rid)

"""B1 C selection: grouped 5-fold CV inside the training data only, ties to the smaller C."""
import numpy as np

import detectors as D
import pipeline as P


def test_folds_respect_groups_and_cover_all():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(100, int), np.ones(80, int)]
    groups = rng.integers(0, 60, size=180)
    folds = D.b1_cv_folds(y, groups)
    assert len(folds) == 5
    seen = np.concatenate([te for _, te in folds])
    assert sorted(seen) == list(range(180))
    for tr, te in folds:
        assert not set(groups[tr]) & set(groups[te])


def test_ties_go_to_smaller_C():
    rng = np.random.default_rng(1)
    z = np.r_[rng.normal(-5, 0.1, (50, 3)), rng.normal(5, 0.1, (50, 3))]
    y = np.r_[np.zeros(50, int), np.ones(50, int)]
    C, table = D.select_C(z, y, np.arange(100))
    assert all(t["mean_auroc"] == 1.0 for t in table)
    assert C == 0.01


def test_cv_sees_only_training_partition_rows(monkeypatch):
    seen_sizes = []
    orig = D.select_C

    def spy(z, y, groups, C_grid=None):
        seen_sizes.append(len(z))
        return orig(z, y, groups, C_grid)

    monkeypatch.setattr(D, "select_C", spy)
    rng = np.random.default_rng(2)
    t = {"R_train": {"h_post": rng.normal(size=(40, 4)), "row_ids": np.arange(40).astype(str), "groups": np.arange(40)},
         "F_seen_train": {"h_post": rng.normal(1, 1, size=(30, 4)), "groups": np.arange(40, 70)}}
    D.fit_methods(t)
    assert seen_sizes == [70]


def test_training_partitions_are_train_split():
    m = P.meta()
    for p in ("R_train", "F_seen_train"):
        assert set(m.loc[P.part_ids(p), "split"]) == {"train"}

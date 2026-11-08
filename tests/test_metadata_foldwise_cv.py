"""Regression checks for the post-review metadata preprocessing correction."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import detectors as D
import metadata_baseline as M


def test_scaler_never_fits_held_out_fold(monkeypatch):
    rng = np.random.default_rng(17)
    X = np.column_stack([np.arange(80), rng.normal(size=(80, 3))])
    y = np.tile([0, 1], 40)
    groups = np.arange(80)
    expected = [set(tr) for tr, _ in D.b1_cv_folds(y, groups)]
    fitted = []
    original = M.StandardScaler.fit

    def spy(self, X, *args, **kwargs):
        fitted.append(set(X[:, 0].astype(int)))
        return original(self, X, *args, **kwargs)

    monkeypatch.setattr(M.StandardScaler, "fit", spy)
    M.select_metadata_C(X, y, groups, C_grid=[0.1, 1.0])
    assert fitted == expected * 2
    assert all(len(rows) < len(X) for rows in fitted)


def test_selection_matches_independent_sklearn_pipelines():
    rng = np.random.default_rng(29)
    X = rng.normal(size=(100, 4)) * [0.1, 1, 10, 100]
    y = (X[:, 0] + rng.normal(0, 0.2, 100) > 0).astype(int)
    groups = np.arange(100)
    grid = [0.01, 0.1, 1.0]
    C, table = M.select_metadata_C(X, y, groups, grid)
    reference = []
    for c in grid:
        scores = []
        for tr, te in D.b1_cv_folds(y, groups):
            model = make_pipeline(StandardScaler(), D._lr(c)).fit(X[tr], y[tr])
            scores.append(roc_auc_score(y[te], model.decision_function(X[te])))
        reference.append(float(np.mean(scores)))
    assert np.allclose([r["mean_auroc"] for r in table], reference, rtol=0, atol=1e-12)
    assert C == grid[int(np.argmax(reference))]


def test_validation_rows_rejected_before_design():
    m = pd.DataFrame({"split": ["validation"]}, index=["held-out"])
    with pytest.raises(ValueError, match="training-split"):
        M.MetaBaseline(m, ["held-out"])

"""B1, O1, O2, M0 and their pre-registered exploratory variants (PROTOCOL.md Section 5).

Larger scores always mean "more synthetic". All arithmetic in float64.
Fitting functions take arrays only; which rows reach them is decided in
`fit_methods`, the single entry point used by every experiment, so that the
label rules (O1, O2, M0 never see synthetic labels; B1 sees only R_train and
F_seen_train) are enforced in one place and tested.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import config

CORE_METHODS = ["B1", "O1", "O2", "M0"]


def l2n(h: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    return h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), eps)


# --------------------------------------------------------------------------- B1
def _lr(C: float) -> LogisticRegression:
    b = config.load("core")["methods"]["B1"]
    return LogisticRegression(C=float(C), l1_ratio=0.0, solver=b["solver"], max_iter=int(b["max_iter"]),
                              class_weight=b["class_weight"], tol=1e-4, fit_intercept=True, random_state=42)


def b1_cv_folds(y: np.ndarray, groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    sel = config.load("core")["methods"]["B1"]["selection"]
    cv = StratifiedGroupKFold(n_splits=int(sel["n_splits"]), shuffle=True,
                              random_state=int(config.load("core")["seeds"]["cv"]))
    return list(cv.split(np.zeros(len(y)), y, groups))


def select_C(z: np.ndarray, y: np.ndarray, groups: np.ndarray, C_grid=None) -> tuple[float, list[dict]]:
    """Grouped 5-fold CV inside the training data only; mean fold AUROC; ties to the smaller C."""
    if C_grid is None:
        C_grid = config.load("core")["methods"]["B1"]["C_grid"]
    folds = b1_cv_folds(y, groups)
    table, best_C, best = [], None, -np.inf
    for C in sorted(float(c) for c in C_grid):
        aucs, iters = [], []
        for tr, te in folds:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                m = _lr(C).fit(z[tr], y[tr])
            aucs.append(roc_auc_score(y[te], m.decision_function(z[te])))
            iters.append(int(m.n_iter_[0]))
        mean = float(np.mean(aucs))
        table.append({"C": C, "mean_auroc": mean, "fold_auroc": [float(a) for a in aucs], "max_n_iter": max(iters)})
        if mean > best:  # strict: ties keep the smaller C (ascending order)
            best_C, best = C, mean
    return best_C, table


@dataclass
class B1:
    C: float
    model: LogisticRegression = field(repr=False)
    cv_table: list | None = None
    n_iter: int = 0

    @classmethod
    def fit(cls, z_real, z_fake, g_real, g_fake, C: float | None = None) -> "B1":
        z = np.vstack([l2n(z_real), l2n(z_fake)])
        y = np.r_[np.zeros(len(z_real), int), np.ones(len(z_fake), int)]
        g = np.r_[g_real, g_fake]
        table = None
        if C is None:
            C, table = select_C(z, y, g)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            m = _lr(C).fit(z, y)
        return cls(C=float(C), model=m, cv_table=table, n_iter=int(m.n_iter_[0]))

    def score(self, h_post: np.ndarray) -> np.ndarray:
        return self.model.decision_function(l2n(h_post)).astype(np.float64)


# --------------------------------------------------------------------------- O1
@dataclass
class O1:
    ref: np.ndarray = field(repr=False)
    ref_ids: np.ndarray = field(repr=False)

    @classmethod
    def fit(cls, h_post_real: np.ndarray, ids_real) -> "O1":
        return cls(ref=l2n(h_post_real), ref_ids=np.asarray(ids_real).astype(str))

    def score(self, h_post: np.ndarray, ids=None, k: int = 5) -> np.ndarray:
        """Cosine distance to the k-th nearest reference image (a query in the reference set excludes itself)."""
        q = l2n(h_post)
        d = 1.0 - q @ self.ref.T
        if ids is not None:
            pos = {r: i for i, r in enumerate(self.ref_ids)}
            for qi, r in enumerate(np.asarray(ids).astype(str)):
                if r in pos:
                    d[qi, pos[r]] = np.inf
        return np.partition(d, k - 1, axis=1)[:, k - 1]


# --------------------------------------------------------------------------- O2
@dataclass
class O2:
    mu: np.ndarray = field(repr=False)
    cov: np.ndarray = field(repr=False)
    shrinkage: float = 0.0
    condition_number: float = 0.0

    @classmethod
    def fit(cls, h_post_real: np.ndarray) -> "O2":
        lw = LedoitWolf(assume_centered=False).fit(l2n(h_post_real))
        cov = np.asarray(lw.covariance_, dtype=np.float64)
        return cls(mu=np.asarray(lw.location_, dtype=np.float64), cov=cov, shrinkage=float(lw.shrinkage_),
                   condition_number=float(np.linalg.cond(cov)))

    def score(self, h_post: np.ndarray) -> np.ndarray:
        d = l2n(h_post) - self.mu
        sol = np.linalg.solve(self.cov, d.T).T
        return np.einsum("ij,ij->i", d, sol)


# --------------------------------------------------------------------------- M0
def m0(h_pre: np.ndarray) -> np.ndarray:
    """M0 = -||h_pre||_1, orientation fixed, never flipped."""
    return -np.abs(np.asarray(h_pre, dtype=np.float64)).sum(axis=1)


def m0_l2(h_pre: np.ndarray) -> np.ndarray:
    return -np.linalg.norm(np.asarray(h_pre, dtype=np.float64), axis=1)


def m0_ratio(h_pre: np.ndarray) -> np.ndarray:
    h = np.asarray(h_pre, dtype=np.float64)
    return -(np.abs(h).sum(axis=1) / np.linalg.norm(h, axis=1))


def dominant_dims(h_pre_rtrain: np.ndarray, factor: float = 10.0) -> dict:
    """Dimension-dominance check (PROTOCOL.md 4): dims with mean |h| > factor x median."""
    a = np.abs(np.asarray(h_pre_rtrain, dtype=np.float64)).mean(axis=0)
    share = a / a.sum()
    dims = np.nonzero(a > factor * np.median(a))[0]
    order = np.argsort(-share)
    return {"dominant_dims": [int(d) for d in dims], "dominant_share": float(share[dims].sum()),
            "median_mean_abs": float(np.median(a)), "max_over_median": float(a.max() / np.median(a)),
            "top10_dims": [int(d) for d in order[:10]], "top10_shares": [float(share[d]) for d in order[:10]],
            "uniform_share": 1.0 / len(a)}


def m0_x(h_pre: np.ndarray, excluded: list[int]) -> np.ndarray:
    h = np.asarray(h_pre, dtype=np.float64)
    keep = np.setdiff1d(np.arange(h.shape[1]), np.asarray(excluded, dtype=int))
    return -np.abs(h[:, keep]).sum(axis=1)


# --------------------------------------------------------------------------- fitting entry point
@dataclass
class Fitted:
    b1: B1
    b1_c1: B1
    o1: O1
    o2: O2
    m0x_dims: list[int]

    def diagnostics(self) -> dict:
        return {"B1_selected_C": self.b1.C, "B1_cv_table": self.b1.cv_table, "B1_n_iter": self.b1.n_iter,
                "B1_C1_n_iter": self.b1_c1.n_iter, "O2_shrinkage": self.o2.shrinkage,
                "O2_condition_number": self.o2.condition_number, "M0x_excluded_dims": self.m0x_dims}


def fit_methods(train: dict, m0x_dims: list[int] | None = None) -> Fitted:
    """Fit all methods from a training-feature dict produced by `features.training_inputs`.

    `train` holds exactly: R_train (h_post, row_ids, groups) and F_seen_train (h_post, groups).
    O1 and O2 receive only R_train arrays; B1 receives R_train (label 0) and F_seen_train (label 1).
    """
    allowed = {"R_train", "F_seen_train"}
    if set(train) != allowed:
        raise ValueError(f"training dict must contain exactly {allowed}, got {set(train)}")
    r, f = train["R_train"], train["F_seen_train"]
    b1 = B1.fit(r["h_post"], f["h_post"], r["groups"], f["groups"])
    b1_c1 = B1.fit(r["h_post"], f["h_post"], r["groups"], f["groups"],
                   C=float(config.load("core")["methods"]["B1"]["sensitivity_C"]))
    o1 = O1.fit(r["h_post"], r["row_ids"])
    o2 = O2.fit(r["h_post"])
    return Fitted(b1=b1, b1_c1=b1_c1, o1=o1, o2=o2, m0x_dims=list(m0x_dims or []))


def score_all(fit: Fitted, h_pre: np.ndarray, h_post: np.ndarray, ids) -> dict[str, np.ndarray]:
    """Scores of core and exploratory methods; keys are method names."""
    out = {
        "B1": fit.b1.score(h_post), "O1": fit.o1.score(h_post, ids, k=5), "O2": fit.o2.score(h_post), "M0": m0(h_pre),
        "B1-C1": fit.b1_c1.score(h_post), "O1-k1": fit.o1.score(h_post, ids, k=1),
        "O1-k10": fit.o1.score(h_post, ids, k=10), "M0-L2": m0_l2(h_pre), "M0-ratio": m0_ratio(h_pre),
    }
    if fit.m0x_dims:
        out["M0-x"] = m0_x(h_pre, fit.m0x_dims)
    return out


FEATURE_STAGE = {"B1": "post_ln_cls_l2", "O1": "post_ln_cls_l2", "O2": "post_ln_cls_l2", "M0": "pre_ln_cls",
                 "B1-C1": "post_ln_cls_l2", "O1-k1": "post_ln_cls_l2", "O1-k10": "post_ln_cls_l2",
                 "M0-L2": "pre_ln_cls", "M0-ratio": "pre_ln_cls", "M0-x": "pre_ln_cls"}

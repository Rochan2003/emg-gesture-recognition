"""Cross-subject domain adaptation so a model trained on some people works on a new person."""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from . import config


def _sym_sqrt(M: np.ndarray, inverse: bool = False, eps: float = 1e-6) -> np.ndarray:
    """Symmetric (inverse) square root of an SPD matrix."""
    M = 0.5 * (M + M.T)
    vals, vecs = np.linalg.eigh(M)
    vals = np.clip(vals, eps, None)
    p = -0.5 if inverse else 0.5
    return (vecs * (vals ** p)) @ vecs.T


class CoralAdapter:
    """CORAL: recolor source features so their covariance matches the target's."""

    def __init__(self, eps: float = 1e-6, mean_align: bool = True):
        self.eps = eps
        self.mean_align = mean_align

    def fit(self, Xs: np.ndarray, Xt: np.ndarray) -> "CoralAdapter":
        Xs = np.asarray(Xs, dtype=float)
        Xt = np.asarray(Xt, dtype=float)
        d = Xs.shape[1]
        Cs = np.cov(Xs, rowvar=False) + self.eps * np.eye(d)
        Ct = np.cov(Xt, rowvar=False) + self.eps * np.eye(d)
        # A = Cs^{-1/2} @ Ct^{1/2}
        self.A_ = _sym_sqrt(Cs, inverse=True, eps=self.eps) @ _sym_sqrt(Ct, eps=self.eps)
        self.mu_s_ = Xs.mean(axis=0)
        self.mu_t_ = Xt.mean(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.mean_align:
            return (X - self.mu_s_) @ self.A_ + self.mu_t_
        return X @ self.A_


def per_subject_standardize(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Z-score each subject's rows by that subject's own mean/std."""
    X = np.asarray(X, dtype=float)
    groups = np.asarray(groups)
    out = np.empty_like(X)
    for g in np.unique(groups):
        m = groups == g
        block = X[m]
        mu = block.mean(axis=0, keepdims=True)
        sd = block.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        out[m] = (block - mu) / sd
    return out


def loso_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    estimator: BaseEstimator,
    adapt: str = "none",
    calibration_k: int = 0,
    scale: bool = True,
    seed: int = config.RANDOM_STATE,
) -> Dict[str, object]:
    """Leave-one-subject-out evaluation with optional domain adaptation.

    adapt is 'none', 'subject_zscore', or 'coral'. If calibration_k > 0, that
    many labeled windows per class move from the target subject into training.
    Returns per_subject accuracies plus mean accuracy, std, and macro f1.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    logo = LeaveOneGroupOut()

    accs: List[float] = []
    f1s: List[float] = []
    per_subject: "OrderedDict[int, float]" = OrderedDict()

    for tr_idx, te_idx in logo.split(X, y, groups):
        Xtr, Xte = X[tr_idx].copy(), X[te_idx].copy()
        ytr, yte = y[tr_idx].copy(), y[te_idx].copy()
        subj = int(groups[te_idx][0])

        # move a few labeled target windows per class into training
        if calibration_k > 0:
            move_local: List[int] = []
            for cls in np.unique(yte):
                cls_local = np.where(yte == cls)[0]
                if len(cls_local) <= calibration_k:
                    continue
                chosen = rng.choice(cls_local, size=calibration_k, replace=False)
                move_local.extend(chosen.tolist())
            if move_local:
                move_mask = np.zeros(len(yte), dtype=bool)
                move_mask[move_local] = True
                Xtr = np.vstack([Xtr, Xte[move_mask]])
                ytr = np.concatenate([ytr, yte[move_mask]])
                Xte, yte = Xte[~move_mask], yte[~move_mask]

        if len(yte) == 0 or len(np.unique(ytr)) < 2:
            continue

        if scale:
            scaler = StandardScaler().fit(Xtr)
            Xtr = scaler.transform(Xtr)
            Xte = scaler.transform(Xte)

        if adapt == "subject_zscore":
            # train uses its pooled stats; test uses its own
            mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True)
            sd[sd == 0] = 1.0
            Xtr = (Xtr - mu) / sd
            mu_t, sd_t = Xte.mean(0, keepdims=True), Xte.std(0, keepdims=True)
            sd_t[sd_t == 0] = 1.0
            Xte = (Xte - mu_t) / sd_t
        elif adapt == "coral":
            adapter = CoralAdapter().fit(Xtr, Xte)  # align train onto the target distribution
            Xtr = adapter.transform(Xtr)
        elif adapt != "none":
            raise ValueError(f"Unknown adapt strategy: {adapt}")

        model = clone(estimator)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        acc = accuracy_score(yte, pred)
        accs.append(acc)
        f1s.append(f1_score(yte, pred, average="macro", zero_division=0))
        per_subject[subj] = float(acc)

    return {
        "per_subject": per_subject,
        "accuracy": float(np.mean(accs)) if accs else float("nan"),
        "accuracy_std": float(np.std(accs)) if accs else float("nan"),
        "f1": float(np.mean(f1s)) if f1s else float("nan"),
        "n_folds": len(accs),
    }


def compare_adaptation(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    estimator: BaseEstimator,
    calibration_k: int = 5,
    seed: int = config.RANDOM_STATE,
) -> "OrderedDict[str, Dict[str, object]]":
    """Run LOSO for none, subject z-score, CORAL, and CORAL plus few-shot calibration."""
    results: "OrderedDict[str, Dict[str, object]]" = OrderedDict()
    results["No adaptation"] = loso_evaluate(X, y, groups, estimator, adapt="none", seed=seed)
    results["Subject z-score"] = loso_evaluate(X, y, groups, estimator, adapt="subject_zscore", seed=seed)
    results["CORAL"] = loso_evaluate(X, y, groups, estimator, adapt="coral", seed=seed)
    results[f"CORAL + {calibration_k}-shot calib"] = loso_evaluate(
        X, y, groups, estimator, adapt="coral", calibration_k=calibration_k, seed=seed
    )
    return results

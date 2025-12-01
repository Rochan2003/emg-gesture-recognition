"""Let the classifier abstain when it is not confident enough."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator

REJECT = -1  # sentinel label for an abstained window


def confidence_scores(proba: np.ndarray, mode: str = "max_prob") -> np.ndarray:
    """Per-sample confidence: top probability (max_prob) or top-two gap (margin)."""
    if mode == "max_prob":
        return proba.max(axis=1)
    if mode == "margin":
        # gap between the two most likely classes
        s = np.sort(proba, axis=1)
        return s[:, -1] - s[:, -2]
    raise ValueError(f"Unknown confidence mode: {mode}")


def predict_with_reject(
    model: BaseEstimator, X: np.ndarray, threshold: float, mode: str = "max_prob"
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict, returning REJECT for samples below the confidence threshold."""
    proba = model.predict_proba(X)
    conf = confidence_scores(proba, mode)
    pred = np.asarray(model.classes_)[np.argmax(proba, axis=1)]
    # abstain wherever confidence falls short of the threshold
    pred = np.where(conf >= threshold, pred, REJECT)
    return pred, conf


def accuracy_coverage_curve(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    mode: str = "max_prob",
    n_points: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep the threshold, returning (thresholds, coverage, accuracy).

    Coverage is the fraction of windows accepted; accuracy is measured on those.
    """
    proba = model.predict_proba(X)
    conf = confidence_scores(proba, mode)
    pred = np.asarray(model.classes_)[np.argmax(proba, axis=1)]
    correct = pred == np.asarray(y)
    # quantile-spaced thresholds give roughly even coverage steps
    thresholds = np.quantile(conf, np.linspace(0.0, 0.95, n_points))
    cov, acc = [], []
    for t in thresholds:
        accept = conf >= t
        cov.append(float(accept.mean()))
        acc.append(float(correct[accept].mean()) if accept.any() else float("nan"))
    return thresholds, np.asarray(cov), np.asarray(acc)


def operating_point(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    target_accuracy: float = 0.95,
    mode: str = "max_prob",
) -> Optional[Dict[str, float]]:
    """Threshold that reaches target_accuracy with the most coverage, or None."""
    thr, cov, acc = accuracy_coverage_curve(model, X, y, mode=mode, n_points=60)
    ok = np.where(acc >= target_accuracy)[0]
    if len(ok) == 0:
        return None
    best = ok[int(np.argmax(cov[ok]))]
    return {"threshold": float(thr[best]), "coverage": float(cov[best]), "accuracy": float(acc[best])}

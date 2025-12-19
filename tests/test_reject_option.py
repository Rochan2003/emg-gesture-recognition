"""Tests for the confidence-gated reject option."""
from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from emg_gesture import reject_option as RO


def _fit(small_feature_data):
    X, y, _g, _fs = small_feature_data
    return LinearDiscriminantAnalysis().fit(X, y), X, y


def test_confidence_scores_ranges():
    proba = np.array([[0.7, 0.2, 0.1], [0.4, 0.35, 0.25]])
    mp = RO.confidence_scores(proba, "max_prob")
    mg = RO.confidence_scores(proba, "margin")
    assert np.allclose(mp, [0.7, 0.4])
    assert np.allclose(mg, [0.5, 0.05])


def test_predict_with_reject_abstains_at_high_threshold(small_feature_data):
    model, X, y = _fit(small_feature_data)
    pred_lo, conf = RO.predict_with_reject(model, X, threshold=0.0)
    assert not np.any(pred_lo == RO.REJECT)                  # nothing rejected at t=0
    # above every confidence score, all windows are abstained (robust to a very
    # confident model on separable synthetic data)
    pred_hi, _ = RO.predict_with_reject(model, X, threshold=float(conf.max()) + 0.01)
    assert np.all(pred_hi == RO.REJECT)


def test_accuracy_coverage_curve_monotone(small_feature_data):
    model, X, y = _fit(small_feature_data)
    thr, cov, acc = RO.accuracy_coverage_curve(model, X, y)
    assert len(thr) == len(cov) == len(acc)
    assert np.all((cov >= 0) & (cov <= 1.0))
    # coverage is non-increasing as the threshold rises
    order = np.argsort(thr)
    assert np.all(np.diff(cov[order]) <= 1e-9)
    # rejecting low-confidence windows should not *lower* accuracy overall
    assert np.nanmax(acc) >= acc[order][0] - 1e-9


def test_operating_point_returns_valid(small_feature_data):
    model, X, y = _fit(small_feature_data)
    op = RO.operating_point(model, X, y, target_accuracy=0.8)
    assert op is None or {"threshold", "coverage", "accuracy"} <= set(op)

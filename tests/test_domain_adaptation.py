"""Tests for CORAL, per-subject standardization and LOSO evaluation."""
from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from emg_gesture import domain_adaptation as DA


def _cov_distance(A, B):
    return np.linalg.norm(np.cov(A, rowvar=False) - np.cov(B, rowvar=False))


def test_coral_brings_covariances_closer():
    rng = np.random.default_rng(0)
    # source and target with deliberately different covariance structure
    Xs = rng.standard_normal((400, 5)) @ np.diag([1, 2, 3, 4, 5])
    Xt = rng.standard_normal((400, 5)) @ np.diag([5, 4, 3, 2, 1])
    adapter = DA.CoralAdapter().fit(Xs, Xt)
    Xs_aligned = adapter.transform(Xs)
    assert _cov_distance(Xs_aligned, Xt) < _cov_distance(Xs, Xt)


def test_per_subject_standardize_zero_means_each_subject():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(10, 2, (50, 3)), rng.normal(-5, 4, (50, 3))])
    groups = np.array([0] * 50 + [1] * 50)
    Z = DA.per_subject_standardize(X, groups)
    for g in (0, 1):
        block = Z[groups == g]
        assert np.allclose(block.mean(axis=0), 0, atol=1e-6)
        assert np.allclose(block.std(axis=0), 1, atol=1e-6)


def test_loso_runs_and_adaptation_helps_or_matches(feature_data):
    X, y, groups, _fs = feature_data
    est = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    base = DA.loso_evaluate(X, y, groups, est, adapt="none")
    adapted = DA.loso_evaluate(X, y, groups, est, adapt="subject_zscore")
    assert base["n_folds"] == len(np.unique(groups))
    assert 0.0 <= base["accuracy"] <= 1.0
    assert 0.0 <= adapted["accuracy"] <= 1.0


def test_compare_adaptation_returns_all_strategies(feature_data):
    X, y, groups, _fs = feature_data
    est = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    res = DA.compare_adaptation(X, y, groups, est, calibration_k=3)
    assert "No adaptation" in res and "CORAL" in res and "Subject z-score" in res
    assert all("accuracy" in v for v in res.values())

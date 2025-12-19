"""Tests for the soft-voting ensemble."""
from __future__ import annotations

import numpy as np

from emg_gesture import ensemble as E


def test_ensemble_has_three_constituents():
    members = E.constituent_models("enhanced")
    assert set(members) == {"SVM-RBF", "KNN", "Logistic Regression"}


def test_ensemble_fits_and_proba_sums_to_one(small_feature_data):
    X, y, _g, _fs = small_feature_data
    ens = E.build_voting_ensemble("enhanced").fit(X, y)
    proba = ens.predict_proba(X)
    assert proba.shape == (len(y), len(np.unique(y)))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert ens.predict(X).shape == y.shape

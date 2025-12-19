"""Tests for the model zoo, per-model enhancements and tuning."""
from __future__ import annotations

import numpy as np
from sklearn.base import clone

from emg_gesture import models as M


def test_get_models_returns_all_variants():
    enhanced = M.get_models("enhanced")
    vanilla = M.get_models("vanilla")
    assert len(enhanced) == 8 == len(vanilla)
    assert "LDA" in enhanced and "HistGradientBoosting" in enhanced


def test_every_model_fits_and_predicts(small_feature_data):
    X, y, _groups, _fs = small_feature_data
    for name, model in M.get_models("enhanced").items():
        m = clone(model).fit(X, y)
        pred = m.predict(X)
        assert pred.shape == y.shape, name
        assert set(np.unique(pred)).issubset(set(np.unique(y))), name


def test_ccp_decision_tree_is_cloneable_and_prunes(small_feature_data):
    X, y, _g, _fs = small_feature_data
    tree = M.CCPDecisionTree(random_state=0, cv=3)
    cloned = clone(tree)  # must support get/set params
    cloned.fit(X, y)
    assert hasattr(cloned, "best_ccp_alpha_")
    assert cloned.best_ccp_alpha_ >= 0
    assert cloned.predict(X).shape == y.shape


def test_cross_validate_models_runs(small_feature_data):
    X, y, _g, _fs = small_feature_data
    # use the two fast models to keep the test snappy
    subset = {k: M.get_models("enhanced")[k] for k in ("LDA", "Gaussian NB")}
    cv = M.cross_validate_models(subset, X, y, cv=3)
    assert set(cv) == set(subset)
    for mean, std in cv.values():
        assert 0.0 <= mean <= 1.0


def test_novelty_table_covers_all_models():
    nt = dict(M.novelty_table())
    assert len(nt) == 8
    assert all(isinstance(v, str) and v for v in nt.values())

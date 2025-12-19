"""Tests for the soft-voting ensemble search."""
from __future__ import annotations

import numpy as np

from emg_gesture import ensemble_search as ES
from emg_gesture import models as M


def _fast_models():
    full = M.get_models("enhanced")
    # a few quick, proba-capable models keep the test snappy
    return {k: full[k] for k in ("LDA", "Gaussian NB", "Random Forest")}


def test_oof_probabilities_shapes(small_feature_data):
    X, y, _g, _fs = small_feature_data
    probas, classes = ES.get_oof_probabilities(_fast_models(), X, y, cv=3)
    n_classes = len(np.unique(y))
    for name, p in probas.items():
        assert p.shape == (len(y), n_classes), name
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6), name
    accs = ES.oof_accuracies(probas, y, classes)
    assert all(0.0 <= a <= 1.0 for a in accs.values())


def test_evaluate_subsets_sorted_and_valid(small_feature_data):
    X, y, _g, _fs = small_feature_data
    probas, classes = ES.get_oof_probabilities(_fast_models(), X, y, cv=3)
    weights = ES.oof_accuracies(probas, y, classes)
    board = ES.evaluate_subsets(probas, y, classes, min_size=2, weighted=True, model_weights=weights)
    # 3 models -> C(3,2)+C(3,3) = 4 subsets
    assert len(board) == 4
    accs = [d["accuracy"] for d in board]
    assert accs == sorted(accs, reverse=True)        # sorted desc
    assert all(0.0 <= a <= 1.0 for a in accs)
    assert all(len(d["members"]) >= 2 for d in board)


def test_build_voting_from_search_fits(small_feature_data):
    X, y, _g, _fs = small_feature_data
    members = ("Random Forest", "LDA")
    ens = ES.build_voting(members).fit(X, y)
    assert ens.predict(X).shape == y.shape
    assert ES.short_label(members) == "RF+LDA"

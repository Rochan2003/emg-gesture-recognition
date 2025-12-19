"""Shared small-data fixtures for fast tests."""
from __future__ import annotations

import numpy as np
import pytest

from emg_gesture.config import Config
from emg_gesture.data_loader import estimate_fs, generate_synthetic_emg
from emg_gesture.features import extract_features
from emg_gesture.preprocessing import preprocess_dataframe


@pytest.fixture(scope="session")
def synthetic_df():
    return generate_synthetic_emg(n_subjects=3, reps_per_gesture=4, duration_s=0.8, seed=0)


@pytest.fixture(scope="session")
def windows_and_labels(synthetic_df):
    cfg = Config()
    cfg.fs = estimate_fs(synthetic_df)
    X_win, y, groups = preprocess_dataframe(synthetic_df, cfg=cfg)
    return X_win, y, groups, cfg.fs


@pytest.fixture(scope="session")
def feature_data(windows_and_labels):
    X_win, y, groups, fs = windows_and_labels
    X = extract_features(X_win, fs=fs)
    return X, y, groups, fs


@pytest.fixture(scope="session")
def small_feature_data(feature_data):
    """A subsampled feature set so model-fitting tests stay fast."""
    X, y, groups, fs = feature_data
    rng = np.random.default_rng(0)
    n = min(300, len(y))
    idx = rng.choice(len(y), size=n, replace=False)
    return X[idx], y[idx], groups[idx], fs

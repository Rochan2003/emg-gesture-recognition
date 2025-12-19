"""Smoke tests: visualization functions render to non-empty PNG files."""
from __future__ import annotations

import numpy as np

from emg_gesture import visualize
from emg_gesture.config import GESTURE_NAMES
from emg_gesture.features import feature_names


def _exists_nonempty(path):
    return path.exists() and path.stat().st_size > 0


def test_raw_and_filtering_plots(synthetic_df, tmp_path):
    p1 = visualize.plot_raw_signals(synthetic_df, tmp_path / "raw.png")
    p2 = visualize.plot_filtering_effect(synthetic_df, 1000.0, tmp_path / "filt.png")
    assert _exists_nonempty(p1) and _exists_nonempty(p2)


def test_class_and_feature_plots(feature_data, tmp_path):
    X, y, groups, _fs = feature_data
    p1 = visualize.plot_class_distribution(y, GESTURE_NAMES, tmp_path / "cls.png")
    p2 = visualize.plot_feature_distributions(
        X, y, feature_names(), GESTURE_NAMES, tmp_path / "feat.png"
    )
    p3 = visualize.plot_subject_variability(X, groups, tmp_path / "subj.png")
    assert all(_exists_nonempty(p) for p in (p1, p2, p3))


def test_psd_and_augmentation_plots(windows_and_labels, tmp_path):
    X_win, y, _g, fs = windows_and_labels
    p1 = visualize.plot_psd_per_gesture(X_win, y, fs, GESTURE_NAMES, tmp_path / "psd.png")
    p2 = visualize.plot_augmentation(X_win[0], tmp_path / "aug.png")
    assert _exists_nonempty(p1) and _exists_nonempty(p2)

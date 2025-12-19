"""Tests for time- and frequency-domain feature extraction."""
from __future__ import annotations

import numpy as np

from emg_gesture import features as F
from emg_gesture.config import N_CHANNELS


def test_feature_vector_length():
    win = np.random.RandomState(0).randn(200, N_CHANNELS)
    vec = F.extract_features_window(win, fs=1000.0)
    assert vec.shape == (N_CHANNELS * F.N_FEATURES_PER_CHANNEL,)
    assert len(F.feature_names(N_CHANNELS)) == vec.shape[0]


def test_feature_matrix_shape():
    W = np.random.RandomState(0).randn(10, 200, N_CHANNELS)
    X = F.extract_features(W, fs=1000.0)
    assert X.shape == (10, N_CHANNELS * F.N_FEATURES_PER_CHANNEL)
    assert np.isfinite(X).all()


def test_mav_and_rms_of_constant():
    x = np.full(200, 3.0)
    mav, rms, wl, zc, ssc, var, iemg, tk = F.time_domain_features(x)
    assert np.isclose(mav, 3.0)
    assert np.isclose(rms, 3.0)
    assert np.isclose(wl, 0.0)      # no waveform length for a constant
    assert np.isclose(var, 0.0)


def test_mean_frequency_tracks_dominant_tone():
    fs = 1000.0
    t = np.arange(256) / fs
    x = np.sin(2 * np.pi * 100 * t)
    mnf, mdf, ttp = F.freq_domain_features(x, fs=fs)
    assert 70 < mnf < 130       # mean frequency near the 100 Hz tone
    assert 70 < mdf < 130
    assert ttp > 0


def test_zero_crossings_increase_with_frequency():
    fs = 1000.0
    t = np.arange(1000) / fs
    low = np.sin(2 * np.pi * 10 * t)
    high = np.sin(2 * np.pi * 100 * t)
    zc_low = F.time_domain_features(low)[3]
    zc_high = F.time_domain_features(high)[3]
    assert zc_high > zc_low

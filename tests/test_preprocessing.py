"""Tests for filtering, normalization, TKEO and segmentation."""
from __future__ import annotations

import numpy as np
import pytest

from emg_gesture import preprocessing as pp


def test_bandpass_clamps_above_nyquist():
    # 450 Hz cutoff is invalid at fs=200; it must clamp (and warn), not crash.
    with pytest.warns(UserWarning):
        sos = pp.design_bandpass(20, 450, fs=200.0)
    assert sos.shape[1] == 6  # second-order-sections format


def test_notch_removes_powerline():
    fs = 1000.0
    t = np.arange(fs) / fs  # 1 s
    sig = np.sin(2 * np.pi * 60 * t) + np.sin(2 * np.pi * 200 * t)
    filt = pp.apply_filters(sig[:, None], fs=fs)[:, 0]

    def power_at(x, f):
        spec = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(len(x), 1 / fs)
        k = int(np.argmin(np.abs(freqs - f)))
        return spec[k]

    # 60 Hz power should drop a lot; 200 Hz (in-band) should survive.
    assert power_at(filt, 60) < 0.3 * power_at(sig, 60)
    assert power_at(filt, 200) > 0.3 * power_at(sig, 200)


def test_normalize_zscore():
    x = np.random.RandomState(0).randn(500, 4) * 5 + 3
    z = pp.normalize(x, "zscore")
    assert np.allclose(z.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(z.std(axis=0), 1, atol=1e-6)


def test_tkeo_shape_and_positive_on_oscillation():
    t = np.arange(1000) / 1000.0
    x = np.sin(2 * np.pi * 50 * t)
    psi = pp.tkeo(x)
    assert psi.shape == x.shape
    # for a sinusoid the TKEO is ~ constant positive (A^2 * w^2)
    assert np.mean(psi[5:-5]) > 0


def test_segmentation_counts_and_purity():
    fs = 1000.0
    n = 2000
    x = np.random.RandomState(1).randn(n, 8)
    labels = np.array([1] * 1000 + [2] * 1000)
    win, step = 200, 100
    W, y = pp.segment_windows(x, labels, win_len=win, step=step, purity=0.75)
    assert W.shape[1:] == (win, 8)
    # only pure windows kept; the boundary-straddling window is dropped
    assert set(np.unique(y)).issubset({1, 2})
    assert W.shape[0] == len(y)


def test_segmentation_drops_unmarked():
    x = np.random.RandomState(2).randn(1000, 8)
    labels = np.zeros(1000, dtype=int)  # all unmarked
    W, y = pp.segment_windows(x, labels, win_len=200, step=100, drop_labels=(0,))
    assert W.shape[0] == 0

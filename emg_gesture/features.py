"""Time- and frequency-domain features per channel."""
from __future__ import annotations

from typing import List

import numpy as np
from scipy import signal as sp_signal

from . import config
from .preprocessing import tkeo

# feature names per channel, in the order produced below
_PER_CHANNEL_FEATURES: List[str] = [
    "MAV", "RMS", "WL", "ZC", "SSC", "VAR", "IEMG", "TKEO", "MNF", "MDF", "TTP",
]
N_FEATURES_PER_CHANNEL = len(_PER_CHANNEL_FEATURES)


def feature_names(n_channels: int = config.N_CHANNELS) -> List[str]:
    """Feature names in the same order as the feature vector."""
    names: List[str] = []
    for ch in range(1, n_channels + 1):
        names.extend(f"ch{ch}_{f}" for f in _PER_CHANNEL_FEATURES)
    return names


# Time-domain features (single channel)
def _zero_crossings(x: np.ndarray, threshold: float) -> int:
    diff_sign = (x[:-1] * x[1:]) < 0
    big_enough = np.abs(x[:-1] - x[1:]) >= threshold
    return int(np.sum(diff_sign & big_enough))


def _slope_sign_changes(x: np.ndarray, threshold: float) -> int:
    d1 = x[1:-1] - x[:-2]
    d2 = x[1:-1] - x[2:]
    cond = (d1 * d2 > 0) & ((np.abs(d1) >= threshold) | (np.abs(d2) >= threshold))
    return int(np.sum(cond))


def time_domain_features(x: np.ndarray) -> List[float]:
    """Eight time-domain features (including mean TKEO energy) for one channel."""
    x = np.asarray(x, dtype=float)
    # dead-zone threshold scaled to the channel so tiny noise doesn't count as crossings
    thr = 1e-3 * (np.std(x) + 1e-12)
    mav = float(np.mean(np.abs(x)))
    rms = float(np.sqrt(np.mean(x ** 2)))
    wl = float(np.sum(np.abs(np.diff(x))))
    zc = float(_zero_crossings(x, thr))
    ssc = float(_slope_sign_changes(x, thr))
    var = float(np.var(x))
    iemg = float(np.sum(np.abs(x)))
    tk = float(np.mean(np.abs(tkeo(x))))
    return [mav, rms, wl, zc, ssc, var, iemg, tk]


# Frequency-domain features (single channel)
def freq_domain_features(x: np.ndarray, fs: float) -> List[float]:
    """Mean freq, median freq and total band power for one channel."""
    x = np.asarray(x, dtype=float)
    nperseg = min(len(x), 256)
    freqs, psd = sp_signal.periodogram(x, fs=fs, nfft=max(nperseg, len(x)))
    total = float(np.sum(psd))
    if total <= 0:
        return [0.0, 0.0, 0.0]
    mnf = float(np.sum(freqs * psd) / total)
    cumulative = np.cumsum(psd)
    mdf_idx = int(np.searchsorted(cumulative, total / 2.0))
    mdf_idx = min(mdf_idx, len(freqs) - 1)
    mdf = float(freqs[mdf_idx])
    ttp = total
    return [mnf, mdf, ttp]


def extract_features_window(window: np.ndarray, fs: float = config.FS_DEFAULT) -> np.ndarray:
    """Feature vector for one (win_len, n_channels) window."""
    window = np.asarray(window, dtype=float)
    if window.ndim == 1:
        window = window[:, None]
    feats: List[float] = []
    for c in range(window.shape[1]):
        ch = window[:, c]
        feats.extend(time_domain_features(ch))
        feats.extend(freq_domain_features(ch, fs))
    # keep features finite so the classifier never sees nan/inf
    return np.nan_to_num(np.asarray(feats, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def extract_features(windows: np.ndarray, fs: float = config.FS_DEFAULT) -> np.ndarray:
    """Feature matrix for a (n_windows, win_len, n_channels) tensor."""
    windows = np.asarray(windows, dtype=float)
    if windows.ndim == 2:  # single window
        windows = windows[None, ...]
    out = np.empty((windows.shape[0], windows.shape[2] * N_FEATURES_PER_CHANNEL))
    for i in range(windows.shape[0]):
        out[i] = extract_features_window(windows[i], fs=fs)
    return out

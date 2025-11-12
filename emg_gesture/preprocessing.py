"""Signal conditioning: filtering, normalization, TKEO, and windowing."""
from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

from . import config
from .data_loader import CHANNEL_COLS


# Filter design
def design_bandpass(
    low: float, high: float, fs: float, order: int = config.BANDPASS_ORDER
) -> np.ndarray:
    """Second-order sections for a Butterworth band-pass.

    The high cutoff is clamped to 0.95 * Nyquist so the design stays valid at
    low sampling rates (e.g. the 200 Hz UCI recordings) instead of raising.
    """
    nyq = 0.5 * fs
    high_eff = min(high, 0.95 * nyq)
    low_eff = max(low, 0.1)
    if high_eff <= low_eff:
        raise ValueError(
            f"Invalid band-pass: low={low_eff} high={high_eff} for fs={fs}"
        )
    if high_eff < high:
        warnings.warn(
            f"Band-pass high cutoff {high} Hz exceeds Nyquist for fs={fs}; "
            f"clamped to {high_eff:.1f} Hz.",
            stacklevel=2,
        )
    sos = sp_signal.butter(order, [low_eff, high_eff], btype="bandpass", fs=fs, output="sos")
    return sos


def design_notch(freq: float, fs: float, q: float = config.NOTCH_Q) -> Tuple[np.ndarray, np.ndarray]:
    """Return (b, a) for an IIR notch at freq Hz, or None if it's above Nyquist."""
    nyq = 0.5 * fs
    if freq >= nyq:
        return None  # type: ignore[return-value]
    b, a = sp_signal.iirnotch(freq, q, fs=fs)
    return b, a


# Filtering
def apply_filters(
    x: np.ndarray,
    fs: float,
    low: float = config.BANDPASS_LOW_HZ,
    high: float = config.BANDPASS_HIGH_HZ,
    order: int = config.BANDPASS_ORDER,
    notch_freqs: Sequence[float] = tuple(config.NOTCH_FREQS_HZ),
    notch_q: float = config.NOTCH_Q,
) -> np.ndarray:
    """Zero-phase band-pass + notch filtering of a (n_samples, n_channels) array."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    sos = design_bandpass(low, high, fs, order)
    # sosfiltfilt needs enough samples vs filter length; bail out if too short
    padlen = 3 * (2 * order + 1)
    if n <= padlen:
        return x.copy()
    y = sp_signal.sosfiltfilt(sos, x, axis=0)
    for f in notch_freqs:
        ba = design_notch(f, fs, notch_q)
        if ba is None:
            continue
        b, a = ba
        y = sp_signal.filtfilt(b, a, y, axis=0)
    return y


def normalize(x: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Per-channel normalization, either zscore (default) or minmax."""
    x = np.asarray(x, dtype=float)
    if method == "zscore":
        mu = x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        return (x - mu) / sd
    if method == "minmax":
        lo = x.min(axis=0, keepdims=True)
        hi = x.max(axis=0, keepdims=True)
        rng = hi - lo
        rng[rng == 0] = 1.0
        return (x - lo) / rng
    raise ValueError(f"Unknown normalization method: {method}")


# Teager-Kaiser Energy Operator (TKEO)
def tkeo(x: np.ndarray) -> np.ndarray:
    """Discrete Teager-Kaiser Energy Operator along axis 0.

    psi[n] = x[n]^2 - x[n-1] * x[n+1]. Emphasizes instantaneous energy, which
    sharpens muscle-activation onset. Same shape as the input.
    """
    x = np.asarray(x, dtype=float)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    psi = np.empty_like(x)
    psi[1:-1] = x[1:-1] ** 2 - x[:-2] * x[2:]
    # copy the neighbors into the two ends so length is preserved
    psi[0] = psi[1]
    psi[-1] = psi[-2]
    return psi[:, 0] if squeeze else psi


def tkeo_envelope(x: np.ndarray, fs: float, smooth_ms: float = 50.0) -> np.ndarray:
    """Smoothed TKEO energy envelope (mean across channels), for onset plots."""
    psi = np.abs(tkeo(x))
    env = psi.mean(axis=1) if psi.ndim == 2 else psi
    win = max(1, int(round(smooth_ms * 1e-3 * fs)))
    kernel = np.ones(win) / win
    return np.convolve(env, kernel, mode="same")


# Segmentation
def segment_windows(
    x: np.ndarray,
    labels: np.ndarray,
    win_len: int,
    step: int,
    purity: float = config.WINDOW_LABEL_PURITY,
    drop_labels: Sequence[int] = (0,),
    onset_gate: bool = False,
    onset_percentile: float = 5.0,
    fs: float = config.FS_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slice a recording into labeled windows.

    A window is kept only if one label covers at least purity of its samples and
    that label isn't in drop_labels (class 0 = unmarked). With onset_gate on, we
    also drop windows below the onset_percentile of TKEO energy; off by default
    since 'rest' is a real class. Returns (windows, win_labels).
    """
    x = np.asarray(x, dtype=float)
    labels = np.asarray(labels)
    n = x.shape[0]
    if n < win_len:
        return np.empty((0, win_len, x.shape[1])), np.empty((0,), dtype=int)

    starts = range(0, n - win_len + 1, step)
    windows: List[np.ndarray] = []
    win_labels: List[int] = []
    energies: List[float] = []
    drop = set(drop_labels)
    for s in starts:
        seg_lab = labels[s : s + win_len]
        vals, counts = np.unique(seg_lab, return_counts=True)
        j = int(np.argmax(counts))
        maj = int(vals[j])
        if counts[j] / win_len < purity:
            continue
        if maj in drop:
            continue
        seg = x[s : s + win_len]
        windows.append(seg)
        win_labels.append(maj)
        if onset_gate:
            energies.append(float(np.abs(tkeo(seg)).mean()))

    if not windows:
        return np.empty((0, win_len, x.shape[1])), np.empty((0,), dtype=int)

    W = np.stack(windows)
    y = np.asarray(win_labels, dtype=int)
    if onset_gate and energies:
        e = np.asarray(energies)
        thresh = np.percentile(e, onset_percentile)
        keep = e >= thresh
        W, y = W[keep], y[keep]
    return W, y


def preprocess_dataframe(
    df: pd.DataFrame,
    cfg: Optional[config.Config] = None,
    fs: Optional[float] = None,
    onset_gate: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter, normalize and window every (subject, trial) recording.

    Windows never cross recording boundaries. Returns (X, y, groups): the
    windows, the gesture label per window, and the subject id per window (the
    last one is what leave-one-subject-out CV groups on).
    """
    cfg = cfg or config.Config()
    if fs is not None:
        cfg.fs = fs
    win_len = cfg.window_length_samples()
    step = cfg.window_step_samples()

    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    g_parts: List[np.ndarray] = []

    for (subject, _trial), grp in df.groupby(["subject", "trial"], sort=False):
        raw = grp[CHANNEL_COLS].to_numpy(dtype=float)
        labels = grp["gesture"].to_numpy()
        filt = apply_filters(
            raw,
            fs=cfg.fs,
            low=cfg.bandpass_low_hz,
            high=cfg.bandpass_high_hz,
            order=cfg.bandpass_order,
            notch_freqs=cfg.notch_freqs_hz,
            notch_q=cfg.notch_q,
        )
        norm = normalize(filt, method=cfg.normalize_method)
        W, yy = segment_windows(
            norm,
            labels,
            win_len=win_len,
            step=step,
            purity=cfg.window_label_purity,
            onset_gate=onset_gate,
            fs=cfg.fs,
        )
        if W.shape[0] == 0:
            continue
        X_parts.append(W)
        y_parts.append(yy)
        g_parts.append(np.full(yy.shape[0], int(subject)))

    if not X_parts:
        raise RuntimeError("Segmentation produced no windows; check fs / window size.")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    groups = np.concatenate(g_parts, axis=0)
    return X, y, groups

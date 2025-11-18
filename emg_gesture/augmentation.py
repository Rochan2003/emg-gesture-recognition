"""Data augmentation for EMG windows (training set only, never the test set).

The main one is electrode shift: rolling the 8 channels fakes the armband being
worn at a slightly different rotation, which helps across subjects/sessions.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from . import config


def electrode_shift(window: np.ndarray, shift: int) -> np.ndarray:
    """Roll a (win_len, n_channels) window around across the channel axis."""
    return np.roll(window, shift, axis=-1)


def jitter(window: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean Gaussian noise (relative to the window's own scale)."""
    scale = std * (np.std(window) + 1e-8)
    return window + rng.normal(0.0, scale, size=window.shape)


def amplitude_scale(window: np.ndarray, factor: float) -> np.ndarray:
    """Scale the whole window's amplitude (electrode-pressure variation)."""
    return window * factor


def augment_windows(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    n_aug: int = 1,
    max_shift: int = 2,
    jitter_std: float = 0.05,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    include_original: bool = True,
    seed: int = config.RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Return an augmented copy of the windowed dataset.

    Each window gets n_aug variants made from a random mix of electrode-shift,
    jitter and amplitude-scale. The originals are kept if include_original is
    True. Returns (X_aug, y_aug, groups_aug), and groups_aug is None if groups
    was None.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    n, _win, n_ch = X.shape

    Xs = [X] if include_original else []
    ys = [y] if include_original else []
    gs = [groups] if (include_original and groups is not None) else []

    # each augmented copy = random channel shift + amplitude scale + jitter
    for _ in range(n_aug):
        out = np.empty_like(X)
        for i in range(n):
            shift = int(rng.integers(-max_shift, max_shift + 1)) % n_ch
            factor = float(rng.uniform(*scale_range))
            w = electrode_shift(X[i], shift)
            w = amplitude_scale(w, factor)
            w = jitter(w, jitter_std, rng)
            out[i] = w
        Xs.append(out)
        ys.append(y)
        if groups is not None:
            gs.append(groups)

    X_aug = np.concatenate(Xs, axis=0)
    y_aug = np.concatenate(ys, axis=0)
    groups_aug = np.concatenate(gs, axis=0) if groups is not None else None
    return X_aug, y_aug, groups_aug

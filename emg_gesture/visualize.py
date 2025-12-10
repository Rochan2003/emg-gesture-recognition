"""Plots for exploring the data and the signal-processing pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import signal as sp_signal  # noqa: E402

from . import augmentation, config  # noqa: E402
from .data_loader import CHANNEL_COLS  # noqa: E402
from .preprocessing import apply_filters, tkeo_envelope  # noqa: E402

_GESTURE_COLORS = plt.get_cmap("tab10")


def plot_raw_signals(
    df: pd.DataFrame,
    path: Path,
    subject: Optional[int] = None,
    trial: Optional[int] = None,
    max_samples: int = 4000,
) -> Path:
    """Plot the raw channels with gesture-colored background shading."""
    if subject is None:
        subject = int(df["subject"].iloc[0])
    sub = df[df["subject"] == subject]
    if trial is None:
        trial = int(sub["trial"].iloc[0])
    sub = sub[sub["trial"] == trial].iloc[:max_samples]

    t = sub["time"].to_numpy()
    fig, axes = plt.subplots(config.N_CHANNELS, 1, figsize=(11, 10), sharex=True)
    gestures = sub["gesture"].to_numpy()
    for ci, col in enumerate(CHANNEL_COLS):
        ax = axes[ci]
        ax.plot(t, sub[col].to_numpy(), lw=0.4, color="black")
        ax.set_ylabel(col, fontsize=8)
        ax.tick_params(labelsize=7)
    # shade each gesture span
    changes = np.where(np.diff(gestures) != 0)[0] + 1
    bounds = [0, *changes.tolist(), len(gestures)]
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        g = int(gestures[b0])
        for ax in axes:
            ax.axvspan(t[b0], t[min(b1, len(t) - 1)], color=_GESTURE_COLORS(g % 10), alpha=0.12)
    axes[-1].set_xlabel("time (ms)")
    fig.suptitle(f"Raw EMG -- subject {subject}, trial {trial} (gesture shading)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_filtering_effect(
    df: pd.DataFrame, fs: float, path: Path, channel: str = "ch1", n_samples: int = 2000
) -> Path:
    """Raw vs filtered signal in time, plus their power spectra."""
    sub = df.iloc[:n_samples]
    raw = sub[channel].to_numpy(dtype=float)
    filt = apply_filters(raw[:, None], fs=fs)[:, 0]
    t = np.arange(len(raw)) / fs

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes[0, 0].plot(t, raw, lw=0.5)
    axes[0, 0].set_title(f"Raw {channel}")
    axes[0, 1].plot(t, filt, lw=0.5, color="#c1440e")
    axes[0, 1].set_title("Band-pass (20-450 Hz) + 50/60 Hz notch")
    for ax in (axes[0, 0], axes[0, 1]):
        ax.set_xlabel("time (s)")

    for ax, sig, label, color in (
        (axes[1, 0], raw, "raw", "#1f77b4"),
        (axes[1, 1], filt, "filtered", "#c1440e"),
    ):
        f, pxx = sp_signal.welch(sig, fs=fs, nperseg=min(len(sig), 512))
        ax.semilogy(f, pxx + 1e-12, color=color)
        ax.set_title(f"PSD ({label})")
        ax.set_xlabel("frequency (Hz)")
        # dotted lines at the 50/60 Hz mains hum the notch removes
        for line_f in (50, 60):
            if line_f < fs / 2:
                ax.axvline(line_f, color="grey", ls=":", alpha=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_class_distribution(
    y: np.ndarray, class_names: Dict[int, str], path: Path
) -> Path:
    vals, counts = np.unique(y, return_counts=True)
    names = [class_names.get(int(v), str(v)) for v in vals]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, counts, color=[_GESTURE_COLORS(int(v) % 10) for v in vals])
    ax.set_ylabel("# windows")
    ax.set_title("Window count per gesture class")
    ax.tick_params(axis="x", rotation=30)
    for i, c in enumerate(counts):
        ax.text(i, c, str(int(c)), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_windowing(
    signal_1d: np.ndarray, fs: float, win_len: int, step: int, path: Path, n_windows: int = 6
) -> Path:
    t = np.arange(len(signal_1d)) / fs
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, signal_1d, lw=0.6, color="black")
    for k in range(n_windows):
        s = k * step
        if s + win_len > len(signal_1d):
            break
        ax.axvspan(t[s], t[s + win_len - 1], color=_GESTURE_COLORS(k % 10), alpha=0.25)
    ax.set_title(f"Sliding windows: {win_len} samples, {int(100*(1-step/win_len))}% overlap")
    ax.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_tkeo_onset(signal_2d: np.ndarray, fs: float, path: Path) -> Path:
    env = tkeo_envelope(signal_2d, fs)
    t = np.arange(signal_2d.shape[0]) / fs
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(t, signal_2d[:, 0], lw=0.5, color="black")
    axes[0].set_title("Filtered channel 1")
    axes[1].plot(t, env, color="#c1440e")
    axes[1].set_title("TKEO energy envelope (sharpened muscle-activation onset)")
    axes[1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_augmentation(window: np.ndarray, path: Path) -> Path:
    shifted = augmentation.electrode_shift(window, shift=2)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    im0 = axes[0].imshow(window.T, aspect="auto", cmap="RdBu_r")
    axes[0].set_title("Original window (channels x time)")
    axes[0].set_ylabel("channel")
    im1 = axes[1].imshow(shifted.T, aspect="auto", cmap="RdBu_r")
    axes[1].set_title("Electrode-shift augmented (rolled 2 channels)")
    for ax in axes:
        ax.set_xlabel("sample")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_psd_per_gesture(
    windows: np.ndarray, y: np.ndarray, fs: float, class_names: Dict[int, str], path: Path, channel: int = 0
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    for g in np.unique(y):
        sigs = windows[y == g][:, :, channel]
        psds = []
        for s in sigs[:200]:
            f, pxx = sp_signal.welch(s, fs=fs, nperseg=min(len(s), 128))
            psds.append(pxx)
        mean_psd = np.mean(psds, axis=0)
        ax.semilogy(f, mean_psd + 1e-12, label=class_names.get(int(g), str(g)))
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title(f"Average power spectrum per gesture (channel {channel + 1})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_feature_distributions(
    X_feat: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    class_names: Dict[int, str],
    path: Path,
    features: Optional[Sequence[str]] = None,
) -> Path:
    if features is None:
        features = ["ch1_MAV", "ch1_RMS", "ch1_MNF", "ch1_WL"]
    idx = [list(feature_names).index(f) for f in features if f in feature_names]
    labels = np.unique(y)
    fig, axes = plt.subplots(1, len(idx), figsize=(4 * len(idx), 4.5))
    if len(idx) == 1:
        axes = [axes]
    for ax, fi in zip(axes, idx):
        # one box per gesture for this feature
        data = [X_feat[y == g, fi] for g in labels]
        ax.boxplot(data, labels=[class_names.get(int(g), str(g)) for g in labels], showfliers=False)
        ax.set_title(feature_names[fi])
        ax.tick_params(axis="x", rotation=45, labelsize=7)
    fig.suptitle("Feature distributions by gesture")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# colors cluster by subject -> shows the cross-subject shift
def plot_subject_variability(
    X_feat: np.ndarray, groups: np.ndarray, path: Path
) -> Path:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # standardize first so PCA isn't dominated by the largest-scale features
    Xs = StandardScaler().fit_transform(X_feat)
    pcs = PCA(n_components=2, random_state=config.RANDOM_STATE).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(8, 6))
    for g in np.unique(groups):
        m = groups == g
        ax.scatter(pcs[m, 0], pcs[m, 1], s=6, alpha=0.5, label=f"subj {int(g)}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Feature space colored by subject\n(clusters by subject => cross-subject shift)")
    if len(np.unique(groups)) <= 12:
        ax.legend(fontsize=7, markerscale=2, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_loso_adaptation(results: Dict[str, Dict[str, object]], path: Path) -> Path:
    names = list(results.keys())
    accs = [float(results[n]["accuracy"]) for n in names]
    stds = [float(results[n]["accuracy_std"]) for n in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, accs, yerr=stds, capsize=4, color="#3a7ca5")
    ax.set_ylabel("LOSO accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Cross-subject (LOSO) accuracy by adaptation strategy")
    ax.tick_params(axis="x", rotation=20)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a, f"{a:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_accuracy_coverage(
    coverage: np.ndarray,
    accuracy: np.ndarray,
    path: Path,
    full_accuracy: Optional[float] = None,
    title: str = "Reject option: accuracy vs coverage",
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    # sort by coverage so the trade-off curve reads left-to-right
    order = np.argsort(coverage)
    ax.plot(coverage[order], accuracy[order], "-o", ms=3, color="#3a7ca5")
    ax.set_xlabel("coverage (fraction of windows the model decides on)")
    ax.set_ylabel("accuracy on accepted windows")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if full_accuracy is not None:
        ax.axhline(full_accuracy, color="#c1440e", ls="--", lw=1.2,
                   label=f"accuracy at 100% coverage ({full_accuracy:.3f})")
        ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_decision_smoothing(
    true: np.ndarray,
    raw: np.ndarray,
    smoothed: np.ndarray,
    class_names: Dict[int, str],
    path: Path,
) -> Path:
    x = np.arange(len(true))
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, sharey=True)
    # top panel: true vs raw per-window; bottom: true vs smoothed
    for ax, pred, label in ((axes[0], raw, "raw (per-window)"), (axes[1], smoothed, "smoothed")):
        ax.step(x, true, where="mid", color="black", lw=2, alpha=0.5, label="true")
        ax.step(x, pred, where="mid", color="#c1440e", lw=1, label=label)
        ax.set_ylabel("gesture")
        ax.legend(fontsize=8, loc="upper right")
    labels = sorted(set(int(v) for v in np.unique(np.concatenate([true, raw, smoothed]))))
    axes[1].set_yticks(labels, [class_names.get(l, str(l)) for l in labels], fontsize=7)
    axes[1].set_xlabel("window index (time)")
    axes[0].set_title("Real-time decision smoothing: fewer spurious gesture flips")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_ensemble_search(
    leaderboard, path: Path, top_n: int = 12, best_single: Optional[float] = None
) -> Path:
    """Bar chart of the top-N searched soft-voting ensembles."""
    from .ensemble_search import short_label

    top = leaderboard[:top_n]
    labels = [short_label(d["members"]) for d in top][::-1]
    accs = [float(d["accuracy"]) for d in top][::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(top) + 1)))
    bars = ax.barh(range(len(top)), accs, color="#3a7ca5")
    ax.set_yticks(range(len(top)), labels, fontsize=8)
    ax.set_xlabel("CV accuracy (train OOF)")
    ax.set_title(f"Top {len(top)} soft-voting ensemble permutations")
    lo = min(accs) - 0.01
    ax.set_xlim(lo, max(accs) + 0.005)
    for b, a in zip(bars, accs):
        ax.text(a, b.get_y() + b.get_height() / 2, f" {a:.3f}", va="center", fontsize=7)
    if best_single is not None:
        ax.axvline(best_single, color="#c1440e", ls="--", lw=1.5,
                   label=f"best single model ({best_single:.3f})")
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_vanilla_vs_enhanced(
    vanilla: Dict[str, float], enhanced: Dict[str, float], path: Path, metric: str = "accuracy"
) -> Path:
    names = list(enhanced.keys())
    v = [vanilla.get(n, np.nan) for n in names]
    e = [enhanced.get(n, np.nan) for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.1), 5))
    ax.bar(x - 0.2, v, 0.4, label="vanilla", color="#9aa5b1")
    ax.bar(x + 0.2, e, 0.4, label="enhanced (novelty)", color="#c1440e")
    ax.set_xticks(x, names, rotation=40, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel(metric)
    ax.set_title(f"Per-model novelty: vanilla vs enhanced ({metric})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path

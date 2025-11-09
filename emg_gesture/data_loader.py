"""Load EMG data into a single DataFrame (real UCI data or a synthetic fallback)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from . import config

CHANNEL_COLS: List[str] = [f"ch{i}" for i in range(1, config.N_CHANNELS + 1)]
CANONICAL_COLS: List[str] = ["time", *CHANNEL_COLS, "gesture", "subject", "trial"]


# Real UCI loader
def load_subject_file(path: Union[str, Path], subject: int, trial: int) -> pd.DataFrame:
    """Parse one tab-separated UCI raw_data file into our standard columns."""
    df = pd.read_csv(path, sep="\t")
    rename = {"class": "gesture"}
    for i in range(1, config.N_CHANNELS + 1):
        rename[f"channel{i}"] = f"ch{i}"
    df = df.rename(columns=rename)
    # some raw files have a stray NaN row (e.g. subject 34), so drop bad rows before casting
    df = df.dropna(subset=["time", *CHANNEL_COLS, "gesture"])
    df["subject"] = subject
    df["trial"] = trial
    df["gesture"] = df["gesture"].astype(int)
    return df[CANONICAL_COLS]


def load_uci_dataset(
    root: Optional[Union[str, Path]] = None,
    subjects: Optional[Sequence[int]] = None,
    max_subjects: Optional[int] = None,
    drop_unmarked: bool = config.DROP_UNMARKED,
    download_if_missing: bool = True,
) -> pd.DataFrame:
    """Load the UCI gesture dataset, optionally limited to some subjects.

    drop_unmarked drops the class-0 transition samples; max_subjects caps how
    many subjects we load to keep runtime down.
    """
    if root is None:
        if download_if_missing:
            from .download_data import ensure_dataset

            root = ensure_dataset()
        else:
            root = config.UCI_DATA_DIR
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"UCI dataset not found at {root}. Run `python -m emg_gesture.download_data`."
        )

    subject_dirs = sorted(
        d for d in root.iterdir() if d.is_dir() and d.name.isdigit()
    )
    available = [int(d.name) for d in subject_dirs]
    if subjects is not None:
        keep = set(subjects)
        subject_dirs = [d for d in subject_dirs if int(d.name) in keep]
    if max_subjects is not None:
        subject_dirs = subject_dirs[:max_subjects]

    frames: List[pd.DataFrame] = []
    for sdir in subject_dirs:
        subject = int(sdir.name)
        for f in sorted(sdir.glob("*.txt")):
            # filenames look like "1_raw_data_...txt", so the leading digit is the trial
            try:
                trial = int(f.name.split("_", 1)[0])
            except ValueError:
                trial = 1
            frames.append(load_subject_file(f, subject=subject, trial=trial))

    if not frames:
        raise RuntimeError(
            f"No subject files parsed from {root} (available subjects: {available})."
        )

    data = pd.concat(frames, ignore_index=True)
    if drop_unmarked:
        data = data[data["gesture"] != 0].reset_index(drop=True)
    return data


def estimate_fs(df: pd.DataFrame, default: float = config.FS_DEFAULT) -> float:
    """Estimate sampling rate (Hz) from the millisecond time column.

    Uses the median inter-sample interval per recording then the median across
    recordings, so a few large gaps don't throw off the estimate.
    """
    if "time" not in df.columns:
        return default
    medians: List[float] = []
    for _, grp in df.groupby(["subject", "trial"], sort=False):
        dt = np.diff(grp["time"].to_numpy(dtype=float))
        dt = dt[dt > 0]
        if dt.size:
            medians.append(float(np.median(dt)))
    if not medians:
        return default
    median_dt_ms = float(np.median(medians))
    if median_dt_ms <= 0:
        return default
    return 1000.0 / median_dt_ms


# Synthetic EMG-like fallback
def generate_synthetic_emg(
    n_subjects: int = 6,
    gestures: Sequence[int] = (1, 2, 3, 4, 5, 6),
    reps_per_gesture: int = 8,
    duration_s: float = 1.5,
    fs: float = config.FS_DEFAULT,
    add_line_noise: bool = True,
    seed: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Generate an EMG-like dataset so the whole pipeline can run without real data.

    Each gesture gets its own spatial pattern across channels and a dominant
    frequency. Each subject gets its own channel gains and a small electrode
    rotation, so there's a real cross-subject shift for LOSO to deal with.
    Optional 50/60 Hz line noise gives the notch filter something to remove.
    """
    rng = np.random.default_rng(seed)
    n_ch = config.N_CHANNELS
    n_gest = len(gestures)
    samples_per_rep = int(round(duration_s * fs))
    t_axis = np.arange(samples_per_rep) / fs

    base_patterns = rng.uniform(0.2, 1.0, size=(n_gest, n_ch))
    # square it so the patterns are peakier and gestures look more distinct
    base_patterns = base_patterns ** 2
    gesture_freqs = np.linspace(60.0, 140.0, n_gest)  # Hz, within EMG band

    rows = []
    time_counter_step_ms = 1000.0 / fs
    for subj in range(1, n_subjects + 1):
        subject_gain = rng.uniform(0.7, 1.3, size=n_ch)
        electrode_shift = int(rng.integers(0, 3))  # pretend the armband is rotated 0-2 channels
        baseline = rng.normal(0.0, 0.02, size=n_ch)
        for trial in (1, 2):
            time_ms = 0.0
            for g_idx, gesture in enumerate(gestures):
                pattern = np.roll(base_patterns[g_idx], electrode_shift) * subject_gain
                f0 = gesture_freqs[g_idx]
                for _ in range(reps_per_gesture):
                    # build each channel as a carrier plus broadband noise to look EMG-ish
                    sig = np.empty((samples_per_rep, n_ch))
                    for c in range(n_ch):
                        carrier = np.sin(2 * np.pi * f0 * t_axis + rng.uniform(0, 2 * np.pi))
                        broadband = rng.standard_normal(samples_per_rep)
                        emg = pattern[c] * (0.4 * carrier + 0.9 * broadband)
                        line = 0.0
                        if add_line_noise:
                            line = 0.15 * np.sin(2 * np.pi * 50.0 * t_axis)
                        sig[:, c] = emg + line + baseline[c] + 0.05 * rng.standard_normal(samples_per_rep)
                    block = pd.DataFrame(sig, columns=CHANNEL_COLS)
                    block.insert(0, "time", time_ms + np.arange(samples_per_rep) * time_counter_step_ms)
                    block["gesture"] = gesture
                    block["subject"] = subj
                    block["trial"] = trial
                    rows.append(block[CANONICAL_COLS])
                    time_ms += samples_per_rep * time_counter_step_ms
    data = pd.concat(rows, ignore_index=True)
    return data


def load_emg(
    data_dir: Optional[Union[str, Path]] = None,
    use_synthetic: bool = False,
    max_subjects: Optional[int] = None,
    subjects: Optional[Sequence[int]] = None,
    drop_unmarked: bool = config.DROP_UNMARKED,
    synthetic_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """Return the real UCI data if available, otherwise the synthetic fallback.

    Pass use_synthetic=True to force the synthetic generator (handy for tests).
    """
    if use_synthetic:
        return generate_synthetic_emg(**(synthetic_kwargs or {}))
    try:
        return load_uci_dataset(
            root=data_dir,
            subjects=subjects,
            max_subjects=max_subjects,
            drop_unmarked=drop_unmarked,
        )
    except (FileNotFoundError, RuntimeError) as exc:  # pragma: no cover - fallback path
        print(f"[data_loader] real dataset unavailable ({exc}); using synthetic fallback.")
        return generate_synthetic_emg(**(synthetic_kwargs or {}))

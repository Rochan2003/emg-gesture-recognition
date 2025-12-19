"""Tests for the data loader and synthetic generator."""
from __future__ import annotations

import numpy as np

from emg_gesture import data_loader as dl
from emg_gesture.config import N_CHANNELS


def test_synthetic_schema(synthetic_df):
    expected = {"time", *dl.CHANNEL_COLS, "gesture", "subject", "trial"}
    assert expected.issubset(set(synthetic_df.columns))
    assert synthetic_df["subject"].nunique() == 3
    # six gestures, no unmarked class in synthetic data
    assert set(synthetic_df["gesture"].unique()) == {1, 2, 3, 4, 5, 6}


def test_synthetic_channels_present(synthetic_df):
    assert all(c in synthetic_df.columns for c in dl.CHANNEL_COLS)
    assert len(dl.CHANNEL_COLS) == N_CHANNELS


def test_estimate_fs_near_1000(synthetic_df):
    fs = dl.estimate_fs(synthetic_df)
    assert 900 < fs < 1100  # synthetic generator uses ~1000 Hz spacing


def test_load_emg_synthetic_fallback():
    df = dl.load_emg(use_synthetic=True, synthetic_kwargs={"n_subjects": 2, "reps_per_gesture": 2})
    assert df["subject"].nunique() == 2
    assert np.isfinite(df[dl.CHANNEL_COLS].to_numpy()).all()


def test_load_subject_file_drops_nan_rows(tmp_path):
    # mirror the real subject-34 quirk: a stray NaN in the class column
    import pandas as pd

    cols = ["time"] + [f"channel{i}" for i in range(1, 9)] + ["class"]
    rows = [[i] + [0.01 * i] * 8 + [1] for i in range(5)]
    rows.append([5] + [0.0] * 8 + [float("nan")])  # bad row
    df = pd.DataFrame(rows, columns=cols)
    path = tmp_path / "1_raw_data_test.txt"
    df.to_csv(path, sep="\t", index=False)

    out = dl.load_subject_file(path, subject=34, trial=1)
    assert out["gesture"].dtype.kind == "i"          # cast to int succeeded
    assert len(out) == 5                              # NaN row dropped
    assert not out[dl.CHANNEL_COLS].isna().any().any()

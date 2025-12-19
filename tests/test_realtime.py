"""Tests for the real-time single-window predictor."""
from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from emg_gesture.config import Config
from emg_gesture.data_loader import CHANNEL_COLS
from emg_gesture.preprocessing import apply_filters
from emg_gesture.realtime import (
    RealTimePredictor,
    StreamSmoother,
    evaluate_stream_smoothing,
    real_time_predict,
)


def _train_simple_model(feature_data):
    X, y, _g, _fs = feature_data
    return LinearDiscriminantAnalysis().fit(X, y), y


def test_real_time_predict_returns_valid_gesture(feature_data, synthetic_df):
    model, y = _train_simple_model(feature_data)
    fs = feature_data[3]
    cfg = Config()
    cfg.fs = fs
    win_len = cfg.window_length_samples()

    rec = synthetic_df[synthetic_df["subject"] == synthetic_df["subject"].iloc[0]]
    rec = rec[rec["trial"] == rec["trial"].iloc[0]]
    raw = rec[CHANNEL_COLS].to_numpy(float)
    filt = apply_filters(raw, fs=fs)
    stats = (filt.mean(axis=0), filt.std(axis=0))

    predictor = RealTimePredictor(model=model, fs=fs, channel_stats=stats)
    raw_window = raw[:win_len]
    label = predictor.predict(raw_window)
    name = real_time_predict(raw_window, predictor)
    assert int(label) in set(np.unique(y).tolist())
    assert isinstance(name, str) and name != ""


def test_predictor_save_load_roundtrip(feature_data, synthetic_df, tmp_path):
    model, _y = _train_simple_model(feature_data)
    fs = feature_data[3]
    predictor = RealTimePredictor(model=model, fs=fs)
    path = predictor.save(tmp_path / "p.joblib")
    loaded = RealTimePredictor.load(path)

    cfg = Config()
    cfg.fs = fs
    raw = synthetic_df[CHANNEL_COLS].to_numpy(float)[: cfg.window_length_samples()]
    assert loaded.predict(raw) == predictor.predict(raw)


def test_stream_smoothing_reduces_flicker(feature_data, synthetic_df):
    model, _y = _train_simple_model(feature_data)
    fs = feature_data[3]
    cfg = Config()
    cfg.fs = fs
    win_len, step = cfg.window_length_samples(), cfg.window_step_samples()

    rec = synthetic_df[synthetic_df["subject"] == synthetic_df["subject"].iloc[0]]
    rec = rec[rec["trial"] == rec["trial"].iloc[0]]
    raw = rec[CHANNEL_COLS].to_numpy(float)
    labels = rec["gesture"].to_numpy()
    filt = apply_filters(raw, fs=fs)
    predictor = RealTimePredictor(
        model=model, fs=fs, channel_stats=(filt.mean(0), filt.std(0))
    )

    windows, true = [], []
    for s in range(0, len(raw) - win_len + 1, step):
        windows.append(raw[s : s + win_len])
        seg = labels[s : s + win_len]
        vals, counts = np.unique(seg, return_counts=True)
        true.append(int(vals[int(np.argmax(counts))]))

    res = evaluate_stream_smoothing(predictor, windows, np.asarray(true), window=7)
    # smoothing should not *increase* spurious transitions
    assert res["smooth_transitions"] <= res["raw_transitions"]
    assert 0.0 <= res["smooth_acc"] <= 1.0


def test_stream_smoother_update_returns_pair(feature_data, synthetic_df):
    model, _y = _train_simple_model(feature_data)
    fs = feature_data[3]
    cfg = Config()
    cfg.fs = fs
    raw = synthetic_df[CHANNEL_COLS].to_numpy(float)[: cfg.window_length_samples()]
    sm = StreamSmoother(RealTimePredictor(model=model, fs=fs), window=5)
    raw_label, smoothed = sm.update(raw)
    assert isinstance(raw_label, int) and isinstance(smoothed, int)

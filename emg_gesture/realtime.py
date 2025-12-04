"""Real-time single-window inference (raw window -> filter -> normalize -> features -> label)."""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np

from . import config
from .features import extract_features_window
from .preprocessing import apply_filters


class RealTimePredictor:
    """Wraps a trained classifier with the full raw-window -> label path."""

    def __init__(
        self,
        model,
        fs: float = config.FS_DEFAULT,
        class_names: Optional[Dict[int, str]] = None,
        channel_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        filter_kwargs: Optional[dict] = None,
    ):
        self.model = model
        self.fs = fs
        self.class_names = class_names or config.GESTURE_NAMES
        self.channel_stats = channel_stats
        self.filter_kwargs = filter_kwargs or {}

    def _to_features(self, raw_window: np.ndarray) -> np.ndarray:
        x = np.asarray(raw_window, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        filt = apply_filters(x, fs=self.fs, **self.filter_kwargs)
        # use calibration stats if we have them, else z-score the window itself
        if self.channel_stats is not None:
            mean, std = self.channel_stats
            std = np.where(std == 0, 1.0, std)
            normed = (filt - mean) / std
        else:
            mu = filt.mean(axis=0, keepdims=True)
            sd = filt.std(axis=0, keepdims=True)
            sd[sd == 0] = 1.0
            normed = (filt - mu) / sd
        return extract_features_window(normed, fs=self.fs)[None, :]

    def predict(self, raw_window: np.ndarray) -> int:
        """Predicted integer gesture label for one raw window."""
        return int(self.model.predict(self._to_features(raw_window))[0])

    def predict_proba(self, raw_window: np.ndarray) -> np.ndarray:
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("Underlying model has no predict_proba.")
        return self.model.predict_proba(self._to_features(raw_window))[0]

    def predict_gesture(self, raw_window: np.ndarray) -> str:
        """Predicted gesture name for one raw window."""
        return self.class_names.get(self.predict(raw_window), "unknown")

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "fs": self.fs,
                "class_names": self.class_names,
                "channel_stats": self.channel_stats,
                "filter_kwargs": self.filter_kwargs,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "RealTimePredictor":
        blob = joblib.load(path)
        return cls(
            model=blob["model"],
            fs=blob["fs"],
            class_names=blob["class_names"],
            channel_stats=blob["channel_stats"],
            filter_kwargs=blob["filter_kwargs"],
        )


def real_time_predict(
    window: np.ndarray, predictor: RealTimePredictor, as_name: bool = True
):
    """Raw window -> predicted gesture (name by default)."""
    return predictor.predict_gesture(window) if as_name else predictor.predict(window)


class StreamSmoother:
    """Smooths a stream of per-window predictions to cut down flicker.

    Single windows flip easily, so we look back over the last `window` of them:
    method="prob" averages the class probabilities then argmax; method="majority"
    takes a majority vote of the recent labels. Costs a little startup latency.
    """

    def __init__(self, predictor: RealTimePredictor, window: int = 5, method: str = "prob"):
        self.predictor = predictor
        self.window = window
        self.method = method
        self._proba_buf: deque = deque(maxlen=window)
        self._label_buf: deque = deque(maxlen=window)

    def reset(self) -> None:
        self._proba_buf.clear()
        self._label_buf.clear()

    def update(self, raw_window: np.ndarray) -> Tuple[int, int]:
        """Push one raw window; return (raw_label, smoothed_label)."""
        proba = self.predictor.predict_proba(raw_window)
        classes = np.asarray(self.predictor.model.classes_)
        raw_label = int(classes[int(np.argmax(proba))])
        self._proba_buf.append(proba)
        self._label_buf.append(raw_label)
        if self.method == "prob":
            # average the recent windows' probabilities, then take the top class
            smoothed = int(classes[int(np.argmax(np.mean(self._proba_buf, axis=0)))])
        elif self.method == "majority":
            vals, counts = np.unique(list(self._label_buf), return_counts=True)
            smoothed = int(vals[int(np.argmax(counts))])
        else:
            raise ValueError(f"Unknown smoothing method: {self.method}")
        return raw_label, smoothed


def evaluate_stream_smoothing(
    predictor: RealTimePredictor,
    raw_windows: List[np.ndarray],
    true_labels: np.ndarray,
    window: int = 5,
    method: str = "prob",
) -> Dict[str, object]:
    """Run a stream raw vs smoothed; report accuracy and transition (flicker) counts."""
    smoother = StreamSmoother(predictor, window=window, method=method)
    raw_preds, smooth_preds = [], []
    for w in raw_windows:
        r, s = smoother.update(w)
        raw_preds.append(r)
        smooth_preds.append(s)
    raw_preds = np.asarray(raw_preds)
    smooth_preds = np.asarray(smooth_preds)
    true = np.asarray(true_labels)

    # count label changes = how much the prediction "flickers"
    def transitions(a: np.ndarray) -> int:
        return int(np.sum(a[1:] != a[:-1])) if len(a) > 1 else 0

    return {
        "raw_acc": float((raw_preds == true).mean()),
        "smooth_acc": float((smooth_preds == true).mean()),
        "raw_transitions": transitions(raw_preds),
        "smooth_transitions": transitions(smooth_preds),
        "true_transitions": transitions(true),
        "raw_preds": raw_preds,
        "smooth_preds": smooth_preds,
        "true": true,
    }

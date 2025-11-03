"""Multi-channel EMG hand-gesture recognition package."""
from __future__ import annotations

__version__ = "1.0.0"

from . import (  # noqa: F401
    augmentation,
    config,
    data_loader,
    domain_adaptation,
    ensemble,
    evaluate,
    features,
    models,
    preprocessing,
    realtime,
    visualize,
)
from .config import Config  # noqa: F401
from .realtime import RealTimePredictor, real_time_predict  # noqa: F401

__all__ = [
    "config",
    "Config",
    "data_loader",
    "preprocessing",
    "features",
    "augmentation",
    "models",
    "ensemble",
    "domain_adaptation",
    "evaluate",
    "visualize",
    "realtime",
    "RealTimePredictor",
    "real_time_predict",
]

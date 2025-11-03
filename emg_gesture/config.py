"""All the tunable constants for the EMG pipeline in one place."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# Paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
# The UCI archive unzips into this sub-folder; the loader also searches for it.
UCI_DATA_DIR: Path = DATA_DIR / "emg_gestures" / "EMG_data_for_gestures-master"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
MODEL_DIR: Path = PROJECT_ROOT / "models_store"

# Signal / acquisition constants
N_CHANNELS: int = 8

# The raw timestamps have ~1ms gaps, so the real rate is closer to 1000 Hz than
# the 200 Hz the dataset is usually quoted as. Loader can still estimate fs.
FS_DEFAULT: float = 1000.0

# Most surface-EMG energy sits in 20-450 Hz.
BANDPASS_LOW_HZ: float = 20.0
BANDPASS_HIGH_HZ: float = 450.0
BANDPASS_ORDER: int = 4

# Notch out both mains frequencies since we don't know where it was recorded.
NOTCH_FREQS_HZ: List[float] = [50.0, 60.0]
NOTCH_Q: float = 30.0

# Windowing
WINDOW_MS: float = 200.0
WINDOW_OVERLAP: float = 0.5  # 50% overlap
# Only keep a window if this fraction of its samples agree on one label.
WINDOW_LABEL_PURITY: float = 0.75

# Labels
GESTURE_NAMES: Dict[int, str] = {
    0: "unmarked",
    1: "rest",
    2: "fist",
    3: "wrist_flexion",
    4: "wrist_extension",
    5: "radial_deviation",
    6: "ulnar_deviation",
    7: "extended_palm",
}
# Class 0 is the transition data between gestures, so drop it by default.
DROP_UNMARKED: bool = True

# Modelling
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.25
CV_FOLDS: int = 5


@dataclass
class Config:
    """Holds all the pipeline settings; defaults come from the constants above."""

    # acquisition
    n_channels: int = N_CHANNELS
    fs: float = FS_DEFAULT

    # filtering
    bandpass_low_hz: float = BANDPASS_LOW_HZ
    bandpass_high_hz: float = BANDPASS_HIGH_HZ
    bandpass_order: int = BANDPASS_ORDER
    notch_freqs_hz: List[float] = field(default_factory=lambda: [50.0, 60.0])
    notch_q: float = NOTCH_Q
    normalize_method: str = "zscore"

    # windowing
    window_ms: float = WINDOW_MS
    window_overlap: float = WINDOW_OVERLAP
    window_label_purity: float = WINDOW_LABEL_PURITY

    # labels
    drop_unmarked: bool = DROP_UNMARKED

    # modelling
    random_state: int = RANDOM_STATE
    test_size: float = TEST_SIZE
    cv_folds: int = CV_FOLDS

    # paths
    data_dir: Path = DATA_DIR
    results_dir: Path = RESULTS_DIR
    model_dir: Path = MODEL_DIR

    def window_length_samples(self) -> int:
        """How many samples are in one window at the current fs."""
        return max(1, int(round(self.window_ms * 1e-3 * self.fs)))

    def window_step_samples(self) -> int:
        """Hop size in samples, based on the overlap fraction."""
        win = self.window_length_samples()
        return max(1, int(round(win * (1.0 - self.window_overlap))))

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.results_dir, self.model_dir):
            Path(d).mkdir(parents=True, exist_ok=True)

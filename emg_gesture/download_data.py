"""Download and unzip the UCI 'EMG Data for Gestures' dataset (ID 481)."""
from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

from . import config

UCI_URL = "https://archive.ics.uci.edu/static/public/481/emg+data+for+gestures.zip"


def ensure_dataset(
    data_dir: Path = config.DATA_DIR,
    url: str = UCI_URL,
    force: bool = False,
) -> Path:
    """Download the dataset if needed and return the EMG_data_for_gestures-master folder.

    Skips the download when the folder is already there (unless force is set).
    """
    data_dir = Path(data_dir)
    target_root = data_dir / "emg_gestures" / "EMG_data_for_gestures-master"
    if target_root.exists() and not force:
        return target_root

    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "emg_gestures.zip"

    if not zip_path.exists() or force:
        print(f"[download_data] fetching {url}")
        urllib.request.urlretrieve(url, zip_path)  # noqa: S310 (trusted UCI host)
        print(f"[download_data] saved {zip_path} ({zip_path.stat().st_size/1e6:.1f} MB)")

    extract_dir = data_dir / "emg_gestures"
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download_data] extracting into {extract_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    if not target_root.exists():
        # Some mirrors nest the files differently, so look for the first dir
        # that has a "01" subject folder inside it.
        for candidate in extract_dir.rglob("*"):
            if candidate.is_dir() and (candidate / "01").exists():
                return candidate
        raise FileNotFoundError(
            f"Could not locate the extracted dataset under {extract_dir}"
        )
    return target_root


if __name__ == "__main__":
    root = ensure_dataset(force="--force" in sys.argv)
    print(f"[download_data] dataset ready at: {root}")

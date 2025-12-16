"""Convenience entry point.

Equivalent to ``python -m emg_gesture.pipeline``.  Run from the project root::

    python main.py                 # real UCI data (downloads if missing)
    python main.py --synthetic     # synthetic fallback
    python main.py --quick         # tiny smoke run
"""
from emg_gesture.pipeline import main

if __name__ == "__main__":
    main()

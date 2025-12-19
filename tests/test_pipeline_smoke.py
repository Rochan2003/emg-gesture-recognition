"""End-to-end smoke test on the tiny synthetic '--quick' configuration."""
from __future__ import annotations

import pandas as pd

from emg_gesture.pipeline import build_argparser, run_pipeline


def test_pipeline_quick_runs(tmp_path):
    args = build_argparser().parse_args(
        ["--quick", "--no-tuning", "--no-augment", "--output-dir", str(tmp_path)]
    )
    out = run_pipeline(args)

    table = out["results_table"]
    assert isinstance(table, pd.DataFrame)
    assert len(table) >= 9            # 8 models + spec ensemble + searched ensemble
    assert "Voting ensemble" in table.index
    assert "Best ensemble (search)" in table.index
    # ensemble search ran over multiple permutations
    assert len(out["ensemble_search"]) >= 2
    assert len(out["best_ensemble"]["members"]) >= 2
    # core artifacts written
    assert (tmp_path / "results_table.csv").exists()
    assert (tmp_path / "RESULTS.md").exists()
    assert (tmp_path / "01_raw_signals.png").exists()
    assert (tmp_path / "16_ensemble_search.png").exists()
    assert (tmp_path / "17_accuracy_coverage.png").exists()
    assert (tmp_path / "18_decision_smoothing.png").exists()
    # reject-option + smoothing summaries produced
    assert "full_accuracy" in out["reject_option"]
    assert {"raw_acc", "smooth_acc"} <= set(out["smoothing"])
    # LOSO ran (3 synthetic subjects in --quick)
    assert out["loso"], "expected LOSO/domain-adaptation results"

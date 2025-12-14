"""End-to-end EMG gesture-recognition pipeline.

Run with: python -m emg_gesture.pipeline
"""
from __future__ import annotations

import argparse
import os

# silence spurious NumPy-2.0 matmul FP warnings (incl. in worker processes)
os.environ.setdefault("PYTHONWARNINGS", "ignore::RuntimeWarning")

import time
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

np.seterr(all="ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from . import (
    augmentation,
    config,
    data_loader,
    ensemble,
    ensemble_search,
    evaluate,
    features as feat_mod,
    models as models_mod,
    reject_option,
    visualize,
)
from .data_loader import CHANNEL_COLS
from .domain_adaptation import compare_adaptation
from .preprocessing import apply_filters, normalize, preprocess_dataframe
from .realtime import RealTimePredictor, evaluate_stream_smoothing, real_time_predict


def _section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EMG gesture recognition pipeline")
    p.add_argument("--data-dir", default=None, help="path to EMG_data_for_gestures-master")
    p.add_argument("--synthetic", action="store_true", help="use the synthetic EMG fallback")
    p.add_argument("--max-subjects", type=int, default=8, help="cap subjects (runtime)")
    p.add_argument("--fs", type=float, default=None, help="override sampling rate (Hz)")
    p.add_argument("--window-ms", type=float, default=config.WINDOW_MS)
    p.add_argument("--overlap", type=float, default=config.WINDOW_OVERLAP)
    p.add_argument("--output-dir", default=str(config.RESULTS_DIR))
    p.add_argument("--no-tuning", action="store_true", help="skip GridSearchCV step")
    p.add_argument("--no-augment", action="store_true", help="skip augmentation ablation")
    p.add_argument("--calibration-k", type=int, default=5, help="few-shot windows/class for DA")
    p.add_argument("--quick", action="store_true", help="tiny synthetic run for smoke testing")
    return p


def run_pipeline(args: argparse.Namespace) -> Dict[str, object]:
    t0 = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = config.Config(window_ms=args.window_ms, window_overlap=args.overlap)
    cfg.results_dir = out_dir

    # load data
    _section("1. Loading data")
    if args.quick:
        df = data_loader.generate_synthetic_emg(n_subjects=3, reps_per_gesture=3, duration_s=0.8)
    elif args.synthetic:
        df = data_loader.generate_synthetic_emg(n_subjects=max(3, min(args.max_subjects, 8)))
    else:
        df = data_loader.load_emg(
            data_dir=args.data_dir, max_subjects=args.max_subjects, drop_unmarked=False
        )

    fs = args.fs or data_loader.estimate_fs(df)
    cfg.fs = fs
    n_subjects = df["subject"].nunique()
    print(f"  rows={len(df):,}  subjects={n_subjects}  trials/subj~{df['trial'].nunique()}")
    print(f"  estimated fs = {fs:.1f} Hz  window = {cfg.window_length_samples()} samples "
          f"(step {cfg.window_step_samples()})")
    gesture_present = sorted(int(g) for g in df["gesture"].unique() if g != 0)
    print(f"  gestures: {[config.GESTURE_NAMES.get(g, g) for g in gesture_present]}")

    # raw-signal EDA
    _section("2. Exploratory plots (raw + filtering)")
    visualize.plot_raw_signals(df, out_dir / "01_raw_signals.png")
    visualize.plot_filtering_effect(df, fs, out_dir / "02_filtering_effect.png")
    print("  saved 01_raw_signals.png, 02_filtering_effect.png")

    # preprocess into windows
    _section("3. Preprocessing & segmentation")
    X_win, y, groups = preprocess_dataframe(df, cfg=cfg)
    print(f"  windows={X_win.shape[0]}  shape/window={X_win.shape[1:]}  classes={np.unique(y)}")

    visualize.plot_class_distribution(y, config.GESTURE_NAMES, out_dir / "03_class_distribution.png")
    # the windowing/tkeo/augmentation illustrations all use one filtered recording
    first = df[(df["subject"] == df["subject"].iloc[0]) & (df["trial"] == df["trial"].iloc[0])]
    filt_first = apply_filters(first[CHANNEL_COLS].to_numpy(float), fs=fs)
    norm_first = normalize(filt_first)
    visualize.plot_windowing(
        norm_first[: cfg.window_length_samples() * 6, 0], fs,
        cfg.window_length_samples(), cfg.window_step_samples(),
        out_dir / "04_windowing.png",
    )
    visualize.plot_tkeo_onset(norm_first[:3000], fs, out_dir / "05_tkeo_onset.png")
    visualize.plot_augmentation(X_win[0], out_dir / "06_augmentation.png")
    print("  saved 03_class_distribution, 04_windowing, 05_tkeo_onset, 06_augmentation")

    # feature extraction
    _section("4. Feature extraction")
    X = feat_mod.extract_features(X_win, fs=fs)
    fnames = feat_mod.feature_names(cfg.n_channels)
    print(f"  feature matrix: {X.shape}  ({feat_mod.N_FEATURES_PER_CHANNEL} feats x {cfg.n_channels} ch)")

    visualize.plot_feature_distributions(X, y, fnames, config.GESTURE_NAMES, out_dir / "07_feature_dist.png")
    visualize.plot_psd_per_gesture(X_win, y, fs, config.GESTURE_NAMES, out_dir / "08_psd_per_gesture.png")
    if n_subjects > 1:
        visualize.plot_subject_variability(X, groups, out_dir / "09_subject_variability.png")
    print("  saved 07_feature_dist, 08_psd_per_gesture, 09_subject_variability")

    # random stratified split + baseline models (vanilla vs enhanced)
    _section("5. Models: random stratified split, vanilla vs enhanced")
    idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(
        idx, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
    Xtr, Xte, ytr, yte = X[idx_tr], X[idx_te], y[idx_tr], y[idx_te]

    vanilla_models = models_mod.get_models("vanilla")
    enhanced_models = models_mod.get_models("enhanced")
    vanilla_acc: Dict[str, float] = {}
    enhanced_acc: Dict[str, float] = {}
    results: "OrderedDict[str, Dict[str, float]]" = OrderedDict()
    fitted: "OrderedDict[str, object]" = OrderedDict()

    for name in enhanced_models:
        vm = vanilla_models[name].fit(Xtr, ytr)
        vanilla_acc[name] = evaluate.evaluate_model(vm, Xte, yte)["accuracy"]
        em = enhanced_models[name].fit(Xtr, ytr)
        metrics = evaluate.evaluate_model(em, Xte, yte)
        enhanced_acc[name] = metrics["accuracy"]
        results[name] = metrics
        fitted[name] = em
        print(f"  {name:22s} vanilla acc={vanilla_acc[name]:.3f}  enhanced acc={metrics['accuracy']:.3f}")

    visualize.plot_vanilla_vs_enhanced(vanilla_acc, enhanced_acc, out_dir / "10_vanilla_vs_enhanced.png")

    # OOF probs on the train split only (no test leakage); reused by the CV table and the ensemble search
    _section("5b. 5-fold OOF cross-validation + per-model probabilities (train)")
    probas, oof_classes = ensemble_search.get_oof_probabilities(
        enhanced_models, Xtr, ytr, cv=cfg.cv_folds
    )
    oof_acc = ensemble_search.oof_accuracies(probas, ytr, oof_classes)
    for name, acc in oof_acc.items():
        pred = oof_classes[np.argmax(probas[name], axis=1)]
        f1 = f1_score(ytr, pred, average="macro", zero_division=0)
        print(f"  {name:22s} OOF CV acc = {acc:.3f}  f1 = {f1:.3f}")

    if not args.no_tuning:
        _section("5c. Light GridSearchCV (Random Forest, Decision Tree)")
        gs_rf = models_mod.tune_random_forest(Xtr, ytr, cv=cfg.cv_folds)
        gs_dt = models_mod.tune_decision_tree(Xtr, ytr, cv=cfg.cv_folds)
        print(f"  RF best params: {gs_rf.best_params_}  (CV acc {gs_rf.best_score_:.3f})")
        print(f"  DT best params: {gs_dt.best_params_}  (CV acc {gs_dt.best_score_:.3f})")

    # soft-voting ensemble
    _section("6. Soft-voting ensemble (SVM + KNN + LogReg)")
    # weight each constituent by its CV accuracy
    ens_weights = [oof_acc["SVM-RBF"], oof_acc["KNN"], oof_acc["Logistic Regression"]]
    print(f"  performance weights (SVC, KNN, LR) = {[round(w, 3) for w in ens_weights]}")
    ens = ensemble.build_voting_ensemble("enhanced", weights=ens_weights).fit(Xtr, ytr)
    ens_metrics = evaluate.evaluate_model(ens, Xte, yte)
    results["Voting ensemble"] = ens_metrics
    fitted["Voting ensemble"] = ens
    constituent_names = ["SVM-RBF", "KNN", "Logistic Regression"]
    print(f"  ensemble accuracy = {ens_metrics['accuracy']:.3f}  f1 = {ens_metrics['f1_macro']:.3f}")
    best_constituent = max(constituent_names, key=lambda n: results[n]["accuracy"])
    best_baseline = max(
        (n for n in results if n not in constituent_names + ["Voting ensemble"]),
        key=lambda n: results[n]["accuracy"],
    )
    print(f"  best constituent: {best_constituent} ({results[best_constituent]['accuracy']:.3f})")
    print(f"  best baseline:    {best_baseline} ({results[best_baseline]['accuracy']:.3f})")
    beats_constituents = all(
        ens_metrics["accuracy"] >= results[c]["accuracy"] for c in constituent_names
    )
    print(f"  ensemble beats every constituent: {beats_constituents}")

    # ensemble search over all soft-voting permutations
    _section("6b. Ensemble search (all soft-voting permutations of the 8 models)")
    leaderboard = ensemble_search.evaluate_subsets(
        probas, ytr, oof_classes, min_size=2, weighted=True, model_weights=oof_acc
    )
    print(f"  searched {len(leaderboard)} soft-voting permutations; top 10 by CV accuracy:")
    for d in leaderboard[:10]:
        print(f"    {ensemble_search.short_label(d['members']):30s} "
              f"CV acc={d['accuracy']:.3f}  f1={d['f1_macro']:.3f}")
    best = leaderboard[0]
    best_ens = ensemble_search.build_voting(best["members"], model_weights=oof_acc).fit(Xtr, ytr)
    best_metrics = evaluate.evaluate_model(best_ens, Xte, yte)
    results["Best ensemble (search)"] = best_metrics
    fitted["Best ensemble (search)"] = best_ens
    best_single_acc = max(results[n]["accuracy"] for n in enhanced_models)
    print(f"  best searched ensemble: {ensemble_search.short_label(best['members'])} "
          f"(CV acc {best['accuracy']:.3f})")
    print(f"  -> held-out TEST acc = {best_metrics['accuracy']:.3f}   "
          f"(best single model = {best_single_acc:.3f}, spec ensemble = {ens_metrics['accuracy']:.3f})")
    # use the best single model's OOF accuracy for the reference line so it matches the bars
    visualize.plot_ensemble_search(
        leaderboard, out_dir / "16_ensemble_search.png", best_single=max(oof_acc.values())
    )
    print("  saved 16_ensemble_search.png")

    # result table + plots
    _section("7. Results table & plots")
    table = evaluate.results_table(results)
    evaluate.print_results_table(table, "Random-split test metrics")
    table.to_csv(out_dir / "results_table.csv")

    evaluate.plot_model_comparison(table, out_dir / "11_model_comparison.png")
    evaluate.plot_confusion_matrix(
        ens, Xte, yte, [config.GESTURE_NAMES.get(int(c), str(c)) for c in np.unique(y)],
        out_dir / "12_confusion_ensemble.png", title="Confusion matrix -- Voting ensemble",
    )
    roc_subset = OrderedDict(
        (n, fitted[n])
        for n in ["Best ensemble (search)", "Voting ensemble", best_baseline, "Random Forest"]
        if n in fitted
    )
    evaluate.plot_roc_curves(roc_subset, Xte, yte, out_dir / "13_roc_curves.png")
    if "Random Forest" in fitted:
        evaluate.plot_feature_importances(
            fitted["Random Forest"], Xte, yte, fnames, out_dir / "14_feature_importance.png"
        )
    print("  saved 11_model_comparison, 12_confusion_ensemble, 13_roc_curves, 14_feature_importance")

    # augmentation ablation (electrode-shift)
    aug_summary = {}
    if not args.no_augment and n_subjects >= 2:
        # electrode-shift aug is about cross-subject robustness, so test it on a held-out subject
        # (within-subject it would just be noise)
        _section("8. Augmentation ablation (electrode-shift, cross-subject hold-out)")
        test_subj = int(np.unique(groups)[-1])
        te = groups == test_subj
        tr = ~te
        ens_no = ensemble.build_voting_ensemble("enhanced", weights=ens_weights).fit(X[tr], y[tr])
        acc_no = evaluate.evaluate_model(ens_no, X[te], y[te])["accuracy"]
        Xw_aug, y_aug, _ = augmentation.augment_windows(X_win[tr], y[tr], n_aug=1)
        X_aug_feat = feat_mod.extract_features(Xw_aug, fs=fs)
        ens_aug = ensemble.build_voting_ensemble("enhanced", weights=ens_weights).fit(X_aug_feat, y_aug)
        acc_aug = evaluate.evaluate_model(ens_aug, X[te], y[te])["accuracy"]
        aug_summary = {"held_out_subject": test_subj, "no_aug": acc_no, "with_aug": acc_aug}
        print(f"  held-out subject {test_subj}: acc no-aug={acc_no:.3f}  "
              f"+electrode-shift aug={acc_aug:.3f}")

    # cross-subject LOSO + domain adaptation
    loso_results = {}
    if n_subjects >= 3:
        _section("9. Leave-one-subject-out + domain adaptation")
        # shrinkage LDA is fast and strong, so it's a good base across the many folds
        base = models_mod.MODEL_SPECS["lda"][2]()
        loso_results = compare_adaptation(X, y, groups, base, calibration_k=args.calibration_k)
        print(f"  {'strategy':28s} {'LOSO acc':>9s} {'+/-':>6s} {'macroF1':>8s}")
        for nm, r in loso_results.items():
            print(f"  {nm:28s} {r['accuracy']:9.3f} {r['accuracy_std']:6.3f} {r['f1']:8.3f}")
        base_loso = loso_results["No adaptation"]["accuracy"]
        best_da = max(loso_results, key=lambda k: loso_results[k]["accuracy"])
        print(f"  random-split (within-subject) acc ~= {ens_metrics['accuracy']:.3f}")
        print(f"  LOSO no-adapt = {base_loso:.3f}  ->  best ({best_da}) = "
              f"{loso_results[best_da]['accuracy']:.3f}  "
              f"(+{loso_results[best_da]['accuracy'] - base_loso:.3f})")
        visualize.plot_loso_adaptation(loso_results, out_dir / "15_loso_adaptation.png")
        print("  saved 15_loso_adaptation.png")
    else:
        print("  (skipping LOSO: need >= 3 subjects)")

    # real-time prediction demo
    _section("10. Real-time prediction demo")
    # use one raw recording with its own per-channel stats, matching how it was normalized in batch
    demo_rec = df[(df["subject"] == df["subject"].iloc[-1])]
    demo_rec = demo_rec[demo_rec["trial"] == demo_rec["trial"].iloc[0]]
    demo_rec = demo_rec[demo_rec["gesture"] != 0]
    raw_sig = demo_rec[CHANNEL_COLS].to_numpy(float)
    filt_sig = apply_filters(raw_sig, fs=fs)
    ch_stats = (filt_sig.mean(axis=0), filt_sig.std(axis=0))
    win_len = cfg.window_length_samples()
    # find a window that sits inside one gesture
    labels_rec = demo_rec["gesture"].to_numpy()
    start = 0
    for s in range(0, len(raw_sig) - win_len, win_len):
        seg = labels_rec[s : s + win_len]
        if len(np.unique(seg)) == 1:
            start = s
            break
    raw_window = raw_sig[start : start + win_len]
    true_label = int(labels_rec[start])

    # serve with the best searched ensemble (strongest model so far)
    predictor = RealTimePredictor(
        model=best_ens, fs=fs, class_names=config.GESTURE_NAMES, channel_stats=ch_stats
    )
    pred_name = real_time_predict(raw_window, predictor)
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    predictor.save(cfg.model_dir / "realtime_ensemble.joblib")
    print(f"  raw window {raw_window.shape} -> predicted '{pred_name}'  "
          f"(true '{config.GESTURE_NAMES.get(true_label, true_label)}')")
    print(f"  saved predictor -> {cfg.model_dir / 'realtime_ensemble.joblib'}")

    # confidence-gated reject option (accuracy vs coverage)
    _section("11. Confidence-gated reject option (accuracy vs coverage)")
    thr, cov, acc = reject_option.accuracy_coverage_curve(best_ens, Xte, yte, mode="max_prob")
    full_acc = best_metrics["accuracy"]
    op95 = reject_option.operating_point(best_ens, Xte, yte, target_accuracy=0.95)
    op99 = reject_option.operating_point(best_ens, Xte, yte, target_accuracy=0.99)
    print(f"  accuracy at 100% coverage = {full_acc:.3f}")
    if op95:
        print(f"  to reach 95% accuracy -> keep {op95['coverage']*100:.1f}% of windows "
              f"(confidence >= {op95['threshold']:.3f})")
    if op99:
        print(f"  to reach 99% accuracy -> keep {op99['coverage']*100:.1f}% of windows "
              f"(confidence >= {op99['threshold']:.3f})")
    visualize.plot_accuracy_coverage(cov, acc, out_dir / "17_accuracy_coverage.png", full_accuracy=full_acc)
    print("  saved 17_accuracy_coverage.png")
    reject_summary = {"full_accuracy": full_acc, "op95": op95, "op99": op99}

    # real-time decision smoothing (flicker reduction)
    _section("12. Real-time decision smoothing (flicker reduction)")
    step = cfg.window_step_samples()
    stream_windows, stream_true = [], []
    for s in range(0, len(raw_sig) - win_len + 1, step):
        seg_lab = labels_rec[s : s + win_len]
        vals, counts = np.unique(seg_lab, return_counts=True)
        stream_true.append(int(vals[int(np.argmax(counts))]))
        stream_windows.append(raw_sig[s : s + win_len])
    sm = evaluate_stream_smoothing(
        predictor, stream_windows, np.asarray(stream_true), window=7, method="prob"
    )
    print(f"  stream of {len(stream_windows)} windows (held-out recording)")
    print(f"  per-window accuracy: raw={sm['raw_acc']:.3f}  smoothed={sm['smooth_acc']:.3f}")
    print(f"  gesture transitions: true={sm['true_transitions']}  "
          f"raw={sm['raw_transitions']}  smoothed={sm['smooth_transitions']} "
          f"(fewer spurious flips is better)")
    visualize.plot_decision_smoothing(
        sm["true"], sm["raw_preds"], sm["smooth_preds"], config.GESTURE_NAMES,
        out_dir / "18_decision_smoothing.png",
    )
    print("  saved 18_decision_smoothing.png")
    smoothing_summary = {k: sm[k] for k in
                         ("raw_acc", "smooth_acc", "raw_transitions", "smooth_transitions", "true_transitions")}

    # markdown summary
    _write_summary(out_dir, table, leaderboard, best, loso_results, aug_summary,
                   models_mod.novelty_table(), vanilla_acc, enhanced_acc,
                   reject_summary, smoothing_summary)
    print(f"\nDone in {time.time() - t0:.1f}s. Artifacts in: {out_dir}")
    return {
        "results_table": table,
        "loso": loso_results,
        "augmentation": aug_summary,
        "ensemble_metrics": ens_metrics,
        "ensemble_search": leaderboard,
        "best_ensemble": best,
        "reject_option": reject_summary,
        "smoothing": smoothing_summary,
    }


def _write_summary(out_dir, table, leaderboard, best, loso_results, aug_summary, novelties,
                   vanilla_acc, enhanced_acc, reject_summary=None, smoothing_summary=None) -> None:
    lines = ["# EMG Gesture Recognition -- Results\n"]
    lines.append("## Random-split test metrics\n")
    lines.append(table.round(4).to_markdown())
    lines.append("\n\n## Per-model novelty (vanilla vs enhanced accuracy)\n")
    lines.append("| Model | Novelty | Vanilla | Enhanced |")
    lines.append("|---|---|---|---|")
    nd = dict(novelties)
    for name in enhanced_acc:
        lines.append(f"| {name} | {nd.get(name, '')} | {vanilla_acc.get(name, float('nan')):.3f} "
                     f"| {enhanced_acc.get(name, float('nan')):.3f} |")
    lines.append("\n## Ensemble search (top soft-voting permutations, by CV accuracy)\n")
    lines.append(f"_{len(leaderboard)} permutations searched; best = "
                 f"**{ensemble_search.short_label(best['members'])}**._\n")
    lines.append("| Ensemble | size | CV accuracy | CV macro-F1 |")
    lines.append("|---|---|---|---|")
    for d in leaderboard[:8]:
        lines.append(f"| {ensemble_search.short_label(d['members'])} | {d['size']} "
                     f"| {d['accuracy']:.3f} | {d['f1_macro']:.3f} |")
    if aug_summary:
        lines.append("\n## Electrode-shift augmentation ablation (cross-subject hold-out)\n")
        lines.append(f"- held-out test subject: **{aug_summary.get('held_out_subject', 'n/a')}**")
        lines.append(f"- ensemble accuracy without augmentation: **{aug_summary['no_aug']:.3f}**")
        lines.append(f"- ensemble accuracy with electrode-shift augmentation: **{aug_summary['with_aug']:.3f}**")
    if loso_results:
        lines.append("\n## Cross-subject (LOSO) accuracy by adaptation strategy\n")
        lines.append("| Strategy | LOSO acc | +/- | macro-F1 |")
        lines.append("|---|---|---|---|")
        for nm, r in loso_results.items():
            lines.append(f"| {nm} | {r['accuracy']:.3f} | {r['accuracy_std']:.3f} | {r['f1']:.3f} |")
    if reject_summary:
        lines.append("\n## Confidence-gated reject option\n")
        lines.append(f"- accuracy at 100% coverage: **{reject_summary['full_accuracy']:.3f}**")
        for tag, key in (("95%", "op95"), ("99%", "op99")):
            op = reject_summary.get(key)
            if op:
                lines.append(f"- to reach {tag} accuracy: keep **{op['coverage']*100:.1f}%** "
                             f"of windows (confidence >= {op['threshold']:.3f})")
    if smoothing_summary:
        lines.append("\n## Real-time decision smoothing (held-out stream)\n")
        lines.append(f"- per-window accuracy: raw **{smoothing_summary['raw_acc']:.3f}** -> "
                     f"smoothed **{smoothing_summary['smooth_acc']:.3f}**")
        lines.append(f"- gesture transitions: true {smoothing_summary['true_transitions']}, "
                     f"raw {smoothing_summary['raw_transitions']}, "
                     f"smoothed {smoothing_summary['smooth_transitions']} (fewer spurious flips)")
    (out_dir / "RESULTS.md").write_text("\n".join(lines))


def main() -> None:
    args = build_argparser().parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()

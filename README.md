# Multi-Channel EMG Hand-Gesture Recognition

Classify hand gestures from 8-channel surface EMG (sEMG) time-series, compare
several classical ML models, and search for a soft-voting ensemble that beats them.
Use case: myoelectric prosthetic / gesture-based HCI control.

```bash
python -m emg_gesture.pipeline          # full run on the real UCI dataset
```

---

## Beyond the baseline pipeline

The basic recipe (band-pass → window → time/freq features → RF/LR/NB/DT → voting)
is standard. On top of that baseline this project adds four extensions aimed at the
harder parts of EMG:

### 1. Cross-subject domain adaptation
EMG models trained on some people generalize poorly to a **new** person
(different anatomy, skin impedance, electrode placement). The leave-one-subject-out
(**LOSO**) accuracy is therefore far below the within-subject random-split number —
this is *the* open problem in practical EMG interfaces. We narrow the gap with
three label-free / few-label strategies (`emg_gesture/domain_adaptation.py`):

| Strategy | Idea | Target labels needed |
|---|---|---|
| **Subject z-score** | standardize each subject's features by their own mean/std | none |
| **CORAL** | re-color the training feature covariance to match the new subject's (Sun et al., 2016) | none (unlabeled) |
| **Few-shot calibration** | fold *k* labeled windows/class from the new subject into training | k per class |

The cross-subject feature shift is visualized directly in
`results/09_subject_variability.png` (a PCA scatter that clusters by subject), and
the LOSO lift is reported in `results/15_loso_adaptation.png`.

### 2. One significant, principled enhancement per model
Every classifier is run **vanilla vs. enhanced**, where each enhancement fixes a
weakness that model has *specifically on EMG feature vectors* — not a grab-bag of
tricks. The pipeline reports the accuracy of both so each novelty has to earn its
place (`results/10_vanilla_vs_enhanced.png`).

| Model | Enhancement | Why it helps EMG |
|---|---|---|
| Logistic Regression | Elastic-net multinomial (CV-tuned L1/L2) | embedded feature selection on the 88-D vector |
| Decision Tree | Cost-complexity pruning (CV-selected `ccp_alpha`) | principled regularization vs. a guessed `max_depth` |
| Random Forest | Balanced sub-sampling + tuned forest; permutation importance | impurity importance is biased on continuous features |
| Gaussian NB | Yeo-Johnson power transform | repairs GNB's per-feature Gaussian assumption (EMG power is skewed) |
| LDA | Ledoit-Wolf shrinkage covariance | robust on correlated, high-dimensional channels |
| SVM-RBF | Probability calibration (`CalibratedClassifierCV`) | uncalibrated SVC probabilities corrupt the soft vote |
| KNN | NCA-learned metric | neighbors defined in a discriminative subspace |
| HistGradientBoosting | early stopping + class weighting | regularized, imbalance-aware |

### 3. EMG-specific signal processing
Beyond the required band-pass + notch + normalization:
- **TKEO** (Teager-Kaiser Energy Operator) — sharpens muscle-activation onset; used
  as an extra feature *and* an optional onset gate (`results/05_tkeo_onset.png`).
- **Electrode-shift augmentation** — circularly rotating the 8 channels simulates
  the armband being worn rotated, a dominant source of cross-subject variability.
  An ablation reports the ensemble accuracy with/without it
  (`results/06_augmentation.png`).

### 4. Deployment-oriented real-time inference
Two features for using the model in a live stream, not just on a benchmark:
- **Confidence-gated reject option** (`reject_option.py`) — the classifier abstains
  ("uncertain") when its top-class probability is low, trading **coverage** for
  **accuracy**. The pipeline traces the accuracy–coverage curve and reports, e.g.,
  "to hit 99% accuracy, keep X% of windows" (`results/17_accuracy_coverage.png`).
  In a prosthetic, *no action* beats a *wrong action*.
- **Real-time decision smoothing** (`StreamSmoother` in `realtime.py`) — averages
  class probabilities over the last *N* windows so a single noisy window can't flip
  the decision. Reported as the reduction in spurious gesture transitions vs. the
  raw per-window trace (`results/18_decision_smoothing.png`).

---

## Dataset

**UCI "EMG Data for Gestures" (ID 481)** — a Myo Thalmic bracelet, 8 EMG channels,
36 subjects × 2 sessions, gestures: rest, fist, wrist flexion, wrist extension,
radial deviation, ulnar deviation (+ extended palm for some subjects). Class 0 is
unmarked transition data and is dropped at the window level.

- **Auto-download**: `python -m emg_gesture.download_data` (≈17 MB zip). The loader
  also calls this automatically if the data is missing.
- **Configurable path**: point the loader at any 8-channel EMG CSV/TSV folder via
  `--data-dir`, or call `data_loader.load_uci_dataset(root=...)`.
- **Synthetic fallback**: if no dataset is present, `generate_synthetic_emg()`
  produces an EMG-like dataset (distinct per-gesture spatial + spectral patterns,
  per-subject gains + electrode rotation) so the entire pipeline still runs
  end-to-end. Use `--synthetic`.

> **Sampling-rate note.** This set is widely cited as 200 Hz, but the millisecond
> timestamps have a **median delta of 1 ms ⇒ an effective ~1000 Hz**. The loader
> estimates `fs` from the data, and the band-pass design **clamps its upper cutoff
> to the Nyquist frequency**, so the 20–450 Hz band stays valid here *and* on a
> genuine 200 Hz CSV (where it auto-clamps to ~95 Hz).

---

## Project structure

```
Multi channel EMG Classifier/
├── emg_gesture/
│   ├── config.py             # all tunable constants + Config dataclass
│   ├── download_data.py      # fetch + unpack the UCI dataset (idempotent)
│   ├── data_loader.py        # UCI parser, fs estimation, synthetic generator
│   ├── preprocessing.py      # Butterworth band-pass + notch, normalize, TKEO, windowing
│   ├── augmentation.py       # electrode-shift / jitter / amplitude-scale augmentation
│   ├── features.py           # 11 time+freq features/channel → 88-D vector/window
│   ├── models.py             # 8 models, vanilla + enhanced, CV, GridSearchCV
│   ├── ensemble.py           # soft-voting SVC+KNN+LogReg (performance-weighted)
│   ├── ensemble_search.py    # search all soft-voting permutations for the best combo
│   ├── domain_adaptation.py  # CORAL, per-subject z-score, few-shot, LOSO eval
│   ├── reject_option.py      # confidence-gated abstention + accuracy-coverage curve
│   ├── evaluate.py           # metrics + confusion/ROC/comparison/importance plots
│   ├── visualize.py          # EDA + signal-processing + novelty visualizations
│   ├── realtime.py           # real_time_predict + StreamSmoother (decision smoothing)
│   └── pipeline.py           # end-to-end orchestration (python -m emg_gesture.pipeline)
├── tests/                    # pytest suite (preprocessing, features, models, DA, ensemble, realtime, e2e)
├── requirements.txt
└── README.md
```

---

## Pipeline stages (`pipeline.py`)

1. **Load** real or synthetic data; estimate `fs`.
2. **EDA plots** — raw 8-channel traces with gesture shading; filtering effect (time + spectrum).
3. **Preprocess** — band-pass (20–450 Hz) → 50/60 Hz notch → per-channel z-score →
   200 ms / 50 %-overlap windows, each labeled by majority vote with a purity gate.
4. **Features** — 88-D vector per window; feature-distribution, per-gesture PSD, and
   cross-subject PCA plots.
5. **Models** — vanilla vs. enhanced on a stratified split; 5-fold CV; light
   GridSearchCV for RF & Decision Tree.
6. **Ensemble** — performance-weighted soft-voting SVC+KNN+LogReg.
   **6b. Ensemble search** — score all 247 soft-voting permutations of the 8 models
   by CV (using cached out-of-fold probabilities), select the best, report its
   held-out test accuracy.
7. **Plots** — model-comparison bar chart, confusion matrix, ROC curves, RF
   permutation importances, ensemble-search leaderboard.
8. **Augmentation ablation** — *cross-subject* (held-out-subject) test of
   electrode-shift augmentation.
9. **LOSO + domain adaptation** — subject-independent accuracy for each adaptation strategy.
10. **Real-time demo** — one raw window → `real_time_predict` → gesture; predictor saved with joblib.
11. **Reject option** — accuracy–coverage curve + operating points (e.g. coverage at 95/99% accuracy).
12. **Decision smoothing** — held-out stream, raw vs. smoothed predictions + transition (flicker) counts.

---

## Quick start

```bash
# from the project root (a .venv already exists)
pip install -r requirements.txt

# (optional) pre-fetch the dataset; the pipeline does this automatically otherwise
python -m emg_gesture.download_data

# run on the real data (--max-subjects defaults to 8; pass more for a bigger run)
python -m emg_gesture.pipeline --max-subjects 10

# faster / data-free options
python -m emg_gesture.pipeline --synthetic      # synthetic EMG, no download
python -m emg_gesture.pipeline --quick           # tiny synthetic smoke run
python -m emg_gesture.pipeline --max-subjects 36 # full dataset (slower)

# tests
pytest
```

Useful flags: `--data-dir PATH`, `--fs HZ`, `--window-ms`, `--overlap`,
`--no-tuning`, `--no-augment`, `--calibration-k K`, `--output-dir DIR`.

All artifacts (plots, `results_table.csv`, `RESULTS.md`) are written to `results/`.

### Real-time inference

```python
from emg_gesture.realtime import RealTimePredictor, real_time_predict
predictor = RealTimePredictor.load("models_store/realtime_ensemble.joblib")
gesture = real_time_predict(raw_window, predictor)   # raw (200, 8) window → "fist"
```

---

## Results

> Numbers below are from a 10-subject run (`--max-subjects 10`, ~3.8k windows).
> Re-running regenerates `results/RESULTS.md`, `results/results_table.csv`, and all plots.
> Full-dataset (36-subject) runs follow the same trends.

<!-- RESULTS_START -->

**Random-split test metrics** (10 subjects, ~3.8k windows, 6 gestures):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Best ensemble (search: RF+SVM+HGB)** | **0.945** | 0.945 | 0.945 | 0.945 | **0.996** |
| SVM-RBF (calibrated) | 0.940 | 0.940 | 0.941 | 0.940 | 0.995 |
| HistGradientBoosting | 0.933 | 0.934 | 0.933 | 0.933 | 0.995 |
| Voting ensemble (SVC+KNN+LR) | 0.932 | 0.933 | 0.932 | 0.932 | 0.995 |
| Random Forest | 0.924 | 0.924 | 0.924 | 0.924 | 0.993 |
| Logistic Regression | 0.903 | 0.904 | 0.903 | 0.903 | 0.991 |
| KNN | 0.898 | 0.900 | 0.898 | 0.898 | 0.983 |
| LDA | 0.875 | 0.878 | 0.875 | 0.876 | 0.986 |
| Decision Tree | 0.873 | 0.877 | 0.874 | 0.873 | 0.968 |
| Gaussian NB | 0.849 | 0.854 | 0.849 | 0.850 | 0.977 |

**The searched ensemble beats every individual model and the spec ensemble.** The
spec's `SVC+KNN+LR` voting ensemble (0.932) actually tracks *under* the lone
calibrated SVM (0.940) because its three members make correlated errors. Searching
all 247 soft-voting permutations (selected by CV, scored on the held-out test set)
surfaces **`RF + SVM + HGB`** — a kernel learner + two tree learners whose
*uncorrelated* errors the average can correct — at **0.945**, the top model overall.

**Per-model novelty** — most enhancements help; the wins are Gaussian NB **+0.041**
(Yeo-Johnson), SVM **+0.028** (calibration), KNN **+0.015** (NCA). A few are flat or
slightly negative on this split (expected — reported honestly, see `RESULTS.md`).

**Cross-subject (LOSO) + domain adaptation** — the core hard problem:

| Strategy | LOSO accuracy | macro-F1 |
|---|---|---|
| No adaptation | 0.806 | 0.791 |
| **Subject z-score** | **0.827** | **0.815** |
| CORAL | 0.768 | 0.766 |
| CORAL + 5-shot calibration | 0.778 | 0.776 |

Within-subject accuracy is ~0.93–0.95 but subject-independent (LOSO) accuracy drops
to ~0.81 — quantifying the cross-subject gap. **Per-subject standardization** is the
most effective label-free adaptation here (**+0.020**); plain CORAL underperforms on
this 10-subject subset (an honest negative — covariance alignment is noisier with
few subjects).

**Confidence-gated reject option** — abstaining on the least-confident windows trades
coverage for accuracy:

| Target accuracy | Coverage kept | Confidence threshold |
|---|---|---|
| 0.945 (no reject) | 100% | — |
| 0.95 | 98.3% | 0.530 |
| **0.99** | **67.8%** | 0.898 |

Dropping the least-confident ~32% of windows lifts accuracy to **99%** — exactly the
knob a prosthetic controller wants (a wrong action is worse than no action).

**Real-time decision smoothing** (held-out stream of 204 windows, prob-averaging over
the last 7): spurious gesture flips fall from **18 → 11**, *exactly* recovering the
ground-truth transition count (11). Per-window accuracy dips 0.961 → 0.853 — the
expected stability/latency trade-off (the smoother lags a few windows at each gesture
boundary; see `results/18_decision_smoothing.png`).

### Scale-up: full 36-subject run (`results_full36/`)

Running all **36 subjects** (14,245 windows, **7 classes** including the rare
`extended_palm`) confirms the conclusions hold at scale and surfaces a ranking shift:

| Model | 10-subj acc (6-class) | 36-subj acc (7-class) |
|---|---|---|
| **Best ensemble (search)** | **0.945** | **0.951** |
| HistGradientBoosting | 0.933 | 0.944 |
| Random Forest | 0.924 | 0.942 |
| Voting ensemble (spec) | 0.932 | 0.934 |
| SVM-RBF | 0.940 | 0.928 |

- **The ensemble search still wins** at full scale (0.951, best F1 and ROC-AUC).
- **Ranking flips:** the calibrated SVM led at 10 subjects but the **tree models overtake
  it at 36** — they exploit the larger/more varied training set better than the RBF kernel.
- The spec ensemble's macro-F1 sags (0.897) on the imbalanced 7-class set; the tree-heavy
  searched ensemble is far more robust (F1 0.942).

**Cross-subject generalization improves with more subjects** — the key scale result:

| LOSO strategy | 10-subj | 36-subj |
|---|---|---|
| No adaptation | 0.806 | **0.854** |
| Subject z-score (best) | 0.827 | **0.873** |
| CORAL | 0.768 | 0.807 |

Training on more people makes the model generalize better to an unseen person (LOSO
0.806 → 0.854 with no adaptation), and per-subject standardization lifts it further to
**0.873** — a strong subject-independent number. **Reject option:** abstaining on the
least-confident ~21% of windows reaches **99% accuracy**. **Decision smoothing:** spurious
gesture flips cut 21 → 11, exactly recovering the true transition count. Full breakdown in
`results_full36/RESULTS.md`.

<!-- RESULTS_END -->

### Generated plots (`results/`)

| File | Content |
|---|---|
| `01_raw_signals.png` | raw 8-channel EMG with gesture-colored shading |
| `02_filtering_effect.png` | raw vs. filtered signal + power spectra (notch visible at 50/60 Hz) |
| `03_class_distribution.png` | window count per gesture |
| `04_windowing.png` | sliding-window segmentation illustration |
| `05_tkeo_onset.png` | TKEO energy envelope (onset enhancement) |
| `06_augmentation.png` | original vs. electrode-shift augmented window |
| `07_feature_dist.png` | per-gesture feature distributions |
| `08_psd_per_gesture.png` | average power spectrum per gesture |
| `09_subject_variability.png` | PCA of features colored by subject (motivates domain adaptation) |
| `10_vanilla_vs_enhanced.png` | per-model novelty: vanilla vs. enhanced accuracy |
| `11_model_comparison.png` | accuracy / F1 / ROC-AUC across all models |
| `12_confusion_ensemble.png` | confusion matrix of the voting ensemble |
| `13_roc_curves.png` | macro one-vs-rest ROC curves |
| `14_feature_importance.png` | top Random-Forest permutation importances |
| `15_loso_adaptation.png` | LOSO accuracy by adaptation strategy |
| `16_ensemble_search.png` | top soft-voting ensemble permutations (vs. best single model) |
| `17_accuracy_coverage.png` | reject option: accuracy vs. coverage trade-off |
| `18_decision_smoothing.png` | real-time decision smoothing: raw vs. smoothed prediction trace |

---

## Notes & honest caveats

- Not every per-model enhancement wins on every dataset — that's expected and the
  point of reporting vanilla vs. enhanced side by side. The Gaussian-NB
  (Yeo-Johnson), SVM (calibration) and KNN (NCA) gains are the most consistent.
- CORAL is **transductive** — it uses the (unlabeled) target subject's windows to
  estimate covariance. `Subject z-score` is fully inductive and is often the
  stronger, simpler baseline; both are reported.
- **Electrode-shift augmentation did *not* help on this 10-subject subset** (0.888 →
  0.836 on the held-out subject). It's a physically-motivated augmentation, but here
  the dominant inter-subject differences aren't pure armband rotation, so rotating
  channels mostly adds noise. Reported honestly rather than hidden; it may help on
  the full 36-subject set or with a smaller `max_shift`.
- The ensemble search selects the winning combination **by cross-validation on the
  training set**, then reports its accuracy on the untouched test set — no test-set
  peeking. The search is cheap because it averages cached out-of-fold probabilities
  rather than refitting models.
- LOSO uses shrinkage-LDA as a fast workhorse across all folds; the within-subject
  numbers use the full model zoo and ensembles.
- Default run caps subjects at 10 for runtime; pass `--max-subjects 36` for the full set.

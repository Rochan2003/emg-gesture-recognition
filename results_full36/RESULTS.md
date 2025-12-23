# EMG Gesture Recognition -- Results

## Random-split test metrics

|                        |   accuracy |   precision_macro |   recall_macro |   f1_macro |   roc_auc_ovr_macro |
|:-----------------------|-----------:|------------------:|---------------:|-----------:|--------------------:|
| Best ensemble (search) |     0.9509 |            0.9528 |         0.9323 |     0.9416 |              0.9975 |
| HistGradientBoosting   |     0.9444 |            0.9344 |         0.9267 |     0.9303 |              0.9965 |
| Random Forest          |     0.9419 |            0.9446 |         0.9161 |     0.9285 |              0.9964 |
| SVM-RBF                |     0.9278 |            0.9257 |         0.8872 |     0.9026 |              0.9927 |
| KNN                    |     0.925  |            0.9068 |         0.8932 |     0.8993 |              0.9835 |
| Voting ensemble        |     0.9332 |            0.943  |         0.8749 |     0.897  |              0.9963 |
| Decision Tree          |     0.8703 |            0.7951 |         0.7995 |     0.7968 |              0.9492 |
| LDA                    |     0.877  |            0.8142 |         0.7797 |     0.7882 |              0.9864 |
| Logistic Regression    |     0.8908 |            0.8361 |         0.7791 |     0.7832 |              0.9896 |
| Gaussian NB            |     0.8492 |            0.7635 |         0.7855 |     0.7667 |              0.9815 |


## Per-model novelty (vanilla vs enhanced accuracy)

| Model | Novelty | Vanilla | Enhanced |
|---|---|---|---|
| Logistic Regression | Elastic-net (L1/L2) feature selection | 0.895 | 0.891 |
| Decision Tree | Cost-complexity pruning | 0.872 | 0.870 |
| Random Forest | Balanced sub-sampling + tuned forest | 0.940 | 0.942 |
| Gaussian NB | Yeo-Johnson power transform | 0.806 | 0.849 |
| LDA | Ledoit-Wolf shrinkage | 0.880 | 0.877 |
| SVM-RBF | Probability calibration | 0.914 | 0.928 |
| KNN | NCA learned metric | 0.887 | 0.925 |
| HistGradientBoosting | Early stopping + class weights | 0.945 | 0.944 |

## Ensemble search (top soft-voting permutations, by CV accuracy)

_247 permutations searched; best = **RF+SVM+KNN+HGB**._

| Ensemble | size | CV accuracy | CV macro-F1 |
|---|---|---|---|
| RF+SVM+KNN+HGB | 4 | 0.940 | 0.919 |
| SVM+KNN+HGB | 3 | 0.939 | 0.917 |
| RF+SVM+HGB | 3 | 0.937 | 0.913 |
| SVM+HGB | 2 | 0.937 | 0.913 |
| DT+SVM+KNN+HGB | 4 | 0.937 | 0.915 |
| KNN+HGB | 2 | 0.937 | 0.913 |
| RF+KNN+HGB | 3 | 0.937 | 0.913 |
| DT+RF+SVM+KNN+HGB | 5 | 0.937 | 0.915 |

## Electrode-shift augmentation ablation (cross-subject hold-out)

- held-out test subject: **36**
- ensemble accuracy without augmentation: **0.962**
- ensemble accuracy with electrode-shift augmentation: **0.956**

## Cross-subject (LOSO) accuracy by adaptation strategy

| Strategy | LOSO acc | +/- | macro-F1 |
|---|---|---|---|
| No adaptation | 0.854 | 0.137 | 0.803 |
| Subject z-score | 0.873 | 0.126 | 0.821 |
| CORAL | 0.807 | 0.134 | 0.728 |
| CORAL + 5-shot calib | 0.811 | 0.132 | 0.731 |

## Confidence-gated reject option

- accuracy at 100% coverage: **0.951**
- to reach 95% accuracy: keep **100.0%** of windows (confidence >= 0.342)
- to reach 99% accuracy: keep **79.1%** of windows (confidence >= 0.832)

## Real-time decision smoothing (held-out stream)

- per-window accuracy: raw **0.940** -> smoothed **0.852**
- gesture transitions: true 11, raw 21, smoothed 11 (fewer spurious flips)
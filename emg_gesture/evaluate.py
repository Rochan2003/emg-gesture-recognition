"""Metrics and result plots."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import BaseEstimator  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize  # noqa: E402


def _proba(model: BaseEstimator, X: np.ndarray) -> Optional[np.ndarray]:
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except Exception:  # pragma: no cover
            return None
    return None


def evaluate_model(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Accuracy, macro precision/recall/F1, and macro one-vs-rest ROC AUC."""
    pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_macro": float(precision_score(y_test, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "roc_auc_ovr_macro": float("nan"),
    }
    # ROC-AUC needs probabilities and every class present; leave it NaN otherwise
    proba = _proba(model, X_test)
    if proba is not None:
        classes = getattr(model, "classes_", np.unique(y_test))
        try:
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(
                    y_test, proba, multi_class="ovr", average="macro", labels=classes
                )
            )
        except Exception:  # pragma: no cover - degenerate class coverage
            pass
    return metrics


def results_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """DataFrame of model -> metrics, sorted by F1 descending."""
    df = pd.DataFrame(results).T
    cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr_macro"]
    df = df[[c for c in cols if c in df.columns]]
    return df.sort_values("f1_macro", ascending=False)


def print_results_table(df: pd.DataFrame, title: str = "Results") -> None:
    print(f"\n=== {title} ===")
    with pd.option_context("display.float_format", lambda v: f"{v:0.4f}"):
        print(df.to_string())


def plot_confusion_matrix(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Sequence[str],
    path: Path,
    title: str = "Confusion matrix",
    normalize: bool = True,
) -> Path:
    pred = model.predict(X_test)
    labels = getattr(model, "classes_", np.unique(y_test))
    cm = confusion_matrix(y_test, pred, labels=labels)
    if normalize:
        # row-normalize so each true-class row sums to 1 (diagonal = recall)
        cm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(len(labels))
    names = [class_names[i] if i < len(class_names) else str(l) for i, l in enumerate(labels)]
    ax.set_xticks(ticks, names, rotation=45, ha="right")
    ax.set_yticks(ticks, names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = (cm.max() + cm.min()) / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]:.2f}" if normalize else int(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] < thresh else "black", fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_model_comparison(df: pd.DataFrame, path: Path, title: str = "Model comparison") -> Path:
    metrics = [m for m in ["accuracy", "f1_macro", "roc_auc_ovr_macro"] if m in df.columns]
    models = list(df.index)
    x = np.arange(len(models))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.1), 5))
    for k, m in enumerate(metrics):
        ax.bar(x + k * width, df[m].values, width, label=m)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2, models, rotation=40, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_roc_curves(
    fitted_models: Dict[str, BaseEstimator],
    X_test: np.ndarray,
    y_test: np.ndarray,
    path: Path,
    title: str = "Macro-average ROC (one-vs-rest)",
) -> Path:
    classes = np.unique(y_test)
    y_bin = label_binarize(y_test, classes=classes)
    grid = np.linspace(0, 1, 200)

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, model in fitted_models.items():
        proba = _proba(model, X_test)
        if proba is None:
            continue
        # line up proba columns with classes so the ROC is per-class correct
        mclasses = list(getattr(model, "classes_", classes))
        col = [mclasses.index(c) for c in classes if c in mclasses]
        if len(col) != len(classes):
            continue
        # macro ROC: interpolate each class curve onto a shared grid, then average
        tprs = []
        for k in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_bin[:, k], proba[:, col[k]])
            tprs.append(np.interp(grid, fpr, tpr))
        mean_tpr = np.mean(tprs, axis=0)
        try:
            auc = roc_auc_score(y_test, proba[:, col], multi_class="ovr", average="macro", labels=classes)
        except Exception:
            auc = float("nan")
        ax.plot(grid, mean_tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_feature_importances(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Sequence[str],
    path: Path,
    top_n: int = 20,
    use_permutation: bool = True,
    seed: int = 42,
) -> Path:
    """Plot the top feature importances (permutation by default)."""
    # permutation importance is less biased than tree impurity on continuous features
    if use_permutation:
        r = permutation_importance(model, X_test, y_test, n_repeats=8, random_state=seed, n_jobs=-1)
        importances = r.importances_mean
        kind = "permutation importance"
    else:
        importances = getattr(model, "feature_importances_", None)
        kind = "impurity importance"
        if importances is None:
            r = permutation_importance(model, X_test, y_test, n_repeats=8, random_state=seed)
            importances = r.importances_mean
            kind = "permutation importance"

    order = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(order)), importances[order][::-1], color="#3a7ca5")
    ax.set_yticks(range(len(order)), [feature_names[i] for i in order][::-1])
    ax.set_xlabel(kind)
    ax.set_title(f"Top {top_n} features ({kind})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path

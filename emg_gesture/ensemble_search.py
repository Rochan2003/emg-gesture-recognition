"""Search soft-voting model combinations for the one with the best CV accuracy."""
from __future__ import annotations

from collections import OrderedDict
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config
from .models import MODEL_SPECS

RS = config.RANDOM_STATE

# short labels for plotting and printing
SHORT_NAMES: Dict[str, str] = {
    "Logistic Regression": "LR",
    "Decision Tree": "DT",
    "Random Forest": "RF",
    "Gaussian NB": "GNB",
    "LDA": "LDA",
    "SVM-RBF": "SVM",
    "KNN": "KNN",
    "HistGradientBoosting": "HGB",
}
_NAME_TO_KEY = {spec[0]: key for key, spec in MODEL_SPECS.items()}


def get_oof_probabilities(
    models: Dict[str, object], X: np.ndarray, y: np.ndarray, cv: int = config.CV_FOLDS
) -> Tuple["OrderedDict[str, np.ndarray]", np.ndarray]:
    """Out-of-fold predict_proba for each model, returned as (probas, classes).

    OOF probabilities come from held-out folds, so scoring subsets on them
    does not leak the test set.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RS)
    classes = np.unique(y)
    probas: "OrderedDict[str, np.ndarray]" = OrderedDict()
    for name, model in models.items():
        probas[name] = cross_val_predict(
            clone(model), X, y, cv=skf, method="predict_proba", n_jobs=-1
        )
    return probas, classes


def oof_accuracies(
    probas: Dict[str, np.ndarray], y: np.ndarray, classes: np.ndarray
) -> "OrderedDict[str, float]":
    """Per-model OOF accuracy from the stored probabilities."""
    out: "OrderedDict[str, float]" = OrderedDict()
    for name, p in probas.items():
        out[name] = float(accuracy_score(y, classes[np.argmax(p, axis=1)]))
    return out


def evaluate_subsets(
    probas: Dict[str, np.ndarray],
    y: np.ndarray,
    classes: np.ndarray,
    min_size: int = 2,
    max_size: Optional[int] = None,
    weighted: bool = True,
    model_weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, object]]:
    """Score every soft-voting subset between min_size and max_size members.

    Returns dicts of {members, size, accuracy, f1_macro} sorted by accuracy.
    With weighting, stronger members (by model_weights) count more in the average.
    """
    names = list(probas.keys())
    max_size = max_size or len(names)
    results: List[Dict[str, object]] = []
    for k in range(min_size, max_size + 1):
        for combo in combinations(names, k):
            if weighted and model_weights:
                # weighted average of the members' probabilities
                w = np.array([model_weights[n] for n in combo], dtype=float)
                w = w / w.sum()
                avg = sum(wi * probas[n] for wi, n in zip(w, combo))
            else:
                avg = np.mean([probas[n] for n in combo], axis=0)
            pred = classes[np.argmax(avg, axis=1)]
            results.append(
                {
                    "members": combo,
                    "size": k,
                    "accuracy": float(accuracy_score(y, pred)),
                    "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
                }
            )
    results.sort(key=lambda d: d["accuracy"], reverse=True)
    return results


def short_label(members: Sequence[str]) -> str:
    """Join member short-names with '+', e.g. SVM+RF."""
    return "+".join(SHORT_NAMES.get(m, m) for m in members)


def build_voting(
    members: Sequence[str],
    variant: str = "enhanced",
    model_weights: Optional[Dict[str, float]] = None,
) -> VotingClassifier:
    """Build a soft-voting ensemble from a list of model names."""
    idx = 2 if variant == "enhanced" else 1
    estimators = [(_NAME_TO_KEY[m], MODEL_SPECS[_NAME_TO_KEY[m]][idx]()) for m in members]
    weights = [model_weights[m] for m in members] if model_weights else None
    return VotingClassifier(estimators=estimators, voting="soft", weights=weights, n_jobs=-1)

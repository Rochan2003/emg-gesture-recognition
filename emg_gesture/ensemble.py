"""Soft-voting ensemble built from an SVM, a KNN, and logistic regression."""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Optional, Sequence

from sklearn.base import BaseEstimator
from sklearn.ensemble import VotingClassifier

from . import models


def constituent_models(variant: str = "enhanced") -> "OrderedDict[str, BaseEstimator]":
    """The three ensemble members as standalone estimators (for comparison)."""
    idx = 2 if variant == "enhanced" else 1
    specs = models.MODEL_SPECS
    return OrderedDict(
        [
            ("SVM-RBF", specs["svm"][idx]()),
            ("KNN", specs["knn"][idx]()),
            ("Logistic Regression", specs["logreg"][idx]()),
        ]
    )


def build_voting_ensemble(
    variant: str = "enhanced", weights: Optional[Sequence[float]] = None
) -> VotingClassifier:
    """Build the soft-voting ensemble of SVC + KNN + logistic regression.

    weights are in SVC, KNN, LogReg order (e.g. each model's CV accuracy so the
    stronger ones count for more); None means equal weights.
    """
    members = constituent_models(variant)
    estimators = [
        ("svc", members["SVM-RBF"]),
        ("knn", members["KNN"]),
        ("logreg", members["Logistic Regression"]),
    ]
    return VotingClassifier(
        estimators=estimators, voting="soft", weights=list(weights) if weights else None, n_jobs=-1
    )

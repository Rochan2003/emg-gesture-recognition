"""Each classifier comes in a vanilla and an enhanced version."""
from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import config

RS = config.RANDOM_STATE


def _scaled(estimator: BaseEstimator) -> Pipeline:
    """Put a StandardScaler in front of the estimator."""
    return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])


# decision tree that picks ccp_alpha by cross-validation instead of a fixed max_depth
class CCPDecisionTree(BaseEstimator, ClassifierMixin):
    """Decision tree that picks ccp_alpha by cross-validation."""

    def __init__(self, random_state: Optional[int] = None, cv: int = 3, max_alphas: int = 10):
        self.random_state = random_state
        self.cv = cv
        self.max_alphas = max_alphas

    def fit(self, X, y):
        base = DecisionTreeClassifier(random_state=self.random_state)
        path = base.cost_complexity_pruning_path(X, y)
        alphas = np.unique(path.ccp_alphas)
        alphas = alphas[alphas >= 0]
        if len(alphas) > self.max_alphas:
            idx = np.linspace(0, len(alphas) - 1, self.max_alphas).astype(int)
            alphas = alphas[idx]
        cv = StratifiedKFold(self.cv, shuffle=True, random_state=self.random_state)
        best_alpha, best_score = 0.0, -np.inf
        for a in alphas:
            tree = DecisionTreeClassifier(random_state=self.random_state, ccp_alpha=a)
            score = float(np.mean(cross_val_score(tree, X, y, cv=cv)))
            if score > best_score:
                best_score, best_alpha = score, float(a)
        self.best_ccp_alpha_ = best_alpha
        self.tree_ = DecisionTreeClassifier(
            random_state=self.random_state, ccp_alpha=best_alpha
        ).fit(X, y)
        self.classes_ = self.tree_.classes_
        return self

    def predict(self, X):
        return self.tree_.predict(X)

    def predict_proba(self, X):
        return self.tree_.predict_proba(X)


def _logreg_vanilla():
    return _scaled(LogisticRegression(max_iter=1000))


def _logreg_enhanced():
    # elastic-net does feature selection through the L1/L2 mix; saga needs headroom to converge
    return _scaled(
        LogisticRegressionCV(
            Cs=6,
            penalty="elasticnet",
            solver="saga",
            l1_ratios=[0.2, 0.5, 0.8],
            max_iter=10000,
            tol=1e-3,
            cv=3,
            n_jobs=-1,
            random_state=RS,
        )
    )


def _dtree_vanilla():
    return DecisionTreeClassifier(random_state=RS)


def _dtree_enhanced():
    return CCPDecisionTree(random_state=RS, cv=3)


def _rf_vanilla():
    return RandomForestClassifier(n_estimators=200, random_state=RS, n_jobs=-1)


def _rf_enhanced():
    # balanced sub-sampling handles class imbalance
    return RandomForestClassifier(
        n_estimators=400,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RS,
        n_jobs=-1,
    )


def _gnb_vanilla():
    return GaussianNB()


def _gnb_enhanced():
    # Yeo-Johnson makes the skewed power features more Gaussian, which is what GNB assumes
    return Pipeline(
        [
            ("power", PowerTransformer(method="yeo-johnson", standardize=True)),
            ("clf", GaussianNB()),
        ]
    )


def _lda_vanilla():
    return LinearDiscriminantAnalysis()


def _lda_enhanced():
    # shrinkage stabilizes the covariance when channels are correlated
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def _svm_vanilla():
    return _scaled(SVC(probability=True, random_state=RS))


def _svm_enhanced():
    # calibrate so the probabilities are usable in the soft-voting ensemble
    return _scaled(
        CalibratedClassifierCV(
            SVC(C=10.0, gamma="scale", probability=False, random_state=RS),
            method="sigmoid",
            cv=3,
        )
    )


def _knn_vanilla():
    return _scaled(KNeighborsClassifier())


def _knn_enhanced():
    # NCA learns a better distance metric before the neighbor vote
    return _scaled(
        Pipeline(
            [
                ("nca", NeighborhoodComponentsAnalysis(n_components=10, max_iter=50, random_state=RS)),
                ("knn", KNeighborsClassifier(weights="distance")),
            ]
        )
    )


def _hgb_vanilla():
    return HistGradientBoostingClassifier(random_state=RS)


def _hgb_enhanced():
    # early stopping keeps it from overfitting; class weights handle imbalance
    return HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        class_weight="balanced",
        random_state=RS,
    )


# key -> (display name, vanilla factory, enhanced factory, description)
MODEL_SPECS: "OrderedDict[str, Tuple[str, Callable, Callable, str]]" = OrderedDict(
    [
        ("logreg", ("Logistic Regression", _logreg_vanilla, _logreg_enhanced, "Elastic-net (L1/L2) feature selection")),
        ("dtree", ("Decision Tree", _dtree_vanilla, _dtree_enhanced, "Cost-complexity pruning")),
        ("rf", ("Random Forest", _rf_vanilla, _rf_enhanced, "Balanced sub-sampling + tuned forest")),
        ("gnb", ("Gaussian NB", _gnb_vanilla, _gnb_enhanced, "Yeo-Johnson power transform")),
        ("lda", ("LDA", _lda_vanilla, _lda_enhanced, "Ledoit-Wolf shrinkage")),
        ("svm", ("SVM-RBF", _svm_vanilla, _svm_enhanced, "Probability calibration")),
        ("knn", ("KNN", _knn_vanilla, _knn_enhanced, "NCA learned metric")),
        ("hgb", ("HistGradientBoosting", _hgb_vanilla, _hgb_enhanced, "Early stopping + class weights")),
    ]
)


def get_models(variant: str = "enhanced") -> "OrderedDict[str, BaseEstimator]":
    """Build a fresh dict of estimators. variant is "enhanced" or "vanilla"."""
    idx = 2 if variant == "enhanced" else 1
    return OrderedDict(
        (spec[0], spec[idx]()) for spec in MODEL_SPECS.values()
    )


def novelty_table() -> List[Tuple[str, str]]:
    """List of (model name, description) for the report."""
    return [(spec[0], spec[3]) for spec in MODEL_SPECS.values()]


def cross_validate_models(
    models: Dict[str, BaseEstimator],
    X: np.ndarray,
    y: np.ndarray,
    cv: int = config.CV_FOLDS,
    scoring: str = "accuracy",
) -> "OrderedDict[str, Tuple[float, float]]":
    """Stratified CV; returns name -> (mean, std) of the score."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RS)
    out: "OrderedDict[str, Tuple[float, float]]" = OrderedDict()
    for name, model in models.items():
        scores = cross_val_score(clone(model), X, y, cv=skf, scoring=scoring, n_jobs=-1)
        out[name] = (float(scores.mean()), float(scores.std()))
    return out


def tune_random_forest(
    X: np.ndarray, y: np.ndarray, cv: int = config.CV_FOLDS
) -> GridSearchCV:
    """Light GridSearchCV for Random Forest (spec requirement)."""
    grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 12, 20],
        "min_samples_leaf": [1, 2],
    }
    gs = GridSearchCV(
        RandomForestClassifier(random_state=RS, n_jobs=-1),
        grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )
    gs.fit(X, y)
    return gs


def tune_decision_tree(
    X: np.ndarray, y: np.ndarray, cv: int = config.CV_FOLDS
) -> GridSearchCV:
    """Light GridSearchCV for Decision Tree (spec requirement)."""
    grid = {
        "max_depth": [None, 6, 10, 16],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"],
    }
    gs = GridSearchCV(
        DecisionTreeClassifier(random_state=RS),
        grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )
    gs.fit(X, y)
    return gs

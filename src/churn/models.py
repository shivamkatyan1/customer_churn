"""Decision-tree candidate configurations, selection and evaluation.

Selection happens ONLY on the training partition via seeded stratified
cross-validation; the untouched test set is used exactly once for the final,
honest evaluation of the chosen model.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

POSITIVE_LABEL = "Yes"
LABELS = ["No", "Yes"]
N_FOLDS = 5
CV_RANDOM_STATE = 42

# The two (or more) explicit configurations compared in the brief.
CANDIDATE_CONFIGS = {
    "baseline_unconstrained": {},  # grows until pure/leaf-limited by sklearn defaults
    "pruned_depth6_minleaf5": {"max_depth": 6, "min_samples_leaf": 5},
}
# Bonus candidate (kept separate so it never displaces a mandatory one).
BONUS_CONFIGS = {
    "pruned_balanced_depth6_minleaf5": {
        "max_depth": 6, "min_samples_leaf": 5, "class_weight": "balanced",
    },
}


@dataclass
class FoldMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float


@dataclass
class CandidateResult:
    name: str
    params: dict
    fold_metrics: list  # list[FoldMetrics]
    mean: FoldMetrics


def make_classifier(params: dict) -> DecisionTreeClassifier:
    """A reproducible DecisionTreeClassifier (seed pinned; never leave it global)."""
    p = dict(params)
    p.setdefault("random_state", 42)
    return DecisionTreeClassifier(**p)


def run_cross_validation(build_pipeline_fn, X_train, y_train,
                         folds=N_FOLDS, random_state=CV_RANDOM_STATE) -> CandidateResult:
    """Seeded stratified k-fold CV on the TRAINING partition only."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

    def metrics(y_true, y_pred) -> FoldMetrics:
        return FoldMetrics(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL)),
            recall=float(recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL)),
            f1=float(f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL)),
        )

    out = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_fold, y_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        pipe = build_pipeline_fn()
        pipe.fit(X_fold, y_fold)
        preds = pipe.predict(X_val)
        out.append(metrics(y_val, preds))

    mean = FoldMetrics(*[round(float(np.mean([getattr(f, k) for f in out])), 4)
                         for k in ("accuracy", "precision", "recall", "f1")])
    return CandidateResult(name="", params={}, fold_metrics=out, mean=mean)


def select_best(candidate_results: list) -> CandidateResult:
    """Select the candidate with the best mean positive-class F1 (CV, train-only).

    F1 balances catching churners (recall) with limiting unnecessary outreach
    (precision); the precision/recall values of every candidate are reported
    alongside so the trade-off stays visible.
    """
    return max(candidate_results, key=lambda r: r.mean.f1)


def evaluate_on_test(pipeline, X_test, y_test) -> dict:
    """The single, honest, untouched-test evaluation of the chosen model."""
    y_pred = pipeline.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=LABELS
    )
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_positive": float(precision[1]),
        "recall_positive": float(recall[1]),
        "f1_positive": float(f1[1]),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=LABELS).tolist(),
        "confusion_labels": LABELS,  # [row, col] = [actual, predicted]
    }


def feature_importance(pipeline, feature_names: list[str]) -> list[dict]:
    """Top feature importances mapped back to business names."""
    clf = pipeline.named_steps["classifier"]
    values = clf.feature_importances_
    pairs = sorted(zip(feature_names, values), key=lambda kv: kv[1], reverse=True)
    return [{"feature": f, "importance": round(float(v), 4)} for f, v in pairs]

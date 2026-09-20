"""Data loading and the deterministic stratified 70:30 split."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Churn"
ID_COLUMN = "customerID"
TEST_SIZE = 0.3
RANDOM_STATE = 42

from .preprocessing import PREDICTOR_COLUMNS  # noqa: E402


def load_data(path: str) -> pd.DataFrame:
    """Load the supplied Telco CSV exactly as shipped (no online substitute)."""
    return pd.read_csv(path, na_values=[], keep_default_na=False)


def stratify_split(data: pd.DataFrame, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Stratified 70:30 split on ``Churn`` with ``random_state=42``.

    The split happens before any learned statistic/preprocessing so that the
    test set never informs training decisions. Returns training/test data
    frames, the target series and the customer IDs for disjointness checks.
    """
    X = data[PREDICTOR_COLUMNS]
    y = data[TARGET]
    ids = data[ID_COLUMN]

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, id_train, id_test


def split_report(X_train, X_test, y_train, y_test, id_train, id_test) -> dict:
    """Summary facts about the split, used for verification."""
    train_ids, test_ids = set(id_train), set(id_test)
    return {
        "n_total": len(X_train) + len(X_test),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_size_fraction": round(len(X_test) / (len(X_train) + len(X_test)), 4),
        "churn_train": int((y_train == "Yes").sum()),
        "churn_train_pct": round(float((y_train == "Yes").mean()) * 100, 2),
        "churn_test": int((y_test == "Yes").sum()),
        "churn_test_pct": round(float((y_test == "Yes").mean()) * 100, 2),
        "disjoint_ids": len(train_ids.intersection(test_ids)) == 0,
        "expected_sizes": f"{len(X_train)}/{len(X_test)} (70/30 of {len(X_train) + len(X_test)})",
    }

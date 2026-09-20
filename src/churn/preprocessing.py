"""Column layouts and the reusable preprocessing pipeline.

Everything here is *fitted on training data only* and shipped inside the saved
``Pipeline`` so that new/unseen data receives exactly the same transformation.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .features import ADDON_COLUMNS, AddOnCount, CleanDataFrame, TenureBand

# Canonical predictor order (19 features, target ``Churn`` and id excluded).
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
PREDICTOR_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

# The categorical levels seen in the supplied dataset; used by the API to give
# clear validation errors instead of silently accepting unknown categories.
KNOWN_CATEGORIES = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["No", "DSL", "Fiber optic"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

# Sanity bounds used by the API (documented; the trained model saw the dataset
# ranges span these observed values: tenure 0-72, MonthlyCharges 18.25-118.75,
# TotalCharges 0-8684.8).
MAX_TENURE = 240
MAX_MONTHLY_CHARGES = 10_000.0
MAX_TOTAL_CHARGES = 1_000_000.0


def build_preprocessor():
    """Return the fitted-on-train-only ColumnTransformer (not yet fitted)."""
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_COLUMNS),
            ("addons", AddOnCount(), ADDON_COLUMNS),
            ("tenure_band", TenureBand(), ["tenure"]),
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=np.nan,
                ),
                CATEGORICAL_COLUMNS,
            ),
        ],
        verbose_feature_names_out=False,
        remainder="drop",
    )


def build_pipeline(classifier=None):
    """Assemble the full pipeline.

    ``classifier`` may be ``None`` (preprocessing-only, useful in the notebook)
    or a ``DecisionTreeClassifier`` instance used for the train step.
    """
    steps = [("clean", CleanDataFrame()), ("preprocess", build_preprocessor())]
    if classifier is not None:
        steps.append(("classifier", classifier))
    return Pipeline(steps)


def feature_names(preprocessor) -> list[str]:
    """Model-ready feature names in output order (drives feature importance)."""
    names = list(preprocessor.get_feature_names_out())
    assert len(names) == len(NUMERIC_COLUMNS) + 1 + 1 + len(CATEGORICAL_COLUMNS), (
        f"unexpected feature count {len(names)}"
    )
    return names

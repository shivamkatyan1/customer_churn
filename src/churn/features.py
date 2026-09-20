"""Feature engineering and input-cleaning transformers.

All transformers are ``scikit-learn`` compatible so they can live inside the
saved ``Pipeline`` and are applied identically during training and inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

#
# Column constants (must match the supplied data dictionary)
#
ADDON_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Fixed tenure bands (months). Bands are a business rule, not learned from data.
TENURE_BAND_EDGES = [12, 24, 48]  # 0-12, 13-24, 25-48, 49+
TENURE_BAND_LABELS = ["0-12", "13-24", "25-48", "49+"]


class AddOnCount(BaseEstimator, TransformerMixin):
    """Count how many internet add-on services a customer actually subscribed to.

    Input : 2D array of the 6 add-on columns (Any order of ``Yes`` / ``No`` /
            ``No internet service``).
    Output: single integer column ``internet_addon_count`` between 0 and 6.

    The idea: a customer who subscribes to several add-ons is more entangled
    with the product (higher switching cost, more ways to reach support), so a
    low bundle count may correlate with churn. Only an exact ``Yes`` counts;
    ``No`` and ``No internet service`` both mean *not subscribed*.
    """

    def fit(self, X, y=None):  # noqa: D102
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=object)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        cleaned = np.char.strip(arr.astype(str))
        counts = np.sum(cleaned == "Yes", axis=1).astype(np.int64)
        return counts.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(["internet_addon_count"])


class TenureBand(BaseEstimator, TransformerMixin):
    """Map tenure (months) onto a fixed business band (1..4).

    Bands: 0-12 -> 1, 13-24 -> 2, 25-48 -> 3, 49+ -> 4.
    Invalid tenure (negative or missing) raises ``ValueError``: upstream code
    must validate before this runs, and the API enforces the same rules.
    """

    def fit(self, X, y=None):  # noqa: D102
        return self

    def transform(self, X):
        x = np.asarray(X, dtype=float).ravel()
        if np.isnan(x).any():
            raise ValueError("tenure must not contain NaN/missing values")
        if (x < 0).any():
            raise ValueError("tenure must be non-negative")
        # np.select: first true -> band
        bands = np.select(
            [x <= 12, x <= 24, x <= 48, True],
            [1, 2, 3, 4],
        )
        return bands.astype(np.int64).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(["tenure_band"])


class BlankTotalChargesRule(BaseEstimator, TransformerMixin):
    """Apply the zero-tenure ``TotalCharges`` domain rule.

    A customer with ``tenure == 0`` has not accumulated any charges yet, so a
    blank/whitespace ``TotalCharges`` is treated as ``0.0`` (this matches every
    one of the 11 blank rows in the supplied dataset).

    A blank ``TotalCharges`` with ``tenure > 0`` is treated as invalid instead
    of being imputed -- new data must follow the same rule, and the REST API
    validates it before it ever reaches the model.
    """

    def __init__(self, total_charges: str = "TotalCharges", tenure: str = "tenure"):
        self.total_charges = total_charges
        self.tenure = tenure

    def fit(self, X, y=None):  # noqa: D102
        required = {self.total_charges, self.tenure}
        missing = required.difference(X.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        self._columns_ = list(X.columns)
        return self

    def transform(self, X):
        X = X.copy()
        tc = pd.to_numeric(X[self.total_charges], errors="coerce").astype(float)
        blank = tc.isna()
        if blank.any():
            tenure_vals = pd.to_numeric(X[self.tenure], errors="coerce")
            invalid = blank & (tenure_vals != 0)
            if invalid.any():
                raise ValueError(
                    "TotalCharges is missing but tenure != 0 for some rows; "
                    "only tenure == 0 customers may have a blank TotalCharges"
                )
            tc = tc.fillna(0.0)
        X[self.total_charges] = tc
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self._columns_)


class CleanDataFrame(BaseEstimator, TransformerMixin):
    """Normalise a raw customer frame into the internal feature frame.

    1. strips whitespace from every string cell,
    2. applies the zero-tenure ``TotalCharges`` rule,
    3. casts ``SeniorCitizen`` to int 0/1 and the numeric money/tenure columns
       to float, so training and inference see identical types.
    """

    def __init__(self, numeric_columns=("tenure", "MonthlyCharges", "TotalCharges"),
                 senior_column="SeniorCitizen"):
        self.numeric_columns = list(numeric_columns)
        self.senior_column = senior_column

    def fit(self, X, y=None):  # noqa: D102
        BlankTotalChargesRule().fit(X)
        return self

    def transform(self, X):
        X = X.copy()
        # 1. whitespace-strip every non-numeric cell (covers object AND string dtypes)
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].astype("string").str.strip()
        # 2. zero-tenure TotalCharges rule
        X = BlankTotalChargesRule().transform(X)
        # 3. stable numeric types
        for col in self.numeric_columns:
            X[col] = pd.to_numeric(X[col], errors="raise")
        X[self.senior_column] = pd.to_numeric(
            X[self.senior_column], errors="raise"
        ).astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(input_features) if input_features is not None else self._columns_)

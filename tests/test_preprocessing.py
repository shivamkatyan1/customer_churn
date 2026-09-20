"""Deterministic unit tests for feature engineering and cleaning transformers.

Run with:  pytest tests/      (from the customer_churn directory)
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.churn.features import (
    ADDON_COLUMNS,
    AddOnCount,
    BlankTotalChargesRule,
    CleanDataFrame,
    TenureBand,
)


def addon_frame(*yes_columns) -> pd.DataFrame:
    """Build a 6-column add-on frame with 'Yes' only in the given columns."""
    data = {col: ["Yes" if col in yes_columns else "No"] for col in ADDON_COLUMNS}
    return pd.DataFrame(data)


class TestAddOnCount:
    def test_yes_count(self):
        out = AddOnCount().transform(addon_frame("OnlineSecurity", "TechSupport")).ravel()
        assert int(out[0]) == 2

    def test_no_service_counts_zero(self):
        X = pd.DataFrame(
            {
                "OnlineSecurity": ["No internet service"],
                "OnlineBackup": ["No internet service"],
                "DeviceProtection": ["No internet service"],
                "TechSupport": ["No internet service"],
                "StreamingTV": ["No internet service"],
                "StreamingMovies": ["No internet service"],
            }
        )
        assert int(AddOnCount().transform(X).ravel()[0]) == 0

    def test_whitespace_tolerant(self):
        X = addon_frame()
        X.loc[0, "OnlineBackup"] = "  Yes  "
        assert int(AddOnCount().transform(X).ravel()[0]) == 1


class TestTenureBand:
    def test_bands(self):
        X = pd.DataFrame({"tenure": [0, 1, 12, 13, 24, 25, 48, 49, 72]})
        assert list(TenureBand().transform(X).ravel()) == [1, 1, 1, 2, 2, 3, 3, 4, 4]

    def test_boundary_49_plus(self):
        X = pd.DataFrame({"tenure": [49, 120]})
        assert list(TenureBand().transform(X).ravel()) == [4, 4]

    def test_negative_rejected(self):
        with pytest.raises(ValueError):
            TenureBand().transform(pd.DataFrame({"tenure": [-1]}))


class TestBlankTotalChargesRule:
    def test_zero_tenure_filled(self):
        X = pd.DataFrame(
            {"tenure": [0, 0], "TotalCharges": ["", " "], "MonthlyCharges": [50.0, 20.0]}
        )
        out = BlankTotalChargesRule().transform(X)
        assert list(out["TotalCharges"]) == [0.0, 0.0]

    def test_blank_with_positive_tenure_rejected(self):
        with pytest.raises(ValueError):
            BlankTotalChargesRule().transform(
                pd.DataFrame({"tenure": [3], "TotalCharges": [""]})
            )

    def test_present_value_kept_and_stripped(self):
        X = pd.DataFrame({"tenure": [5], "TotalCharges": ["1889.5 "]})
        assert float(BlankTotalChargesRule().transform(X)["TotalCharges"][0]) == 1889.5


class TestCleanDataFrame:
    def test_strips_and_types(self):
        X = pd.DataFrame(
            {
                "tenure": [" 3 "],
                "MonthlyCharges": ["62.5"],
                "TotalCharges": ["187.5"],
                "SeniorCitizen": ["1"],
                "gender": [" Male "],
            }
        )
        out = CleanDataFrame().transform(X)
        assert out["SeniorCitizen"].dtype.kind == "i"
        assert out["gender"].iloc[0] == "Male"
        assert out["tenure"].iloc[0] == 3.0

    def test_zero_tenure_rule_applied(self):
        X = pd.DataFrame(
            {
                "tenure": ["0"],
                "MonthlyCharges": ["62.5"],
                "TotalCharges": [""],
                "SeniorCitizen": ["0"],
                "gender": ["Female"],
            }
        )
        out = CleanDataFrame().transform(X)
        assert out["TotalCharges"].iloc[0] == 0.0

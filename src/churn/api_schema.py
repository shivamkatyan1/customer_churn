"""Pydantic request/response models for ``POST /predict``.

Validation policy (documented in the README):
- Required: all 19 predictor fields with their types and ranges.
- ``customerID`` is OPTIONAL and IGNORED (may be used for tracking).
- Unknown or unexpected fields -- including the target ``Churn`` -- are
  rejected (``extra='forbid'``).
- ``TotalCharges`` may be null ONLY when ``tenure == 0`` (zero-tenure rule);
  otherwise it must be a finite number >= 0.
- Unknown categorical levels are rejected with a clear message (the underlying
  model also has an unseen-category policy, but the API refuses them up-front).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .. import churn  # noqa: F401  (ensures shared constants live in one place)
from .preprocessing import KNOWN_CATEGORIES, MAX_MONTHLY_CHARGES, MAX_TENURE, MAX_TOTAL_CHARGES


def _finite_zero_or_more(value: float, field: str, upper: float):
    if isinstance(value, bool) or not float(value) == value:
        raise ValueError(f"{field} must be a number")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a number") from None
    if not (v >= 0 and v <= upper) or v != v or v in (float("inf"), float("-inf")):
        raise ValueError(f"{field} must be a finite number between 0 and {upper:g}")
    return v


class ChurnRequest(BaseModel):
    """Raw customer information accepted by the endpoint."""

    model_config = ConfigDict(extra="forbid")

    customerID: Optional[str] = Field(default=None, description="Ignored for prediction; optional tracking id")

    gender: str
    SeniorCitizen: int = Field(ge=0, le=1, description="0 or 1 (binary)")
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0, le=MAX_TENURE, description="months with the company")
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: Optional[float] = Field(default=None, description="may be null only when tenure == 0")

    # category validation -----------------------------------------------------
    @field_validator(*list(KNOWN_CATEGORIES), mode="before")
    @classmethod
    def validate_category(cls, value, info):
        allowed = set(KNOWN_CATEGORIES[info.field_name])
        if not isinstance(value, str):
            raise ValueError(f"{info.field_name} must be a string")
        v = value.strip()
        if v not in allowed:
            raise ValueError(
                f"unknown value {value!r} for {info.field_name}; allowed: {sorted(allowed)}"
            )
        return v

    # numeric validation ------------------------------------------------------
    @field_validator("MonthlyCharges", mode="before")
    @classmethod
    def validate_monthly(cls, value):
        return _finite_zero_or_more(value, "MonthlyCharges", MAX_MONTHLY_CHARGES)

    @field_validator("TotalCharges", mode="before")
    @classmethod
    def validate_total(cls, value):
        if value is None:
            return None
        return _finite_zero_or_more(value, "TotalCharges", MAX_TOTAL_CHARGES)

    @model_validator(mode="after")
    def totalcharges_rule(self):
        if self.TotalCharges is None and self.tenure != 0:
            raise ValueError(
                "TotalCharges may be omitted only when tenure == 0 "
                "(a new customer has no accumulated charges yet)"
            )
        return self

    def to_feature_row(self) -> dict:
        """Return a row dict in the canonical predictor-column order (training order)."""
        return {
            "tenure": self.tenure,
            "MonthlyCharges": self.MonthlyCharges,
            "TotalCharges": self.TotalCharges if self.TotalCharges is not None else 0.0,
            "SeniorCitizen": self.SeniorCitizen,
            "gender": self.gender.strip(),
            "Partner": self.Partner.strip(),
            "Dependents": self.Dependents.strip(),
            "PhoneService": self.PhoneService.strip(),
            "MultipleLines": self.MultipleLines.strip(),
            "InternetService": self.InternetService.strip(),
            "OnlineSecurity": self.OnlineSecurity.strip(),
            "OnlineBackup": self.OnlineBackup.strip(),
            "DeviceProtection": self.DeviceProtection.strip(),
            "TechSupport": self.TechSupport.strip(),
            "StreamingTV": self.StreamingTV.strip(),
            "StreamingMovies": self.StreamingMovies.strip(),
            "Contract": self.Contract.strip(),
            "PaperlessBilling": self.PaperlessBilling.strip(),
            "PaymentMethod": self.PaymentMethod.strip(),
        }


class PredictionResponse(BaseModel):
    prediction: str  # "Yes" or "No"
    churn_probability: float  # finite, in [0, 1], probability of Churn == "Yes"

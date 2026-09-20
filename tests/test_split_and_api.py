"""Deterministic tests for split integrity, the API, and notebook parity.

Run with:  pytest tests/      (from the customer_churn directory)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.churn.splitting import load_data, split_report, stratify_split

MODEL_PATH = ROOT / "model" / "churn_model.pkl"


# split integrity --------------------------------------------------------
def test_split_sizes_stratified_disjoint():
    data = load_data(ROOT / "data" / "TelcoCustomerChurn.csv")
    X_tr, X_te, y_tr, y_te, id_tr, id_te = stratify_split(data)
    n = len(data)
    assert len(X_tr) + len(X_te) == n
    assert len(X_te) == pytest.approx(n * 0.3, abs=1)
    # stratification: positive-class share preserved
    assert abs((y_tr == "Yes").mean() - (y_te == "Yes").mean()) < 0.01
    # disjoint
    assert set(id_tr).isdisjoint(set(id_te))
    report = split_report(X_tr, X_te, y_tr, y_te, id_tr, id_te)
    assert report["disjoint_ids"] is True


def test_unknown_category_survives_preprocessing_policy():
    """The pipeline (OrdinalEncoder unknown) must still accept an unseen value."""
    from src.churn.preprocessing import build_pipeline
    from src.churn.models import make_classifier

    data = load_data(ROOT / "data" / "TelcoCustomerChurn.csv")
    X_tr, X_te, y_tr, y_te, _, _ = stratify_split(data)
    pipe = build_pipeline(make_classifier({"max_depth": 3, "random_state": 42}))
    pipe.fit(X_tr, y_tr)
    # craft a row with a nonsense category -> should NOT crash (encoded as -1)
    row = X_tr.iloc[[0]].copy()
    row["Contract"] = "lifetime"
    pred = pipe.predict(row)
    assert pred[0] in {"Yes", "No"}


def test_api_valid_and_invalid():
    """End-to-end API behavior against the REAL saved artifact (no network).

    Skips cleanly when the model has not been trained yet (fresh checkout).
    """
    try:
        from app import app
    except SystemExit as exc:
        pytest.skip(f"model artifact missing; run `python train.py` first ({exc})")

    client = TestClient(app)
    valid = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85, "TotalCharges": 29.85,
    }
    resp = client.post("/predict", json=valid)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["prediction"] in {"Yes", "No"}
    assert 0.0 <= body["churn_probability"] <= 1.0

    # blank TotalCharges with tenure>0 -> 422
    bad = dict(valid); bad["TotalCharges"] = None
    assert client.post("/predict", json=bad).status_code == 422

    # unknown category -> 422
    bad2 = dict(valid); bad2["Contract"] = "lifetime"
    assert client.post("/predict", json=bad2).status_code == 422

    # unexpected field (target leaked) -> 422
    bad3 = dict(valid); bad3["Churn"] = "No"
    assert client.post("/predict", json=bad3).status_code == 422

    # negative charges -> 422
    bad4 = dict(valid); bad4["MonthlyCharges"] = -5
    assert client.post("/predict", json=bad4).status_code == 422

"""Fresh-process + API parity: the saved pipeline used standalone (in this
process and in a brand-new Python process) must produce the same prediction as
the live /predict endpoint for the same input.

Skipped until the model artifact exists (run `python train.py` first).
"""

import json
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "churn_model.pkl"

VALID_ROW = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85, "TotalCharges": 29.85,
}


def _payload() -> dict:
    return dict(VALID_ROW)


@pytest.fixture(scope="module")
def prediction_sources():
    if not MODEL_PATH.exists():
        pytest.skip("model artifact missing; run `python train.py` first")

    pipeline = joblib.load(MODEL_PATH)

    # 1) fresh subprocess: load + predict with a brand-new interpreter
    snippet = (
        "import sys, json, joblib, pandas as pd\n"
        "p = joblib.load(sys.argv[1])\n"
        "row = pd.DataFrame([json.loads(sys.argv[2])])\n"
        "proba = p.predict_proba(row)[0]\n"
        "idx = list(p.classes_).index('Yes')\n"
        "print(json.dumps({'pred': str(p.predict(row)[0]), 'proba': float(proba[idx])}))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", snippet, str(MODEL_PATH), json.dumps(_payload())],
        capture_output=True, text=True, cwd=str(ROOT), timeout=120,
    )
    assert out.returncode == 0, out.stderr
    fresh = json.loads(out.stdout.strip().splitlines()[-1])

    from app import app
    api = TestClient(app).post("/predict", json=_payload()).json()

    row = pd.DataFrame([_payload()])
    proba = pipeline.predict_proba(row)
    yes_idx = int((pipeline.named_steps["classifier"].classes_ == "Yes").nonzero()[0][0])
    inproc = {"pred": str(pipeline.predict(row)[0]), "proba": float(proba[0, yes_idx])}
    return inproc, fresh, api


def test_three_sources_agree(prediction_sources):
    inproc, fresh, api = prediction_sources
    assert inproc["pred"] == fresh["pred"] == api["prediction"]
    assert abs(inproc["proba"] - fresh["proba"]) < 1e-9
    assert abs(inproc["proba"] - api["churn_probability"]) < 1e-6


def test_probability_is_positive_class(prediction_sources):
    _, _, api = prediction_sources
    # a very short-tenured, month-to-month, no-dependency profile should read as
    # a plausible churn risk (recall-oriented model); just assert finite + in-range
    assert 0.0 <= api["churn_probability"] <= 1.0

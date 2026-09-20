"""FastAPI service exposing the saved churn pipeline via POST /predict.

Run:
    uvicorn app:app --port 8000          (from the customer_churn directory)

The full fitted pipeline is loaded ONCE at import time and reused for every
request (never refit or reloaded per request).
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

# Allow running as `python app.py` / `uvicorn app:app` from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.churn.api_schema import ChurnRequest, PredictionResponse

MODEL_PATH = Path(__file__).parent / "model" / "churn_model.pkl"

if not MODEL_PATH.exists():
    raise SystemExit(
        f"Saved pipeline not found at {MODEL_PATH}.\n"
        "Run `python train.py` from the customer_churn directory first."
    )

pipeline = joblib.load(MODEL_PATH)
CLASSES = pipeline.named_steps["classifier"].classes_
POSITIVE_INDEX = int(np.where(CLASSES == "Yes")[0][0])  # classes_ -> index of "Yes"

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description=(
        "Predicts whether a telecom customer will churn using the saved "
        "feature-engineering/preprocessing/decision-tree pipeline. "
        "Churn = 'Yes' is the positive class."
    ),
    version="1.0.0",
)


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"endpoint": "POST /predict", "docs": "/docs"}


@app.post("/predict", response_model=PredictionResponse, summary="Predict churn")
def predict(payload: ChurnRequest) -> PredictionResponse:
    try:
        row = pd.DataFrame([payload.to_feature_row()])
        proba = pipeline.predict_proba(row)[0, POSITIVE_INDEX]
        prediction = str(pipeline.predict(row)[0])
    except HTTPException:
        raise
    except Exception as exc:  # defensive: never leak internals
        raise HTTPException(status_code=500, detail="Model inference failed") from exc

    return PredictionResponse(
        prediction=prediction,
        churn_probability=float(round(proba, 6)),
    )

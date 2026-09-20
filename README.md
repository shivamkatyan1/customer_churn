# Customer Churn Prediction - Decision Tree + REST API

End-to-end machine-learning solution that predicts which telecom customers are
likely to churn, built on the supplied **IBM Telco Customer Churn** dataset
(7,043 customers). The complete workflow runs in an executed Jupyter notebook,
the reusable code lives in `src/churn`, and the trained model is served over a
FastAPI endpoint.

```
Business problem → Data → Preparation → EDA → Feature engineering →
Modelling (Decision Tree) → Evaluation → Interpretation → Saved pipeline → API
```

## DEMO Video

[`DEMO_VIDEO.mp4`](DEMO_VIDEO.mp4)

## Public repository

**Public repo link (complete source for this project):**
`[⬆ fill your public GitHub repo URL here before submission]`

## Requirements

- Python **3.11**; no API keys needed.
- Internet on first run for pip packages only.
- No GPU needed; no paid service needed.
- The demo video is linked above (`DEMO_VIDEO.mp4`).

## TL;DR (commands run from this directory)

```bash
# 1) Environment
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Train + save the full pipeline (audit → split → CV selection → save)
python train.py

# 3) (Recommended) re-run the notebook top-to-bottom in a fresh kernel
jupyter nbconvert --to notebook --execute --inplace notebook/churn_analysis.ipynb

# 4) Start the API
uvicorn app:app --port 8000

# 5) Predict
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' -d @samples/sample_request.json
# → {"prediction":"Yes","churn_probability":0.931422}

# 6) Tests (deterministic, keyless)
pytest tests/
```

Interactive API docs: <http://localhost:8000/docs>.

## What is inside

| Path | Purpose |
|---|---|
| `notebook/churn_analysis.ipynb` | **Executed** end-to-end analysis: audit, EDA (6 figures), features, tree comparison + selection, metrics, interpretation, save |
| `src/churn/features.py` | Shared transformers: `internet_addon_count`, `tenure_band`, zero-tenure `TotalCharges` rule, frame cleaning |
| `src/churn/preprocessing.py` | Column layout, ordinal encoding policy, `build_pipeline()` |
| `src/churn/models.py` | Candidate Decision Tree configs, seeded 5-fold CV selection, test evaluation, feature importance |
| `src/churn/splitting.py` | Stratified 70:30 split, `random_state=42`, disjointness reporting |
| `src/churn/api_schema.py` | Pydantic request/response + validation policy (extra fields rejected, blank-charge rule, category whitelist) |
| `src/churn/audit.py` | Dataset audit helpers |
| `train.py` | One command: audit → split → CV → refit → evaluate → save pipeline + `model/metadata.json` |
| `app.py` | FastAPI `POST /predict` (model loaded once) |
| `model/churn_model.pkl` | Complete fitted pipeline (`clean → preprocess → classifier`) + `metadata.json` |
| `samples/` | Real request/response + invalid payloads (see `samples/README.md`) |
| `data/` | Supplied CSV + data dictionary + `PROVENANCE.md` (SHA-256) |
| `tests/` | Unit + parity + API tests (`pytest tests/`) |

## Key decisions (all detailed in the notebook)

- **Split first, no leakage**: 70:30, stratified on `Churn`, `random_state=42`
  (4,930 train / 2,113 test - the actual integer sizes). All learned
  preprocessing is fitted on training folds only and shipped **inside** the
  saved pipeline, so new/unseen data is transformed identically.
- **Missing `TotalCharges`** (11 rows, every one with `tenure=0`): domain rule - zero-tenure customers have not accumulated charges, so blank ⇒ `0.0`. Blank
  with `tenure>0` is invalid input (the API rejects it, 422).
- **Categoricals**: `No internet service` / `No phone service` are real
  categories (never null); `SeniorCitizen` is binary; ordinal-encoded with an
  explicit unseen-category policy (`unknown_value=-1` at model level, and the
  API rejects unknown levels with a clear message).
- **Features**: `internet_addon_count` (0–6 add-ons) and `tenure_band`
  (0–12/13–24/25–48/49+ months), shared importable transformers.
- **Model**: compared unconstrained vs pruned (`max_depth=6, min_samples_leaf=5`)
  vs pruned + `class_weight="balanced"` using 5-fold stratified CV on the
  training partition (selection rule: best positive-class F1). Winner:
  **pruned + balanced**, refit on training, evaluated once on the untouched
  test set - ≈0.74 accuracy, **0.77 recall**, 0.50 precision (recall preferred
  for a retention campaign; rationale in the notebook).
- **Saved**: full pipeline + metadata (versions, seed, schema, split/CV/test
  metrics); loaded in a fresh process in `tests/test_parity.py`.

## API contract - `POST /predict`

- **Body**: JSON with the 19 predictor fields (schema in `src/churn/api_schema.py`).
  `customerID` is optional and ignored. `TotalCharges` may be omitted only when
  `tenure == 0`. Types/ranges/category values are validated; unknown or
  extra fields (including `Churn`) are rejected with HTTP 422 and a clear message.
- **Response** (200):
  ```json
  { "prediction": "Yes", "churn_probability": 0.931422 }
  ```
  `churn_probability` is the model-estimated probability of `Churn = "Yes"`
  (positive class), finite and in [0,1], resolved through the pipeline's
  `classes_` mapping.

Real examples (valid + invalid) are in `samples/` and were captured from a
live local run - see `samples/README.md`.

## Notes

- Requires Python 3.11; no API keys needed.
- Ranges used for validation sanity checks are documented in
  `src/churn/preprocessing.py` (the model itself saw tenure 0–72, monthly
  charges 18.25–118.75, total charges 0–8684.8).
- Model outputs are estimates for prioritising outreach - not guarantees about
  individual behaviour.

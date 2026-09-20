# Sample API requests and recorded responses

All outputs below were captured from a real local run on **19 September 2026**
(model trained with the same seed/code in the repo). They are real responses,
not illustrative values.

## 1. Valid request — high churn risk (returns `Yes`)

Request: `samples/sample_request.json` →

```bash
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' -d @samples/sample_request.json
```

Recorded response (`samples/sample_response.json`):

```json
{"prediction": "Yes", "churn_probability": 0.931422}
```

Business read: 1-month tenure, month-to-month, fiber optic with no add-ons,
electronic check — a classic high-churn profile; retention group should contact.

## 2. Valid request — low churn risk (returns `No`)

```bash
curl -s -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "gender":"Female","SeniorCitizen":1,"Partner":"Yes","Dependents":"Yes","tenure":72,
  "PhoneService":"Yes","MultipleLines":"Yes","InternetService":"DSL","OnlineSecurity":"Yes",
  "OnlineBackup":"Yes","DeviceProtection":"Yes","TechSupport":"Yes","StreamingTV":"No",
  "StreamingMovies":"Yes","Contract":"Two year","PaperlessBilling":"No",
  "PaymentMethod":"Bank transfer (automatic)","MonthlyCharges":89.9,"TotalCharges":6472.8
}'
```

Recorded response:

```json
{"prediction": "No", "churn_probability": 0.0}
```

Business read: 6-year customer on a two-year contract with automatic bank
transfer and multiple add-ons — very low churn probability.

## 3. Invalid requests — all rejected with HTTP 422 + clear messages

| Case | Behaviour observed |
|---|---|
| Unknown category (`"Contract": "Lifetime"`) | `422` — `unknown value 'Lifetime' for Contract; allowed: ['Month-to-month', 'One year', 'Two year']` |
| Omitted `TotalCharges` with `tenure=72` | `422` — `TotalCharges may be omitted only when tenure == 0 (…zero-tenure rule)` |
| `"Churn"` field leaked into the body | `422` — `Extra inputs are not permitted` (target/unexpected fields rejected) |
| `"MonthlyCharges": -5` | `422` — bound violation |
| Missing required field (e.g. no `gender`) | `422` — field required |

See `samples/invalid_payloads/` for runnable example payloads.

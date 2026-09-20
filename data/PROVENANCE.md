# Data provenance

| File | SHA-256 | Bytes | Rows (excl. header) | Columns |
|---|---|---|---|---|
| `TelcoCustomerChurn.csv` | `3d5c233415c1b42bdea7172c73e620819f507f0a8294bc2337a1d8a8877feef0` | 970456 | 7,043 | 21 |
| `TelcoCustomerChurn - Data Dictionary.csv` | (see `shasum -a 256` output at copy time) | 2004 | 22 (header + 21 field rows) | 3 |

- Files were supplied with the assessment package on 16 Sep 2026 and copied here unchanged on **18 Sep 2026**.
- `TelcoCustomerChurn.csv` is the IBM Telco Customer Churn dataset as supplied; no online substitute was used.
- The 11 blank `TotalCharges` values and all other facts stated in `notebook/churn_analysis.ipynb` come from this exact file.
- Publication: this is the dataset assigned for the exercise and included so the projects run out-of-the-box; see the runbook for any organisational data-publication policy.

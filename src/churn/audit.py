"""Read-only data audit helpers used by ``train.py`` and the notebook.

Everything here is descriptive; it never mutates the source file.
"""

from __future__ import annotations

import pandas as pd


def audit_dataframe(data: pd.DataFrame) -> dict:
    """Compute the dataset-level facts needed by section DS-1 of the brief."""
    n_rows, n_cols = data.shape
    full_dupes = int(data.duplicated().sum())
    id_dupes = int(data.duplicated(subset=["customerID"]).sum()) if "customerID" in data else None

    blanks = {}
    for col in data.columns:
        # blank incl. whitespace-only, and explicit NaN
        missing = data[col].isna()
        if data[col].dtype == object:
            missing = missing | data[col].astype(str).str.strip().eq("")
        blanks[col] = int(missing.sum())

    target_col = "Churn"
    target = data[target_col].value_counts().to_dict() if target_col in data else None
    target_pct = round(data[target_col].value_counts(normalize=True).get("Yes", 0) * 100, 2)

    numeric = {c: round(float(data[c].min()), 2) for c in data.select_dtypes("number").columns}
    numeric_max = {c: round(float(data[c].max()), 2) for c in data.select_dtypes("number").columns}

    return {
        "n_rows": n_rows,
        "n_columns": n_cols,
        "duplicate_rows": full_dupes,
        "duplicate_customer_ids": id_dupes,
        "blank_counts": blanks,
        "target_distribution": target,
        "target_positive_pct": target_pct,
        "numeric_min": numeric,
        "numeric_max": numeric_max,
    }


def blank_totalcharges_with_tenure_zero(data: pd.DataFrame) -> int:
    """Confirm every blank ``TotalCharges`` row has ``tenure == 0``."""
    tc = pd.to_numeric(data["TotalCharges"], errors="coerce")
    blank = tc.isna()
    tenure = pd.to_numeric(data["tenure"], errors="coerce")
    return int((blank & (tenure != 0)).sum())


def format_audit(report: dict) -> str:
    """Render the audit dict as a readable text block."""
    lines = [
        f"Rows x columns : {report['n_rows']} x {report['n_columns']}",
        f"Duplicate rows : {report['duplicate_rows']}",
        f"Duplicate ids  : {report['duplicate_customer_ids']}",
        f"Churn          : No={report['target_distribution'].get('No')}, "
        f"Yes={report['target_distribution'].get('Yes')} "
        f"({report['target_positive_pct']}% positive)",
        "Blanks per column: " + ", ".join(f"{c}={v}" for c, v in report["blank_counts"].items() if v),
    ]
    return "\n".join(lines)

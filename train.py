"""End-to-end training entry point.

Audit -> stratified split -> cross-validated decision-tree selection ->
refit on the training partition -> one evaluation on the untouched test set ->
save the complete fitted pipeline + metadata.

Usage:
    python train.py                     # uses data/TelcoCustomerChurn.csv
    python train.py --data path.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as SKLEARN_VERSION

# Allow running as `python train.py` from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.churn import audit as audit_mod
from src.churn.models import (
    BONUS_CONFIGS,
    CANDIDATE_CONFIGS,
    POSITIVE_LABEL,
    evaluate_on_test,
    feature_importance,
    make_classifier,
    run_cross_validation,
    select_best,
)
from src.churn.preprocessing import build_pipeline, feature_names
from src.churn.splitting import load_data, split_report, stratify_split

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "TelcoCustomerChurn.csv"
MODEL_DIR = ROOT / "model"
PIPELINE_PATH = MODEL_DIR / "churn_model.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def main(data_path: Path = DATA_PATH) -> None:
    data = load_data(data_path)
    audit_report = audit_mod.audit_dataframe(data)
    print(audit_mod.format_audit(audit_report))
    assert audit_report["duplicate_rows"] == 0, "unexpected duplicate rows"
    assert audit_mod.blank_totalcharges_with_tenure_zero(data) == 0, (
        "a blank TotalCharges row has tenure != 0 (rule would be violated)"
    )

    # split (before any learned statistic) --
    X_train, X_test, y_train, y_test, id_train, id_test = stratify_split(data)
    split = split_report(X_train, X_test, y_train, y_test, id_train, id_test)
    print("Split:", split["n_train"], "train /", split["n_test"], "test")

    # cross-validated selection on training partition only --
    candidates = list(CANDIDATE_CONFIGS.items()) + list(BONUS_CONFIGS.items())
    results = []
    for name, params in candidates:
        def build(cfg=params):
            return build_pipeline(make_classifier(cfg))

        res = run_cross_validation(build, X_train, y_train)
        res.name = name
        res.params = params
        results.append(res)
        print(
            f"CV {name:32s} acc={res.mean.accuracy:.4f} prec={res.mean.precision:.4f} "
            f"rec={res.mean.recall:.4f} f1={res.mean.f1:.4f}"
        )

    chosen = select_best(results)
    print("Selected:", chosen.name)

    # refit the winner on the full training partition --
    final = build_pipeline(make_classifier(chosen.params))
    final.fit(X_train, y_train)

    # single evaluation on the untouched test set --
    test_metrics = evaluate_on_test(final, X_test, y_test)
    print("Test metrics:", json.dumps(test_metrics, indent=2))
    imp = feature_importance(final, feature_names(final.named_steps["preprocess"]))

    # save pipeline + metadata --
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final, PIPELINE_PATH)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            "scikit-learn": SKLEARN_VERSION,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "seed": 42,
        "data": {
            "path": str(data_path),
            "sha256": sha256(Path(data_path)),
            "n_rows": audit_report["n_rows"],
            "n_columns": audit_report["n_columns"],
            "churn_positive_pct": audit_report["target_positive_pct"],
        },
        "split": split,
        "model_selection": {
            "cv": "StratifiedKFold(5, shuffle=True, random_state=42)",
            "selection_metric": "mean positive-class F1 on training folds",
            "candidates": [
                {"name": r.name, "params": r.params, "mean_f1": r.mean.f1,
                 "mean_precision": r.mean.precision, "mean_recall": r.mean.recall,
                 "mean_accuracy": r.mean.accuracy}
                for r in results
            ],
            "selected": chosen.name,
        },
        "test_metrics": test_metrics,
        "feature_names": feature_names(final.named_steps["preprocess"]),
        "class_mapping": {label: int(index) for index, label in enumerate(final.classes_)},
        "positive_class": POSITIVE_LABEL,
        "pipeline_file": PIPELINE_PATH.name,
        "feature_importance_top10": imp[:10],
        "input_schema": "see src/churn/api_schema.py ChurnRequest",
    }
    with open(METADATA_PATH, "w") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"\nSaved pipeline -> {PIPELINE_PATH}")
    print(f"Saved metadata -> {METADATA_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DATA_PATH), help="path to the Telco CSV")
    args = parser.parse_args()
    sys.exit(main(Path(args.data)))

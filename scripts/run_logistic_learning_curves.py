"""Run ADA-ML-08 Logistic Regression learning curves on training data only."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "santander-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "santander-cache"))

from src.data import load_dataset  # noqa: E402
from src.learning_curves import (  # noqa: E402
    compute_learning_curve,
    create_learning_curve_figures,
    summarize_learning_curve,
)
from src.modeling import create_logistic_regression_pipeline  # noqa: E402
from src.validation import create_stratified_cv, create_train_test_split, split_fingerprint  # noqa: E402

EXPECTED_TRAIN = "61c403ec521d15ab9d6316606eba5acdfc22381cb764b6da76a65041ec11f477"
EXPECTED_TEST = "bf7d43e492967dad1358676e9bf8355a910823077b59118014db12af3e26f586"
FRACTIONS = (.05, .10, .25, .50, .75, 1.00)
TABLE_DIR = PROJECT_ROOT / "reports/tables"
FIGURE_DIR = PROJECT_ROOT / "reports/figures"
FOLDS_PATH = TABLE_DIR / "logistic_learning_curve_folds.csv"
SUMMARY_PATH = TABLE_DIR / "logistic_learning_curve_summary.csv"
SUMMARY_JSON_PATH = TABLE_DIR / "logistic_learning_curve_summary.json"
DECISION_PATH = TABLE_DIR / "logistic_learning_curve_decision.csv"
FIGURE_PATHS = tuple(FIGURE_DIR / name for name in (
    "logistic_learning_curve_roc_auc.pdf",
    "logistic_learning_curve_average_precision.pdf",
    "logistic_learning_curve_fit_time.pdf",
    "logistic_learning_curve_threshold_metrics.pdf",
))
OUTPUTS = (FOLDS_PATH, SUMMARY_PATH, SUMMARY_JSON_PATH, DECISION_PATH, *FIGURE_PATHS)


def refuse_existing_outputs() -> None:
    """Protect completed scientific artifacts from accidental replacement."""
    existing = [path.relative_to(PROJECT_ROOT).as_posix() for path in OUTPUTS if path.exists()]
    if existing:
        raise FileExistsError("Learning-curve outputs already exist and were not overwritten:\n- " + "\n- ".join(existing))


def _strict_records(table: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(table.to_json(orient="records"))


def build_decision_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Build observed incremental-gain and interpretation rows."""
    rows: list[dict[str, Any]] = []
    for configuration, group in summary.groupby("configuration_id", sort=True):
        group = group.sort_values("train_fraction").reset_index(drop=True)
        total_roc = float(group.iloc[-1]["validation_roc_auc_mean"] - group.iloc[0]["validation_roc_auc_mean"])
        total_ap = float(group.iloc[-1]["validation_average_precision_mean"] - group.iloc[0]["validation_average_precision_mean"])
        gain_75_roc = float(group.iloc[-1]["validation_roc_auc_mean"] - group[group["train_fraction"].eq(.75)].iloc[0]["validation_roc_auc_mean"])
        gain_75_ap = float(group.iloc[-1]["validation_average_precision_mean"] - group[group["train_fraction"].eq(.75)].iloc[0]["validation_average_precision_mean"])
        previous_roc: float | None = None
        previous_ap: float | None = None
        for _, item in group.iterrows():
            roc = float(item["validation_roc_auc_mean"])
            ap = float(item["validation_average_precision_mean"])
            incremental_roc = np.nan if previous_roc is None else roc - previous_roc
            incremental_ap = np.nan if previous_ap is None else ap - previous_ap
            if previous_roc is None:
                interpretation = "Initial observed training fraction; no preceding gain."
            elif item["train_fraction"] == 1.0:
                interpretation = (
                    f"Observed 75%→100% gains: ROC-AUC {gain_75_roc:+.6f}, AP {gain_75_ap:+.6f}; "
                    + ("late gains are small (empirical plateau)." if abs(gain_75_roc) < .001 and abs(gain_75_ap) < .002 else "performance is still changing materially.")
                )
            else:
                interpretation = f"Observed gain from preceding fraction: ROC-AUC {incremental_roc:+.6f}, AP {incremental_ap:+.6f}."
            rows.append({
                "configuration_id": configuration,
                "train_fraction": float(item["train_fraction"]),
                "train_size": float(item["train_size_mean"]),
                "validation_roc_auc": roc,
                "validation_average_precision": ap,
                "validation_f1": float(item["validation_f1_mean"]),
                "validation_precision": float(item["validation_precision_mean"]),
                "validation_recall": float(item["validation_recall_mean"]),
                "validation_balanced_accuracy": float(item["validation_balanced_accuracy_mean"]),
                "generalization_gap": float(item["roc_auc_generalization_gap"]),
                "fit_time": float(item["fit_time_mean"]),
                "incremental_roc_auc_gain": incremental_roc,
                "incremental_average_precision_gain": incremental_ap,
                "roc_auc_gain_5_to_100": total_roc,
                "average_precision_gain_5_to_100": total_ap,
                "roc_auc_gain_75_to_100": gain_75_roc,
                "average_precision_gain_75_to_100": gain_75_ap,
                "interpretation": interpretation,
            })
            previous_roc, previous_ap = roc, ap
    return pd.DataFrame(rows)


def main() -> None:
    refuse_existing_outputs()
    X, y, metadata = load_dataset(optimize_memory=True)
    if metadata["openml_id"] != 45566 or X.shape != (200000, 200):
        raise RuntimeError(f"Unexpected dataset identity/shape: {metadata['openml_id']=}, {X.shape=}.")
    if not all(dtype == np.dtype("float32") for dtype in X.dtypes):
        raise RuntimeError("Memory optimization failed: every feature must be float32.")
    X_train, X_reserved, y_train, y_reserved = create_train_test_split(X, y)
    train_hash = split_fingerprint(X_train.index)
    reserved_hash = split_fingerprint(X_reserved.index)
    if train_hash != EXPECTED_TRAIN or reserved_hash != EXPECTED_TEST:
        raise RuntimeError(f"Shared split fingerprint mismatch: train={train_hash}, reserved={reserved_hash}")
    del X_reserved, y_reserved, X, y

    configurations = {
        "LR-LEARNING-ROC": None,
        "LR-LEARNING-BALANCED": "balanced",
    }
    cv = create_stratified_cv(n_splits=5, shuffle=True, random_state=42)
    started = time.perf_counter()
    fold_tables = []
    for configuration_id, class_weight in configurations.items():
        print(f"Running {configuration_id} ...", flush=True)
        estimator = create_logistic_regression_pipeline(
            penalty="l2", C=.01, class_weight=class_weight, solver="saga",
            max_iter=2000, random_state=42,
        )
        fold_tables.append(compute_learning_curve(
            estimator, X_train, y_train, configuration_id=configuration_id,
            train_size_fractions=FRACTIONS, cv=cv, random_state=42,
        ))
    duration = time.perf_counter() - started
    folds = pd.concat(fold_tables, ignore_index=True)
    summary = summarize_learning_curve(folds)
    decision = build_decision_table(summary)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    folds.to_csv(FOLDS_PATH, index=False, encoding="utf-8")
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8")
    SUMMARY_JSON_PATH.write_text(json.dumps({
        "metadata": {
            "ticket_id": "ADA-ML-08", "openml_id": 45566,
            "train_fingerprint": train_hash,
            "reserved_test_fingerprint_verified_only": reserved_hash,
            "reserved_test_usage": "Fingerprint verification only; never scored or predicted.",
            "fractions": list(FRACTIONS), "cv_folds": 5,
            "duration_seconds": duration,
        },
        "results": _strict_records(summary),
    }, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    decision.to_csv(DECISION_PATH, index=False, encoding="utf-8")
    create_learning_curve_figures(summary, FIGURE_DIR)

    print(f"Campaign duration: {duration:.2f} seconds")
    print(f"Converged fits: {(~folds['convergence_warning']).sum()}/{len(folds)}")
    print(summary[["configuration_id", "train_fraction", "train_size_mean", "validation_roc_auc_mean", "validation_average_precision_mean", "roc_auc_generalization_gap", "fit_time_mean"]].to_string(index=False))
    print("Reserved final test: fingerprint verification only; no metric or prediction used it.")


if __name__ == "__main__":
    main()

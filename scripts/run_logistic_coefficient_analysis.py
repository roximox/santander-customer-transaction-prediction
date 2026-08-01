"""Run the four-configuration, training-only Logistic coefficient audit."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "santander-matplotlib"))

from src.data import load_dataset  # noqa: E402
from src.logistic_coefficient_analysis import (  # noqa: E402
    CONFIGURATIONS,
    audit_configurations_by_fold,
    create_convergence_figure,
    create_l1_sparsity_figure,
    create_stability_figure,
    create_top_coefficients_figure,
    save_analysis_figure,
    strict_json_records,
    summarize_coefficient_stability,
    summarize_l1_sparsity,
)
from src.validation import create_train_test_split, split_fingerprint  # noqa: E402

EXPECTED_TRAIN = "61c403ec521d15ab9d6316606eba5acdfc22381cb764b6da76a65041ec11f477"
EXPECTED_TEST = "bf7d43e492967dad1358676e9bf8355a910823077b59118014db12af3e26f586"
TABLES = PROJECT_ROOT / "reports/tables"
FIGURES = PROJECT_ROOT / "reports/figures"
OUTPUTS = [
    TABLES / "logistic_convergence_audit.csv", TABLES / "logistic_convergence_audit.json",
    TABLES / "logistic_coefficients_by_fold.csv", TABLES / "logistic_coefficient_stability.csv",
    TABLES / "logistic_l1_sparsity_summary.csv", TABLES / "logistic_l1_sparsity_summary.json",
    TABLES / "logistic_model_selection_summary.csv",
    FIGURES / "logistic_top_coefficients.pdf", FIGURES / "logistic_coefficient_stability.pdf",
    FIGURES / "logistic_l1_sparsity.pdf", FIGURES / "logistic_convergence_iterations.pdf",
]


def _refuse_existing() -> None:
    existing = [path.relative_to(PROJECT_ROOT).as_posix() for path in OUTPUTS if path.exists()]
    if existing:
        raise FileExistsError("Coefficient-audit outputs already exist and were not overwritten:\n- " + "\n- ".join(existing))


def _save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _selection_summary(audit: pd.DataFrame, coefficients: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    intended = {
        "LR-SELECTED-ROC": "primary ROC-AUC selected candidate",
        "LR-SELECTED-AP": "Average Precision and sparse alternative",
        "LR-SELECTED-BALANCED": "recall-oriented default-threshold alternative",
        "LR-L1-WEAK-REG": "weak-regularization stability stress test",
    }
    rows = []
    for configuration_id, group in audit.groupby("configuration_id", sort=False):
        coef = coefficients[coefficients["configuration_id"] == configuration_id]
        stable = stability[stability["configuration_id"] == configuration_id]
        is_l1 = CONFIGURATIONS[configuration_id]["regularization"] == "l1"
        rows.append({
            "configuration_id": configuration_id,
            "ROC-AUC": group["validation_roc_auc"].mean(),
            "Average Precision": group["validation_average_precision"].mean(),
            "F1": group["validation_f1"].mean(),
            "Balanced Accuracy": group["validation_balanced_accuracy"].mean(),
            "convergence_rate": 1.0 - group["convergence_warning"].mean(),
            "mean_n_iter": group["n_iter"].mean(),
            "nonzero_coefficient_count": (~coef["is_zero"]).groupby(coef["fold"]).sum().mean(),
            "selection_stability": stable["selection_frequency"].mean() if is_l1 else stable["sign_consistency"].mean(),
            "fit_time": group["fit_time"].mean(), "intended_use": intended[configuration_id],
            "strengths": "supports exact zeros; weak sparsity observed" if is_l1 else "stable dense ranking model",
            "limitations": "default-threshold trade-off; non-causal coefficients" + ("; convergence must be checked" if group["convergence_warning"].any() else ""),
        })
    return pd.DataFrame(rows)


def main() -> None:
    _refuse_existing()
    X, y, _ = load_dataset(optimize_memory=True)
    X_train, X_reserved, y_train, y_reserved = create_train_test_split(X, y)
    if split_fingerprint(X_train.index) != EXPECTED_TRAIN or split_fingerprint(X_reserved.index) != EXPECTED_TEST:
        raise RuntimeError("Common split fingerprint mismatch; audit stopped.")
    del X_reserved, y_reserved, X, y
    print(f"Auditing {len(CONFIGURATIONS)} configurations × 5 folds on {X_train.shape}; final test reserved.")
    audit, coefficients = audit_configurations_by_fold(X_train, y_train)
    stability = summarize_coefficient_stability(coefficients)
    sparsity, sparsity_details = summarize_l1_sparsity(coefficients)
    selection = _selection_summary(audit, coefficients, stability)

    TABLES.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUTPUTS[0], index=False); _save_json(OUTPUTS[1], strict_json_records(audit))
    coefficients.to_csv(OUTPUTS[2], index=False); stability.to_csv(OUTPUTS[3], index=False)
    sparsity.to_csv(OUTPUTS[4], index=False)
    _save_json(OUTPUTS[5], {"fold_summary": strict_json_records(sparsity), "configuration_summary": sparsity_details})
    selection.to_csv(OUTPUTS[6], index=False)
    save_analysis_figure(create_top_coefficients_figure(stability), OUTPUTS[7])
    save_analysis_figure(create_stability_figure(stability), OUTPUTS[8])
    save_analysis_figure(create_l1_sparsity_figure(sparsity), OUTPUTS[9])
    save_analysis_figure(create_convergence_figure(audit), OUTPUTS[10])

    print("\nConvergence summary:")
    print(selection[["configuration_id", "convergence_rate", "mean_n_iter", "fit_time", "ROC-AUC", "Average Precision"]].to_string(index=False))
    print("\nL1 sparsity by fold:")
    print(sparsity.to_string(index=False))
    print("\nL1 support summary:")
    for key, value in sparsity_details.items():
        print(f"{key}: intersection={value['intersection_feature_count']}, union={value['union_feature_count']}")
    top = stability[stability["configuration_id"] == "LR-SELECTED-ROC"].nlargest(20, "mean_absolute_coefficient")
    print("\nTop standardized coefficients for LR-SELECTED-ROC:")
    print(top[["feature", "mean_coefficient", "std_coefficient", "sign_consistency", "stability_ratio"]].to_string(index=False))
    print("Final test partition remained reserved and was not evaluated.")


if __name__ == "__main__":
    main()

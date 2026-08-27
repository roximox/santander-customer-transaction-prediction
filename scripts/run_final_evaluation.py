"""Run the single reserved-test evaluation for the collectively locked model."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_dataset  # noqa: E402
from src.final_evaluation import evaluate_fitted_model_once  # noqa: E402
from src.validation import create_train_test_split, split_fingerprint  # noqa: E402

LOCK_PATH = PROJECT_ROOT / "reports/model_selection/final_model_lock.json"
SUMMARY_PATH = PROJECT_ROOT / "reports/experiments/M04-HGB-002_summary.json"
OUTPUT_DIR = PROJECT_ROOT / "reports/final_evaluation"
RESULT_PATH = OUTPUT_DIR / "M04-HGB-002_final_test_results.json"


def load_and_validate_lock() -> tuple[dict, dict]:
    """Refuse execution unless the collective lock and source summary agree."""
    if RESULT_PATH.exists():
        raise FileExistsError(
            f"Final-test result already exists at {RESULT_PATH.relative_to(PROJECT_ROOT)}; "
            "a second evaluation is prohibited."
        )
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if lock.get("selected_experiment_id") != "M04-HGB-002":
        raise RuntimeError("The collective lock does not select M04-HGB-002.")
    if lock.get("status") != "locked_pending_final_test" or lock.get("final_test_used") is not False:
        raise RuntimeError("The final-test lock is not in its single-use pre-evaluation state.")
    if summary.get("experiment_id") != lock["selected_experiment_id"]:
        raise RuntimeError("Locked experiment and registered summary do not match.")
    for name, value in lock["estimator_parameters"].items():
        if summary["estimator_parameters"].get(name) != value:
            raise RuntimeError(f"Locked parameter mismatch for {name}.")
    return lock, summary


def main() -> None:
    """Fit the frozen pipeline on development data and score the reserve once."""
    lock, summary = load_and_validate_lock()
    X, y, metadata = load_dataset(optimize_memory=True)
    X_train, X_final, y_train, y_final = create_train_test_split(X, y)
    del X, y
    train_hash = split_fingerprint(X_train.index)
    final_hash = split_fingerprint(X_final.index)
    protocol = lock["data_protocol"]
    if train_hash != protocol["train_fingerprint"]:
        raise RuntimeError(f"Development fingerprint mismatch: {train_hash}.")
    if final_hash != protocol["reserved_test_fingerprint"]:
        raise RuntimeError(f"Reserved-test fingerprint mismatch: {final_hash}.")

    estimator = HistGradientBoostingClassifier(**lock["estimator_parameters"])
    started = time.perf_counter()
    estimator.fit(X_train, y_train)
    fit_time = time.perf_counter() - started
    metrics = evaluate_fitted_model_once(
        estimator, X_final, y_final, threshold=float(lock["classification_threshold"])
    )
    result = {
        "evaluation_id": "FINAL-M04-HGB-002-001",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": lock["selected_experiment_id"],
        "model_name": lock["model_name"],
        "member": lock["member"],
        "status": "completed",
        "execution_count": 1,
        "selection_reopened": False,
        "n_train": int(len(X_train)),
        "n_final_test": int(len(X_final)),
        "n_features": int(X_train.shape[1]),
        "train_fingerprint": train_hash,
        "final_test_fingerprint": final_hash,
        "numeric_dtype": metadata.get("numeric_dtype"),
        "estimator_class": summary["estimator_class"],
        "estimator_parameters": lock["estimator_parameters"],
        "fit_time_seconds": float(fit_time),
        "metrics": metrics,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lock["status"] = "final_test_completed"
    lock["final_test_used"] = True
    lock["final_test_status"] = "COMPLETED_ONCE"
    lock["final_evaluation_id"] = result["evaluation_id"]
    lock["final_result_file"] = RESULT_PATH.relative_to(PROJECT_ROOT).as_posix()
    LOCK_PATH.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    print(f"Final evaluation completed once for {result['experiment_id']}.")
    for name, value in metrics.items():
        if name != "confusion_matrix":
            print(f"  {name}: {value:.6f}")
    print(f"  confusion_matrix: {metrics['confusion_matrix']}")
    print(f"Result: {RESULT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

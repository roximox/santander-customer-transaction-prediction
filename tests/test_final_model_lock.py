"""Integrity checks for the collective final-model decision."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "reports/model_selection/final_model_lock.json"
SUMMARY_PATH = ROOT / "reports/experiments/M04-HGB-002_summary.json"


def test_lock_matches_the_selected_experiment() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    assert lock["selected_experiment_id"] == summary["experiment_id"] == "M04-HGB-002"
    assert lock["model_name"] == summary["model_name"]
    assert lock["member"] == summary["member"]
    assert lock["selection_basis"]["roc_auc_mean"] == summary["metrics"]["roc_auc"]["validation_mean"]
    assert lock["selection_basis"]["average_precision_mean"] == summary["metrics"]["average_precision"]["validation_mean"]
    for parameter, value in lock["estimator_parameters"].items():
        assert summary["estimator_parameters"][parameter] == value


def test_lock_records_exactly_one_completed_final_evaluation() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))

    assert lock["collective_confirmation"] is True
    assert lock["status"] == "final_test_completed"
    assert lock["final_test_used"] is True
    assert lock["final_test_status"] == "COMPLETED_ONCE"
    assert lock["classification_threshold"] == 0.5
    assert (ROOT / lock["meeting_record"]).is_file()
    result = json.loads((ROOT / lock["final_result_file"]).read_text(encoding="utf-8"))
    assert result["execution_count"] == 1
    assert result["selection_reopened"] is False
    assert result["experiment_id"] == lock["selected_experiment_id"]
    assert not any(str(value).startswith("/") for value in lock.values())

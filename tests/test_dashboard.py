"""Offline tests for the read-only Streamlit dashboard."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from streamlit.testing.v1 import AppTest

from src.dashboard import charts
from src.dashboard.components import arrow_safe_dataframe
from src.dashboard.loaders import (
    filter_experiments, load_csv, load_experiment_summary, load_experiments,
    load_json, load_learning_curves, load_model_comparison, load_registry,
    load_selection_outputs, selection_status,
)


ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_registry_and_summary_loaders(tmp_path: Path) -> None:
    registry = pd.DataFrame([{
        "experiment_id": "M02-RF-001", "member": "Member 02",
        "model_name": "Random Forest", "status": "completed",
        "summary_file": "reports/experiments/M02-RF-001_summary.json",
    }])
    path = tmp_path / "reports/experiments/experiment_registry.csv"
    path.parent.mkdir(parents=True)
    registry.to_csv(path, index=False)
    summary = {"experiment_id": "M02-RF-001", "member": "Member 02", "model_name": "Random Forest"}
    _write(path.with_name("M02-RF-001_summary.json"), json.dumps(summary))
    assert load_registry(tmp_path).iloc[0]["experiment_id"] == "M02-RF-001"
    assert load_experiment_summary("M02-RF-001", tmp_path) == summary


def test_missing_and_invalid_artifacts_are_non_fatal(tmp_path: Path) -> None:
    missing = load_csv("missing.csv", ("required",), tmp_path)
    assert missing.empty and "not available" in missing.attrs["error"]
    _write(tmp_path / "broken.json", "{")
    assert "_error" in load_json("broken.json", tmp_path)
    _write(tmp_path / "wrong.csv", "other\n1\n")
    wrong = load_csv("wrong.csv", ("required",), tmp_path)
    assert wrong.empty and "missing columns" in wrong.attrs["error"]


def test_experiment_discovery_supports_future_member_02(tmp_path: Path) -> None:
    summary = {
        "experiment_id": "M02-RF-001", "member": "Member 02",
        "model_name": "Random Forest", "primary_metric": "roc_auc",
        "primary_score_mean": .87, "status": "completed", "final_test_used": False,
        "metrics": {"roc_auc": {"validation_mean": .87, "validation_std": .01}},
    }
    _write(tmp_path / "reports/experiments/M02-RF-001_summary.json", json.dumps(summary))
    frame = load_experiments(tmp_path)
    assert frame.iloc[0]["model_family"] == "RANDOM_FOREST"
    assert bool(frame.iloc[0]["final_test_used"]) is False


def test_filtering_sorting_and_dummy_control() -> None:
    frame = pd.DataFrame([
        {"experiment_id": "D", "member": "Member 01", "model_family": "DUMMY", "roc_auc_mean": .5},
        {"experiment_id": "H", "member": "Member 04", "model_family": "HIST_GRADIENT_BOOSTING", "roc_auc_mean": .89},
        {"experiment_id": "P", "member": "Member 03", "model_family": "PCA", "roc_auc_mean": .86},
    ])
    filtered = filter_experiments(frame, members=["Member 04"], include_dummy=False)
    assert filtered["experiment_id"].tolist() == ["H"]
    assert filter_experiments(frame, include_dummy=False).sort_values("roc_auc_mean", ascending=False).iloc[0]["experiment_id"] == "H"


def test_learning_curve_and_selection_loaders(tmp_path: Path) -> None:
    logistic = tmp_path / "reports/tables/logistic_learning_curve_summary.csv"
    logistic.parent.mkdir(parents=True)
    pd.DataFrame([{"configuration_id": "LR", "train_size_mean": 10, "train_roc_auc_mean": .8, "validation_roc_auc_mean": .7}]).to_csv(logistic, index=False)
    _write(tmp_path / "reports/model_selection/model_selection_summary.json", json.dumps({"selection_status": "waiting_for_additional_models", "missing_expected_model_families": ["RANDOM_FOREST", "EXTRA_TREES"], "final_test_used": False}))
    _write(tmp_path / "reports/model_selection/final_model_lock.json", json.dumps({"selected_experiment_id": "M04-HGB-002", "final_test_used": False}))
    curves = load_learning_curves(tmp_path)
    assert len(curves["Logistic Regression"]) == 1
    assert curves["HistGradientBoosting Tuned"].empty
    outputs = load_selection_outputs(tmp_path)
    assert outputs["summary"]["final_test_used"] is False
    assert selection_status(tmp_path) == "waiting_for_additional_models"
    assert outputs["summary"]["missing_expected_model_families"] == ["RANDOM_FOREST", "EXTRA_TREES"]
    assert outputs["lock"]["selected_experiment_id"] == "M04-HGB-002"
    assert outputs["lock"]["final_test_used"] is False


def test_model_comparison_loader_validates_columns(tmp_path: Path) -> None:
    path = tmp_path / "reports/model_selection/model_comparison_eligible.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame([{"experiment_id": "X"}]).to_csv(path, index=False)
    assert load_model_comparison(root=tmp_path).empty


def test_chart_builders_return_plotly_figures() -> None:
    comparison = pd.DataFrame([{
        "experiment_id": "M04-HGB-002", "member": "Member 04",
        "model_name": "HGB", "model_family": "HIST_GRADIENT_BOOSTING",
        "roc_auc_mean": .89, "roc_auc_std": .01, "fit_time_mean": 10,
        "precision_mean": .7, "recall_mean": .3, "f1_mean": .42,
        "balanced_accuracy_mean": .64,
    }])
    folds = pd.DataFrame([{"fold": 1, "validation_roc_auc": .89, "validation_average_precision": .6, "validation_f1": .42}])
    curve = pd.DataFrame([{"train_size": 100, "train_roc_auc_mean": .95, "validation_roc_auc_mean": .89}])
    figures = (
        charts.metric_ranking(comparison, "roc_auc"), charts.roc_auc_vs_time(comparison),
        charts.threshold_metrics(comparison), charts.fold_metrics(folds, "M04-HGB-002"),
        charts.learning_curve(curve, "Curve"), charts.class_distribution({"False": .9, "True": .1}),
        charts.memory_comparison(300, 150), charts.empty_figure(),
    )
    assert all(isinstance(figure, go.Figure) for figure in figures)


def test_dataframe_display_normalizes_mixed_arrow_columns() -> None:
    frame = pd.DataFrame({"Control": ["status", "count", "reopened"], "Value": ["completed", 1, False]})

    safe = arrow_safe_dataframe(frame)

    assert safe["Value"].tolist() == ["completed", "1", "False"]
    assert frame["Value"].tolist() == ["completed", 1, False]


def test_dashboard_has_no_training_or_final_test_action() -> None:
    paths = [ROOT / "app.py", *sorted((ROOT / "src/dashboard").glob("*.py"))]
    trees = [ast.parse(path.read_text(encoding="utf-8")) for path in paths]
    calls = [node for tree in trees for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(isinstance(node.func, ast.Attribute) and node.func.attr == "fit" for node in calls)
    imports = {alias.name for tree in trees for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    assert "openml" not in imports and "src.data" not in imports
    source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "Evaluate final test" not in source
    assert "Final test evaluation: LOCKED" in source
    assert "Final test evaluation: COMPLETED ONCE" in source


def test_loaders_do_not_mutate_repository_artifacts() -> None:
    targets = [ROOT / "reports/experiments/experiment_registry.csv", *sorted((ROOT / "reports/experiments").glob("*_summary.json"))]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in targets}
    load_registry(); load_experiments(); load_model_comparison(); load_selection_outputs(); load_learning_curves()
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in targets}
    assert before == after


def test_streamlit_overview_smoke() -> None:
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert [title.value for title in app.title] == ["Project Overview"]


def test_model_selection_separates_pre_and_post_selection_states() -> None:
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.selectbox[0].select("Model Selection").run()

    assert not app.exception
    subheaders = [item.value for item in app.subheader]
    assert "Pre-selection comparability" in subheaders
    assert "Post-selection state" in subheaders
    assert any("Overall pre-selection audit: COMPARABLE" in item.value for item in app.success)

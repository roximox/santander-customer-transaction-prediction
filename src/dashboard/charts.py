"""Pure Plotly chart builders for scientific dashboard data."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.dashboard.formatting import metric_label


def empty_figure(message: str = "No data available") -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(text=message, showarrow=False, x=.5, y=.5)
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return figure


def metric_ranking(frame: pd.DataFrame, metric: str, show_variability: bool = True) -> go.Figure:
    column, std = f"{metric}_mean", f"{metric}_std"
    if frame.empty or column not in frame:
        return empty_figure(f"No {metric_label(metric)} results available")
    data = frame.dropna(subset=[column]).sort_values(column, ascending=True)
    error = data[std] if show_variability and std in data else None
    figure = px.bar(
        data, x=column, y="experiment_id", orientation="h", color="model_family",
        error_x=error, hover_data=[c for c in ("member", "model_name", std) if c in data],
        title=f"Interim ranking by {metric_label(metric)} (0–1 scale)",
        labels={column: f"Mean CV {metric_label(metric)}", "experiment_id": "Experiment"},
    )
    figure.update_xaxes(range=[0, 1])
    return figure


def fold_metrics(frame: pd.DataFrame, experiment_id: str) -> go.Figure:
    metrics = [m for m in ("roc_auc", "average_precision", "f1") if f"validation_{m}" in frame]
    if frame.empty or not metrics:
        return empty_figure("Fold metrics not available")
    long = frame.melt("fold", [f"validation_{m}" for m in metrics], var_name="metric", value_name="score")
    long["metric"] = long["metric"].str.replace("validation_", "", regex=False).map(metric_label)
    figure = px.line(long, x="fold", y="score", color="metric", markers=True, title=f"{experiment_id}: validation metrics by fold")
    figure.update_yaxes(range=[0, 1], title="Validation score")
    figure.update_xaxes(dtick=1, title="CV fold")
    return figure


def roc_auc_vs_time(frame: pd.DataFrame) -> go.Figure:
    required = {"fit_time_mean", "roc_auc_mean", "experiment_id", "model_family"}
    if frame.empty or not required <= set(frame):
        return empty_figure("Performance/time data not available")
    data = frame.dropna(subset=["fit_time_mean", "roc_auc_mean"])
    figure = px.scatter(data, x="fit_time_mean", y="roc_auc_mean", color="model_family", text="experiment_id", hover_data=["model_name", "member"], title="Mean CV ROC-AUC vs mean fit time")
    figure.update_yaxes(range=[0, 1], title="Mean CV ROC-AUC")
    figure.update_xaxes(title="Mean fit time (seconds)")
    return figure


def threshold_metrics(frame: pd.DataFrame) -> go.Figure:
    metrics = [m for m in ("precision", "recall", "f1", "balanced_accuracy") if f"{m}_mean" in frame]
    if frame.empty or not metrics:
        return empty_figure("Threshold metrics not available")
    long = frame.melt("experiment_id", [f"{m}_mean" for m in metrics], var_name="metric", value_name="score")
    long["metric"] = long["metric"].str.replace("_mean", "", regex=False).map(metric_label)
    figure = px.bar(long, x="experiment_id", y="score", color="metric", barmode="group", title="Validation threshold metrics")
    figure.update_yaxes(range=[0, 1], title="Mean CV score")
    return figure


def class_distribution(proportions: dict[str, float]) -> go.Figure:
    frame = pd.DataFrame({"class": list(proportions), "proportion": list(proportions.values())})
    figure = px.bar(frame, x="class", y="proportion", text_auto=".1%", title="Target class distribution")
    figure.update_yaxes(range=[0, 1], tickformat=".0%", title="Proportion")
    return figure


def memory_comparison(original: float, optimized: float) -> go.Figure:
    frame = pd.DataFrame({"representation": ["float64", "float32"], "memory_mb": [original, optimized]})
    return px.bar(frame, x="representation", y="memory_mb", text_auto=".1f", title="Feature memory before and after optimization", labels={"memory_mb": "Memory (MB)"})


def learning_curve(frame: pd.DataFrame, title: str) -> go.Figure:
    if frame.empty:
        return empty_figure("Learning-curve data not available")
    size = "train_size_mean" if "train_size_mean" in frame else "train_size"
    required = {size, "train_roc_auc_mean", "validation_roc_auc_mean"}
    if not required <= set(frame):
        return empty_figure("Learning-curve schema incomplete")
    long = frame.melt(size, ["train_roc_auc_mean", "validation_roc_auc_mean"], var_name="series", value_name="score")
    long["series"] = long["series"].map({"train_roc_auc_mean": "Train ROC-AUC", "validation_roc_auc_mean": "Validation ROC-AUC"})
    figure = px.line(long, x=size, y="score", color="series", markers=True, title=title)
    figure.update_yaxes(range=[0, 1], title="Mean ROC-AUC")
    figure.update_xaxes(title="Training observations")
    return figure


def search_response(frame: pd.DataFrame, metric: str) -> go.Figure:
    column = f"validation_{metric}_mean"
    if frame.empty or not {"C", "penalty", "class_weight", column} <= set(frame):
        return empty_figure("Search results not available")
    data = frame.copy()
    data["class_weight"] = data["class_weight"].fillna("unweighted")
    figure = px.line(data.sort_values("C"), x="C", y=column, color="class_weight", line_dash="penalty", markers=True, title=f"Grid search: {metric_label(metric)} vs C", hover_data=["candidate_id"])
    figure.update_xaxes(type="log", title="Inverse regularization strength C")
    figure.update_yaxes(range=[0, 1], title=f"Mean CV {metric_label(metric)}")
    return figure


def parameter_search_response(frame: pd.DataFrame, parameter: str, metric: str = "roc_auc") -> go.Figure:
    column = f"validation_{metric}_mean"
    if frame.empty or parameter not in frame or column not in frame:
        return empty_figure("Parameter-search results not available")
    data = frame.dropna(subset=[parameter, column]).sort_values(parameter)
    figure = px.scatter(
        data, x=parameter, y=column, color="rank_roc_auc" if "rank_roc_auc" in data else None,
        hover_data=[c for c in ("candidate_id", "learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization") if c in data],
        title=f"Search candidates: {metric_label(metric)} by {parameter}",
    )
    figure.update_yaxes(range=[0, 1], title=f"Mean CV {metric_label(metric)}")
    return figure


def coefficient_stability(frame: pd.DataFrame, top_n: int = 15) -> go.Figure:
    if frame.empty or not {"feature", "mean_coefficient", "sign_consistency"} <= set(frame):
        return empty_figure("Coefficient stability data not available")
    data = frame.assign(magnitude=frame["mean_coefficient"].abs()).nlargest(top_n, "magnitude").sort_values("mean_coefficient")
    return px.bar(data, x="mean_coefficient", y="feature", orientation="h", color="sign_consistency", title="Largest stable logistic coefficients", labels={"mean_coefficient": "Mean fold coefficient", "feature": "Feature"})

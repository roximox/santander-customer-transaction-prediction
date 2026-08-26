"""Streamlit entry point for the read-only scientific-results dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard import charts
from src.dashboard.components import artifact_warning, dataframe, final_test_banner, metric_cards, page_header, status_pill
from src.dashboard.formatting import METRIC_LABELS, format_number, metric_label
from src.dashboard.loaders import (
    filter_experiments, load_coefficient_artifacts, load_dataset_audit,
    load_experiment_summary, load_experiments, load_fold_results,
    load_learning_curves, load_model_comparison, load_registry,
    load_search_candidates, load_selection_outputs,
)


st.set_page_config(page_title="Santander ADA | Scientific Dashboard", page_icon="📊", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.7rem;max-width:1500px}.stMetric{border:1px solid #e2e8f0;border-radius:10px;padding:12px;background:#fff}
[data-testid="stSidebar"]{border-right:1px solid #e2e8f0}.locked{padding:.8rem 1rem;border-radius:8px;background:#ecfdf5;border:1px solid #86efac;color:#14532d;font-weight:700}
</style>""", unsafe_allow_html=True)

PAGES = (
    "Project Overview", "Dataset", "Reproducibility & Validation",
    "Experiment Explorer", "Model Comparison", "Logistic Regression Analysis",
    "Feature Selection & PCA", "HistGradientBoosting", "Learning Curves",
    "Model Selection", "Scientific Conclusions",
)
METRICS = tuple(m for m in METRIC_LABELS if m != "accuracy")


@st.cache_data(show_spinner=False)
def dashboard_data() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    return load_experiments(), load_model_comparison(), load_dataset_audit(), load_selection_outputs()


experiments, eligible, audit, selection = dashboard_data()
with st.sidebar:
    st.header("Santander Customer Transaction Prediction")
    page = st.selectbox("Page", PAGES)
    professor_mode = st.toggle("Professor mode", value=True)
    members = st.multiselect("Member", sorted(experiments.get("member", pd.Series(dtype=str)).dropna().unique()))
    families = st.multiselect("Model family", sorted(experiments.get("model_family", pd.Series(dtype=str)).dropna().unique()))
    experiment_ids = st.multiselect("Experiment", sorted(experiments.get("experiment_id", pd.Series(dtype=str)).dropna().unique()))
    metric = st.selectbox("Metric", METRICS, format_func=metric_label)
    include_dummy = st.checkbox("Include Dummy baselines", value=False)
    show_variability = st.checkbox("Show CV variability", value=True)
    sort_criterion = st.selectbox("Sort criterion", ("Metric descending", "Fit time ascending", "Experiment ID"))
    st.divider()
    st.markdown('<div class="locked">Final test status: RESERVED / NOT USED</div>', unsafe_allow_html=True)
    st.caption("Primary metric: ROC-AUC")
    summary = selection["summary"]
    st.caption("Available: " + ", ".join(summary.get("model_families_represented", [])))
    st.caption("Missing: " + ", ".join(summary.get("missing_expected_model_families", [])))

filtered = filter_experiments(experiments, members=members, families=families, experiment_ids=experiment_ids, include_dummy=include_dummy)


def overview() -> None:
    page_header("Project Overview", "Executive view of saved cross-validation evidence")
    cards = st.columns(4)
    values = (("Total observations", "200,000"), ("Development observations", "160,000"), ("Reserved final test", "40,000"), ("Numerical features", "200"), ("CV folds", "5"), ("Automated tests", "182"), ("Registered experiments", str(len(load_registry()))), ("Eligible candidates", str(len(eligible))))
    for index, (label, value) in enumerate(values): cards[index % 4].metric(label, value)
    st.subheader("Scientific workflow")
    stages = ["OpenML", "Data Audit", "float32 optimization", "Stratified Split", "5-fold CV", "Experiments", "Model Comparison", "Group Model Lock", "🔒 Final Test"]
    st.markdown(" → ".join(f"**{stage}**" for stage in stages))
    final_test_banner()
    st.info("The dashboard reports cross-validation evidence only. Final test evaluation remains locked until group model selection is complete.")


def dataset_page() -> None:
    page_header("Dataset", "Stored data-audit evidence; no OpenML request is made")
    data, dtype = audit["audit"], audit["dtype"]
    if artifact_warning(data, "Dataset audit unavailable"):
        return
    cards = st.columns(4)
    for col, (label, key) in zip(cards, (("OpenML ID", "openml_id"), ("Samples", "n_rows"), ("Features", "n_features"), ("Missing values", "missing_values_X"))): col.metric(label, data.get(key, "Not recorded"))
    left, right = st.columns(2)
    left.plotly_chart(charts.class_distribution(data.get("target_proportions", {})), width="stretch")
    right.plotly_chart(charts.memory_comparison(dtype.get("original_memory_mb", 0), dtype.get("converted_memory_mb", 0)), width="stretch")
    st.write({"Feature type": "Numerical", "Duplicate rows": data.get("duplicate_rows_X"), "Memory reduction": f"{data.get('memory_reduction_percentage', 0):.2f}%", "Maximum absolute conversion error": data.get("maximum_absolute_error")})
    st.info("float32 was retained because it halves feature memory while preserving shape, index, missing values, infinities, and a validated small numerical conversion error.")


def reproducibility() -> None:
    page_header("Reproducibility & Validation", "Shared split and cross-validation protocol")
    split = audit["split"]
    cards = st.columns(4)
    cards[0].metric("Train", f"{split.get('train_dimensions', ['?'])[0]:,}" if isinstance(split.get("train_dimensions"), list) else "Not recorded")
    cards[1].metric("Reserved test", f"{split.get('test_dimensions', ['?'])[0]:,}" if isinstance(split.get("test_dimensions"), list) else "Not recorded")
    cards[2].metric("test_size", split.get("test_size", "Not recorded"))
    cards[3].metric("random_state", split.get("random_state", "Not recorded"))
    st.code(f"Train fingerprint: {split.get('train_indices_sha256', 'Not recorded')}\nReserved-test fingerprint: {split.get('test_indices_sha256', 'Not recorded')}")
    st.subheader("5-fold StratifiedKFold")
    st.markdown(" | ".join(f"**Fold {fold}** · ≈128,000 train / ≈32,000 validation" for fold in range(1, 6)))
    st.write("Every development observation is used once for validation and four times for training. Shuffle is enabled and random_state is 42.")
    final_test_banner()


def explorer() -> None:
    page_header("Experiment Explorer", "Dynamic view of registered experiment summaries and fold results")
    view = filtered.copy()
    statuses = sorted(view.get("status", pd.Series(dtype=str)).dropna().unique())
    selected_statuses = st.multiselect("Status", statuses, default=statuses)
    if selected_statuses and "status" in view:
        view = view[view["status"].isin(selected_statuses)]
    if sort_criterion == "Metric descending" and f"{metric}_mean" in view: view = view.sort_values(f"{metric}_mean", ascending=False)
    elif sort_criterion == "Fit time ascending" and "fit_time_mean" in view: view = view.sort_values("fit_time_mean")
    elif "experiment_id" in view: view = view.sort_values("experiment_id")
    columns = [c for c in ("experiment_id", "member", "model_name", "model_family", "roc_auc_mean", "roc_auc_std", "average_precision_mean", "f1_mean", "precision_mean", "recall_mean", "balanced_accuracy_mean", "roc_auc_generalization_gap", "fit_time_mean", "source_type", "status") if c in view]
    dataframe(view[columns] if columns else view)
    if view.empty: return
    selected = st.selectbox("Inspect experiment", view["experiment_id"].tolist())
    summary = load_experiment_summary(selected)
    if not artifact_warning(summary, "Summary unavailable"):
        with st.expander("Complete summary", expanded=not professor_mode): st.json(summary)
    folds = load_fold_results(selected)
    if not folds.empty:
        st.plotly_chart(charts.fold_metrics(folds, selected), width="stretch")
        fold = st.selectbox("Fold", sorted(folds["fold"].unique()))
        dataframe(folds[folds["fold"].eq(fold)])


def model_comparison() -> None:
    page_header("INTERIM MODEL COMPARISON — Member 02 models pending", "Eligible saved CV candidates only; no final winner is declared")
    data = filter_experiments(eligible, members=members, families=families, experiment_ids=experiment_ids, include_dummy=include_dummy)
    st.plotly_chart(charts.metric_ranking(data, metric, show_variability), width="stretch")
    left, right = st.columns(2)
    left.plotly_chart(charts.roc_auc_vs_time(data), width="stretch")
    right.plotly_chart(charts.threshold_metrics(data), width="stretch")
    ranking = data.sort_values(f"{metric}_mean", ascending=False) if f"{metric}_mean" in data else data
    dataframe(ranking)
    if not data.empty:
        best = data.dropna(subset=["roc_auc_mean"]).sort_values("roc_auc_mean", ascending=False).iloc[0]
        st.info(f"{best['model_name']} currently has the highest recorded mean CV ROC-AUC ({best['roc_auc_mean']:.6f}). This is interim evidence, not a final selection.")
    st.caption("Competitiveness label: CV variability heuristic — not a formal statistical test")


def logistic_page() -> None:
    page_header("Logistic Regression Analysis", "Member 01: baselines, weighting, search, coefficients and learning curves")
    logistic = experiments[experiments["model_family"].isin(["DUMMY", "LOGISTIC_REGRESSION"])]
    st.subheader("Dummy baselines")
    st.plotly_chart(charts.metric_ranking(logistic[logistic.model_family.eq("DUMMY")], "roc_auc", True), width="stretch")
    st.info("Approximately 90% accuracy is possible by predicting the majority class, while ROC-AUC remains 0.5 because ranking ability is absent.")
    st.subheader("Logistic baseline and class weighting")
    official = logistic[logistic.experiment_id.isin(["M01-LR-001", "M01-LR-002"])]
    st.plotly_chart(charts.threshold_metrics(official), width="stretch")
    dataframe(official)
    st.subheader("Hyperparameter search")
    search = load_search_candidates("M01-LR-SEARCH-001")
    search_columns = st.columns(3)
    penalties = search_columns[0].multiselect("Penalty", sorted(search.get("penalty", pd.Series(dtype=str)).dropna().unique()))
    weights = search.get("class_weight", pd.Series(dtype=object)).fillna("unweighted")
    selected_weights = search_columns[1].multiselect("Class weight", sorted(weights.unique()))
    c_values = sorted(search.get("C", pd.Series(dtype=float)).dropna().unique())
    selected_c = search_columns[2].multiselect("C", c_values)
    search_view = search.copy()
    if penalties: search_view = search_view[search_view["penalty"].isin(penalties)]
    if selected_weights: search_view = search_view[weights.loc[search_view.index].isin(selected_weights)]
    if selected_c: search_view = search_view[search_view["C"].isin(selected_c)]
    st.plotly_chart(charts.search_response(search_view, metric if metric in ("roc_auc", "average_precision") else "roc_auc"), width="stretch")
    st.subheader("Coefficient stability")
    coefficients = load_coefficient_artifacts()
    st.plotly_chart(charts.coefficient_stability(coefficients["stability"]), width="stretch")
    st.warning("Model coefficients indicate association, not causality.")
    st.subheader("Learning curves")
    curves = load_learning_curves()["Logistic Regression"]
    configurations = curves.get("configuration_id", pd.Series(dtype=str)).dropna().unique().tolist()
    chosen = st.selectbox("Configuration", configurations) if configurations else None
    selected_curve = curves[curves.configuration_id.eq(chosen)] if chosen else curves
    st.plotly_chart(charts.learning_curve(selected_curve, "Logistic Regression learning curve"), width="stretch")
    st.caption("Validation performance empirically plateaus after approximately 75% of development data; this does not establish complete saturation.")


def m03_page() -> None:
    page_header("Feature Selection & PCA", "Member 03 registered CV evidence")
    ids = ["M01-LR-001", "M03-FS-001", "M03-PCA-001"]
    data = experiments[experiments.experiment_id.isin(ids)]
    st.plotly_chart(charts.metric_ranking(data, metric, show_variability), width="stretch")
    dataframe(data)
    st.info("Feature Selection and PCA currently produce validation ranking performance close to the Logistic Regression baseline. This does not imply that dimensionality reduction is universally ineffective.")
    pca = load_experiment_summary("M03-PCA-001")
    components = pca.get("n_components") or pca.get("explained_variance")
    st.write("PCA explained-variance/component metadata:", components if components is not None else "Metadata not recorded")


def hgb_page() -> None:
    page_header("HistGradientBoosting", "Member 04 official experiments and auxiliary analyses")
    data = experiments[experiments.experiment_id.isin(["M04-HGB-001", "M04-HGB-002"])]
    st.subheader("Official registered experiments")
    st.plotly_chart(charts.threshold_metrics(data), width="stretch")
    dataframe(data)
    st.info("HGB currently shows stronger ranking performance, alongside a larger train-validation gap and higher computational cost than Logistic Regression.")
    st.subheader("Auxiliary analyses — SEARCH / LC / OOF / COMP")
    search = load_search_candidates("M04-HGB-SEARCH-001")
    parameters = [name for name in ("learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization") if name in search]
    parameter = st.selectbox("Search parameter", parameters) if parameters else "learning_rate"
    st.plotly_chart(charts.parameter_search_response(search, parameter), width="stretch")
    curve = load_learning_curves()["HistGradientBoosting Tuned"]
    st.plotly_chart(charts.learning_curve(curve, "Tuned HGB learning curve"), width="stretch")
    st.caption("Auxiliary artifacts are diagnostics and are distinct from official registered experiments.")


def learning_curves_page() -> None:
    page_header("Learning Curves", "Training-only learning evidence from saved artifacts")
    curves = load_learning_curves()
    available = [name for name, frame in curves.items() if not frame.empty]
    choice = st.selectbox("Model/configuration", available) if available else None
    if not choice: st.warning("No learning-curve artifacts are available."); return
    frame = curves[choice]
    if choice == "Logistic Regression" and "configuration_id" in frame:
        config = st.selectbox("Logistic configuration", sorted(frame.configuration_id.unique()))
        frame = frame[frame.configuration_id.eq(config)]
    st.plotly_chart(charts.learning_curve(frame, f"{choice}: ROC-AUC learning curve"), width="stretch")
    if "validation_average_precision_mean" in frame:
        st.line_chart(frame.set_index("train_size_mean" if "train_size_mean" in frame else "train_size")[["validation_average_precision_mean"]])
    size = "train_size_mean" if "train_size_mean" in frame else "train_size"
    if "validation_roc_auc_mean" in frame and size in frame:
        frame = frame.sort_values(size).copy()
        frame["incremental_validation_roc_auc_gain"] = frame["validation_roc_auc_mean"].diff()
    dataframe(frame)


def selection_page() -> None:
    page_header("Model Selection", "Transparent multi-criteria review of saved CV evidence")
    summary, coverage, comparability, decision = selection["summary"], selection["coverage"], selection["comparability"], selection["decision"]
    status_pill("Selection status", summary.get("selection_status"))
    st.subheader("Current best by metric")
    best = summary.get("best_by_metric", {})
    st.write(pd.DataFrame([{"Metric": metric_label(key), "Experiment": value} for key, value in best.items()]))
    st.subheader("Candidate coverage")
    dataframe(coverage)
    for family in summary.get("missing_expected_model_families", []): st.warning(f"No {family.replace('_', ' ').title()} results are currently registered.")
    st.subheader("Comparability")
    checks = comparability.get("checks", {})
    dataframe(pd.DataFrame([{"Check": key, **value} for key, value in checks.items()]))
    if not professor_mode:
        st.subheader("Selection decision evidence")
        dataframe(decision)
    final_test_banner()


def conclusions() -> None:
    page_header("Scientific Conclusions", "Interim evidence for professor and group review")
    st.markdown("""
### Data
The clean numerical dataset is imbalanced; validated float32 conversion halves feature memory.
### Logistic Regression
The baseline is stable with a small generalization gap. Class weighting changes the precision/recall trade-off.
### Dimensionality reduction
Feature Selection and PCA currently have limited effect on ranking performance relative to the Logistic baseline.
### Gradient Boosting
HistGradientBoosting is the current strongest ranking candidate, with higher cost and a larger train-validation gap.
### Current limitation
Member 02 Random Forest and Extra Trees results remain missing.
### Next decision
Integrate remaining models, rerun pre-final selection, lock one group pipeline, and only then unlock final-test evaluation.
""")
    final_test_banner()


ROUTES = {
    "Project Overview": overview, "Dataset": dataset_page,
    "Reproducibility & Validation": reproducibility, "Experiment Explorer": explorer,
    "Model Comparison": model_comparison, "Logistic Regression Analysis": logistic_page,
    "Feature Selection & PCA": m03_page, "HistGradientBoosting": hgb_page,
    "Learning Curves": learning_curves_page, "Model Selection": selection_page,
    "Scientific Conclusions": conclusions,
}
ROUTES[page]()

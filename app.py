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
    final_status = "COMPLETED ONCE" if selection["final_test"].get("status") == "completed" else "RESERVED / NOT USED"
    st.markdown(f'<div class="locked">Final test status: {final_status}</div>', unsafe_allow_html=True)
    st.caption("Primary metric: ROC-AUC")
    summary = selection["summary"]
    st.caption("Available: " + ", ".join(summary.get("model_families_represented", [])))
    st.caption("Missing: " + ", ".join(summary.get("missing_expected_model_families", [])))

filtered = filter_experiments(experiments, members=members, families=families, experiment_ids=experiment_ids, include_dummy=include_dummy)


def overview() -> None:
    page_header("Project Overview", "Executive view of saved cross-validation evidence")
    cards = st.columns(4)
    values = (("Total observations", "200,000"), ("Development observations", "160,000"), ("Reserved final test", "40,000"), ("Numerical features", "200"), ("CV folds", "5"), ("Selected model", selection["lock"].get("selected_experiment_id", "Not locked")), ("Registered experiments", str(len(load_registry()))), ("Eligible candidates", str(len(eligible))))
    for index, (label, value) in enumerate(values): cards[index % 4].metric(label, value)
    st.subheader("Scientific workflow")
    stages = ["OpenML", "Data Audit", "float32 optimization", "Stratified Split", "5-fold CV", "Experiments", "Model Comparison", "Group Model Lock", "🔒 Final Test"]
    st.markdown(" → ".join(f"**{stage}**" for stage in stages))
    final_test_banner(selection["final_test"])
    if selection["final_test"].get("status") == "completed":
        st.info("The group model choice remains frozen. Its single reserved final-test evaluation is complete and cannot be rerun for model selection.")
    else:
        st.info("The dashboard reports cross-validation evidence only. The group model choice is locked; the reserved final-test evaluation has not been executed.")


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
    final_test_banner(selection["final_test"])


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
    missing = selection["summary"].get("missing_expected_model_families", [])
    pending = ", ".join(family.replace("_", " ").title() for family in missing)
    title = "INTERIM MODEL COMPARISON" + (f" — {pending} pending" if pending else " — full expected coverage")
    page_header(title, "Eligible saved CV candidates only; no final winner is declared")
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
    summary, lock, coverage, comparability, decision = selection["summary"], selection["lock"], selection["coverage"], selection["comparability"], selection["decision"]
    status_pill("Generated comparison status", summary.get("selection_status"))
    status_pill("Current project status", lock.get("status", "not_locked"))
    if lock.get("selected_experiment_id"):
        decision_message = (
            "completed its single final evaluation"
            if lock.get("final_test_status") == "COMPLETED_ONCE"
            else "is locked for the single final evaluation"
        )
        st.success(
            f"Collective decision: {lock['selected_experiment_id']} — {lock.get('model_name', 'model')} "
            f"{decision_message}."
        )
        st.write({
            "Decision status": lock.get("status"),
            "Selected member": lock.get("member"),
            "Classification threshold": lock.get("classification_threshold"),
            "Final test used": lock.get("final_test_used"),
        })
    st.subheader("Current best by metric")
    best = summary.get("best_by_metric", {})
    st.write(pd.DataFrame([{"Metric": metric_label(key), "Experiment": value} for key, value in best.items()]))
    st.subheader("Candidate coverage")
    dataframe(coverage)
    for family in summary.get("missing_expected_model_families", []): st.warning(f"No {family.replace('_', ' ').title()} results are currently registered.")
    st.subheader("Pre-selection comparability")
    st.caption(
        "This audit describes the cross-validation evidence used before the collective model lock; "
        "the final-test result is intentionally excluded from candidate comparison."
    )
    checks = comparability.get("checks", {})
    core_protocol = ("n_samples", "n_splits", "cv_strategy", "random_state", "primary_metric")
    core_consistent = all(checks.get(name, {}).get("status") == "consistent" for name in core_protocol)
    target_known = int(eligible.get("target_distribution", pd.Series(dtype=object)).notna().sum())
    source_known = int(eligible.get("data_source", pd.Series(dtype=object)).notna().sum())
    total_candidates = len(eligible)
    comparison_rows = [
        {
            "Scope": "Core methodological protocol",
            "Status": "CONSISTENT" if core_consistent else "INCOMPATIBLE",
            "Detail": "160,000 development rows; 5-fold StratifiedKFold; random_state=42; primary metric ROC-AUC.",
        },
        {
            "Scope": "Target-distribution metadata",
            "Status": checks.get("target_distribution", {}).get("status", "not verifiable").upper(),
            "Detail": f"Recorded for {target_known}/{total_candidates} candidates; missing metadata does not demonstrate a protocol difference.",
        },
        {
            "Scope": "Data-source metadata",
            "Status": checks.get("data_source", {}).get("status", "not verifiable").upper(),
            "Detail": f"Recorded for {source_known}/{total_candidates} candidates; the shared split evidence remains consistent.",
        },
        {
            "Scope": "Final-test use during selection",
            "Status": checks.get("final_test_usage", {}).get("status", "unknown").upper(),
            "Detail": "No candidate used final-test metrics for selection. The selected model was evaluated only after the collective lock.",
        },
    ]
    dataframe(pd.DataFrame(comparison_rows))
    comparability_status = str(comparability.get("comparability_status", "unknown"))
    if comparability_status == "comparable":
        st.success(
            "Overall pre-selection audit: COMPARABLE. The shared CV protocol and required provenance metadata "
            "are consistent across all eligible candidates."
        )
    else:
        st.info(
            f"Overall pre-selection audit: {comparability_status.replace('_', ' ').upper()}. "
            "Any partial status reflects missing provenance metadata unless a check is explicitly marked incompatible."
        )
    st.subheader("Post-selection state")
    dataframe(pd.DataFrame([
        {"Control": "Selected pipeline", "Value": lock.get("selected_experiment_id", "Not locked")},
        {"Control": "Collective model lock", "Value": lock.get("status", "Not available")},
        {"Control": "Final evaluation", "Value": lock.get("final_test_status", "Not available")},
        {"Control": "Execution count", "Value": selection["final_test"].get("execution_count", "Not available")},
        {"Control": "Selection reopened", "Value": selection["final_test"].get("selection_reopened", "Not available")},
    ]))
    if not professor_mode:
        st.subheader("Selection decision evidence")
        dataframe(decision)
    final_test_banner(selection["final_test"])


def conclusions() -> None:
    page_header("Scientific Conclusions", "Collective model decision based on complete cross-validation evidence")
    selection_summary = selection["summary"]
    model_lock = selection["lock"]
    available = eligible.set_index("experiment_id", drop=False) if not eligible.empty else pd.DataFrame()

    def candidate(experiment_id: str) -> pd.Series:
        if available.empty or experiment_id not in available.index:
            return pd.Series(dtype=object)
        row = available.loc[experiment_id]
        return row.iloc[0] if isinstance(row, pd.DataFrame) else row

    hgb = candidate("M04-HGB-002")
    logistic = candidate("M01-LR-001")
    balanced = candidate("M01-LR-SEARCH-001::candidate_004")
    extra_trees = candidate("M01-ET-001")
    random_forest = candidate("M02-RF-001")
    decision_tree = candidate("M02-DT-001")
    feature_selection = candidate("M03-FS-001")
    pca = candidate("M03-PCA-001")

    st.subheader("Executive evidence summary")
    cards = st.columns(4)
    cards[0].metric("Registered experiments", selection_summary.get("number_of_discovered_experiments", "Not recorded"))
    cards[1].metric("Eligible candidates", selection_summary.get("number_of_eligible_candidates", "Not recorded"))
    cards[2].metric("Expected families missing", len(selection_summary.get("missing_expected_model_families", [])))
    cards[3].metric("Selection status", str(selection_summary.get("selection_status", "unknown")).replace("_", " ").upper())

    st.subheader("Data and reproducibility")
    data_audit, split = audit["audit"], audit["split"]
    st.markdown(
        f"""
- The dataset contains **{data_audit.get('n_rows', 200000):,} observations** and **{data_audit.get('n_features', 200)} numerical features**, with no recorded missing feature values or duplicate rows.
- The positive class represents **{data_audit.get('target_proportions', {}).get('True', .10049):.2%}** of observations, so Accuracy alone is not an adequate selection metric.
- Validated float32 conversion reduces feature memory by **{data_audit.get('memory_reduction_percentage', 50):.2f}%**, with a maximum recorded absolute conversion error of **{data_audit.get('maximum_absolute_error', 0):.3e}**.
- All candidates use the **{split.get('train_dimensions', [160000])[0]:,}-row development partition**, five-fold StratifiedKFold, and `random_state=42`.
"""
    )

    st.subheader("Logistic Regression")
    st.markdown(
        f"""
- The unweighted baseline is stable: mean CV ROC-AUC **{format_number(logistic.get('roc_auc_mean'), 6)}** with a train-validation gap of **{format_number(logistic.get('roc_auc_generalization_gap'), 6)}**.
- Class weighting preserves similar ranking performance but changes the operating trade-off: the balanced candidate reaches recall **{format_number(balanced.get('recall_mean'), 6)}** and balanced accuracy **{format_number(balanced.get('balanced_accuracy_mean'), 6)}**, with precision **{format_number(balanced.get('precision_mean'), 6)}**.
- Logistic Regression remains substantially cheaper to fit than the tree ensembles and HGB in the recorded experiments.
"""
    )

    st.subheader("Feature Selection and PCA")
    st.markdown(
        f"""
- Feature Selection records ROC-AUC **{format_number(feature_selection.get('roc_auc_mean'), 6)}**, close to the Logistic baseline.
- PCA records ROC-AUC **{format_number(pca.get('roc_auc_mean'), 6)}**, also close to the Logistic baseline.
- Current evidence therefore shows no material ranking improvement from these two transformations in this configuration; it does not imply that dimensionality reduction is universally ineffective.
"""
    )

    st.subheader("Tree ensembles and Decision Tree")
    st.markdown(
        f"""
- Decision Tree provides the weakest recorded ranking result among eligible candidates, with ROC-AUC **{format_number(decision_tree.get('roc_auc_mean'), 6)}**.
- Random Forest improves over the single Decision Tree to ROC-AUC **{format_number(random_forest.get('roc_auc_mean'), 6)}** and balanced accuracy **{format_number(random_forest.get('balanced_accuracy_mean'), 6)}**, but remains below the linear and HGB candidates on ranking metrics.
- Member 01 Extra Trees reaches ROC-AUC **{format_number(extra_trees.get('roc_auc_mean'), 6)}**, recall **{format_number(extra_trees.get('recall_mean'), 6)}**, and the highest recorded mean CV F1, **{format_number(extra_trees.get('f1_mean'), 6)}**.
- Extra Trees offers a different threshold-metric trade-off; it is not the strongest ROC-AUC candidate and is not automatically the final choice.
"""
    )

    st.subheader("HistGradientBoosting")
    st.markdown(
        f"""
- Tuned HGB currently has the highest mean CV ROC-AUC (**{format_number(hgb.get('roc_auc_mean'), 6)}**), Average Precision (**{format_number(hgb.get('average_precision_mean'), 6)}**), and precision (**{format_number(hgb.get('precision_mean'), 6)}**).
- Its recorded fit time (**{format_number(hgb.get('fit_time_mean'), 2)} seconds per fold**) and train-validation gap (**{format_number(hgb.get('roc_auc_generalization_gap'), 6)}**) are substantially larger than those of Logistic Regression.
- The tuned configuration improves the recorded HGB baseline, but the competitiveness label remains a **CV variability heuristic — not a formal statistical test**.
"""
    )

    st.subheader("Final model-selection conclusion")
    best = selection_summary.get("best_by_metric", {})
    dataframe(pd.DataFrame([{"Criterion": metric_label(name), "Current experiment": experiment_id} for name, experiment_id in best.items()]))
    st.success(
        f"The team collectively selected and locked {model_lock.get('selected_experiment_id', 'M04-HGB-002')} — "
        f"{model_lock.get('model_name', 'HistGradientBoosting Tuned')}. It leads the recorded candidates on both "
        "ROC-AUC and Average Precision, the two ranking metrics most relevant to this imbalanced problem. "
        "This is a pragmatic multi-criteria decision, not a claim of statistical superiority."
    )
    final_result = selection["final_test"]
    if final_result.get("status") == "completed":
        st.subheader("Final-test result")
        metrics = final_result.get("metrics", {})
        metric_cards(metrics, ("roc_auc", "average_precision", "f1", "precision", "recall", "balanced_accuracy"))
        st.caption("Single execution on the 40,000-row reserved partition; model selection was not reopened.")
    else:
        st.subheader("Remaining final step")
        st.markdown(
            "The selected pipeline, estimator parameters, and classification threshold are frozen in the model-lock artifact. "
            "The only remaining scientific evaluation is one execution on the reserved final-test partition; it has not yet been run."
        )
    final_test_banner(final_result)


ROUTES = {
    "Project Overview": overview, "Dataset": dataset_page,
    "Reproducibility & Validation": reproducibility, "Experiment Explorer": explorer,
    "Model Comparison": model_comparison, "Logistic Regression Analysis": logistic_page,
    "Feature Selection & PCA": m03_page, "HistGradientBoosting": hgb_page,
    "Learning Curves": learning_curves_page, "Model Selection": selection_page,
    "Scientific Conclusions": conclusions,
}
ROUTES[page]()

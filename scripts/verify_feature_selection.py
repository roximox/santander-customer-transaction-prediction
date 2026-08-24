"""Offline smoke test for the M03 feature selection and PCA pipeline factories.

Uses only a small synthetic dataset and never accesses the Santander dataset,
the final test partition, or the scientific experiment registry.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import evaluate_model_cv  # noqa: E402
from src.feature_selection import (  # noqa: E402
    FS_EXPERIMENT_ID,
    FS_MODEL_NAME,
    PCA_EXPERIMENT_ID,
    PCA_MODEL_NAME,
    build_fs_pca_comparison,
    create_feature_selection_pipeline,
    create_pca_pipeline,
)
from src.validation import create_stratified_cv  # noqa: E402


def _make_synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return a small synthetic binary classification dataset."""
    values, labels = make_classification(
        n_samples=400,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42,
    )
    X = pd.DataFrame(values, columns=[f"var_{i}" for i in range(values.shape[1])])
    y = pd.Series(labels.astype(str), name="target", index=X.index)
    return X, y


def main() -> None:
    """Run offline structural and pipeline smoke tests without touching real data."""
    print("TECHNICAL SMOKE TEST ONLY — synthetic data, not a Santander experiment.")

    fs_pipeline = create_feature_selection_pipeline()
    pca_pipeline = create_pca_pipeline()

    assert list(fs_pipeline.named_steps) == ["scaler", "selector", "classifier"], \
        "Feature selection pipeline step names are incorrect."
    assert list(pca_pipeline.named_steps) == ["scaler", "pca", "classifier"], \
        "PCA pipeline step names are incorrect."

    for pipeline in (fs_pipeline, pca_pipeline):
        for step_name, step in pipeline.named_steps.items():
            assert not hasattr(step, "coef_"), \
                f"Step '{step_name}' must be unfitted."
    print("  Pipeline structures verified — both pipelines are unfitted.")

    X, y = _make_synthetic_data()
    cv = create_stratified_cv(n_splits=2)

    fs_fold_results, fs_summary = evaluate_model_cv(
        create_feature_selection_pipeline(), X, y,
        model_name="Technical FS smoke check",
        experiment_id="TECH-FS-SMOKE",
        member="Member 03",
        branch="feature/feature-selection",
        cv=cv,
        n_jobs=1,
    )
    assert fs_summary["status"] == "completed", "FS smoke evaluation did not complete."
    print(f"  FS smoke: {fs_summary['primary_metric']}={fs_summary['primary_score_mean']:.4f}")

    pca_fold_results, pca_summary = evaluate_model_cv(
        create_pca_pipeline(), X, y,
        model_name="Technical PCA smoke check",
        experiment_id="TECH-PCA-SMOKE",
        member="Member 03",
        branch="feature/pca",
        cv=cv,
        n_jobs=1,
    )
    assert pca_summary["status"] == "completed", "PCA smoke evaluation did not complete."
    print(f"  PCA smoke: {pca_summary['primary_metric']}={pca_summary['primary_score_mean']:.4f}")

    comparison = build_fs_pca_comparison([fs_summary, pca_summary])
    print(f"  Comparison table: {comparison.shape[0]} rows, {comparison.shape[1]} columns.")
    json.dumps(comparison.to_dict(orient="records"), allow_nan=False)
    print("  Comparison table is JSON-serialisable.")

    print("All offline verification checks passed.")
    print("No files were saved and no scientific experiment was registered.")
    print("The final test partition was not used.")


if __name__ == "__main__":
    main()

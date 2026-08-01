"""Run a synthetic technical smoke test of the evaluation framework."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (  # noqa: E402
    evaluate_model_cv,
    get_primary_metric_name,
    get_scoring_metrics,
)
from src.validation import create_stratified_cv, get_cv_split_fingerprints  # noqa: E402


def main() -> None:
    """Exercise CV and reporting on synthetic data without saving an experiment."""
    values, target = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        weights=[0.9, 0.1],
        random_state=42,
    )
    X = pd.DataFrame(values, columns=[f"synthetic_{index}" for index in range(5)])
    y = pd.Series(target, name="synthetic_target")
    cv = create_stratified_cv()
    fingerprints = get_cv_split_fingerprints(cv, X, y)

    print("TECHNICAL SMOKE TEST ONLY — synthetic data, not a Santander experiment.")
    print(f"Available metrics: {list(get_scoring_metrics())}")
    print(f"Primary metric: {get_primary_metric_name()}")
    print("CV fingerprints:")
    for fold in fingerprints:
        print(
            f"  Fold {fold['fold']}: train={fold['train_indices_sha256']}, "
            f"validation={fold['validation_indices_sha256']}"
        )

    fold_results, summary = evaluate_model_cv(
        DummyClassifier(strategy="prior"),
        X,
        y,
        model_name="Synthetic technical DummyClassifier",
        experiment_id="TECH-SMOKE-ONLY",
        member="technical-smoke-test",
        branch="not-recorded",
        cv=cv,
        n_jobs=1,
    )
    print(f"Fold result columns: {list(fold_results.columns)}")
    print(fold_results.to_string(index=False))
    print("Serializable summary preview:")
    print(
        json.dumps(
            {
                "experiment_id": summary["experiment_id"],
                "n_samples": summary["n_samples"],
                "n_features": summary["n_features"],
                "cv_strategy": summary["cv_strategy"],
                "primary_metric": summary["primary_metric"],
                "primary_score_mean": summary["primary_score_mean"],
                "status": summary["status"],
            },
            indent=2,
            allow_nan=False,
        )
    )
    print("No files were saved and no scientific experiment was registered.")


if __name__ == "__main__":
    main()

"""Run the Member 4 training-only HistGradientBoosting randomized search."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_dataset  # noqa: E402
from src.gradient_boosting_search import (  # noqa: E402
    HIST_GRADIENT_BOOSTING_SEARCH_ID,
    create_hist_gradient_boosting_randomized_search,
    hist_gradient_boosting_search_results_to_dataframe,
    summarize_hist_gradient_boosting_search,
)
from src.search import save_grid_search_results  # noqa: E402
from src.validation import create_train_test_split, split_fingerprint  # noqa: E402

EXPECTED_TRAIN_FINGERPRINT = "61c403ec521d15ab9d6316606eba5acdfc22381cb764b6da76a65041ec11f477"
EXPECTED_TEST_FINGERPRINT = "bf7d43e492967dad1358676e9bf8355a910823077b59118014db12af3e26f586"
SEARCH_DIR = PROJECT_ROOT / "reports/searches"
OUTPUTS = (
    SEARCH_DIR / f"{HIST_GRADIENT_BOOSTING_SEARCH_ID}_candidates.csv",
    SEARCH_DIR / f"{HIST_GRADIENT_BOOSTING_SEARCH_ID}_summary.json",
)


def refuse_existing_outputs() -> None:
    """Refuse duplicate HGB search artifacts before any training begins."""
    existing = [path.relative_to(PROJECT_ROOT).as_posix() for path in OUTPUTS if path.exists()]
    if existing:
        raise FileExistsError(
            f"{HIST_GRADIENT_BOOSTING_SEARCH_ID} outputs already exist and were not "
            "overwritten. Remove only the targeted artifacts manually before rerunning:\n- "
            + "\n- ".join(existing)
        )


def print_result_summary(summary: dict[str, object]) -> None:
    """Print the requested best-candidate validation metrics."""
    metrics = summary["target_metric_summaries"]
    roc_auc = metrics["roc_auc"]
    average_precision = metrics["average_precision"]
    print(f"Best candidate ID: {summary['best_candidate_id']}")
    print(f"Best ROC-AUC: {summary['best_score']:.6f}")
    print(f"Best parameters: {summary['best_parameters']}")
    print(f"Train ROC-AUC: {roc_auc['train_mean']:.6f}")
    print(f"Validation ROC-AUC: {roc_auc['validation_mean']:.6f}")
    print(f"Generalization gap: {roc_auc['generalization_gap']:.6f}")
    print(f"Average Precision: {average_precision['validation_mean']:.6f}")
    print(f"Number of candidates: {summary['n_candidates']}")
    print(f"Number of CV folds: {summary['n_splits']}")
    print(f"Total number of fits: {summary['total_fits']}")


def main() -> None:
    """Fit and save the HGB search using only the verified training partition."""
    refuse_existing_outputs()
    X, y, _ = load_dataset(optimize_memory=True)
    X_train, X_reserved, y_train, y_reserved = create_train_test_split(X, y)
    train_fingerprint = split_fingerprint(X_train.index)
    reserved_fingerprint = split_fingerprint(X_reserved.index)
    if train_fingerprint != EXPECTED_TRAIN_FINGERPRINT:
        raise RuntimeError(f"Train fingerprint mismatch: {train_fingerprint}; search stopped.")
    if reserved_fingerprint != EXPECTED_TEST_FINGERPRINT:
        raise RuntimeError(
            f"Reserved test fingerprint mismatch: {reserved_fingerprint}; search stopped."
        )
    del X_reserved, y_reserved, X, y

    search = create_hist_gradient_boosting_randomized_search(n_iter=20, n_jobs=1)
    search.fit(X_train, y_train)
    results = hist_gradient_boosting_search_results_to_dataframe(search)
    summary = summarize_hist_gradient_boosting_search(
        search,
        results,
        member="Member 04",
        branch="feature/model-optimization",
    )
    summary.update(
        {
            "n_samples": int(X_train.shape[0]),
            "n_features": int(X_train.shape[1]),
            "n_jobs": search.n_jobs,
            "train_fingerprint": train_fingerprint,
            "reserved_test_fingerprint_verified_only": reserved_fingerprint,
        }
    )
    save_grid_search_results(results, summary)

    print_result_summary(summary)
    print("Final test partition remained reserved and was not evaluated.")


if __name__ == "__main__":
    main()

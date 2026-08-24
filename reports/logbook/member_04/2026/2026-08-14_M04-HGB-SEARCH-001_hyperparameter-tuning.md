# Logbook Entry

## Metadata

Date: 2026-08-14
Member: Chaymae Akouaouch
Sprint: Not confirmed
Ticket ID: M04-HGB-SEARCH-001
Branch: feature/model-optimization
Pull Request: Not created yet
Time spent: Not recorded yet
Related meeting: No related meeting yet

## Title

HistGradientBoosting hyperparameter tuning

## Goal

Improve the untuned HistGradientBoosting baseline `M04-HGB-001` through
training-only hyperparameter optimization while keeping the final test
partition closed.

Baseline reference:

- Validation ROC-AUC: 0.884596
- Average Precision: 0.572879
- Train ROC-AUC: 0.975659
- Generalization gap: 0.091063

## Search design

`RandomizedSearchCV` was selected instead of a full `GridSearchCV` because the
predeclared HGB space has 768 Cartesian combinations; randomized search makes it
possible to evaluate a deterministic, bounded subset while retaining the common
cross-validation protocol.

The search space was:

- `learning_rate`: [0.03, 0.05, 0.1, 0.15]
- `max_iter`: [150, 300, 500, 700]
- `max_leaf_nodes`: [15, 31, 63]
- `min_samples_leaf`: [10, 20, 50, 100]
- `l2_regularization`: [0.0, 0.1, 1.0, 10.0]

The actual search used 20 candidates, 5-fold stratified cross-validation, and
100 total fits with `random_state=42` and `n_jobs=1`. ROC-AUC was the primary
metric, common project metrics were retained, and only the training partition
was used.

## Implementation

`src/gradient_boosting_search.py` defines the search ID and HGB search space,
creates an unfitted `RandomizedSearchCV`, converts fitted search results to a
candidate-level DataFrame, and creates a JSON-compatible summary.

`scripts/run_gradient_boosting_search.py` loads the dataset, reproduces the
common shared split, verifies train and reserved-test fingerprints, removes
reserved test data from the tuning workflow, runs `search.fit(X_train, y_train)`,
converts and summarizes results, and reuses `src.search.save_grid_search_results()`
for persistence.

Member 1's `src/search.py` was not modified for HGB search logic; only its
existing generic save helper was reused.

## Work steps

1. Reviewed Member 1's existing `src/search.py`.
2. Identified that its result conversion was Logistic-specific.
3. Decided not to modify shared Member 1 search logic.
4. Created a separate Member 4 HGB search module.
5. Defined a bounded HGB search space.
6. Prepared deterministic `RandomizedSearchCV`.
7. Added HGB-specific result conversion and search summary helpers.
8. Created the HGB search runner.
9. Verified the search object before execution.
10. Executed `python scripts/run_gradient_boosting_search.py`.
11. Reviewed the best candidate and compared it with the HGB baseline.
12. Confirmed the final test partition was not evaluated.

## Results

Best candidate: `candidate_011`

Best parameters:

- `min_samples_leaf = 100`
- `max_leaf_nodes = 31`
- `max_iter = 700`
- `learning_rate = 0.05`
- `l2_regularization = 10.0`

- Best Validation ROC-AUC: 0.891449
- Train ROC-AUC: 0.973580
- Generalization gap: 0.082131
- Validation Average Precision: 0.591089
- Candidates: 20
- CV folds: 5
- Total fits: 100

## Baseline comparison

| Metric | M04-HGB-001 Baseline | M04-HGB-SEARCH-001 Best Candidate | Change |
|---|---:|---:|---:|
| Validation ROC-AUC | 0.884596 | 0.891449 | +0.006853 |
| Average Precision | 0.572879 | 0.591089 | +0.018210 |
| Train ROC-AUC | 0.975659 | 0.973580 | -0.002079 |
| Generalization gap | 0.091063 | 0.082131 | -0.008932 |

## Interpretation

The tuned candidate improves validation ROC-AUC relative to the baseline, and
Average Precision also improves. Train ROC-AUC decreases slightly while
validation ROC-AUC increases, making the train/validation gap smaller. This is
consistent with improved generalization, but it does not establish that
overfitting has been eliminated.

The best candidate uses a lower learning rate, more boosting iterations, a
larger minimum leaf size, and stronger L2 regularization. These settings are
consistent with a more regularized, slower-learning model. It is the best HGB
configuration found in this bounded 20-candidate search, not necessarily the
global optimum, and it is not yet the final group model.

## Reproducibility and leakage safeguards

- Same shared train/test split.
- Fingerprint verification.
- `random_state=42`.
- Training data only.
- 5-fold stratified CV.
- Same shared metrics.
- No final-test predictions.
- No final-test scoring.
- No test-based model selection.

## Generated artifacts

- `reports/searches/M04-HGB-SEARCH-001_candidates.csv`
- `reports/searches/M04-HGB-SEARCH-001_summary.json`

## Next planned work

Learning-curve analysis for the selected/tuned HistGradientBoosting
configuration. This has not yet been implemented or executed.

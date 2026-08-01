# Logbook Entry

## Metadata

- Date: 2026-08-01
- Member: Yassine Elhari
- Sprint: Sprint 1
- Ticket ID: ADA-ML-00
- Branch: feature/data_processing
- Pull Request: To be updated after Pull Request creation
- Time spent: TO BE COMPLETED BY YASSINE ELHARI
- Related meeting: TO BE COMPLETED BY YASSINE ELHARI

## Title

Common model-evaluation framework

## Objective

Standardize training-only model evaluation so results produced by different
members and model families remain scientifically comparable and reproducible.

## Comparability risk

Independent fold generation, metrics, timing conventions, or preprocessing
placement could produce apparently comparable scores from materially different
protocols. The shared framework centralizes these decisions and exposes no final
test-set argument, reducing both accidental protocol drift and leakage risk.

## Cross-validation design

The framework uses five-fold `StratifiedKFold` with shuffling and the shared
`random_state=42`. Stratification preserves the imbalanced target distribution
within each validation fold. Ordered train and validation indices receive
deterministic SHA-256 fingerprints so members can verify that models used the
same folds without persisting row-level data.

## Metrics

ROC-AUC is the configured primary metric because it measures ranking over all
classification thresholds. Average Precision is reported because it emphasizes
positive-class retrieval under class imbalance. F1, precision, recall, accuracy,
and balanced accuracy provide complementary threshold-based and class-balanced
views. Safe binary scorers support both numeric labels and the Santander string
labels and use zero-division handling where relevant.

## Result format and timing

`evaluate_model_cv` returns one row per fold with train/validation metrics, fit
time, score time, and fold sizes. Its serializable summary records aggregate
means and population standard deviations, configuration, estimator class and
parameters, target distribution, CV fingerprints, authorship metadata, and a
completed status. No fitted estimator object is serialized in the summary.

The export helper writes a fold CSV and summary JSON without silent overwrite.
The registry helper stores one traceability row per unique experiment ID and
rejects duplicates. Neither helper records absolute local paths.

## Leakage prevention

Only training data may be passed to cross-validation. Any learned scaling,
imputation, PCA, or feature selection must be inside the estimator's
scikit-learn `Pipeline`, ensuring it is fitted separately within each fold. The
framework does not perform any transformation before `cross_validate`. The
final test set remains closed and was not used in this work.

## Synthetic verification

Offline tests and the verification script use generated classification data and
a minimal `DummyClassifier` strictly as a technical smoke test. No Santander
model was trained or evaluated, no synthetic result was saved as a scientific
experiment, and no experiment registry was created.

## Limits

This infrastructure does not choose a decision threshold, tune parameters,
compare scientific models, or evaluate the final test set. Execution time can
vary with hardware and parallel scheduling even when folds and scores are
reproducible.

## Next step

Implement the first scientific `DummyClassifier` baseline on the shared
training split using a unique experiment ID and the common framework.

## Decision

Use one five-fold stratified protocol and seven common metrics for every model,
with all learned preprocessing inside each fold's Pipeline.

## Difficulties

Probability-based and label-based metrics require different estimator outputs,
and summaries must remain JSON serializable across pandas and NumPy types.

## Adaptations and deviations from the plan

The smoke verification uses synthetic data only and writes no scientific result.

## Rejected approaches

Global preprocessing, accuracy-only comparison, implicit test-set evaluation,
and registering smoke-test output as science were rejected.

## Files changed

- `src/evaluation.py`
- `scripts/verify_evaluation_framework.py`
- `tests/test_evaluation.py`
- `configs/config.yaml`

## Code references

`create_cv`, scoring construction, `evaluate_model_cv`, validation, and export
helpers in `src/evaluation.py`.

## Figure and table references

None; the verification is synthetic and intentionally not persisted.

## Reproducibility notes

Five-fold `StratifiedKFold` uses shuffling and `random_state=42`. The API accepts
training data only; the final test set remained closed.

## Sources and tools used

scikit-learn, pandas, NumPy, pytest, JSON, and Python.

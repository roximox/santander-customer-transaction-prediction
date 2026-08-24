# Logbook Entry

## Metadata

Date: 2026-08-10
Member: Chaymae Akouaouch
Sprint: Not confirmed
Ticket ID: M04-HGB-001
Branch: feature/model-optimization
Pull Request: Not created yet
Time spent: Not recorded yet
Related meeting: No related meeting yet

## Title

HistGradientBoosting baseline

## Goal

Establish the first Member 4 HistGradientBoosting baseline using the common
project infrastructure created by Member 1, while keeping the reserved final
test set completely untouched.

## Implementation

Created `src/gradient_boosting.py` with experiment ID `M04-HGB-001` and model
name `HistGradientBoosting Baseline`. The module reuses
`create_hist_gradient_boosting_classifier()` from `src/modeling.py` and defines
the following baseline configuration:

- `learning_rate=0.1`
- `max_iter=300`
- `max_leaf_nodes=31`
- `l2_regularization=0.0`
- `random_state=42`

`min_samples_leaf=20` and `early_stopping="auto"` remain scikit-learn defaults.

## Experiment runner

Created `scripts/run_gradient_boosting_baseline.py`. It reuses
`load_dataset(optimize_memory=True)`, the shared train/test split, train/test
fingerprint verification, `run_and_save_experiment`, the common five-fold
stratified cross-validation, and the common project metrics. After fingerprint
verification, the reserved test partition is removed from the experiment
workflow and is not evaluated.

## Work steps

1. Reviewed the existing project structure and the shared HistGradientBoosting
   factory in `src/modeling.py`.

2. Confirmed that `create_hist_gradient_boosting_classifier()` is a generic
   factory that creates an unfitted estimator and does not perform Member 4's
   experiment automatically.

3. Created `src/gradient_boosting.py` to define the concrete Member 4 baseline
   configuration for experiment `M04-HGB-001`.

4. Created `scripts/run_gradient_boosting_baseline.py` to connect the baseline
   with the existing shared data, validation, evaluation, and experiment
   infrastructure.

5. Verified that the new baseline could be imported and instantiated without
   training the model.

6. Inspected the estimator with `get_params()` and confirmed the intended
   baseline parameters and relevant scikit-learn defaults.

7. Executed:

   `python scripts/run_gradient_boosting_baseline.py`

   to run the first five-fold cross-validation experiment for `M04-HGB-001`.

8. Reviewed the validation metrics, per-fold results, and the train/validation
   ROC-AUC generalization gap.

9. Confirmed that the reserved final test partition was not evaluated.

10. Investigated why the generated M04 experiment artifacts did not appear in
    `git status` and identified the existing `.gitignore` rule
    `reports/experiments/*`. No Git configuration was changed.

    
## Verification before training

The new baseline factory was imported successfully. It instantiated
`HistGradientBoostingClassifier(max_iter=300, random_state=42)`, and its actual
parameters were verified with `get_params()`.

## Baseline experiment results

Experiment `M04-HGB-001` produced the following training-cross-validation
results:

| Metric | Value |
|---|---:|
| Validation ROC-AUC | 0.884596 ± 0.003278 |
| Validation Average Precision | 0.572879 ± 0.009277 |
| F1 | 0.387255 |
| Precision | 0.782671 |
| Recall | 0.257307 |
| Accuracy | 0.918181 |
| Balanced Accuracy | 0.624658 |
| Train ROC-AUC | 0.975659 |
| Validation ROC-AUC | 0.884596 |
| Generalization gap | 0.091063 |

## Fold results

| Fold | ROC-AUC | Average Precision |
|---:|---:|---:|
| 1 | 0.879752 | 0.566940 |
| 2 | 0.885401 | 0.568957 |
| 3 | 0.889694 | 0.589841 |
| 4 | 0.882831 | 0.563554 |
| 5 | 0.885303 | 0.575105 |

## Interpretation

The untuned HistGradientBoosting baseline already provides strong validation
ROC-AUC, and results are relatively stable across the five folds. The train
ROC-AUC of 0.975659 versus validation ROC-AUC of 0.884596 produces a
generalization gap of 0.091063. This indicates that model complexity and
generalization should be investigated during upcoming hyperparameter
optimization. Precision is high relative to recall, so later evaluation should
also inspect Precision-Recall behavior. This model is not claimed to be final
or optimal.

## Generated artifacts

- `reports/experiments/M04-HGB-001_fold_results.csv`
- `reports/experiments/M04-HGB-001_summary.json`
- `reports/experiments/experiment_registry.csv` was updated by the experiment
  infrastructure.

## Git/artifact observation

The two generated M04 experiment result files exist locally. They are currently
ignored by the existing `.gitignore` rule `reports/experiments/*`. `.gitignore`
was not changed as part of this work; the Git/artifact question will be handled
later.

## Reproducibility notes

The shared train/test split and both expected fingerprints were verified. Only
the training partition entered the common five-fold stratified
cross-validation. The final test partition remained reserved and was not
evaluated.

## Next planned work

Hyperparameter optimization for HistGradientBoosting is planned but has not yet
been implemented or executed.

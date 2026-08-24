# Logbook Entry

## Metadata

Date: 2026-08-22
Member: Chaymae Akouaouch
Sprint: Not confirmed
Ticket ID: M04-HGB-002
Branch: feature/model-optimization
Pull Request: Not created yet
Time spent: Not recorded yet
Related meeting: No related meeting yet

## Title

Registered tuned HistGradientBoosting experiment

## Goal

Document the registered training-only evaluation of the already selected tuned
HistGradientBoosting configuration. `M04-HGB-002` was not another
hyperparameter search. Its purpose was to evaluate the frozen best
configuration selected by `M04-HGB-SEARCH-001` through the same common
experiment framework used by `M04-HGB-001`, so that baseline and tuned HGB are
consistently represented in the shared experiment registry.

## Tuned configuration

The following parameters were frozen from `M04-HGB-SEARCH-001`:

- `learning_rate = 0.05`
- `max_iter = 700`
- `max_leaf_nodes = 31`
- `min_samples_leaf = 100`
- `l2_regularization = 10.0`
- `random_state = 42`

These parameters were not selected using `M04-HGB-002` results.

## Method

- Reused the common dataset loader.
- Recreated the common 80/20 train/reserved-test split.
- Verified the training fingerprint.
- Verified the reserved-test fingerprint.
- Deleted the reserved test partition before evaluation.
- Evaluated the training partition only.
- Reused the common `StratifiedKFold` cross-validation strategy with 5 folds
  and `random_state = 42`.
- Reused the common project evaluation metrics.
- Used `n_jobs = 1`.
- Performed evaluation and persistence through `run_and_save_experiment()`.
- Performed no threshold tuning.
- Performed no new hyperparameter tuning.
- Performed no final-test evaluation.

## Results

| Metric | Value |
|---|---:|
| Validation ROC-AUC | 0.891449 ± 0.002836 |
| Validation Average Precision | 0.591089 ± 0.010028 |
| F1 | 0.415248 |
| Precision | 0.795747 |
| Recall | 0.280942 |
| Accuracy | 0.920487 |
| Balanced Accuracy | 0.636438 |
| Train ROC-AUC | 0.973580 |
| Validation ROC-AUC | 0.891449 |
| Generalization gap | 0.082131 |

## Fold results

| Fold | ROC-AUC | Average Precision |
|---:|---:|---:|
| 1 | 0.887573 | 0.586082 |
| 2 | 0.891348 | 0.586814 |
| 3 | 0.896350 | 0.609012 |
| 4 | 0.890377 | 0.579664 |
| 5 | 0.891598 | 0.593873 |

## Interpretation

The registered tuned experiment reproduces the best
`M04-HGB-SEARCH-001` validation ROC-AUC of 0.891449. This consistency is
expected because `M04-HGB-002` evaluates the same frozen configuration with
the same common five-fold CV methodology.

| Metric | M04-HGB-001 baseline | M04-HGB-002 tuned |
|---|---:|---:|
| ROC-AUC | 0.884596 | 0.891449 |
| Average Precision | 0.572879 | 0.591089 |
| F1 | 0.387255 | 0.415248 |

The tuned registered experiment improves over the baseline. These are
training-only cross-validation results, not final-test performance. The tuned
experiment is not claimed to be the globally best project model.

## Generated artifacts

- `reports/experiments/M04-HGB-002_fold_results.csv`
- `reports/experiments/M04-HGB-002_summary.json`
- One `M04-HGB-002` row appended by the common experiment framework to
  `reports/experiments/experiment_registry.csv`.

## Relationship to previous Member 4 work

`M04-HGB-001`  
→ established the HGB baseline

`M04-HGB-SEARCH-001`  
→ selected the tuned configuration

`M04-HGB-002`  
→ registered that frozen configuration using the common experiment framework

`M04-HGB-LC-001`  
→ analyzed learning behavior of the tuned configuration

`M04-HGB-OOF-001`  
→ provided training-only OOF diagnostics

`M04-HGB-COMP-001`  
→ compared baseline and tuned performance

No previous experiment needs to be rerun because of `M04-HGB-002`.

## Reproducibility and leakage prevention

- Final test observations were not evaluated.
- Only the reserved-test fingerprint was verified.
- `X_reserved` and `y_reserved` were removed before evaluation.
- No test metric was used for tuning or selection.
- `M04-HGB-002` did not introduce a new tuning decision.

## Conclusion

`M04-HGB-002` provides the common-framework registered version of Member 4's
tuned HGB candidate. The tuned HGB is ready to be handed to later group-level
model selection. The final reserved test set must remain untouched until the
group has selected its final model.

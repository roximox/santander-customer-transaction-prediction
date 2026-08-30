# Logbook Entry

## Metadata

- Date: 2026-08-11
- Member: Yassine Elhari
- Sprint: Sprint 1
- Ticket ID: ADA-ML-08
- Branch: feature/data_processing
- Pull Request: [#7 — develop → main](https://github.com/roximox/santander-customer-transaction-prediction/pull/7) (integrated into `develop` in `ff3c2f8`)
- Time spent: 4 hours
- Related meeting: [2026-08-16 — First Individual Analysis and Machine Learning Progress](../../../meetings/2026-08-16_first-individual-analysis-and-machine-learning-progress.md)

## Title

Logistic Regression Learning Curves

## Objective

Measure how training volume affects the selected ROC-AUC Logistic Regression
and its recall-oriented balanced alternative without opening the final test.

## Context

Both configurations use L2 regularization, `C=0.01`, `solver="saga"`,
`max_iter=2000`, and `random_state=42`; only `class_weight` differs. The shared
160,000-row training partition and five-fold stratified CV remain unchanged.

## Work performed

Added a reusable learning-curve module, an offline test suite, a protected
scientific runner, four vector figures, result and decision tables, notebook
reporting cells, and project documentation.

## Methodology

Each CV training fold is stratified down independently to 5%, 10%, 25%, 50%,
75%, and 100%. A fresh clone of the shared factory pipeline is fitted for every
fold/fraction pair. Train metrics use the actual subset; validation metrics use
the complete validation fold. Timing, iterations, and convergence warnings are
captured per fit. The reserved test fingerprint is checked before its objects
are deleted; it is never scored or predicted.

## Results

For LR-LEARNING-ROC, validation ROC-AUC progresses from 0.839782 at 5% to
0.849354, 0.855389, 0.857732, 0.858745, and 0.859201 at 100%. Average Precision
progresses from 0.457493 to 0.481314, 0.498673, 0.504275, 0.506432, and 0.507592.
For LR-LEARNING-BALANCED, ROC-AUC progresses from 0.834234 to 0.845888,
0.854096, 0.857267, 0.858418, and 0.859017; Average Precision progresses from
0.443401 to 0.472769, 0.494634, 0.502437, 0.504984, and 0.506454.

All 60 fits converged. The measured campaign duration was 203.18 seconds.

## Interpretation

Both configurations continue improving through 100%, with diminishing gains.
From 5%→100%, ROC/AP gains are +0.019418/+0.050099 unweighted and
+0.024783/+0.063053 balanced. From 75%→100%, they fall to
+0.000456/+0.001160 and +0.000599/+0.001470 respectively, supporting an
empirical late plateau. ROC-AUC train–validation gaps contract from 0.049999 to
0.002325 unweighted and from 0.060521 to 0.002805 balanced. Mean fit time rises
from 0.148 to 3.084 seconds unweighted and 0.444 to 4.733 seconds balanced.
Fold standard deviations at 100% are 0.003236/0.008491 (ROC/AP) unweighted and
0.003122/0.008518 balanced. Balanced retains much higher recall at 100%
(0.773666 versus 0.267758) but lower precision (0.284593 versus 0.691427).

## Decision

The observed 75%→100% gains support a late empirical plateau for both models;
the unweighted configuration remains marginally stronger on ranking metrics,
while balanced remains the recall-oriented alternative. This analysis does
not select a final model or threshold.

## Difficulties

The campaign requires 60 SAGA fits on 200 numerical features. Runtime is
reported separately from predictive performance and is machine-dependent.

## Adaptations and deviations from the plan

The common model factory, shared metrics, split utility, CV factory, and
fingerprint implementation were reused. The public learning-curve API has no
test-set parameter.

## Rejected approaches

- scikit-learn's generic `learning_curve` helper was not used because the task
  requires per-fit counts, warnings, iterations, and all project metrics.
- The validation folds were not subsampled.
- Existing historical scientific outputs were not changed.
- No trained model or complete dataset was persisted.

## Files changed

- `src/learning_curves.py`
- `scripts/run_logistic_learning_curves.py`
- `tests/test_learning_curves.py`
- `notebooks/03_logistic_regression.ipynb`
- `README.md`
- `CONTRIBUTING.md`
- learning-curve tables and figures listed below

## Code references

- `validate_train_size_fractions`
- `create_stratified_subsample`
- `compute_learning_curve`
- `summarize_learning_curve`
- `create_learning_curve_figures`
- `build_decision_table`

## Figure and table references

- `reports/tables/logistic_learning_curve_folds.csv`
- `reports/tables/logistic_learning_curve_summary.csv`
- `reports/tables/logistic_learning_curve_summary.json`
- `reports/tables/logistic_learning_curve_decision.csv`
- `reports/figures/logistic_learning_curve_roc_auc.pdf`
- `reports/figures/logistic_learning_curve_average_precision.pdf`
- `reports/figures/logistic_learning_curve_fit_time.pdf`
- `reports/figures/logistic_learning_curve_threshold_metrics.pdf`

## Reproducibility notes

OpenML ID 45566, float32 features, shared split fingerprints, random state 42,
and five-fold `StratifiedKFold` are checked or constructed explicitly. The
runner refuses to overwrite any existing target artifact.

## Next step

Any later final-test evaluation requires a separate, explicitly authorized
task.

## Sources and tools used

Project source code and configuration, pandas, NumPy, scikit-learn, Matplotlib,
pytest, nbformat, OpenML's scikit-learn loader, and Git inspection commands.

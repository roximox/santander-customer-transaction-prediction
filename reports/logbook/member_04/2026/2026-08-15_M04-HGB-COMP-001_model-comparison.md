# Logbook Entry

## Metadata

Date: 2026-08-15
Member: Chaymae Akouaouch
Sprint: Not confirmed
Ticket ID: M04-HGB-COMP-001
Branch: feature/model-optimization
Pull Request: Not created yet
Time spent: Not recorded yet
Related meeting: No related meeting yet

## Title

HistGradientBoosting baseline vs tuned model comparison

## Goal

Document the comparison between the original Member 4 HistGradientBoosting
baseline (`M04-HGB-001`) and the tuned HistGradientBoosting model evaluated
with training-only out-of-fold predictions (`M04-HGB-OOF-001`). The goal is to
determine whether the tuning work improved the Member 4 HGB model before
group-level model selection.

## Data sources

No model was trained during this step. The comparison reused existing
artifacts:

- `reports/experiments/M04-HGB-001_summary.json`
- `reports/tables/M04-HGB-OOF-001_metrics.json`
- `reports/searches/M04-HGB-SEARCH-001_summary.json`

Baseline values come from five-fold validation means. Tuned values come from
aggregated five-fold out-of-fold predictions. Both are training-only estimates,
not final-test evaluation.

## Tuned configuration

The selected configuration from `M04-HGB-SEARCH-001` was:

- `learning_rate = 0.05`
- `max_iter = 700`
- `max_leaf_nodes = 31`
- `min_samples_leaf = 100`
- `l2_regularization = 10.0`
- `random_state = 42`

No new tuning was performed during comparison.

## Results

| Metric | Baseline HGB | Tuned HGB OOF | Absolute change |
|---|---:|---:|---:|
| ROC-AUC | 0.884596 | 0.891438 | +0.006842 |
| Average Precision | 0.572879 | 0.590860 | +0.017981 |
| F1 | 0.387255 | 0.415242 | +0.027987 |
| Precision | 0.782671 | 0.795527 | +0.012856 |
| Recall | 0.257307 | 0.280943 | +0.023636 |
| Accuracy | 0.918181 | 0.920488 | +0.002307 |
| Balanced Accuracy | 0.624658 | 0.636438 | +0.011780 |

All seven compared metrics increased.

## Interpretation

ROC-AUC improved from 0.884596 to 0.891438. Average Precision improved from
0.572879 to 0.590860, and F1 improved from 0.387255 to 0.415242. Recall
improved from 0.257307 to 0.280943, while precision also increased from
0.782671 to 0.795527. Accuracy and Balanced Accuracy also increased.

Therefore, the tuned HGB configuration shows an improvement over the original
Member 4 baseline across all seven reported metrics.

This is not final-test performance. The baseline and tuned values come from
closely related but not identical aggregation procedures: baseline values are
mean five-fold validation metrics, whereas tuned values are metrics from
aggregated OOF predictions. Therefore, the comparison is evidence of
improvement, but it must not be presented as an independent final-test
comparison. The tuned HGB must not yet be called the globally best project
model.

## Implementation

`src/gradient_boosting_comparison.py` contains reusable comparison logic for:

- Loading existing Member 4 results.
- Validating source artifacts.
- Constructing the comparison table.
- Computing absolute metric changes.
- Saving CSV.
- Saving JSON.
- Creating the comparison figure.

`scripts/run_gradient_boosting_comparison.py` provides the lightweight runner
that reads existing result artifacts, builds the comparison, saves outputs, and
prints the improvement summary.

No Santander dataset loading, model fitting, cross-validation, OOF prediction,
or hyperparameter search was performed in this comparison step.

`save_figure()` from `src/visualization.py` was reused for figure persistence
without modifying that shared file.

## Generated artifacts

- `reports/tables/M04-HGB-model-comparison.csv`
- `reports/tables/M04-HGB-model-comparison.json`
- `reports/figures/M04-HGB-model-comparison.pdf`

Their Git tracking status was not verified in this entry.

## Reproducibility and leakage prevention

- Comparison based only on already-generated training-only results.
- No dataset loading.
- No model training.
- No new cross-validation.
- No OOF recomputation.
- No hyperparameter tuning.
- No threshold tuning.
- No model selection.
- Final reserved test partition was not accessed or evaluated.

## Conclusion

Member 4's tuning improved the HistGradientBoosting baseline across all seven
compared metrics. The tuned HGB result can now be handed over as Member 4's
candidate for the group-level model-selection stage. It is not the final
project model.

## Next planned work

The next step is to verify and consolidate all Member 4 artifacts, tests,
documentation, and Git status before handing the tuned HGB candidate to the
common group-level model-selection procedure. The reserved final test set must
remain untouched until the group-level selection procedure has selected the
final model.

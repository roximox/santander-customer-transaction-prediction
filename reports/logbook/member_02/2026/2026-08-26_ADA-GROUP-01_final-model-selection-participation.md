# Logbook Entry

## Metadata

- Date: 2026-08-26
- Member: Aya Olali
- Sprint: Final model selection
- Ticket ID: ADA-GROUP-01
- Branch: develop
- Pull Request: Committed to develop through merge commit `b7c5bce`
- Time spent: Not recorded — to be completed by Aya Olali
- Related meeting: [2026-08-26 — Final model selection decision](../../../meetings/2026-08-26_final-model-selection-decision.md)

## Title

Member 02 contribution to the collective final-model selection

## Objective

Review how the Member 02 Decision Tree and Random Forest baselines compare with
the other eligible project experiments and contribute their evidence to the
collective model-selection decision without reopening the reserved final test.

## Context

The EDA and tree experiments had been merged into `develop`. Experiments
`M02-DT-001` and `M02-RF-001` used the official 160,000-row development
partition and the shared five-fold stratified cross-validation protocol. The
group meeting compared the eligible models using consistent validation
evidence before locking a final model.

## Work performed

- Reviewed the registered Decision Tree and Random Forest summaries.
- Confirmed the Member 02 results were represented in the common comparison.
- Compared ROC-AUC, Average Precision, balanced accuracy, stability, and
  train-validation gaps rather than relying on accuracy alone.
- Presented the Random Forest as the stronger Member 02 tree baseline while
  retaining its larger generalization gap as a limitation.
- Participated in the collective decision to keep model selection separate
  from the still-closed final-test partition.

## Methodology

Only repository-local cross-validation summaries and the common model-selection
reports were used. No estimator was refitted, no hyperparameter was changed,
and no final-test result was available during the selection discussion.

## Results

`M02-RF-001` remained stronger than `M02-DT-001`: validation ROC-AUC was
0.793736 versus 0.633935, and Average Precision was 0.369114 versus 0.163973.
The Random Forest did not outperform the strongest eligible project models and
therefore was not selected as the final model. The group selected and locked
`M04-HGB-002` using the collective comparison evidence.

## Interpretation

The Random Forest demonstrated useful nonlinear predictive signal and provided
a meaningful ensemble baseline. Its lower validation performance and larger
train-validation gap relative to the selected candidate justified retaining it
as supporting evidence rather than choosing it as the final model.

## Decision

Keep `M02-DT-001` and `M02-RF-001` as completed Member 02 baselines. Accept the
collective lock of `M04-HGB-002`; do not tune or select another model after the
final-test partition is opened.

## Difficulties

The model families differed in complexity and threshold behaviour. A fair
comparison required separating ranking metrics, threshold-dependent metrics,
computational cost, and overfitting evidence.

## Adaptations and deviations from the plan

Extra Trees was part of the initial Member 02 assignment, but the registered
Member 02 evidence comprised Decision Tree and Random Forest. The collective
decision used only completed, comparable experiments rather than adding a late
Member 02 experiment.

## Rejected approaches

- Selecting Random Forest only because it beat the single Decision Tree.
- Using the reserved final-test set to resolve the model choice.
- Hiding the Random Forest train-validation gap.
- Starting a new tuning cycle after the collective comparison.

## Files changed

- This Logbook entry only; the underlying model-selection artifacts were
  collective project files already present in `develop`.

## Code references

- `scripts/run_tree_models.py`
- `src/model_selection.py`
- `scripts/build_model_selection_report.py`

## Figure and table references

- `reports/tables/tree_model_comparison.csv`
- `reports/figures/tree_model_metrics.pdf`
- `reports/model_selection/model_comparison_eligible.csv`
- `reports/model_selection/model_selection_decision.csv`

## Reproducibility notes

The Member 02 scores come from the registered shared-CV summaries. The final
test remained closed during this decision and was not used to rank candidates.

## Sources and tools used

Registered experiment JSON/CSV files, collective model-selection reports, Git
history, and the 2026-08-26 meeting record.

## Next step

Prepare the most relevant EDA and tree-model evidence for the final project
presentation while preserving the distinction between individual and
collective results.

# Logbook Entry

## Metadata

- Date: 2026-08-09
- Member: Aya Olali
- Sprint: Validation and individual analysis start
- Ticket ID: ADA-MEET-03
- Branch: develop
- Pull Request: Not applicable — meeting, preparation, and analysis work
- Time spent: 6 hours (retrospective estimate)
- Related meeting: [2026-08-09 — Data processing, validation strategy and start of individual analysis](../../../meetings/2026-08-09_data-processing-validation-strategy-and-start-of-individual-analysis.md)

## Title

Validation-method discussion and leakage-safe EDA preparation

## Objective

Understand the official train/test and cross-validation rules and incorporate
them into the Member 02 EDA and planned tree experiments.

## Context

The group finalized the 80/20 stratified split, five-fold StratifiedKFold,
`random_state=42`, common metrics, and the closed final-test boundary.

## Work performed

- Reviewed the 160,000-development / 40,000-reserved split.
- Discussed leakage risks in EDA, feature choice, tuning, and threshold selection.
- Reviewed why class imbalance makes accuracy insufficient.
- Connected ROC-AUC, Average Precision, recall, and balanced accuracy to the
  planned tree comparison.
- Prepared the transition from EDA observations to baseline model questions.
- Discussed reproducibility requirements with colleagues.

## Methodology

All exploratory and model-development decisions were restricted to the
development partition. Five shared folds would support comparable evidence.

## Results

The methodological boundary for Member 02 became explicit: EDA on training data
only, tree evaluation through shared CV, and no final-test model comparison.

## Interpretation

For an approximately 10% positive class, ranking and precision-recall evidence
are required alongside threshold-based metrics.

## Decision

Use the shared split, seed, folds, and metrics in every Member 02 workflow.

## Difficulties

The initial notebook approach needed to be checked carefully because direct
test scoring would violate the agreed development protocol.

## Adaptations and deviations from the plan

The scientific plan was tightened to make the reserved-test restriction explicit.

## Rejected approaches

Non-stratified splitting, accuracy-only reporting, global preprocessing, and
iterative test evaluation were rejected.

## Files changed

- This retrospective Logbook entry only.

## Code references

- `src/validation.py`
- `src/evaluation.py`
- `configs/config.yaml`

## Figure and table references

- `reports/tables/train_test_split_summary.json`

## Reproducibility notes

The final test remained closed. This work defined how later EDA and experiments
would respect that boundary.

## Sources and tools used

Shared validation modules, configuration, project documentation, colleague
discussion, and meeting record.

## Next step

Complete the first EDA implementation and prepare its findings for peer review.

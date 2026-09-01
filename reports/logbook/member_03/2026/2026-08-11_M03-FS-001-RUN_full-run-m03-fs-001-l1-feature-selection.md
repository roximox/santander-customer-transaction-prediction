# 2026-08-11 · M03-FS-001-RUN — Full run M03-FS-001 (L1 feature selection)

| Field | Value |
| --- | --- |
| **Date** | 2026-08-11 |
| **Member** | Ilias El Hamri (Member 03) |
| **Ticket ID** | M03-FS-001-RUN |
| **Branch** | feature/feature-selection |
| **Time spent** | 6.0 h |
| **Related meeting** | 2026-08-09 |

## Objective

Execute the L1 pipeline on the full development partition under 5-fold CV.

## Work performed

- Ran scripts/run_feature_selection.py (wall time ~80 min for 5 folds)
- Collected per-fold metrics and mean fit time
- Wrote M03-FS-001_summary.json
- Generated reports/figures/feature_selection_cv_scores.pdf

## Methodology

5-fold StratifiedKFold, shuffle=True, random_state=42; ROC-AUC primary.

## Results

Mean validation ROC-AUC 0.859187 ± 0.003237; Average Precision 0.507566; mean fit time 15.876 s per fold. Almost identical to baseline.

## Decision

Accept C=0.1 result; record as null finding relative to unreduced LR.

## Rejected approaches

Re-running with different C immediately — deferred to keep single controlled experiment.

## Files changed

reports/experiments/M03-FS-001_summary.json, reports/figures/feature_selection_cv_scores.pdf

## Figure and table references

reports/figures/feature_selection_cv_scores.pdf

## Difficulties / Adaptations

saga solver slow and occasionally hit max_iter; raised max_iter to 2000 and accepted longer fit time.

## Next step

Full run of PCA pipeline.

## Reproducibility notes

All estimators take `random_state = 42` from `configs/config.yaml` where applicable. Shared stratified 5-fold CV with `shuffle = True` is created by `create_stratified_cv()`. Split fingerprints are verified against the shared project constants before any experiment runs. The reserved final test partition was not used for any M03 development experiment.

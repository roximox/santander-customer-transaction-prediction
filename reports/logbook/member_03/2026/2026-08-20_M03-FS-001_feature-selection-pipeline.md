# 2026-08-20 · M03-FS-001 — L1 feature selection + Logistic Regression pipeline (formal ticket)

| Field | Value |
| --- | --- |
| **Date** | 2026-08-20 |
| **Member** | Ilias El Hamri (Member 03) |
| **Ticket ID** | M03-FS-001 |
| **Branch** | develop |
| **Time spent** | 2.0 h |
| **Related meeting** | 2026-08-16 |

## Objective

Formalise the already-executed M03-FS-001 experiment as a complete logbook entry.

## Work performed

- Consolidated all metrics, files and decisions into the standard ticket template
- Recorded saga slow-down and missing retained-feature count as difficulties

## Methodology

Same as earlier runs. StandardScaler → SelectFromModel(L1, C=0.1, saga) → L2 LR inside one Pipeline, refit per fold.

## Results

Mean validation ROC-AUC 0.859187 ± 0.003237; Average Precision 0.507566; F1 0.390561; precision 0.688834; recall 0.272671; balanced accuracy 0.629432; train–validation ROC-AUC gap 0.002338; mean fit time 15.876 s per fold. Final test partition not evaluated.

## Decision

C=0.1 kept; sweep deferred. Result statistically indistinguishable from unreduced baseline M01-LR-001.

## Rejected approaches

Fitting the selector on the full development partition before CV (leaks). Univariate filter selection (ignores multivariate structure).

## Files changed

src/feature_selection.py; scripts/run_feature_selection.py; tests/test_feature_selection.py; notebooks/05_feature_selection.ipynb; reports/logbook/member_03/2026/2026-08-20_M03-FS-001_feature-selection-pipeline.md

## Figure and table references

- `reports/figures/feature_selection_cv_scores.pdf`

## Difficulties / Adaptations

Saga solver slow on 128k×200; memory pressure during float32 conversion on first attempts; retained feature count per fold never logged.

## Next step

Formal PCA ticket.

## Reproducibility notes

All estimators take `random_state = 42` from `configs/config.yaml` where applicable. Shared stratified 5-fold CV with `shuffle = True` is created by `create_stratified_cv()`. Split fingerprints are verified against the shared project constants before any experiment runs. The reserved final test partition was not used for any M03 development experiment.

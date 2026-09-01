# 2026-08-20 · M03-PCA-001 — PCA (95% variance) + Logistic Regression pipeline (formal ticket)

| Field | Value |
| --- | --- |
| **Date** | 2026-08-20 |
| **Member** | Ilias El Hamri (Member 03) |
| **Ticket ID** | M03-PCA-001 |
| **Branch** | feature/pca |
| **Time spent** | 2.0 h |
| **Related meeting** | 2026-08-16 |

## Objective

Formalise the already-executed M03-PCA-001 experiment.

## Work performed

- Consolidated metrics and limitations (component count not stored)
- Noted that comparison figure referenced by scripts was never auto-generated

## Methodology

Same CV protocol. StandardScaler → PCA(n_components=0.95) → L2 LR inside one Pipeline, refit per fold.

## Results

Mean validation ROC-AUC 0.858865 ± 0.003267; Average Precision 0.506556; F1 0.389600; precision 0.689166; recall 0.271676; balanced accuracy 0.628973; train–validation ROC-AUC gap 0.002340; mean fit time 2.197 s per fold. Final test partition not evaluated.

## Decision

n_components=0.95 kept. Difference from baseline −0.000322 ROC-AUC is within fold noise; pure overhead.

## Rejected approaches

Fitting PCA globally before CV (leaks). Fixed number of components without justification.

## Files changed

src/feature_selection.py; scripts/run_pca.py; tests/test_feature_selection.py; notebooks/06_pca.ipynb; reports/logbook/member_03/2026/2026-08-20_M03-PCA-001_pca-pipeline.md

## Figure and table references

reports/figures/pca_cv_scores.pdf. Note: comparison figure feature_selection_vs_pca_metrics.pdf and table feature_selection_pca_comparison.csv referenced by run scripts were never generated; portfolio Figures 1–3 replace them.

## Difficulties / Adaptations

Per-fold component count not recorded; comparison artefacts had to be produced manually for the portfolio.

## Next step

Consolidation work for group comparison.

## Reproducibility notes

All estimators take `random_state = 42` from `configs/config.yaml` where applicable. Shared stratified 5-fold CV with `shuffle = True` is created by `create_stratified_cv()`. Split fingerprints are verified against the shared project constants before any experiment runs. The reserved final test partition was not used for any M03 development experiment.

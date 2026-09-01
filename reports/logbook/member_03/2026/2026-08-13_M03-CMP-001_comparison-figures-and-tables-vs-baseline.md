# 2026-08-13 · M03-CMP-001 — Comparison figures and tables vs baseline

| Field | Value |
| --- | --- |
| **Date** | 2026-08-13 |
| **Member** | Ilias El Hamri (Member 03) |
| **Ticket ID** | M03-CMP-001 |
| **Branch** | develop |
| **Time spent** | 4.0 h |
| **Related meeting** | 2026-08-16 |

## Objective

Produce side-by-side ROC-AUC, multi-metric and fit-time figures for the three pipelines.

## Work performed

- Loaded the three summary JSONs
- Built Figure 1 (per-fold ROC-AUC), Figure 2 (six metrics), Figure 3 (fit time log scale)
- Created comparison table for the portfolio
- Verified that fold indices align across experiments

## Methodology

All numbers taken directly from the JSON artefacts; no re-computation.

## Results

Differences << fold sd; L1 ~25× slower, PCA ~3.4× slower than baseline.

## Decision

Treat both reductions as pure overhead on this dataset.

## Rejected approaches

Statistical equivalence test — descriptive comparison only, as planned.

## Files changed

reports/figures/* (local for portfolio), comparison notes

## Figure and table references

- `reports/figures/feature_selection_cv_scores.pdf`
- `reports/figures/pca_cv_scores.pdf`

## Difficulties / Adaptations

The automated comparison figure referenced by the run scripts was never generated; produced manually for the portfolio.

## Next step

Write notebook versions of the two experiments.

## Reproducibility notes

All estimators take `random_state = 42` from `configs/config.yaml` where applicable. Shared stratified 5-fold CV with `shuffle = True` is created by `create_stratified_cv()`. Split fingerprints are verified against the shared project constants before any experiment runs. The reserved final test partition was not used for any M03 development experiment.

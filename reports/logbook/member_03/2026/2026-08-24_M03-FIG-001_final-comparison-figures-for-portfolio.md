# 2026-08-24 · M03-FIG-001 — Final comparison figures for portfolio

| Field | Value |
| --- | --- |
| **Date** | 2026-08-24 |
| **Member** | Ilias El Hamri (Member 03) |
| **Ticket ID** | M03-FIG-001 |
| **Branch** | develop |
| **Time spent** | 3.5 h |
| **Related meeting** | 2026-08-23 |

## Objective

Produce publication-ready Figures 1–3 used in section 3.

## Work performed

- Re-generated per-fold ROC-AUC, multi-metric bar chart, log-scale fit-time chart
- Truncated y-axis on ROC-AUC plot so differences are visible
- Exported high-resolution PDF/PNG

## Methodology

All values from the registered summary JSONs.

## Results

Figures ready for insertion into the portfolio.

## Decision

Use truncated axis with explicit caption warning.

## Rejected approaches

0–1 axis — would make the three bars indistinguishable.

## Files changed

reports/figures/ (portfolio versions)

## Figure and table references

- `reports/figures/fig1_fold_roc_auc.png`
- `reports/figures/fig2_metric_profile.png`
- `reports/figures/fig3_fit_time.png`

## Difficulties / Adaptations

Matplotlib version differences caused minor layout shifts; fixed by explicit rcParams.

## Next step

Draft section 3 of the portfolio.

## Reproducibility notes

All estimators take `random_state = 42` from `configs/config.yaml` where applicable. Shared stratified 5-fold CV with `shuffle = True` is created by `create_stratified_cv()`. Split fingerprints are verified against the shared project constants before any experiment runs. The reserved final test partition was not used for any M03 development experiment.

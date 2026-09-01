# 2026-08-12 · M03-PCA-001-RUN — Full run M03-PCA-001 (PCA 95%)

| Field | Value |
| --- | --- |
| **Date** | 2026-08-12 |
| **Member** | Ilias El Hamri (Member 03) |
| **Ticket ID** | M03-PCA-001-RUN |
| **Branch** | feature/pca |
| **Time spent** | 3.5 h |
| **Related meeting** | 2026-08-09 |

## Objective

Execute the PCA pipeline on the full development partition.

## Work performed

- Ran scripts/run_pca.py
- Collected metrics; mean fit time 2.20 s/fold
- Wrote M03-PCA-001_summary.json
- Generated pca_cv_scores.pdf

## Methodology

Same CV protocol as FS run.

## Results

Mean validation ROC-AUC 0.858865 ± 0.003267; Average Precision 0.506556; mean fit time 2.197 s per fold. Slightly lower than baseline but within noise.

## Decision

Record as second null finding; PCA cheaper than L1 but still overhead.

## Rejected approaches

None.

## Files changed

reports/experiments/M03-PCA-001_summary.json, reports/figures/pca_cv_scores.pdf

## Figure and table references

reports/figures/pca_cv_scores.pdf

## Difficulties / Adaptations

Number of components selected per fold was not logged — noted as reporting gap.

## Next step

Compare both against M01-LR-001 baseline.

## Reproducibility notes

All estimators take `random_state = 42` from `configs/config.yaml` where applicable. Shared stratified 5-fold CV with `shuffle = True` is created by `create_stratified_cv()`. Split fingerprints are verified against the shared project constants before any experiment runs. The reserved final test partition was not used for any M03 development experiment.

# 2026-08-03 · M03-FS-000 — Skeleton of feature-selection module

| Field | Value |
| --- | --- |
| **Date** | 2026-08-03 |
| **Member** | Ilias El Hamri (Member 03) |
| **Ticket ID** | M03-FS-000 |
| **Branch** | develop |
| **Time spent** | 3.0 h |
| **Related meeting** | 2026-08-02 |

## Objective

Create src/feature_selection.py with empty pipeline factory functions.

## Work performed

- Added create_feature_selection_pipeline() stub
- Added create_pca_pipeline() stub
- Imported StandardScaler, SelectFromModel, PCA, LogisticRegression
- Added config loading for C and n_components from configs/config.yaml

## Methodology

All transformations must live inside a single Pipeline object.

## Results

Module compiles; stubs return Pipeline skeletons.

## Decision

Default C=0.1 for L1 selector; n_components=0.95 for PCA.

## Rejected approaches

Separate scaler outside the pipeline — rejected because of leakage risk.

## Files changed

src/feature_selection.py

## Difficulties / Adaptations

None.

## Next step

Write unit tests that will fail until implementation is complete.

## Reproducibility notes

All estimators take `random_state = 42` from `configs/config.yaml` where applicable. Shared stratified 5-fold CV with `shuffle = True` is created by `create_stratified_cv()`. Split fingerprints are verified against the shared project constants before any experiment runs. The reserved final test partition was not used for any M03 development experiment.

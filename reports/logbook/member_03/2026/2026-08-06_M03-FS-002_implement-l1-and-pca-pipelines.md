# 2026-08-06 · M03-FS-002 — Implement L1 SelectFromModel pipeline and PCA variance-threshold pipeline

| Field | Value |
| --- | --- |
| **Date** | 2026-08-06 |
| **Member** | Ilias El Hamri (Member 03) |
| **Ticket ID** | M03-FS-002 |
| **Branch** | feature/feature-selection |
| **Time spent** | 8.5 h |
| **Related meeting** | 2026-08-09 |

## Objective

Fill create_feature_selection_pipeline and create_pca_pipeline with working leakage-safe Pipelines.

## Work performed

- StandardScaler → SelectFromModel(LogisticRegression(penalty='l1', C=0.1, solver='saga')) → LogisticRegression(penalty='l2')
- StandardScaler → PCA(n_components=0.95) → LogisticRegression(penalty='l2')
- Exposed random_state from config; max_iter=1000 for saga
- Verified pipeline step names; synthetic tests now pass for feature reduction

## Methodology

saga is the only solver that supports L1 on dense float32 matrices of this size. Variance threshold lets each fold decide its own component count.

## Results

Both pipeline objects build; synthetic tests pass.

## Decision

C=0.1 as moderate regularisation; n_components=0.95 as default; sweeps deferred.

## Rejected approaches

liblinear solver — does not scale well to 128k samples. Fixed n_components=50 — arbitrary.

## Files changed

src/feature_selection.py

## Figure and table references

—

## Difficulties / Adaptations

saga was slow on first synthetic runs; increased max_iter and set tol=1e-3.

## Next step

Write the run scripts that call the shared evaluation function.

## Reproducibility notes

All estimators take `random_state = 42` from `configs/config.yaml` where applicable. Shared stratified 5-fold CV with `shuffle = True` is created by `create_stratified_cv()`. Split fingerprints are verified against the shared project constants before any experiment runs. The reserved final test partition was not used for any M03 development experiment.

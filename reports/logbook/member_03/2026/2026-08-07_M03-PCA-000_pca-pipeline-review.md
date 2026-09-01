# 2026-08-07 · M03-PCA-000 — PCA pipeline review

| Field | Value |
| --- | --- |
| **Date** | 2026-08-07 |
| **Member** | Ilias El Hamri (Member 03) |
| **Ticket ID** | M03-PCA-000 |
| **Branch** | develop |
| **Time spent** | 3.0 h |
| **Related meeting** | 2026-08-09 |

## Objective

Finalise PCA pipeline and smoke tests.

## Work performed

- Confirmed PCA(n_components=0.95, random_state=42)
- Confirmed pipeline order: scale first, then project, then classify
- Smoke test on synthetic data

## Methodology

Variance threshold lets each fold decide its own component count, avoiding an arbitrary fixed integer.

## Results

Pipeline builds; synthetic tests pass.

## Decision

Keep 0.95 as default; document that a sweep is future work.

## Rejected approaches

Fixed n_components=50 — arbitrary and not data-driven.

## Files changed

src/feature_selection.py

## Difficulties / Adaptations

None.

## Next step

Write the run scripts that call the shared evaluation function.

## Reproducibility notes

All estimators take `random_state = 42` from `configs/config.yaml` where applicable. Shared stratified 5-fold CV with `shuffle = True` is created by `create_stratified_cv()`. Split fingerprints are verified against the shared project constants before any experiment runs. The reserved final test partition was not used for any M03 development experiment.

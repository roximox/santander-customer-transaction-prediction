# 2026-08-10 · M03-RUN-002 — PCA run script

| Field | Value |
| --- | --- |
| **Date** | 2026-08-10 |
| **Member** | Ilias El Hamri (Member 03) |
| **Ticket ID** | M03-RUN-002 |
| **Branch** | feature/pca |
| **Time spent** | 2.5 h |
| **Related meeting** | 2026-08-09 |

## Objective

scripts/run_pca.py analogous to the feature-selection runner.

## Work performed

- Mirrored structure of run_feature_selection.py
- Experiment ID M03-PCA-001
- Export CV scores figure
- Dry-run on subsample

## Methodology

Identical evaluation protocol.

## Results

Script ready.

## Decision

Both runners share the same fingerprint helper.

## Rejected approaches

None.

## Files changed

scripts/run_pca.py

## Figure and table references

—

## Difficulties / Adaptations

None.

## Next step

Full run of M03-FS-001 on the 160k development set.

## Reproducibility notes

All estimators take `random_state = 42` from `configs/config.yaml` where applicable. Shared stratified 5-fold CV with `shuffle = True` is created by `create_stratified_cv()`. Split fingerprints are verified against the shared project constants before any experiment runs. The reserved final test partition was not used for any M03 development experiment.

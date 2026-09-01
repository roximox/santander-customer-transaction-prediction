# 2026-08-09 · M03-RUN-001 — Feature-selection run script + fingerprint check

| Field | Value |
| --- | --- |
| **Date** | 2026-08-09 |
| **Member** | Ilias El Hamri (Member 03) |
| **Ticket ID** | M03-RUN-001 |
| **Branch** | develop |
| **Time spent** | 4.0 h |
| **Related meeting** | 2026-08-09 |

## Objective

scripts/run_feature_selection.py that verifies fingerprints and registers the experiment.

## Work performed

- Load development partition only
- Verify SHA-256 against shared constants
- Call create_feature_selection_pipeline()
- Call shared evaluate_model_cv (5-fold StratifiedKFold)
- Write reports/experiments/M03-FS-001_summary.json and fold CSV

## Methodology

All metrics come from the shared evaluate_model_cv so numbers are comparable.

## Results

Script skeleton ready; first dry-run on a 10k subsample succeeded.

## Decision

Experiment ID M03-FS-001 fixed.

## Rejected approaches

Custom CV loop — would break comparability with other members.

## Files changed

scripts/run_feature_selection.py

## Difficulties / Adaptations

Fingerprint mismatch on first attempt because of float32 vs float64; fixed by explicit astype.

## Next step

Same for PCA run script.

## Reproducibility notes

All estimators take `random_state = 42` from `configs/config.yaml` where applicable. Shared stratified 5-fold CV with `shuffle = True` is created by `create_stratified_cv()`. Split fingerprints are verified against the shared project constants before any experiment runs. The reserved final test partition was not used for any M03 development experiment.

# 2026-07-31 · M03-DATA-001 — Inspect shared data loader and split

| Field | Value |
| --- | --- |
| **Date** | 2026-07-31 |
| **Member** | Ilias El Hamri (Member 03) |
| **Ticket ID** | M03-DATA-001 |
| **Branch** | main |
| **Time spent** | 2.5 h |
| **Related meeting** | 2026-08-02 |

## Objective

Understand the stratified 80/20 split and fingerprint mechanism.

## Work performed

- Read src/data.py and the split summary JSON
- Verified SHA-256 fingerprints of development and reserved partitions
- Confirmed stratified split preserves ~10% positive class
- Checked that reserved partition is never loaded by development scripts

## Methodology

Fingerprint verification is a hard precondition for every experiment script.

## Results

160 000 development / 40 000 reserved; class balance preserved.

## Decision

Every M03 script will call the shared fingerprint check before running.

## Rejected approaches

Loading the full 200k and splitting locally — rejected to keep comparability.

## Files changed

reports/tables/train_test_split_summary.json

## Figure and table references

—

## Difficulties / Adaptations

Fingerprint helper initially raised on Windows path separators; fixed by normalising paths.

## Next step

Prepare skeleton of feature-selection module.

## Reproducibility notes

All estimators take `random_state = 42` from `configs/config.yaml` where applicable. Shared stratified 5-fold CV with `shuffle = True` is created by `create_stratified_cv()`. Split fingerprints are verified against the shared project constants before any experiment runs. The reserved final test partition was not used for any M03 development experiment.

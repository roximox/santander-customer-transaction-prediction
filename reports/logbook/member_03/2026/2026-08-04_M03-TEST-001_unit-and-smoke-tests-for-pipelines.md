# 2026-08-04 · M03-TEST-001 — Unit and smoke tests for pipelines

| Field | Value |
| --- | --- |
| **Date** | 2026-08-04 |
| **Member** | Ilias El Hamri (Member 03) |
| **Ticket ID** | M03-TEST-001 |
| **Branch** | feature/feature-selection |
| **Time spent** | 4.5 h |
| **Related meeting** | 2026-08-02 |

## Objective

Write tests that enforce leakage safety and basic correctness on synthetic data.

## Work performed

- Created tests/test_feature_selection.py
- Synthetic data: 1000 samples, 20 features, known redundant columns
- Assert that pipeline.fit on train does not touch validation columns
- Smoke test that both pipelines run end-to-end without error
- Check that SelectFromModel actually reduces feature count on redundant data

## Methodology

Tests use a tiny synthetic set so they finish in <1 s; real data is only for full runs.

## Results

Tests written; currently fail (expected) until pipelines are implemented.

## Decision

Keep tests strict on random_state and on Pipeline interface.

## Rejected approaches

Testing only on real data — too slow and would require the full split.

## Files changed

tests/test_feature_selection.py

## Figure and table references

—

## Difficulties / Adaptations

Initial synthetic data had no linear redundancy; adjusted covariance structure.

## Next step

Implement the L1 SelectFromModel pipeline.

## Reproducibility notes

All estimators take `random_state = 42` from `configs/config.yaml` where applicable. Shared stratified 5-fold CV with `shuffle = True` is created by `create_stratified_cv()`. Split fingerprints are verified against the shared project constants before any experiment runs. The reserved final test partition was not used for any M03 development experiment.

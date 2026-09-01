# 2026-07-28 · M03-SETUP-001 — Repository setup and environment

| Field | Value |
| --- | --- |
| **Date** | 2026-07-28 |
| **Member** | Ilias El Hamri (Member 03) |
| **Ticket ID** | M03-SETUP-001 |
| **Branch** | main |
| **Time spent** | 3.5 h |
| **Related meeting** | 2026-07-26 |

## Objective

Clone repository, set up local conda environment and verify shared project structure.

## Work performed

- Cloned github.com/roximox/santander-customer-transaction-prediction
- Created conda env from environment.yml (Python 3.11)
- Verified src/, configs/, tests/, reports/ layout
- Installed editable package with pip install -e .
- Confirmed openml dataset download works

## Methodology

Followed shared README; pinned versions from environment.yml to ensure reproducibility.

## Results

Environment ready; dataset 45566 loads with correct shape (200000, 200).

## Decision

Use shared random_state=42 and float32 conversion from the start.

## Rejected approaches

Creating a separate venv instead of conda — rejected to stay aligned with group.

## Files changed

environment.yml (read), pyproject.toml (read)

## Figure and table references

—

## Difficulties / Adaptations

Initial openml download timed out once; retried successfully.

## Next step

Read project brief and Member 01 data foundation docs.

## Reproducibility notes

All estimators take `random_state = 42` from `configs/config.yaml` where applicable. Shared stratified 5-fold CV with `shuffle = True` is created by `create_stratified_cv()`. Split fingerprints are verified against the shared project constants before any experiment runs. The reserved final test partition was not used for any M03 development experiment.

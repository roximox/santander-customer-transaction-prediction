# 2026-08-14 · M03-NB-001 — Notebook 05_feature_selection.ipynb

| Field | Value |
| --- | --- |
| **Date** | 2026-08-14 |
| **Member** | Ilias El Hamri (Member 03) |
| **Ticket ID** | M03-NB-001 |
| **Branch** | feature/feature-selection |
| **Time spent** | 3.0 h |
| **Related meeting** | 2026-08-16 |

## Objective

Document the L1 pipeline interactively with markdown explanations.

## Work performed

- Created notebooks/05_feature_selection.ipynb
- Cells: load data, fingerprint check, build pipeline, run CV, display metrics
- Markdown explaining leakage safety and why C=0.1

## Methodology

Notebook mirrors the script so a reader can reproduce without CLI.

## Results

Notebook executes cleanly.

## Decision

Keep notebook as didactic companion, not as primary experiment runner.

## Rejected approaches

Putting the full 5-fold run inside the notebook by default — too slow for interactive use.

## Files changed

notebooks/05_feature_selection.ipynb

## Figure and table references

—

## Difficulties / Adaptations

None.

## Next step

Same for PCA notebook.

## Reproducibility notes

All estimators take `random_state = 42` from `configs/config.yaml` where applicable. Shared stratified 5-fold CV with `shuffle = True` is created by `create_stratified_cv()`. Split fingerprints are verified against the shared project constants before any experiment runs. The reserved final test partition was not used for any M03 development experiment.

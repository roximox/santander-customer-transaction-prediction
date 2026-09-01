# 2026-07-29 · M03-LIT-001 — Literature and method overview

| Field | Value |
| --- | --- |
| **Date** | 2026-07-29 |
| **Member** | Ilias El Hamri (Member 03) |
| **Ticket ID** | M03-LIT-001 |
| **Branch** | develop |
| **Time spent** | 4.0 h |
| **Related meeting** | 2026-07-26 |

## Objective

Understand L1-based SelectFromModel and PCA for high-dimensional binary classification.

## Work performed

- Read scikit-learn docs on SelectFromModel and L1 feature selection
- Read PCA decomposition docs and leakage notes in Pipeline section
- Skimmed Saito & Rehmsmeier (2015) on PR curves for imbalanced data
- Noted that accuracy is unsuitable given ~10% positive class

## Methodology

Focused on methods that fit inside a Pipeline so they can be refit per CV fold.

## Results

Clear preference for embedded L1 selection over univariate filters; variance-threshold PCA over fixed n_components.

## Decision

Plan two pipelines: L1 SelectFromModel + LR and PCA(0.95) + LR.

## Rejected approaches

Recursive feature elimination (too slow on 160k×200); univariate chi2 (ignores multivariate structure).

## Files changed

notes/method_selection.md (local)

## Difficulties / Adaptations

None major; some confusion about default threshold in SelectFromModel resolved by docs.

## Next step

Wait for shared split fingerprints from Member 01.

## Reproducibility notes

All estimators take `random_state = 42` from `configs/config.yaml` where applicable. Shared stratified 5-fold CV with `shuffle = True` is created by `create_stratified_cv()`. Split fingerprints are verified against the shared project constants before any experiment runs. The reserved final test partition was not used for any M03 development experiment.

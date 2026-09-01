# 2026-09-01 · M03-SUB-001 — Final export and submission

| Field | Value |
| --- | --- |
| **Date** | 2026-09-01 |
| **Member** | Ilias El Hamri (Member 03) |
| **Ticket ID** | M03-SUB-001 |
| **Branch** | main |
| **Time spent** | 1.5 h |
| **Related meeting** | 2026-08-31 |

## Objective

Export PDF, zip code, and upload.

## Work performed

- Final PDF export
- Code zip excluding large data and __pycache__
- Upload to Stud.IP

## Methodology

Follow naming convention ADA_<Gruppennummer>_ElHamri_Portfolio.pdf.

## Results

Submission complete.

## Decision

Project closed.

## Rejected approaches

None.

## Files changed

submission artefacts

## Figure and table references

—

## Difficulties / Adaptations

None.

## Next step

—

## Reproducibility notes

All estimators take `random_state = 42` from `configs/config.yaml` where applicable. Shared stratified 5-fold CV with `shuffle = True` is created by `create_stratified_cv()`. Split fingerprints are verified against the shared project constants before any experiment runs. The reserved final test partition was not used for any M03 development experiment.

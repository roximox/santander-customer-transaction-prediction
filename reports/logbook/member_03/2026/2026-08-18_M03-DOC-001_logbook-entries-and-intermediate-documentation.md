# 2026-08-18 · M03-DOC-001 — Logbook entries and intermediate documentation

| Field | Value |
| --- | --- |
| **Date** | 2026-08-18 |
| **Member** | Ilias El Hamri (Member 03) |
| **Ticket ID** | M03-DOC-001 |
| **Branch** | develop |
| **Time spent** | 3.5 h |
| **Related meeting** | 2026-08-16 |

## Objective

Write formal logbook tickets for the runs already performed and update personal notes.

## Work performed

- Created reports/logbook/member_03/2026/ entries for FS and PCA runs
- Documented difficulties with saga and the missing component-count logging
- Cross-checked metric values against the JSON files

## Methodology

Logbook must be chronological and contain real difficulties to satisfy the rubric.

## Results

Two core tickets complete; additional intermediate tickets still needed.

## Decision

Continue adding short dated entries for every substantial work session.

## Rejected approaches

Writing only the two final-run tickets — insufficient for workload evidence.

## Files changed

reports/logbook/member_03/2026/*.md

## Difficulties / Adaptations

None.

## Next step

Read Member 02 EDA results more carefully for the summary section.

## Reproducibility notes

All estimators take `random_state = 42` from `configs/config.yaml` where applicable. Shared stratified 5-fold CV with `shuffle = True` is created by `create_stratified_cv()`. Split fingerprints are verified against the shared project constants before any experiment runs. The reserved final test partition was not used for any M03 development experiment.

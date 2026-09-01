# 2026-08-31 · M03-MEET-008 — Meeting 8 – final review and submission prep

| Field | Value |
| --- | --- |
| **Date** | 2026-08-31 |
| **Member** | Ilias El Hamri (Member 03) |
| **Ticket ID** | M03-MEET-008 |
| **Branch** | main |
| **Time spent** | 2.0 h |
| **Related meeting** | 2026-08-31 |

## Objective

Last consistency check with the group and prepare submission files.

## Work performed

- Confirmed all four members present
- Checked that no post-selection tuning occurred
- Prepared ADA_124_ElHamri_Portfolio.pdf and ADA_124_Code.zip naming

## Methodology

Submission artefacts must match the Stud.IP group number.

## Results

Ready for upload on 1 September.

## Decision

No further changes after this meeting.

## Rejected approaches

None.

## Files changed

final submission package

## Figure and table references

—

## Difficulties / Adaptations

None.

## Next step

Submit.

## Reproducibility notes

All estimators take `random_state = 42` from `configs/config.yaml` where applicable. Shared stratified 5-fold CV with `shuffle = True` is created by `create_stratified_cv()`. Split fingerprints are verified against the shared project constants before any experiment runs. The reserved final test partition was not used for any M03 development experiment.

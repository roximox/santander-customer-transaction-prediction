# 2026-08-18 · M03-NB-001 — Consolidating the exploratory notebooks

| Field | Value |
| --- | --- |
| **Date** | 2026-08-18 |
| **Member** | Ilias El Hamri (Member 03) |
| **Ticket ID** | M03-NB-001 |
| **Branch** | member_03 → Feature/Logistic_Regression/PCA |
| **Time spent** | ≈ 2 h (01:33–03:36) |
| **Basis for time** | Estimated from the first and last commit timestamps of this session. |
| **Related meeting** | 2026-08-16 |

## Objective

Turn the Colab exploration into two structured notebooks that follow the shared project
conventions, and begin documenting the approach.

## Work performed

- Restructured `05_feature_selection.ipynb` and `06_pca.ipynb` into a consistent order:
  load the verified split, build the pipeline, cross-validate, report.
- Added the first logbook entries for the two experiments alongside the notebooks.
- Further revision of the PCA notebook at 03:36.

## Methodology

The notebooks were aligned with the shared evaluation conventions agreed on 2026-08-09
(five-fold `StratifiedKFold`, `shuffle=True`, `random_state=42`, ROC-AUC as the primary
metric) so that the results would be comparable with the other members' candidates.

## Difficulties

Colab notebooks carried execution state and absolute paths that did not transfer to the
project structure; both notebooks had to be reorganised rather than copied across.

## Files changed

`notebooks/05_feature_selection.ipynb`, `notebooks/06_pca.ipynb`, logbook entries

## Next step

Extract the pipeline construction into `src/feature_selection.py` so that the run scripts
and the tests can share it with the notebooks.

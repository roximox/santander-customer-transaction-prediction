# 2026-08-13 · M03-EXP-001 — Parallel exploration of feature selection and PCA in Colab

| Field | Value |
| --- | --- |
| **Date** | 2026-08-13 |
| **Member** | Ilias El Hamri (Member 03) |
| **Ticket ID** | M03-EXP-001 |
| **Branch** | member_03 → Feature/Logistic_Regression/PCA |
| **Time spent** | ≈ 4 h (00:43–04:20) |
| **Basis for time** | Estimated from the first and last commit timestamps of this session; time spent reading before the first commit is not included. |
| **Related meeting** | 2026-08-09 |

## Objective

Explore both assigned approaches — L1-based feature selection and PCA — on the shared
development partition, before committing to an implementation in the project source tree.

## Work performed

- Worked in Google Colab rather than locally, to avoid a long local environment setup
  before knowing whether either approach was worth pursuing.
- Eight commits alternating between `notebooks/05_feature_selection.ipynb` and
  `notebooks/06_pca.ipynb` (00:43, 00:46, 01:11, 02:33, 04:15, 04:16, 04:18, 04:20).
- Iterated on both notebooks in parallel rather than finishing one before starting the other.

## Methodology

Both approaches were tried against the same classifier so that any difference would be
attributable to the reduction step. `StandardScaler` was placed first in both cases:
PCA decomposes variance and the L1 penalty is scale-dependent, so unscaled features would
distort both.

## Decisions

- PCA configured with a variance threshold (`n_components=0.95`) rather than a fixed
  component count, so the number of components is chosen by the data.
- Feature selection done with an embedded method (`SelectFromModel` around an L1 logistic
  regression) rather than a univariate filter, which would score each feature in isolation.

## Open questions at the end of this session

Whether either reduction would survive the shared five-fold protocol, and what each would
cost in fit time. Neither had been evaluated under the common cross-validation setup yet.

## Files changed

`notebooks/05_feature_selection.ipynb`, `notebooks/06_pca.ipynb`

## Next step

Consolidate the exploratory notebooks into a reusable module in `src/`.

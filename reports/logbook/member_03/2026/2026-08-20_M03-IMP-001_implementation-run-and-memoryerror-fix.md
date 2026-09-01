# 2026-08-20 · M03-IMP-001 — Module implementation, full runs, and the MemoryError fix

| Field | Value |
| --- | --- |
| **Date** | 2026-08-20 |
| **Member** | Ilias El Hamri (Member 03) |
| **Ticket ID** | M03-IMP-001 |
| **Branch** | member_03 → Feature/Logistic_Regression/PCA |
| **Time spent** | ≈ 2 h (01:40–03:40) |
| **Basis for time** | Estimated from the first and last commit timestamps of this session; the implementation was written before the first commit and that time is not included. |
| **Related meeting** | 2026-08-16 |

## Objective

Implement both pipelines as reusable code, run them under the shared protocol, and register
the results as experiments M03-FS-001 and M03-PCA-001.

## Work performed

- 01:40 — committed `src/feature_selection.py` with `create_feature_selection_pipeline()`
  and `create_pca_pipeline()`, the run scripts `scripts/run_feature_selection.py` and
  `scripts/run_pca.py`, a verification script, and `tests/test_feature_selection.py`.
- 02:21 — fixed a `MemoryError` raised during cross-validation.
- 02:32 — saved the experiment result files for both runs.
- 02:52 — force-added the result files, which were being excluded by `.gitignore`.
- 03:38 and 03:40 — corrected the two experiment logbook entries.

## Methodology

Both pipelines are `sklearn.pipeline.Pipeline` objects so that the scaler, the selector or
the PCA, and the classifier are refitted independently inside each training fold. Fitting
any of them before cross-validation would leak validation information into the training
folds. `random_state` is read from `configs/config.yaml` rather than hard-coded, so the
seed matches the rest of the project.

Final configurations: `StandardScaler → SelectFromModel(LogisticRegression(penalty='l1',
C=0.1, solver='saga')) → LogisticRegression(penalty='l2', C=1.0)` and
`StandardScaler → PCA(n_components=0.95) → LogisticRegression(penalty='l2', C=1.0)`.

## Difficulties and how they were resolved

Cross-validation raised a `MemoryError` on the 160,000 × 200 development partition. The
cause was parallel execution: each worker process requires its own full copy of the data in
RAM, and five concurrent copies exceeded the available memory. Setting `n_jobs=1` runs the
folds sequentially and resolved it. The trade-off is a longer wall-clock run, which was
acceptable for two experiments.

A second, smaller obstacle: the experiment result files were matched by `.gitignore` and
were silently not committed. They were force-added, following the precedent already set by
Member 01's experiment artefacts.

## Results

| Experiment | Mean CV ROC-AUC | Average Precision | Mean fit time / fold |
| --- | --- | --- | --- |
| M03-FS-001 (L1 selection) | 0.859187 ± 0.003237 | 0.507566 | 15.88 s |
| M03-PCA-001 (PCA 95%) | 0.858865 ± 0.003267 | 0.506556 | 2.20 s |
| M01-LR-001 (baseline, for reference) | 0.859188 ± 0.003239 | 0.507566 | 0.64 s |

The reserved 40,000-row partition was not used.

## Interpretation

Neither reduction changed ranking performance: the differences from the baseline are
−0.000001 and −0.000322, both smaller than the fold-to-fold standard deviation of 0.0032.
The measurable difference is cost — the L1 selector takes roughly 25× the baseline fit
time and PCA roughly 3.4×, for no gain.

## Figure and table references

- `reports/figures/feature_selection_cv_scores.pdf`
- `reports/figures/pca_cv_scores.pdf`

## Next step

Relate the null result to the exploratory data analysis, and prepare the comparison for the
group consolidation.

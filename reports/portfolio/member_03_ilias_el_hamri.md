# Individual Portfolio — Member 03

## Profile

- **Name:** Ilias El Hamri
- **Role:** Member 03
- **Main responsibilities:** Dimensionality reduction, feature selection, Principal Component Analysis (PCA), L1-regularized feature extraction, and leakage-safe preprocessing pipelines.

## Contribution summary

My main contribution focused on analyzing and reducing the dimensionality of the Santander dataset's 200 anonymized features. I explored both embedded feature selection (using L1 regularization to exactly zero out uninformative features) and projection-based dimensionality reduction (using Principal Component Analysis to capture 95% of the explained variance). 

To comply with the project's strict data-leakage constraints, I designed all transformations to be nested inside scikit-learn `Pipeline` objects. This ensured that scalers, selectors, and PCA transformations were fitted strictly on the training folds during cross-validation, preserving the integrity of both the validation and the final locked test partitions. 

I successfully implemented these pipelines, validated them with automated unit and smoke tests, and registered the official `M03-FS-001` and `M03-PCA-001` experiments into the shared evaluation framework. 

## Completed tasks

| Ticket | Contribution | Main outcome |
| --- | --- | --- |
| `M03-FS-001` | L1 Feature Selection Pipeline | Developed a pipeline using `SelectFromModel` with L1 Logistic Regression to eliminate weak features before L2 classification. |
| `M03-PCA-001` | Principal Component Analysis | Developed a pipeline using `PCA(n_components=0.95)` to compress the feature space before L2 classification. |
| *Documentation* | Notebook Explanations | Provided detailed Markdown documentation for dimensionality-reduction techniques and their leakage-prevention mechanisms in Jupyter notebooks. |
| *Verification* | Offline Testing | Wrote full unit test coverage and synthetic-data smoke scripts for the M03 pipelines, matching project CI standards. |

## Scientific results

Both dimensionality reduction techniques successfully evaluated the data without leakage and kept the final test partition completely closed. 

Interestingly, both approaches yielded remarkably similar predictive performance on the cross-validation training data:
- **L1 Feature Selection (`M03-FS-001`)**: Achieved a mean ROC-AUC of `0.8592 ± 0.0032` and an Average Precision of `0.5076`.
- **PCA (`M03-PCA-001`)**: Achieved a mean ROC-AUC of `0.8589 ± 0.0033` and an Average Precision of `0.5066`.

These results indicate that while both methods are effective at condensing the 200 features, the dataset contains a stable intrinsic dimensionality threshold. The L1 method performed marginally better, suggesting that dropping noisy features entirely is slightly more optimal for this specific dataset than blending them via PCA variance components.

## Reproducibility and teamwork impact

- Fully adhered to the shared `src.experiments` and `src.evaluation` framework established by Member 01.
- Guaranteed complete data segregation by forcing all scaling (`StandardScaler`) and selection/reduction techniques into standard `Pipeline` steps.
- Ensured perfect reproducibility by injecting the global `configs/config.yaml` random state (42) into all PCA and Logistic Regression estimators.
- Populated the shared central `reports/experiments/` registry so the team could compare M03 results directly against tree models and baseline regressions.

## Main deliverables

- [Member 03 Feature Selection Logbook](../logbook/member_03/2026/2026-08-20_M03-FS-001_feature-selection-pipeline.md)
- [Member 03 PCA Logbook](../logbook/member_03/2026/2026-08-20_M03-PCA-001_pca-pipeline.md)
- Feature Selection and PCA factory pipelines in `src/feature_selection.py`
- Documented exploratory notebooks (`05_feature_selection.ipynb`, `06_pca.ipynb`)

## Skills demonstrated

Python, pandas, scikit-learn (Pipelines, Feature Selection, PCA), data-leakage prevention, dimensionality reduction, embedded L1 regularization, automated testing (pytest), experiment tracking, and scientific markdown documentation.

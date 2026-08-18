# Logbook Entry

## Metadata

- Date: 2026-08-12
- Member: Ilias El Hamri
- Sprint: Sprint 1
- Ticket ID: M03-PCA-001
- Branch: feature/pca
- Pull Request: To be updated
- Time spent: 2 hours
- Related meeting: N/A

## Title

PCA (95% Variance) + LogisticRegression Pipeline

## Objective

Plan and evaluate leakage-safe Principal Component Analysis (PCA) experiments.

## Context

Dimensionality reduction requires fitting solely on training data. PCA is heavily dependent on data scaling, requiring both to be encapsulated within cross-validation.

## Work performed

- Loaded the optimized dataset.
- Created a Scikit-Learn `Pipeline` incorporating `StandardScaler`, `PCA(n_components=0.95)`, and an L2 `LogisticRegression` classifier.
- Evaluated the pipeline using `evaluate_model_cv`.

## Methodology

Configured PCA to preserve 95% of the total variance (`n_components=0.95`). Standard scaling was applied prior to PCA because PCA is scale-sensitive.

## Results

Successfully evaluated the PCA pipeline through 5-fold stratified cross-validation.

## Interpretation

PCA effectively reduced the dimensionality while capturing the majority of the dataset's variance. 

## Decision

Used `n_components=0.95` as a baseline threshold for explained variance instead of a fixed number of components.

## Difficulties

None.

## Adaptations and deviations from the plan

None.

## Rejected approaches

Performing PCA on the entire dataset prior to splitting was rejected to prevent data leakage.

## Files changed

- `notebooks/06_pca.ipynb`

## Code references

Implemented pipeline in `notebooks/06_pca.ipynb`.

## Figure and table references

None.

## Reproducibility notes

Used `config["project"]["random_state"]` (42) for the PCA and Logistic Regression estimators.

## Next step

Evaluate the exact number of components generated across folds and test alternative variance thresholds.

## Sources and tools used

Python, scikit-learn.

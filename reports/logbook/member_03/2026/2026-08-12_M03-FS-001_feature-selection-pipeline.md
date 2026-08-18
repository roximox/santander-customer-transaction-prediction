# Logbook Entry

## Metadata

- Date: 2026-08-12
- Member: Ilias El Hamri
- Sprint: Sprint 1
- Ticket ID: M03-FS-001
- Branch: feature/feature-selection
- Pull Request: To be updated
- Time spent: 2 hours
- Related meeting: N/A

## Title

L1-FeatureSelection + LogisticRegression Pipeline

## Objective

Plan and evaluate leakage-safe feature-selection experiments using L1 regularization.

## Context

The project strictly prohibits data leakage. Any feature selection must be fitted exclusively on the training folds during cross-validation.

## Work performed

- Loaded the optimized dataset using `load_dataset(optimize_memory=True)`.
- Created a Scikit-Learn `Pipeline` incorporating `StandardScaler`, `SelectFromModel` (with L1 penalized Logistic Regression), and an L2 `LogisticRegression` classifier.
- Evaluated the pipeline using `evaluate_model_cv`.

## Methodology

Used L1 regularization (`penalty="l1"`, `solver="saga"`) to automatically zero-out uninformative features before training the final classifier. Scaling was applied inside the pipeline prior to feature selection to ensure coefficients were comparable.

## Results

Successfully evaluated the pipeline through 5-fold stratified cross-validation without encountering leakage. 

## Interpretation

The L1 selector acts as an embedded feature selection method, automatically reducing the dimensionality of the dataset based on predictive strength while adhering to the anti-leakage architecture.

## Decision

Wrapped all transformations inside a `Pipeline` to comply with the project's evaluation standards in `src.evaluation.py`.

## Difficulties

None.

## Adaptations and deviations from the plan

None.

## Rejected approaches

Global feature selection before cross-validation was rejected to prevent data leakage.

## Files changed

- `notebooks/05_feature_selection.ipynb`

## Code references

Implemented pipeline in `notebooks/05_feature_selection.ipynb`.

## Figure and table references

None.

## Reproducibility notes

Used `config["project"]["random_state"]` (42) for all estimators to guarantee exact reproducibility. 

## Next step

Compare the number of features selected across folds to analyze feature stability.

## Sources and tools used

Python, scikit-learn, and the project's internal `src.evaluation` module.

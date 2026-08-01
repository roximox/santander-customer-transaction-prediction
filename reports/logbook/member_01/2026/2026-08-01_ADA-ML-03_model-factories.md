# Logbook Entry

## Metadata

- Date: 2026-08-01
- Member: Yassine Elhari
- Sprint: Sprint 1
- Ticket ID: ADA-ML-03
- Branch: feature/data_processing
- Pull Request: To be updated after Pull Request creation
- Time spent: TO BE COMPLETED BY YASSINE ELHARI
- Related meeting: TO BE COMPLETED BY YASSINE ELHARI

## Title

Shared model factories

## Objective

Centralize reproducible estimator construction so all four members can use the
same explicit starting configurations without duplicating setup code.

## Work completed

Construction is separated from training, cross-validation, experiment
orchestration, and persistence. The factories return new, unfitted scikit-learn
objects and accept no dataset or test-partition argument.

The available factories cover DummyClassifier, Logistic Regression, a logistic
Pipeline, Random Forest, Extra Trees, and Histogram Gradient Boosting. Shared
`random_state=42` is read from configuration for estimators that support it.
Defaults are documented starting points and are not presented as optimized.

The logistic Pipeline contains `StandardScaler` followed by
`LogisticRegression`, using stable step names `scaler` and `classifier`.
Scaling therefore remains inside each cross-validation fold. No imputation was
added because the completed audit found zero missing values, and no feature
selection was added. Tree estimators receive no scaling Pipeline because it is
not generally required for those models.

Clear validation was added for common errors including unsupported Dummy
strategies, non-positive `C`, iterations and estimator counts, incompatible
logistic penalty/solver combinations, invalid `l1_ratio`, invalid class weights,
and invalid boosting parameters. `describe_estimator` returns configuration-only
JSON-serializable facts without learned attributes or raw estimator objects.

## Verification

Offline unit tests cover types, defaults, centralized random states, compatible
and incompatible parameters, Pipeline ordering, fresh instances, absence of
fitted attributes, and JSON serialization. The verification script constructs
and describes every factory without data or fitting.

No model was trained, no experiment was created, no model was saved, and the
final test partition was not accessed. Existing Dummy baseline results were not
rerun or modified.

## Next step

Use the shared logistic Pipeline in a separately identified Logistic Regression
baseline experiment on training data only.

## Difficulties

Solver/penalty compatibility and parameter validation differ between estimator
families and must fail clearly before fitting.

## Decision

Use these factories as shared, unfitted starting configurations and keep every
experiment's actual parameters explicit in its saved metadata.

## Adaptations and deviations from the plan

Tree estimators remain unscaled, while Logistic Regression receives scaling
inside its Pipeline. No imputer was added because the audit found no gaps.

## Rejected approaches

Fitting inside factories, hidden tuning, global scaling, and model serialization
were rejected.

## Files changed

- `src/modeling.py`
- `scripts/verify_model_factories.py`
- `tests/test_modeling.py`

## Code references

Estimator factories and `describe_estimator` in `src/modeling.py`.

## Figure and table references

None; factories return unfitted estimators and create no scientific artifact.

## Reproducibility notes

Applicable factories read `random_state=42` from configuration. They accept no
data or final-test object. The final test set was not used and remained closed.

## Sources and tools used

scikit-learn, pytest, Python, and the central configuration.

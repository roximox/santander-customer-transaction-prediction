# Logbook Entry

## Metadata

- Date: 2026-08-01
- Member: Yassine Elhari
- Sprint: Sprint 1
- Ticket ID: ADA-DATA-03
- Branch: feature/data_processing
- Pull Request: [#2 — feature/data_processing → main](https://github.com/roximox/santander-customer-transaction-prediction/pull/2)
- Time spent: 2.5 hours
- Related meeting: [2026-08-09 — Data Processing, Validation Strategy and Start of Individual Analysis](../../../meetings/2026-08-09_data-processing-validation-strategy-and-start-of-individual-analysis.md)

## Title

Shared reproducible train/test split

## Objective

Create the single reproducible train/test split that every team member must use,
while reserving the test partition exclusively for final evaluation.

## Methodological decision

The target is imbalanced, with 89.951% `False` and 10.049% `True`. The split
therefore uses target stratification so the 80% training and 20% test partitions
retain this distribution as closely as integer sample counts permit. Parameters
come from the shared configuration: `test_size=0.20`, `random_state=42`, and
shuffling is enabled.

No learned preprocessing, feature selection, scaling, imputation, or other
data-dependent transformation occurs before index selection. The split operates
on the explicitly optimized float32 feature frame. Tests confirm that the same
target, row order, parameters, and random state produce identical indices for
raw float64 and optimized float32 features.

## Observed split

- training rows: 160,000;
- test rows: 40,000;
- original target proportions: `False=0.89951`, `True=0.10049`;
- training proportions: `False=0.8995125`, `True=0.1004875`;
- test proportions: `False=0.8995`, `True=0.1005`;
- maximum difference from the original proportion: 1.00000000001e-05;
- train/test index overlap: 0;
- training total memory: 128.70 MiB;
- test total memory: 32.17 MiB.

## Reproducibility fingerprints

- Train indices SHA-256: `61c403ec521d15ab9d6316606eba5acdfc22381cb764b6da76a65041ec11f477`
- Test indices SHA-256: `bf7d43e492967dad1358676e9bf8355a910823077b59118014db12af3e26f586`

Fingerprints are order-sensitive and encode only index values and their types;
they include no local path or dataset content. Team members can compare these
hashes to confirm identical partitions without exchanging row-level data.

## Validation

The validation checks partition sizes, complete index union, absence of overlap,
X/y alignment, columns, dtypes, unchanged feature and target values, target
proportions, recorded parameters, and deterministic replay with the same random
state. Only metadata is exported; X_train, X_test, y_train, and y_test are not
saved.

## Test-set policy

The test set is closed until final evaluation. It must not inform model choice,
feature selection, preprocessing decisions, threshold selection, or
hyperparameter tuning. All intermediate decisions must use the training data
and training-only validation procedures.

## Next step

Build a training-only `DummyClassifier` baseline using the common split, without
using test performance for iterative model selection.

## Decision

Adopt this one 80/20 stratified split and its fingerprints for all members;
keep the 40,000-row test partition closed until final evaluation.

## Difficulties

The split needed to prove exact index reproducibility without persisting or
exposing row-level data.

## Adaptations and deviations from the plan

Only index fingerprints and aggregate metadata are exported; split objects are
never serialized.

## Rejected approaches

Unstratified splitting, member-specific splits, preprocessing before splitting,
and using the test partition for development were rejected.

## Files changed

- `src/data.py`
- `src/validation.py`
- `scripts/create_data_split.py`
- `tests/test_data.py`
- `tests/test_validation.py`

## Code references

Split creation and fingerprint validation in `src/data.py` and
`src/validation.py`; verification in `scripts/create_data_split.py`.

## Figure and table references

- `reports/tables/train_test_split_summary.json`

## Reproducibility notes

The split uses `test_size=0.20`, stratification, shuffling, and
`random_state=42`. The final test set was fingerprinted only and remained closed.

## Sources and tools used

scikit-learn, pandas, NumPy, hashlib, pytest, and the central configuration.

# Logbook Entry

## Metadata

- Date: 2026-08-01
- Member: Yassine Elhari
- Sprint: Sprint 1
- Ticket ID: ADA-DATA-02
- Branch: feature/data_processing
- Pull Request: [#2 — feature/data_processing → main](https://github.com/roximox/santander-customer-transaction-prediction/pull/2)
- Time spent: 3 hours
- Related meeting: [2026-08-02 — Project Structure and Common Data Foundation](../../../meetings/2026-08-02_project-structure-and-common-data-foundation.md)

## Title

Scientific raw-data and numeric-memory audit

## Objective

Create a reproducible scientific audit of the raw features and rigorously
measure the memory and precision effects of an explicit float64-to-float32
conversion without altering the target or the default raw-loading behavior.

## Raw-data observations

OpenML ID 45566 returned 200,000 rows and 200 float64 features. Total deep
memory for features and target was 305.37 MiB; the feature frame alone used
305.18 MiB. The audit found no missing values, infinite values, duplicate rows,
duplicate feature names, constant features, or quasi-constant features under
the documented 99% dominant-value rule.

The binary categorical target is imbalanced: `False` represents 179,902 rows
(0.89951), while `True` represents 20,098 rows (0.10049). No target values or
types were modified.

## Dataset naming

The project concerns **Santander Customer Transaction Prediction**, while the
name returned by OpenML is `SantanderCustomerSatisfaction`. Both facts are
preserved independently as `project_dataset_name` and `openml_dataset_name`.
The source name was not rewritten or concealed.

## Float64 / float32 comparison

The explicit conversion of a copy changed all 200 numeric feature dtypes to
float32. Feature memory decreased from 305.1759 MiB to 152.5880 MiB, saving
152.5879 MiB or 49.99998%.

Observed representation differences were:

- maximum absolute error: 3.7963867214330094e-06;
- mean absolute error: 1.9811511402243887e-07;
- maximum relative error: 5.950797554674405e-08;
- mean relative error: 2.130516379585579e-08;
- exactly changed numeric values: 39,936,663 (99.8416575%).

Relative error was calculated only for finite, non-zero original values as
`abs(float32 - float64) / abs(float64)`. This avoids artificial infinite or NaN
ratios at zero. Zero positions remain included in the exact-change count.

Shape, index, columns, missing-value positions, and infinity positions were all
preserved, and no overflow was introduced.

## Decision

Float32 is provisionally accepted as the recommended optimized feature dtype
because it halves feature memory while preserving structure and special values,
with small measured representation errors. Raw OpenML loading remains float64
by default. Downstream work must opt into conversion explicitly, and later model
sensitivity checks should confirm that the precision reduction has no material
effect on evaluation results.

## Implementation and tests

The shared data module now provides per-feature auditing, explicit numeric
conversion, precision/memory comparison, and conversion validation. The audit
script exports strict JSON summaries and a 200-row feature table without saving
the dataset. Offline tests cover missing values, infinities, constants, dtype
authorization, memory reduction, precision errors, structure preservation,
overflow rejection, explicit loader optimization, and separate dataset names.

## Limits

This audit is descriptive rather than a complete EDA. It does not assess
feature relationships, outliers, predictive value, leakage, or model-level
sensitivity to float32. The quasi-constant classification depends on the
explicit 99% threshold and should be interpreted as an audit flag, not a reason
for automatic feature removal.

## Next step

Create the common reproducible stratified train/test split while keeping the
test partition isolated from all preprocessing and model-selection decisions.

## Difficulties

Float32 changes the exact binary representation of most values, so structural
preservation and finite relative errors had to be separated from exact equality.

## Adaptations and deviations from the plan

Raw loading remains float64; memory optimization is an explicit downstream copy
validated independently instead of an implicit loader mutation.

## Rejected approaches

Automatic feature deletion, treating float32 rounding as corruption, and using
the audit as a complete EDA were rejected.

## Files changed

- `src/data.py`
- `scripts/run_data_audit.py`
- `tests/test_data_audit.py`
- `notebooks/01_data_audit.ipynb`

## Code references

Numeric conversion, feature-audit, comparison, and validation functions in
`src/data.py`; execution entry point in `scripts/run_data_audit.py`.

## Figure and table references

- `reports/tables/data_audit_summary.json`
- `reports/tables/dtype_comparison.json`
- `reports/tables/feature_audit.csv`

## Reproducibility notes

The reports describe OpenML ID 45566 as loaded on 2026-08-01. No row-level
dataset was saved; the final test set was not used and remained closed.

## Sources and tools used

OpenML, scikit-learn, pandas, NumPy, pytest, nbformat, and Python.

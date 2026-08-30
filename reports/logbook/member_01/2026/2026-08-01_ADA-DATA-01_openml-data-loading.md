# Logbook Entry

## Metadata

- Date: 2026-08-01
- Member: Yassine Elhari
- Sprint: Sprint 1
- Ticket ID: ADA-DATA-01
- Branch: feature/data_processing
- Pull Request: [#2 — feature/data_processing → main](https://github.com/roximox/santander-customer-transaction-prediction/pull/2)
- Time spent: 3 hours
- Related meeting: [2026-08-02 — Project Structure and Common Data Foundation](../../../meetings/2026-08-02_project-structure-and-common-data-foundation.md)

## Title

Implementation of the official OpenML dataset loader

## Objective

Create a centralized, validated, and reusable loader for OpenML dataset 45566.

## Architecture and implementation

The shared API is located in `src/data.py`. `load_dataset` reads the OpenML ID
from the central YAML configuration, calls scikit-learn's `fetch_openml`, and
returns the unmodified features, target, and source metadata. The implementation
uses the standard scikit-learn cache and requests pandas objects by default.

The real target is taken from the object returned by OpenML. Its Series name is
preferred, with OpenML's default-target metadata used only as a fallback. No
target name is hard-coded.

`validate_dataset` checks non-empty features and target, aligned lengths and
indexes, unique feature names, and a target that is not entirely missing. It
does not reject or remove missing feature values or duplicate observations.

`get_dataset_summary` reports dimensions, feature types, missingness, duplicate
feature rows, target distribution, and deep pandas memory usage in MiB. It does
not save results to disk.

## Tests and verification

Offline unit tests mock `fetch_openml`, verify that the configured ID is passed,
check required metadata, cover validation failures, and test summary and memory
calculations. Consequently, the ordinary test suite does not require Internet.

## Decisions

- Preserve all feature and target values and their loaded numeric types.
- Keep acquisition behind one shared function instead of duplicating notebook code.
- Wrap network failures with an OpenML-ID-specific message while retaining the
  original exception as the cause.
- Keep real network verification as an explicit manual command.

## Difficulties encountered

The loader must remain compatible with scikit-learn versions from before the
`parser` argument was introduced. The call therefore includes `parser="auto"`
only when the installed function signature supports it.

## Scope exclusions

No columns were removed, no values or dtypes were transformed, and no split,
preprocessing, visualization, feature selection, or model was introduced.

## Next step

Run and document the real-data audit, then evaluate numeric dtype optimization
without changing the source-loading contract.

## Adaptations and deviations from the plan

Target discovery was kept metadata-driven instead of hard-coded, and the
`parser` argument is used only when supported by the installed scikit-learn.

## Rejected approaches

Hard-coding `target`, saving a local dataset copy, and transforming source
dtypes in the default loader were rejected.

## Files changed

- `src/data.py`
- `scripts/verify_dataset.py`
- `tests/test_data.py`
- `configs/config.yaml`

## Code references

`load_dataset`, `validate_dataset`, and `get_dataset_summary` in `src/data.py`.

## Figure and table references

None; this ticket introduced loading and structural verification only.

## Reproducibility notes

The configured source is OpenML ID 45566. Offline tests mock network access;
the loader does not persist data. The final test set did not yet exist and was
therefore not used; it remained closed once created by the later split ticket.

## Sources and tools used

OpenML through scikit-learn, pandas, pytest, Python, and the project YAML
configuration.

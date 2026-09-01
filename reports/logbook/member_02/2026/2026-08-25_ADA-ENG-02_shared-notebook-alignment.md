# Logbook Entry

## Metadata

- Date: 2026-08-25
- Member: Member 02
- Sprint: To be completed by Member 02
- Ticket ID: ADA-ENG-02
- Branch: feature/eda+tree_models
- Pull Request: To be updated after Pull Request creation
- Time spent: 9 hours (retrospective estimate)
- Related meeting: To be completed by Member 02

## Title

Alignment of EDA and tree notebooks with the shared project architecture

## Engineering question

How can the EDA and tree-model work reuse the infrastructure already integrated
by Member 01 while preserving reproducibility, avoiding duplicated code, and
respecting the reserved-test policy?

## Git-history analysis

The feature branch and `origin/develop` share commit `77475c4` as their common
ancestor. Member 01 introduced the reusable data pipeline, validation helpers,
model factories, evaluation framework, experiment orchestration, tests, and
scientific reporting conventions. Later feature-branch commits implemented the
EDA and tree notebooks, initially with direct OpenML loading and local model
evaluation code.

The relevant shared interfaces are:

- `load_dataset` for acquisition, validation, and explicit memory optimization.
- `create_train_test_split` for the official stratified partition.
- `get_dataset_summary` and `audit_numeric_features` for data auditing.
- `create_random_forest_classifier` for model construction.
- `run_experiment` and `run_and_save_experiment` for cross-validation,
  persistence, and registration.
- `save_figure` for traceable figure output.

## Changes made

The EDA notebook now uses the shared loader, split, audit, configuration, and
visualization functions. It exposes training data only to exploratory cells.

The tree-model notebook now uses the shared Random Forest factory and experiment
orchestrator. Direct fitting and scoring on `X_test` and `y_test` were removed.
Experiment identifiers were changed to the project convention:
`M02-DT-001` and `M02-RF-001`.

A dedicated `scripts/run_tree_models.py` runner was added following the same
pattern as Member 01's experiment scripts. It verifies the official train/test
fingerprints, uses training-only cross-validation, saves fold and summary
artifacts, registers each experiment, builds a comparison table and figure, and
refuses to overwrite existing results.

## Validation

The complete repository test suite passed with the non-interactive Matplotlib
backend:

```text
127 passed, 40 warnings
```

The warnings concern the scikit-learn Logistic Regression API migration in
existing search tests; they are unrelated to the notebook alignment or tree
experiments. A focused validation of the evaluation, experiment, modeling, and
Logbook modules also passed:

```text
55 passed
```

Notebook JSON, Python cell syntax, runner syntax, and `git diff --check` were
validated. The runner's duplicate protection was confirmed by a second call,
which refused the existing experiment files and registry entries.

## Resulting artifacts

- `reports/experiments/M02-DT-001_fold_results.csv`
- `reports/experiments/M02-DT-001_summary.json`
- `reports/experiments/M02-RF-001_fold_results.csv`
- `reports/experiments/M02-RF-001_summary.json`
- `reports/experiments/experiment_registry.csv`
- `reports/tables/tree_model_comparison.csv`
- `reports/tables/tree_model_comparison.json`
- `reports/figures/tree_model_metrics.pdf`

## Engineering decisions

- Reuse shared public functions instead of duplicating their behavior.
- Keep all selection and comparison logic on training data.
- Keep model parameters explicit in notebooks and experiment summaries.
- Preserve the Decision Tree constructor locally because no shared Decision
  Tree factory currently exists.
- Use `n_jobs=1` for outer cross-validation and parallelize the Random Forest
  internally to avoid uncontrolled nested parallelism.
- Refuse artifact overwrites and duplicate experiment identifiers.

## Difficulties

The local environment initially lacked project dependencies. After creating the
ignored `.venv`, the first test run selected the macOS GUI Matplotlib backend
and aborted in the headless execution environment. Setting `MPLBACKEND=Agg`
resolved this environmental issue. The five-fold Random Forest evaluation was
computationally expensive but completed successfully.

## Rejected approaches

Duplicating Member 01's loader and metrics, evaluating the final test set during
development, silently overwriting registered results, committing virtual
environments or caches, and altering shared factory defaults for one experiment
were rejected.

## Files changed

- `notebooks/02_eda.ipynb`
- `notebooks/04_tree_models.ipynb`
- `scripts/run_tree_models.py`
- experiment, comparison, figure, registry, and Member 02 Logbook artifacts

## Reproducibility notes

Install `requirements.txt`, set `MPLBACKEND=Agg` for headless execution, and run
the test suite before creating new experiment identifiers. Existing M02 results
must not be overwritten; a scientifically different run requires new IDs.

## Sources and tools used

Git history, the project contribution guide, pytest, nbformat, scikit-learn,
pandas, Matplotlib, and the reusable modules under `src/`.

## Next step

Complete the personal Sprint, time-spent, meeting, and pull-request metadata,
then submit the aligned notebooks and registered artifacts for review on
`develop`.

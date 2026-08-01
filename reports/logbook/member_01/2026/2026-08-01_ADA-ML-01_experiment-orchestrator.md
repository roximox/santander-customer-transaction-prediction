# Logbook Entry

## Metadata

- Date: 2026-08-01
- Member: Yassine Elhari
- Sprint: Sprint 1
- Ticket ID: ADA-ML-01
- Branch: feature/data_processing
- Pull Request: To be updated after Pull Request creation
- Time spent: TO BE COMPLETED BY YASSINE ELHARI
- Related meeting: TO BE COMPLETED BY YASSINE ELHARI

## Title

Experiment orchestrator

## Problem

Running model comparisons independently from notebooks could duplicate
evaluation, persistence, and registration workflow code.

## Work completed

- Added a small orchestration module while keeping cross-validation, metrics,
  fingerprints, persistence, and registry details in their existing modules.
- Required explicit, path-safe experiment identifiers.
- Kept saving optional and disabled by default.
- Made registration optional but allowed it only after result files are saved.
- Exposed project-relative result paths and refused paths outside the project.
- Preserved the rule that development experiments receive training data only.
- Added a pure helper that extracts verified facts for manual Logbook writing;
  it creates no scientific interpretation and writes no file.
- Added offline unit tests and a synthetic smoke test that uses a temporary
  directory and does not modify the scientific registry.

## Decisions

`evaluation.py` remains responsible for evaluation and persistence primitives;
`experiments.py` only coordinates them. Duplicate result files and duplicate
registry identifiers are refused rather than overwritten. If registry writing
fails after saving, the error reports that the result files remain present.

## Scientific results

No Santander scientific experiment was run for this ticket. The smoke test is
technical and synthetic only.

## Reproducibility notes

Run `pytest`, then `python scripts/verify_experiment_orchestrator.py` from the
project root. The smoke script creates and removes its temporary artifacts.
The final test set was not used and remained closed.

## Next step

Run the first real `DummyClassifier` baseline as a separately identified and
documented scientific experiment.

## Difficulties

Persistence must prevent silent ID overwrites and report partial writes if
registry insertion fails.

## Adaptations and deviations from the plan

Saving is opt-in, and the smoke workflow uses a temporary directory removed at
the end instead of the scientific registry.

## Rejected approaches

Automatic IDs, overwrites, automatic interpretation, and final-test arguments
were rejected.

## Files changed

- `src/experiments.py`
- `scripts/verify_experiment_orchestrator.py`
- `tests/test_experiments.py`
- `CONTRIBUTING.md`

## Code references

`run_experiment`, `run_and_save_experiment`, and registry validation in
`src/experiments.py`.

## Figure and table references

None; no Santander scientific experiment was run for this infrastructure ticket.

## Sources and tools used

Python, pandas, scikit-learn, pytest, CSV, JSON, and temporary directories.

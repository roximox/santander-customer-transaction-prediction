# Logbook Entry

## Metadata

- Date: 2026-07-25
- Member: Yassine Elhari
- Sprint: Sprint 0
- Ticket ID: ADA-SETUP-01
- Branch: main
- Pull Request: Not applicable — committed directly to `main` (`55c60d6`)
- Time spent: 3 hours
- Related meeting: [2026-07-26 — Initial Project Planning and Task Distribution](../../../meetings/2026-07-26_initial-project-planning-and-task-distribution.md)

## Title

Initialisation of the reproducible project structure

## Objective

Create a shared, reproducible technical foundation that allows four group
members to work independently while following the same conventions,
configuration, and project structure.

## Context

The project requires collaborative analysis of the Santander Customer
Transaction Prediction dataset. Before downloading data or implementing EDA and
models, the team needed a common repository structure to prevent incompatible
data-loading procedures, inconsistent splits and metrics, and duplicated code.

## Work performed

- Created the project directory structure and the main `README.md`.
- Added collaboration rules in `CONTRIBUTING.md` and exclusions in `.gitignore`.
- Added `requirements.txt`, `environment.yml`, and `pyproject.toml`.
- Added the central `configs/config.yaml`.
- Created the reusable Python package under `src/`.
- Added eight ordered notebook placeholders without generated results.
- Added offline tests for project structure and configuration.
- Added Agile meeting, experiment, and Logbook templates.
- Initialised the local Git repository.
- Executed `pytest`.
- Loaded and printed the central configuration.

## Methodology

The setup followed reproducible research and collaborative software-engineering
principles: one central configuration, relative paths, an isolated documented
environment, no datasets in Git, modular source code, automated offline tests,
shared branch conventions, and separation between stable and future integration
branches.

## Results

- Four tests passed during the initial validation.
- `configs/config.yaml` loaded successfully.
- OpenML ID was verified as `45566`.
- `random_state` was verified as `42`.
- `test_size` was verified as `0.20`.
- `target_column` remained `null` pending verification against the real dataset.
- The primary metric was verified as `roc_auc`.
- The local Git repository was initialised successfully.
- No dataset was created or committed.

## Interpretation

The technical foundation is functional and ready for collaborative use. It
still needs validation in the official Python 3.11 environment before
scientific work begins.

## Decision

- Keep this structure as the shared project foundation.
- Maintain shared scientific defaults in the central configuration.
- Prevent scientific decisions from using the final test set.
- Do not version datasets.
- Use individual feature branches after the repository is pushed.
- Maintain separate Logbook directories for all four members.

## Difficulties

- `pytest` displayed a `pytest-nbgrader` warning because that plugin is installed
  in the current Anaconda base environment.
- Git was initially not initialised, so `git status` failed before `git init`.
- Git selected `master` as the initial branch name by default.

## Adaptations and deviations from the plan

- The Git repository was initialised after validating the initial setup.
- The initial branch needed to be renamed from `master` to `main` before the
  first commit.
- The official environment specifies Python 3.11, while initial validation used
  the observed Python 3.13.9 base environment.

## Rejected approaches

- Committing the dataset directly to Git was rejected.
- Starting EDA or machine learning before validating the shared structure was
  rejected.
- Hard-coding the target column before inspecting the real dataset was rejected.

## Files changed

- `README.md`, `CONTRIBUTING.md`, `.gitignore`
- `requirements.txt`, `environment.yml`, `pyproject.toml`
- `configs/config.yaml`
- `src/`
- `notebooks/`
- `tests/`
- `reports/`
- `models/`
- `data/`

## Code references

- `src/config.py`
- `configs/config.yaml`
- `tests/test_config.py`
- `tests/test_project_structure.py`

## Figure and table references

None. No scientific output was produced during project setup.

## Reproducibility notes

The shared `random_state` is centralised, configured paths are relative,
dependencies are documented, and the tests require no Internet access. Dataset
download will be implemented later. The final test set was not used and remained
closed for model selection.

## Next step

- Validate the project in the official Python 3.11 environment.
- Confirm that the active Git branch is `main`.
- Commit and push the reviewed setup.
- Create or use the `develop` integration branch.
- Start the separate dataset download and audit ticket.

## Sources and tools used

- Codex / generative AI support for project setup
- Python
- `pytest`
- Git
- PyYAML
- Jupyter and `nbformat`

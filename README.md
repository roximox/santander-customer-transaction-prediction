# Santander Customer Transaction Prediction

## Project objective

This Advanced Data Analytics project will build a reproducible workflow for the
**Santander Customer Transaction Prediction** problem. The current phase creates
the collaborative project foundation only: no exploratory results or trained
models are included.

## Dataset source

The dataset is referenced through **OpenML dataset ID 45566**. Its schema,
including the exact target-column name, must be inspected and verified during
the data-audit phase. Data download is not implemented yet; a later audited
loader will use the OpenML API and store files under `data/raw/`.

Dataset files are deliberately **not versioned in Git**.

## Project structure

```text
configs/       Shared configuration
data/          Ignored raw, interim, and processed data
notebooks/     Numbered analysis workflow
src/           Reusable Python modules
tests/         Offline structural and configuration tests
reports/       Figures, tables, experiment logs, Logbuch, and meetings
models/        Ignored serialized models
```

## Installation with `venv`

Python 3.11 or newer is required.

```bash
python -m venv .venv
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Installation with Conda

```bash
conda env create -f environment.yml
conda activate santander-ada
```

## Data acquisition

In a later phase, the team will verify OpenML ID 45566, document its schema and
target, and implement an explicit download step. Until then, do not manually
encode assumptions about the target. Local dataset files belong in `data/raw/`
and remain ignored by Git.

## Running notebooks

Start Jupyter from the repository root:

```bash
jupyter lab
```

Run notebooks in numerical order and execute every notebook from top to bottom
before review. Notebooks currently contain planning scaffolds only.

## Team workflow and Git strategy

`main` is stable, while `develop` is the integration branch. Create one focused
branch per task, open a pull request into `develop`, and obtain at least one
review. Direct commits to `main` are prohibited. Detailed branch, commit, and
review rules are in [CONTRIBUTING.md](CONTRIBUTING.md).

## Experiment tracking

Every experiment must add a row based on
`reports/experiments/experiment_template.csv` and document its scientific work
in the Logbuch. Code, parameters, preprocessing, timing, metrics, feature count,
interpretation, branch, member, and shared random state must be traceable.

## Individual Logbooks

Each member maintains a chronological Logbook in
`reports/logbook/member_01/` through `reports/logbook/member_04/`. Entries use
the naming convention `YYYY-MM-DD_ticket-short-title.md` and should be written
soon after the related work. Every entry must document relevant decisions,
difficulties, adaptations, and code or report references. Each member is
responsible for the accuracy and completeness of their own directory; see
`reports/logbook/README.md` for the full convention.

## Reproducibility and leakage prevention

All team members must use:

- the same `random_state`;
- the same train/test split;
- the same metrics;
- the same experiment conventions.

Shared values live in `configs/config.yaml`. Fit preprocessing and feature
selection on training folds only. Keep the test set isolated and never use it
for model selection, threshold tuning, or iterative feature decisions.

## Agile workflow

Work is organized into sprints with tickets, short team meetings, documented
decisions, blockers, assignments, deadlines, and retrospectives. Use the
templates in `reports/meetings/` and `reports/logbook/`.

## Authors / team

- Team member 1: _TBD_
- Team member 2: _TBD_
- Team member 3: _TBD_
- Team member 4: _TBD_

## Current status

**Phase 1 — project scaffold.** Structure, environment definitions, templates,
configuration, placeholder modules, notebooks, and offline tests are present.
No dataset has been downloaded, no complete EDA has been performed, and no
machine-learning model has been trained.

//comment

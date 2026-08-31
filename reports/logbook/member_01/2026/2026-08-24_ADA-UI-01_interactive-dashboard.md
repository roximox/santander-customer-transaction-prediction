# Logbook Entry

## Metadata

- Date: 2026-08-24
- Member: Yassine Elhari
- Sprint: Sprint 2
- Ticket ID: ADA-UI-01
- Branch: develop
- Pull Request: [#7 — develop → main](https://github.com/roximox/santander-customer-transaction-prediction/pull/7) (feature merged into `develop` in `37830b3`)
- Time spent: 6 hours
- Related meeting: [2026-08-23 — Model Progress, Optimization and Evaluation](../../../meetings/2026-08-23_model-progress-optimization-and-evaluation.md)

## Title

Interactive Scientific Results Dashboard

## Objective

Provide a professional Streamlit interface for exploring the project's saved
scientific evidence without manually opening CSV, JSON, and PDF artifacts.

## Context

The repository now contains registered Logistic Regression, Feature Selection,
PCA, and HistGradientBoosting results. A shared read-only presentation layer is
needed for professor demonstrations and the e-Portfolio while Member 02 results
remain unavailable.

Before starting the UI work, the team contributions had to be consolidated on
the shared `develop` branch. Member 03's Feature Selection and PCA work and
Member 04's HistGradientBoosting optimization work were reviewed and integrated
with the common data, validation, evaluation, experiment, and model-selection
infrastructure maintained by Member 01.

## Team integration and merge-conflict resolution

The integration work covered the following activities:

- synchronized the feature branches with the current remote repository state;
- merged Member 03's Feature Selection and PCA contribution into `develop`;
- merged Member 04's `feature/model-optimization` contribution into `develop`;
- resolved the content conflict in
  `reports/experiments/experiment_registry.csv` by retaining the existing M03
  experiment rows and adding the M04 registered experiment rows;
- preserved the unique experiment identifiers `M03-FS-001`, `M03-PCA-001`,
  `M04-HGB-001`, and `M04-HGB-002`;
- adapted integration tests that still assumed Member 04 had no Logbook entries;
- verified that the merged experiment artifacts, summaries, fold results, and
  registry entries remained consistent;
- ran the complete test suite after conflict resolution before pushing the
  validated `develop` state.

The conflict was additive rather than scientific: both branches had appended
valid experiment rows to the same registry location. The resolution therefore
kept both sets of records and did not recalculate or alter any recorded score.

## Post-merge scientific compatibility review

After integration, the merged Member 03 and Member 04 implementations were
checked against the shared Member 01 infrastructure. The review confirmed the
common 160,000-row development partition, five-fold `StratifiedKFold`, shared
metrics, and `random_state=42` for the registered scripts. It also identified a
methodological problem in the two Member 03 notebooks: they could evaluate the
pipeline on the complete dataset. Those notebooks were corrected to verify the
official split fingerprints, delete the reserved partition, reuse the M03
production factories, and display the existing experiment artifacts without
rerunning the official experiments.

The pre-final model-selection report was then reviewed after the M03/M04 merge.
An initially stale report still showed Member 01 only because historical output
files were protected against overwrite. Once the report was regenerated through
the intended workflow, the saved comparison exposed the available Logistic
Regression, Feature Selection, PCA, and HistGradientBoosting candidates while
continuing to report Random Forest and Extra Trees as missing.

## Work performed

Following the team integration and scientific compatibility review, implemented
a thin `app.py`, defensive cached artifact loaders, reusable Plotly charts,
formatting helpers, Streamlit components, eleven scientific pages, global
filters, and a professor mode. Added offline tests, structure checks, launch
documentation, and explicit final-test safeguards.

## Methodology

All displayed values come from existing files under `reports/`. The dashboard
does not import dataset-loading or experiment-execution modules. It never fits
an estimator, contacts OpenML during startup, updates the registry, or creates a
final-test metric. Charts retain 0–1 metric axes where comparisons could
otherwise exaggerate small differences.

## Results

The team contributions are consolidated on `develop`, the registry contains the
M03 and M04 registered experiments without duplicate identifiers, and the
merged suite passed before the dashboard work began. The interface now exposes
the data audit, split reproducibility, experiment registry, fold metrics,
interim model comparison, Member 01 analyses, Member 03 dimensionality-reduction
evidence, Member 04 HGB evidence, learning curves, selection coverage,
comparability, and interim conclusions. Missing artifacts produce readable
warnings.

## Interpretation

The dashboard separates registered CV experiments from auxiliary analyses and
keeps the current comparison explicitly interim. It supports scientific review
without becoming a second machine-learning implementation.

## Decision

Use Streamlit with Pandas and Plotly as the single visualization layer over the
existing scientific artifacts. Preserve one result source for both normal and
professor modes.

## Difficulties

Artifact schemas vary across registered experiments, search results, learning
curves, and auxiliary analyses. Defensive loaders and missing-column checks are
therefore necessary. Member 02 coverage is intentionally incomplete. Earlier in
the workflow, parallel additions to the shared experiment registry also caused
a merge conflict that required an additive, traceability-preserving resolution.

## Adaptations and deviations from the plan

PDF artifacts are represented through their underlying stored CSV/JSON data
when available so the dashboard remains interactive. No fake screenshots or
missing Member 02 results were created.

## Rejected approaches

- Retraining models inside Streamlit.
- Loading OpenML during application startup.
- Writing regenerated experiment or selection artifacts.
- Adding an action that unlocks final-test evaluation.
- Maintaining separate professor-mode result sources.

## Files changed

- `reports/experiments/experiment_registry.csv` during team integration
- `tests/test_logbooks.py` and `tests/test_project_structure.py` for integration expectations
- `notebooks/05_feature_selection.ipynb`
- `notebooks/06_pca.ipynb`
- `tests/test_feature_selection.py`
- `app.py`
- `src/dashboard/`
- `tests/test_dashboard.py`
- `tests/test_project_structure.py`
- `tests/test_logbooks.py`
- `requirements.txt`
- `README.md`
- this Logbook entry

## Code references

Team integration reused `src/data.py`, `src/validation.py`, `src/evaluation.py`,
`src/experiments.py`, and `src/model_selection.py`. The corrected Member 03
notebooks reuse factories from `src/feature_selection.py` and read their saved
M03 result artifacts without training.

Cached artifact access is implemented in `src/dashboard/loaders.py`; interactive
figures are implemented in `src/dashboard/charts.py`; reusable presentation
elements are implemented in `src/dashboard/components.py`; page orchestration
and navigation are implemented in `app.py`.

## Figure and table references

- `reports/experiments/experiment_registry.csv`
- `reports/experiments/M03-FS-001_summary.json`
- `reports/experiments/M03-PCA-001_summary.json`
- `reports/experiments/M04-HGB-001_summary.json`
- `reports/experiments/M04-HGB-002_summary.json`
- `reports/model_selection/model_comparison_eligible.csv`
- `reports/model_selection/model_selection_summary.json`
- `reports/tables/data_audit_summary.json`
- `reports/tables/train_test_split_summary.json`
- `reports/tables/logistic_learning_curve_summary.csv`
- `reports/tables/M04-HGB-learning-curve.csv`

## Reproducibility notes

Install dependencies with `pip install -r requirements.txt`, then launch from
the repository root with `streamlit run app.py`. The same saved artifacts drive
both display modes. The final test remains reserved and no final test metric is
read or computed; the reserved partition is not used.

## Next step

Run the complete offline test suite and startup smoke test, then demonstrate the
interim dashboard to the group. Integrate Member 02 artifacts automatically
once their registered experiments are available.

## Sources and tools used

- Streamlit documentation and installed runtime.
- Plotly and Pandas APIs.
- Existing project artifact schemas and shared model-selection utilities.
- Repository-local pytest and nbformat validation.
- No external model training or final test evaluation was used.

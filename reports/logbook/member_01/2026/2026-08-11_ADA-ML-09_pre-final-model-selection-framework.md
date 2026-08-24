# Logbook Entry

## Metadata

- Date: 2026-08-11
- Member: Yassine Elhari
- Sprint: Sprint 2
- Ticket ID: ADA-ML-09
- Branch: feature/data_processing
- Pull Request: To be updated after Pull Request creation
- Time spent: TO BE COMPLETED BY YASSINE ELHARI
- Related meeting: TO BE COMPLETED BY YASSINE ELHARI

## Title

Pre-Final Model Selection Framework

## Objective

Build a common read-only framework that compares recorded training-CV results
before the group locks one final pipeline. The framework must remain useful
while some members' candidates are absent.

## Context

Different model families and result producers may expose compatible information
under different JSON keys. A normalized comparison is necessary to make model
coverage, protocol metadata, metric trade-offs, and missing evidence explicit.

## Work performed

Implemented deterministic summary discovery, robust normalization, eligibility
and exclusion reporting, protocol comparability checks, metric-specific ranks,
a CV-variability competitiveness heuristic, multi-criteria decisions, expected
family coverage, portfolio exports, meeting notes, figures, and an offline test
suite. The script reads six registered M01 experiments and the two retained
Logistic candidates from the recorded grid search.

## Methodology

ROC-AUC remains the predeclared primary metric. Average Precision, F1,
precision, recall, accuracy, balanced accuracy, fold dispersion,
train-validation gap, fit time, convergence, feature count, and provenance are
kept separately. No weighted or averaged composite rank is computed.
Comparability checks distinguish consistent, incompatible, and not-verifiable
metadata. Missing candidates are reported rather than fabricated.

## Results

The generated report records four eligible Logistic candidates and excludes
four Dummy baselines. Members 02–04 and their expected RF, Extra Trees, PCA,
Feature Selection, and HGB candidates are missing. Consequently selection
status is `waiting_for_additional_models`; no group winner is declared.

## Interpretation

Metric winners and competitive candidates are meeting inputs, not a final
selection. The CV variability rule uses one standard deviation of the best
recorded ROC-AUC and is explicitly a heuristic rather than a formal
non-inferiority test.

## Decision

Preserve a transparent multi-criteria review. Candidate status is distinct from
final-model status. The final pipeline decision is deferred to the group after
coverage and protocol comparability are complete.

## Difficulties

Registered experiment summaries and grid-search summaries use different field
names. Search candidates also lack some protocol metadata available in the
registered experiment files, so comparability can be only partially verified.

## Adaptations and deviations from the plan

The two selected Logistic configurations are normalized directly from the
existing candidate CSV with `source_type=grid_search_candidate`; neither the
experiment registry nor historical scientific artifacts are modified.

## Rejected approaches

- Retraining candidates or loading OpenML data.
- Reading or calculating final-test metrics.
- Selecting by Accuracy alone.
- An arbitrary weighted composite score or naive average rank.
- Inventing missing Member 02–04 experiments or a group decision.

## Files changed

- `src/model_selection.py`
- `scripts/build_model_selection_report.py`
- `tests/test_model_selection.py`
- `tests/test_project_structure.py`
- `notebooks/08_final_evaluation.ipynb`
- `README.md`
- `CONTRIBUTING.md`
- `reports/model_selection/` exports and meeting notes
- four pre-final comparison figures

## Code references

Summary discovery/loading/normalization, model-family inference, eligibility,
comparability, rankings, competitiveness, decisions, coverage, portfolio, and
figure functions in `src/model_selection.py`; report orchestration in
`scripts/build_model_selection_report.py`.

## Figure and table references

- `reports/model_selection/model_comparison_all.csv`
- `reports/model_selection/model_comparison_eligible.csv`
- `reports/model_selection/model_comparison_excluded.csv`
- `reports/model_selection/model_selection_decision.csv`
- `reports/model_selection/model_selection_coverage.csv`
- `reports/model_selection/model_selection_summary.json`
- `reports/model_selection/model_selection_comparability.json`
- `reports/model_selection/model_comparison_portfolio.csv`
- `reports/model_selection/model_comparison_portfolio.md`
- `reports/model_selection/group_model_selection_notes.md`
- `reports/figures/final_model_comparison_roc_auc.pdf`
- `reports/figures/final_model_comparison_average_precision.pdf`
- `reports/figures/final_model_comparison_threshold_metrics.pdf`
- `reports/figures/final_model_performance_vs_time.pdf`

## Reproducibility notes

Inputs are immutable recorded JSON/CSV artifacts. Discovery and tie-breaking
are deterministic. Paths stored in outputs are project-relative. No dataset,
estimator, model serialization, Internet service, or final test set is used.

## Next step

Group review and final model lock after Members 02–04 publish their retained,
common-protocol candidates. The final test remains closed until that lock.

## Sources and tools used

Existing experiment/search reports, project evaluation conventions, Python,
pandas, NumPy, Matplotlib, JSON, pytest, nbformat, and Git inspection commands.

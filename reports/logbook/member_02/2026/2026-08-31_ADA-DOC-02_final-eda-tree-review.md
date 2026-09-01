# Logbook Entry

## Metadata

- Date: 2026-08-31
- Member: Aya Olali
- Sprint: Final submission review
- Ticket ID: ADA-DOC-02
- Branch: develop
- Pull Request: Committed directly to develop as documentation
- Time spent: 4 hours (retrospective estimate)
- Related meeting: [2026-08-31 — Final submission and presentation review](../../../meetings/2026-08-31_final-submission-and-presentation-review.md)

## Title

Final consistency review of the Member 02 EDA and tree-model material

## Objective

Check that Member 02 documentation, experiment references, reported metrics,
and presentation statements remain consistent with the final repository and
with the separation between individual development and collective outcomes.

## Context

Model development, selection, and the controlled final evaluation were closed.
The last regular meeting focused on documentation, individual portfolios,
presentation readiness, and consistency across the repository.

## Work performed

- Reviewed the three existing Member 02 Logbook entries and their artifact
  references.
- Checked the Decision Tree and Random Forest identifiers in the experiment
  registry.
- Verified the main validation means and standard deviations against the saved
  summaries and comparison table.
- Checked that Member 02 documentation states that EDA and development
  evaluation used training data only.
- Confirmed that the Random Forest is described as the stronger Member 02
  baseline, not as the collectively selected final model.
- Reviewed the presentation narrative for consistent terminology and limits.

## Methodology

This was a read-only consistency audit of committed Markdown, CSV, JSON,
notebook, registry, and figure references. No model fitting, threshold tuning,
model selection, or final-test access occurred.

## Results

The central Member 02 figures remain consistent: Decision Tree validation
ROC-AUC 0.633935 ± 0.003119 and Random Forest validation ROC-AUC 0.793736 ±
0.002392. The comparison and experiment summaries use the same identifiers and
protocol. The final project decision remains attributed to the group and to the
locked `M04-HGB-002` experiment.

The review also identified that personal administrative metadata, especially
actual time spent, must be supplied by Aya rather than inferred from Git
timestamps or repository artifacts.

## Interpretation

The Member 02 contribution is traceable from training-only EDA through the two
registered tree baselines and into the collective comparison. The repository
supports the technical and scientific claims, but it cannot prove personal
hours that were not recorded contemporaneously.

## Decision

Keep the scientific workflow closed. Complete only truthful personal metadata
and final communication material; do not add new experiments or reinterpret
the reserved final-test result.

## Difficulties

The repository contains development CV, collective selection, and final-test
artifacts. These evidence types must remain clearly separated. Git timestamps
also establish when files were committed, not the duration of the work.

## Adaptations and deviations from the plan

The final review was limited to documentation and evidence consistency, in line
with the meeting decision not to reopen modeling.

## Rejected approaches

- Inventing or backdating working hours from commit timestamps.
- Creating new experiments after final-model lock.
- Attributing the collective final-test result to Member 02.
- Changing stored metrics to simplify the presentation.

## Files changed

- This Logbook entry only.

## Code references

- `notebooks/02_eda.ipynb`
- `notebooks/04_tree_models.ipynb`
- `scripts/run_tree_models.py`
- `src/model_selection.py`

## Figure and table references

- `reports/experiments/experiment_registry.csv`
- `reports/tables/tree_model_comparison.csv`
- `reports/model_selection/model_selection_decision.csv`
- `reports/scientific_conclusions.md`

## Reproducibility notes

The review reads existing artifacts only. The final test was not rerun or used
for further selection. Reported Member 02 metrics remain development
cross-validation results.

## Sources and tools used

Git history, meeting notes, Member 02 Logbooks, registered experiment JSON/CSV
files, comparison reports, and final project documentation.

## Next step

Aya completes the real time-spent fields and any personal presentation details,
then verifies the final rendered submission material.

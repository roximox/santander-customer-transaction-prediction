# Logbook Entry

## Metadata

- Date: 2026-07-26
- Member: Aya Olali
- Sprint: Project planning
- Ticket ID: ADA-MEET-01
- Branch: develop
- Pull Request: Not applicable — meeting, preparation, and analysis work
- Time spent: 4 hours (retrospective estimate)
- Related meeting: [2026-07-26 — Initial project planning and task distribution](../../../meetings/2026-07-26_initial-project-planning-and-task-distribution.md)

## Title

Project selection, responsibility definition, and initial methodological discussion

## Objective

Understand the project options, participate in selecting Santander, and define
an individual contribution combining EDA with tree-based Machine Learning.

## Context

This was the first group meeting. No shared implementation or experiment was
available yet, so the work focused on preparation, discussion, responsibilities,
risks, and the scientific sequence of the project.

## Work performed

- Reviewed the proposed project alternatives before the meeting.
- Discussed the suitability of the Santander classification problem.
- Participated in distributing the four scientific work packages.
- Defined the Member 02 responsibility: EDA, Random Forest, and Extra Trees.
- Discussed anonymized variables, class imbalance, reproducibility, and leakage risks.
- Agreed to contribute to later reviews, conclusions, portfolio, and presentation work.

## Methodology

The initial plan separated shared infrastructure from individual scientific
work while requiring common validation and comparable reporting.

## Results

Santander was selected and Member 02 received a clear analysis and ML scope.
The meeting established the dependency on a common loader, split, and evaluation
protocol before model comparisons could be scientifically meaningful.

## Interpretation

Combining EDA with tree models was appropriate because anonymous numerical
features may contain nonlinear relationships that a single linear summary
cannot expose.

## Decision

Prepare a training-only EDA and later compare reproducible tree baselines using
the common project methodology.

## Difficulties

The exact dataset interface and validation framework were not yet available.

## Adaptations and deviations from the plan

None at this stage; this entry records the initial plan.

## Rejected approaches

Independent datasets, incompatible local splits, and premature final-test use
were rejected during planning.

## Files changed

- This retrospective Logbook entry only.

## Code references

- Later implementation: `notebooks/02_eda.ipynb` and `notebooks/04_tree_models.ipynb`.

## Figure and table references

- None at the planning stage.

## Reproducibility notes

The final test was not available or used. This entry documents discussion and
planning rather than a model result.

## Sources and tools used

Project brief, group discussion, repository plan, and meeting record.

## Next step

Prepare the EDA approach while waiting for the common data foundation.

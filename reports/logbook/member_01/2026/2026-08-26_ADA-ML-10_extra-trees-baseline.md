# Logbook Entry

## Metadata

- Date: 2026-08-26
- Member: Yassine Elhari
- Sprint: Sprint 2
- Ticket ID: ADA-ML-10
- Branch: develop
- Pull Request: Not applicable — committed directly to `develop` (`190c7c5`)
- Time spent: 3 hours
- Related meeting: [2026-08-26 — Final Model-Selection Decision](../../../meetings/2026-08-26_final-model-selection-decision.md)

## Title

Member 01 Extra Trees Baseline

## Objective

Complete the expected pre-final model-family coverage by implementing and
registering one reproducible Extra Trees baseline under Member 01.

## Context

After Member 02's Decision Tree and Random Forest results were integrated, the
model-selection coverage still reported `EXTRA_TREES` as missing. The team
assigned this remaining baseline to Member 01. The experiment must use the same
development split, validation folds, metrics, and random state as every other
registered candidate.

## Work performed

Added a dedicated training-only entry point for `M01-ET-001`, reused the shared
Extra Trees factory and experiment orchestrator, verified both official split
fingerprints, added overwrite protection and offline safeguards, ran the
registered five-fold experiment, and prepared the model-selection reports for
regeneration from the new saved evidence.

## Methodology

The estimator uses 200 trees, `max_depth=8`, `class_weight="balanced"`,
`random_state=42`, and parallel tree construction. Evaluation uses the common
five-fold shuffled `StratifiedKFold` on the 160,000-row development partition.
ROC-AUC is primary; Average Precision, F1, precision, recall, accuracy, and
balanced accuracy are reported through the common scoring infrastructure.

## Results

The registered five-fold development-CV results are:

- ROC-AUC: `0.847946 ± 0.003073`;
- Average Precision: `0.475459 ± 0.007537`;
- F1: `0.441943 ± 0.002290`;
- precision: `0.330575 ± 0.001983`;
- recall: `0.666500 ± 0.004820`;
- accuracy: `0.830856 ± 0.001176`;
- balanced accuracy: `0.757859 ± 0.002129`.

After regeneration, 13 registered experiments were discovered, 11 candidates
were eligible, no expected model family was missing, and selection status became
`ready_for_group_review`. Extra Trees recorded the highest current mean CV F1,
while `M04-HGB-002` retained the highest mean ROC-AUC and Average Precision.

## Interpretation

Extra Trees provides the previously missing ensemble-family evidence for the
interim comparison. Its performance must be reviewed alongside ranking quality,
threshold metrics, fit time, and generalization gap; its presence does not imply
that it is the final model.

## Decision

Register `M01-ET-001` as a baseline candidate and include it in the transparent
multi-criteria group review. Keep the final test locked until the group selects
one final pipeline.

## Difficulties

Historical model-selection outputs are overwrite-protected and must be backed
up before regeneration. The experiment is computationally heavier than the
linear baselines because five independent tree ensembles are evaluated.

## Adaptations and deviations from the plan

Extra Trees was initially listed under Member 02 coverage. Responsibility was
reassigned to Member 01 after Member 02 delivered Decision Tree and Random
Forest. Only the expected ownership metadata changed; the common scientific
protocol did not change.

## Rejected approaches

- Fabricating an Extra Trees result to clear the coverage status.
- Reusing the reserved final test for model selection.
- Tuning parameters against final-test performance.
- Overwriting an existing experiment or registry entry.
- Declaring a final winner from this single additional baseline.

## Files changed

- `scripts/run_extra_trees_baseline.py`
- `src/model_selection.py`
- `tests/test_extra_trees_experiment.py`
- `tests/test_project_structure.py`
- `tests/test_logbooks.py`
- `reports/experiments/M01-ET-001_fold_results.csv`
- `reports/experiments/M01-ET-001_summary.json`
- `reports/experiments/experiment_registry.csv`
- regenerated files under `reports/model_selection/`

## Code references

Estimator construction uses `create_extra_trees_classifier()` from
`src/modeling.py`. Data loading, split verification, evaluation, persistence,
and registration use `src/data.py`, `src/validation.py`, `src/evaluation.py`,
and `src/experiments.py`.

## Figure and table references

- `reports/experiments/M01-ET-001_fold_results.csv`
- `reports/experiments/M01-ET-001_summary.json`
- `reports/experiments/experiment_registry.csv`
- `reports/model_selection/model_comparison_eligible.csv`
- `reports/model_selection/model_selection_coverage.csv`
- `reports/model_selection/model_selection_summary.json`

## Reproducibility notes

Run from the repository root with
`MPLCONFIGDIR=/tmp/ada-mpl-cache python scripts/run_extra_trees_baseline.py`.
The command refuses existing outputs. The reserved partition is fingerprinted,
deleted before evaluation, and not used.

## Next step

Regenerate the pre-final model-selection outputs, review complete expected-family
coverage with the group, and lock one pipeline before final-test evaluation.

## Sources and tools used

- Existing shared Extra Trees factory and experiment framework.
- scikit-learn ExtraTreesClassifier documentation reflected by the factory API.
- Repository-local pytest safeguards and recorded CV artifacts.
- No final-test result or external unrecorded metric was used.

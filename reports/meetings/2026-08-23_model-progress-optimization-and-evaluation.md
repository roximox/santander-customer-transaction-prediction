# Model Progress, Optimization and Evaluation

- **Date:** 2026-08-23
- **Participants:** Member 01 — Yassine El Hari; Member 02 — Aya Olali; Member 03 — Iliase Lhamri; Member 04 — Chaymae Akouaouch.
- **Sprint goal:** Review the more mature results across the ML approaches, consolidate optimization and diagnostic evidence, and prepare the individual model tracks for a consistent later cross-model comparison.
- **Results from previous sprint:** All four individual work packages progressed toward more mature analysis and modeling results. Member 01 continued Logistic Regression and baseline work; Member 02 progressed EDA and tree-based modeling; Member 03 progressed PCA and feature-selection analysis; and Member 04 completed the main HistGradientBoosting optimization and diagnostic workflow and formally registered the tuned HGB configuration.

## Discussion

The team reviewed progress from all four scientific areas.

Logistic Regression and baseline work continued to provide an important reference while the shared validation and reproducibility foundation supported comparable experimentation.

EDA and tree-based modeling progressed, contributing to the understanding of dataset characteristics and the behaviour of tree-based approaches.

PCA and feature-selection work progressed, focusing on whether reduced or selected feature representations could remain competitive while reducing the input representation.

For HistGradientBoosting, the baseline had already been followed by a training-only hyperparameter search, learning-curve analysis, out-of-fold evaluation, and a baseline-versus-tuned comparison. The tuned configuration had also been formally registered as `M04-HGB-002`. The available evidence showed an improvement over the HGB baseline, while the remaining train-validation gap and the precision-recall trade-off still required careful interpretation.

The team agreed that the later cross-model comparison should consider predictive performance, generalization behaviour, stability, computational aspects, and methodological consistency rather than relying on a single metric.

## Decisions

The individual model tracks will now focus on consolidating comparable evidence rather than introducing unnecessary additional optimization.

The reserved final-test partition remains untouched and must not be used for model selection.

Model comparison will use consistent evaluation criteria, and results from the individual work packages will be prepared for the later team-level cross-model review.

`M04-HGB-002` remains the registered tuned HGB candidate for that comparison, but no final model was selected during this meeting.

Final model selection remains a collective team decision.

Members will continue documenting important methodological decisions, results, and limitations.

## Task Assignment

- **Member 01** will continue evaluation and documentation of Logistic Regression and baseline results and support consistency of the common validation and reproducibility framework.
- **Member 02** will consolidate tree-based modeling results, connect relevant findings with EDA, and prepare important observations for comparison with other model families.
- **Member 03** will consolidate PCA and feature-selection evaluation and prepare a clear comparison of original and reduced or selected feature representations.
- **Member 04** will consolidate the registered tuned HistGradientBoosting evidence, its diagnostic results, and the baseline-versus-tuned comparison, and prepare the HGB results for reporting and later cross-model review.

All members will prepare their most relevant results, methodological decisions, limitations, and supporting artifacts for the later team-level model comparison.

## Risks and Blockers

As the number of completed experiments increases, differences in preprocessing, validation, or reporting could undermine comparability between model tracks.

Improved validation performance must not automatically be interpreted as universal model superiority. Generalization behaviour, class-specific metrics, computational cost, and methodological consistency should also be considered.

The reserved final-test partition must remain isolated until the group has completed model selection.

Documentation must remain synchronized with the actual experiments. No specific technical failure was identified.

## Changes from Initial Plan

No major responsibility change was made. The project has moved from initial model development and optimization toward evidence consolidation and preparation for cross-model comparison.

For Member 04, the main HGB optimization and diagnostic workflow is now complete, and the focus shifts toward consolidation, reporting, and preparation for the later group-level model review.

## Deadlines

No exact deadlines were set. Before the final project-phase meeting, each member will consolidate their main results, complete remaining documentation and reporting work, prepare important figures and tables, and be ready to discuss the cross-model comparison and final project tasks.

## Next Meeting

Consolidate results from all four members; conduct the cross-model comparison; discuss the strongest model candidates; make the collective final-model decision before opening the reserved test partition; address remaining technical or documentation work; review logbook and portfolio completeness; and prepare the final presentation and remaining tasks before submission.

## Retrospective

- **What went well:** The individual ML work packages had progressed sufficiently to move from model development and optimization toward evidence consolidation and cross-model comparison.
- **What did not go well:** As the number of experiments increased, keeping evaluation, interpretation, and documentation consistent required more coordination.
- **Improvement action:** Consolidate the most relevant results and artifacts before the next meeting so the team can perform a clear and methodologically consistent cross-model comparison before any final-test evaluation.

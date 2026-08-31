# Logbook Entry
 
## Metadata
 
**Date:** 2026-08-26
 
**Member:** Chaymae Akouaouch
 
**Phase:** Collective model selection
 
**Ticket ID:** M04-SEL-001
 
**Branch:** feature/model-optimization
 
**Related meeting:** [2026-08-26 — Final Model-Selection Decision](../../../meetings/2026-08-26_final-model-selection-decision.md)
 
## Title
 
Collective Final-Model Selection Participation and Confirmation
 
## Objective
 
Take part in the group-level review of the saved candidate evidence and in the collective confirmation of one frozen pipeline, before the reserved final-test partition was opened.
 
## Context
 
By this point, my Member 04 work had already produced and formally registered `M04-HGB-002`, using the configuration selected earlier in `M04-HGB-SEARCH-001`. The group review weighed this candidate against the completed Logistic Regression, Decision Tree, Random Forest, Extra Trees, feature-selection, and PCA evidence.
 
## Work Performed
 
1. Took part in the collective review of the saved eligible candidate results.
2. Provided the previously completed `M04-HGB-002` cross-validation evidence, along with its known limitations, for that review.
3. Took part in the collective confirmation of `M04-HGB-002` as the final pipeline.
4. Confirmed that no additional Member 04 hyperparameter tuning or threshold optimization happened during this selection stage.
## Evidence Used
 
The shared comparison used only the 160,000-row development partition with five-fold `StratifiedKFold` and ROC-AUC as the predefined primary metric.
 
| Evidence | Recorded value |
|---|---:|
| `M04-HGB-002` mean CV ROC-AUC | 0.891449 |
| `M04-HGB-002` mean CV Average Precision | 0.591089 |
| `M04-HGB-002` mean CV precision | 0.795747 |
| `M04-HGB-002` mean CV recall | 0.280942 |
| `M04-HGB-002` ROC-AUC train-validation gap | 0.082131 |
| `M04-HGB-002` mean fit time per fold | 115.013859 seconds |
 
`M04-HGB-002` led the eligible candidates on mean CV ROC-AUC and Average Precision. The review also kept the other operating-point trade-offs in view: Extra Trees had the strongest recorded mean F1, while the balanced Logistic Regression search candidate had the strongest recorded recall and balanced accuracy.
 
## Interpretation
 
Average Precision was weighed alongside ROC-AUC because the target is imbalanced. The evidence supported `M04-HGB-002` as the strongest recorded ranking candidate, though not as the best model on every threshold metric. Its larger train-validation gap, higher computational cost, and limited recall at the fixed threshold remained explicit limitations going into the decision.
 
## Decision
 
The group collectively selected `M04-HGB-002` (`HistGradientBoosting Tuned`) as the final pipeline. Its previously selected estimator configuration and classification threshold of 0.5 were frozen before final evaluation. The reserved final-test partition was not used for this selection decision.
 
## Role Boundaries
 
My contribution at this stage was participating in the review and the collective confirmation, backed by the HGB evidence I had already completed. I did not implement the group model-selection framework, the final model-lock artifact, dashboard integration, or the final-evaluation pipeline, and I did not execute or persist the final-test workflow. Those implementation activities are documented under Member 01's `ADA-ML-11` entry.
 
## Difficulties
 
The main discussion point was weighing strong ranking performance against other models' different operating-point strengths, alongside HGB's runtime, generalization gap, and recall limitation.
 
## Adaptations and Deviations from the Plan
 
No new Member 04 experiment, tuning stage, or threshold change was introduced. The group waited until coverage across the expected model families was complete before making the collective decision.
 
## Rejected Approaches
 
- Reopening Member 04 hyperparameter optimization during group selection.
- Changing the 0.5 classification threshold based on candidate-comparison evidence.
- Opening or using the reserved final-test partition to select a model.
## Files / References
 
- `reports/experiments/M04-HGB-002_summary.json`
- `reports/model_selection/model_comparison_eligible.csv`
- `reports/model_selection/model_selection_decision.csv`
- `reports/meetings/2026-08-26_final-model-selection-decision.md`
- `reports/model_selection/final_model_lock.json`
- `reports/logbook/member_01/2026/2026-08-26_ADA-ML-11_final-model-lock.md`
## Reproducibility Notes
 
The selection evidence relied on saved, training-only cross-validation artifacts. The HGB configuration had already been frozen before this stage, and the reserved final-test partition stayed outside candidate selection, tuning, and threshold decisions.
 
## Next Step
 
The next controlled project step was a single final evaluation of the already frozen collective choice, without reopening selection or modifying the Member 04 configuration.
 
## Sources and Tools Used
 
Saved experiment summaries, model-selection reports, the final model-lock record, and the final model-selection meeting record.
 

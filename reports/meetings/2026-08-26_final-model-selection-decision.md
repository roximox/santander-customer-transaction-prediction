# Final Model Selection and Controlled Final Evaluation

- **Date:** 2026-08-26
- **Participants:** Member 01 — Yassine El Hari; Member 02 — Aya Olali; Member 03 — Iliase Lhamri; Member 04 — Chaymae Akouaouch.
- **Sprint goal:** Compare the eligible model candidates, collectively select and freeze one final model, and only then perform the single controlled evaluation on the reserved final-test partition.
- **Results from previous sprint:** Thirteen registered experiments were discovered, eleven candidates were eligible for model selection, and the expected model families were represented. The individual model tracks had produced sufficient comparable evidence for the collective review.

## Discussion

The group reviewed the eleven eligible candidates using the common development-data evidence. The comparison considered the primary ranking metric ROC-AUC together with Average Precision, F1, precision, recall, balanced accuracy, variability, train-validation generalization gaps, and computational cost.

`M04-HGB-002` achieved the highest mean CV ROC-AUC (0.891449) and Average Precision (0.591089) among the eligible candidates. Other candidates showed different operating-point advantages: Extra Trees achieved the highest recorded mean F1, while a balanced Logistic Regression candidate achieved the highest recall and balanced accuracy. Logistic Regression also remained computationally cheaper and showed a considerably smaller train-validation gap than HGB.

The group therefore treated the decision as a multi-criteria model-selection problem rather than assuming that one model was superior on every metric.

## Decisions

The group collectively selected `M04-HGB-002` (`HistGradientBoosting Tuned`) as the final model.

Its recorded estimator configuration and classification threshold of `0.5` were frozen before opening the reserved final-test partition.

The final test must not be used for additional model selection, hyperparameter tuning, threshold optimization, or reopening the model decision.

Only after the model lock and execution procedure were verified could the single controlled final-test evaluation be performed.

## Task Assignment

- **Member 01** will record and expose the collective model-selection decision, maintain the final lock artifact, and execute/record the controlled final-evaluation workflow.
- **Members 02, 03, and 04** will contribute to the collective review and verify that the final decision is consistent with the available model evidence.
- **Member 04** contributes the HGB evidence used in the comparison but does not implement the group-level selection framework, model-lock mechanism, or final-evaluation pipeline.

All members will preserve the final selection and use the resulting evidence consistently in the final documentation and presentation.

## Risks and Blockers

The selected HGB model has a larger train-validation gap and higher computational cost than Logistic Regression. Its recall at the fixed `0.5` threshold also remains limited.

These limitations must remain visible in the final interpretation. Selection of HGB should therefore not be presented as proof that it is universally superior or statistically superior to every alternative.

No remaining blocker prevented the controlled final evaluation once the model lock was verified.

## Changes from Initial Plan

No responsibility areas were changed. The project moved from individual model development and evidence consolidation into collective model selection and final evaluation.

The expected model-family coverage was complete for the final comparison, including Extra Trees.

## Deadlines

The reserved final-test evaluation may be executed only after verification of the final model lock and execution procedure. After the one-time evaluation, no further tuning or model reselection is permitted.

## Controlled Final-Evaluation Outcome

After the collective model decision was frozen, `M04-HGB-002` was evaluated once on the reserved 40,000-observation final-test partition.

- **Execution count:** 1
- **Selection reopened:** No
- **Final-test size:** 40,000 observations
- **ROC-AUC:** 0.891214
- **Average Precision:** 0.584385
- **F1:** 0.403632
- **Precision:** 0.791424
- **Recall:** 0.270896
- **Accuracy:** 0.919550
- **Balanced Accuracy:** 0.631459
- **Classification threshold:** 0.5

The final ROC-AUC of 0.891214 was close to the development CV mean of 0.891449. Average Precision was also close to the development estimate (0.584385 final test versus 0.591089 CV). The limited recall observed during development remained visible on the final test.

These differences are interpreted descriptively rather than as a formal statistical test. No post-test tuning was performed and model selection was not reopened.

## Next Meeting

Review the completed final-test evidence, finalize scientific conclusions and documentation, check logbook and portfolio completeness, and prepare the final presentation without reopening model selection.

## Retrospective

- **What went well:** The shared validation and experiment infrastructure provided sufficiently consistent and traceable evidence for the collective model-selection decision, and the reserved test partition remained isolated until after the final model lock.
- **What did not go well:** The candidate models showed different trade-offs across ranking metrics, threshold-dependent metrics, generalization behaviour, and computational cost, so the final choice could not be justified by a single metric alone.
- **Improvement action:** Preserve the final model lock, report both strengths and limitations of the selected model, and keep development/CV evidence clearly separated from the one-time final-test result.

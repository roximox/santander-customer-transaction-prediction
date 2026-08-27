# Final model-selection decision

- Date: 2026-08-26
- Participants: Members 01, 02, 03, and 04; collective confirmation reported to Member 01
- Sprint goal: Select and freeze one model before the single reserved final-test evaluation.
- Results from previous sprint: Thirteen experiments were discovered, eleven candidates were eligible, and all expected model families were covered.
- Discussion: The group compared ranking metrics, threshold metrics, variability, generalization gaps, and computational cost. M04-HGB-002 leads the eligible candidates on mean CV ROC-AUC and Average Precision. Extra Trees has the strongest recorded mean F1, while Logistic Regression remains cheaper and more stable.
- Decisions: Select `M04-HGB-002` (`HistGradientBoosting Tuned`) as the final pipeline. Freeze its recorded estimator parameters and the 0.5 classification threshold. Do not use the final test for any further selection or tuning.
- Task assignment: Member 01 records and exposes the collective decision; the final-test run remains a separate controlled action.
- Risks and blockers: HGB has a larger train-validation gap and computational cost than Logistic Regression, and recall at threshold 0.5 remains limited.
- Changes from initial plan: The previously missing Extra Trees family was added before the collective review, completing expected-family coverage.
- Deadlines: Final-test evaluation only after verification of the lock artifact and execution procedure.
- Next meeting: Review the single final-test result and finalize the report without reopening model selection.

## Retrospective

- What went well: Shared validation infrastructure made results traceable and comparable enough for a collective decision.
- What did not go well: Some candidates expose different operating-point trade-offs, and the overall comparability assessment remains partially comparable.
- Improvement action: Preserve the lock, execute the reserved test once, and report both ranking and threshold metrics with limitations.

## Controlled final-evaluation outcome

- Execution count: One.
- Selection reopened: No.
- Final-test size: 40,000 observations.
- ROC-AUC: 0.891214.
- Average Precision: 0.584385.
- F1 at the frozen 0.5 threshold: 0.403632.
- Precision: 0.791424.
- Recall: 0.270896.
- Balanced Accuracy: 0.631459.
- Interpretation: Final ROC-AUC is close to the CV mean of 0.891449. The recorded recall limitation remains present and no post-test tuning is authorized.

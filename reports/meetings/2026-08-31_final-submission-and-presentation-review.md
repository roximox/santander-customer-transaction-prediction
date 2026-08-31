# Final Submission and Presentation Review

- **Date:** 2026-08-31
- **Participants:** Member 01 — Yassine El Hari; Member 02 — Aya Olali; Member 03 — Iliase Lhamri; Member 04 — Chaymae Akouaouch.
- **Sprint goal:** Review the final project status before submission, check consistency between experimental evidence and documentation, and coordinate the remaining portfolio and presentation work.
- **Results from previous sprint:** The modeling and model-selection phases were completed. `M04-HGB-002` had been collectively selected and locked as the final model before the reserved final-test partition was opened. The final model was subsequently evaluated once on the 40,000-observation reserved final-test set, and model selection was not reopened.

## Discussion

The team reviewed the overall project status with the focus now on documentation, portfolios, presentation readiness, and consistency rather than additional model development.

The completed experimental workflow was reviewed conceptually to ensure that development cross-validation results, out-of-fold diagnostic evidence, and the one-time final-test results remain clearly separated in the final project materials.

The team also discussed the importance of maintaining a clear distinction between individual contributions and collective project outcomes. Individual model development and analysis should remain attributable to the responsible member, while shared methodology, cross-model comparison, collective final-model selection, and overall conclusions should be presented as group-level activities where appropriate.

For the remaining presentation work, the team agreed to focus on communicating the main scientific story clearly: dataset and methodology, individual modeling approaches, candidate comparison, final-model decision, controlled final-test result, limitations, and conclusions.

No additional model tuning, threshold optimization, or model-selection activity was planned.

## Decisions

The experimental and model-selection phases remain closed.

`M04-HGB-002` remains the locked final model, and the collective model-selection decision will not be reopened.

The one-time final-test result will remain clearly separated from development CV and OOF evidence.

No additional tuning or threshold optimization based on final-test results will be performed.

Remaining work will focus on documentation, individual portfolios, presentation preparation, consistency checks, and submission readiness.

Final claims must remain supported by the recorded experimental evidence and should include the relevant limitations.

## Task Assignment

- **Member 01** will review the common methodology, reproducibility documentation, Logistic Regression material, and shared final-model and final-evaluation documentation.
- **Member 02** will review the EDA and tree-based modeling material and prepare the most relevant findings and visualizations for final communication.
- **Member 03** will review the PCA, feature-selection, and reduced-feature modeling material and prepare the corresponding methodological findings.
- **Member 04** will review the HistGradientBoosting baseline, optimization, learning-curve, OOF, and comparison material and ensure that the HGB contribution to the collective final-model decision is represented accurately.

All members will continue checking their individual documentation and coordinate the remaining presentation and submission-related work.

## Risks and Blockers

The main remaining risks concern documentation and communication rather than model development.

Potential issues include inconsistent metric values between artifacts, mixing CV, OOF, and final-test results, unclear distinction between individual and collective contributions, overstating the superiority of the selected model, excessive technical detail in the presentation, and inconsistent terminology or conclusions.

No remaining modeling issue requires reopening the completed experimental workflow.

## Changes from Initial Plan

No change was made to the scientific responsibility distribution.

The project is now in its final documentation and submission-preparation phase. Modeling, optimization, collective model selection, and controlled final evaluation are complete.

## Deadlines

The remaining work before submission focuses on completing individual portfolio and documentation tasks, preparing the presentation, checking reported results and conclusions, and resolving remaining consistency issues.

## Next Meeting

No further regular project meeting is planned.

Any additional communication before submission will focus only on remaining documentation, presentation, or submission issues and will not reopen model development or selection.

## Retrospective

- **What went well:** The team completed the experimental workflow from common validation through individual modeling, collective model selection, and controlled final evaluation while preserving the reserved test partition until the final stage.
- **What did not go well:** The growing number of experiments, metrics, figures, and documentation artifacts increased the effort required to maintain consistent reporting.
- **Improvement action:** Complete the remaining documentation and presentation work and perform final consistency checks so that reported metrics, methodological statements, individual contributions, and collective conclusions remain aligned.

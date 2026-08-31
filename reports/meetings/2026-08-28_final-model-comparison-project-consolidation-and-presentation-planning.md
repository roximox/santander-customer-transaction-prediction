# Final Project Consolidation and Presentation Planning

- **Date:** 2026-08-28
- **Participants:** Member 01 — Yassine El Hari; Member 02 — Aya Olali; Member 03 — Iliase Lhamri; Member 04 — Chaymae Akouaouch.
- **Sprint goal:** Consolidate the completed work of all four members, review the final model-selection and final-evaluation evidence, align the documentation, and organize the remaining work for presentation and submission.
- **Results from previous sprint:** The individual model tracks had been consolidated and compared under the common evaluation framework. On 2026-08-26, the group collectively selected and locked `M04-HGB-002` as the final model before opening the reserved test partition. The selected model was subsequently evaluated once on the 40,000-observation reserved final-test set, and model selection was not reopened.

## Discussion

The team reviewed the completed contributions from all four work packages and discussed how they should be presented as one coherent project.

Member 01's data, validation, reproducibility, baseline, and Logistic Regression work provided the common technical foundation and an important reference for comparing more complex approaches.

Member 02's EDA and tree-based modeling contributed both dataset understanding and additional predictive model families.

Member 03's PCA and feature-selection work examined whether reduced or selected feature representations could remain competitive while reducing the input representation.

Member 04's HistGradientBoosting work contributed the HGB baseline, controlled hyperparameter optimization, learning-curve analysis, out-of-fold diagnostics, baseline-versus-tuned comparison, and the registered tuned candidate `M04-HGB-002`.

The team reviewed the completed model-selection decision. `M04-HGB-002` had achieved the strongest mean CV ROC-AUC and Average Precision among the eligible candidates, while other models showed advantages on different metrics. The group therefore retained the collective multi-criteria decision made on 2026-08-26 rather than interpreting HGB as universally superior.

The controlled final-test evaluation was also reviewed. The selected model achieved a final-test ROC-AUC of 0.891214 and Average Precision of 0.584385 on the 40,000 reserved observations. These results were consistent with the development evidence, and no further tuning or model reselection was performed after seeing the final-test result.

The remaining project work therefore focuses on documentation, scientific interpretation, portfolio preparation, presentation quality, and consistency between individual and group-level reporting.

## Decisions

The model-selection phase is complete and will not be reopened.

`M04-HGB-002` remains the locked final model, and the one-time reserved final-test result will be reported separately from development CV and OOF evidence.

No new model tuning or selection based on the final-test result will be performed.

Final conclusions must reflect both the strengths and limitations of the selected model and remain consistent with the recorded experimental evidence.

Documentation and logbooks must accurately reflect the work performed and distinguish individual contributions from shared group work.

Presentation sections will follow the members' responsibility areas, while the overall methodology, final model-selection decision, final-test result, limitations, and conclusions will be presented as collective project outcomes.

## Task Assignment

- **Member 01** will consolidate the common methodology, data and reproducibility description, baseline and Logistic Regression results, final model-selection and final-evaluation documentation, and corresponding presentation content.
- **Member 02** will consolidate EDA findings and tree-model results and select the most relevant visualizations and observations for the portfolio and presentation.
- **Member 03** will consolidate PCA, feature-selection, and reduced-feature comparisons and prepare the key methodological findings and presentation material.
- **Member 04** will consolidate the HistGradientBoosting baseline, optimization, learning-curve and OOF diagnostics, baseline-versus-tuned comparison, and the HGB contribution to the final model decision, and prepare the most relevant figures, tables, and presentation content.

All members will review their personal logbooks, check documentation completeness, prepare their individual portfolio material, coordinate the final scientific conclusions, and review the presentation together.

## Risks and Blockers

The main remaining risks are no longer related to model development but to reporting and communication.

These include incomplete or inconsistent documentation, mixing CV, OOF, and final-test results, overstating the superiority of the selected model, unclear distinction between individual and team contributions, excessive technical detail for the available presentation time, and inconsistent terminology or conclusions across presentation sections.

Last-minute changes to completed experimental results should be avoided so that the final documentation remains reproducible and consistent with the recorded evidence.

No technical failure currently blocks project completion.

## Changes from Initial Plan

No major change was made to the original responsibility distribution.

The project has moved from individual modeling and experimentation through collective model selection and controlled final evaluation into its final consolidation phase.

The remaining emphasis is therefore on scientific interpretation, documentation, individual portfolios, presentation preparation, and consistent communication of the completed results rather than further model development.

## Deadlines

No additional internal deadlines were set during this meeting. The remaining work before submission will prioritize:

- finalizing documentation and personal logbooks;
- completing individual portfolio material;
- preparing and reviewing presentation slides;
- checking that CV, OOF, and final-test results are clearly distinguished;
- confirming that individual and group contributions are accurately represented;
- ensuring that all final conclusions match the recorded experimental evidence.

## Next Meeting

As this is the final regular weekly project-phase meeting, any remaining synchronization will focus only on final presentation and submission review, unresolved documentation items, consistency checks, and final quality assurance.

The completed model selection and final-test evaluation will not be reopened.

## Retrospective

- **What went well:** The team progressed from a shared technical foundation to four identifiable scientific and ML work packages, completed a comparable model review, collectively selected and locked a final model, and preserved the reserved test partition until the controlled final evaluation.
- **What did not go well:** As the number of experiments, metrics, and documentation artifacts increased, maintaining a clear distinction between development evidence, final-test evidence, individual contributions, and group-level results became more demanding.
- **Improvement action:** Complete the remaining documentation and portfolio work, simplify the main scientific findings for the presentation, and conduct a final consistency review so that the repository, portfolios, and presentation communicate the same evidence and conclusions.

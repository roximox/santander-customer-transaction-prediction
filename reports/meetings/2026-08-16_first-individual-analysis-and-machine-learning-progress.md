# First Individual Analysis and Machine Learning Progress

- **Date:** 2026-08-16
- **Participants:** Member 01 — Yassine El Hari; Member 02 — Aya Olali; Member 03 — Iliase Lhamri; Member 04 — Chaymae Akouaouch.
- **Sprint goal:** Review the first concrete progress from the individual work packages, discuss methodological and technical observations, and ensure that the different ML approaches remain comparable under the common validation strategy.
- **Results from previous sprint:** The common data and validation foundation enabled more independent work. Member 01 progressed with baseline and Logistic Regression work; Member 02 moved into exploratory analysis; Member 03 started feature-selection and dimensionality-reduction work; and Member 04 progressed from the initial HistGradientBoosting baseline into optimization and diagnostic evaluation.

## Discussion

The team reviewed progress across all four responsibility areas.

Member 01 progressed the common data and validation workflow, while baseline and Logistic Regression work provided an important reference point for later model comparison.

Member 02 advanced the exploratory data analysis through investigation of dataset characteristics, distributions, variable relationships, and relevant visualizations, while tree-based modeling remained part of the planned modeling work.

Member 03 progressed feature engineering and dimensionality reduction, including PCA and feature-selection work aimed at investigating whether the feature representation could be reduced while retaining useful predictive performance.

Member 04 presented progress on the HistGradientBoosting track. An initial HGB baseline had been established, followed by a training-only hyperparameter search. The selected tuned configuration was then examined through learning-curve and out-of-fold diagnostics and compared with the baseline. The results indicated improved validation performance for the tuned configuration, while the remaining train-validation gap and the precision-recall trade-off showed that generalization and class-specific behaviour still required careful interpretation.

The team emphasized that all model tracks should remain grounded in the common validation methodology and comparable evaluation metrics before any later cross-model or final-model decision.

## Decisions

All model tracks will continue using the common validation principles. Baseline results will remain reference points for evaluating later improvements, and optimization results must be interpreted together with diagnostic evidence rather than from a single metric alone.

The HGB tuning result will remain fixed for subsequent evaluation and reporting steps rather than starting another immediate tuning cycle.

Members will continue documenting methodological decisions, limitations, and important observations.

No final model selection was performed at this meeting, and the reserved final-test partition remains untouched.

## Task Assignment

- **Member 01** will continue Logistic Regression and baseline evaluation and maintain the common validation and reproducibility foundation.
- **Member 02** will continue EDA and progress Random Forest and Extra Trees modeling while documenting relevant observations.
- **Member 03** will continue PCA and feature-selection work and prepare comparisons between feature representations using the common validation methodology.
- **Member 04** will continue evaluating and documenting the tuned HistGradientBoosting configuration, consolidate the baseline-versus-tuned evidence, and prepare the later formal experiment registration and reporting steps.

All members will document their work and ensure that results can later be compared under common evaluation principles.

## Risks and Blockers

Different preprocessing or validation choices could reduce comparability between model tracks. More complex models may increase computational cost, and optimization must avoid over-interpreting repeated validation results.

For the HGB track in particular, improved validation performance does not by itself demonstrate that overfitting has disappeared, so the observed train-validation gap and class-specific performance need to remain visible in the interpretation.

Dependencies on shared project components still require communication. No concrete technical failure was identified.

## Changes from Initial Plan

No major responsibility change was made. The project is now in a more parallel phase, with each member increasingly producing concrete results in their assigned scientific and ML area.

For Member 04, the work progressed beyond the initial HGB baseline into hyperparameter optimization and diagnostic evaluation, while remaining within the originally assigned model-optimization and evaluation responsibility.

## Deadlines

No exact deadlines were set. Before the next meeting, Member 01 will progress Logistic Regression and baseline evaluation; Member 02 will progress EDA and tree-based models; Member 03 will progress PCA and feature-selection work; and Member 04 will consolidate the tuned HGB evidence and prepare the subsequent registration and reporting work.

All members will prepare their main results, limitations, and methodological observations for later cross-model comparison.

## Next Meeting

Review the more mature results from all model tracks, check result comparability, discuss remaining methodological or technical issues, and prepare the evidence required for later cross-model comparison and final-model selection.

## Retrospective

- **What went well:** The shared project foundation enabled all four members to make increasingly independent progress, and the first concrete modeling and diagnostic results were becoming available.
- **What did not go well:** As the number and complexity of experiments increased, keeping evaluation and interpretation consistent across the different model tracks became more important.
- **Improvement action:** Keep evaluation conventions aligned, document limitations alongside performance improvements, and prepare comparable evidence before the later group-level model-selection discussion.
# Individual Portfolio — Member 04
 
**Chaymae Akouaouch**
**Individual focus:** Model Optimization and Evaluation — HistGradientBoosting
**Project:** Santander Customer Transaction Prediction
 
## Profile and Individual Focus
 
As Member 04, I was responsible for the HistGradientBoosting (HGB) modeling and optimization track. My guiding question was whether a controlled, reproducible optimization process could improve the initial HGB baseline while keeping its generalization behaviour visible throughout.
 
I built on shared project infrastructure — OpenML 45566, the stratified 80/20 split, five-fold `StratifiedKFold`, `random_state=42`, shared metrics, and the experiment framework — but none of that shared foundation was my individual implementation. My contribution was the HGB baseline, the hyperparameter search, the learning-curve and training-only OOF diagnostics, the baseline-versus-tuned comparison, the formal experiment registration, the reporting-notebook consolidation, and my participation in the collective final-model review.
 
## My Optimization Journey
 
`M04-HGB-001 (baseline) → M04-HGB-SEARCH-001 (hyperparameter search) → M04-HGB-LC-001 (learning curve) → M04-HGB-OOF-001 (OOF diagnostics) → M04-HGB-COMP-001 (baseline vs. tuned) → M04-HGB-002 (formal registration) → M04-NB-001 (notebook consolidation) → M04-SEL-001 (collective selection)`
 
On 2026-08-14, `M04-HGB-SEARCH-001` evaluated 20 candidates with five-fold training-only cross-validation and selected a configuration with a lower learning rate, more iterations, a larger minimum leaf size, and stronger L2 regularization than the baseline. That frozen configuration carried unchanged through the learning-curve, OOF, and comparison analyses. `M04-HGB-002` (2026-08-22) later registered that same configuration through the common experiment framework — a registration step, not another tuning stage. `M04-NB-001` consolidated the evidence into the reporting notebook, and on 2026-08-26 I took part in the collective review that confirmed the final candidate (`M04-SEL-001`).
 
## Key Scientific Results
 
| Metric | M04-HGB-001 baseline CV | M04-HGB-002 registered tuned CV |
|---|---:|---:|
| ROC-AUC | 0.884596 ± 0.003278 | 0.891449 ± 0.002836 |
| Average Precision | 0.572879 | 0.591089 ± 0.010028 |
| F1 | 0.387255 | 0.415248 |
| Train-validation ROC-AUC gap | 0.091063 | 0.082131 |
 
The tuned configuration improved ranking performance and narrowed the train-validation gap, though the remaining gap still called for caution.
 
`M04-HGB-OOF-001` confirmed this on training-only out-of-fold predictions, where every observation was scored by a fold model that never trained on it: OOF ROC-AUC was 0.891438 and AP was 0.590860, closely matching the CV result. At the fixed 0.5 threshold, precision was high (0.795527) but recall was limited (0.280943) — the model's positive predictions were reliable, but it missed a large share of actual positives. No threshold optimization was performed.
 
`M04-HGB-COMP-001` compared baseline and tuned evidence directly: all seven reported metrics improved, with ROC-AUC up by +0.006842 and Average Precision by +0.017981. Since the baseline uses fold-mean CV and the tuned result uses aggregated OOF, this is evidence of improvement within the HGB track rather than an independent final-test comparison.
 
**Figure 1.** [Learning curve for tuned HistGradientBoosting](../figures/M04-HGB-learning-curve.pdf).
**Figure 2.** [OOF Precision-Recall curve](../figures/M04-HGB-OOF-001_precision_recall_curve.pdf).
**Figure 3.** [Baseline-versus-tuned HGB comparison](../figures/M04-HGB-model-comparison.pdf).
 
## Evaluation and Scientific Interpretation
 
The learning curve showed that more development data improved validation performance: ROC-AUC rose from 0.851063 at 12,800 effective training rows to 0.891245 at 128,000, while the ROC-AUC gap dropped from 0.138206 to 0.082098 over the same range. This points to improving generalization with more data, though it doesn't prove overfitting disappeared entirely.
 
ROC-AUC was the main ranking metric, but Average Precision and the precision-recall trade-off mattered just as much given the imbalanced target — a model can rank well overall while still missing a large share of positive cases at a fixed threshold. Throughout development, tuning, and diagnostics, the 40,000-row reserved test partition stayed untouched; it was opened only after the group froze `M04-HGB-002` on 2026-08-26.
 
## Challenges and Lessons Learned
 
The main challenge was distinguishing a better development estimate from a universally better model. Before this work, I mainly associated model optimization with improving validation scores. Through the HGB experiments, I learned that optimization also requires checking generalization gaps, class-specific behaviour, leakage risks, and reproducibility — not chasing a single score.
 
## Contribution to the Final Model Decision
 
I contributed the full HGB evidence chain — baseline, tuned configuration, learning curve, OOF diagnostics, comparison, and formal registration — and took part in the collective final-model review on 2026-08-26.
 
The group compared 11 eligible candidates before opening the reserved test partition. `M04-HGB-002` was collectively selected after the group-level comparison. It achieved the best mean CV ROC-AUC (0.891449) and Average Precision (0.591089), while other candidates led on different metrics — Extra Trees had the highest F1, and a balanced Logistic Regression candidate had the highest recall. The decision was multi-criteria and pragmatic, not a claim that HGB was best on every metric. I did not implement the group selection framework, the final model lock, or the final-evaluation pipeline myself.
 
After the collective model lock, `M04-HGB-002` was evaluated once on the reserved 40,000-row final-test partition through the group workflow. It achieved ROC-AUC 0.891214 and AP 0.584385, closely matching the development evidence (CV ROC-AUC 0.891449). The final test was not used to reopen model selection or perform further tuning.
 
## Reproducibility and Main Deliverables
 
My deliverables include reusable HGB components in `src/`, reproducible HGB scripts in `scripts/`, the reporting notebook [`notebooks/07_gradient_boosting.ipynb`](../../notebooks/07_gradient_boosting.ipynb), the persisted HGB experiment, search, and diagnostic artifacts, dedicated tests for the HGB workflow, and the chronological [Member 04 logbook](../logbook/member_04/README.md).
 
## Skills Demonstrated
 
Python, scikit-learn, HistGradientBoosting, RandomizedSearchCV, cross-validation, learning curves, OOF evaluation, imbalanced classification, ROC/Precision-Recall analysis, reproducible experimentation, scientific interpretation, and collaborative model review.
 
## Sources, Tools and AI Usage
 
**Sources and tools:** official ADA project specification; Santander Customer Transaction Prediction dataset accessed through OpenML dataset 45566; Python, NumPy, pandas, scikit-learn, matplotlib, Jupyter, pytest, and Git/GitHub.
 
**AI tools:** ChatGPT and Codex were used as supporting tools for conceptual explanations, documentation, and code review.
 

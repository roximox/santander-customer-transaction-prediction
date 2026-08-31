# Individual Portfolio — Member 04

**Chaymae Akouaouch**
**Individual focus:** Model Optimization & Evaluation — HistGradientBoosting
**Project:** Santander Customer Transaction Prediction
**Course:** Advanced Data Analytics / Scientific Programming and Data Analysis


## Profile and Individual Focus

As Member 04, I was responsible for the Gradient Boosting track of the project: building a reproducible HistGradientBoosting baseline, improving it through controlled optimization, and interpreting its behaviour beyond a single score. My guiding question was whether a systematically optimized HistGradientBoosting model can improve predictive performance over its initial baseline while maintaining acceptable generalization performance.

The task itself was an imbalanced binary classification problem, so ROC-AUC served as the primary metric, complemented by Average Precision (AP) to judge how well the model ranks the rare positive class. I worked on top of infrastructure shared across the group — OpenML 45566, the stratified 80/20 split, five-fold `StratifiedKFold`, `random_state=42`, shared metrics, and the experiment framework. None of that was my individual work. What I contributed was the HGB-specific baseline, the hyperparameter search, the tuned configuration, the learning-curve and OOF diagnostics, the comparison between baseline and tuned model, and the interpretation behind all of it.

## My Optimization Journey

`Baseline → Hyperparameter Optimization → Tuned Configuration → Learning Curve → OOF Diagnostics → Baseline-vs-Tuned Comparison`

I started with M04-HGB-001 to establish a proper reference point before doing any tuning. The baseline used `learning_rate=0.1`, `max_iter=300`, `max_leaf_nodes=31`, `l2_regularization=0.0`, and `random_state=42`. Its five-fold ROC-AUC came out to 0.884596 ± 0.003278, while the train ROC-AUC of 0.975659 left a gap of 0.091063 — large enough to warrant a closer look at generalization.

Next, M04-HGB-SEARCH-001 ran a training-only `RandomizedSearchCV` over 20 candidates, five folds, and 100 fits, selecting on ROC-AUC. The best configuration was `learning_rate=0.05`, `max_iter=700`, `max_leaf_nodes=31`, `min_samples_leaf=100`, `l2_regularization=10.0`, and `random_state=42`. A lower learning rate paired with more iterations, stronger regularization, and a larger minimum leaf size points toward a more gradual, regularized fit — though the experiment shows an association, not proof of parameter-level causality.

M04-HGB-002 didn't run a second search. Instead, it registered the frozen candidate through the same shared framework used for the baseline, allowing both experiments to be evaluated within the same cross-validation record. From there, the learning curve checked whether validation performance kept improving with more data. OOF predictions then gave every development observation a prediction from a model that had never seen it during training, which let me compute one coherent set of diagnostic metrics and plots. The final comparison pulled these results together without fitting anything new.

## Key Scientific Results

| Metric | Baseline CV | Tuned OOF | Change |
|---|---:|---:|---:|
| ROC-AUC | 0.884596 | 0.891438 | +0.006842 |
| Average Precision | 0.572879 | 0.590860 | +0.017981 |
| F1 | 0.387255 | 0.415242 | +0.027986 |
| Precision | 0.782671 | 0.795527 | +0.012856 |
| Recall | 0.257307 | 0.280943 | +0.023636 |
| Accuracy | 0.918181 | 0.920488 | +0.002306 |
| Balanced Accuracy | 0.624658 | 0.636438 | +0.011780 |

The registered tuned CV result was ROC-AUC **0.891449 ± 0.002836**, AP **0.591089 ± 0.010028**, F1 0.415248, precision 0.795747, recall 0.280942, accuracy 0.920487, and balanced accuracy 0.636438. Train ROC-AUC was 0.973580, reducing the train-validation ROC-AUC gap to 0.082131. All seven reported metrics improved relative to the baseline.

**Figure 1.** [Learning curve for tuned HistGradientBoosting](../figures/M04-HGB-learning-curve.pdf). Validation ROC-AUC rose from 0.851063 at 12,800 effective training rows to 0.891245 at 128,000; validation AP rose from 0.475333 to 0.589899. The ROC-AUC gap dropped from 0.138206 to 0.082098 over the same range.

**Figure 2.** [OOF Precision-Recall curve](../figures/M04-HGB-OOF-001_precision_recall_curve.pdf).
**Figure 3.** [Baseline-versus-tuned HGB comparison](../figures/M04-HGB-model-comparison.pdf).
**Supplementary evidence:** [OOF confusion matrix](../figures/M04-HGB-OOF-001_confusion_matrix.pdf).

## Evaluation and Scientific Interpretation

ROC-AUC measures ranking quality across thresholds, and AP matters here in particular because positive cases are rare. Both improved with tuning, yet the OOF diagnostics demonstrate why model quality should not be assessed using a single metric. At an OOF ROC-AUC of 0.891438 and AP of 0.590860, the model's precision at the unchanged 0.5 threshold was 0.795527 and recall only 0.280943. The confusion matrix (TN=142,761, FP=1,161, FN=11,561, TP=4,517) confirms this: positive predictions were fairly reliable, but a large share of positive cases still went undetected.

The learning curve supports a cautiously optimistic reading of generalization: more data improved both validation ROC-AUC and AP and narrowed the train-validation gap, though the gap never closed entirely. No additional threshold optimization was performed, so the reported precision-recall trade-off reflects the predefined threshold of 0.5. Leakage prevention mattered just as much: every step of HGB optimization, cross-validation, learning-curve analysis, and OOF evaluation used only the 160,000-row development partition. The 40,000-row reserved test partition was never touched or used in any Member 04 decision.

## Challenges and Lessons Learned

A central challenge was evaluating model improvement under class imbalance. ROC-AUC alone was not sufficient, which made Average Precision and the precision-recall trade-off particularly important for judging the tuned model fairly. The learning curve also showed that improved validation performance does not automatically mean overfitting has disappeared, since a train-validation gap remained even after more data was added.

The main lesson from this work was that model optimization is not only about finding better hyperparameters. Reliable validation, leakage prevention, and the interpretation of multiple metrics together are equally important for judging whether an improvement is scientifically meaningful.

## Contribution to the Final Model Decision

The optimized M04-HGB-002 model achieved the strongest performance among the compared candidates on the main ranking metrics, particularly ROC-AUC and Average Precision. Based on the final group-level comparison, the team selected M04-HGB-002 as the final model. My contribution to this decision consisted of developing the HistGradientBoosting baseline, systematically optimizing it, and providing the learning-curve and OOF diagnostics used to assess its performance and generalization.

The final selection was therefore based not only on a single performance score, but on the combined evidence from cross-validation, class-imbalance-aware metrics, generalization analysis, and comparison with the other candidate models.

## Reproducibility and Main Deliverables

My contribution was implemented as reusable HGB components in `src/`, reproducible experiment workflows in `scripts/`, and scientific reporting in [`notebooks/07_gradient_boosting.ipynb`](../../notebooks/07_gradient_boosting.ipynb). Persisted artifacts include experiment summaries, learning-curve and OOF diagnostics, comparison tables, and figures. Dedicated tests cover the main Member 04 workflow components.

The chronological development and decisions are documented in the [Member 04 logbook](../logbook/member_04/README.md).

## Skills Demonstrated

Python, scikit-learn, HistGradientBoosting, hyperparameter optimization, RandomizedSearchCV, cross-validation, learning curves, out-of-fold evaluation, imbalanced classification, ROC/PR analysis, reproducible machine learning, scientific interpretation, and Git.

## Sources, Tools and AI Usage

**Sources and tools used:** official ADA project specification; OpenML dataset 45566 through the project's scikit-learn loader; Santander Customer Transaction Prediction / Kaggle competition dataset; Python, NumPy, pandas, scikit-learn, matplotlib, Jupyter, pytest, and Git/GitHub.

**AI tools:** ChatGPT and Codex were used as supporting tools for conceptual explanations, documentation, and code review.


# Scientific Conclusions

## Study objective and protocol

The project compared candidate models for the Santander Customer Transaction
Prediction task under a shared protocol. Model development and selection used
only the 160,000-row development partition. All eligible candidates used
five-fold `StratifiedKFold`, `random_state=42`, and ROC-AUC as the primary
selection metric. The 40,000-row final partition remained reserved until the
team collectively selected and locked one pipeline.

## Model-family comparison

The final portfolio covered Logistic Regression, Decision Tree, Random Forest,
Extra Trees, feature selection, PCA, and HistGradientBoosting. Logistic
Regression provided a stable and computationally efficient reference. Feature
selection and PCA did not materially improve ranking performance in their
recorded configurations. Decision Tree was the weakest ranking model, while
Random Forest improved on the individual tree. Extra Trees produced the highest
recorded mean CV F1, but not the strongest ranking metrics.

`M04-HGB-002` (`HistGradientBoosting Tuned`) led the eligible portfolio on the
two ranking metrics prioritized for this imbalanced problem: mean CV ROC-AUC
and mean CV Average Precision. The team therefore selected this model before
opening the final partition. Its estimator parameters, `random_state=42`, and
classification threshold of 0.5 were frozen in
`reports/model_selection/final_model_lock.json`.

## Cross-validation and final-test results

| Metric | CV mean | CV standard deviation | Final test | Final minus CV |
|---|---:|---:|---:|---:|
| ROC-AUC | 0.891449 | 0.002836 | 0.891214 | -0.000235 |
| Average Precision | 0.591089 | 0.010028 | 0.584385 | -0.006704 |
| F1 | 0.415248 | 0.007033 | 0.403632 | -0.011616 |
| Precision | 0.795747 | 0.018308 | 0.791424 | -0.004323 |
| Recall | 0.280942 | 0.004640 | 0.270896 | -0.010047 |
| Balanced Accuracy | 0.636438 | 0.002587 | 0.631459 | -0.004978 |

The final ROC-AUC differs from the CV mean by only 0.000235. Average Precision
is 0.006704 below its CV mean. These small descriptive differences support the
conclusion that the locked model's ranking performance transferred well to the
held-out partition. They do not constitute a formal hypothesis test.

At the frozen threshold of 0.5, the final confusion matrix contains 35,693 true
negatives, 287 false positives, 2,931 false negatives, and 1,089 true positives.
The resulting precision is high (0.791424), but recall remains limited
(0.270896). The model is therefore more conservative in assigning the positive
class and misses a substantial share of positive cases.

## Final conclusion

The final result supports the collective choice of `M04-HGB-002`: it preserves
the strong ranking performance observed during cross-validation without a
material ROC-AUC deterioration on unseen data. The result does not justify
reopening selection, changing the estimator, or tuning the threshold from the
final-test labels.

The main scientific limitations remain the larger recorded train-validation
ROC-AUC gap of the HGB model, its higher computational cost than Logistic
Regression, and its low recall at the frozen operating threshold. Any future
threshold change must be motivated by an independently specified operational
cost and validated on new data, not on this final partition.

## Reproducibility and final-test policy

The final evaluation was executed exactly once as
`FINAL-M04-HGB-002-001`. The result is stored in
`reports/final_evaluation/M04-HGB-002_final_test_results.json`. Model selection
was not reopened, and the execution guard now prohibits a second final-test run.
No prediction-level final-test data were persisted.

# Logbook Entry
 
## Metadata
 
**Date:** 2026-08-14
**Member:** Chaymae Akouaouch
**Phase:** Model optimization and hyperparameter tuning
**Ticket ID:** M04-HGB-SEARCH-001
**Branch:** feature/model-optimization
**Related meeting:** 2026-08-09 — Data Processing, Validation Strategy and Start of Individual Analysis
 
## Title
 
HistGradientBoosting Hyperparameter Optimization
 
## Context and Goal
 
After establishing the `M04-HGB-001` baseline, I continued the Member 04 model-optimization track, following the validation strategy the group agreed on at the meeting on 2026-08-09.
 
The baseline reached a validation ROC-AUC of 0.884596 and Average Precision of 0.572879, with a train ROC-AUC of 0.975659 that left a train-validation gap of 0.091063.
 
My goal for this step was to see whether controlled hyperparameter optimization could improve validation performance while also narrowing that generalization gap. As before, the reserved final-test partition stayed outside the optimization and model-selection process.
 
## Search Design
 
I went with `RandomizedSearchCV` rather than an exhaustive `GridSearchCV`. The predefined HGB search space had 768 possible parameter combinations, so a randomized search let me evaluate a bounded, reproducible subset while keeping the same cross-validation protocol.
 
The search space was:
 
- `learning_rate`: [0.03, 0.05, 0.1, 0.15]
- `max_iter`: [150, 300, 500, 700]
- `max_leaf_nodes`: [15, 31, 63]
- `min_samples_leaf`: [10, 20, 50, 100]
- `l2_regularization`: [0.0, 0.1, 1.0, 10.0]
The search evaluated 20 candidates using five-fold stratified cross-validation, for 100 total fits. I used `random_state=42` for reproducibility and `n_jobs=1`.
 
ROC-AUC stayed the primary selection metric, with the common project metrics tracked alongside it for additional evaluation.
 
## Implementation
 
I created `src/gradient_boosting_search.py` to define the HGB search space, build the unfitted `RandomizedSearchCV`, convert the fitted search results into a candidate-level DataFrame, and produce a JSON-compatible summary.
 
I also created `scripts/run_gradient_boosting_search.py` to reproduce the shared split, verify the dataset fingerprints, run the search on the development partition, summarize the results, and persist the search artifacts.
 
Before writing any of this, I looked at the existing shared `src/search.py` implementation. Its result-conversion logic turned out to be specific to the Logistic Regression workflow, so rather than adapting it, I kept it untouched and wrote the HGB-specific conversion in the Member 04 module, reusing the existing generic persistence helper where it still applied.
 
## Work Performed
 
1. Reviewed the existing shared search infrastructure.
2. Identified that the existing result-conversion logic was specific to the Logistic Regression workflow.
3. Decided to leave the shared implementation unchanged and build a separate HGB-specific search module.
4. Defined a bounded HistGradientBoosting search space.
5. Configured a deterministic `RandomizedSearchCV`.
6. Implemented HGB-specific candidate-result conversion and summary helpers.
7. Created the reproducible HGB search runner.
8. Verified the search configuration before execution.
9. Executed `python scripts/run_gradient_boosting_search.py`.
10. Reviewed the 20 evaluated candidates and identified the best one by validation ROC-AUC.
11. Compared the selected candidate with `M04-HGB-001`.
12. Confirmed that the reserved final-test partition was not used for fitting, tuning, model selection, or scoring.
## Results
 
The search selected `candidate_011`.
 
The selected configuration was:
 
- `learning_rate=0.05`
- `max_iter=700`
- `max_leaf_nodes=31`
- `min_samples_leaf=100`
- `l2_regularization=10.0`
- `random_state=42`
The selected candidate achieved:
 
- Validation ROC-AUC: 0.891449
- Average Precision: 0.591089
- Train ROC-AUC: 0.973580
- Generalization gap: 0.082131
The search evaluated 20 candidates across five folds, for 100 model fits in total.
 
## Baseline Comparison
 
| Metric | M04-HGB-001 Baseline | M04-HGB-SEARCH-001 Best Candidate | Change |
|---|---:|---:|---:|
| Validation ROC-AUC | 0.884596 | 0.891449 | +0.006853 |
| Average Precision | 0.572879 | 0.591089 | +0.018210 |
| Train ROC-AUC | 0.975659 | 0.973580 | -0.002079 |
| Generalization gap | 0.091063 | 0.082131 | -0.008932 |
 
## Interpretation and Decision
 
The selected candidate improved validation ROC-AUC from 0.884596 to 0.891449 and Average Precision from 0.572879 to 0.591089 compared with the baseline.
 
At the same time, train ROC-AUC dropped slightly while validation ROC-AUC rose, narrowing the train-validation ROC-AUC gap from 0.091063 to 0.082131. I take this as evidence of improved generalization relative to the baseline, though not as proof that overfitting has been eliminated.
 
The selected configuration pairs a lower learning rate with more boosting iterations, a larger minimum leaf size, and stronger L2 regularization — settings that fit the picture of a slower, more regularized model. That said, the search results don't establish the individual causal contribution of each hyperparameter on their own.
 
It's also worth noting that this candidate was the best among the 20 evaluated, not necessarily the global optimum across the full search space.
 
Given these results, I froze this configuration for the next diagnostic step instead of running another search. The next question was whether its validation behaviour held up as the amount of training data increased.
 
## Difficulties and Observations
 
One design decision was whether to modify the existing shared search infrastructure or build HGB-specific handling instead. Since part of the existing result conversion was tied to the Logistic Regression workflow, I left the shared implementation alone and kept the HGB-specific logic isolated in the Member 04 module.
 
Balancing search coverage against computational cost was another consideration. The full predefined search space had 768 combinations; sampling 20 reproducible candidates kept the workload manageable, but it also means the selected configuration can't be claimed as the global optimum.
 
## Generated Artifacts
 
- `reports/searches/M04-HGB-SEARCH-001_candidates.csv`
- `reports/searches/M04-HGB-SEARCH-001_summary.json`
## Reproducibility
 
The search used the shared train/test split, verified dataset fingerprints, `random_state=42`, and five-fold stratified cross-validation.
 
Only development data was used for hyperparameter optimization and candidate selection. The reserved final-test partition was not used for model fitting, hyperparameter tuning, candidate selection, or scoring.
 
## Next Step
 
Use the selected HistGradientBoosting configuration in a training-only learning-curve analysis to investigate validation performance and the train-validation gap as the amount of training data increases.
 

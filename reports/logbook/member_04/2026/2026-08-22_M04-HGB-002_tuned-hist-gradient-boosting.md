# Logbook Entry
 
## Metadata
 
**Date:** 2026-08-22
**Member:** Chaymae Akouaouch
**Phase:** Experiment consolidation and registration
**Ticket ID:** M04-HGB-002
**Branch:** feature/model-optimization
**Related meeting:** 2026-08-16 — First Individual Analysis and Machine Learning Progress
 
## Title
 
Registered Tuned HistGradientBoosting Experiment
 
## Context and Goal
 
The tuned HistGradientBoosting configuration was already selected in `M04-HGB-SEARCH-001` on 2026-08-14 and had since been used for both the learning-curve and OOF diagnostic work.
 
What this step adds is formal registration: I ran that same frozen configuration as `M04-HGB-002` through the common experiment framework used for the original `M04-HGB-001` baseline, so it produces the standard fold-level and summary artifacts and shows up consistently in the shared experiment registry.
 
This was not another hyperparameter search or a new tuned model — the parameters themselves don't change here. It's the point where the already-selected configuration gets formally logged alongside the other project experiments, which is why it appears more than a week after the search that actually found it.
 
## Tuned Configuration
 
The following parameters had already been selected by `M04-HGB-SEARCH-001` on 2026-08-14:
 
- `learning_rate = 0.05`
- `max_iter = 700`
- `max_leaf_nodes = 31`
- `min_samples_leaf = 100`
- `l2_regularization = 10.0`
- `random_state = 42`
The configuration stayed unchanged throughout `M04-HGB-002`. None of these parameters were selected based on the `M04-HGB-002` results — that selection happened earlier, on 2026-08-14.
 
## Method
 
- Reused the common project dataset loader.
- Recreated the common 80/20 train/reserved-test split.
- Verified the training fingerprint.
- Verified the reserved-test fingerprint.
- Removed the reserved-test objects before evaluation.
- Evaluated only the 160,000-row development/training partition.
- Reused the common five-fold `StratifiedKFold` cross-validation strategy with `random_state = 42`.
- Reused the common project evaluation metrics.
- Used `n_jobs = 1`.
- Evaluated and persisted the experiment through `run_and_save_experiment()`.
- Performed no new hyperparameter tuning.
- Performed no threshold tuning.
- Performed no final-test evaluation.
## Work Performed
 
1. Reused the configuration previously selected by `M04-HGB-SEARCH-001`.
2. Kept all tuned hyperparameters unchanged.
3. Recreated and verified the common project data split.
4. Removed the reserved-test objects before model evaluation.
5. Evaluated the frozen HGB configuration using the common five-fold CV experiment framework.
6. Generated the standard fold-level experiment results.
7. Generated the standard experiment summary.
8. Registered `M04-HGB-002` in the common experiment registry.
9. Compared the registered metrics with the earlier search result as a consistency check.
10. Confirmed that this registration step introduced no new tuning and no final-test decision.
## Results
 
| Metric | Value |
|---|---:|
| Validation ROC-AUC | 0.891449 ± 0.002836 |
| Validation Average Precision | 0.591089 ± 0.010028 |
| F1 | 0.415248 |
| Precision | 0.795747 |
| Recall | 0.280942 |
| Accuracy | 0.920487 |
| Balanced Accuracy | 0.636438 |
| Train ROC-AUC | 0.973580 |
| Generalization gap | 0.082131 |
 
## Fold Results
 
| Fold | ROC-AUC | Average Precision |
|---:|---:|---:|
| 1 | 0.887573 | 0.586082 |
| 2 | 0.891348 | 0.586814 |
| 3 | 0.896350 | 0.609012 |
| 4 | 0.890377 | 0.579664 |
| 5 | 0.891598 | 0.593873 |
 
## Interpretation
 
The registered experiment produced a validation ROC-AUC of 0.891449 and Average Precision of 0.591089 — consistent with the best configuration identified earlier in `M04-HGB-SEARCH-001`.
 
That consistency is expected: `M04-HGB-002` doesn't introduce another optimization stage, it evaluates the same frozen configuration through the common experiment framework, more than a week after that configuration was originally chosen.
 
For reference, the registered tuned result also remained better than the original Member 04 baseline:
 
| Metric | M04-HGB-001 Baseline | M04-HGB-002 Tuned |
|---|---:|---:|
| ROC-AUC | 0.884596 | 0.891449 |
| Average Precision | 0.572879 | 0.591089 |
| F1 | 0.387255 | 0.415248 |
 
These are training-only cross-validation results and do not represent final-test performance.
 
At this stage, `M04-HGB-002` is the formally registered Member 04 candidate. No group-level final-model decision was made as part of this registration step.
 
## Relationship to Previous Member 04 Work
 
The chronological workflow was:
 
`M04-HGB-001`
→ established the initial HGB baseline on 2026-08-10.
 
`M04-HGB-SEARCH-001`
→ selected and froze the tuned configuration on 2026-08-14.
 
`M04-HGB-LC-001`
→ analyzed the learning behaviour of that frozen configuration on 2026-08-14, using the parameters `SEARCH-001` had already selected.
 
`M04-HGB-OOF-001`
→ generated training-only OOF diagnostics for the same configuration on 2026-08-15, again without touching the parameters.
 
`M04-HGB-COMP-001`
→ compared the tuned OOF results with the original HGB baseline on 2026-08-15.
 
`M04-HGB-002`
→ formally registered the already selected configuration through the common experiment framework on 2026-08-22.
 
The gap between 2026-08-14 and 2026-08-22 reflects when this configuration was registered as a standalone experiment record, not when it was selected. The parameters, the learning-curve analysis, the OOF diagnostics, and the baseline comparison were all already based on the `SEARCH-001` configuration well before this registration step. `M04-HGB-002` didn't replace or re-trigger any of that earlier work — it consolidated the already-selected tuned configuration in the shared experiment registry so it could sit alongside the other registered project experiments.
 
## Generated Artifacts
 
- `reports/experiments/M04-HGB-002_fold_results.csv`
- `reports/experiments/M04-HGB-002_summary.json`
- One `M04-HGB-002` entry in `reports/experiments/experiment_registry.csv`
## Difficulties and Observations
 
The main documentation challenge was keeping "selecting the tuned configuration" and "formally registering it" clearly separate.
 
The hyperparameters had already been chosen during `M04-HGB-SEARCH-001` and used in all the subsequent diagnostic work, so I documented `M04-HGB-002` purely as a registration and standardized-evaluation step, not as another tuning experiment.
 
That distinction matters for keeping the experimental chronology clear — without it, a reader could easily assume the later registration date meant the parameters were only decided on 2026-08-22, when in fact they had already been driving the learning-curve, OOF, and comparison work for over a week.
 
## Reproducibility and Leakage Prevention
 
- The tuned configuration was fixed before this experiment.
- `random_state = 42`.
- Common five-fold stratified cross-validation was reused.
- Shared train and reserved-test fingerprints were verified.
- Only the development/training partition was evaluated.
- The reserved final-test partition was not evaluated.
- No final-test prediction or metric was produced.
- No new hyperparameter tuning was performed.
- No threshold tuning was performed.
- No group-level model selection was performed during this step.
## Conclusion
 
`M04-HGB-002` formally registered the previously selected tuned HistGradientBoosting configuration in the common experiment framework.
 
The experiment confirmed that the frozen configuration could be represented consistently alongside the other registered project experiments, with the reserved final-test partition kept outside this evaluation.
 
The registered tuned HGB result was therefore ready to serve as the Member 04 candidate in the later group-level model-selection process.
 

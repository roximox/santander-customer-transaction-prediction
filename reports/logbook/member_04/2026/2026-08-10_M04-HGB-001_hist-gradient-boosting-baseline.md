# Logbook Entry
 
## Metadata
 
**Date:** 2026-08-10
**Member:** Chaymae Akouaouch
**Phase:** Individual analysis and initial modeling
**Ticket ID:** M04-HGB-001
**Branch:** feature/model-optimization
**Related meeting:** 2026-08-09 — Data Processing, Validation Strategy and Start of Individual Analysis
 
## Title
 
HistGradientBoosting Baseline
 
## Context and Goal
 
At the group meeting on 2026-08-09, we agreed on a shared train/test split and validation strategy for the individual modeling tracks, with the reserved final-test partition staying untouched throughout development and optimization.
 
Building on that, my goal for this session was to set up the first reproducible Member 04 HistGradientBoosting baseline on top of the shared project infrastructure. This baseline would give me something concrete to compare against once I started tuning hyperparameters.
 
## Implementation
 
I created `src/gradient_boosting.py` for experiment `M04-HGB-001` with an initial HistGradientBoosting configuration:
 
- `learning_rate=0.1`
- `max_iter=300`
- `max_leaf_nodes=31`
- `l2_regularization=0.0`
- `random_state=42`
The model reuses the shared `create_hist_gradient_boosting_classifier()` factory from `src/modeling.py`. I kept the scikit-learn defaults `min_samples_leaf=20` and `early_stopping="auto"` unchanged.
 
I also wrote `scripts/run_gradient_boosting_baseline.py` to wire the Member 04 model configuration into the shared dataset loading, split, fingerprint verification, cross-validation, metrics, and experiment infrastructure.
 
## Work Performed
 
1. Reviewed the shared project structure and existing modeling infrastructure.
2. Checked the common HistGradientBoosting factory and confirmed it returns an unfitted estimator rather than a full experiment setup.
3. Defined the Member 04 baseline configuration as `M04-HGB-001`.
4. Built the reproducible baseline experiment runner.
5. Verified the estimator configuration with `get_params()`.
6. Ran the five-fold stratified cross-validation experiment.
7. Reviewed the validation metrics and fold-level results.
8. Compared training and validation ROC-AUC to get a first read on generalization.
9. Confirmed that only the development partition entered cross-validation and that the reserved final-test partition was not used for model fitting, tuning, or scoring.
10. Tracked down why the generated experiment artifacts weren't showing up in `git status`, and found the existing `.gitignore` rule for `reports/experiments/*`.
## Results
 
Experiment `M04-HGB-001` produced:
 
| Metric | Result |
|---|---:|
| Validation ROC-AUC | 0.884596 ± 0.003278 |
| Average Precision | 0.572879 ± 0.009277 |
| F1 | 0.387255 |
| Precision | 0.782671 |
| Recall | 0.257307 |
| Accuracy | 0.918181 |
| Balanced Accuracy | 0.624658 |
| Train ROC-AUC | 0.975659 |
| Generalization gap | 0.091063 |
 
The fold-level ROC-AUC values were:
 
| Fold | ROC-AUC | Average Precision |
|---:|---:|---:|
| 1 | 0.879752 | 0.566940 |
| 2 | 0.885401 | 0.568957 |
| 3 | 0.889694 | 0.589841 |
| 4 | 0.882831 | 0.563554 |
| 5 | 0.885303 | 0.575105 |
 
## Interpretation and Decision
 
The first HistGradientBoosting baseline already reached a validation ROC-AUC of 0.884596, and the results stayed fairly stable across the five folds.
 
At the same time, the train ROC-AUC of 0.975659 versus the validation ROC-AUC of 0.884596 left a generalization gap of 0.091063 — large enough that I wanted to look at generalization more carefully before treating this as a final candidate.
 
Precision also came in noticeably higher than recall. Since the target is imbalanced, I decided that later evaluation should look at Average Precision and precision-recall behaviour alongside ROC-AUC, not ROC-AUC on its own.
 
Given these results, the sensible next step was controlled hyperparameter optimization, rather than adding model complexity without evidence to support it.
 
## Difficulties and Observations
 
The main technical hiccup was that the generated Member 04 experiment artifacts didn't show up in `git status` at first. Digging into the repository configuration, I found that `reports/experiments/*` was already covered by a `.gitignore` rule. I left `.gitignore` unchanged and deferred the artifact-versioning decision to a later repository-integration step.
 
On the methodology side, the train-validation gap is worth flagging but isn't conclusive on its own — I'm treating it as motivation for the upcoming optimization and learning-curve analysis rather than as proof of overfitting.
 
## Generated Artifacts
 
- `reports/experiments/M04-HGB-001_fold_results.csv`
- `reports/experiments/M04-HGB-001_summary.json`
- update to `reports/experiments/experiment_registry.csv`
## Reproducibility
 
I verified the shared train/test split and expected dataset fingerprints before running the experiment. Only the development partition was used in the five-fold stratified cross-validation.
 
The reserved final-test partition stayed outside the Member 04 model development workflow, in line with the validation strategy the group agreed on.
 
## Next Step
 
Perform controlled hyperparameter optimization for HistGradientBoosting using training-only cross-validation and compare the resulting configuration with `M04-HGB-001`.
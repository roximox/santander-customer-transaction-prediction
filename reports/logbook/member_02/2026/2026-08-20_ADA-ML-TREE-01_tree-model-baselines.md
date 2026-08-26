# Logbook Entry

## Metadata

- Date: 2026-08-20
- Member: Member 02
- Sprint: To be completed by Member 02
- Ticket ID: ADA-ML-TREE-01
- Branch: feature/eda+tree_models
- Pull Request: To be updated after Pull Request creation
- Time spent: To be completed by Member 02
- Related meeting: To be completed by Member 02

## Title

Decision Tree and Random Forest training-only baselines

## Scientific question

Does a Random Forest provide stronger and more stable discrimination than a
depth-limited Decision Tree under the shared cross-validation protocol?

## Pipeline and protocol

Experiments `M02-DT-001` and `M02-RF-001` used the shared OpenML loader, the
explicit `float32` memory optimization, and the official stratified 80/20 split.
The train fingerprint was
`61c403ec521d15ab9d6316606eba5acdfc22381cb764b6da76a65041ec11f477` and
the reserved-test fingerprint was
`bf7d43e492967dad1358676e9bf8355a910823077b59118014db12af3e26f586`.
Only the 160,000 training rows entered the shared shuffled five-fold
`StratifiedKFold` evaluation. The final test partition was not evaluated.

The Decision Tree used `max_depth=5`, `class_weight="balanced"`, and
`random_state=42`. The Random Forest was created with the shared factory and
used `n_estimators=200`, `max_depth=8`, `max_features="sqrt"`,
`class_weight="balanced"`, `random_state=42`, and `n_jobs=-1`. No scaling,
feature selection, threshold tuning, or hyperparameter search was performed.

## Results

| Model | Metric | Train mean | Validation mean | Validation std |
|---|---|---:|---:|---:|
| Decision Tree | ROC-AUC | 0.650853 | 0.633935 | 0.003119 |
| Decision Tree | Average Precision | 0.187590 | 0.163973 | 0.002675 |
| Decision Tree | F1 | 0.246164 | 0.237988 | 0.003366 |
| Decision Tree | Precision | 0.157048 | 0.151806 | 0.004748 |
| Decision Tree | Recall | 0.572645 | 0.554300 | 0.034961 |
| Decision Tree | Accuracy | 0.647242 | 0.642962 | 0.027270 |
| Decision Tree | Balanced accuracy | 0.614110 | 0.603584 | 0.003300 |
| Random Forest | ROC-AUC | 0.868594 | 0.793736 | 0.002392 |
| Random Forest | Average Precision | 0.605988 | 0.369114 | 0.006802 |
| Random Forest | F1 | 0.451497 | 0.368647 | 0.002791 |
| Random Forest | Precision | 0.323146 | 0.261355 | 0.002962 |
| Random Forest | Recall | 0.749052 | 0.625451 | 0.002073 |
| Random Forest | Accuracy | 0.817102 | 0.784700 | 0.002939 |
| Random Forest | Balanced accuracy | 0.786878 | 0.713970 | 0.001351 |

Random Forest validation ROC-AUC by fold was 0.792238, 0.792798, 0.798432,
0.791942, and 0.793268. Decision Tree validation ROC-AUC by fold was 0.633601,
0.631447, 0.633458, 0.639867, and 0.631305.

## Interpretation and limitations

The Random Forest improved mean validation ROC-AUC by 0.159801 and Average
Precision by 0.205141 over the Decision Tree. Its smaller ROC-AUC standard
deviation also indicates stable ranking performance across these folds. The
Random Forest train-to-validation ROC-AUC gap was 0.074858, compared with
0.016918 for the Decision Tree, so the ensemble exhibits more overfitting even
though its validation performance remains substantially stronger.

These are baseline comparisons, not optimized or final results. The experiment
does not establish that the selected depths, number of trees, class weights, or
classification threshold are optimal. No causal interpretation is made.

## Outputs and decision

Keep `M02-DT-001` as an interpretable tree baseline and `M02-RF-001` as the
stronger tree-ensemble baseline. Any tuning must use a new search identifier
and training data only. The final test set remains closed.

## Difficulties

Five-fold evaluation of 200 Random Forest trees is computationally expensive.
Class imbalance also makes accuracy insufficient, so ROC-AUC, Average
Precision, recall, and balanced accuracy were reported together.

## Adaptations and deviations from the plan

The initial notebook evaluated fitted models on the final test partition. That
step was removed and replaced by the shared training-only experiment runner.
The shared Random Forest factory was used; no Decision Tree factory currently
exists in `src.modeling`, so its constructor remains explicit.

## Rejected approaches

Direct OpenML loading in the notebook, test-set model comparison, hard-coded
result claims, overwriting experiment IDs, and unregistered scientific results
were rejected.

## Files changed

- `notebooks/02_eda.ipynb`
- `notebooks/04_tree_models.ipynb`
- `scripts/run_tree_models.py`
- `reports/experiments/M02-DT-001_fold_results.csv`
- `reports/experiments/M02-DT-001_summary.json`
- `reports/experiments/M02-RF-001_fold_results.csv`
- `reports/experiments/M02-RF-001_summary.json`
- `reports/experiments/experiment_registry.csv`
- `reports/tables/tree_model_comparison.csv`
- `reports/tables/tree_model_comparison.json`
- `reports/figures/tree_model_metrics.pdf`

## Code references

The reproducible runner is `scripts/run_tree_models.py`. Model construction,
evaluation, persistence, and split validation reuse the shared `src` modules.

## Reproducibility notes

The two explicit experiment IDs, official split fingerprints,
`random_state=42`, shared metrics, and shared five folds were recorded. The
runner refuses to overwrite existing artifacts or duplicate registry entries.

## Sources and tools used

scikit-learn, pandas, Matplotlib, pytest, Python, and the shared project APIs.

## Next step

Define a reviewed training-only Random Forest search space under a new search
ID, while keeping the final test partition reserved.

# Logbook Entry

## Metadata

- Date: 2026-08-25
- Member: Member 02
- Sprint: To be completed by Member 02
- Ticket ID: ADA-DATA-EDA-01
- Branch: feature/eda+tree_models
- Pull Request: To be updated after Pull Request creation
- Time spent: 12 hours (retrospective estimate)
- Related meeting: To be completed by Member 02

## Title

Training-only exploratory data analysis

## Scientific question

What structural properties, class imbalance, feature distributions, linear
relationships, and potential outliers are visible in the official training
partition without using information from the reserved final test set?

## Data and protocol

The analysis uses the shared `load_dataset(optimize_memory=True)` entry point.
The configured OpenML dataset is converted explicitly to `float32`, then split
with the common stratified 80/20 protocol and `random_state=42`. Only the
160,000 training observations are exposed to the EDA cells. The 40,000-row test
partition is created to preserve the official split but is not inspected.

The structural audit reuses `get_dataset_summary` and
`audit_numeric_features`. Figures are persisted with the shared `save_figure`
helper. This replaces direct OpenML access and duplicated structural checks in
the original notebook.

## Analyses performed

- Dataset shape, dtypes, memory use, missing values, infinities, constant
  features, and quasi-constant features.
- Training-target counts and proportions.
- Descriptive statistics for all 200 numerical features.
- Histograms for representative features.
- Class-conditional means and standardized mean differences.
- Class-conditional distributions for the three largest standardized
  differences.
- Absolute pairwise Pearson correlations.
- Potential outlier counts using the 1.5-IQR rule.

## Results

The training partition contains 160,000 observations and 200 anonymous numeric
features. The target is strongly imbalanced, with approximately 90% negative
and 10% positive observations. Pairwise linear feature correlations are weak.
Some features show class-conditional distribution differences and several
features contain observations outside the conventional 1.5-IQR interval.

These results are descriptive. Correlation does not establish causality,
standardized mean differences do not by themselves justify feature selection,
and an IQR flag does not prove that an observation is erroneous.

## Interpretation and limitations

The weak pairwise correlations do not exclude nonlinear or multivariate
predictive relationships. The anonymized feature names prevent domain-level
interpretation. The class imbalance means accuracy alone will not be an
adequate model-selection metric. Potential outliers were retained because
removing them without domain evidence could discard predictive information.

The exploratory selection of example and top-difference features was performed
on the training partition only. No conclusion in this analysis uses the final
test partition.

## Decision

Retain all features and potential outliers for the baseline modeling stage.
Use stratified evaluation and imbalance-aware metrics, especially ROC-AUC,
Average Precision, recall, and balanced accuracy.

## Difficulties

The dataset is large and contains 200 anonymous variables, making exhaustive
visual inspection expensive and domain interpretation impossible. Plotting all
pairwise relationships would also add substantial noise without a specific
hypothesis.

## Adaptations and deviations from the plan

The original EDA loaded OpenML directly and mixed reusable checks with notebook
logic. It was aligned with the project loader, official split, audit functions,
configuration, and figure-saving convention. Stored Colab outputs were removed
so results can be regenerated from the shared environment.

## Rejected approaches

EDA on the complete dataset, inspection of the final test distribution,
automatic deletion of IQR-flagged observations, causal interpretation, and
feature removal based only on univariate exploration were rejected.

## Files changed

- `notebooks/02_eda.ipynb`
- `reports/figures/eda_feature_distributions.pdf` after notebook execution
- `reports/figures/eda_top_features_by_target.pdf` after notebook execution

## Code references

- `src/data.py`
- `src/validation.py`
- `src/visualization.py`
- `notebooks/02_eda.ipynb`

## Reproducibility notes

Run the notebook from the repository root with the project environment. The
shared loader, configured split, and fixed random state reproduce the same
training partition. Execute cells from top to bottom without manually replacing
`X` or `y`.

## Sources and tools used

OpenML through the shared project loader, pandas, NumPy, Matplotlib, and the
project's reusable data, validation, and visualization modules.

## Next step

Compare baseline models under the shared training-only cross-validation
framework without using the final test set.

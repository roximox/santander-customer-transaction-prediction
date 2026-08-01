# Santander Customer Transaction Prediction

## Project objective

This Advanced Data Analytics project will build a reproducible workflow for the
**Santander Customer Transaction Prediction** problem. The current phase creates
the collaborative project foundation only: no exploratory results or trained
models are included.

## Dataset source

The official loader retrieves **OpenML dataset ID 45566** through scikit-learn.
It uses scikit-learn's standard cache, and detects the real target name from the
objects returned by OpenML instead of encoding a target-column assumption.
OpenML currently reports `SantanderCustomerSatisfaction`; metadata keeps this
source name as `openml_dataset_name`, separately from the project name
`Santander Customer Transaction Prediction` in `project_dataset_name`.
Verify the download and display a concise, read-only audit from the project root:

```bash
python scripts/verify_dataset.py
```

Dataset files are deliberately **not versioned in Git**. This loading stage does
not modify types, values, or columns and does not write a dataset into the
repository.

## Project structure

```text
configs/       Shared configuration
data/          Ignored raw, interim, and processed data
notebooks/     Numbered analysis workflow
src/           Reusable Python modules
tests/         Offline structural and configuration tests
reports/       Figures, tables, experiment logs, Logbuch, and meetings
models/        Ignored serialized models
```

## Installation with `venv`

Python 3.11 or newer is required.

```bash
python -m venv .venv
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Installation with Conda

```bash
conda env create -f environment.yml
conda activate santander-ada
```

## Data acquisition

Use `from src.data import load_dataset` as the shared access point. The call
`X, y, metadata = load_dataset()` retrieves OpenML ID 45566 and validates basic
structural invariants. Local dataset files, if created during later phases,
belong in `data/raw/` and remain ignored by Git.

### Raw-data audit and optional memory optimization

OpenML currently loads the 200 numeric features as `float64`. Raw loading remains
the default and preserves every source value and dtype:

```python
X_raw, y, metadata = load_dataset()
```

An explicit `load_dataset(optimize_memory=True)` applies the recommended
`numeric_dtype` from `configs/config.yaml` only after structural validation and
adds a precision/memory comparison to the metadata. The `float32` evaluation is
intended to reduce memory while quantifying representation differences; it does
not treat those expected differences as data errors and never changes the target.

Run the reproducible audit with:

```bash
python scripts/run_data_audit.py
```

It generates `reports/tables/data_audit_summary.json`,
`reports/tables/feature_audit.csv`, and
`reports/tables/dtype_comparison.json`. These are audit reports, not copies of
the dataset. No complete raw or optimized dataset is saved or versioned in Git.

## Shared Train/Test Split

Every team member must use the shared shuffled 80/20 split created with target
stratification and `random_state=42`, as configured in `configs/config.yaml`.
The split is applied to the explicitly optimized `float32` features, while its
row indices are identical to a split of the raw features because no feature
values influence index selection.

Generate and validate the common split metadata with:

```bash
python scripts/create_data_split.py
```

The command saves only `reports/tables/train_test_split_summary.json`. SHA-256
fingerprints for ordered train and test indices allow all members to confirm
that they have exactly the same partitions. Feature and target datasets are not
saved. The test set is reserved for final evaluation and must never be used for
model selection, preprocessing decisions, feature selection, threshold tuning,
or hyperparameter tuning.

## Common Evaluation Framework

All project models use the same five-fold `StratifiedKFold` protocol with
shuffling and the shared `random_state=42`. Cross-validation receives training
data only; the final test partition remains closed. ROC-AUC is the primary
metric, while Average Precision is especially important for interpreting the
imbalanced target. F1, precision, recall, accuracy, and balanced accuracy are
reported under the same folds for every model.

The framework returns a table with one row per fold and a serializable summary
containing aggregate metrics, timing, estimator parameters, target distribution,
and deterministic fold fingerprints. Scientific results can be exported as a
fold CSV plus summary JSON and registered under a unique experiment ID.

Any learned preprocessing—including scaling, imputation, PCA, or feature
selection—must be inside a scikit-learn `Pipeline` so it is fitted independently
within each training fold. The framework never performs global preprocessing.

Run the infrastructure smoke test with:

```bash
python scripts/verify_evaluation_framework.py
```

This command uses only a small synthetic dataset and a technical
`DummyClassifier`. It saves no files, creates no registry entry, reports no
Santander model result, and does not access the final test set.

## Experiment Orchestration

`src/experiments.py` provides the common interface for model experiments while
`evaluate_model_cv` remains responsible for cross-validation and metrics.
`run_experiment` orchestrates evaluation and returns both fold results and the
summary; `save_results=False` is the default, so persistence and registration
are always explicit. Each scientific experiment must use a unique ID selected
by its author—IDs are never generated automatically.

The orchestrator accepts training data and an estimator or already constructed
Pipeline. The final test set is never passed to it, and it performs no automatic
preprocessing. Scientific interpretation and Logbook writing remain manual.

```python
from src.experiments import run_and_save_experiment

fold_results, summary = run_and_save_experiment(
    estimator=pipeline,
    X=X_train,
    y=y_train,
    experiment_id="M01-LR-001",
    model_name="Logistic Regression",
    member="Member 01",
    branch="feature/data_processing",
)
```

Verify the workflow offline with:

```bash
python scripts/verify_experiment_orchestrator.py
```

This technical check uses only synthetic data and a temporary directory; it
does not alter the project experiment registry.

## Dummy Baselines

Naive baselines show what the shared metrics look like without learning any
relationship between Santander features and the target. The first scientific
comparison covers `most_frequent` (`M01-DUMMY-001`), `prior`
(`M01-DUMMY-002`), `stratified` (`M01-DUMMY-003`), and `uniform`
(`M01-DUMMY-004`). All four use the common stratified cross-validation metrics:
ROC-AUC, Average Precision, F1, precision, recall, accuracy, and balanced
accuracy. Accuracy is insufficient by itself because predicting only the
majority class can score close to 90% on an imbalanced target while failing to
identify positive cases.

Run the registered training-only comparison once with:

```bash
python scripts/run_dummy_baselines.py
```

The command refuses existing IDs and outputs rather than overwriting them. It
creates fold CSV and summary JSON files plus `experiment_registry.csv` under
`reports/experiments/`, the comparison files
`reports/tables/dummy_baseline_comparison.csv` and `.json`, and the PDF figure
`reports/figures/dummy_baseline_metrics.pdf`. It verifies the official split
fingerprints and never evaluates the reserved final test partition.

## Model Factories

Shared estimators and pipelines are constructed through `src/modeling.py`.
Factories expose their hyperparameters, return a new unfitted object on every
call, and never load data, train a model, evaluate a split, or save an artifact.
Their defaults are reproducible project starting points—not optimized
configurations—and every scientific experiment must record the actual
hyperparameters it uses.

Logistic Regression scaling is kept inside a scikit-learn Pipeline so
`StandardScaler` is learned independently within each training fold. Tree-based
factories return estimators directly without unnecessary scaling.

```python
from src.modeling import create_logistic_regression_pipeline

pipeline = create_logistic_regression_pipeline(
    penalty="l2",
    C=1.0,
    class_weight=None,
)
```

Inspect every factory without loading data or fitting models:

```bash
python scripts/verify_model_factories.py
```

## Logistic Regression L2 Baseline

Experiment `M01-LR-001` evaluates the shared `StandardScaler` →
`LogisticRegression` Pipeline with L2 regularization, `C=1.0`, no class
weighting, and `max_iter=1000`. Scaling is learned inside each cross-validation
training fold because linear optimization is sensitive to feature scales. These
parameters define an untuned baseline and are not claimed to be optimal.

The experiment uses the common five-fold stratified protocol on training data
only and compares its ROC-AUC, Average Precision, F1, precision, recall,
accuracy, and balanced accuracy with the registered Dummy baselines. The final
test partition remains closed.

Run the experiment once with:

```bash
python scripts/run_logistic_baseline.py
```

The command refuses duplicate outputs. It writes fold results and a summary to
`reports/experiments/`, updates the registry, creates CSV and JSON comparisons
in `reports/tables/`, and exports `logistic_vs_dummy_metrics.pdf` plus
`logistic_cv_scores.pdf` in `reports/figures/`.

## Logistic Regression Class Weighting

Experiment `M01-LR-002` tests whether `class_weight="balanced"` changes
positive-class detection under imbalance. It is a controlled comparison with
`M01-LR-001`: only class weighting changes from `None` to `"balanced"`; the
Pipeline, L2 penalty, `C=1.0`, solver, iteration limit, random seed, folds, and
metrics remain fixed.

Balanced weights make errors on the minority class contribute more strongly to
the training objective. This can improve recall while reducing precision and
accuracy, so the result must be assessed as a full trade-off rather than chosen
from recall alone. The final test partition is not used.

Run the registered comparison once with:

```bash
python scripts/run_logistic_class_weight_comparison.py
```

The command creates the `M01-LR-002` fold and summary reports, updates the
registry, writes the class-weight comparison CSV and JSON under
`reports/tables/`, and exports two PDF figures under `reports/figures/`.

## Logistic Regression Grid Search

Search `M01-LR-SEARCH-001` exhaustively compares penalties L1/L2, `C` values
0.01, 0.1, 1, 10, and 100, and class weights `None`/`balanced`: 20 candidates,
five shared stratified folds, and 100 fits. Every candidate uses the common
`StandardScaler` → `LogisticRegression(solver="saga", max_iter=2000,
random_state=42)` Pipeline. The seven shared metrics are retained and ROC-AUC
is the predeclared `GridSearchCV` refit metric.

Run the protected, one-time search from the repository root:

```bash
python scripts/run_logistic_grid_search.py
```

Full candidate results and the summary are under `reports/searches/`; top-ten
and decision tables are under `reports/tables/`; ROC-AUC, metric-trade-off, and
train/validation figures are under `reports/figures/`. The final test partition
is fingerprint-verified only and is never scored or predicted during search.

## Logistic Regression Coefficient Audit

Performance and numerical convergence are separate requirements. Run the
targeted four-configuration, five-fold audit with:

```bash
python scripts/run_logistic_coefficient_analysis.py
```

For scikit-learn 1.8, this audit uses the recommended correspondence
`l1_ratio=0` for L2 and `l1_ratio=1` for L1 with `solver="saga"`; historical
experiments retain their original API. The exported coefficients are learned
after fold-specific `StandardScaler` fitting, so they describe standardized
features and are comparable in scale. They are predictive associations, not
causal effects. Interpret their signs and magnitudes jointly with stability
across folds, exact-zero selection frequency, convergence, and validation
metrics. The reserved final test partition is never evaluated by this audit.

## Running notebooks

Start Jupyter from the repository root:

```bash
jupyter lab
```

Run notebooks in numerical order and execute every notebook from top to bottom
before review. Notebooks currently contain planning scaffolds only.

## Team workflow and Git strategy

`main` is stable, while `develop` is the integration branch. Create one focused
branch per task, open a pull request into `develop`, and obtain at least one
review. Direct commits to `main` are prohibited. Detailed branch, commit, and
review rules are in [CONTRIBUTING.md](CONTRIBUTING.md).

## Experiment tracking

Every experiment must add a row based on
`reports/experiments/experiment_template.csv` and document its scientific work
in the Logbuch. Code, parameters, preprocessing, timing, metrics, feature count,
interpretation, branch, member, and shared random state must be traceable.

## Individual Logbooks

Each member maintains a chronological Logbook in
`reports/logbook/member_01/` through `reports/logbook/member_04/`. Entries use
the naming convention `YYYY-MM-DD_ticket-short-title.md` and should be written
soon after the related work. Every entry must document relevant decisions,
difficulties, adaptations, and code or report references. Each member is
responsible for the accuracy and completeness of their own directory; see
`reports/logbook/README.md` for the full convention.

## Reproducibility and leakage prevention

All team members must use:

- the same `random_state`;
- the same train/test split;
- the same metrics;
- the same experiment conventions.

Shared values live in `configs/config.yaml`. Fit preprocessing and feature
selection on training folds only. Keep the test set isolated and never use it
for model selection, threshold tuning, or iterative feature decisions.

## Agile workflow

Work is organized into sprints with tickets, short team meetings, documented
decisions, blockers, assignments, deadlines, and retrospectives. Use the
templates in `reports/meetings/` and `reports/logbook/`.

## Authors / team

- Team member 1: Yassine Elhari
- Team member 2: _TBD_
- Team member 3: _TBD_
- Team member 4: _TBD_

## Current status

**Phase 2 — reproducible data and Logistic Regression workflow.** OpenML
loading, structural and memory auditing, the shared stratified split, common
evaluation and experiment infrastructure, Dummy baselines, Logistic Regression
comparisons, grid search, and coefficient-stability analysis are present. The
reserved final test set remains closed.

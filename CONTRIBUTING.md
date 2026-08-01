# Contributing

## Git workflow

- `main` contains stable, reviewed work only.
- `develop` is the integration branch.
- Use one branch per task; never commit directly to `main`.
- Every change requires a pull request and at least one review by another member.
- Keep pull requests focused and document every relevant decision.

Branch conventions:

- `feature/data-audit`
- `feature/eda`
- `feature/feature-selection`
- `feature/boosting`
- `experiment/<short-name>`
- `fix/<short-name>`
- `docs/<short-name>`

Commit prefixes:

- `feat:` new functionality
- `fix:` defect correction
- `docs:` documentation
- `test:` tests
- `refactor:` behavior-preserving code changes
- `chore:` maintenance
- `experiment:` scientific experiment work

## Data and security

- Never commit dataset files, secrets, credentials, or local environment files.
- Store data only in the ignored `data/` subdirectories.
- Do not use test data for model selection, feature selection, preprocessing
  decisions, threshold tuning, or hyperparameter tuning.

## Scientific standards

- Notebooks must run from top to bottom from the repository root.
- Scientific results must be reproducible with the shared configuration.
- Every completed scientific or technical ticket requires a Logbook entry.
- Commit Logbook entries with the related work or immediately afterwards.
- Never fabricate or backdate entries; reported time spent must be truthful.
- Each member edits only their own individual Logbook, except for formatting
  corrections made through a reviewed pull request.
- Document every relevant scientific or engineering decision.
- Export figures as PDF, SVG, or high-resolution PNG.

## Shared model evaluation

- All models must use the shared evaluation framework in `src/evaluation.py`.
- Pass training data only to `evaluate_model_cv`; final test data is prohibited.
- Put every learned preprocessing step inside a scikit-learn `Pipeline`,
  including scaling, imputation, PCA, and feature selection.
- Each scientific experiment requires a unique `experiment_id`.
- Experiment IDs follow `<member>-<model>-<sequence>`, for example
  `M01-LR-001`, `M02-RF-001`, `M03-PCA-001`, or `M04-HGB-001`.
- Every scientific experiment must produce both fold-level results and a
  serializable summary, then add one unique row to the experiment registry.
- Synthetic smoke tests must not be recorded as scientific experiments.
- Use `run_experiment` or `run_and_save_experiment` for scientific model
  comparisons.
- Select experiment IDs explicitly; never generate them automatically or reuse
  an existing ID.
- Save and register only validated scientific experiments. Never register smoke
  tests.
- Write scientific interpretation and the individual Logbook entry manually.
- Never pass the final test set to the experiment orchestrator during model
  development.

Every experiment must log:

- `experiment_id`
- `model_name`
- `random_state`
- `preprocessing`
- `metrics`
- `parameters`
- fit time
- number of features
- interpretation

Use `reports/experiments/experiment_template.csv` as the canonical schema and
include its additional traceability fields.

## Model factories

- Prefer shared factories in `src/modeling.py` when they cover the required
  estimator.
- Do not hide experimental hyperparameters; pass and record them explicitly.
- Do not change shared factory defaults to improve one experiment without team
  review.
- Keep every preprocessing step learned from data inside a Pipeline.
- Factories must never call `fit` or perform evaluation or persistence.
- Factory defaults are baseline starting points, not optimized configurations.

## Hyperparameter searches

- Assign every search a unique, manually selected search ID and never reuse it.
- Use training data only; the final test partition is prohibited during search.
- Define and document the complete search space before execution.
- Declare the primary metric before execution and do not change it after seeing
  results.
- Save complete candidate-level results, not only the winning configuration.
- Report convergence warnings, total duration, fit count, and computational
  cost.
- Treat search cross-validation scores as model-selection evidence, never as
  final test performance.

## Coefficient analysis

- Report numerical convergence separately from predictive performance.
- Prefer the installed scikit-learn API: for version 1.8 Logistic Regression,
  use `l1_ratio=0` for L2 and `l1_ratio=1` for L1 when the chosen solver supports
  both.
- Keep scaling inside each cross-validation Pipeline; coefficients from a
  scaled Pipeline refer to standardized features.
- Never describe predictive coefficients as causal effects.
- Check coefficient sign, magnitude, exact-zero selection, and stability across
  folds before interpreting or selecting features.
- Preserve historical experiment parameterizations; API migrations belong to
  new, explicitly documented analyses.

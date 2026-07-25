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
- Add every scientific task to the Logbuch.
- Document every relevant scientific or engineering decision.
- Export figures as PDF, SVG, or high-resolution PNG.

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

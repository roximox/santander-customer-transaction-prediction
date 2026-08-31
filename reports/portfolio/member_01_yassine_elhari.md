# Individual Portfolio — Member 01

## Profile

- **Name:** Yassine Elhari
- **Role:** Member 01
- **Main responsibilities:** project setup, data engineering, reproducibility,
  shared evaluation methodology, Logistic Regression, results integration, and
  final model governance
- **Additional contributions:** Extra Trees baseline, interactive dashboard,
  cross-model comparison, and final documentation

## Contribution summary

My main contribution was to establish the common technical and scientific
foundation used by the four project members. I implemented reproducible data
access, dataset validation, the protected train/test split, common model
evaluation, experiment registration, and reusable model factories. This work
allowed the different model families to be evaluated under the same data and
cross-validation rules.

I was also responsible for the complete Logistic Regression track. I created
naive reference baselines, trained the initial L2 model, studied class
weighting, performed a controlled grid search, examined convergence and
coefficient stability, and generated learning curves. Near the end of the
project, I consolidated the model evidence, added the missing Extra Trees
baseline, implemented the scientific-results dashboard, recorded the
collective model-selection decision, and locked the selected pipeline before
the single final-test evaluation.

## Completed tasks

| Ticket | Contribution | Main outcome |
| --- | --- | --- |
| `ADA-SETUP-01` | Initial project setup | Created the reproducible repository structure, configuration, testing conventions, and shared working foundation. |
| `ADA-DATA-01` | Official OpenML data loading | Implemented centralized and validated loading of OpenML dataset 45566 without persisting raw row-level data. |
| `ADA-DATA-02` | Numeric and memory audit | Audited feature types, missing and constant values, target imbalance, and the memory/precision trade-off of `float32`. |
| `ADA-DATA-03` | Shared train/test split | Created the reproducible stratified 80/20 split and protected the 40,000-row final-test partition from iterative use. |
| `ADA-ML-00` | Common evaluation framework | Standardized five-fold stratified cross-validation, metrics, timing, generalization diagnostics, and result persistence. |
| `ADA-ML-01` | Experiment orchestrator | Centralized experiment execution, validation, saving, and registry updates to avoid duplicated workflows. |
| `ADA-ML-02` | Dummy baselines | Registered four `DummyClassifier` strategies as naive scientific reference points. |
| `ADA-ML-03` | Shared model factories | Implemented validated, reproducible factories for Logistic Regression and other shared estimators. |
| `ADA-ML-04` | Logistic Regression L2 baseline | Established the first learned linear baseline and demonstrated predictive signal beyond the dummy models. |
| `ADA-ML-05` | Class-weight comparison | Quantified the trade-off between the unweighted Logistic Regression and the recall-oriented balanced alternative. |
| `ADA-ML-06` | Logistic Regression grid search | Evaluated the predefined combinations of L1/L2 regularization, `C`, and class weighting using training-only CV. |
| `ADA-ML-07` | Coefficient-stability analysis | Investigated convergence warnings, L1 sparsity, coefficient stability, and the limitations of coefficient interpretation. |
| `ADA-ML-08` | Logistic learning curves | Measured the effect of training-set size on the selected unweighted and balanced Logistic Regression candidates. |
| `ADA-ML-09` | Pre-final selection framework | Built the read-only cross-model comparison, eligibility, coverage, ranking, and decision-report workflow. |
| `ADA-UI-01` | Interactive dashboard | Developed a Streamlit dashboard for exploring saved project evidence without retraining models or opening the final test. |
| `ADA-ML-10` | Extra Trees baseline | Filled the remaining expected-family coverage gap with the reproducible `M01-ET-001` experiment. |
| `ADA-ML-11` | Final-model lock | Recorded the collective selection of `M04-HGB-002`, froze its parameters and threshold, and preserved the single-use final-test policy. |

## Scientific results

My Logistic Regression work established a strong, inexpensive, and stable
linear reference. The unweighted L2 baseline reached a mean cross-validated
ROC-AUC of approximately `0.8592`. Balanced class weighting substantially
improved recall and balanced accuracy, but reduced precision, illustrating why
threshold-dependent model decisions require an explicit operational objective.

The Extra Trees baseline broadened the comparison with a nonlinear ensemble
and recorded the strongest pre-final mean F1 among the eligible candidates.
The complete portfolio nevertheless showed that the tuned
HistGradientBoosting model from Member 04 led the principal ranking metrics.
This evidence supported the team's collective decision to select and freeze
`M04-HGB-002` rather than favoring my own model family.

## Reproducibility and teamwork impact

- Established the common dataset, split fingerprints, random state, metrics,
  and validation rules used by the team.
- Kept preprocessing inside model pipelines to reduce leakage risk.
- Ensured that iterative development used training data only.
- Made results traceable through experiment identifiers, CSV/JSON reports,
  scripts, tests, logbooks, and meeting decisions.
- Integrated evidence from the four members into a comparable model portfolio.
- Preserved the collective lock before the reserved final partition was used
  exactly once.

## Main deliverables

- [Member 01 logbook](../logbook/member_01/README.md)
- [Pre-final model-comparison portfolio](../model_selection/model_comparison_portfolio.md)
- [Collective model-selection notes](../model_selection/group_model_selection_notes.md)
- [Final scientific conclusions](../scientific_conclusions.md)
- [Interactive dashboard](../../app.py)

## Skills demonstrated

Python, pandas, NumPy, scikit-learn, OpenML, reproducible data pipelines,
cross-validation, imbalanced classification, Logistic Regression, Extra Trees,
hyperparameter search, learning curves, model comparison, automated testing,
Streamlit, Git, scientific documentation, and collaborative model governance.

# Logbook Entry

## Metadata

- Member: Yassine Elhari
- Ticket ID: ADA-ML-11
- Date: 2026-08-26
- Time spent: TO BE COMPLETED BY YASSINE ELHARI
- Related meeting: TO BE COMPLETED BY YASSINE ELHARI
- Pull Request: To be updated after Pull Request creation

## Title

Collective final-model selection and reproducible pipeline lock

## Objective

Record the team's confirmed model choice without opening or evaluating the reserved final test.

## Context

Expected-family coverage was complete after adding Extra Trees. The generated comparison identified M04-HGB-002 as the only competitive candidate and the leader on mean cross-validated ROC-AUC and Average Precision.

## Work performed

Recorded M04-HGB-002 as the collectively selected pipeline, captured its fixed estimator parameters and threshold, added the meeting decision, and exposed the lock in the read-only UI.

## Methodology

The choice uses only saved five-fold StratifiedKFold evidence from the 160,000-row development partition. No experiment was retrained. The final test remained closed and was not used.

## Results

The selected model has mean CV ROC-AUC 0.891449 and Average Precision 0.591089. Its locked classification threshold is 0.5 and its recorded random state is 42. The single final-test execution produced ROC-AUC 0.891214, Average Precision 0.584385, F1 0.403632, precision 0.791424, recall 0.270896, and balanced accuracy 0.631459.

## Interpretation

HGB provides the best recorded ranking evidence for the imbalanced prediction problem, while its larger generalization gap, runtime, and limited recall at the fixed threshold remain explicit limitations.

## Decision

M04-HGB-002 remains the locked final model after one controlled final-test evaluation. Model selection was not reopened. This is a collective multi-criteria choice, not a formal statistical-superiority claim.

## Difficulties

Different models lead different metrics, so the decision required prioritizing ranking quality while retaining threshold and computational limitations.

## Adaptations and deviations from the plan

The group waited for complete expected-family coverage before locking the model.

## Rejected approaches

Rejected reopening hyperparameter search, changing the threshold after selection, and consulting the final test before the model lock.

## Files changed

- `reports/model_selection/final_model_lock.json`
- `reports/final_evaluation/M04-HGB-002_final_test_results.json`
- `reports/meetings/2026-08-26_final-model-selection-decision.md`
- `app.py`
- `src/dashboard/loaders.py`
- dashboard and logbook tests

## Code references

- `src/model_selection.py`
- `src/dashboard/loaders.py`
- `app.py`

## Figure and table references

- `reports/model_selection/model_comparison_eligible.csv`
- `reports/model_selection/model_selection_decision.csv`
- `reports/model_selection/model_selection_comparability.json`

## Reproducibility notes

The lock records the shared split fingerprints, StratifiedKFold configuration, estimator parameters, classification threshold, and the unique final-evaluation identifier. The final test was not used during model selection; it was evaluated exactly once after the collective lock.

## Next step

Finalize the scientific report using the single recorded result. Do not rerun the final evaluation or tune from its metrics.

## Sources and tools used

Saved experiment summaries, generated model-selection reports, the shared dashboard loaders, pytest, and Streamlit AppTest.

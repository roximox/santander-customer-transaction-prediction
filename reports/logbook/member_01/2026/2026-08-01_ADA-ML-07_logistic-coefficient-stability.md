# Logbook Entry

## Metadata

- Date: 2026-08-01
- Member: Yassine Elhari
- Sprint: Sprint 1
- Ticket ID: ADA-ML-07
- Branch: feature/data_processing
- Pull Request: [#2 — feature/data_processing → main](https://github.com/roximox/santander-customer-transaction-prediction/pull/2)
- Time spent: 4 hours
- Related meeting: [2026-08-16 — First Individual Analysis and Machine Learning Progress](../../../meetings/2026-08-16_first-individual-analysis-and-machine-learning-progress.md)

## Title

Logistic Regression convergence, sparsity and coefficient stability

## Reason for the audit

M01-LR-SEARCH-001 produced six convergence warnings that could not be mapped to
parallel candidates. This audit isolates four scientifically relevant
configurations and examines convergence, exact L1 sparsity, and coefficient
stability without starting another grid search.

## Configurations and API

The audit covers L2/C=0.01 unweighted, L1/C=0.1 unweighted, L2/C=0.01 balanced,
and L1/C=100 unweighted. Each uses `StandardScaler` and `LogisticRegression`
inside a Pipeline on each of the five common training folds. Under scikit-learn
1.8, `l1_ratio=0` represents L2 and `l1_ratio=1` represents L1 with `saga`.
Synthetic tests confirm equivalence with the deprecated `penalty` formulations
for these cases. Historical experiments were not changed.

## Convergence results

All 20 fits converged without `ConvergenceWarning`. Mean `n_iter_` was 19.6 for
LR-SELECTED-ROC, 23.0 for LR-SELECTED-AP, 31.4 for LR-SELECTED-BALANCED, and
21.6 for LR-L1-WEAK-REG, all far below `max_iter=2000`. Mean fit times were
3.33, 5.01, 5.53, and 4.73 seconds respectively. The grid warnings therefore
do not reproduce with isolated fits and the recommended 1.8 API.

## Sparsity and stability

L1/C=0.1 retained 196–200 features per fold (98–100%), with 193 features in the
five-fold intersection and all 200 in the union. L1/C=100 retained all 200
features in every fold. Consequently neither L1 candidate offers meaningful
parsimony at these C values.

For LR-SELECTED-ROC, leading standardized coefficients—including `var_81`,
`var_139`, `var_6`, `var_12`, and `var_76`—share the same sign in every fold and
have small fold standard deviations. Coefficients describe predictive
associations after scaling, not causal effects.

## Performance and decision

LR-SELECTED-ROC retains the best ROC-AUC (0.859201), converges fastest on
average, and has stable leading coefficients. LR-SELECTED-AP retains the best
AP (0.507626) by a very small margin but does not materially reduce dimension.
LR-SELECTED-BALANCED retains the strongest F1 (0.416119) and balanced accuracy
(0.778194) for recall-oriented use. LR-L1-WEAK-REG adds no parsimony or relevant
metric advantage.

The provisional primary ranking candidate remains LR-SELECTED-ROC. Preserve
LR-SELECTED-BALANCED as the threshold-dependent recall alternative and do not
claim a final business choice before costs and threshold policy are specified.

## Limits

No final-test data, threshold optimization, calibration, causal coefficient
interpretation, or final comparison with other members' models was used.
Stability is measured across five folds from one configured split and seed.

## Next step

Define the operational error-cost question before threshold analysis, while
keeping the final test partition closed.

## Difficulties

Historical parallel warnings were not attributable to candidates, and explicit
`penalty` is deprecated in scikit-learn 1.8. Exact-zero sparsity also required
distinguishing true selection from an arbitrary magnitude threshold.

## Adaptations and deviations from the plan

Four targeted candidates were fitted fold by fold with `l1_ratio=0` or `1`, so
warnings and iteration counts are attributable without rerunning the grid.

## Rejected approaches

Rerunning the costly search, applying an arbitrary coefficient threshold,
changing historical results, and causal coefficient claims were rejected.

## Files changed

- `src/logistic_coefficient_analysis.py`
- `scripts/run_logistic_coefficient_analysis.py`
- `tests/test_logistic_coefficient_analysis.py`

## Code references

Target configurations, fold audit, exact sparsity, stability summaries, and
figure factories in `src/logistic_coefficient_analysis.py`.

## Figure and table references

- `reports/tables/logistic_convergence_audit.csv`
- `reports/tables/logistic_coefficients_by_fold.csv`
- `reports/tables/logistic_coefficient_stability.csv`
- `reports/tables/logistic_l1_sparsity_summary.csv`
- `reports/tables/logistic_model_selection_summary.csv`
- `reports/figures/logistic_top_coefficients.pdf`
- `reports/figures/logistic_coefficient_stability.pdf`
- `reports/figures/logistic_l1_sparsity.pdf`
- `reports/figures/logistic_convergence_iterations.pdf`

## Reproducibility notes

Twenty isolated fits use the shared five folds, `random_state=42`, and
`max_iter=2000`. The final test set remained closed and was never scored.

## Sources and tools used

scikit-learn 1.8, pandas, NumPy, Matplotlib, pytest, and Python.

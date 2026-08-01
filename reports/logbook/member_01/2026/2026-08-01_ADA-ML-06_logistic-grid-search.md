# Logbook Entry

## Metadata

- Date: 2026-08-01
- Member: Yassine Elhari
- Sprint: Sprint 1
- Ticket ID: ADA-ML-06
- Branch: feature/data_processing
- Pull Request: To be updated after Pull Request creation
- Time spent: TO BE COMPLETED BY YASSINE ELHARI
- Related meeting: TO BE COMPLETED BY YASSINE ELHARI

## Title

Logistic Regression grid search

## Scientific question

Which combination of L1/L2 regularization, `C`, and class weighting gives the
best cross-validated Logistic Regression behavior on the common training set?

## Protocol and justification

`GridSearchCV` exhaustively evaluated the predeclared 2 × 5 × 2 space: penalties
L1/L2, `C` in 0.01, 0.1, 1, 10, 100, and class weights `None`/`balanced`. The 20
candidates produced 100 fits over the common five-fold stratified CV. Every fit
used `StandardScaler` → `LogisticRegression(solver="saga", max_iter=2000,
random_state=42)`. ROC-AUC was declared as the refit metric before execution.

Only the 160,000-row float32 training partition was passed to the search. The
reserved 40,000-row test fingerprint was verified, then its objects were
deleted without calling `score`, `predict`, or `predict_proba`.

## Computational cost and convergence

The complete search and refit took 974.66 seconds (16.24 minutes), averaging
48.73 seconds per candidate and 9.75 seconds per fit with `n_jobs=-1`. Six
`ConvergenceWarning` messages stated that `max_iter=2000` was reached. Warnings
emitted by parallel workers cannot be mapped reliably to candidate IDs through
`GridSearchCV`; no scores were hidden, changed, or silently rerun. A corrective
study, if needed, must use a new Search ID.

## Results and metric-specific winners

- ROC-AUC: `candidate_002`, L2, C=0.01, unweighted — 0.859201.
- Average Precision: `candidate_005`, L1, C=0.1, unweighted — 0.507626.
- F1: `candidate_004`, L2, C=0.01, balanced — 0.416119.
- Balanced Accuracy: `candidate_004` — 0.778194.

`candidate_002` has precision 0.691427 and recall 0.267758. In contrast,
`candidate_004` has precision 0.284599 and recall 0.773666. Thus class weighting
changes default-threshold behavior substantially while ROC-AUC remains almost
unchanged.

## Comparison and provisional decision

The registered neutral baseline M01-LR-001 has ROC-AUC 0.859188 and AP 0.507566;
the ROC-AUC winner improves these by only 0.000013 and 0.000027. M01-LR-002 has
F1 0.416059 and balanced accuracy 0.778128; `candidate_004` improves these by
only 0.000060 and 0.000066. These differences are small relative to fold
variation. No single configuration is declared the final business choice:
retain `candidate_002` as the ROC-AUC-selected candidate, `candidate_005` as the
AP alternative, and `candidate_004` as the recall-oriented alternative.

## Figures and artifacts

The ROC-AUC/C figure includes fold uncertainty, the trade-off figure contrasts
threshold metrics, and the train/validation figure shows small generalization
gaps without a strong overfitting signal. Full candidate results and summary are
stored under `reports/searches/`; decision and top-ten tables are under
`reports/tables/`.

## Limits

Cross-validation is model-selection evidence, not final test performance. No
threshold optimization, calibration, definitive coefficient interpretation,
or nonlinear-model comparison was performed. Explicit `penalty` is deprecated
by scikit-learn 1.8, although required by and functional for this search.

## Next step

Analyze coefficient stability for selected linear candidates, or define a
separate corrective convergence search with a new Search ID.

## Difficulties

Six warnings emitted by parallel workers could not be attributed reliably to
individual candidates. Finalization also required JSON-safe handling of missing
candidate values; the completed search itself was not rerun or altered.

## Adaptations and deviations from the plan

Existing complete candidate and summary outputs were finalized after correcting
serialization, avoiding a second costly 100-fit search.

## Rejected approaches

Mapping parallel warnings speculatively, selecting on the final test set,
overwriting the Search ID, and hiding convergence warnings were rejected.

## Files changed

- `src/search.py`
- `scripts/run_logistic_grid_search.py`
- `tests/test_search.py`
- `notebooks/03_logistic_regression.ipynb`

## Code references

Search-space, serialization, ranking, and figure helpers in `src/search.py`;
protected execution and finalization in `scripts/run_logistic_grid_search.py`.

## Figure and table references

- `reports/searches/M01-LR-SEARCH-001_candidates.csv`
- `reports/searches/M01-LR-SEARCH-001_summary.json`
- `reports/tables/logistic_grid_search_decision_table.csv`
- `reports/tables/logistic_grid_search_top_candidates.csv`
- `reports/figures/logistic_grid_search_roc_auc.pdf`
- `reports/figures/logistic_grid_search_tradeoff.pdf`
- `reports/figures/logistic_grid_search_train_validation.pdf`

## Reproducibility notes

The search used the 160,000-row training partition, five stratified folds,
`random_state=42`, and 20 predeclared candidates. The final test set was
fingerprint-verified only and remained closed.

## Sources and tools used

scikit-learn `GridSearchCV`, pandas, NumPy, Matplotlib, pytest, JSON, and Python.

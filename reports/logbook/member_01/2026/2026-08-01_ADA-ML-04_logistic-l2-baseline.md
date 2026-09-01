# Logbucheintrag

## Metadaten

- Datum: 2026-08-01
- Mitglied: Yassine Elhari
- Sprint: Sprint 1
- Ticket-ID: ADA-ML-04
- Branch: feature/data_processing
- Pull-Anforderung: [#2 — feature/data_processing → main](https://github.com/roximox/santander-customer-transaction-prediction/pull/2)
- Zeitaufwand: 3 Stunden
- Zugehörige Besprechung: [2026-08-16 — Erster individueller Analyse- und Maschinelles Lernenfortschritt](../../../meetings/2026-08-16_first-individual-analysis-and-machine-learning-progress.md)

## Titel

Logistische Regressionsl2-Baselin

## Wissenschaftliche Frage

Versteht man, ob ein nicht geoptimiertes L2-regulierter lineares Modell unter der gemeinsamen Kreuzvalidierungsprotokoll diskriminative Signal lernen kann, das über die registrierten Naiven Basen hinausgeht?

## Pipeline und Protokoll

Experiment `M01-LR-001`, benannt `Logistic Regression L2 Baseline`, verwendete die gemeinsame Fabrik, um `StandardScaler` zu erstellen, gefolgt von `LogisticRegression`. Skalierung ist notwendig für vergleichbare Feature-Skalen während der Koeffizientenoptimierung und blieb innerhalb des Pipelins, so dass sie separat in jedem Fold gelernt wurde. L2 und `C=1.0` definieren eine Baselin, nicht einen optimierten Wert.

Parameter waren `penalty="l2"`, `C=1.0`, `class_weight=None`, `solver="lbfgs"`, `max_iter=1000`, `random_state=42`, `with_mean=True`, und `with_std=True`. Die offiziellen Trainings- und Reserveteilungsabdrücke wurden überprüft. Nur die 160.000 Trainingszeilen gingen in fünf-fach stratifizierten Kreuzvalidierung; der Testteil wurde nicht bewertet.

## Ergebnisse

| Metrik | Trainingsmittelwert | Validiermittelwert | ValidierMittelstand |
|---|---:|---:|---:|
| ROC-AUC | 0.861525 | 0.859188 | 0.003239 |
| Durchschnittliche Präzision | 0.513131 | 0.507566 | 0.008490 |
| F1 | 0.395235 | 0.390361 | 0.002371 |
| Genauigkeit | 0.915025 | 0.914481 | 0.000830 |
| Gleichgewichtsgenauigkeit | 0.631351 | 0.629343 | 0.001565 |

Die Validier-ROC-AUC pro Fold betrug 0.858534, 0.857496, 0.865475, 0.858185 und 0.856249. Sein Standarddeviation von 0.003239 zeigt stabiles Verhalten über diese Folds. Der Mittelwert des Trainingsminus-Validier-ROC-AUC-Gap betrug 0.002337.

## Dummy-Vergleich

Im Vergleich zu `M01-DUMMY-001` stieg die ROC-AUC absolut um 0.359188, die Durchschnittliche Präzision stieg absolut um 0.407078 und relativ um 4.051032 (405.10%) und die Gleichgewichtsgenauigkeit stieg um 0.129343. Die Logistische Mittelfold-Fitzeit betrug 0.644985 Sekunden, 3.739 Mal so lange wie die aufgezeichnete Mehrheits-Dummy-Fit-Zeit.

## Konvergenz

Kein `ConvergenceWarning` wurde mit `lbfgs` und `max_iter=1000` detektiert. Das aufgezeichnete Experiment wurde nicht nach der Bewertung geändert.

## Interpretation und Einschränkungen

Das Baseline lernt erheblich mehr diskriminatives Signal als die Dummy-Referenzen, aber es ist nicht festgestellt worden. Die Standardentscheidungs-Schwelle gibt immer noch einen Wert von 0.272484 zurück, und dieses Ticket wurde keine Schwellenwerte, Regulierung, Klassen- Gewichte, Kalibrierung oder andere Hyperparameter nicht geoptimiert. Keine kausale Interpretation ist gemacht worden.

## Ausgabewerte und nächster Schritt

Die Ergebnisse und die Zusammenfassung werden unter `M01-LR-001` registriert. Die Vergleichstabelle befindet sich in `reports/tables/`, und die Metrik- und Folds-Figuren befinden sich in `reports/figures/`. Ein späteres Experiment kann eine explizit motivierte Alternative unter einem neuen ID untersuchen; der letzte Testteil bleibt geschlossen.

## Entscheidung

M01-LR-001 als neutrales gelerntes Baseline behalten; es ist nicht ein optimiertes oder endgültiges Modell.

## Schwierigkeiten

Die Klassenungleichheit macht die Genauigkeit unzureichend und die Standard-Schwelle produziert begrenzte positive Erträge trotz starker Rang-Metriken.

## Anpassungen und Abweichungen vom Plan

Skalierung wurde in den Pipelins innerhalb platziert, sodass jede Fold seine eigenen Parameter lernte.

## Abgelehnte Ansätze

Globale Skalierung, Schwellen-Tuning, Test-Set-Vergleich und die Aussage des Baselines als optimal waren abgelehnt.

## Geänderte Dateien

- `src/logistic_baseline.py`
- `scripts/run_logistic_baseline.py`
- `tests/test_logistic_baseline.py`

## Code-Referenzen

Experiment und Vergleichshilfen in `src/logistic_baseline.py`.

## Figuren- und Tabellen-Bezüge

- `reports/experiments/M01-LR-001_fold_results.csv`
- `reports/experiments/M01-LR-001_summary.json`
- `reports/tables/logistic_baseline_comparison.csv`
- `reports/figures/logistic_vs_dummy_metrics.pdf`
- `reports/figures/logistic_cv_scores.pdf`

## Reproduzierbarkeitshinweise

Die offizielle Trainingsabdruck, fünf gemeinsame Folds und `random_state=42` wurden verwendet. Der letzte Testteil wurde nur und blieb geschlossen.

## Verwendete Quellen und Werkzeuge

scikit-learn, pandas, Matplotlib, pytest, Python und die gemeinsame Experimentier-API.

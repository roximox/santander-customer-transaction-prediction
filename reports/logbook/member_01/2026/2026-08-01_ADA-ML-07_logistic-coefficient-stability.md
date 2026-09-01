# Logbucheintrag

## Metadaten

- Datum: 2026-08-01
- Mitglied: Yassine Elhari
- Sprint: Sprint 1
- Ticket ID: ADA-ML-07
- Branch: feature/data_processing
- Pull Request: [#2 — feature/data_processing → main](https://github.com/roximox/santander-customer-transaction-prediction/pull/2)
- Zeitaufwand: 4 Stunden
- Zugehörige Besprechung: [2026-08-16 — Erster individueller Analyse- und Machine-Learning- Fortschritt](../../../meetings/2026-08-16_first-individual-analysis-and-machine-learning-progress.md)

## Titel

Logistische Regressionskonvergenz, Sparsität und Koeffizientenstabilität

## Grund für die Prüfung

M01-LR-SEARCH-001 produzierte sechs Konvergenzwarnungen, die nicht auf parallele Kandidaten zurückgeführt werden konnten. Diese Prüfung isoliert vier wissenschaftlich relevante Konfigurationen und untersucht Konvergenz, exakte L1-Sparsität und Koeffizientenstabilität ohne ein weiteres Grid-Search.

## Konfigurationen und API

Die Prüfung deckt die Konfigurationen L2/C=0.01 ungewichtete, L1/C=0.1 ungewichtete, L2/C=0.01 ausgewogen und L1/C=100 ungewichtete zu. Jede verwendet `StandardScaler` und `LogisticRegression` in einem Pipelinen auf jedem der fünf üblichen Trainingsfälle. Unter scikit-learn 1.8 stellt `l1_ratio=0` die L2 dar und `l1_ratio=1` die L1 mit `saga`. Synthetische Tests bestätigen die Gleichwertigkeit mit den abgelehnten `penalty`-Formulierungen für diese Fälle. Historische Experimente wurden nicht geändert.

## Konvergenzergebnisse

Alle 20 Fits konvergierten ohne `ConvergenceWarning`. Der durchschnittliche `n_iter_` betrug 19,6 für LR-SELECTED-ROC, 23,0 für LR-SELECTED-AP, 31,4 für LR-SELECTED-BALANCED und 21,6 für LR-L1-WEAK-REG, alle weit unter `max_iter=2000`. Der durchschnittliche Fitzeit betrug 3,33, 5,01, 5,53 und 4,73 Sekunden. Die Grid-Warnungen reproduzieren sich daher nicht mit isolierten Fits und die empfohlene API.

## Sparsität und Stabilität

L1/C=0,1 behielt 196–200 Features pro Fold (98–100%), mit 193 Features in der fünf-Fold-Intersektion und alle 200 Features in der Union. L1/C=100 behielt alle 200 Features in jedem Fold. Folglich bieten weder die L1-Kandidaten bei diesen C-Werten bedeutende Parsimonie.

Für LR-SELECTED-ROC teilen die führenden Standardisierte Koeffizienten – einschließlich `var_81`, `var_139`, `var_6`, `var_12` und `var_76` – in jedem Fold dieselbe Vorzeichen und haben kleine Folds-Standardabweichungen. Die Koeffizienten beschreiben nach Skalierung vorhergesagte Assoziationen, nicht kausale Effekte.

## Leistung und Entscheidung

LR-SELECTED-ROC behält die beste ROC-AUC (0,859201), konvergiert am schnellsten durchschnittlich und hat stabile führende Koeffizienten. LR-SELECTED-AP hält die beste AP (0,507626) durch einen sehr kleinen Abstand, aber reduziert weder Dimension. LR-SELECTED-BALANCED hält die stärkste F1 (0,416119) und eine ausgewogene Genauigkeit (0,778194) für Anwendungszwecke mit Erinnerung an den Vorzug bei der Anforderung nach Recall. LR-L1-WEAK-REG fügt weder Parsimonie noch einen relevanten Metriken-Vorteil hinzu.

Die vorläufige Hauptrangierkandidat bleibt LR-SELECTED-ROC. Bewahren Sie LR-SELECTED-BALANCED als den Schwellenabhängigen Recall-Alternative auf und lassen Sie keinen Anspruch auf eine endgültige Geschäftsentscheidung vor Kosten und Schwellenpolitik angegeben werden.

## Grenzen

Keine endgültige Testdaten, Schwellenoptimierung, Kalibrierung, kausale Koeffizienteninterpretation oder Vergleich mit anderen Mitgliedern' Modellen wurde verwendet.
Stabilität wird über fünf Folds von einer konfigurierten Spaltung und einem festen Sammlungszeitpunkt gemessen.

## Nächster Schritt

Definieren Sie die Betriebsfehlerkostenfrage vor der Schwellenanalyse, während die endgültige Testteilung geschlossen bleibt.

## Schwierigkeiten

Historische parallele Warnungen konnten nicht den Kandidaten zugeordnet werden und explizites `penalty` ist in scikit-learn 1,8 deaktiviert. Exakt Null-Sparsität erforderte die Unterscheidung zwischen wahrer Auswahl und einem arithmetischen Schwellenwert.

## Anpassungen und Abweichungen vom Plan

Vier zielgerichtete Kandidaten wurden in Fold für Fold mit `l1_ratio=0` oder `1` eingesetzt, sodass Warnungen und Iterationszahlen ohne Wiederholung des Grids zugeschrieben werden können.

## Abgelehnte Ansätze

Rerunnen des teuren Suchens, Anwenden eines arithmetischen Koeffizienten-Schwellenwertes, Ändern historischer Ergebnisse und kausale Koeffizientenansprüche wurden abgelehnt.

## Geänderte Dateien

- `src/logistic_coefficient_analysis.py`
- `scripts/run_logistic_coefficient_analysis.py`
- `tests/test_logistic_coefficient_analysis.py`

## Code-Referenzen

Zielkonfigurationen, Folds-Audit, exakte Sparsität, Stabilitätssummarisierungen und Figuren-Factories in `src/logistic_coefficient_analysis.py`.

## Figuren- und Tabellenbezüge

- `reports/tables/logistic_convergence_audit.csv`
- `reports/tables/logistic_coefficients_by_fold.csv`
- `reports/tables/logistic_coefficient_stability.csv`
- `reports/tables/logistic_l1_sparsity_summary.csv`
- `reports/tables/logistic_model_selection_summary.csv`
- `reports/figures/logistic_top_coefficients.pdf`
- `reports/figures/logistic_coefficient_stability.pdf`
- `reports/figures/logistic_l1_sparsity.pdf`
- `reports/figures/logistic_convergence_iterations.pdf`

## Reproduzierbarkeitshinweise

Zwanzig isolierte Fits verwenden die gemeinsame fünf-Fold-Sammlung, `random_state=42`, und `max_iter=2000`. Die endgültige Testset blieb geschlossen und wurde nie abgeschossen.

## Verwendete Quellen und Werkzeuge

scikit-learn 1,8, pandas, NumPy, Matplotlib, pytest und Python.

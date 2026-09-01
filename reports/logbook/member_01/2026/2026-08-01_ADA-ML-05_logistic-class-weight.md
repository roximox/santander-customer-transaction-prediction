# Logbucheintrag

## Metadaten

- Datum: 2026-08-01
- Mitglied: Yassine Elhari
- Sprint: Sprint 1
- Ticket ID: ADA-ML-05
- Branch: feature/data_processing
- Pull Request: [#2 — feature/data_processing → main](https://github.com/roximox/santander-customer-transaction-prediction/pull/2)
- Aufgewendete Zeit: 4,5 Stunden
- Zugehörige Besprechung: [2026-08-16 — Erste individuelle Analyse und Fortschritt in der maschinellen Lernung](../../../meetings/2026-08-16_first-individual-analysis-and-machine-learning-progress.md)

## Titel

Logistische Regressionsklasse mit Gewichtsvergleich

## Wissenschaftliche Frage

Gibt es bei einem ausgewogenen Klassengewicht eine Verbesserung der positiven-Klassenaufrufrate, F1-Wert und ausgeglichenen Genauigkeit ohne erhebliche Verluste in Präzision, ROC-AUC und Durchschnittspräzision?

## Kontrollierte Änderung

`M01-LR-002`, benannt `Logistic Regression L2 Balanced`, änderte nur `class_weight` von `None` zu `"balanced"` im Vergleich zu `M01-LR-001`. Die Pipeline `StandardScaler` → `LogisticRegression`, L2-Penalty, `C=1.0`, `lbfgs`, `max_iter=1000`, `random_state=42`, fünf Fälle, Spaltung und Metriken wurden unverändert. Die offiziellen Splitterfingerabdrücke passierten, und nur das Trainingssatz wurde bewertet.

## Ergebnisse und Differenzen

| Metrik | M01-LR-001 | M01-LR-002 | Ausgewogen - ungewichtet |
|---|---:|---:|---:|
| ROC-AUC | 0,859188 | 0,859011 | -0,000176 |
| Durchschnittliche Präzision | 0,507566 | 0,506430 | -0,001135 |
| Präzision | 0,688813 | 0,284560 | -0,404253 |
| Aufrufrate | 0,272484 | 0,773541 | +0,501057 |
| F1-Wert | 0,390361 | 0,416059 | +0,025698 |
| Genauigkeit | 0,914481 | 0,781794 | -0,132687 |
| Ausgewogene Genauigkeit | 0,629343 | 0,778128 | +0,148786 |

Das ausgewogene Modell hatte eine Trainings-ROC-AUC von 0,861823 und eine Validierungs-ROC-AUC von 0,859011, was einen Abstand von 0,002812 ergab. Die Standardabweichung der Validierungs-ROC-AUC betrug 0,003121 und die Durchschnittliche Präzision standardisierte sich auf 0,008519, was eine ähnliche Faltstabilität wie das ungewichtete Basismodell zeigte. Kein `ConvergenceWarning` wurde ermittelt.

## Interpretation und Entscheidung

Ausgewogenes Gewicht stieg die Aufrufrate und ausgeglichenen Genauigkeit deutlich, produzierte jedoch einen großen Verlust in Präzision und eine erhebliche Genauigkeitsabnahme. ROC-AUC und Durchschnittliche Präzision waren effektiv stabil, aber leicht niedriger, so dass das Klassengewicht die Betriebsgewinn-Abwägung nicht verbesserte, sondern den Betriebsschlüssel änderte.

`M01-LR-002` wird als Alternativ für eine Aufruforientierte Lösung beibehalten und nicht weltweit besser erklärt. Mit ROC-AUC als Hauptmetrik und ohne dokumentierten relativen Verlust für falsche Negativergebnisse und falsche Positivergebnisse bleibt `M01-LR-001` das neutrale Basismodell. Eine Auswahl zwischen ihnen sollte einen expliziten Betriebskostenziel haben.

## Grenzen und nächster Schritt

Diese Vergleichung optimiert nicht eine Schwelle, passt die Regulierung, kalibriert die Wahrscheinlichkeiten oder legt die Betriebskosten fest. Ein zukünftiges Experiment sollte zunächst das gewünschte Präzisionsaufruf- oder Kostenvergleichgewicht unter einem neuen ID definieren. Die endgültige Testpartition bleibt geschlossen.

## Entscheidung

`M01-LR-002` wird als Aufruforientierte Alternative beibehalten und `M01-LR-001` bleibt das neutrale Basismodell, bis Betriebsfehlerkosten definiert werden.

## Schwierigkeiten

Ausgewogenes Gewicht verbessert die Aufrufrate, aber senkt die Präzision stark, so dass kein einziges Schwellenwertmetrik einen ausreichenden Entscheidungsschlüssel liefert.

## Anpassungen und Abweichungen vom Plan

Nur `class_weight` wurde geändert, um eine kontrollierte Vergleichung zu gewährleisten.

## Abgelehnte Ansätze

Die globale Ausgewogenheit des Modells als besser erklärt, die Schwelle zu passen und das endgültige Testset zu verwenden, wurden abgelehnt.

## Dateien geändert

- `src/logistic_class_weight.py`
- `scripts/run_logistic_class_weight_comparison.py`
- `tests/test_logistic_class_weight.py`

## Code-Referenzen

Controlierte Vergleich und Ausgabe in `src/logistic_class_weight.py`.

## Figuren- und Tabellenbezug

- `reports/experiments/M01-LR-002_fold_results.csv`
- `reports/experiments/M01-LR-002_summary.json`
- `reports/tables/logistic_class_weight_comparison.csv`
- `reports/figures/logistic_class_weight_metrics.pdf`
- `reports/figures/logistic_class_weight_cv.pdf`

## Reproduzierbarkeitshinweise

Die geteilte Spaltung, fünf Fälle, Pipeline und Sammlungsschlüssel übersprangen M01-LR-001. Die endgültige Testset blieb geschlossen.

## Verwendete Quellen und Werkzeuge

scikit-learn, pandas, Matplotlib, pytest, Python und das Experimentenregister.

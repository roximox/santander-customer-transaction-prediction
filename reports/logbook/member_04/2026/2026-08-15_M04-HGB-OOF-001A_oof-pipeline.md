# 2026-08-15 – M04-HGB-OOF-001A – Implementierung und Ausführung der HGB-OOF-Evaluation

**Mitglied:** Chaymae Akouaouch (Member 04)

**Kategorie:** Implementierung und Experiment

**Zugehöriges Experiment:** `M04-HGB-OOF-001`

**Zugehöriges Gruppentreffen:** 2026-08-16 – Erster individueller Analyse- und ML-Fortschritt

**Branch:** `feature/model-optimization`

**Zeitaufwand:** 8 h

## Ziel

Ich wollte für jede Entwicklungsbeobachtung eine Out-of-Fold-Prognose von einem Modell erzeugen, das diese Beobachtung nicht im Training gesehen hatte.

## Durchgeführte Arbeiten

- OOF-Evaluationsmodul und Runner implementiert.
- Fold-Zuordnung, positive Wahrscheinlichkeiten und Klassen bei der unveränderten Schwelle `0.5` erzeugt.
- Metriken sowie Daten für ROC-, Precision-Recall- und Confusion-Matrix-Artefakte berechnet und gespeichert.

## Tests und Validierung

Vollständigkeit der 160.000 Vorhersagezeilen, Schema, Fold-Abdeckung, Wahrscheinlichkeitsbereich und Ausschluss der Final-Test-Partition wurden geprüft. Automatisierte Schema- und Metriktests folgten am 22.08.

## Probleme und Herausforderungen

Große Vorhersage- und Kurvenartefakte mussten konsistent gespeichert werden. Ranking-Metriken und Schwellenmetriken mussten getrennt betrachtet werden.

## Ergebnisse

Es entstanden 160.000 OOF-Vorhersagen, 19.212 ROC-Kurvenpunkte, 160.001 Precision-Recall-Kurvenpunkte, eine Metrik-JSON-Datei und drei Abbildungen.

## Wissenschaftliche Interpretation

Die OOF-Vorhersagen ermöglichten eine zusammenhängende Analyse auf den Entwicklungsdaten, ohne die reservierte Testpartition zu öffnen.

## Entscheidungen und Erkenntnisse

Die Schwelle blieb bei `0.5`; es wurde keine Threshold-Optimierung durchgeführt.

## Nächste Schritte

ROC-AUC, Average Precision, Precision/Recall und Confusion Matrix wissenschaftlich interpretieren.

## Repository-Evidenz

- `src/gradient_boosting_evaluation.py`
- `scripts/run_gradient_boosting_evaluation.py`
- `reports/tables/M04-HGB-OOF-001_predictions.csv`
- `reports/tables/M04-HGB-OOF-001_roc_curve.csv`
- `reports/tables/M04-HGB-OOF-001_precision_recall_curve.csv`

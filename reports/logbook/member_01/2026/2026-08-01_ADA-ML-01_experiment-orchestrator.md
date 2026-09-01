# Logbucheintrag

## Metadaten

- Datum: 2026-08-01
- Mitglied: Yassine Elhari
- Sprint: Sprint 1
- Ticket ID: ADA-ML-01
- Branch: feature/data_processing
- Pull Request: [#2 — feature/data_processing → main](https://github.com/roximox/santander-customer-transaction-prediction/pull/2)
- Zeitaufwand: 3 Stunden
- Zugehörige Besprechung: [2026-08-09 — Data Processing, Validation Strategy and Start of Individual Analysis](../../../meetings/2026-08-09_data-processing-validation-strategy-and-start-of-individual-analysis.md)

## Titel

Experimentorchestrator

## Problem

Das Ausführen von Modellvergleichen unabhängig von Notebooks könnte die Wiederholung der Bewertung, Speicherung und Registrierungscode-Durchführung dupizieren.

## Durchgeführte Arbeiten

- Ein kleines Orchestrierungsmodul hinzugefügt, während die Kreuzvalidierung, Metriken, Fingerabdrucke, Speicherung und Registrierungsdaten in ihren bestehenden Modulen erhalten blieben.
- Explicit, path-sichere Experimentidentifizierungen erforderlich.
- Optional und deaktiviert als Standard gespeichert.
- Registrierung optional, aber nur nachdem die Ergebnisdateien gespeichert sind.
- Projektrelativen Ergebnispfade ausgewiesen und Wege außerhalb des Projekts abgelehnt.
- Die Regel erhalten, dass Entwicklungsexperimente nur mit Trainingsdaten versorgt werden.
- Ein reiner Hilfsprogramm hinzugefügt, das überprüfte Fakten für manuelle Logbuch-Eingabe extrahiert; es schafft keine wissenschaftliche Interpretation und schreibt kein Datei.
- Offline-Einheitstests und ein synthetisches Rauchtest hinzugefügt, der einen temporären Ordner verwendet und nicht die wissenschaftliche Registrierung ändert.

## Entscheidungen

`evaluation.py` verantwortlich für Bewertung und Speicherungsprimitiven; `experiments.py` nur sie koordiniert. Duplikate Ergebnisdateien und duplizierte Registrierungsidentifikatoren werden abgelehnt, anstatt überschrieben. Wenn die Registrierungsschreibung nach dem Speichern fehlschlägt, melden sich Fehlerberichte, dass die Ergebnisdateien vorhanden bleiben.

## Wissenschaftliche Ergebnisse

Kein Santander wissenschaftliches Experiment wurde für diesen Ticket durchgeführt. Der Rauchtest ist technisch und synthetisch nur.

## Reproduzierbarkeitsnotizen

Führen Sie `pytest`, dann `python scripts/verify_experiment_orchestrator.py` aus dem Projekt-Root. Das Rauchskript erstellt und löscht seine temporären Artefakte. Die letzte Testset wurde nicht verwendet und blieb geschlossen.

## Nächster Schritt

Führen Sie den ersten reellen `DummyClassifier` Baseline als separat identifizierten und dokumentierten wissenschaftlichen Experiment durch.

## Schwierigkeiten

Die Speicherung muss vorbeugend vor leisen ID-Überprüfungen und Meldungen von teilweiser Schreibfehlern bei der Registrierungsinsertion schützen.

## Anpassungen und Abweichungen vom Plan

Das Speichern ist opt-in, und das Rauchworkflow verwendet einen temporären Ordner entfernt, nachdem er die wissenschaftliche Registrierung verwendet hat.

## Abgelehnte Ansätze

Automatische IDs, Überschreibungen, automatische Interpretationen und Endtestargumente wurden abgelehnt.

## Geänderte Dateien

- `src/experiments.py`
- `scripts/verify_experiment_orchestrator.py`
- `tests/test_experiments.py`
- `CONTRIBUTING.md`

## Code-Referenzen

`run_experiment`, `run_and_save_experiment` und Registrierungsvalidierung in `src/experiments.py`.

## Abbildung und Tabelle-Bezüge

Keine; kein Santander wissenschaftliches Experiment wurde für diesen Infrastruktur-Ticket durchgeführt.

## Verwendete Quellen und Werkzeuge

Python, pandas, scikit-learn, pytest, CSV, JSON und temporäre Ordner.

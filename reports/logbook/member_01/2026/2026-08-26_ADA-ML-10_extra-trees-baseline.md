# Logbucheintrag

## Metadaten

- Datum: 2026-08-26
- Mitglied: Yassine Elhari
- Sprint: Sprint 2
- Ticket ID: ADA-ML-10
- Branch: develop
- Pull Request: Nicht anwendbar — direkt in `develop` (`190c7c5`) übergeben
- Aufgewendete Zeit: 5 Stunden
- Zugehörige Besprechung: [2026-08-26 — Entscheidung über den Endmodellauswahl](../../../meetings/2026-08-26_final-model-selection-decision.md)

## Titel

Mitglied 01 Baseline für Extra Trees

## Ziel

Das erwartete Vorkaliberspannung durch die Implementierung und Registrierung eines reproduzierbaren Extra Trees-Baseline unter Mitglied 01 erreichen.

## Kontext

Nachdem Mitglied 02s Entscheidungsbaum- und Random Forest-Ergebnisse integriert wurden, wurde das Modellauswahlüberblick immer noch `EXTRA_TREES` als fehlend gemeldet. Die Teammitglieder haben dieses verbleibende Baseline an Mitglied 01 zugewiesen. Das Experiment muss die gleichen Entwicklungsunterteilung, Validierungsfelder, Metriken und Zufallszustand wie alle anderen registrierten Kandidaten verwenden.

## Durchgeführte Arbeit

Eine dedizierte Trainings-only-Eingangsstelle für `M01-ET-001` hinzugefügt, die gemeinsame Extra Trees-Fabrik und Experimentorchestrator wiederholt verwendet, bestätigte beide offiziellen Spaltenspuren, fügte eine Überforderungsschutz und Sicherheitsmaßnahmen hinzu, lief das registrierte fünffache Experiment aus, bereitete die Modellauswahlberichte für Wiederaufbereitung aus den neuen gespeicherten Beweisen vor.

## Methodik

Der Estimator verwendet 200 Bäume, `max_depth=8`, `class_weight="balanced"`, `random_state=42` und paralleler Baumbau. Die Bewertung verwendet die gemeinsame fünffache zufällige `StratifiedKFold` auf der 160.000-Spalten-Entwicklungspartition. ROC-AUC ist Hauptwert; Durchschnittliche Präzision, F1, Genauigkeit, Genauigkeitsrate und ausgeglichenen Genauigkeitsrate werden durch die gemeinsame Bewertungseinrichtung berichtet.

## Ergebnisse

Die registrierten fünffachen Entwicklungs-CV-Ergebnisse sind:

- ROC-AUC: `0.847946 ± 0.003073`;
- Durchschnittliche Präzision: `0.475459 ± 0.007537`;
- F1: `0.441943 ± 0.002290`;
- Genauigkeit: `0.330575 ± 0.001983`;
- Genauigkeitsrate: `0.666500 ± 0.004820`;
- ausgeglichenen Genauigkeitsrate: `0.757859 ± 0.002129`.

Nach Wiederaufbereitung wurden 13 registrierte Experimente gefunden, 11 Kandidaten waren für die Auswahl geeignet, keine erwartete Modellfamilie war fehlend und die Auswahlstatus wurde auf `ready_for_group_review` geändert. Extra Trees hat den bisher fehlenden Ensemble-Familienbeweis zur Zwischenvergleichszeit geliefert. Seine Leistung muss neben der Rangqualität, Schwellenwertmetriken, Passungszeit und allgemeiner Generierungslücke überprüft werden; seine Anwesenheit impliziert nicht, dass es sich um das endgültige Modell handelt.

## Entscheidung

`M01-ET-001` als Basiskandidat registrieren und in den transparenten Mehrkriteriengruppenüberprüfung einbeziehen. Den finalen Test bis zur Gruppe auswählen, bevor die Pipeline geöffnet wird.

## Schwierigkeiten

Historische Modellauswahlergebnisse sind über das Verfahren geschützt und müssen vor der Wiederaufbereitung gesichert werden. Das Experiment ist aufgrund der fünf unabhängigen Baumensembles computationally schwerer als die linearen Basenlinien.

## Anpassungen und Abweichungen vom Plan

Extra Trees wurde ursprünglich unter Mitglied 02s Überblick gelistet. Die Verantwortung wurde nach Mitglied 02s Entscheidungsbaum- und Random Forest-Ergebnissen übergeben. Nur die erwartete Eigennamen-Metadaten änderten sich; der gemeinsame wissenschaftliche Protokoll wurde nicht geändert.

## Abgelehnte Ansätze

- Ein fälschlicherweise erstelltes Extra Trees-Ergebnis, um den Überblickstatus zu klären.
- Den reservierten finalen Test für die Modellauswahl verwenden.
- Die Parameter gegenüber der Leistung des finalen Tests anpassen.
- Eine bestehende Experiment oder Registry-Eintrag überreiben.
- Ein endgültiger Sieger aus dieser einzelnen Baseline-Kandidaten Aussage ermitteln.
- Den Überblick für dieses einzelne Baseline-Bewertungsergebnis aussprechen.

## Geänderte Dateien

- `scripts/run_extra_trees_baseline.py`
- `src/model_selection.py`
- `tests/test_extra_trees_experiment.py`
- `tests/test_project_structure.py`
- `tests/test_logbooks.py`
- `reports/experiments/M01-ET-001_fold_results.csv`
- `reports/experiments/M01-ET-001_summary.json`
- `reports/experiments/experiment_registry.csv`
- Regenerierte Dateien unter `reports/model_selection/`

## Code-Referenzen

Estimatorkonstruktion verwendet `create_extra_trees_classifier()` aus `src/modeling.py`. Datenlast, Spaltensicherheit, Bewertung, Speicherung und Registrierung verwenden `src/data.py`, `src/validation.py`, `src/evaluation.py` und `src/experiments.py`.

## Figuren- und Tabellenbezüge

- `reports/experiments/M01-ET-001_fold_results.csv`
- `reports/experiments/M01-ET-001_summary.json`
- `reports/experiments/experiment_registry.csv`
- `reports/model_selection/model_comparison_eligible.csv`
- `reports/model_selection/model_selection_coverage.csv`
- `reports/model_selection/model_selection_summary.json`

## Reproduzierbarkeitshinweise

Von der Repository-Root mit
`MPLCONFIGDIR=/tmp/ada-mpl-cache python scripts/run_extra_trees_baseline.py`.
Die Befehl weigert bestehende Ergebnisse. Die reservierte Spalte wird vor Bewertung gelöscht, und nicht verwendet.

## Nächster Schritt

Regeneriere die Vorkaliberspannungsergebnisse, überprüfe das komplette erwartete-Familienüberblick mit der Gruppe und sperre eine Pipeline bis zur Gruppenauswahl.

## Verwendete Quellen und Werkzeuge

- Bestehende gemeinsame Extra Trees-Fabrik und Experimentframework.
- scikit-learn-Dokumentation für `ExtraTreesClassifier` reflektiert durch die Fabrik-API.
- Repository-lokale pytest-Schutzmaßnahmen und aufgezeichnete CV-Artikel.
- Kein endgültiges Ergebnis oder externer ungespeicherter Wert wurde verwendet.

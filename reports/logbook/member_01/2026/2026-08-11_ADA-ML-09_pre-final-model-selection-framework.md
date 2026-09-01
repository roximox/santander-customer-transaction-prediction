# Logbucheintrag

## Metadaten

- Datum: 2026-08-11
- Mitglied: Yassine Elhari
- Sprint: Sprint 2
- Ticket ID: ADA-ML-09
- Branch: feature/data_processing
- Pull Request: [#7 — develop → main](https://github.com/roximox/santander-customer-transaction-prediction/pull/7) (integriert in `develop` in `ff3c2f8`)
- Aufgewendete Zeit: 8,5 Stunden
- Zugehörige Besprechung: [2026-08-23 — Modellfortschritt, Optimierung und Bewertung](../../../meetings/2026-08-23_model-progress-optimization-and-evaluation.md)

## Titel

Framenummerungsframework für die vorletzte Phase

## Ziel

Ein gemeinsames Leseband, das vergleichbare trainierten-CV-Ergebnisse vor dem Gruppenabschluss vergleicht. Das Framework muss während der Abwesenheit einiger Mitglieder noch nützlich bleiben.

## Kontext

Unterschiedliche Modellfamilien und Ergebnisproduzenten können unter verschiedenen JSON-Schlüsseln kompatibles Information preisen. Eine normalisierte Vergleichung ist erforderlich, um Modelldeckung, Protokollmetadaten, Metrik-Abwägungen und fehlende Beweise explizit zu machen.

## Durchgeführte Arbeit

Das Skript liest die sechs registrierten M01-Experimente und die beiden behaltenen Logistik-Kandidaten aus dem aufgezeichneten Grid-Suchvorgang. Implementiert wurde ein deterministischer Zusammenfassungsentdeckung, robuste Normalisierung, Eligibilitäts- und Exklusionsberichterstattung, Protokollvergleichsprüfung, -spezifische Ränge, eine CV-Variabilitäts-Kompetitivitäts-Funktion, multi-kritische Entscheidungen, erwartete Familiendeckung, Portfolio-Exporte, Besprechungsnotizen, Figuren und ein Offline-Testset. Das Skript liest die sechs registrierten M01-Experimente und die beiden behaltenen Logistik-Kandidaten aus dem aufgezeichneten Grid-Suchvorgang.

## Methodik

ROC-AUC bleibt das vorher festgelegte Hauptmetrik. Durchschnittliche Genauigkeit, F1-Wert, Präzision, Erinnerung, ausgeglichenes Genauigkeitsmaß, Verbreitung in der Ladephase, Anpassungszeit, Konvergenz, Anzahl der Merkmale und Nachweis sind getrennt. Keine gewichtete oder kombinierte Rangliste wird berechnet. Die Vergleichsprüfung unterscheidet zwischen konsistenten, inkonsistenten und nicht überprüfbaren Metadaten. Fehlende Kandidaten werden stattdessen als solche gemeldet.

## Ergebnisse

Das generierte Bericht zeigt vier geeignete Logistik-Kandidaten auf und ausschließt vier Dummy-Baselines. Mitglieder 02-04 und ihre erwarteten RF, Extra Trees, PCA, Feature Selection und HGB-Kandidaten fehlen. Folglich ist die Auswahlstatus `waiting_for_additional_models`; keine Gruppen Sieger wird ausgesprochen.

## Interpretation

Die gewonnenen Metriken und die konkurrierenden Kandidaten sind Eingabewerte, nicht eine endgültige Auswahl. Die CV-Variabilitätsregel verwendet einen Standarddeviationswert des besten aufgezeichneten ROC-AUC und ist explizit ein heuristischer Ansatz und kein formeller nicht-unterlegenheits-Test.

## Entscheidung

Ein transparenter multi-kritischer Überprüfungsprozess. Die Kandidatenstatus ist von der endgültigen Modellentscheidung getrennt. Die endgültige Pipelineentscheidung wird auf die Gruppe nach Abschluss der Deckung und Protokollvergleichs übergeben.

## Schwierigkeiten

Die Registrierungsexperiment-Summarisierungen und die Grid-Suchsummarisierungen verwenden unterschiedliche Feldnamen. Die Suchkandidaten fehlen auch einige Protokollmetadaten, die in den registrierten Experimenten verfügbar sind, so dass die Vergleichbarkeit nur teilweise überprüft werden kann.

## Anpassungen und Abweichungen vom Plan

Die beiden ausgewählten Logistik-Konfigurationen werden direkt von der bestehenden Kandidatencsv mit `source_type=grid_search_candidate` normalisiert. Keine Änderung des Experimentenregistrierungsartikels oder historischer wissenschaftlicher Artefakte.

## Abgelehnte Ansätze

- Wiederholte Kandidaten oder die Lade von OpenML-Daten.
- Die Lesung oder Berechnung der endgültigen Testpunkte.
- Die Auswahl nach Genauigkeit allein.
- Eine arbitrarische gewichtete Rangliste oder eine nahezu durchschnittliche Rangliste.
- Erfinden fehlender Mitglieder 02-04-Experimente oder eine Gruppenentscheidung.

## Geänderte Dateien

- `src/model_selection.py`
- `scripts/build_model_selection_report.py`
- `tests/test_model_selection.py`
- `tests/test_project_structure.py`
- `notebooks/08_final_evaluation.ipynb`
- `README.md`
- `CONTRIBUTING.md`
- `reports/model_selection/` - Aus exports und Besprechungsnotizen
- vier vorletzte Vergleichsfiguren

## Code-Referenzen

Zusammenfassung/Loading/Normalisierung, Modellfamilienabschätzung, Eligibilität, Vergleichbarkeit, Ränge, Wettbewerbsfähigkeit, Entscheidungen, Deckung, Portfolio, und Figuren in `src/model_selection.py`; Berichtsorchestrierung in `scripts/build_model_selection_report.py`.

## Figur- und Tabellennachweise

- `reports/model_selection/model_comparison_all.csv`
- `reports/model_selection/model_comparison_eligible.csv`
- `reports/model_selection/model_comparison_excluded.csv`
- `reports/model_selection/model_selection_decision.csv`
- `reports/model_selection/model_selection_coverage.csv`
- `reports/model_selection/model_selection_summary.json`
- `reports/model_selection/model_selection_comparability.json`
- `reports/model_selection/model_comparison_portfolio.csv`
- `reports/model_selection/model_comparison_portfolio.md`
- `reports/model_selection/group_model_selection_notes.md`
- `reports/figures/final_model_comparison_roc_auc.pdf`
- `reports/figures/final_model_comparison_average_precision.pdf`
- `reports/figures/final_model_comparison_threshold_metrics.pdf`
- `reports/figures/final_model_performance_vs_time.pdf`

## Reproduzierbarkeitshinweise

Eingangsdaten sind unveränderliche aufgezeichnete JSON/CSV-Artikel. Entdeckung und Rangierung sind deterministisch. Pfade in den Ausgabewerten sind Projektbezüglich. Keine Datenbank, Estimator, Modellisierung, Internetdienst oder endgültige Testset wird verwendet.

## Nächster Schritt

Gruppenbesprechung und endgültige Modellentscheidung nach Veröffentlichung der von Mitgliedern 02-04 veröffentlichten, gemeinsamen Protokollkandidaten. Die endgültige Test bleibt geschlossen bis zu diesem Zeitpunkt.

## Quellen und verwendete Werkzeuge

Bestehende Experiment-Summarisierungen und Projektbewertungskonventionen, Python, pandas, NumPy, Matplotlib, JSON, pytest, nbformat und Git-Inspektionsbefehle.

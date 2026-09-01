# Logbucheintrag

## Metadaten

- Datum: 2026-08-01
- Mitglied: Yassine Elhari
- Sprint: Sprint 1
- Ticket ID: ADA-ML-00
- Branch: feature/data_processing
- Pull Request: [#2 — feature/data_processing → main](https://github.com/roximox/santander-customer-transaction-prediction/pull/2)
- Aufgewendete Zeit: 7 Stunden
- Zugehörige Besprechung: [2026-08-09 — Data Processing, Validation Strategy und Start der individuellen Analyse](../../../meetings/2026-08-09_data-processing-validation-strategy-and-start-of-individual-analysis.md)

## Titel

Gemeinsamer Modellbewertungsrahmen

## Ziel

Ein Standard für die Bewertung nur des Trainingsmodells so festlegen, dass die Ergebnisse von verschiedenen Mitgliedern und Modelfamilien wissenschaftlich vergleichbar und wiederholbar sind.

## Vergleichsrisiko

Unabhängige Faltengeneration, Metriken, Zeitkonventionen oder Platzierung der Vorverarbeitung könnten scheinbar vergleichbare Scores aus Materialien unterschiedlicher Protokolle liefern. Der gemeinsame Rahmen zentriert diese Entscheidungen und enthält keinen letzten Testset-Argument, reduziert sowohl ungewollte Protokolldrift als auch das Risiko von Leaks.

## Kreuzvalidierungsdesign

Das Framework verwendet fünf-Fold-StratifiedKFold mit Schütteln und die gemeinsame `random_state=42`. Die Stratification bewahrt die imbalances Zielverteilung innerhalb jeder Validierungsfold. Bestimmte trainings- und Validierungsindizes erhalten deterministische SHA-256-Fingerabdrücke, sodass Mitglieder überprüfen können, dass Modelle denselben Folds ohne persistieren von Zeileniveau verwendet haben.

## Metriken

Die ROC-AUC ist die konfigurierte Hauptmetrik, da sie die Rangierung über alle Klassifizierungsstufen misst. Die durchschnittliche Genauigkeit wird berichtet, da sie den Fokus auf die positive-Klassen-Retriebe unter der Klasse-Unabhängigkeit legt. F1, Präzision, Recall, Genauigkeit und ausgeglichenes Genauigkeitsmaß bieten komplementäre Sichtweisen. Safe binary Scorer unterstützen sowohl numerische Etiketten als auch die Santander-String-Etiketten und verwenden die 0-Trennungshandhabung, wo relevant.

## Ergebnisformat und Zeit

`evaluate_model_cv` gibt eine Zeile pro Falt mit Trainings- und Validierungsmetriken, Fitzeit, Score-Zeit und Faltgrößen zurück. Seine serialisierbare Zusammenfassung enthält aggregierte Mittelwerte und Bevölkerungsschwankungen, Konfiguration, Estimator-Klasse und Parameter, Zielverteilung, CV-Fingerabdrücke, Autorenschaftsdaten und einen abgeschlossenen Status. Kein gefittetes Estimator-Objekt wird in der Zusammenfassung serialisiert.

Der Export-Helfer schreibt eine Falt CSV und eine Zusammenfassungs-JSON ohne stummes Überladen. Der Registrierungs-Helfer speichert eine Zeile pro einzigartiger Experiment-ID und lehnt Doppelte ab. Kein Helfer registriert absolute lokale Wege.

## Leakschutz

Nur das Training-Daten kann an die Kreuzvalidierung weitergegeben werden. Jede gelernte Skalierung, Imputation, PCA oder Feature-Selektion muss innerhalb des scikit-learn `Pipeline` liegen und wird separat in jedem Falt gefittet. Das Framework führt keine Transformation vor `cross_validate`. Der letzte Testset bleibt geschlossen und wurde nicht in dieser Arbeit verwendet.

## Synthetische Überprüfung

Offline-Tests und der Überprüfungs-Script verwenden generierte Klassifizierungsdaten und einen minimalen `DummyClassifier` streng als technischer Rauchtest. Kein Santander-Modell wurde trainiert oder bewertet, keine synthetischen Ergebnisse wurden gespeichert als wissenschaftliche Experimente, und kein Experiment-Registrierung wurde erstellt.

## Grenzen

Diese Infrastruktur wählt keinen Entscheidungs-Schwellenwert, passt Parameter an, vergleicht wissenschaftliche Modelle oder bewertet den letzten Testset. Die Ausführungszeit kann mit der Hardware und dem parallelen Scheduling variieren, auch wenn die Fälle und Scores wiederholbar sind.

## Nächster Schritt

Die erste wissenschaftliche `DummyClassifier`-Baselinen auf dem gemeinsamen Trainings-Split unter Verwendung eines einzigartigen Experiment-ID und des gemeinsamen Rahmens umzusetzen.

## Entscheidung

Ein fünf-Fold-Stratified-KFOLD zu verwenden und sieben gemeinsame Metriken für jedes Modell, mit allen gelernten Vorverarbeitungen in jedem Pipelinen-Fold.

## Schwierigkeiten

Die Wahrscheinlichkeitsbasierte und Etikettbasierten Metriken erfordern unterschiedliche Estimator-Ausgabewerte, und die Zusammenfassung muss immer JSON-serialisierbar bleiben über pandas und NumPy-Typen.

## Anpassungen und Abweichungen vom Plan

Der Rauchtest verwendet nur synthetische Daten und schreibt keine wissenschaftlichen Ergebnisse ab.

## Abgelehnte Ansätze

Globale Vorverarbeitung, Genauigkeits-Only-Vergleich, implizites Testset-Evaluation und das Registrieren des Rauchtests-Output als Wissenschaft waren abgelehnt.

## Geänderte Dateien

- `src/evaluation.py`
- `scripts/verify_evaluation_framework.py`
- `tests/test_evaluation.py`
- `configs/config.yaml`

## Code-Referenzen

`create_cv`, Scoring-Konstruktion, `evaluate_model_cv`, Validierung und Export-Helfer in `src/evaluation.py`.

## Abbildung und Tabelle-Bezug

Keine; der Überprüfung ist synthetisch und nicht persistiert.

## Reproduzierbarkeitshinweise

Fünf-Fold-Stratified-KFOLD verwendet Schütteln und `random_state=42`. Die API akzeptiert nur Trainingsdaten; der letzte Testset blieb geschlossen.

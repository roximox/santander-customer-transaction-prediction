# Logbucheintrag

## Metadaten

- Datum: 2026-08-01
- Mitglied: Yassine Elhari
- Sprint: Sprint 1
- Ticket ID: ADA-DATA-03
- Branch: feature/data_processing
- Pull Request: [#2 — feature/data_processing → main](https://github.com/roximox/santander-customer-transaction-prediction/pull/2)
- Zeitaufwand: 2,5 Stunden
- Zuordnung zu einer Besprechung: [2026-08-09 — Data Processing, Validation Strategy und Start der individuellen Analyse](../../../meetings/2026-08-09_data-processing-validation-strategy-and-start-of-individual-analysis.md)

## Titel

Gemeinsame wiederholbare Trainings-/Testteilung

## Ziel

Eine einzige gemeinsame wiederholbare Trainings-/Testteilung erstellen, die für jeden Teammitglied verwendet werden muss,
während die Testpartition ausschließlich für die endgültige Bewertung reserviert wird.

## Entscheidungsgrundlage

Das Ziel ist ungleichmäßig mit 89,951% `False` und 10,049% `True`. Die Teilung verwendet daher eine Zielstratifikation,
um die 80% Trainings- und 20% Testpartitionen so nahe wie möglich an die ursprüngliche Verteilung anzunähern. Die Parameter kommen aus der gemeinsamen Konfiguration: `test_size=0.20`, `random_state=42` und Schleifen werden aktiviert.

Keine gelernten Vorverarbeitung, Auswahl von Merkmalen, Skalierung, Füllung oder andere Datenabhängige Transformationen erfolgen vor dem Indexauswahl. Die Teilung operiert auf der explizit optimierten float32-Funktionsmatrix. Tests bestätigen, dass die gleiche Ziel, Zeilenordnung, Parameter und zufälliger Zahlen generator für raw float64 und optimierte float32-Funktionen identische Indizes ergeben.

## Beobachtete Teilung

- Trainingszeilen: 160.000;
- Testzeilen: 40.000;
- Ursprüngliche Zielproportionen: `False=0.89951`, `True=0.10049`;
- Trainingsproportionen: `False=0.8995125`, `True=0.1004875`;
- Testproportionen: `False=0,8995`, `True=0,1005`;
- Maximaler Unterschied von der ursprünglichen Proportion: 1,00000000001e-05;
- Trainings-/Testindex Überschneidung: 0;
- Trainingsgescdtes Speichervolumen: 128,70 MiB;
- Testgescdtes Speichervolumen: 32,17 MiB.

## Reproduzierbarkeitssignaturen

- Trainingsindizes SHA-256: `61c403ec521d15ab9d6316606eba5acdfc22381cb764b6da76a65041ec11f477`
- Testindizes SHA-256: `bf7d43e492967dad1358676e9bf8355a910823077b59118014db12af3e26f586`

Die Signaturn sind ordnersensibel und enthalten nur Indexwerte und deren Typen;
sie enthalten keine lokale Pfad oder Dateninhalte. Teammitglieder können diese Hashs vergleichen, um identische Partitionen ohne den Austausch von Zeileniveau-Daten zu bestätigen.

## Validierung

Die Validierung überprüft die Teilungsgrößen, die vollständige Indexunion, Abwesenheit von Überschneidungen,
X/Y-Aligment, Spalten, Datentypen, unveränderte Merkmals- und Zielwerte, Zielproportionen, aufgezeichnete Parameter und deterministische Wiederholung mit dem gleichen zufälligen Zahlen generator. Nur Metadaten werden exportiert; X_train, X_test, y_train und y_test werden nicht gespeichert.

## Testsetpolitik

Das Testset ist bis zur endgültigen Bewertung geschlossen. Es darf keine Modellwahl,
Merkmalsauswahl, Vorverarbeitungsentscheidungen, Schwellenwertbestimmung oder
Hyperparameter-Tuning informieren. Alle mittleren Entscheidungen müssen die Trainingsdaten und Trainings-Only-Validierungsverfahren verwenden.

## Nächster Schritt

Ein Trainings-only `DummyClassifier` Baseline mit der gemeinsamen Teilung erstellen, ohne die Testleistung für iterativen Modellauswahl zu verwenden.

## Entscheidung

Diese 80/20 stratifizierte Teilung und ihre Signaturn für alle Mitglieder übernehmen;
die 40.000-Row-Testpartition bleibt bis zur endgültigen Bewertung geschlossen.

## Schwierigkeiten

Die Teilung musste genau wiederholbare Indizes ohne die Persistenz oder Exposition von Zeileniveau-Daten beweisen.

## Anpassungen und Abweichungen vom Plan

Nur Indexsignaturn und aggregierte Metadaten werden exportiert; Split-Objekte werden nie serialisiert.

## Abgelehnte Ansätze

Unstratifizierte Teilung, Mitgliedspezifische Teilungen, Vorverarbeitung vor der Teilung,
und die Verwendung des Testpartitions für Entwicklung wurden abgelehnt.

## Geänderte Dateien

- `src/data.py`
- `src/validation.py`
- `scripts/create_data_split.py`
- `tests/test_data.py`
- `tests/test_validation.py`

## Code-Referenzen

Split-Erstellung und Signatur-Validierung in `src/data.py` und
`src/validation.py`; Verifizierung in `scripts/create_data_split.py`.

## Figur und Tabelle-Bezugnahmen

- `reports/tables/train_test_split_summary.json`

## Reproduzierbarkeitshinweise

Die Teilung verwendet `test_size=0.20`, Stratifikation, Schleifen und
`random_state=42`. Die endgültige Testset wurde nur gefingeriert und blieb geschlossen.

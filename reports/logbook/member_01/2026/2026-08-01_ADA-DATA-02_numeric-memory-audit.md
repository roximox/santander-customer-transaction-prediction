# Logbucheintrag

## Metadaten

- Datum: 2026-08-01
- Mitglied: Yassine Elhari
- Sprint: Sprint 1
- Ticket ID: ADA-DATA-02
- Branch: feature/data_processing
- Pull Request: [#2 — feature/data_processing → main](https://github.com/roximox/santander-customer-transaction-prediction/pull/2)
- Aufgewendete Zeit: 5 Stunden
- Zugehörige Besprechung: [2026-08-02 — Projektstruktur und gemeinsame Datenbasis](../../../meetings/2026-08-02_project-structure-and-common-data-foundation.md)

## Titel

Wissenschaftliche Raw-Daten- und numerische-Memory-Audit

## Ziel

Eine reproduzierbare wissenschaftliche Prüfung der Raw-Funktionen durchzuführen und die Auswirkungen einer expliziten Umwandlung von float64 in float32 ohne die Ziel- oder die Standardladeverhalten zu ändern.

## Raw-Datenbeobachtungen

Die OpenML-ID 45566 lief mit 200.000 Zeilen und 200 float64-Funktionen. Der Gesamtdatenspeicher für Funktionen und Ziel betrug 305,37 MiB; die Funktionen allein nutzten 305,18 MiB. Die Prüfung fand keine fehlenden Werte, unendliche Werte, wiederholte Zeilen, wiederholte Funktionennamen, konstante Funktionen oder quasi-constante Funktionen unter der dokumentierten 99%-Hochwertregel.

Die binäre kategorische Ziel ist ungleichmäßig: `False` stellt 179.902 Zeilen (0,89951) dar, während `True` 20.098 Zeilen (0,10049) darstellt. Keine Zielwerte oder -typen wurden geändert.

## Datennamen

Das Projekt befasst sich mit **Santander Customer Transaction Prediction**, während der Name durch OpenML als `SantanderCustomerSatisfaction` zurückgegeben wurde. Beide Fakten werden unabhängig voneinander als `project_dataset_name` und `openml_dataset_name` aufbewahrt. Der Quellennamen wurde nicht neu geschrieben oder versteckt.

## float64 / float32-Vergleich

Die explizite Umwandlung einer Kopie änderte alle 200 numerischen Funktionen-Daten typen in float32. Die Speicherung der Funktionen verringerte sich von 305,1759 MiB auf 152,5880 MiB, was einem Speicherplatzverlust von 152,5879 MiB oder 49,99998% entspricht.

Beobachtete Darstellungsunterschiede waren:

- Maximumer absolute Fehler: 3,7963867214330094e-06;
- Durchschnittlicher absolute Fehler: 1,9811511402243887e-07;
- Maximumer relative Fehler: 5,950797554674405e-08;
- Durchschnittlicher relative Fehler: 2,130516379585579e-08;
- Genauigkeitsänderungen bei numerischen Werten: 39.936.663 (99,8416575%).

Der relative Fehler wurde nur für nicht-null, endliche ursprüngliche Werte berechnet und umfasste die Positionen von Null. Diese Positionen blieben im Rahmen der genauen Änderungszahl aufgenommen.

Die Form, Index, Spalten, fehlende Wertpositionen und unendliche Wertpositionen wurden alle erhalten, und keine Überlauf wurde eingeführt.

## Entscheidung

Float32 wird provisorisch als empfohlenes optimiertes Funktion-Typ-Setungsmaßnahme angenommen, da es den Speicherplatz halbiert, während die Struktur und besondere Werte erhalten werden, mit kleinen gemessenen Darstellungsfehlern. Die Raw-OpenML-Ladeverhalten bleibt float64 standardmäßig. Downstream-Arbeit muss explizit in die Umwandlung optieren, und spätere Modellempfindlichkeitsprüfungen sollten bestätigen, dass die Präzisionsreduktion keinen erheblichen Einfluss auf die Bewertungsergebnisse hat.

## Implementierung und Tests

Das gemeinsame Datenmodul bietet nun pro-Funktion eine per-Funktionen-Prüfung, explizite numerische Umwandlung, Darstellungsfehler/Vergleich, und Umwandlungsvalidierung. Die Prüfungsskript exportiert strenge JSON-Summarisierungen und eine 200-reihige Funktionstabelle ohne die Datenbank zu speichern.

## Grenzen

Dieses Audit ist beschreibend und nicht umfassend. Es bewertet keine Funktionenbeziehungen, Ausreißer, Vorhersagewert, Leckagen oder Modellempfindlichkeit bei float32. Die quasi-constante Klassifizierung hängt von der expliziten 99%-Hochwertregel ab und sollte als Audit-Flag interpretiert werden, nicht als Grund für automatische Funktionenlöschung.

## Nächster Schritt

Erstellen Sie eine gemeinsame reproduzierbare getrennte Trainings-Test-Spalte während der Erhaltung der Testpartition von allen Vorbereitungs- und Modellauswahlentscheidungen.

## Schwierigkeiten

Float32 ändert die genaue binäre Darstellung der meisten Werte, sodass die Strukturbeibehaltung und die finiten relativen Fehler von der exakten Gleichheit getrennt werden mussten.

## Anpassungen und Abweichungen vom Plan

Die Raw-Ladeverhalten bleibt float64; die Speicherplatzoptimierung ist ein explizites downstream-Kopie-Validiertes, das unabhängig von einem impliziten Loader-Mutation erfolgt.

## Abgelehnte Ansätze

Automatische Funktionenlöschung, das Runden der Float32-Werte als Korruption und die Verwendung des Audit als umfassendes EDA wurden abgelehnt.

## Geänderte Dateien

- `src/data.py`
- `scripts/run_data_audit.py`
- `tests/test_data_audit.py`
- `notebooks/01_data_audit.ipynb`

## Code-Referenzen

Numerische Umwandlung, Funktionprüfung, Vergleich und Validierung-Funktionen in `src/data.py`; Ausführungs-Eingangs-Punkt in `scripts/run_data_audit.py`.

## Figuren- und Tabellen-Bezüge

- `reports/tables/data_audit_summary.json`
- `reports/tables/dtype_comparison.json`
- `reports/tables/feature_audit.csv`

## Reproduzierbarkeitsnotizen

Die Berichte beschreiben OpenML ID 45566 als geladen auf 2026-08-01. Keine Zeile-level-Datenbank wurde gespeichert; die endgültige Testset wurde nicht verwendet und blieb geschlossen.

## Verwendete Quellen und Werkzeuge

OpenML, scikit-learn, pandas, NumPy, pytest, nbformat, Python

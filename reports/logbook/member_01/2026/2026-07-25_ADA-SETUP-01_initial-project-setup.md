# Logbucheintrag

## Metadaten

- Datum: 2026-07-25
- Mitglied: Yassine Elhari
- Sprint: Sprint 0
- Ticket ID: ADA-SETUP-01
- Branch: main
- Pull Request: Nicht anwendbar — direkt in `main` (`55c60d6`) eingefügt
- Aufgewendete Zeit: 5,5 Stunden
- Zuordnung zu einem Treffen: [2026-07-26 — Initialisierung der Projektplanung und Aufteilung von Aufgaben](../../../meetings/2026-07-26_initial-project-planning-and-task-distribution.md)

## Titel

Initialisierung des gemeinsamen, reproduzierbaren technischen Grundgerüsts

## Ziel

Ein gemeinsames, reproduzierbares technisches Fundament schaffen, das es vier Gruppenmitgliedern ermöglicht, unabhängig voneinander zu arbeiten, während sie dieselben Konventionen, Einstellungen und Projektstruktur befolgen.

## Kontext

Das Projekt erfordert eine gemeinsame Analyse des Santander Customer Transaction Prediction-Datasets. Bevor man die Daten herunterlädt oder EDA- und Modellimplementierungen durchführt, benötigt das Team einen gemeinsamen Repository-Struktur, um unkompatiblen Datenlastprozesse, inkonsistente Splits und wiederholte Code zu vermeiden.

## Durchgeführte Arbeit

- Erstellung des Projektverzeichnissesstrukturs und der Hauptdatei `README.md`.
- Hinzufügung von Zusammenarbeitseingaben in `CONTRIBUTING.md` und Auslassungen in `.gitignore`.
- Hinzufügung von Anforderungsdateien (`requirements.txt`, `environment.yml`, `pyproject.toml`).
- Hinzufügung der zentralen Konfigurationsdatei (`configs/config.yaml`).
- Erstellung des reusiblen Python-Pakets unter `src/`.
- Hinzufügung von acht ordnenden Notizblöcken ohne generierte Ergebnisse.
- Hinzufügung von Offline-Tests für Projektstruktur und Konfiguration.
- Hinzufügung von Agile-Treffen, Experimenten und Logbuchmustern.
- Initialisierung des lokalen Git-Repository.
- Ausführung von `pytest`.
- Lade und Ausgabe der zentralen Konfigurationsdatei.

## Methodik

Die Einrichtung folgte den reproduzierbaren Forschungs- und Software-Engineeringprinzipien: eine zentrale Konfiguration, relative Wege, ein isoliertes dokumentiertes Umfeld, keine Daten in Git, modulares Quellcode, automatisierte Offline-Tests, gemeinsame Branchkonventionen und Trennung zwischen stabilen und zukünftigen Integrationsbrechen.

## Ergebnisse

- Vier Tests lieferten sich während der initialen Validierung.
- Die zentrale Konfigurationsdatei (`configs/config.yaml`) wurde erfolgreich geladen.
- Der OpenML-ID wurde als `45566` bestätigt.
- Das `random_state` wurde als `42` bestätigt.
- Die `test_size` wurde als `0.20` bestätigt.
- Das Zielkolonne blieb `null` und musste gegen das reale Dataset verifiziert werden.
- Der primäre Metrikwert wurde als `roc_auc` bestätigt.
- Das lokale Git-Repository wurde erfolgreich initialisiert.
- Keine Daten wurden erstellt oder eingefügt.

## Interpretation

Das technische Fundament ist funktional und bereit für die Zusammenarbeit. Es benötigt jedoch noch eine Validierung in der offiziellen Python 3.11-Umgebung, bevor wissenschaftliche Arbeit beginnen kann.

## Entscheidung

- Bewahren Sie diese Struktur als gemeinsames Projektgrundgerüst auf.
- Erhalten Sie gemeinsame wissenschaftliche Standards in der zentralen Konfigurationsdatei.
- Vermeiden Sie wissenschaftliche Entscheidungen mit dem Endtestset.
- Lassen Sie keine Daten in Versionierung.
- Verwenden Sie individuelle Featurezweige nachdem das Repository gepusht wurde.
- Bewahren Sie separate Logbuchordner für alle vier Mitglieder auf.

## Schwierigkeiten

- `pytest` zeigte eine `pytest-nbgrader` Warnung, weil dieses Plugin in der aktuellen Anaconda Basisumgebung installiert ist.
- Das Git-Repository war nicht initialisiert, sodass `git status` vor `git init` fehlte.
- Das Git-System wählte `master` als Initialisierungsbranchname automatisch.

## Anpassungen und Abweichungen vom Plan

- Das Git-Repository wurde nach der Validierung des ersten Setup eingefügt.
- Die erste Branch musste von `master` in `main` umbenannt werden, bevor die erste Commit-Menge erstellt wurde.
- Die offizielle Umgebung spezifiziert Python 3.11, während die initialen Tests mit dem beobachteten Python 3.13.9-Basisumfeld verwendet wurden.

## Abgelehnte Ansätze

- Das direkte Einfügen der Daten in Git wurde abgelehnt.
- Die Einführung von EDA oder Modellimplementierungen vor der Validierung des gemeinsamen Grundgerüsts wurde abgelehnt.
- Die Festlegung eines Zielkolonnens vor dem Inspektion des realen Datensatzes wurde abgelehnt.

## Dateien geändert

- `README.md`, `CONTRIBUTING.md`, `.gitignore`
- `requirements.txt`, `environment.yml`, `pyproject.toml`
- `configs/config.yaml`
- `src/`
- `notebooks/`
- `tests/`
- `reports/`
- `models/`
- `data/`

## Code-Referenzen

- `src/config.py`
- `configs/config.yaml`
- `tests/test_config.py`
- `tests/test_project_structure.py`

## Abbildung und Tabelle-Bezug

Keine wissenschaftliche Produktion wurde während der Projekt-Einrichtung erzeugt.

## Reproduzierbarkeitshinweise

Die zentrale `random_state` ist zentralisiert, konfigurierte Wege sind relativ, Abhängigkeiten werden dokumentiert und die Tests benötigen keinen Internetzugriff. Die Endtestset wurde nicht verwendet und blieb geschlossen für Modellauswahl.

## Nächster Schritt

- Validieren Sie das Projekt in der offiziellen Python 3.11-Umgebung.
- Bestätigen Sie, dass die aktive Git-Branche `main` ist.
- Commiten und pushen Sie den überprüften Setup.
- Erstellen oder verwenden Sie die `develop`-Integrationsspalte.
- Beginnen Sie mit der separaten Datenherausforderung und -prüfung.

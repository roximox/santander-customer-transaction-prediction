# Logbucheintrag

## Metadaten

- Datum: 2026-08-24
- Mitglied: Yassine Elhari
- Sprint: Sprint 2
- Ticket ID: ADA-UI-01
- Branch: develop
- Pull Request: [#7 — develop → main](https://github.com/roximox/santander-customer-transaction-prediction/pull/7) (feature merged into `develop` in `37830b3`)
- Aufgewendete Zeit: 10,5 Stunden
- Zugehörige Besprechung: [2026-08-23 — Model Progress, Optimization and Evaluation](../../../meetings/2026-08-23_model-progress-optimization-and-evaluation.md)

## Titel

Interaktive Wissenschaftliche Ergebnis-Dashboard

## Ziel

Eine professionelle Streamlit-Oberfläche bereitzustellen, um das gesparte wissenschaftliche Beweismaterial ohne manuelle Öffnung von CSV, JSON und PDF-Artikeln zu erkunden.

## Kontext

Der Repository enthält jetzt registrierte Logistik-Regression, Feature-Selbstauswahl, PCA und HistGradientBoosting-Ergebnisse. Eine gemeinsame Leseberechtigungsschicht ist erforderlich für die Vorlesungen der Professoren und das e-Portfolio, während die Ergebnisse von Mitglied 02 nicht zugänglich sind.

Bevor man mit der UI-Arbeit begann, mussten die Teambeiträge auf der gemeinsamen `develop`-Zweigbranche zusammengefasst werden. Die Beiträge von Mitglied 03 wurden überprüft und in den gemeinsamen Daten, Validierung, Bewertung, Experiment und Modellauswahl-Infrastruktur eingebunden, die von Mitglied 01 unterhalten wird.

## Teamintegration und Konfliktlösung

Die Integration umfasste folgende Aktivitäten:

- Synchronisierte die Feature-Zweige mit dem aktuellen Remote-Repositoryzustand;
- Mischte die Beiträge von Mitglied 03 in `develop`;
- Mischte die Beiträge von Mitglied 04 in `develop`;
- Löste den Inhaltskonflikt in `reports/experiments/experiment_registry.csv` durch die Aufbewahrung der bestehenden M03-Experimenten und die Hinzufügung der M04 registrierten Experimenten;
- Bewahrte die einzigartigen Experimentidentifikatoren `M03-FS-001`, `M03-PCA-001`, `M04-HGB-001` und `M04-HGB-002`;
- Anpassete die Integrationstests, die angenommen hatten, dass Mitglied 04 keine Logbuch-Einträge hatte;
- Überprüfte, dass die integrierten Experimentartikel, Zusammenfassungen, Faltresultate und Registrierungsdaten konsistent blieben;
- Laufte das komplette Test-Set nach Konfliktlösung vor dem Push der validierten `develop`-Zustand.

Der Konflikt war additiv und nicht wissenschaftlich: Beide Zweige hatten gültige Experimentartikel in derselben Registrierungsstelle platziert. Die Lösung behielt daher beide Datensätze und änderte keine registrierten Punktzahlen.

## Nach-Konflikt-Wissenschaftliche Kompatibilitätsprüfung

Nach der Integration wurden die integrierten Beiträge von Mitglied 03 und Mitglied 04 gegenüber der gemeinsamen Infrastruktur von Mitglied 01 überprüft. Die Prüfung bestätigte den gemeinsamen Partitionssatz von 160.000 Zeilen, fünf-Fache `StratifiedKFold`, die gemeinsame Metrik und `random_state=42` für die registrierten Skripte. Es wurde auch ein methodologischer Fehler in den beiden M03-Notebooks entdeckt: Sie konnten das Pipeline auf dem gesamten Datensatz auswerten. Diese Notebooks wurden korrigiert, um die offiziellen Spaltensignaturen zu überprüfen, die vorbehaltene Partition zu löschen, die bestehenden M03-Produktionsfabriken wiederzunutzen und die bestehenden Experimentartikel ohne Wiederholung der offiziellen Experimente anzuzeigen.

Die vorletzte Modellauswahl-Berichtsübersicht wurde dann nach der M03/M04-Mischung überprüft. Anfangs war der Bericht noch von Mitglied 01, da historische Ausgabe-Dateien vor dem Überreiben geschützt waren. Nachdem der Bericht durch den vorgesehenen Workflow neu generiert wurde, wurden die verfügbaren Logistik-Regression-, Feature-Selbstauswahl-, PCA- und HistGradientBoosting-Kandidaten sichtbar, während weiterhin Random Forest und Extra Trees als fehlend angezeigt wurden.

## Durchgeführte Arbeit

Nach der Teamintegration und wissenschaftlicher Kompatibilitätsprüfung wurde eine dünne `app.py`, defensive geladene Artefakt-Loader, reusbare Plotly-Charts, Formatierungs-Hilfen, Streamlit-Komponenten, elf wissenschaftliche Seiten, globale Filter und einen Professor-Modus implementiert. Die Offline-Tests, Strukturprüfung, Launch-Dokumentation und explizite Endtest-Sicherungen wurden hinzugefügt.

## Methodik

Alle angezeigten Werte stammen aus bestehenden Dateien unter `reports/`. Das Dashboard importiert keine Datensatzlader oder Experimentausführungsmodul. Es passt nie einen Estimator, kontaktiert OpenML während des Startups, aktualisiert die Registrierung oder erstellt ein Endtest-Metric. Die Charts behalten 0-1 Metrikachsen, wobei Vergleiche sonst übertrieben sein könnten.

## Ergebnisse

Die Teambeiträge sind auf `develop` zusammengefasst, die Registrierungsdaten enthalten M03 und M04 registrierte Experimente ohne Duplikate, und das Suite passte vor Beginn der Dashboard-Arbeit. Die Oberfläche zeigt nun die Datenprüfung, Spaltensicherheit, Experiment-Registrierung, Faltmetrik, Interimmodell-Vergleich, Mitglied 01-Analysen, Mitglied 03-Dimensionenauswertungen, Mitglied 04-HGB-Evidenz, Lernkurven, Auswahlkoverage, Vergleichbarkeit und interimme Erkenntnisse. Fehlende Artefakte produzieren lesbare Warnungen.

## Interpretation

Das Dashboard trennt registrierte CV-Experimente von Nebenanalysen und hält die aktuelle Vergleichsweise explizit interim. Es unterstützt wissenschaftliche Überprüfung ohne zu zweiten Maschinenlern-Implementierung werden.

## Entscheidung

Mit Streamlit mit Pandas und Plotly als einziges visuelles Schichtengitter über den bestehenden wissenschaftlichen Artefakten. Bewahren Sie eine Ergebnissequelle für beide Normal- und Professor-Modus vor.

## Schwierigkeiten

Artikel-Schemas variieren sich in den registrierten Experimenten, Suchergebnissen, Lernkurven und Nebenanalysen. Defensive Loader und Überprüfung auf fehlende Spalten sind daher erforderlich. Mitglied 02-Koverage ist absichtlich unvollständig. Früher im Workflow wurden parallele Hinzufügungen zur gemeinsamen Experiment-Registrierung auch zu einem Konflikt geführt, der eine additivere und traceierbare Lösung erforderte.

## Anpassungen und Abweichungen vom Plan

PDF-Artikel werden durch ihre zugrunde liegenden gespeicherten CSV/JSON-Daten dargestellt, wenn verfügbar, damit die Oberfläche interaktiv bleibt. Keine falschen Screenshot oder fehlende Mitglied 02-Ergebnisse wurden erstellt.

## Abgelehnte Ansätze

- Modellneubildung innerhalb von Streamlit.
- Lade von OpenML während der Anwendungsausführung.
- Erstellung neu generierter Experiment- oder Auswahlartikel.
- Hinzufügung eines Aktionen, die den Endtest-Evaluation zugänglich macht.
- Wiederholte Aufrechterhaltung einer separaten Professor-Modus-Quellen.

## Geänderte Dateien

- `reports/experiments/experiment_registry.csv` während der Teamintegration
- `tests/test_logbooks.py` und `tests/test_project_structure.py` für Integrationsempfänglichkeit
- `notebooks/05_feature_selection.ipynb`
- `notebooks/06_pca.ipynb`
- `tests/test_feature_selection.py`
- `app.py`
- `src/dashboard/`
- `tests/test_dashboard.py`
- `tests/test_project_structure.py`
- `tests/test_logbooks.py`
- `requirements.txt`
- `README.md`
- dieser Logbuch-Eintrag

## Code-Referenzen

Die Teamintegration verwendete `src/data.py`, `src/validation.py`, `src/evaluation.py`, `src/experiments.py` und `src/model_selection.py`. Die korrigierten M03-Notebooks verwenden die Fabriken aus `src/feature_selection.py` und lesen ihre gespeicherten M03-Ergebnisartikel ohne Neubildung.

Die geladenen Artefakte werden in `src/dashboard/loaders.py` implementiert; interaktive Figuren werden in `src/dashboard/charts.py` implementiert; reusbare Präsentations-Elemente werden in `src/dashboard/components.py` implementiert; die Seiteorchestrierung und Navigation werden in `app.py` implementiert.

## Figur- und Tabellennachweise

- `reports/experiments/experiment_registry.csv`
- `reports/experiments/M03-FS-001_summary.json`
- `reports/experiments/M03-PCA-001_summary.json`
- `reports/experiments/M04-HGB-001_summary.json`
- `reports/experiments/M04-HGB-002_summary.json`
- `reports/model_selection/model_comparison_eligible.csv`
- `reports/model_selection/model_selection_summary.json`
- `reports/tables/data_audit_summary.json`
- `reports/tables/train_test_split_summary.json`
- `reports/tables/logistic_learning_curve_summary.csv`
- `reports/tables/M04-HGB-learning-curve.csv`

## Reproduzierbarkeitsnotizen

Installiere Abhängigkeiten mit `pip install -r requirements.txt`, dann startet man von der Repository-Root mit `streamlit run app.py`. Die gleichen gespeicherten Artefakte treiben sowohl die Anzeige-Modi vor. Der reservierte Test bleibt vorbehalten und kein Endtest-Metric wird gelesen oder berechnet; die reservierte Partition wird nicht verwendet.

## Nächster Schritt

Das komplette Offline-Test-Set laufen und den Startup-Smoke-Test durchführen, dann demonstrieren Sie das interim Dashboard der Gruppe. Integrieren Sie Mitglied 02-Artikel automatisch, sobald ihre registrierten Experimente verfügbar sind.

## Verwendete Quellen und Werkzeuge

- Streamlit-Dokumentation und installierte Laufzeit.
- Plotly und Pandas-APIs.
- Bestehende Projektartikelschemas und gemeinsame Modellauswahl-Tools.
- Repository-lokale pytest und nbformat-Validierung.
- Keine externe Modellneubildung oder Endtest-Evaluation wurde verwendet.

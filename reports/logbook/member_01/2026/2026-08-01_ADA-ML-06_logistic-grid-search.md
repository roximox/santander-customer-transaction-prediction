# Logbucheintrag

## Metadaten

- Datum: 2026-08-01
- Mitglied: Yassine Elhari
- Sprint: Sprint 1
- Ticket ID: ADA-ML-06
- Branch: feature/data_processing
- Pull Request: [#2 — feature/data_processing → main](https://github.com/roximox/santander-customer-transaction-prediction/pull/2)
- Aufgewendete Zeit: 7 Stunden
- Bezugnahme auf eine Besprechung: [2026-08-16 — Erster individueller Analyse- und Machine-Learning- Fortschritt](../../../meetings/2026-08-16_first-individual-analysis-and-machine-learning-progress.md)

## Titel

Logistische Regressionsuche

## Wissenschaftliche Frage

Welche Kombination aus L1/L2-Normalisierung, `C`, und Gewichtung der Klassen ergibt das beste überprüfbare Verhalten der Logistischen Regression auf dem gemeinsamen Ausbildungssetz?

## Protokoll und Begründung

`GridSearchCV` untersuchte exhaustiv die vorher festgelegte 2 × 5 × 2-Raum: Penalien L1/L2, `C` in 0,01, 0,1, 1, 10, 100 und Klassen- Gewichtung `None`/`ausgewogen`. Die 20 Kandidaten produzierten 100 Fits über das gemeinsame fünffache stratifizierte CV. Jeder Fit verwendete `StandardScaler` → `Logistische Regression (Solver="saga", max_iter=2000, random_state=42)`. Der ROC-AUC-Wert wurde als Refit-Metrik verwendet.

Nur der 160.000-Spalten-Float32-Ausbildungssetz war an den Suchvorgang übergeben. Die reservierte 40.000-Spalten-Testspiegel wurde bestätigt, dann wurden seine Objekte ohne Aufruf von `score`, `predict` oder `predict_proba` gelöscht.

## Rechenkosten und Konvergenz

Der vollständige Suchvorgang und Refit dauerten 974,66 Sekunden (16,24 Minuten), mit einer Durchschnittsdauer von 48,73 Sekunden pro Kandidaten und 9,75 Sekunden pro Fit mit `n_jobs=-1`. Sechs `ConvergenceWarning`-Nachrichten sagten, dass `max_iter=2000` erreicht wurde. Warnungen, die von parallelen Arbeitern ausgestrahlt wurden, konnten nicht zuverlässig auf individuelle Kandidaten zugeordnet werden; keine Scores wurden versteckt, geändert oder leise neu durchgeführt. Eine korrigierende Studie, falls erforderlich, muss mit einem neuen Such-Id verwendet werden.

## Ergebnisse und Metrikenpezitivgewinner

- ROC-AUC: `candidate_002`, L2, C=0,01, ungewichtete — 0,859201.
- Durchschnittliche Präzision: `candidate_005`, L1, C=0,1, ungewichtete — 0,507626.
- F1-Wert: `candidate_004`, L2, C=0,01, ausgewogen — 0,416119.
- Ausgewogene Genauigkeit: `candidate_004` — 0,778194.

`candidate_002` hat eine Präzision von 0,691427 und einen Ertragsgrad von 0,267758. Im Gegensatz dazu hat `candidate_004` eine Präzision von 0,284599 und einen Ertragsgrad von 0,773666. Die Klassen- Gewichtung ändert das Defaultverhalten der Schwellenwerte erheblich, während sich der ROC-AUC-Wert fast unverändert verhält.

## Vergleich und vorläufige Entscheidung

Das registrierte neutrale Basiskonfiguration M01-LR-001 hat einen ROC-AUC-Wert von 0,859188 und eine Durchschnittliche Präzision von 0,507566; der ROC-AUC-Sieger verbessert diese Werte um nur 0,000013 bzw. 0,000027. M01-LR-002 hat einen F1-Wert von 0,416059 und eine ausgewogene Genauigkeit von 0,778128; `candidate_004` verbessert diese Werte um nur 0,000060 bzw. 0,000066. Diese Differenzen sind relativ gering im Vergleich zur Faltveränderung. Kein einzelner Kandidaten ist die endgültige Geschäftsentscheidung ausgewählt: behalte `candidate_002` als den ROC-AUC-Selbstkandidaten, `candidate_005` als AP-Alternative und `candidate_004` als Alternative zur Ertragsgradorientierung.

## Abbildungen und Dokumente

Die ROC-AUC/C-Figur enthält Faltunsicherheit, die Gegenüberstellung der Schwellenwert-Metriken und das Trainings-Validierungs- Bild zeigt kleine allgemeine Gaps ohne starkes Überfitting-Signal. Die vollständigen Kandidaten-Ergebnisse und Zusammenfassungen werden unter `reports/searches/`; die Entscheidung und die Topp-Tabelle unter `reports/tables/`.

## Grenzen

Die Kreuzvalidierung ist Modellauswahlbeweis, nicht endgültiges Testleistung. Keine Schwellenoptimierung, Kalibrierung, definitive Koeffizientenauslegung oder Nichtlineare-Modellvergleich wurde durchgeführt. Der explizite `Penalty` ist durch scikit-learn 1,8 deaktiviert, obwohl er für diesen Suchvorgang erforderlich ist.

## Nächster Schritt

Analyse der Koeffizientenstabilität für ausgewählte lineare Kandidaten oder definieren Sie einen separaten korrigierenden Konvergenz-Suchvorgang mit einem neuen Such-Id.

## Schwierigkeiten

Sechs Warnungen, die von parallelen Arbeitern ausgestrahlt wurden, konnten nicht zuverlässig auf individuelle Kandidaten zugeordnet werden. Die Abschlussarbeit erforderte JSON-sichere Verarbeitung von fehlenden Kandidatenwerten; der vollständige Suchvorgang wurde nicht neu durchgeführt oder geändert.

## Anpassungen und Abweichungen vom Plan

Die vollständigen Kandidaten- und Zusammenfassungsdaten wurden nach der Korrektur der Serialisierung abgeschlossen, ohne einen zweiten teuren 100-Fit-Suchvorgang durchzuführen.

## Abgelehnte Ansätze

Kartieren paralleler Warnungen spekulativ, Auswahl auf dem Endtestset, Überzeichnen des Such-Id und Verstecken von Konvergenz-Warnungen wurden abgelehnt.

## Geänderte Dateien

- `src/search.py`
- `scripts/run_logistic_grid_search.py`
- `tests/test_search.py`
- `notebooks/03_logistic_regression.ipynb`

## Code-Referenzen

Suchraum, Serialisierung, Rangierung und Figuren-Hilfen in `src/search.py`; geschützte Ausführung und Abschluss in `scripts/run_logistic_grid_search.py`.

## Figuren- und Tabellenbezug

- `reports/searches/M01-LR-SEARCH-001_candidates.csv`
- `reports/searches/M01-LR-SEARCH-001_summary.json`
- `reports/tables/logistic_grid_search_decision_table.csv`
- `reports/tables/logistic_grid_search_top_candidates.csv`
- `reports/figures/logistic_grid_search_roc_auc.pdf`
- `reports/figures/logistic_grid_search_tradeoff.pdf`
- `reports/figures/logistic_grid_search_train_validation.pdf`

## Reproduzierbarkeitshinweise

Der Suchvorgang verwendete das 160.000-Spalten-Ausbildungssetz, fünf stratifizierte Fälle, `random_state=42` und 20 vorher festgelegte Kandidaten. Der Endtest-Set wurde nur fingerprint-verifiziert und blieb geschlossen.

## Quellen und Werkzeuge

scikit-learn `GridSearchCV`, pandas, NumPy, Matplotlib, pytest, JSON und Python.

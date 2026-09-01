# Projektbeitrag Member 04

**Chaymae Akouaouch**

**Schwerpunkt:** Modelloptimierung und Evaluation mit HistGradientBoosting

**Projekt:** Santander Customer Transaction Prediction

## Meine Aufgabe

Ich war für den HistGradientBoosting-Bereich (HGB) verantwortlich. Meine Aufgabe war es, eine reproduzierbare Baseline zu entwickeln, das Modell zu optimieren und seine Ergebnisse zu untersuchen.

Dabei nutzte ich die gemeinsame Daten- und Evaluationsinfrastruktur des Projekts. Diese Infrastruktur wurde nicht von mir implementiert.

## Mein Vorgehen

Mein Arbeitsablauf bestand aus folgenden Schritten:

1. HGB-Baseline `M04-HGB-001` erstellen und mit fünfteiliger Cross-Validation auswerten.
2. Mit `RandomizedSearchCV` 20 Hyperparameterkombinationen untersuchen.
3. Die ausgewählte Konfiguration mit einer Learning Curve prüfen.
4. OOF-Vorhersagen für die Entwicklungsdaten erzeugen und auswerten.
5. Baseline und optimiertes HGB-Modell vergleichen.
6. Die ausgewählte Konfiguration als `M04-HGB-002` registrieren.
7. Die zentralen HGB-Funktionen mit fünf eigenen Testdateien prüfen.
8. Ergebnisse und Abbildungen im HGB-Notebook zusammenführen.
9. Die HGB-Ergebnisse in die gemeinsame Modellwahl einbringen.

## Wichtigste Ergebnisse

| Metrik | Baseline `M04-HGB-001` | Optimiertes Modell `M04-HGB-002` |
|---|---:|---:|
| ROC-AUC | 0.884596 ± 0.003278 | 0.891449 ± 0.002836 |
| Average Precision | 0.572879 | 0.591089 ± 0.010028 |
| F1 | 0.387255 | 0.415248 |
| Train-Validierungs-Lücke bei ROC-AUC | 0.091063 | 0.082131 |

Das optimierte Modell erreichte bessere Werte als die Baseline und die Train-Validierungs-Lücke wurde kleiner. Die OOF-Auswertung bestätigte ein ähnliches Ergebnis mit einer ROC-AUC von `0.891438` und einer Average Precision von `0.590860`.

Bei der festen Schwelle `0.5` lag die Precision bei `0.795527`, der Recall aber nur bei `0.280943`. Das Modell erkannte daher weiterhin viele tatsächliche positive Fälle nicht. Eine nachträgliche Threshold-Optimierung wurde nicht durchgeführt.

Die Learning Curve zeigte, dass mehr Entwicklungsdaten die Validierungsleistung verbesserten. Die ROC-AUC stieg von `0.851063` bei 12.800 Trainingszeilen auf `0.891245` bei 128.000 Trainingszeilen. Die verbleibende Lücke zeigt, dass die Generalisierung weiterhin vorsichtig bewertet werden muss.

## Beitrag zur finalen Modellentscheidung

Ich bereitete die Ergebnisse des HGB-Modells für den gemeinsamen Vergleich vor und nahm an der Modellprüfung teil. Die Gruppe wählte `M04-HGB-002` gemeinsam als finales Modell aus. Im gemeinsamen Vergleich erreichte es die höchste mittlere CV-ROC-AUC (0.891449) und Average Precision (0.591089), war jedoch nicht bei jeder Metrik das beste Modell.

Das Model Selection Framework, der Final Model Lock und die Final Evaluation wurden nicht von mir implementiert. Nach der gemeinsamen Auswahl wurde das Modell über den Gruppenworkflow einmal auf der reservierten Final-Test-Partition ausgewertet.

## Wichtigste Dateien und Artefakte

- HGB-Module: `src/gradient_boosting*.py`
- Ausführungsskripte: `scripts/run_gradient_boosting*.py`
- HGB-Notebook: [`notebooks/07_gradient_boosting.ipynb`](../../notebooks/07_gradient_boosting.ipynb)
- Ergebnisse: `reports/experiments`, `reports/searches` und `reports/tables`
- Abbildungen: `reports/figures/M04-HGB-*.pdf`
- Tests: `tests/test_gradient_boosting*.py`
- [Member-04-Logbuch](../logbook/member_04/README.md)

## Verwendete Technologien

Python, scikit-learn, HistGradientBoosting, RandomizedSearchCV, pytest, Jupyter und Git/GitHub.

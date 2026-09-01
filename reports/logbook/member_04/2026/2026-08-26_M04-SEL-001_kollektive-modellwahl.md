# 2026-08-26 – M04-SEL-001 – HGB-Beitrag zur kollektiven Finalmodellprüfung

**Mitglied:** Chaymae Akouaouch (Member 04)
**Kategorie:** Meeting und Review
**Zugehöriges Experiment:** `M04-HGB-002`
**Zugehöriges Gruppentreffen:** 2026-08-26 – Entscheidung über die finale Modellauswahl
**Branch:** `feature/model-optimization`
**Zeitaufwand:** 2.5 h

## Ziel

Ich wollte die Ergebnisse des HGB-Modells und seine Einschränkungen für den gemeinsamen Modellvergleich vorbereiten.

## Durchgeführte Arbeiten

- Gespeicherte HGB-CV-, Learning-Curve-, OOF- und Vergleichsevidenz vorbereitet.
- Kandidaten gemeinsam geprüft und HGB-Stärken sowie Generalisierungs-, Laufzeit- und Recall-Grenzen erläutert.
- An der kollektiven Bestätigung von `M04-HGB-002` teilgenommen.

## Tests und Validierung

Ausgewählte Parameter und Threshold `0.5` wurden mit dem eingefrorenen Kandidaten abgeglichen. Während der Auswahl gab es keine weitere Optimierung.

## Probleme und Herausforderungen

Starke Ranking-Metriken mussten gegen andere Vorteile konkurrierender Modelle abgewogen werden. Extra Trees führte beispielsweise bei F1, ein balancierter Logistic-Regression-Kandidat bei Recall.

## Ergebnisse

Die Gruppe wählte `M04-HGB-002` kollektiv als finale Pipeline. Das Modell führte die aufgezeichneten Kandidaten bei mittlerer CV-ROC-AUC (`0.891449`) und Average Precision (`0.591089`) an, aber nicht bei jeder Metrik.

## Wissenschaftliche Interpretation

Die Auswahl war eine pragmatische Multi-Kriterien-Entscheidung und kein statistischer Überlegenheitsnachweis.

## Entscheidungen und Erkenntnisse

Modellparameter und Threshold `0.5` wurden vor der Final Evaluation eingefroren. Meine Aufgabe bestand darin, die HGB-Ergebnisse in die gemeinsame Auswahl einzubringen. Die technische Umsetzung des Model Selection Frameworks, des Final Model Locks und der Final-Evaluation-Pipeline gehörte nicht zu meinem Aufgabenbereich.

## Nächste Schritte

Als nächster Projektschritt sollte Member 01 den kontrollierten Final-Test-Workflow nach dem Freeze genau einmal ausführen. Danach waren weder Post-Test-Tuning noch eine erneute Auswahl vorgesehen.

## Repository-Evidenz

- `reports/meetings/2026-08-26_final-model-selection-decision.md`
- `reports/model_selection/model_comparison_portfolio.md`
- `reports/model_selection/final_model_lock.json`

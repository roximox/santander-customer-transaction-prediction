# 2026-08-22 – M04-HGB-002 – Registrierung und Reproduzierbarkeitsprüfung des optimierten HGB-Modells

**Mitglied:** Chaymae Akouaouch (Member 04)
**Kategorie:** Experiment und Integration
**Zugehöriges Experiment:** `M04-HGB-002`
**Zugehöriges Gruppentreffen:** 2026-08-23 – Modellfortschritt, Optimierung und Evaluation
**Branch:** `feature/model-optimization`
**Zeitaufwand:** 5 h

## Ziel

Ich wollte die am 14.08. ausgewählte und anschließend unverändert verwendete Konfiguration im gemeinsamen Experimentformat registrieren.

## Durchgeführte Arbeiten

- Runner für das optimierte HGB-Modell mit den bereits ausgewählten Parametern erstellt.
- Gemeinsame fünfteilige CV ausgeführt, Fold-Ergebnisse und Summary gespeichert und Experiment Registry aktualisiert.
- Registrierte Werte mit dem Search-Ergebnis abgeglichen und Branch-Integration vorbereitet.

## Tests und Validierung

Split-Fingerprints, Parameteridentität, `random_state=42`, Entwicklungsdaten-Grenze und Übereinstimmung mit `M04-HGB-SEARCH-001` wurden geprüft. Es gab weder eine neue Suche noch eine Final-Test-Entscheidung.

## Probleme und Herausforderungen

Die Parameterauswahl am 14.08. musste klar von der formalen Registrierung am 22.08. getrennt werden.

## Ergebnisse

| Metrik | Ergebnis |
|---|---:|
| Validierungs-ROC-AUC | 0.891449 ± 0.002836 |
| Average Precision | 0.591089 ± 0.010028 |
| F1 | 0.415248 |
| Precision | 0.795747 |
| Recall | 0.280942 |
| Accuracy | 0.920487 |
| Balanced Accuracy | 0.636438 |
| Train-ROC-AUC | 0.973580 |
| Train-Validierungs-Lücke | 0.082131 |

## Wissenschaftliche Interpretation

Die Übereinstimmung mit dem besten Search-Kandidaten bestätigt die Reproduzierbarkeit. Diese Werte stammen aus einer Cross-Validation ausschließlich auf den Entwicklungsdaten und sind keine Final-Test-Ergebnisse.

## Entscheidungen und Erkenntnisse

Ich stellte `M04-HGB-002` als meinen registrierten HGB-Kandidaten für den späteren Modellvergleich bereit. Die Konfiguration blieb unverändert.

## Nächste Schritte

Die gesamte HGB-Berechnungslogik automatisiert absichern und anschließend im Notebook konsolidieren.

## Repository-Evidenz

- `scripts/run_gradient_boosting_tuned.py`
- `reports/experiments/M04-HGB-002_summary.json`
- `reports/experiments/M04-HGB-002_fold_results.csv`
- `reports/experiments/experiment_registry.csv`

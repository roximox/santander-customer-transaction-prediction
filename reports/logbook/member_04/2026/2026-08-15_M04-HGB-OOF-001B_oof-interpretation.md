# 2026-08-15 – M04-HGB-OOF-001B – ROC-, Precision-Recall- und Confusion-Matrix-Analyse

**Mitglied:** Chaymae Akouaouch (Member 04)

**Kategorie:** Wissenschaftliche Analyse

**Zugehöriges Experiment:** `M04-HGB-OOF-001`

**Zugehöriges Gruppentreffen:** 2026-08-16 – Erster individueller Analyse- und ML-Fortschritt

**Branch:** `feature/model-optimization`

**Zeitaufwand:** 3 h

## Ziel

Ich wollte das optimierte HGB-Modell unter Klassenungleichgewicht anhand von Ranking-Metriken, Schwellenmetriken und den Fehlerklassen interpretieren.

## Durchgeführte Arbeiten

- ROC-AUC, Average Precision, F1, Precision, Recall, Accuracy und Balanced Accuracy geprüft.
- ROC-, Precision-Recall- und Confusion-Matrix-Figures ausgewertet.
- False Negatives und True Positives im Kontext der seltenen positiven Klasse eingeordnet.

## Tests und Validierung

Die dargestellten Werte wurden mit der Metrik-JSON-Datei und den OOF-Tabellen abgeglichen. Die Schwelle `0.5` blieb unverändert.

## Probleme und Herausforderungen

Gute Ranking-Leistung stand einem niedrigen Recall gegenüber. Die vorgegebene Schwelle war daher nicht im Nachhinein als optimal zu interpretieren.

## Ergebnisse

| Metrik | Wert |
|---|---:|
| ROC-AUC | 0.891438 |
| Average Precision | 0.590860 |
| F1 | 0.415242 |
| Precision | 0.795527 |
| Recall | 0.280943 |
| Accuracy | 0.920488 |
| Balanced Accuracy | 0.636438 |
| True Negatives | 142761 |
| False Positives | 1161 |
| False Negatives | 11561 |
| True Positives | 4517 |
| Threshold | 0.5 |

## Wissenschaftliche Interpretation

Das Modell rankte Beobachtungen gut und positive Prognosen waren relativ präzise. Bei der festen Schwelle wurden jedoch viele tatsächliche positive Fälle nicht erkannt. Dies beweist weder Optimalität des Modells noch der Schwelle.

## Entscheidungen und Erkenntnisse

Ich entschied mich gegen eine nachträgliche Threshold-Optimierung und hielt den begrenzten Recall ausdrücklich fest.

## Nächste Schritte

Die Tuned-OOF-Ergebnisse vorsichtig mit der Baseline vergleichen.

## Repository-Evidenz

- `reports/tables/M04-HGB-OOF-001_metrics.json`
- `reports/figures/M04-HGB-OOF-001_roc_curve.pdf`
- `reports/figures/M04-HGB-OOF-001_precision_recall_curve.pdf`
- `reports/figures/M04-HGB-OOF-001_confusion_matrix.pdf`

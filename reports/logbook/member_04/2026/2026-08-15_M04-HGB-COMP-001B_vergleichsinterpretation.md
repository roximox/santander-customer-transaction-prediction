# 2026-08-15 – M04-HGB-COMP-001B – Wissenschaftliche Interpretation der HGB-Optimierungsgewinne

**Mitglied:** Chaymae Akouaouch (Member 04)
**Kategorie:** Analyse und Interpretation
**Zugehöriges Experiment:** `M04-HGB-COMP-001`
**Zugehöriges Gruppentreffen:** 2026-08-16 – Erster individueller Analyse- und ML-Fortschritt
**Branch:** `feature/model-optimization`
**Zeitaufwand:** 2 h

## Ziel

Ich wollte die Verbesserungen und die weiterhin bestehenden Einschränkungen angemessen bewerten.

## Durchgeführte Arbeiten

Ich habe die Änderungen aller Metriken geprüft und die HGB-interne Verbesserung von der späteren gruppenweiten Modellwahl abgegrenzt.

## Tests und Validierung

Alle dargestellten Differenzen wurden gegen die gespeicherten Baseline- und OOF-Werte geprüft.

## Probleme und Herausforderungen

Kleine Veränderungen mussten unter Berücksichtigung der unterschiedlichen Aggregationsverfahren interpretiert werden.

## Ergebnisse

| Metrik | Änderung Baseline → Tuned |
|---|---:|
| ROC-AUC | +0.006842 |
| Average Precision | +0.017981 |
| F1 | +0.027987 |
| Precision | +0.012856 |
| Recall | +0.023636 |
| Accuracy | +0.002307 |
| Balanced Accuracy | +0.011780 |

## Wissenschaftliche Interpretation

Die Optimierung führte zu konsistenten deskriptiven Verbesserungen. Recall- und Generalisierungseinschränkungen blieben bestehen; HGB war damit nicht automatisch für jede Metrik oder Anwendung das beste Modell.

## Entscheidungen und Erkenntnisse

Die eingefrorene Konfiguration soll im gemeinsamen Experimentformat registriert werden.

## Nächste Schritte

`M04-HGB-002` ohne erneute Optimierung formal registrieren.

## Repository-Evidenz

- `reports/tables/M04-HGB-model-comparison.csv`
- `reports/tables/M04-HGB-model-comparison.json`
- `reports/figures/M04-HGB-model-comparison.pdf`

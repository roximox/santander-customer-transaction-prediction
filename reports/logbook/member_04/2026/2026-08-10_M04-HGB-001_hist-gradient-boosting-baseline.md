# 10.–12.08.2026 – M04-HGB-001 – Implementierung und Auswertung der HGB-Baseline

**Zeitraum:** 10.–12.08.2026

**Mitglied:** Chaymae Akouaouch (Member 04)
**Kategorie:** Implementierung, Experiment und Analyse
**Zugehöriges Experiment:** `M04-HGB-001`
**Zugehöriges Gruppentreffen:** 2026-08-09 – Datenverarbeitung und Validierungsstrategie
**Branch:** `feature/model-optimization`
**Zeitaufwand:** 8 h

## Ziel

Ich wollte vor der Optimierung einen reproduzierbaren Ausgangspunkt für HistGradientBoosting schaffen.

## Durchgeführte Arbeiten

- Factory geprüft und `learning_rate=0.1`, `max_iter=300`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=0.0`, `random_state=42` festgelegt.
- HGB-Modul und Baseline-Runner implementiert.
- Fünfteilige CV ausgeführt sowie Fold-, Train- und Validierungswerte analysiert.

## Tests und Validierung

Ich prüfte Estimatorparameter, Fingerprints, Fold-Größen und die ausschließliche Nutzung der Entwicklungsdaten. Der automatisierte Baseline-Test folgte am 22.08. Das Ausführungsartefakt trägt den Zeitstempel 2026-08-12; die Arbeit an der Baseline begann am 10.08. Wie sich die insgesamt acht Stunden auf den Zeitraum verteilten, lässt sich aus dem Repository nicht genauer ableiten.

## Probleme und Herausforderungen

Artefakte erschienen wegen der bestehenden `.gitignore`-Regel zunächst nicht in `git status`. Die Generalisierungslücke durfte außerdem nicht vorschnell als abschließender Overfitting-Nachweis gelten.

## Ergebnisse

| Metrik | Ergebnis |
|---|---:|
| Validierungs-ROC-AUC | 0.884596 ± 0.003278 |
| Average Precision | 0.572879 ± 0.009277 |
| F1 | 0.387255 |
| Precision | 0.782671 |
| Recall | 0.257307 |
| Accuracy | 0.918181 |
| Balanced Accuracy | 0.624658 |
| Train-ROC-AUC | 0.975659 |
| Train-Validierungs-Lücke | 0.091063 |

Fold-ROC-AUC: `0.879752`, `0.885401`, `0.889694`, `0.882831`, `0.885303`.

## Wissenschaftliche Interpretation

Die Baseline war über die Folds stabil, zeigte jedoch eine relevante Generalisierungslücke. Gleichzeitig war die Precision deutlich höher als der Recall. Bei Klassenungleichgewicht müssen Average Precision und Precision-Recall zusätzlich zur ROC-AUC betrachtet werden.

## Entscheidungen und Erkenntnisse

Ich entschied mich für eine kontrollierte Hyperparametersuche. Die reservierte Testpartition blieb dabei unangetastet.

## Nächste Schritte

Einen begrenzten HGB-Suchraum entwerfen und auf Entwicklungsdaten auswerten.

## Repository-Evidenz

- `src/gradient_boosting.py`
- `scripts/run_gradient_boosting_baseline.py`
- `reports/experiments/M04-HGB-001_summary.json`
- `reports/experiments/M04-HGB-001_fold_results.csv`

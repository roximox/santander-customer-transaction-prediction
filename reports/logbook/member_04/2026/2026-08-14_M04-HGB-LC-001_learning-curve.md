# 2026-08-14 – M04-HGB-LC-001 – Learning-Curve-Diagnostik des optimierten HGB-Modells

**Mitglied:** Chaymae Akouaouch (Member 04)
**Kategorie:** Implementierung, Experiment und Analyse
**Zugehöriges Experiment:** `M04-HGB-LC-001`
**Zugehöriges Gruppentreffen:** 2026-08-16 – Erster individueller Analyse- und ML-Fortschritt
**Branch:** `feature/model-optimization`
**Zeitaufwand:** 7 h

## Ziel

Ich wollte untersuchen, wie sich die eingefrorene optimierte Konfiguration bei wachsender Trainingsmenge verhält.

## Durchgeführte Arbeiten

- Learning-Curve-Modul und Runner implementiert.
- ROC-AUC und AP bei 10 %, 25 %, 50 %, 75 % und 100 % berechnet.
- Tabelle und Abbildung erzeugt sowie Train-/Validierungslücken interpretiert.

## Tests und Validierung

Parameter, Output-Schema, Fingerprints und Konsistenz beim größten Trainingsumfang wurden geprüft. Automatisierte Berechnungstests folgten am 22.08. Es gab keine Nachoptimierung anhand der Learning Curve.

## Probleme und Herausforderungen

Eine kleinere Lücke beweist nicht, dass Overfitting vollständig beseitigt ist. AP zeigte eine größere Lücke als ROC-AUC.

## Ergebnisse

| Trainingsgröße | Train-ROC-AUC | Validierungs-ROC-AUC | ROC-AUC-Lücke | Train-AP | Validierungs-AP | AP-Lücke |
|---:|---:|---:|---:|---:|---:|---:|
| 12,800 | 0.989269 | 0.851063 | 0.138206 | 0.968960 | 0.475333 | 0.493628 |
| 32,000 | 0.989880 | 0.877188 | 0.112692 | 0.967501 | 0.550490 | 0.417011 |
| 64,000 | 0.983864 | 0.885981 | 0.097883 | 0.936334 | 0.574884 | 0.361450 |
| 96,000 | 0.980122 | 0.889333 | 0.090789 | 0.917675 | 0.584480 | 0.333195 |
| 128,000 | 0.973342 | 0.891245 | 0.082098 | 0.887635 | 0.589899 | 0.297736 |

## Wissenschaftliche Interpretation

Mehr Entwicklungsdaten verbesserten beide Validierungsmetriken und verringerten die Lücken. Die verbleibenden Lücken blieben als Einschränkung sichtbar.

## Entscheidungen und Erkenntnisse

Ich nutzte die Learning Curve ausschließlich zur Diagnose und nicht für eine weitere Abstimmung der Hyperparameter.

## Nächste Schritte

OOF-Vorhersagen auf den Entwicklungsdaten für die weitere Analyse erzeugen.

## Repository-Evidenz

- `src/gradient_boosting_learning_curve.py`
- `scripts/run_gradient_boosting_learning_curve.py`
- `reports/tables/M04-HGB-learning-curve.csv`
- `reports/figures/M04-HGB-learning-curve.pdf`

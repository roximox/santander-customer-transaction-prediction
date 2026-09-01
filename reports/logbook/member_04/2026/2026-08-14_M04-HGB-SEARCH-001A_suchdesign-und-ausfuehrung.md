# 2026-08-14 – M04-HGB-SEARCH-001A – Entwurf und Ausführung der HGB-Hyperparametersuche

**Mitglied:** Chaymae Akouaouch (Member 04)
**Kategorie:** Implementierung und Experiment
**Zugehöriges Experiment:** `M04-HGB-SEARCH-001`
**Zugehöriges Gruppentreffen:** 2026-08-09 – Datenverarbeitung und Validierungsstrategie
**Branch:** `feature/model-optimization`
**Zeitaufwand:** 8 h

## Ziel

Ich wollte die Hyperparameter reproduzierbar und mit einem begrenzten Rechenaufwand auf den Entwicklungsdaten optimieren.

## Durchgeführte Arbeiten

- Vorhandene Search-Infrastruktur gelesen und HGB-spezifische Ergebnislogik geplant.
- Search-Modul und Runner implementiert.
- 768 mögliche Kombinationen über fünf Hyperparameter definiert.
- `RandomizedSearchCV` mit 20 Kandidaten, fünf Folds und 100 Fits ausgeführt.

## Tests und Validierung

Search-ID, `random_state=42`, `n_jobs=1`, `refit="roc_auc"`, Train-Scores, Fingerprints und Final-Test-Ausschluss wurden geprüft. Automatisierte Tests folgten am 22.08.

## Probleme und Herausforderungen

Vorhandene Ergebnisumwandlung war Logistic-Regression-spezifisch. Eine Vollsuche wäre unverhältnismäßig teuer gewesen.

## Ergebnisse

Zwanzig reproduzierbare Kandidaten wurden bewertet und als CSV/JSON gespeichert. Der Final Test wurde nicht verwendet.

## Wissenschaftliche Interpretation

Die begrenzte Zufallssuche balanciert Abdeckung und Rechenaufwand, beweist aber kein globales Optimum.

## Entscheidungen und Erkenntnisse

Ich hielt die HGB-spezifische Logik in meinem eigenen Modul und musste die gemeinsame Infrastruktur dafür nicht verändern.

## Nächste Schritte

Kandidaten analysieren und die beste Konfiguration einfrieren.

## Repository-Evidenz

- `src/gradient_boosting_search.py`
- `scripts/run_gradient_boosting_search.py`
- `reports/searches/M04-HGB-SEARCH-001_candidates.csv`
- `reports/searches/M04-HGB-SEARCH-001_summary.json`

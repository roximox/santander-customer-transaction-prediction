# 2026-08-15 – M04-HGB-COMP-001A – Reproduzierbarer Vergleich zwischen Baseline und optimiertem HGB-Modell

**Mitglied:** Chaymae Akouaouch (Member 04)

**Kategorie:** Implementierung und Validierung

**Zugehöriges Experiment:** `M04-HGB-COMP-001`

**Zugehöriges Gruppentreffen:** 2026-08-16 – Erster individueller Analyse- und ML-Fortschritt

**Branch:** `feature/model-optimization`

**Zeitaufwand:** 4 h

## Ziel

Ich wollte die Veränderung gegenüber meiner HGB-Baseline über sieben Metriken reproduzierbar darstellen.

## Durchgeführte Arbeiten

- Baseline-Summary und Tuned-OOF-Metriken geladen und validiert.
- Gemeinsame Vergleichstabelle und absolute Änderungen berechnet.
- CSV-, JSON- und PDF-Artefakte erzeugt, ohne Modelle erneut zu fitten.

## Tests und Validierung

Quell-Experiment-ID, erforderliche Metriken, Output-Schema und Überschreibschutz wurden geprüft. Automatisierte Comparison-Tests folgten am 22.08.

## Probleme und Herausforderungen

Die Baseline verwendet Mittelwerte der Fold-Metriken, während die Tuned-Werte aus aggregierten OOF-Vorhersagen stammen. Kleine Differenzen sind deshalb nicht vollständig gleichartig aggregiert.

## Ergebnisse

Alle sieben aufgezeichneten Metriken verbesserten sich gegenüber der Baseline.

## Wissenschaftliche Interpretation

Der Vergleich zeigt eine deskriptive Verbesserung gegenüber meiner HGB-Baseline. Daraus folgt jedoch weder eine gruppenweite noch eine statistische Überlegenheit.

## Entscheidungen und Erkenntnisse

Den Aggregationsunterschied ausdrücklich dokumentieren und die Gruppenauswahl getrennt behandeln.

## Nächste Schritte

Die Veränderungen wissenschaftlich interpretieren und die Tuned-Konfiguration formal registrieren.

## Repository-Evidenz

- `src/gradient_boosting_comparison.py`
- `scripts/run_gradient_boosting_comparison.py`
- `reports/tables/M04-HGB-model-comparison.csv`
- `reports/tables/M04-HGB-model-comparison.json`
- `reports/figures/M04-HGB-model-comparison.pdf`

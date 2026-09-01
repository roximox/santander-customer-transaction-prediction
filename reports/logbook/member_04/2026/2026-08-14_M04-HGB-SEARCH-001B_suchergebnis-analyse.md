# 2026-08-14 – M04-HGB-SEARCH-001B – Analyse der Suchergebnisse und Festlegung der Konfiguration

**Mitglied:** Chaymae Akouaouch (Member 04)

**Kategorie:** Analyse und Interpretation

**Zugehöriges Experiment:** `M04-HGB-SEARCH-001`

**Zugehöriges Gruppentreffen:** 2026-08-16 – Erster individueller Analyse- und ML-Fortschritt

**Branch:** `feature/model-optimization`

**Zeitaufwand:** 2.5 h

## Ziel

Ich wollte den besten gespeicherten Kandidaten nachvollziehbar auswählen und zugleich die Grenzen der Suche festhalten.

## Durchgeführte Arbeiten

Kandidaten nach ROC-AUC verglichen, Average Precision und Generalisierungslücken geprüft und Kandidat 011 gegen die Baseline eingeordnet.

## Tests und Validierung

Ich glich die Bestparameter mit der Kandidaten-CSV und der JSON-Zusammenfassung ab. Die späteren Diagnosen dienten der Bewertung der bereits gewählten Konfiguration und änderten die Auswahl nicht.

## Probleme und Herausforderungen

Einzelne Parameterwirkungen sind nicht kausal isolierbar; der beste von 20 Kandidaten ist nicht zwingend das globale Optimum.

## Ergebnisse

`learning_rate=0.05`, `max_iter=700`, `max_leaf_nodes=31`, `min_samples_leaf=100`, `l2_regularization=10.0`, `random_state=42`. ROC-AUC `0.891449`, AP `0.591089`, Train-ROC-AUC `0.973580`, Lücke `0.082131`.

## Wissenschaftliche Interpretation

Die regularisierte Konfiguration verbesserte die Ranking-Leistung und verringerte die Lücke, beseitigte sie aber nicht.

## Entscheidungen und Erkenntnisse

Die Konfiguration wurde unverändert für Learning Curve, OOF und Vergleich eingefroren.

## Nächste Schritte

Verhalten bei steigenden Trainingsgrößen untersuchen.

## Repository-Evidenz

- `reports/searches/M04-HGB-SEARCH-001_candidates.csv`
- `reports/searches/M04-HGB-SEARCH-001_summary.json`

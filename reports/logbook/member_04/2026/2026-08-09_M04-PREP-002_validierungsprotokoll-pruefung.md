# 2026-08-09 – M04-PREP-002 – Prüfung des Train/Test-/CV-Protokolls für HGB

**Mitglied:** Chaymae Akouaouch (Member 04)

**Kategorie:** Einarbeitung und Reproduzierbarkeit

**Zugehöriges Experiment:** Vorbereitung für `M04-HGB-001`

**Zugehöriges Gruppentreffen:** 2026-08-09 – Datenverarbeitung und Validierungsstrategie

**Branch:** `feature/model-optimization`

**Zeitaufwand:** 2.5 h

## Ziel

Ich wollte meine HGB-Entwicklung korrekt in das gemeinsame Train/Test- und Cross-Validation-Protokoll einordnen.

## Durchgeführte Arbeiten

- Stratifizierten 80/20-Split und fünfteilige `StratifiedKFold`-Validierung nachvollzogen.
- Gemeinsame Metriken, `random_state=42` und Dataset-Fingerprints geprüft.
- Grenze zwischen Entwicklungsdaten und reservierter Final-Test-Partition festgelegt.

## Tests und Validierung

Konzeptionell geprüft wurde, dass Baseline, Suche, Learning Curve, OOF und Vergleich ausschließlich die 160.000 Entwicklungszeilen verwenden.

## Probleme und Herausforderungen

Abweichende Splits, Seeds oder Metriken hätten den späteren Modellvergleich geschwächt.

## Ergebnisse

Ein reproduzierbarer Rahmen für die HGB-Baseline und alle Optimierungsschritte lag vor.

## Wissenschaftliche Interpretation

Die Isolation der Final-Test-Partition verhindert eine Verzerrung iterativer Modellentscheidungen.

## Entscheidungen und Erkenntnisse

Für meine Modellarbeit blieb der Final Test unangetastet. Er durfte erst nach der gemeinsamen Festlegung des finalen Modells verwendet werden.

## Nächste Schritte

Eine reproduzierbare HGB-Baseline implementieren.

## Repository-Evidenz

- `reports/meetings/2026-08-09_data-processing-validation-strategy-and-start-of-individual-analysis.md`
- Fingerprint-Prüfungen in den HGB-Runnern

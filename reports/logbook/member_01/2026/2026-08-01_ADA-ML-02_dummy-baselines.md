# Logbucheintrag

## Metadaten

- Datum: 2026-08-01
- Mitglied: Yassine Elhari
- Sprint: Sprint 1
- Ticket ID: ADA-ML-02
- Branch: feature/data_processing
- Pull Request: [#2 — feature/data_processing → main](https://github.com/roximox/santander-customer-transaction-prediction/pull/2)
- Zeitaufwand: 2 Stunden
- Zugehörige Besprechung: [2026-08-09 — Data Processing, Validation Strategy and Start of Individual Analysis](../../../meetings/2026-08-09_data-processing-validation-strategy-and-start-of-individual-analysis.md)

## Titel

DummyClassifier Baselines

## Ziel

Nahe Referenzwerte vor der Anpassung eines Modells festzulegen, das von den Features lernt. Vier `DummyClassifier` Strategien wurden verglichen: `most_frequent`, `prior`, `stratified` und `uniform`, die jeweils als `M01-DUMMY-001` bis `M01-DUMMY-004` registriert wurden.

## Protokoll

Die OpenML-Funktionen wurden explizit in `float32` umgewandelt, dann wurde der gemeinsame stratifizierte Split neu erstellt. Seine Trainings- und Reservateilfingerabdrücke übersprangen die offiziellen Werte. Nur das 160.000-Satz-Trainingsetteil ging in den gemeinsamen fünf-fach stratifizierten Kreuzvalidierung ein. Das 40.000-Satz-Reserveteil wurde nicht bewertet. Die zufälligen Strategien verwendeten `random_state=42`, und die technischen Bewertungen reproduzierten dieselben Werte.

Die Metriken waren ROC-AUC, Durchschnittsprecision, F1, Präzision, Erinnerung, Genauigkeit und ausgeglichenen Genauigkeit. Der Trainingszielpositivprävalenz war 0,1005.

## Ergebnisse

| Strategie | ROC-AUC | Durchschnittsprecision | F1 | Präzision | Erinnerung | Genauigkeit | Ausgeglichene Genauigkeit |
|---|---:|---:|---:|---:|---:|---:|---:|
| most_frequent | 0,5000 | 0,1005 | 0,0000 | 0,0000 | 0,0000 | 0,8995 | 0,5000 |
| prior | 0,5000 | 0,1005 | 0,0000 | 0,0000 | 0,0000 | 0,8995 | 0,5000 |
| stratified | 0,4995 | 0,1004 | 0,0994 | 0,0996 | 0,0992 | 0,8194 | 0,4995 |
| uniform | 0,5000 | 0,1005 | 0,1673 | 0,1005 | 0,5002 | 0,4997 | 0,4999 |

## Interpretation

`most_frequent` und `prior` vorherrschende Klasse vorhersehen. Ihre Genauigkeit von 0,8995 spiegelt das Zielunverhältnis wider, anstatt positive-Klasse-Detektion: Erinnerung und F1 sind Null, während ausgeglichene Genauigkeit 0,5000 beträgt. Die ROC-AUC von 0,5000 zeigt Zufallsdiskriminierung an. Durchschnittliche Präzision von 0,1005 passt sich der positiven Prävalenz an und ist daher die nahe Referenz-Ebene.

Die zufälligen Strategien ergeben unterschiedliche Schwellenabhängige Genauigkeit, Präzision und F1-Werte, aber ihre ROC-AUC und ausgeglichene Genauigkeit bleiben ungefähr 0,5. Keine der Dummy-Strategien lernt diskriminierende Featureinformation.

## Ausgänge

- Eintragungsberichte und Zusammenfassungen: `reports/experiments/M01-DUMMY-00*_*.csv/json`
- Registrierung: `reports/experiments/experiment_registry.csv`
- Vergleich: `reports/tables/dummy_baseline_comparison.csv` und `.json`
- Abbildung: `reports/figures/dummy_baseline_metrics.pdf`

## Einschränkungen und nächster Schritt

Dummy-Klassifizierer stellen nur Metrik-Ebenen auf und können keine Beziehungen zwischen Kunden-Funktionen und Zielmodellieren. Der nächste Schritt ist eine getrennte identifizierte Logistik-Regression-Experiment mit Training-Fold-Preprocessing.

## Entscheidung

Alle vier registrierten Strategien als Metrik-Ebenen beibehalten und den Mehrheits-/Prior-AP von 0,1005 als Ungleichheitsbewusstsein-Nahe Referenz verwenden.

## Schwierigkeiten

Hohe Mehrheitsklasse-Genauigkeit kann trotz Null positiver Erinnerung stark erscheinen, daher wurde der volle gemeinsame Metrikset verwendet.

## Anpassungen und Abweichungen vom Plan

Zufällige Basen erhielten den gemeinsamen Sitz; bestehende Experiment-IDs wurden vor Rennen geschützt.

## Abgelehnte Ansätze

Genauigkeits-Aberichterstattung und Bewertung auf der Reserveteil waren abgelehnt.

## Geänderte Dateien

- `src/dummy_baselines.py`
- `scripts/run_dummy_baselines.py`
- `tests/test_dummy_baselines.py`

## Code-Referenzen

Baseldefinitionen und Vergleichsbauer in `src/dummy_baselines.py`.

## Abbildung und Tabelle-Bezüge

- `reports/tables/dummy_baseline_comparison.csv`
- `reports/tables/dummy_baseline_comparison.json`
- `reports/figures/dummy_baseline_metrics.pdf`
- `reports/experiments/M01-DUMMY-001_summary.json` bis `M01-DUMMY-004_summary.json`

## Reproduzierbarkeitshinweise

Alle Ergebnisse verwenden die offizielle Trainingspartition, fünf stratifizierte Fälle und `random_state=42`. Der letzte Testset blieb geschlossen.

## Verwendete Quellen und Werkzeuge

scikit-learn, pandas, Matplotlib, pytest, Python und das gemeinsame Experiment-API.

## Nächster Schritt

Ein gemeinsames Logistik-Regression-Baseline auf Training-Fold-Preprocessing anpassen.

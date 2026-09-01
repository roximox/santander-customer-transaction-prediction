# Logbucheintrag

## Metadaten

- Datum: 2026-08-11
- Mitglied: Yassine Elhari
- Sprint: Sprint 1
- Ticket ID: ADA-ML-08
- Branch: feature/data_processing
- Pull Request: [#7 — develop → main](https://github.com/roximox/santander-customer-transaction-prediction/pull/7) (integriert in `develop` in `ff3c2f8`)
- Zeitaufwand: 4 Stunden
- Bezugnahme auf eine Besprechung: [2026-08-16 — Erster individueller Analyse- und Machine-Learning- Fortschritt](../../../meetings/2026-08-16_first-individual-analysis-and-machine-learning-progress.md)

## Titel

Logistische Regressionskurven

## Ziel

Die Messung der Auswirkungen des Trainingsvolumens auf die ausgewählte ROC-AUC-Logistische Regressions und ihre recallorientierte alternative ohne den letzten Test.

## Kontext

Beide Konfigurationen verwenden L2-Regularisierung, `C=0.01`, `solver="saga"`,
`max_iter=2000`, und `random_state=42`; nur `class_weight` unterscheidet sich. Die gemeinsame 160.000-Reihen-Trainingseinheit und die fünffache stratifizierte CV bleiben unverändert.

## Durchgeführte Arbeit

Eine reusable Learning-Kurve-Module, eine Offline-Test-Suite, ein geschützter wissenschaftlicher Runner, vier Vektor-Figuren, Ergebnis- und Entscheidungstabelle, Zellen für das Notebook-Bericht und Projekt-Dokumentation hinzugefügt.

## Methodik

Jeder CV-Training-Fold wird unabhängig auf 5%, 10%, 25%, 50%,
75%, und 100% unterteilt. Eine frische Kopie der gemeinsamen Fabrikpipelinen wird für jeden Fold/Fraction-Paar angepasst. Die Trainingsmetriken verwenden die tatsächliche Subset; die Validierungsmetriken verwenden den kompletten Validierungs-Ordner. Timing, Iterationen und Konvergenzwarnungen werden pro Pass. Das vorbelegte Test-Signatur ist überprüft, bevor seine Objekte gelöscht werden; es wird nie gescored oder vorhergesagt.

## Ergebnisse

Für LR-LEARNING-ROC, wächst die Validierungs-ROC-AUC von 0.839782 bei 5% bis
0.849354, 0.855389, 0.857732, 0.858745 und 0.859201 bei 100%. Die Durchschnittliche Genauigkeit wächst von 0.457493 zu 0.481314, 0.498673, 0.504275, 0.506432 und 0.507592.
Für LR-LEARNING-BALANCED, wächst die ROC-AUC von 0.834234 bis
0.845888, 0.854096, 0.857267, 0.858418 und 0.859017; Durchschnittliche Genauigkeit wächst von
0.443401 zu 0.472769, 0.494634, 0.502437, 0.504984 und 0.506454.

Alle 60 Passen konvergierten. Die gemessene Kampagne-Dauer betrug 203,18 Sekunden.

## Interpretation

Beide Konfigurationen verbessern sich weiterhin durch 100%, mit abnehmenden Gewinnen.
Von 5%→100%, wachsen ROC/AP-Gewinne um +0.019418/+0.050099 ungewichtet und
+0.024783/+0.063053 ausgewogen. Von 75%→100%, sie fallen zu
+0.000456/+0.001160 und +0.000599/+0.001470, was die empirische späte Platte unterstützt. Die ROC-AUC-Train-Valid-Gap schrumpft von 0.049999 zu
0.002325 ungewichtet und von 0.060521 zu 0.002805 ausgewogen. Die durchschnittliche Passzeit steigt von 0,148 Sekunden zu 3,084 Sekunden ungewichtet und 0,444 Sekunden zu 4,733 Sekunden ausgewogen.
Die Falt-Standardabweichung bei 100% beträgt 0,003236/0,008491 (ROC/AP) ungewichtet und
0,003122/0,008518 ausgewogen. Ausgewogen behält die Balanced viel höhere Erinnerung bei 100%
(0,773666 gegenüber 0,267758), aber niedrigere Genauigkeit (0,284593 gegenüber 0,691427).

## Entscheidung

Die beobachteten 75%→100%-Gewinne unterstützen eine späte empirische Platte für beide Modelle; die ungewichtete Konfiguration bleibt marginal stärker in der Rangierung, während ausgewogen die Erinnerungsorientierte Alternative ist. Diese Analyse wählt keinen Endmodell oder Schwellenwert.

## Schwierigkeiten

Die Kampagne erfordert 60 SAGA-Pass auf 200 numerische Merkmale. Die Ausführung wird von der Vorhersage unabhängig gemacht und ist maschinenabhängig.

## Anpassungen und Abweichungen vom Plan

Der gemeinsame Modell-Factory, die gemeinsamen Metriken, die Split-Tools, die CV-Fabrik und die Fingerprint-Implementierung wurden wieder verwendet. Die öffentliche Learning-Kurve-API hat keinen Testsetparameter.

## Abgelehnte Ansätze

- scikit-learns generic `learning_curve` Helper wurde nicht verwendet, weil der Auftrag eine pro-Pass-Zählung, Warnungen, Iterationen und alle Projektmetriken erfordert.
- Die Validierungs-Fold wurden nicht unterbezogen.
- Bestehende historische wissenschaftliche Ausgaben wurden nicht geändert.
- Kein trainiertes Modell oder komplettes Dataset wurde gespeichert.

## Geänderte Dateien

- `src/learning_curves.py`
- `scripts/run_logistic_learning_curves.py`
- `tests/test_learning_curves.py`
- `notebooks/03_logistic_regression.ipynb`
- `README.md`
- `CONTRIBUTING.md`
- Die Learning-Kurve-Tabelle und -Figur unten

## Code-Referenzen

- `validate_train_size_fractions`
- `create_stratified_subsample`
- `compute_learning_curve`
- `summarize_learning_curve`
- `create_learning_curve_figures`
- `build_decision_table`

## Figuren- und Tabellenbezug

- `reports/tables/logistic_learning_curve_folds.csv`
- `reports/tables/logistic_learning_curve_summary.csv`
- `reports/tables/logistic_learning_curve_summary.json`
- `reports/tables/logistic_learning_curve_decision.csv`
- `reports/figures/logistic_learning_curve_roc_auc.pdf`
- `reports/figures/logistic_learning_curve_average_precision.pdf`
- `reports/figures/logistic_learning_curve_fit_time.pdf`
- `reports/figures/logistic_learning_curve_threshold_metrics.pdf`

## Reproduzierbarkeitshinweise

Mit OpenML ID 45566, float32-Features, gemeinsamer Split-Signatur, random State 42,
und fünffacher `StratifiedKFold` wird überprüft oder konstruiert explizit. Der Runner lehnt es ab, ein bestehendes Ziel zu überschreiben.

## Nächster Schritt

Ein späterer Endtest-Evaluation erfordert eine separate, explizit genehmigte Aufgabe.

## Verwendete Quellen und Werkzeuge

Das Projekt-Quellcode und Konfiguration, pandas, NumPy, scikit-learn, Matplotlib,
pytest, nbformat, OpenMLs scikit-learn-Lader und Git-Inspektionsbefehle.

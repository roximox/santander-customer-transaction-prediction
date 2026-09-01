# Logbucheintrag

## Metadaten

- Mitglied: Yassine Elhari
- Sprint: Sprint 2
- Ticket ID: ADA-ML-11
- Datum: 2026-08-26
- Branch: develop
- Pull Request: Nicht anwendbar — direkt in `develop` (`c15f0d6`) übertragen
- Zeitaufwand: 2,5 Stunden
- Zugehörige Besprechung: [2026-08-26 — Entscheidung über den finalen Modellauswahl](../../../meetings/2026-08-26_final-model-selection-decision.md)

## Titel

Sammlungsfinalmodellauswahl und reproduzierbare Pipelinenlöse

## Ziel

Das Team bestätige die gemeinsam ausgewählte Modellwahl ohne den reservierten finalen Test zu öffnen oder zu bewerten.

## Kontext

Die erwartete-Familien-Koverage war nach der Hinzufügung von Extra Trees vollständig. Die Vergleichsidentifizierung identifizierte M04-HGB-002 als einziges wettbewerbsfähiges Kandidat und den Spitzenwert bei der mittleren KreuzvalidierungsROC-AUC und dem Durchschnittswert für die Präzision.

## Durchgeführte Arbeiten

M04-HGB-002 als gemeinsam ausgewählte Pipelinen, ihre feste Parameter festgelegt und den Wert, die Besprechungentscheidung hinzugefügt und in der Lesebenutzerschnittstelle geöffnet.

## Methodik

Die Auswahl verwendet nur gespeicherte fünf-Fold-StratifiedKFold-Evidenz aus der 160.000-Spalten-Entwicklungspartition. Keine Experimentierung wurde wiederholt. Der finale Test blieb geschlossen und wurde nicht verwendet.

## Ergebnisse

Das ausgewählte Modell hat einen mittleren CV-ROC-AUC-Wert von 0,891449 und einen Durchschnittswert für die Präzision von 0,591089. Seine gesicherte Klassifikationswertschranke beträgt 0,5 und sein festgelegter Zufallszustand ist 42. Die einzige finale-Testausführung produzierte einen ROC-AUC-Wert von 0,891214, einen Durchschnittswert für die Präzision von 0,584385, F1-Wert von 0,403632, Präzisionswert von 0,791424, Wiederholungsgrad von 0,270896 und ein ausgeglichenes Genauigkeitsmaß von 0,631459.

## Interpretation

HGB bietet die beste gesicherte Beweise für die imbalancesbehaftete Vorhersagemuster, während seine größeren allgemeinisierten Rückschläge, Laufzeit und begrenzte Wiederholungsrate bei der festen Schranke explizite Einschränkungen darstellen.

## Entscheidung

M04-HGB-002 bleibt die gesicherte finale Modellwahl nach einer kontrollierten finalen Testausführung. Die Modellauswahl wurde nicht wieder geöffnet. Dies ist eine kollektive Mehrkriterien-Auswahl, nicht ein formeller statistischer Vorteilanspruch.

## Schwierigkeiten

Differente Modelle führen zu unterschiedlichen Metriken, so dass die Entscheidung Prioritäten bei der Rangqualität setzen und die Schranke und die Rechenbegrenzungen berücksichtigen musste.

## Anpassungen und Abweichungen von dem Plan

Das Team wartete auf eine vollständige erwartete-Familien-Koverage, bevor es den Modellzugriff öffnete.

## Abgelehnte Ansätze

Abgelehnt wurden die Wiederholung der Hyperparameter-Suche, das Ändern der Schranke nach der Auswahl und die Berücksichtigung des finalen Tests vor dem Modellzugriff.

## Geänderte Dateien

- `reports/model_selection/final_model_lock.json`
- `reports/final_evaluation/M04-HGB-002_final_test_results.json`
- `reports/meetings/2026-08-26_final-model-selection-decision.md`
- `app.py`
- `src/dashboard/loaders.py`
- dashboard und Logbuchtests

## Code-Referenzen

- `src/model_selection.py`
- `src/dashboard/loaders.py`
- `app.py`

## Figuren- und Tabellenbeziehungen

- `reports/model_selection/model_comparison_eligible.csv`
- `reports/model_selection/model_selection_decision.csv`
- `reports/model_selection/model_selection_comparability.json`

## Reproduzierbarkeitsnotizen

Die Löse schreibt die geteilten Spaltenspuren, StratifiedKFold-Konfiguration, Parameter festgelegt und Schranke sowie den einzigartigen finalen Testausführungsidentifikator fest. Der finale Test wurde während der Modellauswahl nicht verwendet; er wurde genau einmal nach der kollektiven Löse überprüft.

## Weitere Schritte

Das wissenschaftliche Bericht abschließen, indem die einzige gesicherte Ergebnisse verwenden werden. Keine weitere Ausführung des finalen Tests oder Tuning aus seinen Metriken.

## Verwendete Quellen und Werkzeuge

Gespeicherte Experimentierungsübersichten, generierte Modellauswahlberichte, die gemeinsame Dashboard-Lader, pytest, und Streamlit AppTest.

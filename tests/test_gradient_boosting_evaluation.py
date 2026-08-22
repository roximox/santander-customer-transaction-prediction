import numpy as np
import pandas as pd
import pytest

import src.gradient_boosting_evaluation as module


def _predictions():
    return pd.DataFrame({"row_index": [0, 1, 2, 3], "fold": [1, 1, 2, 2], "true_target": [False, False, True, True], "predicted_class": [False, True, False, True], "positive_class_probability": [.1, .8, .4, .9]})


def test_oof_metrics_and_curves_are_consistent():
    metrics = module.compute_hist_gradient_boosting_oof_metrics(_predictions())
    assert metrics["true_negatives"] == metrics["false_positives"] == metrics["false_negatives"] == metrics["true_positives"] == 1
    assert metrics["accuracy"] == .5 and set(module.compute_hist_gradient_boosting_oof_roc_curve(_predictions())) == {"false_positive_rate", "true_positive_rate", "threshold"}
    pr = module.compute_hist_gradient_boosting_oof_precision_recall_curve(_predictions())
    assert set(pr) == {"precision", "recall", "threshold"} and np.isnan(pr["threshold"].iloc[-1])


def test_prediction_schema_is_validated():
    with pytest.raises(ValueError): module.compute_hist_gradient_boosting_oof_metrics(pd.DataFrame())
    bad = _predictions().drop(columns="positive_class_probability")
    with pytest.raises(ValueError): module.compute_hist_gradient_boosting_oof_metrics(bad)

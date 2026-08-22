import numpy as np
import pandas as pd
import pytest

import src.gradient_boosting_learning_curve as module


def test_tuned_factory_and_train_sizes():
    params = module.create_tuned_hist_gradient_boosting_estimator().get_params()
    assert module.HIST_GRADIENT_BOOSTING_LEARNING_CURVE_TRAIN_SIZES == [0.10, 0.25, 0.50, 0.75, 1.00]
    assert {key: params[key] for key in ("learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization", "random_state")} == {"learning_rate": .05, "max_iter": 700, "max_leaf_nodes": 31, "min_samples_leaf": 100, "l2_regularization": 10., "random_state": 42}


def test_learning_curve_uses_mocked_arrays(monkeypatch):
    calls = []

    def fake_learning_curve(**kwargs):
        calls.append(kwargs)
        return (
            np.array([1, 2], dtype=int),
            np.array([[0.9, 0.8], [0.8, 0.7]]),
            np.array([[0.7, 0.6], [0.8, 0.7]]),
        )

    monkeypatch.setattr(module, "create_stratified_cv", lambda: object())
    monkeypatch.setattr(module, "get_scoring_metrics", lambda: {"roc_auc": object(), "average_precision": object()})
    monkeypatch.setattr(module, "learning_curve", fake_learning_curve)
    result = module.compute_hist_gradient_boosting_learning_curve(pd.DataFrame({"x": [0, 1]}), pd.Series([0, 1]))
    assert list(result.columns) == [
        "train_size", "train_roc_auc_mean", "train_roc_auc_std",
        "validation_roc_auc_mean", "validation_roc_auc_std",
        "train_average_precision_mean", "train_average_precision_std",
        "validation_average_precision_mean", "validation_average_precision_std",
    ]
    assert list(result["train_size"]) == [1, 2]
    assert result.loc[0, "train_roc_auc_mean"] == pytest.approx(0.85)
    assert result.loc[0, "train_roc_auc_std"] == pytest.approx(0.05)
    assert result.loc[0, "validation_roc_auc_mean"] == pytest.approx(0.65)
    assert result.loc[0, "validation_roc_auc_std"] == pytest.approx(0.05)
    assert len(calls) == 2

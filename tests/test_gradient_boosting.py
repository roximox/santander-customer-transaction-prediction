from sklearn.ensemble import HistGradientBoostingClassifier

from src.gradient_boosting import (
    HIST_GRADIENT_BOOSTING_EXPERIMENT_ID,
    HIST_GRADIENT_BOOSTING_MODEL_NAME,
    create_hist_gradient_boosting_baseline,
)


def test_baseline_identity_and_fresh_frozen_parameters():
    first = create_hist_gradient_boosting_baseline()
    second = create_hist_gradient_boosting_baseline()
    assert HIST_GRADIENT_BOOSTING_EXPERIMENT_ID == "M04-HGB-001"
    assert HIST_GRADIENT_BOOSTING_MODEL_NAME == "HistGradientBoosting Baseline"
    assert isinstance(first, HistGradientBoostingClassifier) and first is not second
    expected = {"learning_rate": 0.1, "max_iter": 300, "max_leaf_nodes": 31, "l2_regularization": 0.0, "random_state": 42, "min_samples_leaf": 20}
    assert all(first.get_params()[name] == value for name, value in expected.items())

"""Baseline configuration for the Member 4 HistGradientBoosting model."""

from sklearn.ensemble import HistGradientBoostingClassifier

from src.modeling import create_hist_gradient_boosting_classifier

HIST_GRADIENT_BOOSTING_EXPERIMENT_ID = "M04-HGB-001"
HIST_GRADIENT_BOOSTING_MODEL_NAME = "HistGradientBoosting Baseline"


def create_hist_gradient_boosting_baseline() -> HistGradientBoostingClassifier:
    """Return the exact unfitted classifier specified for M04-HGB-001."""
    return create_hist_gradient_boosting_classifier(
        learning_rate=0.1,
        max_iter=300,
        max_leaf_nodes=31,
        l2_regularization=0.0,
        random_state=42,
    )

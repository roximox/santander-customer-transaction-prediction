"""Verify model factories without data, fitting, experiments, or file output."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling import (  # noqa: E402
    create_dummy_classifier,
    create_extra_trees_classifier,
    create_hist_gradient_boosting_classifier,
    create_logistic_regression,
    create_logistic_regression_pipeline,
    create_random_forest_classifier,
    describe_estimator,
)


def main() -> None:
    """Construct and describe one unfitted object from every shared factory."""
    estimators = {
        "dummy": create_dummy_classifier(strategy="stratified"),
        "logistic": create_logistic_regression(),
        "logistic_pipeline": create_logistic_regression_pipeline(),
        "random_forest": create_random_forest_classifier(),
        "extra_trees": create_extra_trees_classifier(),
        "hist_gradient_boosting": create_hist_gradient_boosting_classifier(),
    }
    print("Factory verification only — no model was trained.")
    for name, estimator in estimators.items():
        description = describe_estimator(estimator)
        parameters = description["parameters"]
        print(
            f"{name}: class={description['estimator_class']}, "
            f"steps={description['pipeline_steps']}, "
            f"random_state={description['random_state']}, "
            f"main_parameter={parameters.get('classifier__C', parameters.get('n_estimators', parameters.get('max_iter', parameters.get('strategy'))))}"
        )


if __name__ == "__main__":
    main()

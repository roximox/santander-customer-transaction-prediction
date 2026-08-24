import pytest
from sklearn.model_selection import RandomizedSearchCV

from src.gradient_boosting_search import (
    HIST_GRADIENT_BOOSTING_SEARCH_ID, create_hist_gradient_boosting_randomized_search,
    create_hist_gradient_boosting_search_space, hist_gradient_boosting_search_results_to_dataframe,
)


def test_search_space_and_unfitted_configuration():
    space = create_hist_gradient_boosting_search_space()
    assert HIST_GRADIENT_BOOSTING_SEARCH_ID == "M04-HGB-SEARCH-001"
    assert set(space) == {"learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization"}
    assert 0.05 in space["learning_rate"] and 700 in space["max_iter"] and 100 in space["min_samples_leaf"]
    search = create_hist_gradient_boosting_randomized_search(n_iter=20, n_jobs=1)
    assert isinstance(search, RandomizedSearchCV) and search.n_iter == 20 and search.random_state == 42
    assert search.refit == "roc_auc" and search.return_train_score is True


@pytest.mark.parametrize("kwargs", [{"n_iter": 0}, {"n_iter": True}, {"n_jobs": True}])
def test_invalid_search_arguments_and_unfitted_results_are_rejected(kwargs):
    with pytest.raises(ValueError):
        create_hist_gradient_boosting_randomized_search(**kwargs)
    with pytest.raises(Exception):
        hist_gradient_boosting_search_results_to_dataframe(create_hist_gradient_boosting_randomized_search(n_iter=1))

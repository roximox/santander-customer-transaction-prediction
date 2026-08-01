"""Tests for central configuration."""

from src.config import load_config


def test_load_config_values() -> None:
    config = load_config()
    assert isinstance(config, dict)
    assert config["data"]["openml_id"] == 45566
    assert config["project"]["random_state"] == 42
    assert config["data"]["target_column"] is None
    assert config["metrics"]["primary"] == "roc_auc"
    assert config["metrics"]["reporting"] == ["accuracy", "balanced_accuracy"]
    assert config["validation"]["n_splits"] == 5
    assert config["validation"]["shuffle"] is True
    assert config["experiments"]["default_n_jobs"] == -1
    assert config["experiments"]["return_train_score"] is True
    expected_paths = {
        "raw_data",
        "interim_data",
        "processed_data",
        "figures",
        "tables",
        "experiments",
        "models",
    }
    assert expected_paths <= config["paths"].keys()

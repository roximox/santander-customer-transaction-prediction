"""Tests for central configuration."""

from src.config import load_config


def test_load_config_values() -> None:
    config = load_config()
    assert isinstance(config, dict)
    assert config["data"]["openml_id"] == 45566
    assert config["project"]["random_state"] == 42
    assert config["data"]["target_column"] is None
    assert config["metrics"]["primary"] == "roc_auc"
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

"""Central configuration loading utilities."""

from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    """Return the project root derived from this module's location."""
    return Path(__file__).resolve().parents[1]


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load and validate the YAML configuration.

    Args:
        config_path: Optional path to a YAML file. Relative paths are resolved
            from the project root.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the YAML is empty or its root is not a mapping.
        yaml.YAMLError: If the YAML syntax is invalid.
    """
    path = config_path or Path("configs/config.yaml")
    if not path.is_absolute():
        path = get_project_root() / path
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return config

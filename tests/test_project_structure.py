"""Tests for the repository scaffold."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_main_directories_exist() -> None:
    directories = [
        "configs",
        "data/raw",
        "data/interim",
        "data/processed",
        "notebooks",
        "src",
        "reports/figures",
        "reports/tables",
        "reports/experiments",
        "reports/logbook",
        "reports/meetings",
        "reports/portfolio",
        "models",
        "tests",
    ]
    assert all((ROOT / directory).is_dir() for directory in directories)


def test_main_files_exist() -> None:
    files = [
        "README.md",
        "CONTRIBUTING.md",
        "requirements.txt",
        "environment.yml",
        "pyproject.toml",
        ".gitignore",
        "configs/config.yaml",
        "src/config.py",
        "reports/experiments/experiment_template.csv",
        "reports/logbook/logbook_template.md",
        "reports/meetings/meeting_template.md",
    ]
    assert all((ROOT / file).is_file() for file in files)


def test_required_gitkeep_files_exist() -> None:
    directories = [
        "data/raw",
        "data/interim",
        "data/processed",
        "reports/figures",
        "reports/tables",
        "reports/experiments",
        "reports/logbook",
        "reports/meetings",
        "reports/portfolio",
        "models",
    ]
    assert all((ROOT / directory / ".gitkeep").is_file() for directory in directories)

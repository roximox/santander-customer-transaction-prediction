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
        "reports/logbook/README.md",
        "reports/logbook/templates/logbook_entry_template.md",
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
        "reports/meetings",
        "reports/portfolio",
        "models",
    ]
    assert all((ROOT / directory / ".gitkeep").is_file() for directory in directories)


def test_individual_logbook_structure() -> None:
    logbook = ROOT / "reports/logbook"
    members = [logbook / f"member_{number:02d}" for number in range(1, 5)]
    assert all(member.is_dir() for member in members)
    assert all((member / "README.md").is_file() for member in members)
    assert all((member / "2026").is_dir() for member in members)
    assert all((member / "2026/.gitkeep").is_file() for member in members)


def test_logbook_template_has_required_sections() -> None:
    template = (
        ROOT / "reports/logbook/templates/logbook_entry_template.md"
    ).read_text(encoding="utf-8")
    required = [
        "# Logbook Entry",
        "## Metadata",
        "- Ticket ID:",
        "- Time spent:",
        "## Rejected approaches",
        "## Reproducibility notes",
        "## Sources and tools used",
    ]
    assert all(item in template for item in required)


def test_only_member_01_has_an_initial_entry() -> None:
    logbook = ROOT / "reports/logbook"
    first_entry = (
        logbook
        / "member_01/2026/2026-07-25_ADA-SETUP-01_initial-project-setup.md"
    )
    assert first_entry.is_file()
    entry = first_entry.read_text(encoding="utf-8")
    required = [
        "- Date: 2026-07-25",
        "- Member: Member 01",
        "- Ticket ID: ADA-SETUP-01",
        "- Time spent: TO BE COMPLETED BY THE STUDENT",
        "## Reproducibility notes",
        "## Next step",
    ]
    assert all(item in entry for item in required)

    for number in range(2, 5):
        year = logbook / f"member_{number:02d}/2026"
        entries = [path for path in year.iterdir() if path.name != ".gitkeep"]
        assert entries == []

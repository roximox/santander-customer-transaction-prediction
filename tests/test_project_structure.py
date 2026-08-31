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
        "reports/searches",
        "reports/model_selection",
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
        "src/evaluation.py",
        "src/experiments.py",
        "src/dummy_baselines.py",
        "src/modeling.py",
        "src/logistic_baseline.py",
        "src/logistic_class_weight.py",
        "src/search.py",
        "src/logistic_coefficient_analysis.py",
        "src/model_selection.py",
        "app.py",
        "src/dashboard/__init__.py",
        "src/dashboard/loaders.py",
        "src/dashboard/components.py",
        "src/dashboard/charts.py",
        "src/dashboard/formatting.py",
        "scripts/verify_evaluation_framework.py",
        "scripts/verify_experiment_orchestrator.py",
        "scripts/run_dummy_baselines.py",
        "scripts/verify_model_factories.py",
        "scripts/run_logistic_baseline.py",
        "scripts/run_logistic_class_weight_comparison.py",
        "scripts/run_logistic_grid_search.py",
        "scripts/run_logistic_coefficient_analysis.py",
        "scripts/build_model_selection_report.py",
        "scripts/run_extra_trees_baseline.py",
        "tests/test_evaluation.py",
        "tests/test_experiments.py",
        "tests/test_dummy_baselines.py",
        "tests/test_modeling.py",
        "tests/test_logistic_baseline.py",
        "tests/test_logistic_class_weight.py",
        "tests/test_search.py",
        "tests/test_logistic_coefficient_analysis.py",
        "tests/test_model_selection.py",
        "tests/test_dashboard.py",
        "tests/test_extra_trees_experiment.py",
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
        "reports/searches",
        "reports/model_selection",
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
        "- Member: Yassine Elhari",
        "- Ticket ID: ADA-SETUP-01",
        "- Time spent: TO BE COMPLETED BY YASSINE ELHARI",
        "## Reproducibility notes",
        "## Next step",
    ]
    assert all(item in entry for item in required)

    for number in (2, 3, 4):
        year = logbook / f"member_{number:02d}/2026"
        entries = [path for path in year.iterdir() if path.name != ".gitkeep"]
        assert len(entries) >= 1, f"member_{number:02d} should have logbook entries."

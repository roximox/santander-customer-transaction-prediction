"""Structural and traceability checks for Yassine Elhari's Logbooks."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOGBOOK_ROOT = ROOT / "reports" / "logbook"
MEMBER_DIR = LOGBOOK_ROOT / "member_01" / "2026"
EXPECTED_TICKETS = {
    "ADA-SETUP-01",
    "ADA-DATA-01",
    "ADA-DATA-02",
    "ADA-DATA-03",
    "ADA-ML-00",
    "ADA-ML-01",
    "ADA-ML-02",
    "ADA-ML-03",
    "ADA-ML-04",
    "ADA-ML-05",
    "ADA-ML-06",
    "ADA-ML-07",
    "ADA-ML-08",
    "ADA-ML-09",
}
FILENAME_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}_(ADA-(?:SETUP|DATA|ML)-\d{2})_[a-z0-9-]+\.md$"
)
SECTION_EQUIVALENTS = {
    "Objective": ("## Objective", "## Scientific question", "## Problem", "## Reason for the audit"),
    "Context": ("## Context", "## Comparability risk", "## Protocol", "## Protocol and justification", "## Pipeline and protocol", "## Architecture and implementation", "## Reason for the audit", "## Problem", "## Raw-data observations", "## Scientific question", "## Methodological decision", "## Work completed"),
    "Work performed": ("## Work performed", "## Work completed", "## Architecture and implementation", "## Implementation and tests", "## Protocol", "## Pipeline and protocol", "## Controlled change", "## Configurations and API", "## Validation", "## Observed split", "## Synthetic verification"),
    "Methodology": ("## Methodology", "## Protocol", "## Methodological decision", "## Cross-validation design", "## Controlled change", "## Configurations and API", "## Float64 / float32 comparison", "## Pipeline and protocol", "## Architecture and implementation", "## Work completed"),
    "Results": ("## Results", "## Results and deltas", "## Scientific results", "## Verification", "## Tests and verification", "## Convergence results", "## Observed split", "## Raw-data observations", "## Synthetic verification"),
    "Interpretation": ("## Interpretation", "## Interpretation and decision", "## Interpretation and limitations", "## Comparison and provisional decision", "## Performance and decision", "## Decision", "## Decisions", "## Verification"),
    "Decision": ("## Decision", "## Decisions", "## Interpretation and decision", "## Comparison and provisional decision", "## Performance and decision"),
    "Difficulties": ("## Difficulties", "## Difficulties encountered", "## Computational cost and convergence"),
}
REQUIRED_EXACT_SECTIONS = (
    "# Logbook Entry",
    "## Metadata",
    "## Title",
    "## Adaptations and deviations from the plan",
    "## Rejected approaches",
    "## Files changed",
    "## Code references",
    "## Figure and table references",
    "## Reproducibility notes",
    "## Next step",
    "## Sources and tools used",
)
OTHER_MEMBER_HASHES = {
    "member_02/README.md": "8959a652dddc40028c75d44cb686c2cc0606d03607d0037ee7269c4249f6b78f",
    "member_02/2026/.gitkeep": "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b",
    "member_03/README.md": "57b34e97cbe870fd80738fcd7fa7333b56adf05708268d25e633215a7d65f467",
    "member_03/2026/.gitkeep": "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b",
    "member_04/README.md": "bed888e01228e33f60fe27f5982188bda26dadac16e98defb98af267b227ed80",
    "member_04/2026/.gitkeep": "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b",
}


def _entries() -> list[Path]:
    return sorted(MEMBER_DIR.glob("*.md"))


def test_member_01_directory_and_expected_entries_exist() -> None:
    assert MEMBER_DIR.is_dir()
    matches = [FILENAME_PATTERN.fullmatch(path.name) for path in _entries()]
    assert all(matches)
    assert {match.group(1) for match in matches if match} == EXPECTED_TICKETS


def test_ticket_ids_are_unique_and_match_filenames() -> None:
    ticket_ids: list[str] = []
    for path in _entries():
        content = path.read_text(encoding="utf-8")
        metadata_ids = re.findall(r"^- Ticket ID: (\S+)$", content, re.MULTILINE)
        assert len(metadata_ids) == 1
        filename_match = FILENAME_PATTERN.fullmatch(path.name)
        assert filename_match is not None
        assert metadata_ids[0] == filename_match.group(1)
        ticket_ids.extend(metadata_ids)
    assert len(ticket_ids) == len(set(ticket_ids))


def test_entries_identify_yassine_and_have_complete_structure() -> None:
    for path in _entries():
        content = path.read_text(encoding="utf-8")
        assert re.findall(r"^- Member: (.+)$", content, re.MULTILINE) == ["Yassine Elhari"]
        assert "- Member: Member 01" not in content
        for section in REQUIRED_EXACT_SECTIONS:
            assert section in content, f"{path.name}: missing {section}"
        for section, equivalents in SECTION_EQUIVALENTS.items():
            assert any(title in content for title in equivalents), (
                f"{path.name}: missing {section} or a clear equivalent"
            )
        assert "- Time spent: TO BE COMPLETED BY YASSINE ELHARI" in content
        assert "- Related meeting: TO BE COMPLETED BY YASSINE ELHARI" in content
        assert "final test" in content.lower()
        assert "closed" in content.lower() or "not used" in content.lower()


def test_pr_placeholder_and_main_code_references() -> None:
    for path in _entries():
        content = path.read_text(encoding="utf-8")
        assert "- Pull Request: To be updated after Pull Request creation" in content

    required_paths = (
        "src/data.py",
        "src/validation.py",
        "src/evaluation.py",
        "src/experiments.py",
        "src/dummy_baselines.py",
        "src/modeling.py",
        "src/logistic_baseline.py",
        "src/logistic_class_weight.py",
        "src/search.py",
        "src/logistic_coefficient_analysis.py",
        "src/learning_curves.py",
        "src/model_selection.py",
        "scripts/run_data_audit.py",
        "scripts/create_data_split.py",
        "scripts/run_logistic_grid_search.py",
        "scripts/run_logistic_coefficient_analysis.py",
        "scripts/run_logistic_learning_curves.py",
        "scripts/build_model_selection_report.py",
    )
    assert all((ROOT / relative_path).is_file() for relative_path in required_paths)


def test_other_member_logbooks_are_unchanged() -> None:
    for relative_path, expected_hash in OTHER_MEMBER_HASHES.items():
        content = (LOGBOOK_ROOT / relative_path).read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected_hash

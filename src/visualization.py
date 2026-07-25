"""Small, style-neutral visualization utilities."""

from pathlib import Path
from typing import Any


def save_figure(fig: Any, path: str | Path, dpi: int = 300) -> Path:
    """Save a Matplotlib-compatible figure and create its parent directory."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path

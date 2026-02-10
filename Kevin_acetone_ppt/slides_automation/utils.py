"""Common utility functions for the slides_automation package."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Union


def slugify(value: str) -> str:
    """Convert a string to a URL/filename-safe slug."""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "presentation"


def ensure_directory(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist and return the Path object."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

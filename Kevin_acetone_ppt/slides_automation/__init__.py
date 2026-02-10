"""Presentation automation toolkit.

This package provides tools for generating research presentations from
YAML/JSON configuration files, including:

- PPTX generation with professional dark theme styling
- Google Slides automation via API
- Manuscript DOCX generation with Pandoc + post-processing
- Quality assurance (placeholder validation, asset checks, margin audits)

Usage:
    python -m slides_automation.cli --help
    python -m slides_automation.audit_pptx path/to/deck.pptx
"""

from .config import (
    GoogleSlidesConfig,
    ImagePlacement,
    LocalSlideDefinition,
    PPTXConfig,
    ProjectConfig,
    load_config,
)
from .google_slides import GoogleSlidesAutomator
from .pptx_generator import LocalPPTBuilder
from .orchestrator import PresentationProject

__all__ = [
    "GoogleSlidesAutomator",
    "LocalPPTBuilder",
    "PresentationProject",
    "GoogleSlidesConfig",
    "ImagePlacement",
    "LocalSlideDefinition",
    "PPTXConfig",
    "ProjectConfig",
    "load_config",
]

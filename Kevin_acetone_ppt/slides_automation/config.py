from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

import yaml


@dataclass
class ImagePlacement:
    """Definition for injecting an image into a slide."""

    slide_index: int
    path: str
    left: float = 1.0
    top: float = 2.0
    width: float = 4.0
    height: float = 3.0


@dataclass
class GoogleSlidesConfig:
    template_id: str
    slide_index_base: int = 0
    folder_id: Optional[str] = None
    share_emails: List[str] = field(default_factory=list)
    image_uploads: List[ImagePlacement] = field(default_factory=list)


@dataclass
class PPTXTypography:
    title_size: int = 42
    subtitle_size: int = 22
    section_title_size: int = 44
    section_subtitle_size: int = 20
    heading_size: int = 28
    body_size: int = 20
    body_secondary_size: int = 18
    bullet_indent_size: int = 18
    table_header_size: int = 14
    table_body_size: int = 12
    caption_size: int = 10
    slide_number_size: int = 12


@dataclass
class PPTXConfig:
    template_path: Optional[str] = None
    output_dir: str = "output"
    title_layout_index: int = 0
    blank_layout_index: int = 6
    font_name: str = "Calibri"
    font_sizes: PPTXTypography = field(default_factory=PPTXTypography)


@dataclass
class LocalSlideDefinition:
    kind: str
    title: Optional[str] = None
    subtitle: Optional[str] = None
    bullets: List[str] = field(default_factory=list)
    text: Optional[str] = None
    image_path: Optional[str] = None
    image_width: float = 8.0
    image_left: Optional[float] = None
    image_top: Optional[float] = None
    table_source: Optional[str] = None
    notes: Optional[str] = None
    left_content: List[str] = field(default_factory=list)
    right_content: List[str] = field(default_factory=list)
    right_image: Optional[str] = None
    equations: List[str] = field(default_factory=list)
    position: Optional[float] = None
    image_border: bool = False


@dataclass
class ProjectConfig:
    project_name: str
    credentials_file: Optional[str]
    default_title: str
    source_path: Optional[str] = None
    placeholders: Dict[str, str] = field(default_factory=dict)
    google_slides: Optional[GoogleSlidesConfig] = None
    pptx: Optional[PPTXConfig] = None
    slides: List[LocalSlideDefinition] = field(default_factory=list)
    data_sources: Dict[str, str] = field(default_factory=dict)

    def resolve_credentials(self, base_dir: Path) -> Optional[str]:
        if not self.credentials_file:
            return None
        return str(_resolve_path(base_dir, self.credentials_file))


def load_config(config_path: str | Path) -> ProjectConfig:
    """Load a YAML/JSON configuration file into a ProjectConfig object."""

    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    base_dir = path.parent

    google_cfg = None
    if raw_config.get("google_slides"):
        google_cfg = _parse_google_config(raw_config["google_slides"], base_dir)

    pptx_cfg = None
    if raw_config.get("pptx"):
        pptx_cfg = _parse_pptx_config(raw_config["pptx"], base_dir)

    slides = [_parse_slide(defn, base_dir) for defn in raw_config.get("slides", [])]

    data_sources = {
        name: str(_resolve_path(base_dir, location))
        for name, location in raw_config.get("data_sources", {}).items()
    }

    project = ProjectConfig(
        project_name=raw_config.get("project_name", "Automated Presentation"),
        credentials_file=str(_resolve_path(base_dir, raw_config["credentials_file"]))
        if raw_config.get("credentials_file")
        else None,
        default_title=raw_config.get("default_title", "Automated Presentation"),
        source_path=str(path),
        placeholders=dict(raw_config.get("placeholders", {})),
        google_slides=google_cfg,
        pptx=pptx_cfg,
        slides=slides,
        data_sources=data_sources,
    )
    return project


def _parse_google_config(raw_cfg: Mapping[str, Any], base_dir: Path) -> GoogleSlidesConfig:
    uploads = [
        ImagePlacement(
            slide_index=upload["slide_index"],
            path=str(_resolve_path(base_dir, upload["path"])),
            left=upload.get("left", 1.0),
            top=upload.get("top", 2.0),
            width=upload.get("width", 4.0),
            height=upload.get("height", 3.0),
        )
        for upload in raw_cfg.get("image_uploads", [])
    ]

    slide_index_base = raw_cfg.get("slide_index_base", 0)
    try:
        slide_index_base = int(slide_index_base)
    except (TypeError, ValueError):
        slide_index_base = 0

    return GoogleSlidesConfig(
        template_id=raw_cfg["template_id"],
        slide_index_base=slide_index_base,
        folder_id=raw_cfg.get("folder_id"),
        share_emails=list(raw_cfg.get("share_emails", [])),
        image_uploads=uploads,
    )


def _parse_pptx_config(raw_cfg: Mapping[str, Any], base_dir: Path) -> PPTXConfig:
    font_sizes = _parse_typography(raw_cfg.get("font_sizes", {}))
    return PPTXConfig(
        template_path=str(_resolve_path(base_dir, raw_cfg["template_path"]))
        if raw_cfg.get("template_path")
        else None,
        output_dir=str(_resolve_path(base_dir, raw_cfg.get("output_dir", "output"))),
        title_layout_index=raw_cfg.get("title_layout_index", 0),
        blank_layout_index=raw_cfg.get("blank_layout_index", 6),
        font_name=raw_cfg.get("font_name", "Calibri"),
        font_sizes=font_sizes,
    )


def _parse_slide(definition: MutableMapping[str, Any], base_dir: Path) -> LocalSlideDefinition:
    return LocalSlideDefinition(
        kind=definition["kind"],
        title=definition.get("title"),
        subtitle=definition.get("subtitle"),
        bullets=list(definition.get("bullets", [])),
        text=definition.get("text"),
        image_path=str(_resolve_path(base_dir, definition["image_path"]))
        if definition.get("image_path")
        else None,
        image_width=definition.get("image_width", 8.0),
        image_left=definition.get("image_left"),
        image_top=definition.get("image_top"),
        table_source=definition.get("table_source"),
        notes=definition.get("notes"),
        left_content=list(definition.get("left_content", [])),
        right_content=list(definition.get("right_content", [])),
        right_image=str(_resolve_path(base_dir, definition["right_image"]))
        if definition.get("right_image")
        else None,
        equations=list(definition.get("equations", [])),
        position=definition.get("position"),
        image_border=bool(definition.get("image_border", False)),
    )


def _resolve_path(base_dir: Path, location: str) -> Path:
    path = Path(location)
    if not path.is_absolute():
        path = base_dir / path
    return path.expanduser().resolve()


def _parse_typography(raw_typography: Mapping[str, Any]) -> PPTXTypography:
    if not raw_typography:
        return PPTXTypography()
    defaults = PPTXTypography()
    fields = defaults.__dataclass_fields__.keys()
    data = {
        field: raw_typography.get(field, getattr(defaults, field))
        for field in fields
    }
    return PPTXTypography(**data)

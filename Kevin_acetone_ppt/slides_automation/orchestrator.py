from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import ProjectConfig
from .data_sources import load_data_sources
from .google_slides import GoogleSlidesAutomator
from .pptx_generator import LocalPPTBuilder
from .quality import (
    apply_auto_metrics,
    check_placeholder_consistency,
    scan_pptx_for_placeholders,
    validate_project,
    write_manifest,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    title: str
    google_slides_link: Optional[str] = None
    pptx_path: Optional[str] = None
    manifest_path: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class PresentationProject:
    """Coordinate Google Slides + PPTX generation from a config file."""

    def __init__(self, config: ProjectConfig):
        self.config = config

    # ------------------------------------------------------------------
    def run(
        self,
        *,
        title: Optional[str] = None,
        generate_google: bool = True,
        generate_pptx: bool = True,
        extra_placeholders: Optional[Dict[str, str]] = None,
        output_name: Optional[str] = None,
        strict: bool = False,
        validate_only: bool = False,
        write_manifest_file: bool = True,
    ) -> GenerationResult:
        final_title = title or self.config.default_title
        placeholders = dict(self.config.placeholders)
        if extra_placeholders:
            placeholders.update(extra_placeholders)
        placeholders.setdefault("{TITLE}", final_title)

        LOGGER.info("Starting generation for '%s'", final_title)
        data_sources = load_data_sources(self.config.data_sources)

        placeholders = apply_auto_metrics(placeholders=placeholders, data_sources=data_sources)

        consistency_warnings = check_placeholder_consistency(
            placeholders=placeholders,
            data_sources=data_sources,
        )

        warnings = validate_project(
            config=self.config,
            placeholders=placeholders,
            strict=strict,
            generate_google=generate_google and bool(self.config.google_slides),
            generate_pptx=generate_pptx and bool(self.config.pptx),
        )
        warnings.extend(consistency_warnings)
        for warning in warnings:
            LOGGER.warning("%s", warning)

        result = GenerationResult(title=final_title, warnings=warnings)

        if validate_only:
            return result

        if generate_google and self.config.google_slides:
            if not self.config.credentials_file:
                LOGGER.error("credentials_file must be provided for Google Slides runs")
            else:
                automator = GoogleSlidesAutomator(self.config.credentials_file)
                copy_result = automator.generate_from_config(
                    self.config.google_slides,
                    title=final_title,
                    replacements=placeholders,
                )
                result.google_slides_link = copy_result.get("webViewLink")
        elif generate_google:
            LOGGER.warning("Google Slides config missing; skipping remote generation")

        if generate_pptx and self.config.pptx:
            builder = LocalPPTBuilder(self.config.pptx)
            pptx_path = builder.build_presentation(
                title=final_title,
                slides=self.config.slides,
                placeholders=placeholders,
                data_sources=data_sources,
                output_name=output_name,
            )
            result.pptx_path = str(pptx_path)

            pptx_placeholder_hits = scan_pptx_for_placeholders(pptx_path)
            if pptx_placeholder_hits:
                msg = "Unresolved placeholders found in generated PPTX:\n" + "\n".join(
                    f"- {hit}" for hit in pptx_placeholder_hits
                )
                if strict:
                    raise ValueError(msg)
                warnings.append(msg)
                LOGGER.warning("%s", msg)

            if write_manifest_file:
                manifest_path = write_manifest(
                    output_pptx_path=pptx_path,
                    config=self.config,
                    title=final_title,
                    placeholders=placeholders,
                    warnings=warnings,
                    google_slides_link=result.google_slides_link,
                )
                result.manifest_path = str(manifest_path)
                LOGGER.info("Wrote build manifest → %s", manifest_path)
        elif generate_pptx:
            LOGGER.warning("PPTX config missing; skipping local deck")

        return result

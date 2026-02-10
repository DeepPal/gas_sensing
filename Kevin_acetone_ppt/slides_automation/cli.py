"""Command-line interface for the slides_automation toolkit.

Usage:
    python -m slides_automation.cli --config config/presentation_scientific.yaml --no-google
    python -m slides_automation.cli --manuscript --autogen-manuscript --no-google --no-pptx
    python -m slides_automation.cli --audit-pptx --audit-pptx-path dist/deck.pptx
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import click

from .config import load_config
from .data_sources import load_data_sources
from .manuscript import build_manuscript
from .orchestrator import PresentationProject
from .quality import (
    collect_markdown_image_paths,
    collect_referenced_paths,
    scan_markdown_for_missing_assets,
    scan_markdown_for_submission_issues,
    scan_pptx_for_picture_margin_issues_xml,
    write_artifact_bundle,
)

LOG_LEVELS = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def parse_key_value(values: tuple[str, ...]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for entry in values:
        if "=" not in entry:
            raise click.BadParameter(
                f"Placeholder overrides must use NAME=VALUE format. Got: '{entry}'"
            )
        key, value = entry.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


@click.command(context_settings={"show_default": True})
@click.option(
    "--config",
    "config_path",
    required=False,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("config/presentation_scientific.yaml"),
    show_default=True,
    help="Path to the project configuration file (YAML or JSON).",
)
@click.option("--title", "title_override", help="Override the default presentation title.")
@click.option("--no-google", is_flag=True, help="Skip Google Slides generation.")
@click.option("--no-pptx", is_flag=True, help="Skip local PPTX generation.")
@click.option(
    "--set",
    "placeholder_overrides",
    multiple=True,
    help="Override placeholder value, e.g. --set {DATE}=2025-12-09",
)
@click.option(
    "--placeholders-json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Load additional placeholder overrides from a JSON file.",
)
@click.option(
    "--output-name",
    type=str,
    help="File-friendly name for PPTX export (defaults to slugified title).",
)
@click.option(
    "--log-level",
    type=click.Choice(LOG_LEVELS, case_sensitive=False),
    default="INFO",
    help="Logging verbosity.",
)
@click.option("--dry-run", is_flag=True, help="Validate inputs/placeholders without generating files.")
@click.option(
    "--validate-only",
    is_flag=True,
    help="Validate inputs/placeholders and exit (no deck generation).",
)
@click.option(
    "--strict",
    is_flag=True,
    help="Fail the run if any referenced assets are missing or placeholders remain unresolved.",
)
@click.option(
    "--no-manifest",
    is_flag=True,
    help="Do not write the reproducibility manifest next to the PPTX.",
)
@click.option(
    "--manuscript",
    is_flag=True,
    help="Generate a manuscript DOCX export (optionally with auto-generated sections from CSVs).",
)
@click.option(
    "--autogen-manuscript",
    is_flag=True,
    help="Auto-generate key Results/Discussion sections from CSVs before exporting the manuscript.",
)
@click.option(
    "--manuscript-template",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("help_files/MANUSCRIPT_DRAFT.md"),
    show_default=True,
    help="Path to the manuscript Markdown template.",
)
@click.option(
    "--manuscript-md-out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("help_files/MANUSCRIPT_DRAFT.autogen.md"),
    show_default=True,
    help="Output Markdown path when --autogen-manuscript is enabled.",
)
@click.option(
    "--manuscript-docx-out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("help_files/MANUSCRIPT_DRAFT.docx"),
    show_default=True,
    help="Output DOCX path for manuscript export.",
)
@click.option(
    "--bundle-out",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write a reproducible submission bundle ZIP containing generated artifacts + referenced assets.",
)
@click.option(
    "--audit-pptx",
    is_flag=True,
    help="Run fast ZIP/XML-based audit for picture overflow/tight margins on a PPTX.",
)
@click.option(
    "--audit-pptx-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="PPTX file to audit (defaults to generated PPTX when available).",
)
@click.option(
    "--audit-min-margin-in",
    type=float,
    default=0.25,
    show_default=True,
    help="Minimum allowed margin (inches) for pictures.",
)
def main(
    config_path: Path,
    title_override: str | None,
    no_google: bool,
    no_pptx: bool,
    placeholder_overrides: tuple[str, ...],
    placeholders_json: Path | None,
    output_name: str | None,
    log_level: str,
    dry_run: bool,
    validate_only: bool,
    strict: bool,
    no_manifest: bool,
    manuscript: bool,
    autogen_manuscript: bool,
    manuscript_template: Path,
    manuscript_md_out: Path,
    manuscript_docx_out: Path,
    bundle_out: Path | None,
    audit_pptx: bool,
    audit_pptx_path: Path | None,
    audit_min_margin_in: float,
) -> None:
    """Generate research presentations from templates."""

    configure_logging(log_level)
    logger = logging.getLogger("slides_automation.cli")

    config = load_config(config_path)
    project = PresentationProject(config)

    replacements = parse_key_value(placeholder_overrides)
    if placeholders_json:
        json_payload = json.loads(placeholders_json.read_text(encoding="utf-8"))
        if not isinstance(json_payload, dict):
            raise click.BadParameter("JSON placeholder file must contain an object")
        replacements.update({str(k): str(v) for k, v in json_payload.items()})

    title = title_override or config.default_title

    logger.info("Prepared project '%s' (title: %s)", config.project_name, title)
    logger.info("Google Slides enabled: %s", not no_google and bool(config.google_slides))
    logger.info("Local PPTX enabled: %s", not no_pptx and bool(config.pptx))

    if dry_run or validate_only:
        result = project.run(
            title=title,
            generate_google=not no_google,
            generate_pptx=not no_pptx,
            extra_placeholders=replacements,
            output_name=output_name,
            strict=strict,
            validate_only=True,
            write_manifest_file=not no_manifest,
        )
        if result.warnings:
            logger.warning("Validation produced %d warning(s)", len(result.warnings))

        if manuscript:
            project_root = Path(__file__).resolve().parents[1]
            manuscript_warnings = scan_markdown_for_missing_assets(
                markdown_path=manuscript_template,
                project_root=project_root,
            )
            for warning in manuscript_warnings:
                logger.warning("%s", warning)
            if strict and manuscript_warnings:
                raise click.ClickException(
                    "Missing manuscript assets detected (strict mode). Fix missing files and re-run."
                )

            if autogen_manuscript:
                data_sources = load_data_sources(config.data_sources)
                required_keys = [
                    "reference_metrics",
                    "comparison_table",
                    "multigas_results",
                    "voc_protocol",
                ]
                missing_keys = [
                    key for key in required_keys if key not in data_sources or data_sources.get(key) is None
                ]
                if missing_keys:
                    msg = "Missing required data_sources for --autogen-manuscript: " + ", ".join(
                        missing_keys
                    )
                    if strict:
                        raise click.ClickException(msg)
                    logger.warning("%s", msg)

        if audit_pptx:
            if audit_pptx_path is None:
                raise click.ClickException(
                    "--audit-pptx-path is required when running with --dry-run/--validate-only."
                )
            per_slide, meta = scan_pptx_for_picture_margin_issues_xml(
                pptx_path=audit_pptx_path,
                min_margin_in=audit_min_margin_in,
            )
            click.echo(f"Fast PPTX XML audit: {audit_pptx_path}")
            click.echo(f"- Slide size (EMU): {meta.get('slide_w')} x {meta.get('slide_h')}")
            click.echo(f"- Margin threshold: {audit_min_margin_in} in")
            if not per_slide:
                click.echo("OK: No picture margin/overflow issues detected.")
            else:
                click.echo("Slide summary (slide: overflow_count, tight_count):")
                for si in sorted(per_slide):
                    d = per_slide[si]
                    click.echo(f"- {si}: overflow={d['overflow']}, tight={d['tight']}")

        logger.info("Validation complete. No artifacts generated.")
        return

    result = project.run(
        title=title,
        generate_google=not no_google,
        generate_pptx=not no_pptx,
        extra_placeholders=replacements,
        output_name=output_name,
        strict=strict,
        validate_only=False,
        write_manifest_file=not no_manifest,
    )

    if result.google_slides_link:
        logger.info("Google Slides deck: %s", result.google_slides_link)
    if result.pptx_path:
        logger.info("Local PPTX saved to: %s", result.pptx_path)
    if result.manifest_path:
        logger.info("Build manifest: %s", result.manifest_path)

    if audit_pptx:
        target_pptx = audit_pptx_path
        if target_pptx is None and result.pptx_path:
            target_pptx = Path(result.pptx_path)
        if target_pptx is None:
            raise click.ClickException(
                "No PPTX available to audit. Provide --audit-pptx-path or enable PPTX generation."
            )
        per_slide, meta = scan_pptx_for_picture_margin_issues_xml(
            pptx_path=target_pptx,
            min_margin_in=audit_min_margin_in,
        )
        click.echo(f"Fast PPTX XML audit: {target_pptx}")
        click.echo(f"- Slide size (EMU): {meta.get('slide_w')} x {meta.get('slide_h')}")
        click.echo(f"- Margin threshold: {audit_min_margin_in} in")
        if not per_slide:
            click.echo("OK: No picture margin/overflow issues detected.")
        else:
            click.echo("Slide summary (slide: overflow_count, tight_count):")
            for si in sorted(per_slide):
                d = per_slide[si]
                click.echo(f"- {si}: overflow={d['overflow']}, tight={d['tight']}")

    md_path: Path | None = None
    docx_path: Path | None = None

    if manuscript:
        data_sources = load_data_sources(config.data_sources)
        md_path, docx_path = build_manuscript(
            manuscript_template_path=manuscript_template,
            manuscript_markdown_out=manuscript_md_out,
            manuscript_docx_out=manuscript_docx_out,
            data_sources=data_sources,
            autogen=autogen_manuscript,
        )
        logger.info("Manuscript markdown: %s", md_path)
        logger.info("Manuscript docx: %s", docx_path)

        project_root = Path(__file__).resolve().parents[1]
        manuscript_warnings = scan_markdown_for_missing_assets(
            markdown_path=md_path,
            project_root=project_root,
        )
        for warning in manuscript_warnings:
            logger.warning("%s", warning)
        if strict and manuscript_warnings:
            raise click.ClickException(
                "Missing manuscript assets detected (strict mode). Fix missing files and re-run."
            )

        submission_issues = scan_markdown_for_submission_issues(markdown_path=md_path)
        for issue in submission_issues:
            logger.warning("%s", issue)
        if strict and any(issue.startswith("BLOCKER:") for issue in submission_issues):
            raise click.ClickException(
                "Submission blockers detected in manuscript (strict mode). Review warnings and fix before re-run."
            )

        md_text = md_path.read_text(encoding="utf-8")
        embedded_images = collect_markdown_image_paths(md_text)
        if embedded_images:
            try:
                docx_size = docx_path.stat().st_size
            except OSError:
                docx_size = 0
            if docx_size and docx_size < 200_000:
                msg = (
                    "Manuscript Markdown contains embedded images, but the DOCX output is unusually small. "
                    "This often indicates images were not embedded (path resolution issue)."
                )
                if strict:
                    raise click.ClickException(msg)
                logger.warning("%s", msg)

    if bundle_out:
        project_root = Path(__file__).resolve().parents[1]

        bundle_files: list[Path] = []

        bundle_files.append(config_path)

        if result.pptx_path:
            bundle_files.append(Path(result.pptx_path))
        if result.manifest_path:
            bundle_files.append(Path(result.manifest_path))

        if md_path:
            bundle_files.append(md_path)
        if docx_path:
            bundle_files.append(docx_path)

        # Include all referenced project assets (slides + data sources).
        for ref in collect_referenced_paths(config):
            ref_path = Path(ref)
            if not ref_path.is_absolute():
                ref_path = project_root / ref_path
            bundle_files.append(ref_path)

        # Include all manuscript-embedded image assets.
        if md_path:
            md_text = md_path.read_text(encoding="utf-8")
            for rel in collect_markdown_image_paths(md_text):
                img_path = Path(rel)
                if not img_path.is_absolute():
                    img_path = project_root / img_path
                bundle_files.append(img_path)

                # If a corresponding PDF exists next to the referenced image, include it too.
                if img_path.suffix.lower() == ".png":
                    pdf_candidate = img_path.with_suffix(".pdf")
                    if pdf_candidate.exists():
                        bundle_files.append(pdf_candidate)

        bundle_path = write_artifact_bundle(
            bundle_path=bundle_out,
            project_root=project_root,
            files=bundle_files,
        )
        logger.info("Wrote submission bundle → %s", bundle_path)


if __name__ == "__main__":  # pragma: no cover
    main()

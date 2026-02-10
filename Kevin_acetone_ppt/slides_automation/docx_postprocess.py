"""DOCX post-processing for single-column compact journal-like styling.

This module edits the DOCX XML (document.xml, styles.xml) to enforce:
- Single-column layout
- A4 or Letter page size
- Compact margins (default 0.75 inch)
- Times New Roman font with configurable sizes
- Compact heading/paragraph spacing
"""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

LOGGER = logging.getLogger(__name__)

_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _q(local: str) -> str:
    return f"{{{_W_NS}}}{local}"


def _ensure_child(parent: ET.Element, local: str) -> ET.Element:
    child = parent.find(_q(local))
    if child is None:
        child = ET.SubElement(parent, _q(local))
    return child


def _twips_from_inches(value_in: float) -> int:
    return int(round(float(value_in) * 1440.0))


def _set_rpr_font_and_size(*, rpr: ET.Element, font_name: str, size_half_points: int) -> None:
    rfonts = _ensure_child(rpr, "rFonts")
    rfonts.set(_q("ascii"), font_name)
    rfonts.set(_q("hAnsi"), font_name)
    rfonts.set(_q("cs"), font_name)
    rfonts.set(_q("eastAsia"), font_name)

    sz = _ensure_child(rpr, "sz")
    sz.set(_q("val"), str(size_half_points))

    sz_cs = _ensure_child(rpr, "szCs")
    sz_cs.set(_q("val"), str(size_half_points))


def _set_ppr_spacing(*, ppr: ET.Element, before_twips: int, after_twips: int, line: Optional[int] = None) -> None:
    spacing = _ensure_child(ppr, "spacing")
    spacing.set(_q("before"), str(int(before_twips)))
    spacing.set(_q("after"), str(int(after_twips)))
    if line is not None:
        spacing.set(_q("line"), str(int(line)))
        spacing.set(_q("lineRule"), "auto")


def _style_id_matches(style: ET.Element, candidates: set[str]) -> bool:
    style_id = style.get(_q("styleId")) or style.get("styleId")
    if style_id and style_id in candidates:
        return True

    name = style.find(_q("name"))
    if name is not None:
        name_val = name.get(_q("val")) or name.get("val")
        if name_val and name_val in candidates:
            return True

    return False


def enforce_single_column_compact_style(
    *,
    docx_path: str | Path,
    page: str = "A4",
    margin_in: float = 0.75,
    body_font: str = "Times New Roman",
    body_pt: int = 10,
    caption_pt: int = 9,
) -> None:
    docx_path = Path(docx_path)
    if not docx_path.exists():
        raise FileNotFoundError(f"DOCX not found: {docx_path}")

    tmp_path = docx_path.parent / f"{docx_path.name}.tmp"

    body_half_points = int(body_pt) * 2
    caption_half_points = int(caption_pt) * 2

    margin_twips = _twips_from_inches(margin_in)

    if str(page).upper() == "A4":
        pg_w = 11906
        pg_h = 16838
    elif str(page).upper() == "LETTER":
        pg_w = 12240
        pg_h = 15840
    else:
        raise ValueError("page must be 'A4' or 'Letter'")

    with zipfile.ZipFile(docx_path, "r") as zin:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)

                if item.filename == "word/document.xml":
                    root = ET.fromstring(data)
                    for sect_pr in root.findall(f".//{_q('sectPr')}"):
                        cols = _ensure_child(sect_pr, "cols")
                        cols.set(_q("num"), "1")

                        pg_mar = _ensure_child(sect_pr, "pgMar")
                        pg_mar.set(_q("top"), str(margin_twips))
                        pg_mar.set(_q("bottom"), str(margin_twips))
                        pg_mar.set(_q("left"), str(margin_twips))
                        pg_mar.set(_q("right"), str(margin_twips))
                        pg_mar.set(_q("header"), "720")
                        pg_mar.set(_q("footer"), "720")
                        pg_mar.set(_q("gutter"), "0")

                        pg_sz = _ensure_child(sect_pr, "pgSz")
                        pg_sz.set(_q("w"), str(pg_w))
                        pg_sz.set(_q("h"), str(pg_h))

                    data = ET.tostring(root, encoding="utf-8", xml_declaration=True)

                elif item.filename == "word/styles.xml":
                    root = ET.fromstring(data)
                    for style in root.findall(f".//{_q('style')}"):
                        if _style_id_matches(style, {"Normal"}):
                            rpr = _ensure_child(style, "rPr")
                            _set_rpr_font_and_size(
                                rpr=rpr,
                                font_name=body_font,
                                size_half_points=body_half_points,
                            )
                            ppr = _ensure_child(style, "pPr")
                            _set_ppr_spacing(ppr=ppr, before_twips=0, after_twips=60)

                        elif _style_id_matches(style, {"Caption"}):
                            rpr = _ensure_child(style, "rPr")
                            _set_rpr_font_and_size(
                                rpr=rpr,
                                font_name=body_font,
                                size_half_points=caption_half_points,
                            )
                            ppr = _ensure_child(style, "pPr")
                            _set_ppr_spacing(ppr=ppr, before_twips=0, after_twips=60)

                        elif _style_id_matches(style, {"Heading1", "Heading 1"}):
                            rpr = _ensure_child(style, "rPr")
                            _set_rpr_font_and_size(
                                rpr=rpr,
                                font_name=body_font,
                                size_half_points=max(body_half_points + 4, body_half_points),
                            )
                            ppr = _ensure_child(style, "pPr")
                            _set_ppr_spacing(ppr=ppr, before_twips=120, after_twips=60)

                        elif _style_id_matches(style, {"Heading2", "Heading 2"}):
                            rpr = _ensure_child(style, "rPr")
                            _set_rpr_font_and_size(
                                rpr=rpr,
                                font_name=body_font,
                                size_half_points=max(body_half_points + 2, body_half_points),
                            )
                            ppr = _ensure_child(style, "pPr")
                            _set_ppr_spacing(ppr=ppr, before_twips=120, after_twips=60)

                    data = ET.tostring(root, encoding="utf-8", xml_declaration=True)

                zout.writestr(item, data)

    try:
        tmp_path.replace(docx_path)
    except PermissionError as exc:
        raise PermissionError(
            f"Could not update '{docx_path}'. Close the file in Word and try again."
        ) from exc

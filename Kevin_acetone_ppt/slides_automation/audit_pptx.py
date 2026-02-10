"""Standalone PPTX margin/overflow audit CLI.

Usage:
    python -m slides_automation.audit_pptx path/to/deck.pptx --min-margin-in 0.25
    python -m slides_automation.audit_pptx path/to/deck.pptx --out report.json --fail-on-issues
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .quality import scan_pptx_for_picture_margin_issues_xml

LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Run PPTX margin/overflow audit and optionally write JSON report."""
    parser = argparse.ArgumentParser(
        prog="python -m slides_automation.audit_pptx",
        description="Audit PPTX for picture overflow and tight margins.",
    )
    parser.add_argument(
        "pptx",
        type=str,
        help="Path to a .pptx file to audit.",
    )
    parser.add_argument(
        "--min-margin-in",
        type=float,
        default=0.25,
        help="Minimum allowed margin (inches) for pictures.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to write a JSON audit report.",
    )
    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="Exit non-zero if any overflow/tight-margin issues are detected.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    pptx_path = Path(args.pptx)
    if not pptx_path.exists():
        LOGGER.error("PPTX not found: %s", pptx_path)
        return 1

    per_slide, meta = scan_pptx_for_picture_margin_issues_xml(
        pptx_path=pptx_path,
        min_margin_in=float(args.min_margin_in),
    )

    print(f"Fast PPTX XML audit: {pptx_path}")
    print(f"- Slide size (EMU): {meta.get('slide_w')} x {meta.get('slide_h')}")
    print(f"- Margin threshold: {float(args.min_margin_in)} in")

    if not per_slide:
        print("OK: No picture margin/overflow issues detected.")
    else:
        print("Slide summary (slide: overflow_count, tight_count):")
        for si in sorted(per_slide):
            d = per_slide[si]
            print(f"- {si}: overflow={d['overflow']}, tight={d['tight']}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pptx_path": str(pptx_path),
            "min_margin_in": float(args.min_margin_in),
            "meta": meta,
            "per_slide": per_slide,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.fail_on_issues and per_slide:
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

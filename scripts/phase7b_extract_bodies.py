"""Phase 7b: Extract CONFIG-free private functions to new src/ modules.

Creates:
  src/batch/preprocessing.py  -- sort_frame_paths
  src/batch/response.py       -- scale_reference_to_baseline, score_trial_quality,
                                 summarize_responsive_delta, aggregate_responsive_delta_maps
                                 (+ _safe_float helper)

Then updates pipeline.py:
  - Adds imports for both new modules
  - Replaces the 5+1=6 function bodies with thin delegates

Run from project root:
    python scripts/phase7b_extract_bodies.py
"""

from __future__ import annotations

import ast
import pathlib
import sys
import textwrap

PIPELINE = pathlib.Path("gas_analysis/core/pipeline.py")

# ---------------------------------------------------------------------------
# Target functions and their public names in src/
# ---------------------------------------------------------------------------
PREPROCESSING_TARGETS = {
    "_sort_frame_paths": "sort_frame_paths",
}
RESPONSE_TARGETS = {
    "_safe_float": "_safe_float",                            # stays private helper
    "_scale_reference_to_baseline": "scale_reference_to_baseline",
    "_score_trial_quality": "score_trial_quality",
    "_summarize_responsive_delta": "summarize_responsive_delta",
    "_aggregate_responsive_delta_maps": "aggregate_responsive_delta_maps",
}


def extract_fn_lines(tree: ast.AST, src_lines: list[str], fname: str) -> list[str]:
    """Return verbatim source lines for the named top-level function."""
    node = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == fname and n.col_offset == 0),
        None,
    )
    if node is None:
        raise ValueError(f"Function {fname!r} not found")
    return src_lines[node.lineno - 1 : node.end_lineno]


def rename_fn(lines: list[str], old_name: str, new_name: str) -> list[str]:
    """Rename the function definition on the first line."""
    if old_name == new_name:
        return lines
    result = list(lines)
    result[0] = result[0].replace(f"def {old_name}(", f"def {new_name}(", 1)
    return result


def replace_signal_column(lines: list[str]) -> list[str]:
    """Substitute _signal_column( with select_signal_column( throughout."""
    return [l.replace("_signal_column(", "select_signal_column(") for l in lines]


def build_preprocessing_module(pipeline_src: str, tree: ast.AST, src_lines: list[str]) -> str:
    """Build src/batch/preprocessing.py content."""
    fn_lines = extract_fn_lines(tree, src_lines, "_sort_frame_paths")
    fn_lines = rename_fn(fn_lines, "_sort_frame_paths", "sort_frame_paths")

    body = "\n".join(fn_lines)
    return (
        '"""Batch preprocessing utilities — file sorting and frame ordering.\n\n'
        "All functions are CONFIG-free: every parameter is passed explicitly.\n"
        '"""\n'
        "from __future__ import annotations\n\n"
        "import math\n"
        "import os\n"
        "import re\n"
        "from collections.abc import Sequence\n\n\n"
        f"{body}\n"
    )


def build_response_module(pipeline_src: str, tree: ast.AST, src_lines: list[str]) -> str:
    """Build src/batch/response.py content."""
    blocks: list[str] = []
    for old_name, new_name in RESPONSE_TARGETS.items():
        fn_lines = extract_fn_lines(tree, src_lines, old_name)
        fn_lines = rename_fn(fn_lines, old_name, new_name)
        fn_lines = replace_signal_column(fn_lines)
        blocks.append("\n".join(fn_lines))

    sep = "\n\n\n"
    return (
        '"""Batch response-analysis utilities — delta summarisation and aggregation.\n\n'
        "All functions are CONFIG-free: every parameter is passed explicitly.\n"
        '"""\n'
        "from __future__ import annotations\n\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from collections.abc import Sequence\n"
        "from typing import Any, Optional\n\n"
        "from src.reporting.metrics import select_signal_column\n\n\n"
        f"{sep.join(blocks)}\n"
    )


def main() -> None:
    src = PIPELINE.read_text(encoding="utf-8")
    lines = src.splitlines(keepends=False)  # keep without trailing newlines for extraction
    tree = ast.parse(src)

    # ------------------------------------------------------------------
    # 1. Write src/ modules
    # ------------------------------------------------------------------
    pre_path = pathlib.Path("src/batch/preprocessing.py")
    resp_path = pathlib.Path("src/batch/response.py")

    if pre_path.exists():
        print(f"  {pre_path} already exists — skipping creation.")
    else:
        content = build_preprocessing_module(src, tree, lines)
        # Syntax check
        try:
            ast.parse(content)
        except SyntaxError as e:
            print(f"SYNTAX ERROR in preprocessing module: {e}", file=sys.stderr)
            sys.exit(1)
        pre_path.write_text(content, encoding="utf-8")
        print(f"  Created {pre_path} ({content.count(chr(10))} lines).")

    if resp_path.exists():
        print(f"  {resp_path} already exists — skipping creation.")
    else:
        content = build_response_module(src, tree, lines)
        try:
            ast.parse(content)
        except SyntaxError as e:
            print(f"SYNTAX ERROR in response module: {e}", file=sys.stderr)
            sys.exit(1)
        resp_path.write_text(content, encoding="utf-8")
        print(f"  Created {resp_path} ({content.count(chr(10))} lines).")

    # ------------------------------------------------------------------
    # 2. Now patch pipeline.py: add imports + replace bodies
    # ------------------------------------------------------------------
    src = PIPELINE.read_text(encoding="utf-8")  # re-read (files unchanged so far)
    lines = src.splitlines(keepends=True)

    IMPORT_BLOCK = (
        "from src.batch.preprocessing import sort_frame_paths as _sort_frame_paths_src\n"
        "from src.batch.response import (\n"
        "    aggregate_responsive_delta_maps as _aggregate_responsive_delta_maps_src,\n"
        "    scale_reference_to_baseline as _scale_reference_to_baseline_src,\n"
        "    score_trial_quality as _score_trial_quality_src,\n"
        "    summarize_responsive_delta as _summarize_responsive_delta_src,\n"
        ")\n"
    )

    if IMPORT_BLOCK.splitlines()[0] in src:
        print("  Import block already present — skipping injection.")
    else:
        # Parse again on fresh src
        tree2 = ast.parse(src)

        # Collect replacement ranges for the 5 public functions
        BODIES: dict[str, str] = {
            "_sort_frame_paths": (
                "    return _sort_frame_paths_src(paths)\n"
            ),
            "_scale_reference_to_baseline": (
                "    return _scale_reference_to_baseline_src(\n"
                "        ref_df, baseline_frames, percentile=percentile\n"
                "    )\n"
            ),
            "_score_trial_quality": (
                "    return _score_trial_quality_src(\n"
                "        df, roi_bounds=roi_bounds, expected_center=expected_center\n"
                "    )\n"
            ),
            "_summarize_responsive_delta": (
                "    return _summarize_responsive_delta_src(df)\n"
            ),
            "_aggregate_responsive_delta_maps": (
                "    return _aggregate_responsive_delta_maps_src(responsive_delta_by_conc)\n"
            ),
        }

        replacements: list[tuple[int, int, list[str], str, str]] = []
        for fname, new_body in BODIES.items():
            node = next(
                (n for n in ast.walk(tree2)
                 if isinstance(n, ast.FunctionDef) and n.name == fname and n.col_offset == 0),
                None,
            )
            if node is None:
                print(f"  WARNING: {fname} not found — skipping.")
                continue
            start = node.lineno - 1
            end = node.end_lineno
            body_start = node.body[0].lineno - 1
            kept = lines[start:body_start]
            replacements.append((start, end, kept, new_body, fname))

        # Bottom-to-top replacements
        replacements.sort(key=lambda x: x[0], reverse=True)
        total_removed = 0
        for start, end, kept, new_body, fname in replacements:
            old_len = end - start
            new_node_lines = kept + [new_body]
            delta = len(new_node_lines) - old_len
            total_removed -= delta
            lines = lines[:start] + new_node_lines + lines[end:]
            print(f"  {fname}: {old_len} -> {len(new_node_lines)} lines (delta={delta:+d})")

        # Inject import block after Phase 7a block
        ANCHOR = "from src.batch.aggregation import ("
        anchor_idx = next(
            (i for i, l in enumerate(lines) if l.strip().startswith(ANCHOR)), None
        )
        if anchor_idx is None:
            print(f"ERROR: anchor not found", file=sys.stderr)
            sys.exit(1)
        close_idx = anchor_idx
        while close_idx < len(lines) and ")" not in lines[close_idx]:
            close_idx += 1
        insert_at = close_idx + 1
        lines = (
            lines[:insert_at]
            + IMPORT_BLOCK.splitlines(keepends=True)
            + lines[insert_at:]
        )
        print(f"  Injected Phase 7b import block after line {insert_at + 1}.")

        new_src = "".join(lines)
        try:
            ast.parse(new_src)
        except SyntaxError as e:
            print(f"SYNTAX ERROR in pipeline.py after edits: {e}", file=sys.stderr)
            sys.exit(1)

        PIPELINE.write_text(new_src, encoding="utf-8")
        old_count = src.count("\n")
        new_count = new_src.count("\n")
        print(f"\nDone. pipeline.py lines: {old_count} -> {new_count} "
              f"(net -{old_count - new_count}, body savings: {total_removed}).")


if __name__ == "__main__":
    main()

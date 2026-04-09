"""Phase 7a: Redirect 4 batch aggregation functions already in src/batch/aggregation.py.

Functions redirected:
  find_stable_block            (97 lines → 2)
  average_stable_block         (64 lines → 2)
  average_top_frames           (67 lines → 2)
  select_canonical_per_concentration (69 lines → 2)

Run from project root:
    python scripts/phase7a_batch_aggregation.py
"""

import ast
import pathlib
import sys

PIPELINE = pathlib.Path("gas_analysis/core/pipeline.py")

IMPORT_LINE = (
    "from src.batch.aggregation import (\n"
    "    average_stable_block as _average_stable_block_src,\n"
    "    average_top_frames as _average_top_frames_src,\n"
    "    find_stable_block as _find_stable_block_src,\n"
    "    select_canonical_per_concentration as _select_canonical_src,\n"
    ")\n"
)

# Anchor: inject after the last src.reporting.plots import group (the Phase 6 block)
ANCHOR = "from src.reporting.plots import ("

BODIES: dict[str, str] = {
    "find_stable_block": (
        "    return _find_stable_block_src(\n"
        "        frames, diff_threshold=diff_threshold, weight_mode=weight_mode,\n"
        "        min_block=min_block\n"
        "    )\n"
    ),
    "average_stable_block": (
        "    return _average_stable_block_src(frames, start_idx, end_idx, weights=weights)\n"
    ),
    "average_top_frames": (
        "    return _average_top_frames_src(\n"
        "        frames, top_n=top_n, signal_col=signal_col, ascending=ascending\n"
        "    )\n"
    ),
    "select_canonical_per_concentration": (
        "    return _select_canonical_src(stable_results)\n"
    ),
}


def main() -> None:
    src = PIPELINE.read_text(encoding="utf-8")
    lines = src.splitlines(keepends=True)

    if IMPORT_LINE.splitlines()[0] in src:
        print("Import already present — already migrated, exiting.")
        return

    # Parse ONCE
    tree = ast.parse(src)

    # Collect replacement ranges
    replacements: list[tuple[int, int, list[str], str, str]] = []
    for fname, new_body in BODIES.items():
        node = next(
            (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fname),
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

    # Apply bottom-to-top
    replacements.sort(key=lambda x: x[0], reverse=True)
    total_removed = 0
    for start, end, kept, new_body, fname in replacements:
        old_len = end - start
        new_node_lines = kept + [new_body]
        delta = len(new_node_lines) - old_len
        total_removed -= delta
        lines = lines[:start] + new_node_lines + lines[end:]
        print(f"  {fname}: {old_len} -> {len(new_node_lines)} lines (delta={delta:+d})")

    # Inject import block after the Phase 6 plots import group closing ')'
    anchor_idx = next(
        (i for i, l in enumerate(lines) if l.strip().startswith(ANCHOR)), None
    )
    if anchor_idx is None:
        print("ERROR: anchor not found", file=sys.stderr)
        sys.exit(1)
    close_idx = anchor_idx
    while close_idx < len(lines) and ")" not in lines[close_idx]:
        close_idx += 1
    insert_at = close_idx + 1
    lines = lines[:insert_at] + IMPORT_LINE.splitlines(keepends=True) + lines[insert_at:]
    print(f"  Injected Phase 7a import block after line {insert_at + 1}.")

    # Syntax check + write
    new_src = "".join(lines)
    try:
        ast.parse(new_src)
    except SyntaxError as e:
        print(f"SYNTAX ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    PIPELINE.write_text(new_src, encoding="utf-8")
    old_lines = src.count("\n")
    new_lines = new_src.count("\n")
    print(f"\nDone. Lines: {old_lines} -> {new_lines} (net -{old_lines - new_lines}).")


if __name__ == "__main__":
    main()

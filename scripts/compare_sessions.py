"""
Compare Sessions
----------------
Load and overlay pipeline_results.csv from multiple recorded sensor sessions
for comparative analysis (e.g., different gas concentrations, different runs).

Usage::

    # Compare two sessions by path
    python scripts/compare_sessions.py \\
        output/sessions/20260224_145530 \\
        output/sessions/20260224_152130

    # Compare all sessions in output/sessions/
    python scripts/compare_sessions.py --all

    # Show specific metric
    python scripts/compare_sessions.py --all --metric wavelength_shift

    # Save plot to file
    python scripts/compare_sessions.py --all --output comparison.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_session(session_dir: Path) -> tuple:
    """
    Load a session directory.

    Returns:
        (df, meta) where df is the pipeline_results DataFrame
        and meta is the session_meta dict.
    """
    import pandas as pd

    results_path = session_dir / "pipeline_results.csv"
    meta_path = session_dir / "session_meta.json"

    if not results_path.exists():
        raise FileNotFoundError(f"No pipeline_results.csv in {session_dir}")

    df = pd.read_csv(results_path)
    if "timestamp" in df.columns:
        # Normalise timestamp to seconds from session start
        df["time_s"] = df["timestamp"] - df["timestamp"].iloc[0]

    meta = {}
    if meta_path.exists():
        with open(meta_path) as fh:
            meta = json.load(fh)

    return df, meta


def discover_sessions(sessions_root: Path) -> list:
    """Return all session directories (those containing pipeline_results.csv)."""
    return sorted(
        [d for d in sessions_root.iterdir() if d.is_dir() and (d / "pipeline_results.csv").exists()]
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_comparison(
    sessions: list,  # list of (session_dir, df, meta)
    metric: str = "concentration_ppm",
    output_path: str | None = None,
) -> None:
    """Overlay metric time series from multiple sessions."""
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    colors = cm.tab10(np.linspace(0, 1, len(sessions)))

    ax_main = axes[0]
    ax_snr = axes[1]

    for i, (session_dir, df, meta) in enumerate(sessions):
        label = meta.get("gas_label", session_dir.name)
        session_id = meta.get("session_id", session_dir.name)
        color = colors[i]

        x = df.get("time_s", range(len(df)))

        # Main metric
        if metric in df.columns:
            y = df[metric]
            ax_main.plot(
                x, y, color=color, linewidth=1.5, label=f"{label} [{session_id}]", alpha=0.85
            )

        # SNR
        if "snr" in df.columns:
            ax_snr.plot(x, df["snr"], color=color, linewidth=1.0, alpha=0.7)

    ax_main.set_ylabel(metric.replace("_", " ").title())
    ax_main.set_title(f"Session Comparison — {metric.replace('_', ' ').title()}")
    ax_main.legend(fontsize=8, loc="upper right")
    ax_main.grid(True, alpha=0.3)

    ax_snr.set_xlabel("Time (s)")
    ax_snr.set_ylabel("SNR")
    ax_snr.set_title("Signal-to-Noise Ratio")
    ax_snr.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary_table(sessions: list, metric: str = "concentration_ppm") -> None:
    """Print a summary statistics table for all sessions."""
    import pandas as pd

    rows = []
    for session_dir, df, meta in sessions:
        row = {
            "session_id": meta.get("session_id", session_dir.name),
            "gas_label": meta.get("gas_label", "unknown"),
            "samples": len(df),
        }
        if metric in df.columns:
            s = df[metric].dropna()
            row[f"{metric}_mean"] = round(s.mean(), 4)
            row[f"{metric}_std"] = round(s.std(), 4)
            row[f"{metric}_min"] = round(s.min(), 4)
            row[f"{metric}_max"] = round(s.max(), 4)
        if "snr" in df.columns:
            row["snr_mean"] = round(df["snr"].dropna().mean(), 2)
        rows.append(row)

    summary = pd.DataFrame(rows)
    print("\n--- Session Comparison Summary ---")
    print(summary.to_string(index=False))
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare sensor sessions")
    parser.add_argument(
        "session_dirs",
        nargs="*",
        help="Paths to session directories (omit to use --all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Compare all sessions found in output/sessions/",
    )
    parser.add_argument(
        "--sessions-root",
        default="output/sessions",
        help="Root directory to scan when using --all",
    )
    parser.add_argument(
        "--metric",
        default="concentration_ppm",
        help="Column to compare (default: concentration_ppm)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save plot to this file (e.g., comparison.png) instead of displaying",
    )
    args = parser.parse_args()

    # Resolve session directories
    if args.all:
        session_paths = discover_sessions(Path(args.sessions_root))
        if not session_paths:
            print(f"No sessions found in {args.sessions_root}")
            sys.exit(1)
    elif args.session_dirs:
        session_paths = [Path(p) for p in args.session_dirs]
    else:
        parser.print_help()
        sys.exit(0)

    print(f"Loading {len(session_paths)} session(s)…")
    sessions = []
    for sp in session_paths:
        try:
            df, meta = load_session(sp)
            sessions.append((sp, df, meta))
            print(f"  OK  {sp.name}  ({len(df)} samples, gas={meta.get('gas_label', '?')})")
        except Exception as exc:
            print(f"  SKIP {sp.name}: {exc}")

    if not sessions:
        print("No sessions loaded. Exiting.")
        sys.exit(1)

    print_summary_table(sessions, metric=args.metric)
    plot_comparison(sessions, metric=args.metric, output_path=args.output)


if __name__ == "__main__":
    main()

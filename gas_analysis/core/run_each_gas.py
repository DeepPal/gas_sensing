import argparse
from collections.abc import Iterable
import contextlib
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import sys
import traceback

from config.config_loader import load_config
from gas_analysis.core import pipeline as pl

BASE = Path("Joy_Data")
JOBS = {
    "Ethanol": (
        BASE / "Multi mix vary-EtOH",
        BASE / "ref MutiAuMIP-EtOH.csv",
        Path("output") / "ethanol_topavg",
    ),
    "Isopropanol": (
        BASE / "Multi mix vary-IPA",
        BASE / "ref AuMutiMIP-IPA.csv",
        Path("output") / "isopropanol_topavg",
    ),
    "Methanol": (
        BASE / "Multi mix vary-MeOH",
        BASE / "ref AuMutiMIP-MeOH.csv",
        Path("output") / "methanol_topavg",
    ),
    "MixVOC": (
        BASE / "Mixed gas",
        BASE / "ref AuMutiMIP-MIX.csv",
        Path("output") / "mixvoc_topavg",
    ),
}


def _configure_utf8_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        with contextlib.suppress(Exception):
            reconfigure(encoding="utf-8", errors="replace")


def _resolve_gases(gas_args: Iterable[str]) -> list[str]:
    if not gas_args:
        return list(JOBS.keys())
    selected: list[str] = []
    for entry in gas_args:
        # Allow comma-separated lists as a convenience
        tokens = [token.strip() for token in entry.split(",") if token.strip()]
        for token in tokens:
            if token not in JOBS:
                raise ValueError(
                    f"Unknown gas label '{token}'. Valid options: {', '.join(JOBS.keys())}"
                )
            if token not in selected:
                selected.append(token)
    return selected if selected else list(JOBS.keys())


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        with contextlib.suppress(Exception):
            f.flush()
        with contextlib.suppress(Exception):
            os.fsync(f.fileno())

    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except Exception:
        pass


def _run_status_path(out_dir: Path) -> Path:
    return out_dir / "metrics" / "run_status.json"


def run_job(
    label: str,
    data_dir: Path,
    ref_path: Path,
    out_dir: Path,
    *,
    config_path: Path | None,
    diff_threshold: float | None,
    avg_top_n: int | None,
    scan_full: bool,
    top_k: int,
) -> None:
    status_path = _run_status_path(out_dir)
    status_payload = {
        "state": "running",
        "label": label,
        "started_at": _now_iso(),
        "ended_at": None,
        "exit_code": None,
        "error": None,
        "traceback": None,
        "inputs": {
            "data_dir": str(data_dir.resolve()),
            "ref_path": str(ref_path.resolve()),
            "out_dir": str(out_dir.resolve()),
            "config_path": str(config_path) if config_path is not None else None,
            "diff_threshold": diff_threshold,
            "avg_top_n": avg_top_n,
            "scan_full": scan_full,
            "top_k": top_k,
        },
        "runtime": {
            "pid": os.getpid(),
            "ppid": getattr(os, "getppid", lambda: None)(),
            "python": sys.version,
            "argv": sys.argv,
        },
    }
    with contextlib.suppress(Exception):
        _write_json_atomic(status_path, status_payload)

    prev_sigint = signal.getsignal(signal.SIGINT)
    prev_sigterm = signal.getsignal(signal.SIGTERM)
    prev_sigbreak = getattr(signal, "SIGBREAK", None)
    if prev_sigbreak is not None:
        try:
            prev_sigbreak = signal.getsignal(signal.SIGBREAK)
        except Exception:
            prev_sigbreak = None

    def _handle_termination(signum, _frame):
        code = 130 if signum == signal.SIGINT else 143
        status_payload.update(
            {
                "state": "canceled",
                "ended_at": _now_iso(),
                "exit_code": code,
                "error": f"signal:{signum}",
                "traceback": "(terminated by signal)",
            }
        )
        with contextlib.suppress(Exception):
            _write_json_atomic(status_path, status_payload)
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, _handle_termination)
        signal.signal(signal.SIGTERM, _handle_termination)
        if prev_sigbreak is not None:
            signal.signal(signal.SIGBREAK, _handle_termination)
    except Exception:
        prev_sigint = None
        prev_sigterm = None
        prev_sigbreak = None

    config = load_config(str(config_path)) if config_path is not None else load_config()
    pl.CONFIG = config

    stability_cfg = (config.get("stability") or {}) if isinstance(config, dict) else {}
    diff_threshold_val = (
        diff_threshold
        if diff_threshold is not None
        else float(stability_cfg.get("diff_threshold", 0.01))
    )

    print(f"\n=== {label} ===")
    result = None
    try:
        result = pl.run_full_pipeline(
            root_dir=str(data_dir.resolve()),
            ref_path=str(ref_path.resolve()),
            out_root=str(out_dir.resolve()),
            diff_threshold=diff_threshold_val,
            avg_top_n=avg_top_n,
            scan_full=scan_full,
            top_k_candidates=top_k,
            dataset_label=label,
        )
    except KeyboardInterrupt:
        if status_payload.get("state") != "canceled":
            status_payload.update(
                {
                    "state": "canceled",
                    "ended_at": _now_iso(),
                    "exit_code": 130,
                    "error": "KeyboardInterrupt",
                    "traceback": traceback.format_exc(),
                }
            )
        else:
            if status_payload.get("ended_at") is None:
                status_payload["ended_at"] = _now_iso()
            if status_payload.get("exit_code") is None:
                status_payload["exit_code"] = 130

        with contextlib.suppress(Exception):
            _write_json_atomic(status_path, status_payload)
        raise
    except Exception as exc:  # noqa: BLE001
        status_payload.update(
            {
                "state": "failed",
                "ended_at": _now_iso(),
                "exit_code": 1,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        with contextlib.suppress(Exception):
            _write_json_atomic(status_path, status_payload)
        raise
    else:
        status_payload.update(
            {
                "state": "success",
                "ended_at": _now_iso(),
                "exit_code": 0,
                "outputs": {
                    "run_metadata": result.get("run_metadata")
                    if isinstance(result, dict)
                    else None,
                    "calibration_metrics": os.path.join(
                        str(out_dir.resolve()), "metrics", "calibration_metrics.json"
                    ),
                    "dynamics_summary": os.path.join(
                        str(out_dir.resolve()), "metrics", "dynamics_summary.json"
                    ),
                },
            }
        )
        with contextlib.suppress(Exception):
            _write_json_atomic(status_path, status_payload)
    finally:
        if prev_sigint is not None:
            with contextlib.suppress(Exception):
                signal.signal(signal.SIGINT, prev_sigint)
        if prev_sigterm is not None:
            with contextlib.suppress(Exception):
                signal.signal(signal.SIGTERM, prev_sigterm)
        if prev_sigbreak is not None:
            with contextlib.suppress(Exception):
                signal.signal(signal.SIGBREAK, prev_sigbreak)

    if ((config.get("roi", {}) or {}).get("shift", {}) or {}).get("minimal_outputs", False):
        print("  (minimal_outputs=True - diagnostic plots may be suppressed)")

    print(f"Outputs -> {out_dir}")
    print(
        "Top full-scan candidates (main):",
        result.get("fullscan_concentration_response_metrics"),
    )
    for mode, payload in (result.get("top_avg_results") or {}).items():
        print(f"Top full-scan candidates ({mode}): {payload.get('fullscan_metrics_path')}")


def main() -> None:
    _configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description="Run the gas analysis pipeline for one or more gases.",
    )
    parser.add_argument(
        "--data-dir",
        dest="data_dir",
        type=str,
        help="Override data directory for a single run (bypasses built-in gas presets).",
    )
    parser.add_argument(
        "--ref-path",
        dest="ref_path",
        type=str,
        help="Override reference spectrum path for a single run (requires --data-dir).",
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        type=str,
        help="Override output directory for a single run (requires --data-dir).",
    )
    parser.add_argument(
        "--label",
        dest="dataset_label",
        type=str,
        default=None,
        help="Optional dataset label to use in outputs when using --data-dir.",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        type=str,
        help="Optional YAML configuration to load instead of config/config.yaml.",
    )
    parser.add_argument(
        "--gas",
        dest="gases",
        action="append",
        help="Gas label to process (EtOH, IPA, MeOH, MIX). Can be repeated or comma-separated. Defaults to all.",
    )
    parser.add_argument(
        "--diff-threshold",
        type=float,
        default=None,
        help="Frame-to-frame normalised difference threshold for stability filtering (default: uses config value)",
    )
    parser.add_argument(
        "--avg-top-n",
        type=int,
        default=10,
        help="Average the first N frames per concentration (default: 10). Use 0 to disable averaging.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top ROI candidates to record (default: 10)",
    )
    parser.add_argument(
        "--scan-full",
        dest="scan_full",
        action="store_true",
        help="Enable full-spectrum candidate scan (default)",
    )
    parser.add_argument(
        "--no-scan-full",
        dest="scan_full",
        action="store_false",
        help="Disable the full-spectrum candidate scan",
    )
    parser.set_defaults(scan_full=True)

    args = parser.parse_args()

    config_path = Path(args.config_path).resolve() if args.config_path else None
    if config_path and not config_path.is_file():
        parser.error(f"Configuration file not found: {config_path}")

    avg_top_n = None if args.avg_top_n is None else max(0, args.avg_top_n)

    if args.data_dir or args.ref_path or args.out_dir:
        if not (args.data_dir and args.ref_path and args.out_dir):
            parser.error("--data-dir, --ref-path, and --out-dir must be provided together")
        data_dir = Path(args.data_dir).resolve()
        ref_path = Path(args.ref_path).resolve()
        out_dir = Path(args.out_dir).resolve()
        if not data_dir.is_dir():
            parser.error(f"Data directory not found: {data_dir}")
        if not ref_path.is_file():
            parser.error(f"Reference file not found: {ref_path}")
        out_dir.mkdir(parents=True, exist_ok=True)

        label = args.dataset_label or data_dir.name
        try:
            run_job(
                label,
                data_dir,
                ref_path,
                out_dir,
                config_path=config_path,
                diff_threshold=args.diff_threshold,
                avg_top_n=avg_top_n if (avg_top_n is not None and avg_top_n > 0) else None,
                scan_full=args.scan_full,
                top_k=args.top_k,
            )
        except KeyboardInterrupt:
            sys.exit(130)
        return

    try:
        selected_gases = _resolve_gases(args.gases)
    except ValueError as exc:
        parser.error(str(exc))

    for label in selected_gases:
        data_dir, ref_path, out_dir = JOBS[label]
        try:
            run_job(
                label,
                data_dir,
                ref_path,
                out_dir,
                config_path=config_path,
                diff_threshold=args.diff_threshold,
                avg_top_n=avg_top_n if avg_top_n > 0 else None,
                scan_full=args.scan_full,
                top_k=args.top_k,
            )
        except KeyboardInterrupt:
            sys.exit(130)


if __name__ == "__main__":
    main()

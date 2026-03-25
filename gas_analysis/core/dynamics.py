import contextlib
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .pipeline import _read_csv_spectrum


def compute_response_recovery_times(
    root_dir: str,
    out_root: str,
    signal_column: str = "intensity",
    exposure_start_time: float = None,
    purge_start_time: float = None,
    saturation_fraction: float = 0.9,
    baseline_fraction: float = 0.1,
) -> dict[str, object]:
    """Compute response (T90) and recovery (T10) times from time-series CSVs.

    Expected structure:
      root/<concentration>/<trial>/frame_*.csv
    Each frame is a time-series: time,intensity.

    Args:
        root_dir: Experiment root with concentration/trial/frame structure.
        out_root: Output directory for results.
        signal_column: Column to analyze ('intensity' or 'transmittance').
        exposure_start_time: Time when gas exposure begins (default: first timestamp).
        purge_start_time: Time when purge begins (default: midpoint).
        saturation_fraction: Fraction of max signal to define T90.
        baseline_fraction: Fraction of max signal to define T10.

    Returns:
        Dict with keys: 'results' (DataFrame), 'plot_path'.
    """
    results = []

    # Handle both layouts:
    # 1. root/<concentration>/<trial>/frame_*.csv
    # 2. root/<concentration>/frame_*.csv
    if not os.path.isdir(root_dir):
        raise ValueError(f"Not a directory: {root_dir}")

    conc_dirs = [
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]
    if not conc_dirs:
        raise ValueError(f"No concentration directories found in: {root_dir}")

    for conc_dir in conc_dirs:
        conc_name = os.path.basename(conc_dir)
        try:
            conc_val = float("".join(filter(str.isdigit, conc_name))) or 0.0
        except (ValueError, TypeError, AttributeError):
            conc_val = 0.0

        # Check for direct frames or trial subfolders
        direct_frames = [
            os.path.join(conc_dir, f) for f in os.listdir(conc_dir) if f.lower().endswith(".csv")
        ]
        trial_dirs = [
            os.path.join(conc_dir, d)
            for d in os.listdir(conc_dir)
            if os.path.isdir(os.path.join(conc_dir, d))
        ]

        all_trial_frames = []
        if direct_frames:
            all_trial_frames.append(("direct", direct_frames))
        for trial_dir in trial_dirs:
            trial_name = os.path.basename(trial_dir)
            trial_frames = [
                os.path.join(trial_dir, f)
                for f in os.listdir(trial_dir)
                if f.lower().endswith(".csv")
            ]
            if trial_frames:
                all_trial_frames.append((trial_name, trial_frames))

        for trial_name, frame_paths in all_trial_frames:
            dfs = []
            for path in sorted(frame_paths, key=lambda p: os.path.getmtime(p)):
                df = _read_csv_spectrum(path)
                if "time" not in df.columns:
                    df = df.rename(columns={"wavelength": "time"})
                if "time" in df.columns and signal_column in df.columns:
                    dfs.append(df)
            if not dfs:
                continue
            trial_df = pd.concat(dfs, ignore_index=True).sort_values("time")
            t = trial_df["time"].values
            y = trial_df[signal_column].values

            if len(y) < 2:
                continue

            # Normalize signal to [0,1]
            y_min = y.min()
            y_max = y.max()
            if y_max - y_min < 1e-9:
                continue
            y_norm = (y - y_min) / (y_max - y_min)

            # Define time windows
            t_start = exposure_start_time if exposure_start_time is not None else t[0]
            t_purge = purge_start_time if purge_start_time is not None else 0.5 * (t[0] + t[-1])

            # Response time (T90): time to reach 90% of max after exposure
            y90 = saturation_fraction
            idx_exposure = np.where(t >= t_start)[0]
            if len(idx_exposure) == 0:
                continue
            idx_sat = np.where(y_norm[idx_exposure] >= y90)[0]
            t90 = float(t[idx_exposure[idx_sat[0]]]) if len(idx_sat) > 0 else float("nan")

            # Recovery time (T10): time to drop to 10% after purge
            y10 = baseline_fraction
            idx_purge = np.where(t >= t_purge)[0]
            if len(idx_purge) == 0:
                continue
            idx_rec = np.where(y_norm[idx_purge] <= y10)[0]
            t10 = float(t[idx_purge[idx_rec[0]]]) if len(idx_rec) > 0 else float("nan")

            results.append(
                {
                    "concentration": conc_val,
                    "trial": trial_name,
                    "response_time_T90": t90,
                    "recovery_time_T10": t10,
                }
            )

    df_results = pd.DataFrame(results)
    out_dir = os.path.join(out_root, "dynamics")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "response_recovery.csv")
    df_results.to_csv(csv_path, index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    for conc in df_results["concentration"].unique():
        sub = df_results[df_results["concentration"] == conc]
        ax.scatter(sub["response_time_T90"], sub["recovery_time_T10"], label=f"{conc} ppm")
    ax.set_xlabel("Response Time T90 (s)")
    ax.set_ylabel("Recovery Time T10 (s)")
    ax.set_title("Response vs Recovery Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "response_recovery.png")
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    return {
        "results": df_results,
        "plot_path": plot_path,
    }


def compute_t90_t10_from_timeseries(
    time_series_dir: str,
    out_root: str,
    baseline_frames: int = 20,
    steady_state_frames: int = 20,
    frame_rate: float = None,
    min_response_amplitude_nm: float = 0.0,
    smooth_window: int = 1,
) -> dict[str, object]:
    """
    Compute T90 (response time) and T10 (recovery time) from time-series CSV files.

    T90: Time to reach 90% of steady-state response (from baseline)
    T10: Time to decay from peak to 10% of response (recovery)

    Args:
        time_series_dir: Directory containing *_timeseries.csv files
        out_root: Output directory for results
        baseline_frames: Number of initial frames to average for baseline
        steady_state_frames: Number of final frames to average for steady-state
        frame_rate: Frames per second (default: None for auto-detection)

    Returns:
        Dict with keys: 'summary' (dict), 'results' (DataFrame), 'json_path', 'csv_path'
    """
    if not os.path.isdir(time_series_dir):
        raise ValueError(f"Not a directory: {time_series_dir}")

    csv_files = [f for f in os.listdir(time_series_dir) if f.endswith(".csv")]
    if not csv_files:
        print(f"[WARNING] No CSV files found in {time_series_dir}")
        return {
            "summary": {},
            "results": pd.DataFrame(),
            "json_path": None,
            "csv_path": None,
        }

    results = []

    for csv_file in sorted(csv_files):
        csv_path = os.path.join(time_series_dir, csv_file)

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARNING] Failed to read {csv_file}: {e}")
            continue

        if "delta_lambda_nm" not in df.columns:
            print(f"[WARNING] No 'delta_lambda_nm' column in {csv_file}")
            continue

        # Extract metadata from filename (e.g., "Acetone_50_50ppm_t1.csv")
        parts = csv_file.replace(".csv", "").split("_")
        gas_name = parts[0] if len(parts) > 0 else "unknown"
        concentration = 0.0
        trial = "unknown"

        # Try to extract concentration (look for pattern like "50ppm")
        for part in parts:
            if "ppm" in part.lower():
                with contextlib.suppress(ValueError):
                    concentration = float(part.lower().replace("ppm", ""))

        # Extract trial (last part, e.g., "t1", "T1")
        if len(parts) > 0:
            trial = parts[-1]

        delta_raw = df["delta_lambda_nm"].values
        delta = np.asarray(delta_raw, dtype=float)

        # Optional temporal smoothing to suppress frame-to-frame jitter
        if smooth_window is not None:
            try:
                win = int(smooth_window)
            except Exception:
                win = 1
            if win > 1:
                kernel = np.ones(win, dtype=float) / float(win)
                delta = np.convolve(delta, kernel, mode="same")

        if len(delta) < baseline_frames + steady_state_frames:
            print(
                f"[WARNING] Not enough frames in {csv_file} (need {baseline_frames + steady_state_frames})"
            )
            continue

        # Auto-detect frame rate if not provided
        if frame_rate is None:
            # Method 1: Use responsive frames if available
            if "is_responsive" in df.columns:
                responsive_mask = df["is_responsive"] == 1
                if responsive_mask.sum() > 0:
                    # Assume responsive window is ~5 minutes (300s) for gas sensing
                    responsive_frame_count = responsive_mask.sum()
                    estimated_frame_rate = responsive_frame_count / 300.0
                else:
                    # Method 2: Estimate from total frames (assume 10-minute experiment)
                    estimated_frame_rate = len(delta) / 600.0
            else:
                # Method 2: Estimate from total frames (assume 10-minute experiment)
                estimated_frame_rate = len(delta) / 600.0

            # Use estimated rate for this file
            current_frame_rate = max(
                0.1, min(10.0, estimated_frame_rate)
            )  # Clamp to reasonable range
            print(f"[INFO] Auto-detected frame rate: {current_frame_rate:.3f} fps for {csv_file}")
        else:
            current_frame_rate = frame_rate

        # Compute baseline and steady-state
        baseline = np.nanmean(delta[:baseline_frames])
        steady_state = np.nanmean(delta[-steady_state_frames:])

        # Response amplitude
        response_amplitude = abs(steady_state - baseline)

        # Gate out very weak responses from T90/T10 statistics
        try:
            amp_thresh = (
                float(min_response_amplitude_nm) if min_response_amplitude_nm is not None else 0.0
            )
        except Exception:
            amp_thresh = 0.0
        amp_thresh = max(amp_thresh, 1e-6)

        if response_amplitude < amp_thresh:
            print(
                f"[WARNING] Response amplitude {response_amplitude:.4g} nm below threshold {amp_thresh:.4g} nm in {csv_file}"
            )
            t90 = float("nan")
            t10 = float("nan")
        else:
            # T90: Time to reach 90% of response
            threshold_90 = baseline + 0.9 * (steady_state - baseline)

            # Find first frame exceeding threshold
            if steady_state > baseline:
                idx_90 = np.where(delta >= threshold_90)[0]
            else:
                idx_90 = np.where(delta <= threshold_90)[0]

            if len(idx_90) > 0:
                t90_frame = idx_90[0]
                t90 = t90_frame / current_frame_rate
            else:
                t90 = float("nan")

            # T10: Time to decay to 10% of response (recovery)
            # Find peak (max absolute deviation from baseline)
            peak_idx = np.argmax(np.abs(delta - baseline))

            if peak_idx < len(delta) - 1:
                # Look for decay after peak
                delta_after_peak = delta[peak_idx:]
                threshold_10 = baseline + 0.1 * (steady_state - baseline)

                if steady_state > baseline:
                    idx_10 = np.where(delta_after_peak <= threshold_10)[0]
                else:
                    idx_10 = np.where(delta_after_peak >= threshold_10)[0]

                if len(idx_10) > 0:
                    t10_frame = peak_idx + idx_10[0]
                    t10 = t10_frame / current_frame_rate
                else:
                    t10 = float("nan")
            else:
                t10 = float("nan")

        # Δλ(t) plot for this time-series
        try:
            time_axis = np.arange(len(delta), dtype=float) / max(current_frame_rate, 1e-6)
            plots_dir = os.path.join(out_root, "plots", "dynamics")
            os.makedirs(plots_dir, exist_ok=True)
            plot_name = csv_file.replace(".csv", "_delta_lambda_timeseries.png")
            plot_path = os.path.join(plots_dir, plot_name)

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(time_axis, delta, color="tab:blue", linewidth=1.4, label="Δλ(t)")
            ax.axhline(baseline, color="gray", linestyle="--", linewidth=0.9, label="baseline")
            ax.axhline(
                steady_state,
                color="tab:green",
                linestyle="--",
                linewidth=0.9,
                label="steady-state",
            )

            if not np.isnan(t90):
                ax.axvline(t90, color="tab:red", linestyle="--", linewidth=0.9, label="T90")
                ax.axvspan(0.0, t90, color="tab:red", alpha=0.06)
            if not np.isnan(t10) and not np.isnan(t90):
                ax.axvline(t10, color="tab:orange", linestyle="--", linewidth=0.9, label="T10")
                ax.axvspan(t90, t10, color="tab:orange", alpha=0.06)

            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Δλ (nm)")
            ax.set_title(f"Δλ(t): {gas_name} {concentration:g} ppm ({trial})")
            ax.grid(True, alpha=0.3)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(plot_path, dpi=200)
            plt.close(fig)
        except Exception:
            with contextlib.suppress(Exception):
                plt.close(fig)

        results.append(
            {
                "gas": gas_name,
                "concentration_ppm": concentration,
                "trial": trial,
                "filename": csv_file,
                "baseline_nm": baseline,
                "steady_state_nm": steady_state,
                "response_amplitude_nm": response_amplitude,
                "T90_response_s": t90,
                "T10_recovery_s": t10,
                "total_frames": len(delta),
            }
        )

    df_results = pd.DataFrame(results)

    # Create output directory
    out_dir = os.path.join(out_root, "metrics")
    os.makedirs(out_dir, exist_ok=True)

    # Save CSV
    csv_out_path = os.path.join(out_dir, "dynamics_summary.csv")
    df_results.to_csv(csv_out_path, index=False)

    # Compute summary statistics
    summary = {}
    if len(df_results) > 0:
        valid_t90 = df_results["T90_response_s"].dropna()
        valid_t10 = df_results["T10_recovery_s"].dropna()

        summary = {
            "total_trials": len(df_results),
            "T90_mean_s": float(valid_t90.mean()) if len(valid_t90) > 0 else None,
            "T90_std_s": float(valid_t90.std()) if len(valid_t90) > 0 else None,
            "T90_median_s": float(valid_t90.median()) if len(valid_t90) > 0 else None,
            "T90_min_s": float(valid_t90.min()) if len(valid_t90) > 0 else None,
            "T90_max_s": float(valid_t90.max()) if len(valid_t90) > 0 else None,
            "T10_mean_s": float(valid_t10.mean()) if len(valid_t10) > 0 else None,
            "T10_std_s": float(valid_t10.std()) if len(valid_t10) > 0 else None,
            "T10_median_s": float(valid_t10.median()) if len(valid_t10) > 0 else None,
            "T10_min_s": float(valid_t10.min()) if len(valid_t10) > 0 else None,
            "T10_max_s": float(valid_t10.max()) if len(valid_t10) > 0 else None,
            "response_amplitude_mean_nm": float(df_results["response_amplitude_nm"].mean()),
            "response_amplitude_std_nm": float(df_results["response_amplitude_nm"].std()),
        }

    # Save JSON summary
    json_out_path = os.path.join(out_dir, "dynamics_summary.json")
    with open(json_out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Dynamics summary saved to {json_out_path}")
    print(
        f"[INFO] T90 (response): {summary.get('T90_mean_s', 'N/A'):.2f} ± {summary.get('T90_std_s', 0):.2f} s"
    )
    print(
        f"[INFO] T10 (recovery): {summary.get('T10_mean_s', 'N/A'):.2f} ± {summary.get('T10_std_s', 0):.2f} s"
    )

    return {
        "summary": summary,
        "results": df_results,
        "json_path": json_out_path,
        "csv_path": csv_out_path,
    }

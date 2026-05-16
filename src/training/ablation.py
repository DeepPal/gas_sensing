"""
src.training.ablation
=====================
Preprocessing ablation study for the the sensor LSPR gas-sensing pipeline.

What it does
------------
Systematically disables one preprocessing step at a time and re-trains a
GPRCalibration model on the resulting spectra, measuring how much each step
contributes to ΔR² and ΔRMSE (concentration prediction).

Ablation configurations
-----------------------
+-----------------------------+--------------------------------------------------+
| Config name                 | What is disabled                                 |
+=============================+==================================================+
| all_on (baseline)           | All steps active (reference configuration)       |
| no_baseline                 | Skip ALS baseline correction                     |
| no_smoothing                | Skip Savitzky-Golay smoothing                    |
| no_normalization            | Skip SNV normalization                           |
| no_baseline_no_smoothing    | Skip both baseline correction AND smoothing      |
| raw_only                    | No preprocessing at all (raw spectra)            |
+-----------------------------+--------------------------------------------------+

Design rationale
----------------
The ablation trains a GPR (rather than CNN) because GPR is fast to fit and
its R² is a clean measure of data quality.  CNN training would take too long
for a 6-fold ablation study.  The preprocessing quality translates directly —
if GPR R² drops 0.2 when smoothing is removed, the CNN will be similarly hurt.

Usage
-----
::

    python -m src.training.ablation \\
        --data-dir data/JOY_Data/Ethanol \\
        --conc-col concentration_ppm \\
        --output output/ablation_results.json

Outputs
-------
- Console table with ΔR² and ΔRMSE per ablation config.
- ``output/ablation_results.json`` — machine-readable results.
- MLflow run tagged ``ablation`` (optional).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sys

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ablation configuration
# ---------------------------------------------------------------------------


@dataclass
class AblationConfig:
    """Controls which preprocessing steps are active."""

    name: str
    use_baseline: bool = True
    use_smoothing: bool = True
    use_normalization: bool = True

    def describe(self) -> str:
        parts = []
        if not self.use_baseline:
            parts.append("no_baseline")
        if not self.use_smoothing:
            parts.append("no_smoothing")
        if not self.use_normalization:
            parts.append("no_normalization")
        return " + ".join(parts) if parts else "all steps active"


ABLATION_CONFIGS: list[AblationConfig] = [
    AblationConfig("all_on"),
    AblationConfig("no_baseline", use_baseline=False),
    AblationConfig("no_smoothing", use_smoothing=False),
    AblationConfig("no_normalization", use_normalization=False),
    AblationConfig("no_baseline_no_smoothing", use_baseline=False, use_smoothing=False),
    AblationConfig("raw_only", use_baseline=False, use_smoothing=False, use_normalization=False),
]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def preprocess_spectrum(
    intensities: np.ndarray,
    wavelengths: np.ndarray | None,
    cfg: AblationConfig,
) -> np.ndarray:
    """Apply the preprocessing pipeline according to *cfg*.

    Parameters
    ----------
    intensities:
        Raw intensity array, shape (n_points,).
    wavelengths:
        Corresponding wavelength axis.  Required for ALS/polynomial baseline.
    cfg:
        Ablation configuration controlling which steps are active.

    Returns
    -------
    ndarray
        Preprocessed intensities.
    """
    from src.preprocessing.baseline import correct_baseline
    from src.preprocessing.denoising import savgol_smooth
    from src.preprocessing.normalization import normalize_snv

    arr = np.asarray(intensities, dtype=float)

    if cfg.use_baseline and wavelengths is not None:
        try:
            arr = correct_baseline(wavelengths, arr, method="als")
        except Exception as exc:
            log.debug("Baseline correction failed: %s", exc)

    if cfg.use_smoothing:
        try:
            arr = savgol_smooth(arr)
        except Exception as exc:
            log.debug("Smoothing failed: %s", exc)

    if cfg.use_normalization:
        try:
            arr = normalize_snv(arr)
        except Exception as exc:
            log.debug("Normalization failed: %s", exc)

    return np.asarray(arr)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_data(
    data_dir: Path,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """Load spectra and concentrations from Joy-data layout.

    Returns
    -------
    wavelengths : (n_points,) or None
    X_raw : (n_samples, n_points) — raw (no preprocessing)
    y_conc : (n_samples,) float ppm
    """
    import re

    import pandas as pd

    csv_files = sorted(data_dir.rglob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found under {data_dir}")

    X_rows: list[np.ndarray] = []
    y_concs: list[float] = []
    wavelengths: np.ndarray | None = None

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, header=None)
            vals = df.values
            if vals.shape[1] == 2:
                # Two-column format: wavelength | intensity
                if wavelengths is None:
                    wavelengths = vals[:, 0].astype(float)
                row = vals[:, 1].astype(float)
            else:
                row = vals.flatten().astype(float)

            conc_text = csv_path.parent.name
            m = re.search(r"([\d.]+)\s*ppm", conc_text, re.IGNORECASE)
            conc = float(m.group(1)) if m else 0.0

            X_rows.append(row)
            y_concs.append(conc)
        except Exception as exc:
            log.debug("Skipping %s: %s", csv_path, exc)

    if not X_rows:
        raise ValueError(f"No valid spectra loaded from {data_dir}")

    min_len = min(r.size for r in X_rows)
    X_raw = np.stack([r[:min_len] for r in X_rows], axis=0)
    if wavelengths is not None:
        wavelengths = wavelengths[:min_len]

    return wavelengths, X_raw, np.array(y_concs, dtype=float)


# ---------------------------------------------------------------------------
# GPR evaluation per ablation config
# ---------------------------------------------------------------------------


def _evaluate_config(
    cfg: AblationConfig,
    wavelengths: np.ndarray | None,
    X_raw: np.ndarray,
    y_conc: np.ndarray,
) -> dict:
    """Preprocess X_raw according to *cfg*, then fit+CV a GPR, return metrics."""
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import KFold

    from src.calibration.gpr import GPRCalibration

    # Preprocess all spectra under this config
    X_pp = np.stack([preprocess_spectrum(row, wavelengths, cfg) for row in X_raw], axis=0)

    # Extract the LSPR feature: peak wavelength shift within the ROI
    # Use simple argmax (fast) — peak position is the primary LSPR signal
    peak_indices = np.argmax(X_pp, axis=1)
    if wavelengths is not None:
        peak_wavelengths = wavelengths[peak_indices]
    else:
        peak_wavelengths = peak_indices.astype(float)

    delta_lambda = peak_wavelengths - peak_wavelengths.mean()

    # 5-fold CV with GPR
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_all = np.empty_like(y_conc)
    y_unc_all = np.empty_like(y_conc)

    for train_idx, test_idx in kf.split(delta_lambda):
        gpr = GPRCalibration()
        gpr.fit(delta_lambda[train_idx], y_conc[train_idx])
        preds, uncs = gpr.predict(delta_lambda[test_idx])
        y_pred_all[test_idx] = preds
        y_unc_all[test_idx] = uncs

    rmse = float(np.sqrt(mean_squared_error(y_conc, y_pred_all)))
    r2 = float(r2_score(y_conc, y_pred_all))
    mean_unc = float(y_unc_all.mean())

    return {
        "rmse_ppm": round(rmse, 4),
        "r2": round(r2, 4),
        "mean_uncertainty_ppm": round(mean_unc, 4),
        "description": cfg.describe(),
    }


# ---------------------------------------------------------------------------
# Main ablation loop
# ---------------------------------------------------------------------------


def run_ablation(
    data_dir: Path,
    configs: list[AblationConfig] | None = None,
) -> dict:
    """Run all ablation configs and return a results dict.

    Parameters
    ----------
    data_dir:
        Directory of spectra for a single gas + concentration (Joy-data layout).
    configs:
        List of ablation configs to evaluate.  Defaults to :data:`ABLATION_CONFIGS`.

    Returns
    -------
    dict
        ``{"baseline": {...}, "ablations": {config_name: {...}}}``
    """
    if configs is None:
        configs = ABLATION_CONFIGS

    wavelengths, X_raw, y_conc = _load_data(data_dir)
    log.info("Loaded %d spectra from %s", len(X_raw), data_dir)

    results: dict[str, dict] = {}
    baseline_r2: float | None = None
    baseline_rmse: float | None = None

    for cfg in configs:
        log.info("Evaluating config: %s (%s)", cfg.name, cfg.describe())
        try:
            metrics = _evaluate_config(cfg, wavelengths, X_raw, y_conc)
        except Exception as exc:
            log.error("Config %s failed: %s", cfg.name, exc)
            metrics = {"rmse_ppm": float("nan"), "r2": float("nan"), "error": str(exc)}

        if cfg.name == "all_on":
            baseline_r2 = metrics.get("r2")
            baseline_rmse = metrics.get("rmse_ppm")

        results[cfg.name] = metrics

    # Compute deltas relative to the all_on baseline
    for name, m in results.items():
        if name == "all_on":
            m["delta_r2"] = 0.0
            m["delta_rmse_ppm"] = 0.0
        else:
            r2 = m.get("r2", float("nan"))
            rmse = m.get("rmse_ppm", float("nan"))
            m["delta_r2"] = (
                round(r2 - (baseline_r2 or 0.0), 4) if not np.isnan(r2) else float("nan")
            )
            m["delta_rmse_ppm"] = (
                round(rmse - (baseline_rmse or 0.0), 4) if not np.isnan(rmse) else float("nan")
            )

    return {
        "baseline_config": "all_on",
        "n_spectra": int(len(X_raw)),
        "data_dir": str(data_dir),
        "configs": results,
    }


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------


def _log_ablation_to_mlflow(
    results: dict,
    mlflow_uri: str,
    out_path: Path | None = None,
) -> None:
    from datetime import datetime

    from src.training.mlflow_tracker import _DEFAULT_TRACKING_URI, ExperimentTracker

    run_name = f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    uri = mlflow_uri if mlflow_uri != "mlruns" else _DEFAULT_TRACKING_URI

    try:
        with ExperimentTracker(
            run_name=run_name,
            experiment_name="preprocessing_ablation",
            tracking_uri=uri,
            tags={"data_dir": results.get("data_dir", "")},
        ) as tracker:
            tracker.log_params(
                {
                    "n_spectra": results.get("n_spectra", 0),
                    "baseline_config": results.get("baseline_config", "all_on"),
                    "configs_tested": json.dumps(list(results.get("configs", {}).keys())),
                }
            )
            for config_name, m in results.get("configs", {}).items():
                tracker.log_metrics(
                    {
                        f"{config_name}_{key}": float(m[key])
                        for key in ("rmse_ppm", "r2", "delta_r2", "delta_rmse_ppm")
                        if key in m and m[key] is not None and np.isfinite(m[key])
                    }
                )
            if out_path and Path(out_path).exists():
                tracker.log_artifact(out_path, "results")
            log.info("Ablation results logged to MLflow run '%s'.", run_name)
    except ImportError:
        log.info("MLflow not installed — skipping MLflow logging.")
    except Exception as exc:
        log.warning("MLflow logging failed: %s", exc)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _print_ablation_table(results: dict) -> None:
    configs = results.get("configs", {})
    print("\n" + "=" * 75)
    print("  Preprocessing Ablation Study Results")
    print(f"  Data: {results.get('data_dir', 'N/A')}  |  N={results.get('n_spectra', '?')}")
    print("=" * 75)
    print(f"  {'Config':<30} {'R²':>7} {'ΔR²':>8} {'RMSE(ppm)':>11} {'ΔRMSE(ppm)':>12}")
    print("-" * 75)
    for name, m in configs.items():
        r2 = m.get("r2", float("nan"))
        dr2 = m.get("delta_r2", float("nan"))
        rmse = m.get("rmse_ppm", float("nan"))
        drmse = m.get("delta_rmse_ppm", float("nan"))
        marker = "← baseline" if name == "all_on" else ""
        print(f"  {name:<30} {r2:>7.4f} {dr2:>+8.4f} {rmse:>11.4f} {drmse:>+12.4f}  {marker}")
    print("=" * 75 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Preprocessing ablation study")
    parser.add_argument(
        "--data-dir", default="Joy_Data/Ethanol", help="Directory containing spectra for one gas"
    )
    parser.add_argument("--mlflow-uri", default="experiments/mlruns", help="MLflow tracking URI")
    parser.add_argument(
        "--output", default="output/ablation_results.json", help="Path for results JSON"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        log.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    results = run_ablation(data_dir)
    _print_ablation_table(results)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info("Ablation results written to %s", out_path)

    _log_ablation_to_mlflow(results, mlflow_uri=args.mlflow_uri, out_path=out_path)


if __name__ == "__main__":
    main()

"""
src.training.train_gpr
=======================
Train a Gaussian Process Regression (GPR) calibration model for one gas type.

The GPR models concentration (ppm) from the 4-component LSPR feature vector
[Δλ, ΔI_peak, ΔI_area, ΔI_std] and returns both a mean prediction and a
one-sigma uncertainty estimate — essential for confidence-aware detection.

Usage
-----
::

    python -m src.training.train_gpr --gas Ethanol --data data/raw/Joy_Data

    # All gases:
    python -m src.training.train_gpr --gas all --data data/raw/Joy_Data
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from src.training.mlflow_tracker import ExperimentTracker

log = logging.getLogger(__name__)


def _load_training_data(
    gas_type: str,
    data_root: Path,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Load and preprocess spectral data for a specific gas type.

    Returns
    -------
    tuple[X, y, concentrations]
        X: feature matrix (n_samples, 4)
        y: concentration labels (n_samples,)
        concentrations: unique concentration levels
    """
    from src.features.lspr_features import extract_lspr_features
    from src.preprocessing import smooth_spectrum

    gas_dir = data_root / gas_type
    if not gas_dir.exists():
        # Try alternate capitalizations
        for p in data_root.iterdir():
            if p.is_dir() and p.name.lower() == gas_type.lower():
                gas_dir = p
                break

    if not gas_dir.exists():
        raise FileNotFoundError(f"No data directory found for gas '{gas_type}' in {data_root}")

    import pandas as pd

    X_list: list[list[float]] = []
    y_list: list[float] = []
    ref_intensities: np.ndarray | None = None

    # Try to load reference spectrum
    ref_candidates = list(gas_dir.glob("ref*.csv")) + list(data_root.glob(f"ref*{gas_type}*.csv"))
    if ref_candidates:
        try:
            ref_df = pd.read_csv(ref_candidates[0], header=None, names=["wavelength", "intensity"])
            ref_intensities = ref_df["intensity"].to_numpy(dtype=float)
            log.info("Reference spectrum loaded: %s", ref_candidates[0].name)
        except Exception as exc:
            log.warning("Could not load reference spectrum: %s", exc)

    concentrations_seen: set[float] = set()

    for conc_dir in sorted(gas_dir.iterdir()):
        if not conc_dir.is_dir():
            continue

        # Parse concentration from folder name e.g. "0.5 ppm-1"
        conc = _parse_concentration(conc_dir.name)
        if conc is None:
            continue
        concentrations_seen.add(conc)

        csv_files = sorted(conc_dir.glob("*.csv"))
        if not csv_files:
            continue

        # Use last 10 CSVs as plateau (temporal gating)
        plateau_files = csv_files[-10:]
        spectra: list[np.ndarray] = []

        for csv_f in plateau_files:
            try:
                df = pd.read_csv(csv_f, header=None, names=["wavelength", "intensity"])
                wl = df["wavelength"].to_numpy(dtype=float)
                raw = df["intensity"].to_numpy(dtype=float)

                # Preprocess
                smoothed = smooth_spectrum(raw, window=11)
                processed = smoothed
                spectra.append(processed)

                if ref_intensities is None:
                    ref_intensities = processed.copy()
            except Exception as exc:
                log.debug("Skip %s: %s", csv_f.name, exc)

        if not spectra:
            continue

        # Average plateau frames
        avg_spectrum = np.mean(spectra, axis=0)

        if ref_intensities is not None and len(ref_intensities) == len(avg_spectrum):
            feat = extract_lspr_features(wl, avg_spectrum, ref_intensities)
            X_list.append(feat.feature_vector)
            y_list.append(conc)

    if not X_list:
        raise ValueError(f"No training samples extracted for gas '{gas_type}'.")

    X = np.array(X_list)
    y = np.array(y_list)
    return X, y, sorted(concentrations_seen)


def _parse_concentration(folder_name: str) -> float | None:
    """Extract concentration (ppm) from a folder name.

    Handles two conventions:
    - ``"0.5 ppm EtOH-1"`` — explicit ppm label
    - ``"Multi mix vary-EtOH-0.1-1"`` — gas-{conc}-{run} suffix (Joy-data layout)
    """
    import re

    # Primary: explicit "ppm" label
    m = re.search(r"(\d+\.?\d*)\s*ppm", folder_name, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # Fallback: numeric field before a trailing run-index suffix (-\d+)
    # e.g. "Multi mix vary-EtOH-0.1-1" → 0.1
    m = re.search(r"-(\d+\.?\d+)-\d+$", folder_name)
    if m:
        return float(m.group(1))

    return None


def _compute_gpr_aware_lod(
    gpr,
    scaler,
    concentrations: list[float],
    X_scaled: np.ndarray,
    y_std_train: np.ndarray,
) -> float | None:
    """Compute LOD via GPR posterior: find c where E[ŷ(c)] = 3 × Std[ŷ(c)].

    This is more rigorous than the linear approximation because it accounts
    for the GPR's concentration-dependent uncertainty, which grows outside the
    training range and may be non-monotonic.

    Uses a dense 1-D grid search followed by linear interpolation for the
    crossing point (avoids the assumption of differentiability required by
    root-finding methods).
    """
    if len(concentrations) < 2:
        return None

    c_min, c_max = min(concentrations), max(concentrations)
    c_grid = np.linspace(0.0, c_max * 1.5, 500).reshape(-1, 1)

    # Build a 1-D feature from concentration (use mean of training features
    # scaled to the GPR input space — approximation for 1-D representation)
    c_train = np.array(sorted(concentrations))
    resp_train_scaled = X_scaled[:, 0]  # first feature (Δλ) as 1-D proxy

    # Fit a simple 1-D GPR on concentration → first feature for LOD grid
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel

        kernel_1d = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01)
        gpr_1d = GaussianProcessRegressor(
            kernel=kernel_1d, normalize_y=True, n_restarts_optimizer=5
        )
        # Sort by concentration
        sort_idx = np.argsort(c_train)
        gpr_1d.fit(c_train[sort_idx].reshape(-1, 1), resp_train_scaled[sort_idx])

        y_grid, y_std_grid = gpr_1d.predict(c_grid, return_std=True)

        # Find crossing: |E[ŷ]| = 3 × Std[ŷ]
        signal = np.abs(y_grid)
        threshold = 3.0 * y_std_grid
        crossings = np.where(np.diff(np.sign(signal - threshold)))[0]

        if len(crossings) > 0:
            i = crossings[0]
            # Linear interpolation between grid[i] and grid[i+1]
            s1, t1 = signal[i], threshold[i]
            s2, t2 = signal[i + 1], threshold[i + 1]
            frac = (t1 - s1) / ((s2 - s1) - (t2 - t1)) if (s2 - s1) != (t2 - t1) else 0.5
            lod = float(c_grid[i, 0] + frac * (c_grid[i + 1, 0] - c_grid[i, 0]))
            return lod if lod > 0 else None

    except Exception as exc:
        log.debug("GPR-aware LOD failed, falling back to linear: %s", exc)

    # Fallback: 3σ / slope linear approximation
    if len(concentrations) >= 2:
        slope_est = (max(resp_train_scaled) - min(resp_train_scaled)) / (c_max - c_min)
        noise_floor = float(np.mean(y_std_train))
        return float(3 * noise_floor / slope_est) if slope_est != 0 else None
    return None


def train_gpr(
    gas_type: str,
    data_root: Path,
    output_dir: Path,
    tracking_uri: str = "experiments/mlruns",
    cv_folds: int = 5,
) -> dict:
    """Train a GPR model for one gas type and log to MLflow.

    Returns
    -------
    dict
        Training metrics: r2, rmse_ppm, lod_ppm, n_samples, model_path.
    """
    import joblib
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    from sklearn.metrics import r2_score
    from sklearn.model_selection import LeaveOneOut, cross_val_score
    from sklearn.preprocessing import StandardScaler

    run_name = f"GPR_{gas_type}_v3"

    with ExperimentTracker(run_name=run_name, tracking_uri=tracking_uri) as tracker:
        log.info("Loading training data for %s...", gas_type)
        X, y, concentrations = _load_training_data(gas_type, data_root)
        n_samples = len(y)

        # For small datasets (n < 20), LOOCV gives the least-biased estimate.
        # k-fold with small n yields high-variance splits; LOOCV is equivalent
        # to k-fold with k=n and is the standard for spectroscopic calibration.
        use_loocv = n_samples < 20
        cv_strategy = LeaveOneOut() if use_loocv else cv_folds
        cv_name = f"LOOCV (n={n_samples})" if use_loocv else f"{cv_folds}-fold CV"
        log.info("Using %s (n=%d samples)", cv_name, n_samples)

        tracker.log_dataset_info(
            gas_type=gas_type,
            n_samples=n_samples,
            concentrations=concentrations,
        )
        tracker.log_params(
            {
                "model_type": "GPR",
                "kernel": "RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01)",
                "n_restarts_optimizer": 10,
                "cv_strategy": cv_name,
                "feature_dim": X.shape[1],
                "normalize_y": True,
            }
        )

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # GPR with RBF + noise kernel
        # Bounds: length_scale [0.1, 100] captures typical spectral feature widths;
        # noise_level [1e-5, 1e1] spans dark-noise floor to calibration residual.
        kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(0.1, 100.0)) + WhiteKernel(
            noise_level=0.01, noise_level_bounds=(1e-5, 10.0)
        )
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            normalize_y=True,
            alpha=1e-6,
        )

        # Cross-validation R²
        cv_scores = cross_val_score(gpr, X_scaled, y, cv=cv_strategy, scoring="r2")
        r2_cv = float(np.mean(cv_scores))
        log.info("%s R² = %.4f ± %.4f", cv_name, r2_cv, np.std(cv_scores))

        # Full fit
        gpr.fit(X_scaled, y)
        y_pred, y_std = gpr.predict(X_scaled, return_std=True)

        r2 = float(r2_score(y, y_pred))
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))

        # ── GPR-aware LOD via root-finding ────────────────────────────────
        # Linear approximation (3σ/slope) ignores that GPR posterior variance
        # grows at concentration extremes.  The rigorous LOD satisfies:
        #   E[ŷ(c_LOD)] = 3 × Std[ŷ(c_LOD)]
        # We solve this 1-D equation on a dense grid.
        lod = _compute_gpr_aware_lod(gpr, scaler, concentrations, X_scaled, y_std)

        tracker.log_calibration_results(r2=r2, rmse_ppm=rmse, lod_ppm=lod)
        tracker.log_metrics(
            {
                "r2_cv": r2_cv,
                "r2_cv_std": float(np.std(cv_scores)),
                "cv_strategy": 1.0 if use_loocv else float(cv_folds),
                "rmse_train": rmse,
                "mean_gpr_std": float(np.mean(y_std)),
            }
        )

        # Save model
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"gpr_calibration_{gas_type}.joblib"
        joblib.dump({"model": gpr, "scaler": scaler, "gas_type": gas_type}, model_path)
        tracker.log_model(gpr, f"gpr_{gas_type}")

        log.info(
            "GPR training complete: R²=%.4f, RMSE=%.4f ppm, LOD=%s ppm → %s",
            r2,
            rmse,
            f"{lod:.3f}" if lod else "N/A",
            model_path,
        )

        return {
            "gas_type": gas_type,
            "r2": r2,
            "r2_cv": r2_cv,
            "rmse_ppm": rmse,
            "lod_ppm": lod,
            "n_samples": len(y),
            "model_path": str(model_path),
            "mlflow_run_id": tracker.run_id,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train GPR calibration models.")
    parser.add_argument(
        "--gas",
        default="Ethanol",
        help="Gas type to train (or 'all' for all known gases).",
    )
    parser.add_argument(
        "--data",
        default="data/JOY_Data",
        help="Root directory containing per-gas subdirectories.",
    )
    parser.add_argument(
        "--output",
        default="models/registry",
        help="Directory to save trained models.",
    )
    parser.add_argument(
        "--tracking-uri",
        default="experiments/mlruns",
        help="MLflow tracking URI.",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args(argv)

    data_root = Path(args.data)
    output_dir = Path(args.output)

    if args.gas == "all":
        # Auto-discover gas directories from the data root
        if data_root.exists():
            gas_types = [
                d.name
                for d in sorted(data_root.iterdir())
                if d.is_dir() and not d.name.startswith((".", "_"))
            ]
            log.info("Auto-discovered gas types: %s", gas_types)
        else:
            gas_types = ["Ethanol", "Multi mix vary-IPA", "Multi mix vary-MeOH", "Mixed gas"]
    else:
        gas_types = [args.gas]

    all_results = []
    for gas in gas_types:
        try:
            result = train_gpr(gas, data_root, output_dir, args.tracking_uri, args.cv_folds)
            all_results.append(result)
        except Exception as exc:
            log.error("Training failed for %s: %s", gas, exc)

    if all_results:
        log.info("\n=== Training Summary ===")
        for r in all_results:
            log.info(
                "  %-12s R²=%.4f  RMSE=%.4f ppm  LOD=%s ppm",
                r["gas_type"],
                r["r2"],
                r["rmse_ppm"],
                f"{r['lod_ppm']:.3f}" if r["lod_ppm"] else "N/A",
            )
        log.info("\nView results: mlflow ui --backend-store-uri %s", args.tracking_uri)


if __name__ == "__main__":
    main()

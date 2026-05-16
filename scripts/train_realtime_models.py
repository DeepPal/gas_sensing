"""
Train Real-Time Models from Batch Analysis Outputs
----------------------------------------------------
Trains and saves the CNN gas classifier and GPR calibration model
so the SensorOrchestrator can use them during live acquisition.

Run AFTER a successful batch analysis has produced ML feature CSVs.

Usage::

    python scripts/train_realtime_models.py
    python scripts/train_realtime_models.py --ml-dataset output/ml_dataset
    python scripts/train_realtime_models.py --model-dir output/models --gases Ethanol Methanol Isopropanol

Output files (saved to --model-dir, default: output/models/):
    cnn_classifier.pt        - PyTorch CNN gas classifier
    gpr_calibration.joblib   - scikit-learn GPR regression model
    calibration_params.json  - slope/intercept/reference_wavelength
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_ml_features(ml_dataset_dir: str) -> pd.DataFrame:
    """Load and concatenate all ml_features_*.csv files."""
    ml_dir = Path(ml_dataset_dir)
    files = sorted(ml_dir.glob("ml_features_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No ml_features_*.csv files found in {ml_dir}.\n"
            "Run a batch or deployable session first to generate them."
        )
    logger.info("Loading %d ML feature files from %s", len(files), ml_dir)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    logger.info("Total samples: %d", len(df))
    return df


# ---------------------------------------------------------------------------
# CNN training
# ---------------------------------------------------------------------------


def train_cnn(df: pd.DataFrame, model_dir: Path) -> bool:
    """
    Train CNN gas classifier from the ML feature dataset.
    Expects columns: wavelength_shift, snr, concentration_ppm, gas_label (or gas_type).
    """
    try:
        import pandas as pd

        from gas_analysis.core.intelligence.classifier import CNNGasClassifier

        # Check required columns
        label_col = None
        for candidate in ("gas_label", "gas_type", "gas"):
            if candidate in df.columns:
                label_col = candidate
                break
        if label_col is None:
            logger.warning(
                "No gas label column found (gas_label/gas_type). "
                "CNN training skipped — run sessions with --gas argument to record labels."
            )
            return False

        # Spectrum features required for CNN input
        spectrum_cols = [c for c in df.columns if c.startswith("w") and c[1:].isdigit()]
        if not spectrum_cols:
            logger.warning(
                "No spectrum columns (w0..w3647) in ML features. "
                "CNN training skipped — enable save_raw_spectra in config."
            )
            return False

        X = df[spectrum_cols].values.astype(np.float32)
        # Normalise each spectrum
        row_max = X.max(axis=1, keepdims=True) + 1e-9
        X = X / row_max

        y_label = df[label_col].values
        y_conc = df.get("concentration_ppm", pd.Series(np.zeros(len(df)))).values.astype(np.float32)

        # Encode labels
        class_names = sorted(set(y_label))
        label_to_idx = {name: i for i, name in enumerate(class_names)}
        y_label_int = np.array([label_to_idx[l] for l in y_label])

        logger.info(
            "Training CNN on %d samples, %d classes: %s",
            len(X),
            len(class_names),
            class_names,
        )

        clf = CNNGasClassifier()
        clf.fit(X, y_label_int, y_conc, class_names=class_names, epochs=30, batch_size=32)

        model_dir.mkdir(parents=True, exist_ok=True)
        out_path = model_dir / "cnn_classifier.pt"
        clf.save(str(out_path))
        logger.info("CNN classifier saved to %s", out_path)
        return True

    except Exception as exc:
        logger.error("CNN training failed: %s", exc, exc_info=True)
        return False


# ---------------------------------------------------------------------------
# GPR training
# ---------------------------------------------------------------------------


def train_gpr(df: pd.DataFrame, model_dir: Path) -> bool:
    """
    Train GPR concentration model from wavelength shift data.
    Expects columns: wavelength_shift, concentration_ppm.
    """
    try:
        import joblib

        from gas_analysis.core.intelligence.gpr import GPRCalibration

        required = {"wavelength_shift", "concentration_ppm"}
        if not required.issubset(df.columns):
            logger.warning(
                "Columns %s not found. GPR training skipped.", required - set(df.columns)
            )
            return False

        # Drop rows with NaN
        sub = df[["wavelength_shift", "concentration_ppm"]].dropna()
        if len(sub) < 5:
            logger.warning("Too few samples (%d) for GPR. Need at least 5.", len(sub))
            return False

        X = sub["wavelength_shift"].values.reshape(-1, 1).astype(np.float64)
        y = sub["concentration_ppm"].values.astype(np.float64)

        logger.info("Training GPR on %d samples (shift → concentration).", len(X))

        gpr = GPRCalibration()
        gpr.fit(X, y)
        mean, std = gpr.predict(X)
        residuals = y - mean
        rmse = float(np.sqrt(np.mean(residuals**2)))
        logger.info("GPR RMSE: %.4f ppm", rmse)

        model_dir.mkdir(parents=True, exist_ok=True)
        out_path = model_dir / "gpr_calibration.joblib"
        joblib.dump(gpr, str(out_path))
        logger.info("GPR model saved to %s", out_path)
        return True

    except Exception as exc:
        logger.error("GPR training failed: %s", exc, exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Calibration params extraction
# ---------------------------------------------------------------------------


def extract_calibration_params(df: pd.DataFrame, model_dir: Path) -> bool:
    """
    Fit a simple linear model (wavelength_shift → concentration) and save
    as calibration_params.json for ModelRegistry to load.
    """
    try:
        from sklearn.linear_model import LinearRegression

        sub = df[["wavelength_shift", "concentration_ppm"]].dropna()
        if len(sub) < 3:
            return False

        X = sub["wavelength_shift"].values.reshape(-1, 1)
        y = sub["concentration_ppm"].values

        lr = LinearRegression()
        lr.fit(X, y)
        r2 = lr.score(X, y)

        # slope in nm/ppm → invert to get concentration = shift / slope_nm_per_ppm
        # calibration_params uses slope as ppm/nm (concentration per unit shift)
        slope_ppm_per_nm = float(lr.coef_[0])
        intercept_ppm = float(lr.intercept_)

        # RealTimePipeline CalibrationStage uses: concentration = shift / slope_nm_per_ppm
        # so convert: slope_nm_per_ppm = 1 / slope_ppm_per_nm  (if non-zero)
        if abs(slope_ppm_per_nm) > 1e-9:
            slope_nm_per_ppm = 1.0 / slope_ppm_per_nm
        else:
            slope_nm_per_ppm = 0.116  # fallback default

        ref_wl = float(df["peak_wavelength"].median()) if "peak_wavelength" in df.columns else 531.5

        params = {
            "slope": slope_nm_per_ppm,
            "intercept": 0.0,
            "reference_wavelength": ref_wl,
            "r_squared": r2,
            "slope_ppm_per_nm": slope_ppm_per_nm,
            "intercept_ppm": intercept_ppm,
        }

        model_dir.mkdir(parents=True, exist_ok=True)
        out_path = model_dir / "calibration_params.json"
        with open(out_path, "w") as fh:
            json.dump(params, fh, indent=2)
        logger.info(
            "Calibration params saved: slope=%.4f nm/ppm, R²=%.3f → %s",
            slope_nm_per_ppm,
            r2,
            out_path,
        )
        return True

    except Exception as exc:
        logger.error("Calibration param extraction failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN + GPR models from batch outputs")
    parser.add_argument(
        "--ml-dataset",
        default="output/ml_dataset",
        help="Directory containing ml_features_*.csv files",
    )
    parser.add_argument(
        "--model-dir",
        default="output/models",
        help="Directory to save trained model files",
    )
    args = parser.parse_args()

    # Load data
    df = load_ml_features(args.ml_dataset)
    model_dir = Path(args.model_dir)

    # Train models
    results = {
        "calibration": extract_calibration_params(df, model_dir),
        "gpr": train_gpr(df, model_dir),
        "cnn": train_cnn(df, model_dir),
    }

    print("\n--- Training Summary ---")
    for name, ok in results.items():
        status = "OK" if ok else "SKIPPED"
        print(f"  {name:<15} {status}")

    if any(results.values()):
        print(f"\nModels saved to: {model_dir.resolve()}")
        print("Run the live sensor to use them:")
        print("  python run.py --mode sensor --gas Ethanol --duration 3600")
    else:
        print("\nNo models trained. Check logs above for details.")


if __name__ == "__main__":
    main()

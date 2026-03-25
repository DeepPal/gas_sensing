"""
src.training.train_cnn
======================
Training script for the 1-D CNN gas classifier.

Loads Joy_Data CSVs, extracts features, trains the dual-head CNN
(gas-type classification + concentration regression), evaluates with
leave-one-concentration-out cross-validation, and saves the checkpoint.

Usage
-----
::

    # Basic
    python -m src.training.train_cnn --data data/raw/Joy_Data --gas Ethanol

    # Full options
    python -m src.training.train_cnn \\
        --data  data/raw/Joy_Data \\
        --gas   Ethanol \\
        --output models/registry \\
        --epochs 50 \\
        --batch-size 32 \\
        --lr 1e-3 \\
        --n-tail 10 \\
        --tracking-uri experiments/mlruns \\
        --experiment AuMIP_CNN_Training

The script expects Joy_Data to follow the standard layout::

    Joy_Data/<GasType>/<concentration> ppm <label>-<trial>/<frame>.csv

and optionally multiple gas types (for multi-class training).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_training_data(
    data_root: str,
    gas_types: list[str],
    n_tail: int = 10,
    target_length: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load and prepare training arrays from Joy_Data layout.

    Parameters
    ----------
    data_root:
        Root directory; should contain one sub-directory per gas type.
    gas_types:
        List of gas type sub-directory names to include.
    n_tail:
        Number of trailing (plateau) frames to use per trial.
    target_length:
        All spectra are interpolated to this number of points.

    Returns
    -------
    X : ndarray, shape (n_samples, target_length)
        Spectra matrix.
    y_label : ndarray of int, shape (n_samples,)
        Class label indices.
    y_conc : ndarray of float, shape (n_samples,)
        Concentration targets in ppm.
    class_names : list of str
        Ordered list mapping label int → gas name.
    """

    from src.batch.data_loader import load_last_n_frames, scan_experiment_root

    X_list: list[np.ndarray] = []
    y_label_list: list[int] = []
    y_conc_list: list[float] = []
    class_names: list[str] = []

    root = Path(data_root)

    for _gas_idx, gas_type in enumerate(gas_types):
        gas_dir = root / gas_type
        if not gas_dir.is_dir():
            log.warning("Gas directory not found: %s — skipping.", gas_dir)
            continue

        class_names.append(gas_type)
        label_idx = len(class_names) - 1

        try:
            scan = scan_experiment_root(gas_dir, gas_type=gas_type)
        except ValueError as exc:
            log.warning("scan_experiment_root failed for %s: %s", gas_dir, exc)
            continue

        for conc_ppm, conc_trials in scan.trials.items():
            for trial_label, frame_paths in conc_trials.items():
                frames = load_last_n_frames(frame_paths, n=n_tail)
                if not frames:
                    continue
                for df in frames:
                    try:
                        wl = df["wavelength"].to_numpy(dtype=float)
                        it = df["intensity"].to_numpy(dtype=float)
                        # Resample to target_length
                        wl_new = np.linspace(wl[0], wl[-1], target_length)
                        it_rs = np.interp(wl_new, wl, it)
                        X_list.append(it_rs)
                        y_label_list.append(label_idx)
                        y_conc_list.append(float(conc_ppm))
                    except Exception as exc:
                        log.debug("Skipping frame in %s: %s", trial_label, exc)

    if not X_list:
        raise RuntimeError(f"No training samples found in {data_root} for gas types {gas_types}.")

    X = np.vstack(X_list)
    y_label = np.array(y_label_list, dtype=np.int64)
    y_conc = np.array(y_conc_list, dtype=np.float32)

    log.info(
        "Loaded %d training samples | %d classes | concentration range [%.2f, %.2f] ppm",
        len(X),
        len(class_names),
        y_conc.min(),
        y_conc.max(),
    )
    return X, y_label, y_conc, class_names


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate_loocv(
    X: np.ndarray,
    y_label: np.ndarray,
    y_conc: np.ndarray,
    class_names: list[str],
    input_length: int = 1000,
) -> dict[str, float]:
    """Leave-one-concentration-out cross-validation.

    Returns accuracy, mean RMSE concentration, and per-class accuracy.
    Quick check only — not a substitute for a held-out test set.
    """
    from src.models.cnn import CNNGasClassifier

    concentrations = np.unique(y_conc)
    if len(concentrations) < 2:
        return {"loocv_accuracy": float("nan"), "loocv_rmse_ppm": float("nan")}

    correct_cls = 0
    total = 0
    rmse_accum = 0.0

    for hold_out_conc in concentrations:
        test_mask = np.abs(y_conc - hold_out_conc) < 1e-6
        train_mask = ~test_mask

        if train_mask.sum() < 4 or test_mask.sum() == 0:
            continue

        clf = CNNGasClassifier(input_length=input_length, num_classes=len(class_names))
        try:
            clf.fit(
                X[train_mask],
                y_label[train_mask],
                y_conc[train_mask],
                class_names=class_names,
                epochs=15,
                batch_size=16,
            )
            preds, conc_preds = clf.predict(X[test_mask])
        except Exception as exc:
            log.warning("LOOCV fold failed (conc=%.2f): %s", hold_out_conc, exc)
            continue

        true_labels = [class_names[i] for i in y_label[test_mask]]
        correct_cls += sum(p == t for p, t in zip(preds, true_labels))
        total += len(preds)
        rmse_accum += float(np.sqrt(np.mean((conc_preds - y_conc[test_mask]) ** 2)))

    n_folds = len(concentrations)
    return {
        "loocv_accuracy": correct_cls / max(total, 1),
        "loocv_rmse_ppm": rmse_accum / max(n_folds, 1),
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_cnn(
    gas_types: list[str],
    data_root: str,
    output_dir: str = "models/registry",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    n_tail: int = 10,
    input_length: int = 1000,
    tracking_uri: str = "experiments/mlruns",
    experiment_name: str = "AuMIP_CNN_Training",
) -> dict[str, object]:
    """End-to-end CNN training pipeline.

    Parameters
    ----------
    gas_types:
        List of gas type names to include (must match subdirectory names).
    data_root:
        Root directory containing one folder per gas type.
    output_dir:
        Where to save the trained checkpoint.
    epochs, batch_size, lr:
        Training hyperparameters.
    n_tail:
        Number of plateau frames per trial.
    input_length:
        Spectrum length after resampling.
    tracking_uri, experiment_name:
        MLflow tracking configuration.

    Returns
    -------
    dict
        Training summary with metrics and output path.
    """
    try:
        from src.models.cnn import CNNGasClassifier
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for CNN training. Install with: pip install torch"
        ) from exc

    from src.training.mlflow_tracker import ExperimentTracker

    log.info("Loading training data from %s for gas types: %s", data_root, gas_types)
    X, y_label, y_conc, class_names = _load_training_data(
        data_root, gas_types, n_tail=n_tail, target_length=input_length
    )

    params = {
        "gas_types": gas_types,
        "n_samples": len(X),
        "n_classes": len(class_names),
        "input_length": input_length,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "n_tail": n_tail,
    }

    with ExperimentTracker(
        experiment_name=experiment_name,
        run_name=f"cnn_{'_'.join(gas_types)}",
        tracking_uri=tracking_uri,
    ) as tracker:
        tracker.log_params(params)

        # Train
        log.info(
            "Training CNN: %d samples, %d classes, %d epochs", len(X), len(class_names), epochs
        )
        clf = CNNGasClassifier(input_length=input_length, num_classes=len(class_names))
        history = clf.fit(
            X,
            y_label,
            y_conc,
            class_names=class_names,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
        )

        final_acc = history["cls_acc"][-1] if history["cls_acc"] else 0.0
        final_loss = history["loss"][-1] if history["loss"] else 0.0
        log.info("Training complete: acc=%.3f, loss=%.4f", final_acc, final_loss)

        # LOOCV evaluation
        log.info("Running LOOCV evaluation...")
        loocv = _evaluate_loocv(X, y_label, y_conc, class_names, input_length)
        log.info(
            "LOOCV accuracy=%.3f, RMSE=%.4f ppm",
            loocv.get("loocv_accuracy", 0),
            loocv.get("loocv_rmse_ppm", 0),
        )

        metrics = {
            "train_accuracy": final_acc,
            "train_loss": final_loss,
            **loocv,
        }
        tracker.log_metrics(metrics)

        # Save checkpoint
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        ckpt_path = str(out_path / "cnn_classifier.pt")
        clf.save(ckpt_path)

        # Save class map and metadata
        meta = {
            "class_names": class_names,
            "input_length": input_length,
            "trained_on": gas_types,
            **metrics,
        }
        with open(out_path / "cnn_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        log.info("CNN checkpoint saved to %s", ckpt_path)
        return {"output_path": ckpt_path, **metrics}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train the Au-MIP LSPR 1-D CNN gas classifier")
    parser.add_argument(
        "--data",
        required=True,
        help="Root directory of Joy_Data (contains one folder per gas type)",
    )
    parser.add_argument(
        "--gas",
        nargs="+",
        default=["Ethanol", "IPA", "Methanol"],
        help="Gas type names (must match subdirectory names)",
    )
    parser.add_argument(
        "--output",
        default="models/registry",
        help="Output directory for the checkpoint (default: models/registry)",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-tail", type=int, default=10)
    parser.add_argument(
        "--tracking-uri",
        default="experiments/mlruns",
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--experiment",
        default="AuMIP_CNN_Training",
        help="MLflow experiment name",
    )
    args = parser.parse_args(argv)

    result = train_cnn(
        gas_types=args.gas,
        data_root=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_tail=args.n_tail,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

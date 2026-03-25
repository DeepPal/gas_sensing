"""
src.training.cross_gas_eval
===========================
Leave-one-gas-out (LOGO) cross-validation for CNNGasClassifier + GPRCalibration.

What it does
------------
For each gas G in the dataset:
  1. Train on all spectra *except* those labelled G.
  2. Evaluate on held-out G spectra.
  3. Report per-gas metrics: classification accuracy, concentration RMSE, R².

Why this matters
----------------
Standard k-fold on the full dataset can leak within-gas variance between
folds and give an over-optimistic estimate.  LOGO measures true
*generalisation* — can the model handle a gas it has never seen in
training?  For a deployable sensor this is the critical question.

Usage
-----
::

    python -m src.training.cross_gas_eval \\
        --data-dir data/JOY_Data \\
        --model-dir output/models \\
        --mlflow-uri mlruns

Outputs
-------
- Console table with per-gas and overall metrics.
- MLflow run tagged ``cross_gas_eval`` (optional, skipped if MLflow unavailable).
- ``output/cross_gas_eval.json`` — machine-readable results.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_spectra_from_dir(
    data_dir: Path,
    gas_labels: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load spectra from a directory tree matching Joy-data layout.

    Expected layout::

        data_dir/
          Ethanol/
            1 ppm Ethanol-1/  ← sub-runs
              *.csv
          IPA/
            ...

    Each CSV row represents one spectrum (columns = wavelength bins, or two
    columns ``wavelength,intensity`` for single spectra).

    Returns
    -------
    X : ndarray (n_samples, n_points)
    y_label : ndarray (n_samples,) int class index
    y_conc : ndarray (n_samples,) float ppm
    class_names : List[str]
    """
    import pandas as pd

    data_dir = Path(data_dir)
    gas_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if gas_labels:
        gas_dirs = [d for d in gas_dirs if d.name in gas_labels]

    class_names: list[str] = []
    X_rows: list[np.ndarray] = []
    y_labels: list[int] = []
    y_concs: list[float] = []

    for cls_idx, gas_dir in enumerate(gas_dirs):
        gas_name = gas_dir.name
        class_names.append(gas_name)
        csv_files = list(gas_dir.rglob("*.csv"))
        if not csv_files:
            log.warning("No CSV files found under %s", gas_dir)
            continue

        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path, header=None)
                row = df.values.flatten().astype(float)
                if row.size < 10:
                    continue
                # Try to parse concentration from parent folder name
                conc = _parse_concentration(csv_path)
                X_rows.append(row)
                y_labels.append(cls_idx)
                y_concs.append(conc)
            except Exception as exc:
                log.debug("Skipping %s: %s", csv_path, exc)

    if not X_rows:
        raise ValueError(f"No spectra loaded from {data_dir}")

    # Align length to the shortest spectrum (trim to common size)
    min_len = min(r.size for r in X_rows)
    X = np.stack([r[:min_len] for r in X_rows], axis=0)
    return X, np.array(y_labels, dtype=int), np.array(y_concs, dtype=float), class_names


def _parse_concentration(path: Path) -> float:
    """Extract ppm concentration from a Joy-data folder name like '1 ppm Ethanol-1'."""
    import re

    text = path.parent.name
    m = re.search(r"([\d.]+)\s*ppm", text, re.IGNORECASE)
    return float(m.group(1)) if m else 0.0


# ---------------------------------------------------------------------------
# LOGO evaluation
# ---------------------------------------------------------------------------


def run_logo_eval(
    X: np.ndarray,
    y_label: np.ndarray,
    y_conc: np.ndarray,
    class_names: list[str],
    epochs: int = 20,
    n_mc_samples: int = 30,
) -> dict:
    """Leave-one-gas-out evaluation of CNNGasClassifier.

    Parameters
    ----------
    X, y_label, y_conc, class_names:
        Full dataset as returned by :func:`_load_spectra_from_dir`.
    epochs:
        CNN training epochs per fold (keep low for quick iteration; use ≥50
        for publication results).
    n_mc_samples:
        MC Dropout forward passes for uncertainty estimation on held-out fold.

    Returns
    -------
    dict
        ``{"overall": {...}, "per_gas": {gas: {...}}}``
    """
    try:
        from src.models.cnn import CNNGasClassifier
    except ImportError as exc:
        raise ImportError("src.models.cnn not available — check your PYTHONPATH.") from exc

    n_classes = len(class_names)
    per_gas: dict[str, dict] = {}

    all_correct = 0
    all_total = 0
    all_sq_err: list[float] = []
    all_conc_true: list[float] = []

    for held_out_idx, held_out_gas in enumerate(class_names):
        train_mask = y_label != held_out_idx
        test_mask = y_label == held_out_idx

        if not test_mask.any():
            log.warning("No test samples for gas %s — skipping.", held_out_gas)
            continue
        if not train_mask.any():
            log.warning("No train samples for LOGO fold without %s — skipping.", held_out_gas)
            continue

        X_train, y_cls_train, y_conc_train = X[train_mask], y_label[train_mask], y_conc[train_mask]
        X_test, y_conc_test = X[test_mask], y_conc[test_mask]

        # Re-map labels so they are contiguous (0..n-2) for the training fold
        train_classes = [c for i, c in enumerate(class_names) if i != held_out_idx]
        label_remap = {old: new for new, old in enumerate(sorted(set(y_cls_train.tolist())))}
        y_cls_train_remapped = np.array([label_remap[int(lbl)] for lbl in y_cls_train])

        log.info(
            "LOGO fold %d/%d — held-out: %s (%d test samples, %d train samples)",
            held_out_idx + 1,
            n_classes,
            held_out_gas,
            test_mask.sum(),
            train_mask.sum(),
        )

        clf = CNNGasClassifier(input_length=min(X.shape[1], 1000), num_classes=len(train_classes))
        clf.fit(
            X_train,
            y_cls_train_remapped,
            y_conc_train,
            class_names=train_classes,
            epochs=epochs,
        )

        # Evaluate
        conc_preds_mc: list[float] = []
        conc_stds_mc: list[float] = []
        for spec in X_test:
            _, c_mean, c_std, _ = clf.predict_with_uncertainty(spec, n_samples=n_mc_samples)
            conc_preds_mc.append(c_mean)
            conc_stds_mc.append(c_std)

        # Classification: model will predict one of the train_classes — the
        # held-out gas will always be "wrong" (expected for true LOGO).
        gas_names_pred, _ = clf.predict(X_test)
        n_correct = sum(1 for g in gas_names_pred if g == held_out_gas)
        n_test = len(X_test)

        # Concentration RMSE (only meaningful when model has seen this gas)
        sq_errors = [(p - t) ** 2 for p, t in zip(conc_preds_mc, y_conc_test.tolist())]
        rmse = float(np.sqrt(np.mean(sq_errors))) if sq_errors else float("nan")

        per_gas[held_out_gas] = {
            "n_test": n_test,
            "cls_accuracy_pct": round(100 * n_correct / n_test, 2),
            "conc_rmse_ppm": round(rmse, 4),
            "mean_mc_uncertainty_ppm": round(float(np.mean(conc_stds_mc)), 4),
        }

        all_correct += n_correct
        all_total += n_test
        all_sq_err.extend(sq_errors)
        all_conc_true.extend(y_conc_test.tolist())

    overall_rmse = float(np.sqrt(np.mean(all_sq_err))) if all_sq_err else float("nan")
    # R² over all folds
    ss_res = float(np.sum(all_sq_err)) if all_sq_err else float("nan")
    mean_true = float(np.mean(all_conc_true)) if all_conc_true else 0.0
    ss_tot = (
        float(np.sum([(v - mean_true) ** 2 for v in all_conc_true]))
        if all_conc_true
        else float("nan")
    )
    r2 = 1.0 - ss_res / ss_tot if (ss_tot and not np.isnan(ss_tot) and ss_tot > 0) else float("nan")

    return {
        "overall": {
            "cls_accuracy_pct": round(100 * all_correct / max(all_total, 1), 2),
            "conc_rmse_ppm": round(overall_rmse, 4),
            "r2": round(r2, 4),
            "n_folds": len(per_gas),
            "n_total_test": all_total,
        },
        "per_gas": per_gas,
    }


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------


def _log_to_mlflow(
    results: dict,
    mlflow_uri: str,
    model_dir: str,
    out_path: Path | None = None,
    epochs: int = 20,
    n_mc_samples: int = 30,
    gas_labels: list[str] | None = None,
) -> None:
    from datetime import datetime

    from src.training.mlflow_tracker import _DEFAULT_TRACKING_URI, ExperimentTracker

    run_name = f"logo_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    uri = mlflow_uri if mlflow_uri != "mlruns" else _DEFAULT_TRACKING_URI

    try:
        with ExperimentTracker(
            run_name=run_name,
            experiment_name="cross_gas_eval",
            tracking_uri=uri,
            tags={"eval_type": "leave_one_gas_out", "model_dir": model_dir},
        ) as tracker:
            tracker.log_params(
                {
                    "epochs": epochs,
                    "n_mc_samples": n_mc_samples,
                    "gas_labels": json.dumps(gas_labels or results.get("class_names", [])),
                    "n_classes": len(results.get("per_gas", {})),
                }
            )
            overall = results.get("overall", {})
            tracker.log_metrics(
                {k: float(v) for k, v in overall.items() if isinstance(v, (int, float))}
            )
            for gas, metrics in results.get("per_gas", {}).items():
                tracker.log_metrics(
                    {
                        f"{gas}_{k}": float(v)
                        for k, v in metrics.items()
                        if isinstance(v, (int, float))
                    }
                )
            if out_path and Path(out_path).exists():
                tracker.log_artifact(out_path, "results")
            log.info("LOGO results logged to MLflow run '%s'.", run_name)
    except ImportError:
        log.info("MLflow not installed — skipping MLflow logging.")
    except Exception as exc:
        log.warning("MLflow logging failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _print_table(results: dict) -> None:
    overall = results["overall"]
    per_gas = results["per_gas"]
    print("\n" + "=" * 65)
    print("  Leave-One-Gas-Out (LOGO) Cross-Validation Results")
    print("=" * 65)
    print(f"  Overall accuracy : {overall['cls_accuracy_pct']:6.2f}%")
    print(f"  Overall RMSE     : {overall['conc_rmse_ppm']:6.4f} ppm")
    print(f"  Overall R²       : {overall['r2']:6.4f}")
    print(f"  Folds            : {overall['n_folds']}")
    print("-" * 65)
    print(f"  {'Gas':<18} {'N':>5} {'Acc%':>8} {'RMSE(ppm)':>12} {'MC±(ppm)':>10}")
    print("-" * 65)
    for gas, m in per_gas.items():
        print(
            f"  {gas:<18} {m['n_test']:>5} {m['cls_accuracy_pct']:>8.2f}"
            f" {m['conc_rmse_ppm']:>12.4f} {m['mean_mc_uncertainty_ppm']:>10.4f}"
        )
    print("=" * 65 + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Leave-one-gas-out cross-validation")
    parser.add_argument("--data-dir", default="Joy_Data", help="Root data directory")
    parser.add_argument("--model-dir", default="output/models", help="Model output directory")
    parser.add_argument("--mlflow-uri", default="experiments/mlruns", help="MLflow tracking URI")
    parser.add_argument("--epochs", type=int, default=20, help="CNN training epochs per fold")
    parser.add_argument("--mc-samples", type=int, default=30, help="MC Dropout forward passes")
    parser.add_argument("--output", default="output/cross_gas_eval.json", help="Results JSON path")
    parser.add_argument("--gas-labels", nargs="*", help="Subset of gas directories to include")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        log.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    log.info("Loading spectra from %s ...", data_dir)
    X, y_label, y_conc, class_names = _load_spectra_from_dir(data_dir, args.gas_labels)
    log.info("Loaded %d spectra, %d classes: %s", len(X), len(class_names), class_names)

    results = run_logo_eval(
        X, y_label, y_conc, class_names, epochs=args.epochs, n_mc_samples=args.mc_samples
    )

    _print_table(results)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info("Results written to %s", out_path)

    _log_to_mlflow(
        results,
        mlflow_uri=args.mlflow_uri,
        model_dir=args.model_dir,
        out_path=out_path,
        epochs=args.epochs,
        n_mc_samples=args.mc_samples,
        gas_labels=args.gas_labels,
    )


if __name__ == "__main__":
    main()

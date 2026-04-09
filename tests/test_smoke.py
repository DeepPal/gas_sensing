"""
tests/test_smoke.py
===================
End-to-end smoke tests for the the sensor LSPR gas sensing platform.

These tests exercise the full config → preprocess → feature extraction →
calibration chain against synthetic data, without requiring hardware or
trained model files.  They are intentionally broad ("does the pipeline
produce sensible output?") rather than precise ("is the exact value X?").

Run selectively with::

    pytest -m smoke -v

All tests are marked ``@pytest.mark.smoke`` so the normal ``pytest -q``
run (CI) does NOT execute them — they are for pre-release validation only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lspr_spectrum(
    n_points: int = 3648,
    peak_nm: float = 717.9,
    peak_shift_nm: float = 0.0,
    noise_std: float = 30.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic CCS200 LSPR spectrum (Gaussian absorption on flat baseline)."""
    rng = np.random.default_rng(seed)
    wl = np.linspace(500.0, 1000.0, n_points)
    baseline = np.full(n_points, 40_000.0)
    center = peak_nm + peak_shift_nm
    absorption = 8_000.0 * np.exp(-0.5 * ((wl - center) / 3.0) ** 2)
    noise = rng.normal(0.0, noise_std, n_points)
    return wl, baseline - absorption + noise


# ---------------------------------------------------------------------------
# 1. Config loading
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_config_loads_and_has_required_keys(full_config_path: Path) -> None:
    """Real config/config.yaml loads and exposes all top-level sections."""
    from config.config_loader import load_config

    cfg = load_config(str(full_config_path))

    required_sections = [
        "preprocessing",
        "roi",
        "sensor",
        "mlflow",
        "api",
        "agents",
    ]
    for section in required_sections:
        assert section in cfg, f"Missing config section: {section}"


@pytest.mark.smoke
def test_model_yaml_loads(full_config_path: Path) -> None:
    """config/MODEL.yaml is valid YAML and contains cnn + gpr sections."""
    import yaml

    model_yaml = full_config_path.parent / "MODEL.yaml"
    assert model_yaml.exists(), "config/MODEL.yaml not found"

    with model_yaml.open() as f:
        model_cfg = yaml.safe_load(f)

    assert "cnn" in model_cfg, "MODEL.yaml missing 'cnn' section"
    assert "gpr" in model_cfg, "MODEL.yaml missing 'gpr' section"
    assert "model_version" in model_cfg, "MODEL.yaml missing 'model_version'"


# ---------------------------------------------------------------------------
# 2. Preprocessing pipeline
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_preprocessing_produces_clean_spectrum() -> None:
    """ALS baseline + S-G smoothing runs without error and returns same shape."""
    from gas_analysis.core.preprocessing import preprocess_spectrum

    wl, raw = _make_lspr_spectrum()
    result = preprocess_spectrum(wl, raw)

    assert result is not None, "preprocess_spectrum returned None"
    wl_out, clean = result if isinstance(result, tuple) else (wl, result)
    assert clean.shape == raw.shape, "Output shape changed after preprocessing"
    assert not np.any(np.isnan(clean)), "NaN values in preprocessed spectrum"
    assert not np.any(np.isinf(clean)), "Inf values in preprocessed spectrum"


# ---------------------------------------------------------------------------
# 3. Peak detection
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_peak_detection_finds_lspr_peak() -> None:
    """Peak wavelength found within ±2 nm of the planted LSPR extinction peak.

    The LSPR extinction spectrum has a clear Gaussian peak; detect_peaks
    must locate it reliably.
    """
    from gas_analysis.feature_extraction import detect_peaks

    planted_peak = 717.9
    wl = np.linspace(500.0, 1000.0, 3648)
    # Build an extinction (peak-up) spectrum: Gaussian peak on a flat baseline
    extinction = 0.5 * np.exp(-0.5 * ((wl - planted_peak) / 3.0) ** 2)
    df = pd.DataFrame({"wavelength": wl, "intensity": extinction})

    result = detect_peaks(df, height=0.01, distance=20, prominence=0.01)

    assert len(result["peaks"]) > 0, "detect_peaks found no peaks in the extinction spectrum"
    peak_indices = result["peaks"]
    peak_wl = float(wl[peak_indices[np.argmax(extinction[peak_indices])]])
    assert abs(peak_wl - planted_peak) < 2.0, (
        f"Peak found at {peak_wl:.2f} nm, expected ~{planted_peak:.1f} nm"
    )


# ---------------------------------------------------------------------------
# 4. Calibration (GPR) — fit + predict round-trip
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_gpr_calibration_fit_predict_roundtrip() -> None:
    """GPR fits a linear Δλ–concentration curve and recovers concentrations."""
    from src.calibration.gpr import GPRCalibration

    rng = np.random.default_rng(0)
    concentrations = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
    sensitivity = 0.116  # nm/ppm
    # Simulate Δλ with small noise
    delta_lambda = -sensitivity * concentrations + rng.normal(0, 0.01, len(concentrations))

    gpr = GPRCalibration()
    gpr.fit(delta_lambda.reshape(-1, 1), concentrations)

    test_inputs = np.array([-0.058, -0.116, -0.232]).reshape(-1, 1)
    pred, std = gpr.predict(test_inputs, return_std=True)

    assert pred.shape == (3,), "Unexpected prediction shape"
    assert std.shape == (3,), "Unexpected uncertainty shape"
    assert np.all(std >= 0), "Negative uncertainty values"
    # Rough sanity: predictions should be in the calibrated range
    assert np.all(pred > 0), "Negative concentration predictions"
    assert np.all(pred < 20.0), "Unreasonably large concentration predictions"


# ---------------------------------------------------------------------------
# 5. Batch aggregation — stable-plateau detection
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_stable_plateau_detection_returns_canonical_spectrum(
    canonical_spectra: dict,
) -> None:
    """build_canonical_from_scan returns one DataFrame per concentration."""
    from src.batch.aggregation import build_canonical_from_scan

    # Wrap canonical_spectra into the expected scan_data format
    scan_data: dict = {
        conc: {"trial_1": [df.copy() for _ in range(5)]} for conc, df in canonical_spectra.items()
    }

    result = build_canonical_from_scan(scan_data, n_tail=3, weight_mode="max")

    assert len(result) == len(canonical_spectra), (
        f"Expected {len(canonical_spectra)} concentrations, got {len(result)}"
    )
    for conc, df in result.items():
        assert isinstance(df, pd.DataFrame), f"Result for {conc} ppm is not a DataFrame"
        assert not df.empty, f"Empty canonical spectrum at {conc} ppm"
        assert "wavelength" in df.columns, "Missing 'wavelength' column"


# ---------------------------------------------------------------------------
# 6. LiveDataStore — thread safety
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_live_data_store_thread_safe_push_and_read() -> None:
    """Push 100 frames from a background thread; read them from main thread."""
    import threading

    from src.inference.live_state import _LiveDataStore

    store = _LiveDataStore(maxlen=200)
    wl = np.linspace(500, 1000, 3648)
    store.set_wavelengths(wl)

    errors: list[str] = []

    def _writer() -> None:
        for i in range(100):
            result = {"timestamp": float(i), "concentration_ppm": float(i) * 0.1}
            _, inten = _make_lspr_spectrum(seed=i)
            try:
                store.push(result, inten)
            except Exception as exc:
                errors.append(str(exc))

    t = threading.Thread(target=_writer, daemon=True)
    t.start()
    t.join(timeout=10.0)

    assert not errors, f"Writer thread errors: {errors}"
    assert store.get_sample_count() == 100
    assert len(store.get_latest(50)) == 50
    spectrum = store.get_latest_spectrum()
    assert spectrum is not None
    assert spectrum[0].shape == wl.shape


# ---------------------------------------------------------------------------
# 7. ONNX export validation (skipped if torch unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_onnx_export_validates_if_torch_available(tmp_path: Path) -> None:
    """CNN ONNX export produces a file that validates within tolerance."""
    pytest.importorskip("torch", reason="torch not installed")
    pytest.importorskip(
        "onnxscript", reason="onnxscript not installed (required by torch >= 2.0 ONNX export)"
    )
    from src.models.cnn import CNNGasClassifier, GasCNN
    from src.models.onnx_export import export_cnn_to_onnx, validate_onnx_export

    clf = CNNGasClassifier(num_classes=3, input_length=512)
    # Wire a freshly-initialised (random-weight) model so is_fitted=True
    # without running a multi-epoch training loop in a smoke test.
    clf.model = GasCNN(input_length=512, num_classes=3).to(clf.device)
    clf.class_map = {0: "Ethanol", 1: "IPA", 2: "Methanol"}
    clf.is_fitted = True
    onnx_path = str(tmp_path / "test_model.onnx")

    export_cnn_to_onnx(clf, onnx_path)
    assert Path(onnx_path).exists(), "ONNX file not created"

    ok, delta = validate_onnx_export(clf, onnx_path)
    assert ok, f"ONNX validation failed: max delta = {delta:.2e}"
    assert delta < 1e-4, f"ONNX output too far from PyTorch: delta={delta:.2e}"

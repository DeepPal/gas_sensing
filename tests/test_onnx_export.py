"""
tests.test_onnx_export
=======================
Unit tests for src.models.onnx_export:
  - export_cnn_to_onnx  (skipped if torch/onnx unavailable)
  - validate_onnx_export
  - OnnxInferenceWrapper
  - CLI main() argument parsing

All tests that require a fitted CNN are skipped gracefully when PyTorch or
the onnx/onnxruntime packages are not installed, so the test suite stays
green in minimal (CPU-only, no-torch) environments.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Skip guards
# ---------------------------------------------------------------------------

try:
    import torch

    _TORCH = True
except ImportError:
    _TORCH = False

try:
    import onnx

    _ONNX = True
except ImportError:
    _ONNX = False

try:
    import onnxruntime  # noqa: F401

    _ORT = True
except ImportError:
    _ORT = False

_SKIP_TORCH = pytest.mark.skipif(not _TORCH, reason="PyTorch not installed")
_SKIP_ONNX = pytest.mark.skipif(not (_TORCH and _ONNX), reason="torch or onnx not installed")
_SKIP_ORT = pytest.mark.skipif(not (_TORCH and _ORT), reason="torch or onnxruntime not installed")
_SKIP_FULL = pytest.mark.skipif(
    not (_TORCH and _ONNX and _ORT),
    reason="torch, onnx, and onnxruntime all required",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_clf():
    """A minimal fitted CNNGasClassifier (input_length=50, 2 classes)."""
    if not _TORCH:
        pytest.skip("PyTorch not installed")

    from src.models.cnn import CNNGasClassifier

    rng = np.random.default_rng(0)
    n = 20
    X = rng.standard_normal((n, 50)).astype(np.float32)
    y_cls = np.array([i % 2 for i in range(n)], dtype=int)
    y_conc = rng.uniform(0.5, 5.0, n).astype(np.float32)

    clf = CNNGasClassifier(input_length=50, num_classes=2, device="cpu")
    clf.fit(X, y_cls, y_conc, class_names=["GasA", "GasB"], epochs=2, batch_size=8)
    return clf


@pytest.fixture
def onnx_file(tiny_clf, tmp_path):
    """Export tiny_clf to a temporary .onnx file."""
    if not _ONNX:
        pytest.skip("onnx not installed")
    from src.models.onnx_export import export_cnn_to_onnx

    out = tmp_path / "test_model.onnx"
    export_cnn_to_onnx(tiny_clf, str(out), opset_version=17)
    return out


# ---------------------------------------------------------------------------
# export_cnn_to_onnx
# ---------------------------------------------------------------------------


class TestExportCnnToOnnx:
    @_SKIP_ONNX
    def test_file_created(self, tiny_clf, tmp_path):
        from src.models.onnx_export import export_cnn_to_onnx

        out = tmp_path / "model.onnx"
        result = export_cnn_to_onnx(tiny_clf, str(out))
        assert result.exists()
        assert result.suffix == ".onnx"

    @_SKIP_ONNX
    def test_file_is_nonzero(self, tiny_clf, tmp_path):
        from src.models.onnx_export import export_cnn_to_onnx

        out = tmp_path / "model.onnx"
        export_cnn_to_onnx(tiny_clf, str(out))
        assert out.stat().st_size > 0

    @_SKIP_ONNX
    def test_returns_path_object(self, tiny_clf, tmp_path):
        from src.models.onnx_export import export_cnn_to_onnx

        out = tmp_path / "model.onnx"
        result = export_cnn_to_onnx(tiny_clf, str(out))
        assert isinstance(result, Path)

    @_SKIP_ONNX
    def test_class_map_in_metadata(self, tiny_clf, tmp_path):
        from src.models.onnx_export import export_cnn_to_onnx

        out = tmp_path / "model.onnx"
        export_cnn_to_onnx(tiny_clf, str(out))
        import onnx

        proto = onnx.load(str(out))
        keys = {p.key for p in proto.metadata_props}
        assert "class_map" in keys
        assert "input_length" in keys

    @_SKIP_ONNX
    def test_class_map_values_correct(self, tiny_clf, tmp_path):
        from src.models.onnx_export import export_cnn_to_onnx

        out = tmp_path / "model.onnx"
        export_cnn_to_onnx(tiny_clf, str(out))
        import onnx

        proto = onnx.load(str(out))
        for p in proto.metadata_props:
            if p.key == "class_map":
                cm = json.loads(p.value)
                assert set(cm.values()) == {"GasA", "GasB"}

    @_SKIP_TORCH
    def test_unfitted_classifier_raises(self, tmp_path):
        from src.models.cnn import CNNGasClassifier
        from src.models.onnx_export import export_cnn_to_onnx

        clf = CNNGasClassifier(input_length=50, num_classes=2, device="cpu")
        with pytest.raises(RuntimeError, match="fitted"):
            export_cnn_to_onnx(clf, str(tmp_path / "bad.onnx"))


# ---------------------------------------------------------------------------
# validate_onnx_export
# ---------------------------------------------------------------------------


class TestValidateOnnxExport:
    @_SKIP_FULL
    def test_validation_passes(self, tiny_clf, onnx_file):
        from src.models.onnx_export import validate_onnx_export

        passed, max_delta = validate_onnx_export(tiny_clf, str(onnx_file), n_test_inputs=5)
        assert passed
        assert max_delta < 1e-4

    @_SKIP_FULL
    def test_returns_tuple(self, tiny_clf, onnx_file):
        from src.models.onnx_export import validate_onnx_export

        result = validate_onnx_export(tiny_clf, str(onnx_file), n_test_inputs=3)
        assert len(result) == 2
        passed, delta = result
        assert isinstance(passed, bool)
        assert isinstance(delta, float)


# ---------------------------------------------------------------------------
# OnnxInferenceWrapper
# ---------------------------------------------------------------------------


class TestOnnxInferenceWrapper:
    @_SKIP_FULL
    def test_loads_without_error(self, onnx_file):
        from src.models.onnx_export import OnnxInferenceWrapper

        wrapper = OnnxInferenceWrapper(str(onnx_file))
        assert wrapper is not None

    @_SKIP_FULL
    def test_class_map_loaded_from_metadata(self, onnx_file):
        from src.models.onnx_export import OnnxInferenceWrapper

        wrapper = OnnxInferenceWrapper(str(onnx_file))
        assert set(wrapper.class_map.values()) == {"GasA", "GasB"}

    @_SKIP_FULL
    def test_input_length_set(self, onnx_file):
        from src.models.onnx_export import OnnxInferenceWrapper

        wrapper = OnnxInferenceWrapper(str(onnx_file))
        assert wrapper.input_length == 50

    @_SKIP_FULL
    def test_predict_single_returns_three_values(self, onnx_file):
        from src.models.onnx_export import OnnxInferenceWrapper

        wrapper = OnnxInferenceWrapper(str(onnx_file))
        spectrum = np.random.default_rng(0).standard_normal(50).astype(np.float32)
        result = wrapper.predict_single(spectrum)
        assert len(result) == 3

    @_SKIP_FULL
    def test_predict_single_gas_in_class_map(self, onnx_file):
        from src.models.onnx_export import OnnxInferenceWrapper

        wrapper = OnnxInferenceWrapper(str(onnx_file))
        spectrum = np.ones(50, dtype=np.float32)
        gas, conc, conf = wrapper.predict_single(spectrum)
        assert gas in {"GasA", "GasB"}

    @_SKIP_FULL
    def test_predict_single_confidence_in_range(self, onnx_file):
        from src.models.onnx_export import OnnxInferenceWrapper

        wrapper = OnnxInferenceWrapper(str(onnx_file))
        spectrum = np.random.default_rng(1).standard_normal(50).astype(np.float32)
        _, _, conf = wrapper.predict_single(spectrum)
        assert 0.0 <= conf <= 1.0

    @_SKIP_FULL
    def test_predict_single_resamples_input(self, onnx_file):
        """Wrapper should handle spectra of different length via resampling."""
        from src.models.onnx_export import OnnxInferenceWrapper

        wrapper = OnnxInferenceWrapper(str(onnx_file))
        spectrum = np.ones(200, dtype=np.float32)  # different from input_length=50
        gas, conc, conf = wrapper.predict_single(spectrum)
        assert gas in {"GasA", "GasB"}

    @_SKIP_FULL
    def test_predict_batch_shapes(self, onnx_file):
        from src.models.onnx_export import OnnxInferenceWrapper

        wrapper = OnnxInferenceWrapper(str(onnx_file))
        rng = np.random.default_rng(0)
        batch = rng.standard_normal((4, 50)).astype(np.float32)
        gas_names, concs, confs = wrapper.predict_batch(batch)
        assert len(gas_names) == 4
        assert concs.shape == (4,)
        assert confs.shape == (4,)

    @_SKIP_FULL
    def test_predict_batch_confidences_sum_approx_one_per_sample(self, onnx_file):
        """Each sample's confidence is its max softmax prob — not a sum, just in [0,1]."""
        from src.models.onnx_export import OnnxInferenceWrapper

        wrapper = OnnxInferenceWrapper(str(onnx_file))
        batch = np.random.default_rng(2).standard_normal((3, 50)).astype(np.float32)
        _, _, confs = wrapper.predict_batch(batch)
        assert np.all(confs >= 0.0) and np.all(confs <= 1.0)


# ---------------------------------------------------------------------------
# CLI main()
# ---------------------------------------------------------------------------


class TestOnnxExportCli:
    @_SKIP_ONNX
    def test_cli_exports_file(self, tiny_clf, tmp_path):
        """End-to-end: save checkpoint, call main() via argv, check file exists."""
        import sys

        from src.models.onnx_export import main

        ckpt = tmp_path / "clf.pt"
        tiny_clf.save(str(ckpt))
        out = tmp_path / "clf.onnx"

        sys.argv = ["onnx_export", "--checkpoint", str(ckpt), "--output", str(out), "--opset", "17"]
        try:
            main()
        except SystemExit as s:
            if s.code != 0:
                raise
        assert out.exists()
        assert out.stat().st_size > 0

    @_SKIP_FULL
    def test_cli_validate_flag(self, tiny_clf, tmp_path):
        import sys

        from src.models.onnx_export import main

        ckpt = tmp_path / "clf.pt"
        tiny_clf.save(str(ckpt))
        out = tmp_path / "clf.onnx"

        # Should not raise or exit with error
        sys.argv = ["onnx_export", "--checkpoint", str(ckpt), "--output", str(out), "--validate"]
        try:
            main()
        except SystemExit as s:
            if s.code != 0:
                raise

"""
src.models.onnx_export
======================
ONNX export and validation for :class:`~src.models.cnn.CNNGasClassifier`.

Why ONNX?
---------
ONNX (Open Neural Network Exchange) is the standard interchange format for
deploying neural networks on hardware where PyTorch is unavailable or too
heavy — e.g.:

- **Edge devices** (Raspberry Pi, NVIDIA Jetson) using ONNX Runtime
- **Microcontrollers** via conversion to TFLite or TensorFlow Lite
- **C++ inference** in embedded firmware via onnxruntime-c
- **Multi-framework interop** (TensorFlow, CoreML, TensorRT)

The export traces the model's computation graph with a dummy input tensor of
the correct shape ``(1, 1, input_length)`` and saves the static graph to
``.onnx`` format.  The resulting file runs identically on any ONNX Runtime
version ≥ 1.14.

Public API
----------
- ``export_cnn_to_onnx``     — export CNNGasClassifier → .onnx file
- ``validate_onnx_export``   — verify ONNX outputs match PyTorch outputs
- ``OnnxInferenceWrapper``   — production inference wrapper using onnxruntime
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_cnn_to_onnx(
    classifier,  # CNNGasClassifier
    output_path: str,
    opset_version: int = 17,
    dynamic_batch: bool = True,
) -> Path:
    """Export a fitted :class:`~src.models.cnn.CNNGasClassifier` to ONNX format.

    Parameters
    ----------
    classifier:
        A fitted ``CNNGasClassifier`` instance.
    output_path:
        Destination ``.onnx`` file path.
    opset_version:
        ONNX opset version.  17 is widely supported by ONNX Runtime 1.14+.
        Use 12 for maximum compatibility with older runtimes.
    dynamic_batch:
        If True, the batch dimension is dynamic (recommended for production —
        allows batched inference without re-exporting).

    Returns
    -------
    Path
        Absolute path to the exported ``.onnx`` file.

    Raises
    ------
    RuntimeError
        If the classifier is not fitted.
    ImportError
        If PyTorch is not installed.

    Example
    -------
    ::

        from src.models.cnn import CNNGasClassifier
        from src.models.onnx_export import export_cnn_to_onnx, validate_onnx_export

        clf = CNNGasClassifier.load("output/models/cnn_classifier.pt")
        onnx_path = export_cnn_to_onnx(clf, "output/models/cnn_classifier.onnx")
        ok, delta = validate_onnx_export(clf, onnx_path)
        print(f"Validation passed: {ok}, max delta: {delta:.2e}")
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for ONNX export: pip install torch") from exc

    if not classifier.is_fitted or classifier.model is None:
        raise RuntimeError("CNNGasClassifier must be fitted before ONNX export.")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = classifier.model
    model.eval()

    # Dummy input: batch=1, channel=1, spectral_points=input_length
    dummy = torch.zeros(1, 1, classifier.input_length, device=classifier.device)

    # Dynamic axes config
    dynamic_axes: dict | None = None
    if dynamic_batch:
        dynamic_axes = {
            "spectrum": {0: "batch_size"},
            "cls_logits": {0: "batch_size"},
            "conc_pred": {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        (dummy,),  # type: ignore[arg-type]
        str(out_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["spectrum"],
        output_names=["cls_logits", "conc_pred"],
        dynamic_axes=dynamic_axes,
        verbose=False,
    )

    # Embed class_map metadata as custom properties (ONNX metadata_props)
    try:
        import onnx

        model_proto = onnx.load(str(out_path))
        meta = model_proto.metadata_props.add()
        meta.key = "class_map"
        import json

        meta.value = json.dumps({str(k): v for k, v in classifier.class_map.items()})
        meta2 = model_proto.metadata_props.add()
        meta2.key = "input_length"
        meta2.value = str(classifier.input_length)
        onnx.save(model_proto, str(out_path))
        log.info("ONNX metadata (class_map, input_length) embedded in %s", out_path)
    except ImportError:
        log.info("onnx package not installed — metadata not embedded. pip install onnx")
    except Exception as exc:
        log.debug("ONNX metadata embedding failed: %s", exc)

    size_mb = out_path.stat().st_size / 1024 / 1024
    log.info(
        "CNN exported to ONNX: %s (opset=%d, %.2f MB)",
        out_path,
        opset_version,
        size_mb,
    )
    return out_path.resolve()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_onnx_export(
    classifier,  # CNNGasClassifier
    onnx_path: str,
    n_test_inputs: int = 10,
    atol: float = 1e-4,
) -> tuple[bool, float]:
    """Verify that ONNX outputs numerically match PyTorch outputs.

    Generates *n_test_inputs* random spectra, runs them through both the
    original PyTorch model and the ONNX Runtime session, and checks that
    the maximum absolute difference is below *atol*.

    Parameters
    ----------
    classifier:
        The same CNNGasClassifier used for :func:`export_cnn_to_onnx`.
    onnx_path:
        Path to the exported ``.onnx`` file.
    n_test_inputs:
        Number of random test inputs to compare.
    atol:
        Absolute tolerance for numerical agreement.

    Returns
    -------
    passed : bool
        True if all outputs agree within *atol*.
    max_delta : float
        Maximum absolute difference observed across all test inputs and
        both output heads (classification logits + concentration).
    """
    try:
        import onnxruntime as ort
        import torch
    except ImportError as exc:
        raise ImportError(
            "Both torch and onnxruntime are required for validation. "
            f"Missing: {exc}. Install with: pip install onnxruntime"
        ) from exc

    if not classifier.is_fitted or classifier.model is None:
        raise RuntimeError("Classifier is not fitted.")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    model = classifier.model
    model.eval()

    rng = np.random.default_rng(42)
    max_delta = 0.0

    for _ in range(n_test_inputs):
        # Random normalised spectrum
        raw = rng.standard_normal((1, 1, classifier.input_length)).astype(np.float32)

        # PyTorch
        with torch.no_grad():
            t = torch.from_numpy(raw).to(classifier.device)
            logits_pt, conc_pt = model(t)
            logits_np = logits_pt.cpu().numpy()
            conc_np = conc_pt.cpu().numpy()

        # ONNX Runtime
        ort_inputs = {"spectrum": raw}
        logits_ort, conc_ort = sess.run(["cls_logits", "conc_pred"], ort_inputs)

        delta_cls = float(np.abs(logits_np - logits_ort).max())
        delta_reg = float(np.abs(conc_np - conc_ort).max())
        max_delta = max(max_delta, delta_cls, delta_reg)

    passed = max_delta <= atol
    status = "PASSED" if passed else "FAILED"
    log.info(
        "ONNX validation %s: max_delta=%.2e (atol=%.2e, n=%d)",
        status,
        max_delta,
        atol,
        n_test_inputs,
    )
    return passed, max_delta


# ---------------------------------------------------------------------------
# Production inference wrapper
# ---------------------------------------------------------------------------


class OnnxInferenceWrapper:
    """Lightweight ONNX Runtime wrapper for edge deployment.

    Loads a ``.onnx`` file exported by :func:`export_cnn_to_onnx` and
    provides the same ``predict_single()`` interface as
    :class:`~src.models.cnn.CNNGasClassifier`, but **without requiring
    PyTorch** at runtime — only ``onnxruntime`` and ``numpy`` are needed.

    Parameters
    ----------
    onnx_path:
        Path to the exported ``.onnx`` file.
    class_map:
        Dict mapping integer class index → gas name.  If None, attempts to
        read from ONNX metadata (embedded during export).

    Example
    -------
    ::

        from src.models.onnx_export import OnnxInferenceWrapper
        wrapper = OnnxInferenceWrapper("output/models/cnn_classifier.onnx")
        gas, conc, conf = wrapper.predict_single(my_spectrum)
    """

    def __init__(
        self,
        onnx_path: str,
        class_map: dict[int, str] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for OnnxInferenceWrapper. "
                "Install with: pip install onnxruntime"
            ) from exc

        self._path = Path(onnx_path)
        self._sess = ort.InferenceSession(str(self._path), providers=["CPUExecutionProvider"])

        # Infer input_length from ONNX model shape
        input_info = self._sess.get_inputs()[0]
        self.input_length: int = int(input_info.shape[2]) if len(input_info.shape) >= 3 else 1000

        # Class map: explicit arg > ONNX metadata > fallback integers
        if class_map is not None:
            self.class_map = class_map
        else:
            self.class_map = self._load_class_map_from_metadata()

        log.info(
            "OnnxInferenceWrapper loaded: %s (input_length=%d, classes=%s)",
            self._path.name,
            self.input_length,
            list(self.class_map.values()),
        )

    def _load_class_map_from_metadata(self) -> dict[int, str]:
        try:
            import json

            import onnx

            model_proto = onnx.load(str(self._path))
            for prop in model_proto.metadata_props:
                if prop.key == "class_map":
                    raw = json.loads(prop.value)
                    return {int(k): v for k, v in raw.items()}
        except Exception:
            pass
        return {}

    def _prepare(self, spectrum: np.ndarray) -> np.ndarray:
        """Normalise and reshape spectrum to (1, 1, input_length) float32."""
        arr = np.asarray(spectrum, dtype=np.float32).ravel()
        if arr.size != self.input_length:
            x_old = np.linspace(0, 1, arr.size)
            x_new = np.linspace(0, 1, self.input_length)
            arr = np.interp(x_new, x_old, arr).astype(np.float32)
        mu, sd = arr.mean(), arr.std() + 1e-8
        arr = (arr - mu) / sd
        return np.asarray(arr[np.newaxis, np.newaxis, :], dtype=np.float32)  # (1, 1, L)

    def predict_single(self, spectrum: np.ndarray) -> tuple[str, float, float]:
        """Predict gas type, concentration, and classification confidence.

        Parameters
        ----------
        spectrum:
            Raw intensity array (n_points,).

        Returns
        -------
        gas_name : str
        concentration_ppm : float
        confidence : float
            Softmax probability of the top-1 class.
        """
        x = self._prepare(spectrum)
        logits, conc_pred = self._sess.run(["cls_logits", "conc_pred"], {"spectrum": x})
        logits = logits[0]  # (n_classes,)
        # Softmax
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        gas_name = self.class_map.get(top_idx, str(top_idx))
        concentration = float(conc_pred[0, 0])
        return gas_name, concentration, confidence

    def predict_batch(self, spectra: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Batch inference for multiple spectra.

        Parameters
        ----------
        spectra:
            Shape ``(n_samples, n_points)``.

        Returns
        -------
        gas_names : List[str]
        concentrations : ndarray (n_samples,)
        confidences : ndarray (n_samples,)
        """
        batch = np.stack([self._prepare(row)[0] for row in spectra], axis=0)  # (N, 1, L)
        logits, conc_pred = self._sess.run(["cls_logits", "conc_pred"], {"spectrum": batch})
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        top_indices = probs.argmax(axis=1)
        gas_names = [self.class_map.get(int(i), str(i)) for i in top_indices]
        confidences = probs[np.arange(len(probs)), top_indices]
        return gas_names, conc_pred[:, 0].astype(float), confidences.astype(float)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI: export a fitted CNNGasClassifier checkpoint to ONNX format.

    Usage::

        gas-export-onnx --checkpoint output/models/cnn_classifier.pt \\
                         --output output/models/cnn_classifier.onnx \\
                         --validate
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Export CNNGasClassifier to ONNX for edge deployment"
    )
    parser.add_argument(
        "--checkpoint",
        default="output/models/cnn_classifier.pt",
        help="Path to .pt checkpoint file",
    )
    parser.add_argument(
        "--output",
        default="output/models/cnn_classifier.onnx",
        help="Output .onnx file path",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run numerical validation after export (requires onnxruntime)",
    )
    parser.add_argument(
        "--no-dynamic-batch",
        action="store_true",
        help="Disable dynamic batch axis (fixes batch size to 1)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        log.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    try:
        from src.models.cnn import CNNGasClassifier
    except ImportError:
        log.error("src.models.cnn not importable — check PYTHONPATH or venv.")
        sys.exit(1)

    log.info("Loading checkpoint: %s", ckpt_path)
    clf = CNNGasClassifier.load(str(ckpt_path))

    log.info("Exporting to ONNX (opset %d)...", args.opset)
    onnx_path = export_cnn_to_onnx(
        clf,
        args.output,
        opset_version=args.opset,
        dynamic_batch=not args.no_dynamic_batch,
    )
    log.info("Exported: %s", onnx_path)

    if args.validate:
        log.info("Validating ONNX output against PyTorch...")
        passed, max_delta = validate_onnx_export(clf, str(onnx_path))
        if passed:
            log.info("Validation PASSED (max delta: %.2e)", max_delta)
        else:
            log.error("Validation FAILED (max delta: %.2e > 1e-4)", max_delta)
            sys.exit(1)


if __name__ == "__main__":
    main()

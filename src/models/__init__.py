"""
src.models
==========
ML model definitions, the central model registry, ONNX export, and versioning.

Modules
-------
cnn
    ``GasCNN`` (nn.Module) and ``CNNGasClassifier`` (high-level wrapper).
    Requires PyTorch — import is deferred so non-torch code still works.
registry
    ``ModelRegistry`` — loads CNN / GPR / calibration from ``models/registry/``
    and exposes a unified prediction interface with graceful degradation.
versioning
    ``ModelVersionStore`` — timestamped saves, manifests, promotion, rollback.
    No PyTorch/joblib hard dependency at import time.
onnx_export
    ``export_cnn_to_onnx`` — export fitted CNN to ONNX for edge deployment.
    ``validate_onnx_export`` — verify numerical agreement between PyTorch and ONNX.
    ``OnnxInferenceWrapper`` — production inference without PyTorch (onnxruntime only).
"""

from src.models.registry import ModelRegistry
from src.models.versioning import ModelVersionStore, VersionRecord

__all__ = [
    "ModelRegistry",
    "ModelVersionStore",
    "VersionRecord",
    "export_cnn_to_onnx",
    "get_cnn_classifier",
    "get_onnx_inference_wrapper",
    "validate_onnx_export",
]


# CNNGasClassifier imported lazily to avoid torch hard-dependency at module level
def get_cnn_classifier(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Import and return CNNGasClassifier (deferred to avoid torch dependency)."""
    from src.models.cnn import CNNGasClassifier

    return CNNGasClassifier(*args, **kwargs)


def export_cnn_to_onnx(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Deferred import of export_cnn_to_onnx (requires torch + onnx)."""
    from src.models.onnx_export import export_cnn_to_onnx as _fn

    return _fn(*args, **kwargs)


def validate_onnx_export(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Deferred import of validate_onnx_export (requires torch + onnxruntime)."""
    from src.models.onnx_export import validate_onnx_export as _fn

    return _fn(*args, **kwargs)


def get_onnx_inference_wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Deferred import of OnnxInferenceWrapper (requires onnxruntime only)."""
    from src.models.onnx_export import OnnxInferenceWrapper  # noqa: N813

    return OnnxInferenceWrapper(*args, **kwargs)

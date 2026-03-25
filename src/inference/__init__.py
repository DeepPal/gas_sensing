"""
src.inference
=============
Real-time inference layer for the Au-MIP LSPR gas sensing platform.

Modules
-------
realtime_pipeline
    Four-stage signal processing pipeline: preprocessing → feature extraction
    → calibration → quality scoring.  Entry point: ``RealTimePipeline``.
live_state
    Thread-safe circular buffer (``LiveDataStore`` singleton) shared between
    the acquisition thread and the Streamlit dashboard process.
orchestrator
    ``SensorOrchestrator`` — wires hardware acquisition, pipeline, model
    registry, live store, and disk writer into a single session lifecycle.
"""

from src.inference.live_state import LiveDataStore, _LiveDataStore
from src.inference.realtime_pipeline import (
    PipelineConfig,
    PipelineResult,
    RealTimePipeline,
    SpectrumData,
)

__all__ = [
    "RealTimePipeline",
    "PipelineConfig",
    "PipelineResult",
    "SpectrumData",
    "LiveDataStore",
    "_LiveDataStore",
]

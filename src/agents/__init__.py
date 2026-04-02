"""
src.agents
==========
Autonomous monitoring agents for the the sensor LSPR gas sensing platform.

Each agent runs independently and can be composed into an agentic pipeline
(see ``md_specs/AGENTS.md`` for the full architecture).

Agents
------
drift
    ``DriftDetectionAgent`` — monitors baseline peak wavelength for slow
    sensor drift (slope > threshold nm/min or absolute offset > threshold nm).
quality
    ``DataQualityAgent`` — gates incoming spectra through saturation,
    finite-value, SNR, and low-signal checks before any expensive processing.
training
    ``TrainingAgent`` — autonomous closed-loop model retraining; triggered by
    drift alerts, performance degradation, or sample volume thresholds.
"""

from src.agents.drift import DriftAlert, DriftAlertType, DriftDetectionAgent
from src.agents.quality import DataQualityAgent, QualityCode, QualityResult
from src.agents.training import RetrainResult, RetrainTrigger, TrainingAgent

__all__ = [
    "DriftDetectionAgent",
    "DriftAlert",
    "DriftAlertType",
    "DataQualityAgent",
    "QualityResult",
    "QualityCode",
    "TrainingAgent",
    "RetrainTrigger",
    "RetrainResult",
]

"""
spectraagent.knowledge
=======================
Domain knowledge base for optical gas sensors.

This package provides structured, physics-grounded knowledge that is injected
into Claude agent prompts at runtime — enabling agents to reason like a domain
expert rather than relying on generic LLM training data.

Sub-modules
-----------
analytes
    Reference data for target VOC analytes: molecular properties, expected
    sensor response ranges, literature LOD/LOQ values, and known interferents.

failure_modes
    Taxonomy of LSPR sensor failure modes with physical mechanisms,
    drift signatures, and corrective actions.

protocols
    ICH Q2(R1) validation protocol specifications and acceptance criteria
    for analytical method validation of LSPR gas sensors.

context_builders
    Functions that select and format relevant knowledge for agent prompts,
    adapting the depth and focus based on the specific measurement context.
"""
from spectraagent.knowledge.analytes import AnalyteProperties, ANALYTE_REGISTRY, lookup_analyte
from spectraagent.knowledge.failure_modes import FailureMode, FAILURE_TAXONOMY, match_failure_modes
from spectraagent.knowledge.protocols import ValidationRequirement, ICH_Q2_PROTOCOL, ValidationTracker, ValidationStatus
from spectraagent.knowledge.context_builders import (
    build_anomaly_context,
    build_calibration_narration_context,
    build_report_context,
    build_hardware_diagnostics_context,
    build_sensor_physics_preamble,
)

__all__ = [
    "AnalyteProperties",
    "ANALYTE_REGISTRY",
    "lookup_analyte",
    "FailureMode",
    "FAILURE_TAXONOMY",
    "match_failure_modes",
    "ValidationRequirement",
    "ICH_Q2_PROTOCOL",
    "ValidationTracker",
    "ValidationStatus",
    "build_anomaly_context",
    "build_calibration_narration_context",
    "build_report_context",
    "build_hardware_diagnostics_context",
    "build_sensor_physics_preamble",
]

"""
src.schemas.spectrum
====================
Pydantic data contracts for spectral readings and predictions.

These are the **canonical data structures** that flow between every layer
of the pipeline.  All other modules must accept/return these types — never
raw dicts — so type errors are caught at the boundary, not buried in logic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Annotated
import uuid

if TYPE_CHECKING:
    import numpy as np

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Vocabulary constants
# ---------------------------------------------------------------------------

KNOWN_GAS_TYPES: frozenset[str] = frozenset({"Ethanol", "IPA", "Methanol", "MixVOC", "unknown"})

# Normalise alternate spellings at ingest
_GAS_ALIASES: dict[str, str] = {
    "etoh": "Ethanol",
    "ethanol": "Ethanol",
    "ipa": "IPA",
    "isopropanol": "IPA",
    "isopropyl": "IPA",
    "meoh": "Methanol",
    "methanol": "Methanol",
    "mixvoc": "MixVOC",
    "mix": "MixVOC",
    "mixed": "MixVOC",
    "unknown": "unknown",
}


def normalise_gas_type(raw: str) -> str:
    """Return the canonical gas-type string for *raw*, or *raw* if unknown."""
    return _GAS_ALIASES.get(raw.lower().strip(), raw)


# ---------------------------------------------------------------------------
# SpectrumReading — one spectrum acquisition event
# ---------------------------------------------------------------------------


class SpectrumReading(BaseModel):
    """A single raw spectrum acquisition, with all context needed for processing.

    Required fields carry the minimum information for any pipeline stage.
    Optional fields are nullable and default to ``None`` — the system works
    without them; they improve calibration accuracy when available.
    """

    model_config = ConfigDict(extra="forbid", frozen=False)

    # Identity
    spectrum_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier (UUID4) for this spectrum.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC acquisition timestamp.",
    )

    # Spectral data (required)
    wavelengths: list[float] = Field(
        ...,
        min_length=4,
        description="Wavelength axis in nanometres (monotonically increasing, ≥ 4 points).",
    )
    intensities: list[float] = Field(
        ...,
        min_length=4,
        description="Measured intensities (same length as wavelengths, ≥ 4 points).",
    )

    # Sensor identity (required)
    sensor_id: str = Field(
        default="unknown",
        description="Sensor identifier, e.g. 'CCS200-LAB-01' or 'simulation'.",
    )

    # Ground-truth labels (required; use defaults for inference mode)
    gas_type: str = Field(
        default="unknown",
        description="Gas analyte label (normalised to KNOWN_GAS_TYPES vocabulary).",
    )
    concentration_ppm: Annotated[float, Field(ge=0.0)] = Field(
        default=0.0,
        description="Ground-truth concentration in ppm; 0.0 when unknown.",
    )

    # Optional environmental context
    temperature_c: float | None = Field(
        default=None,
        description="Ambient temperature in °C at acquisition time.",
    )
    humidity_pct: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Relative humidity percentage at acquisition time.",
    )

    # Optional reference / dark spectra
    reference_spectrum: list[float] | None = Field(
        default=None,
        description="Reference (baseline) intensities, same length as intensities.",
    )
    dark_spectrum: list[float] | None = Field(
        default=None,
        description="Dark-current spectrum (no light), same length as intensities.",
    )
    integration_time_ms: float | None = Field(
        default=None,
        gt=0.0,
        description="CCS200 integration time in milliseconds.",
    )

    # ---------------------------------------------------------------------------
    # Validators
    # ---------------------------------------------------------------------------

    @field_validator("gas_type", mode="before")
    @classmethod
    def normalise_gas(cls, v: str) -> str:
        return normalise_gas_type(v)

    @model_validator(mode="after")
    def check_array_lengths(self) -> SpectrumReading:
        n = len(self.wavelengths)
        if len(self.intensities) != n:
            raise ValueError(
                f"wavelengths ({n}) and intensities ({len(self.intensities)}) "
                "must have the same length."
            )
        if self.reference_spectrum is not None and len(self.reference_spectrum) != n:
            raise ValueError(
                f"reference_spectrum length ({len(self.reference_spectrum)}) "
                f"must match wavelengths ({n})."
            )
        if self.dark_spectrum is not None and len(self.dark_spectrum) != n:
            raise ValueError(
                f"dark_spectrum length ({len(self.dark_spectrum)}) must match wavelengths ({n})."
            )
        return self

    @model_validator(mode="after")
    def check_wavelengths_monotonic(self) -> SpectrumReading:
        wl = self.wavelengths
        if len(wl) >= 2:
            diffs = [wl[i + 1] - wl[i] for i in range(len(wl) - 1)]
            if any(d <= 0 for d in diffs):
                raise ValueError("wavelengths must be strictly monotonically increasing.")
        return self

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    @property
    def n_points(self) -> int:
        """Number of spectral data points."""
        return len(self.wavelengths)

    @property
    def wavelength_range(self) -> tuple[float, float]:
        """(min_nm, max_nm) of the wavelength axis."""
        return (self.wavelengths[0], self.wavelengths[-1])

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[return]
        """Return ``(wavelengths, intensities)`` as numpy arrays."""
        import numpy as np

        return np.asarray(self.wavelengths, dtype=np.float64), np.asarray(
            self.intensities, dtype=np.float64
        )


# ---------------------------------------------------------------------------
# PredictionResult — output of one pipeline run
# ---------------------------------------------------------------------------


class PredictionResult(BaseModel):
    """Full output of the real-time pipeline for a single spectrum.

    Pairs with a ``SpectrumReading`` via ``spectrum_id``.
    """

    model_config = ConfigDict(extra="forbid", frozen=False)

    # Identity linkage
    spectrum_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Physical measurement
    peak_wavelength: float | None = Field(
        default=None,
        description="Detected LSPR peak wavelength in nm.",
    )
    wavelength_shift_nm: float | None = Field(
        default=None,
        description="Δλ = λ_gas − λ_reference in nm (negative on adsorption).",
    )

    # Concentration estimate
    concentration_ppm: float | None = Field(
        default=None,
        description="Estimated analyte concentration in ppm.",
    )
    concentration_std_ppm: float | None = Field(
        default=None,
        ge=0.0,
        description="One-sigma uncertainty from GPR (ppm); None if GPR not loaded.",
    )

    # Classification (CNN)
    gas_type_predicted: str | None = Field(
        default=None,
        description="Predicted gas type from CNN classifier.",
    )
    gas_type_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Softmax confidence of the top-1 CNN prediction.",
    )

    # Quality metrics
    snr: float | None = Field(default=None, ge=0.0)
    quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall quality score [0, 1]; < 0.5 = flag for review.",
    )

    # Pipeline metadata
    success: bool = Field(default=False)
    processing_time_ms: float = Field(default=0.0, ge=0.0)
    model_version: str = Field(default="none")
    pipeline_version: str = Field(default="3.0.0")
    error_message: str | None = Field(default=None)

    @classmethod
    def failure(cls, spectrum_id: str, reason: str) -> PredictionResult:
        """Convenience constructor for a failed-pipeline result."""
        return cls(
            spectrum_id=spectrum_id,
            success=False,
            quality_score=0.0,
            error_message=reason,
        )

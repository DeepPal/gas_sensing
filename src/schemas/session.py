"""
src.schemas.session
===================
Pydantic schema for session-level metadata (a group of spectra from one experiment run).
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


class SessionMeta(BaseModel):
    """Metadata for one acquisition session."""

    model_config = ConfigDict(extra="forbid", frozen=False)

    session_id: str = Field(..., description="Format: YYYYMMDD_HHMMSS")
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stopped_at: datetime | None = None

    gas_type: str = Field(default="unknown")
    sensor_id: str = Field(default="unknown")

    total_spectra: int = Field(default=0, ge=0)
    valid_spectra: int = Field(default=0, ge=0)

    model_version: str = Field(default="none")
    config_hash: str | None = Field(
        default=None,
        description="SHA-256 hash of config.yaml used for this session.",
    )
    notes: str = Field(default="")

    @property
    def duration_seconds(self) -> float | None:
        if self.stopped_at is None:
            return None
        try:
            return (self.stopped_at - self.started_at).total_seconds()
        except TypeError:
            # Naive/aware datetime mismatch — return None rather than crashing
            return None

    @property
    def valid_rate(self) -> float:
        if self.total_spectra == 0:
            return 0.0
        return self.valid_spectra / self.total_spectra

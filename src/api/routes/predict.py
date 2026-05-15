"""
src.api.routes.predict
=======================
Inference endpoint.

POST /predict
    Input:  SpectrumReading (Pydantic)
    Output: PredictionResult (Pydantic)

The route validates the input via Pydantic, passes raw arrays to
``RealTimePipeline.process_spectrum()``, and marshals the result into
a ``PredictionResult`` response.

Error codes
-----------
200  Successful prediction (check ``success`` field for pipeline quality)
422  Pydantic validation error (bad input shape, type mismatch, etc.)
503  Pipeline not yet initialised
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_pipeline, get_version_store
from src.schemas.spectrum import PredictionResult, SpectrumReading


def _resolve_model_version(pipeline: object, version_store: object | None) -> str:
    """Return the promoted model version ID, falling back to the pipeline's static version."""
    if version_store is not None:
        try:
            vid = version_store.active_version("pipeline")
            if vid:
                return vid
        except Exception:
            pass
    return getattr(pipeline, "_VERSION", "3.0.0")

router = APIRouter(tags=["inference"])


@router.post(
    "/predict",
    response_model=PredictionResult,
    summary="Run the real-time inference pipeline on one spectrum",
)
async def predict(
    reading: SpectrumReading,
    pipeline=Depends(get_pipeline),
    version_store=Depends(get_version_store),
) -> PredictionResult:
    """Process one spectrum through the 4-stage pipeline and return predictions.

    The input is validated by Pydantic before reaching this handler — array
    length mismatches and non-finite values are caught automatically.

    Returns a ``PredictionResult`` with ``success=True`` on a clean prediction.
    ``success=False`` indicates a pipeline-level issue (e.g. saturation) but
    the response is still HTTP 200 — callers should check ``success``.
    """
    t0 = time.perf_counter()

    wl, raw = reading.to_numpy()

    # Run pipeline
    result = pipeline.process_spectrum(
        wavelengths=wl,
        intensities=raw,
        timestamp=reading.timestamp,
        sample_id=reading.spectrum_id,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    sp = result.spectrum

    return PredictionResult(
        spectrum_id=reading.spectrum_id,
        timestamp=reading.timestamp,
        peak_wavelength=sp.peak_wavelength,
        wavelength_shift_nm=sp.wavelength_shift,
        concentration_ppm=sp.concentration_ppm,
        concentration_std_ppm=sp.concentration_std_ppm or sp.gpr_uncertainty,
        gas_type_predicted=sp.gas_type,
        gas_type_confidence=sp.confidence_score if sp.confidence_score > 0 else None,
        snr=sp.snr,
        quality_score=sp.quality_score,
        success=result.success,
        processing_time_ms=elapsed_ms,
        model_version=_resolve_model_version(pipeline, version_store),
        error_message="; ".join(result.errors) if result.errors else None,
    )


@router.post(
    "/predict/batch",
    summary="Run inference on multiple spectra",
)
async def predict_batch(
    readings: list[SpectrumReading],
    pipeline=Depends(get_pipeline),
    version_store=Depends(get_version_store),
) -> list[PredictionResult]:
    """Process a list of spectra.  Useful for batch re-scoring.

    Limit: 1000 spectra per request.
    """
    if len(readings) > 1000:
        raise HTTPException(
            status_code=422,
            detail="Batch size exceeds maximum of 1000 spectra per request.",
        )

    results = []
    for reading in readings:
        wl, raw = reading.to_numpy()
        result = pipeline.process_spectrum(wl, raw, reading.timestamp, reading.spectrum_id)
        sp = result.spectrum
        results.append(
            PredictionResult(
                spectrum_id=reading.spectrum_id,
                timestamp=reading.timestamp,
                peak_wavelength=sp.peak_wavelength,
                wavelength_shift_nm=sp.wavelength_shift,
                concentration_ppm=sp.concentration_ppm,
                concentration_std_ppm=sp.concentration_std_ppm,
                gas_type_predicted=sp.gas_type,
                gas_type_confidence=sp.confidence_score if sp.confidence_score > 0 else None,
                snr=sp.snr,
                quality_score=sp.quality_score,
                success=result.success,
                processing_time_ms=result.processing_time_ms,
                model_version=_resolve_model_version(pipeline, version_store),
                error_message="; ".join(result.errors) if result.errors else None,
            )
        )
    return results

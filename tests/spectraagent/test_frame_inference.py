"""Tests that _process_acquired_frame calls RealTimePipeline when wired."""
import asyncio
import json
from types import SimpleNamespace
from typing import cast

from fastapi import FastAPI
import numpy as np

from spectraagent.__main__ import _process_acquired_frame


class _RecordingBroadcaster:
    def __init__(self):
        self.messages: list[str] = []

    async def broadcast(self, msg: str) -> None:
        self.messages.append(msg)


class _QualityPass:
    def process(self, frame_num, wl_np, intensities):
        return True


class _MockPipelineResult:
    def __init__(self):
        self.success = True
        from src.inference.realtime_pipeline import SpectrumData
        wl = np.linspace(300, 1000, 3648)
        self.spectrum = SpectrumData(wavelengths=wl, intensities=np.ones(3648))
        self.spectrum.concentration_ppm = 2.5
        self.spectrum.ci_low = 2.1
        self.spectrum.ci_high = 2.9
        self.spectrum.wavelength_shift = -1.8
        self.spectrum.snr = 12.0


class _MockPipeline:
    def __init__(self):
        self.calls = []

    def process_spectrum(self, wl, intensities, timestamp=None, sample_id=None):
        self.calls.append((wl, intensities))
        return _MockPipelineResult()


def _make_app(loop):
    pipeline = _MockPipeline()
    state = SimpleNamespace(
        asyncio_loop=loop,
        spectrum_bc=_RecordingBroadcaster(),
        quality_agent=_QualityPass(),
        drift_agent=None,
        plugin=None,
        session_running=True,
        session_frame_count=0,
        latest_spectrum=None,
        pipeline=pipeline,
    )
    return SimpleNamespace(state=state)


def test_pipeline_is_called_when_wired():
    """_process_acquired_frame must call app.state.pipeline.process_spectrum."""
    loop = asyncio.new_event_loop()
    try:
        app = _make_app(loop)
        wl = np.linspace(300, 1000, 3648)
        intensities = np.random.rand(3648)
        _process_acquired_frame(cast(FastAPI, app), wl.tolist(), wl, 1, intensities)
        loop.run_until_complete(asyncio.sleep(0))
        assert len(app.state.pipeline.calls) == 1
    finally:
        loop.close()


def test_broadcast_includes_concentration_when_pipeline_runs():
    """WebSocket broadcast must contain concentration_ppm when pipeline succeeds."""
    loop = asyncio.new_event_loop()
    try:
        app = _make_app(loop)
        wl = np.linspace(300, 1000, 3648)
        intensities = np.random.rand(3648)
        _process_acquired_frame(cast(FastAPI, app), wl.tolist(), wl, 1, intensities)
        loop.run_until_complete(asyncio.sleep(0))
        assert len(app.state.spectrum_bc.messages) == 1
        payload = json.loads(app.state.spectrum_bc.messages[0])
        assert "concentration_ppm" in payload
        assert "ci_low" in payload
        assert "ci_high" in payload
    finally:
        loop.close()


def test_frame_works_without_pipeline():
    """_process_acquired_frame must still broadcast when no pipeline is wired."""
    loop = asyncio.new_event_loop()
    try:
        app = _make_app(loop)
        del app.state.pipeline  # remove pipeline
        wl = np.linspace(300, 1000, 3648)
        intensities = np.random.rand(3648)
        result = _process_acquired_frame(cast(FastAPI, app), wl.tolist(), wl, 1, intensities)
        loop.run_until_complete(asyncio.sleep(0))
        assert result is True
    finally:
        loop.close()

import asyncio
import json
from types import SimpleNamespace
from typing import cast

from fastapi import FastAPI
import numpy as np

from spectraagent.__main__ import _process_acquired_frame


class _RecordingBroadcaster:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def broadcast(self, message: str) -> None:
        self.messages.append(message)


class _QualityPass:
    def process(self, frame_num, wl_np, intensities):
        return True


class _QualityBlock:
    def process(self, frame_num, wl_np, intensities):
        return False


class _DriftRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[int, float]] = []

    def update(self, frame_num: int, peak_wavelength: float) -> None:
        self.calls.append((frame_num, peak_wavelength))


class _PeakPlugin:
    def __init__(self, peak: float | None = 720.0) -> None:
        self.peak = peak

    def detect_peak(self, wl_np, intensities):
        return self.peak


class _FailingPlugin:
    def detect_peak(self, wl_np, intensities):
        raise RuntimeError("peak detection failed")


def _make_app(loop: asyncio.AbstractEventLoop):
    state = SimpleNamespace(
        asyncio_loop=loop,
        spectrum_bc=_RecordingBroadcaster(),
        quality_agent=_QualityPass(),
        drift_agent=_DriftRecorder(),
        plugin=_PeakPlugin(720.5),
        session_running=True,
        session_frame_count=0,
        latest_spectrum=None,
    )
    return SimpleNamespace(state=state)


def test_process_acquired_frame_updates_latest_spectrum_and_frame_count():
    loop = asyncio.new_event_loop()
    try:
        app = _make_app(loop)

        wl = np.array([500.0, 600.0, 700.0])
        intensities = np.array([0.1, 0.2, 0.3])

        processed = _process_acquired_frame(cast(FastAPI, app), wl.tolist(), wl, 3, intensities)
        loop.run_until_complete(asyncio.sleep(0))

        assert processed is True
        assert app.state.latest_spectrum is not None
        assert app.state.latest_spectrum["wl"] == [500.0, 600.0, 700.0]
        assert np.array_equal(app.state.latest_spectrum["intensities"], intensities)
        assert app.state.session_frame_count == 1
        assert app.state.drift_agent.calls == [(3, 720.5)]

        assert len(app.state.spectrum_bc.messages) == 1
        payload = json.loads(app.state.spectrum_bc.messages[0])
        assert payload["wl"] == [500.0, 600.0, 700.0]
        assert payload["i"] == [0.1, 0.2, 0.3]
    finally:
        loop.close()


def test_process_acquired_frame_blocks_broadcast_when_quality_fails():
    loop = asyncio.new_event_loop()
    try:
        app = _make_app(loop)
        app.state.quality_agent = _QualityBlock()
        app.state.plugin = _PeakPlugin(719.8)

        wl = np.array([500.0, 600.0, 700.0])
        intensities = np.array([0.5, 0.4, 0.3])

        processed = _process_acquired_frame(cast(FastAPI, app), wl.tolist(), wl, 8, intensities)
        loop.run_until_complete(asyncio.sleep(0))

        assert processed is False
        assert app.state.session_frame_count == 1
        assert app.state.latest_spectrum is not None
        assert app.state.drift_agent.calls == []
        assert app.state.spectrum_bc.messages == []
    finally:
        loop.close()


def test_process_acquired_frame_continues_when_peak_detection_fails():
    loop = asyncio.new_event_loop()
    try:
        app = _make_app(loop)
        app.state.plugin = _FailingPlugin()

        wl = np.array([500.0, 600.0, 700.0])
        intensities = np.array([0.3, 0.2, 0.1])

        processed = _process_acquired_frame(cast(FastAPI, app), wl.tolist(), wl, 11, intensities)
        loop.run_until_complete(asyncio.sleep(0))

        assert processed is True
        assert app.state.session_frame_count == 1
        assert app.state.drift_agent.calls == []
        assert len(app.state.spectrum_bc.messages) == 1
    finally:
        loop.close()


def test_process_acquired_frame_allows_missing_broadcaster():
    loop = asyncio.new_event_loop()
    try:
        app = _make_app(loop)
        app.state.spectrum_bc = None

        wl = np.array([500.0, 600.0, 700.0])
        intensities = np.array([0.7, 0.6, 0.5])

        processed = _process_acquired_frame(cast(FastAPI, app), wl.tolist(), wl, 12, intensities)
        loop.run_until_complete(asyncio.sleep(0))

        assert processed is True
        assert app.state.session_frame_count == 1
        assert app.state.drift_agent.calls == [(12, 720.5)]
    finally:
        loop.close()

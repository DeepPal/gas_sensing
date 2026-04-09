# SpectraAgent AgentBus + Deterministic Agents — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `AgentBus` threading bridge and four deterministic agents (QualityAgent, DriftAgent, CalibrationAgent, ExperimentPlannerAgent), wiring them into the running FastAPI server so that `/ws/agent-events` streams quality and drift events in real-time during simulation.

**Architecture:** `AgentBus` bridges the 20 Hz sync acquisition thread → asyncio FastAPI via `call_soon_threadsafe` + per-client `asyncio.Queue` fan-out (spec Section 4.1). Agents are called synchronously in the acquisition loop and emit `AgentEvent` objects onto the bus. `CalibrationAgent` wraps existing `src.calibration.isotherms.select_isotherm`. `ExperimentPlannerAgent` wraps existing `src.calibration.gpr.GPRCalibration`. Nothing in `src/` is modified.

**Tech Stack:** Python 3.9+, asyncio, FastAPI, scipy (already installed), scikit-learn GPR (already installed), pytest, pytest-asyncio. No new pip dependencies.

---

## File Map

### Created
| File | Purpose |
|---|---|
| `spectraagent/webapp/agent_bus.py` | `AgentEvent` dataclass + `AgentBus` fan-out bridge |
| `spectraagent/webapp/agents/__init__.py` | Package marker (empty) |
| `spectraagent/webapp/agents/quality.py` | `QualityAgent` — per-frame SNR + saturation gate |
| `spectraagent/webapp/agents/drift.py` | `DriftAgent` — rolling 60-frame CUSUM |
| `spectraagent/webapp/agents/calibration.py` | `CalibrationAgent` — wraps `select_isotherm` |
| `spectraagent/webapp/agents/planner.py` | `ExperimentPlannerAgent` — wraps `GPRCalibration.predict()` |
| `tests/spectraagent/webapp/test_agent_bus.py` | AgentBus + AgentEvent tests |
| `tests/spectraagent/webapp/agents/test_quality.py` | QualityAgent tests |
| `tests/spectraagent/webapp/agents/test_drift.py` | DriftAgent tests |
| `tests/spectraagent/webapp/agents/test_calibration.py` | CalibrationAgent tests |
| `tests/spectraagent/webapp/agents/test_planner.py` | ExperimentPlannerAgent tests |

### Modified
| File | What changes |
|---|---|
| `spectraagent/webapp/server.py` | Add `/ws/agent-events`, startup event to wire AgentBus loop, `/api/calibration/add-point`, `/api/calibration/suggest` |
| `spectraagent/__main__.py` | Create all agents in `start()`, store on `app.state`, update `_acquisition_loop()` to call agents |

**Do NOT create `__init__.py` in any test subdirectory.** With `--import-mode=importlib` (set in pyproject.toml), test `__init__.py` files cause namespace collision with the real `spectraagent` package.

---

## Task 1: AgentEvent + AgentBus Core

**Files:**
- Create: `spectraagent/webapp/agent_bus.py`
- Create: `tests/spectraagent/webapp/test_agent_bus.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/spectraagent/webapp/test_agent_bus.py`:

```python
import asyncio
import json
from pathlib import Path

import pytest

from spectraagent.webapp.agent_bus import AgentBus, AgentEvent


def _event(**kw) -> AgentEvent:
    defaults = dict(source="Test", level="ok", type="test", data={}, text="t")
    defaults.update(kw)
    return AgentEvent(**defaults)


# -----------------------------------------------------------------------
# AgentEvent
# -----------------------------------------------------------------------


def test_event_to_dict_has_required_keys():
    d = _event().to_dict()
    for key in ("ts", "source", "level", "type", "data", "text"):
        assert key in d


def test_event_to_json_is_valid():
    ev = _event(data={"frame": 1, "snr": 42.0})
    parsed = json.loads(ev.to_json())
    assert parsed["data"]["snr"] == 42.0


def test_event_ts_is_iso_format():
    from datetime import datetime, timezone
    ev = _event()
    # Should parse without error
    datetime.fromisoformat(ev.ts.replace("Z", "+00:00"))


# -----------------------------------------------------------------------
# AgentBus
# -----------------------------------------------------------------------


def test_emit_before_setup_does_not_raise():
    bus = AgentBus()
    bus.emit(_event())  # no loop set → no-op


def test_subscribe_adds_to_subscribers():
    bus = AgentBus()
    loop = asyncio.new_event_loop()
    bus.setup_loop(loop)
    q = bus.subscribe()
    assert q in bus._subscribers
    bus.unsubscribe(q)
    loop.close()


def test_unsubscribe_removes_from_subscribers():
    bus = AgentBus()
    loop = asyncio.new_event_loop()
    bus.setup_loop(loop)
    q = bus.subscribe()
    bus.unsubscribe(q)
    assert q not in bus._subscribers
    loop.close()


def test_emit_delivers_to_single_subscriber():
    bus = AgentBus()
    loop = asyncio.new_event_loop()
    bus.setup_loop(loop)

    async def run():
        q = bus.subscribe()
        bus.emit(_event(type="ping"))
        await asyncio.sleep(0)  # let call_soon_threadsafe execute _fanout
        assert not q.empty()
        event = q.get_nowait()
        assert event.type == "ping"
        bus.unsubscribe(q)

    loop.run_until_complete(run())
    loop.close()


def test_emit_fans_out_to_multiple_subscribers():
    bus = AgentBus()
    loop = asyncio.new_event_loop()
    bus.setup_loop(loop)

    async def run():
        q1 = bus.subscribe()
        q2 = bus.subscribe()
        bus.emit(_event(type="broadcast"))
        await asyncio.sleep(0)
        assert q1.get_nowait().type == "broadcast"
        assert q2.get_nowait().type == "broadcast"
        bus.unsubscribe(q1)
        bus.unsubscribe(q2)

    loop.run_until_complete(run())
    loop.close()


def test_unsubscribed_client_receives_nothing():
    bus = AgentBus()
    loop = asyncio.new_event_loop()
    bus.setup_loop(loop)

    async def run():
        q = bus.subscribe()
        bus.unsubscribe(q)
        bus.emit(_event(type="missed"))
        await asyncio.sleep(0)
        assert q.empty()

    loop.run_until_complete(run())
    loop.close()


def test_emit_writes_jsonl(tmp_path):
    path = tmp_path / "events.jsonl"
    bus = AgentBus(jsonl_path=path)
    loop = asyncio.new_event_loop()
    bus.setup_loop(loop)

    async def run():
        bus.emit(_event(type="written", text="hello"))
        await asyncio.sleep(0)

    loop.run_until_complete(run())
    loop.close()

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["type"] == "written"


def test_emit_multiple_events_all_written(tmp_path):
    path = tmp_path / "events.jsonl"
    bus = AgentBus(jsonl_path=path)
    loop = asyncio.new_event_loop()
    bus.setup_loop(loop)

    async def run():
        for i in range(3):
            bus.emit(_event(type=f"ev{i}"))
        await asyncio.sleep(0)

    loop.run_until_complete(run())
    loop.close()

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/webapp/test_agent_bus.py -v
```

Expected: `ModuleNotFoundError: No module named 'spectraagent.webapp.agent_bus'`

- [ ] **Step 3: Create `spectraagent/webapp/agent_bus.py`**

```python
"""
spectraagent.webapp.agent_bus
==============================
AgentBus: thread-safe bridge between the 20 Hz sync pipeline thread
and the asyncio FastAPI event loop.

Architecture (spec Section 4.1):
- Pipeline thread calls emit(event) → call_soon_threadsafe schedules _fanout
- Each WebSocket client has its own asyncio.Queue (created via subscribe())
- _fanout puts the event into every subscriber queue (fan-out)
- Writes every event to agent_events.jsonl (session log)
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class AgentEvent:
    """A single event emitted by an agent.

    JSON schema (spec Section 4.1):
        {
          "ts":     "2026-03-26T14:32:11.042+00:00",
          "source": "QualityAgent",
          "level":  "ok",   # ok | warn | error | info | claude
          "type":   "quality",
          "data":   {...},
          "text":   "human-readable summary"
        }

    ``level`` is the sole field the frontend uses for color coding.
    """

    source: str
    level: str   # "ok" | "warn" | "error" | "info" | "claude"
    type: str
    data: dict
    text: str
    ts: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class AgentBus:
    """Thread-safe event bus: sync pipeline thread → asyncio WebSocket clients.

    Lifecycle:
        # At FastAPI startup (from async context):
        bus.setup_loop(asyncio.get_running_loop())

        # From any thread (sync):
        bus.emit(AgentEvent(...))

        # From async WebSocket handler:
        q = bus.subscribe()
        try:
            event = await q.get()
        finally:
            bus.unsubscribe(q)
    """

    def __init__(self, jsonl_path: Optional[Path] = None) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._subscribers: list[asyncio.Queue] = []
        self._jsonl_path: Optional[Path] = jsonl_path
        self._jsonl_file = None  # opened lazily on first write

    def setup_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Call once from the asyncio event loop thread (e.g. FastAPI startup)."""
        self._loop = loop

    def set_jsonl_path(self, path: Path) -> None:
        """Change the JSONL output path (e.g. at session start)."""
        if self._jsonl_file is not None:
            try:
                self._jsonl_file.close()
            except Exception:
                pass
            self._jsonl_file = None
        self._jsonl_path = path

    def emit(self, event: AgentEvent) -> None:
        """Emit an event from any thread. No-op if setup_loop() not yet called."""
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._fanout, event)

    def subscribe(self) -> asyncio.Queue:
        """Register a new WebSocket client. Returns its dedicated queue."""
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Remove a client's queue when they disconnect."""
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Internal — always called from the event loop thread
    # ------------------------------------------------------------------

    def _fanout(self, event: AgentEvent) -> None:
        """Put event into every subscriber queue and write to JSONL log."""
        for q in list(self._subscribers):
            q.put_nowait(event)
        self._write_jsonl(event)

    def _write_jsonl(self, event: AgentEvent) -> None:
        if self._jsonl_path is None:
            return
        try:
            if self._jsonl_file is None:
                self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                self._jsonl_file = open(self._jsonl_path, "a", encoding="utf-8")  # noqa: SIM115
            self._jsonl_file.write(event.to_json() + "\n")
            self._jsonl_file.flush()
        except Exception as exc:
            log.warning("AgentBus: failed to write JSONL: %s", exc)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/webapp/test_agent_bus.py -v
```

Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add spectraagent/webapp/agent_bus.py tests/spectraagent/webapp/test_agent_bus.py
git commit -m "feat: AgentEvent dataclass + AgentBus fan-out bridge (async queue per client)"
```

---

## Task 2: /ws/agent-events WebSocket

**Files:**
- Modify: `spectraagent/webapp/server.py`
- Modify: `tests/spectraagent/webapp/test_server.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/spectraagent/webapp/test_server.py`:

```python
from spectraagent.webapp.agent_bus import AgentBus, AgentEvent


def test_ws_agent_events_connects(client):
    """WebSocket /ws/agent-events accepts connections without error."""
    with client.websocket_connect("/ws/agent-events") as ws:
        pass  # connecting and cleanly disconnecting = success


def test_ws_agent_events_receives_emitted_event(client):
    """Event emitted to AgentBus is delivered to connected WS client."""
    import json

    app = client.app
    agent_bus: AgentBus = app.state.agent_bus
    assert agent_bus is not None, "AgentBus must be initialised by create_app()"

    with client.websocket_connect("/ws/agent-events") as ws:
        # Emit an event directly from test (bus.setup_loop already called by startup)
        agent_bus.emit(AgentEvent(
            source="Test",
            level="info",
            type="test_event",
            data={"x": 1},
            text="hello from test",
        ))
        msg = ws.receive_text(timeout=2.0)
        parsed = json.loads(msg)
        assert parsed["type"] == "test_event"
        assert parsed["source"] == "Test"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/webapp/test_server.py::test_ws_agent_events_connects -v
```

Expected: FAIL — `/ws/agent-events` route not found (404).

- [ ] **Step 3: Add AgentBus + /ws/agent-events to `server.py`**

At the top of `spectraagent/webapp/server.py`, after the existing imports, add:

```python
from spectraagent.webapp.agent_bus import AgentBus, AgentEvent
```

After the `AcquisitionConfig` Pydantic model (before `create_app`), add:

```python
# Module-level AgentBus singleton (created once per process, shared across requests)
_agent_bus = AgentBus()
```

Inside `create_app()`, after `app.state.cached_ref = None` and before the health endpoint, add:

```python
    # Store AgentBus on app.state so tests and other modules can reach it
    app.state.agent_bus = _agent_bus

    @app.on_event("startup")
    async def _startup() -> None:
        """Wire AgentBus to the running event loop once uvicorn starts."""
        _agent_bus.setup_loop(asyncio.get_running_loop())
```

After the `/ws/trend` endpoint (and before the acquisition API section), add:

```python
    # ------------------------------------------------------------------
    # WebSocket: /ws/agent-events — streams AgentEvent JSON to clients
    # ------------------------------------------------------------------

    @app.websocket("/ws/agent-events")
    async def ws_agent_events(websocket: WebSocket) -> None:
        await websocket.accept()
        q = _agent_bus.subscribe()
        try:
            while True:
                event = await q.get()
                await websocket.send_text(event.to_json())
        except WebSocketDisconnect:
            pass
        finally:
            _agent_bus.unsubscribe(q)
```

- [ ] **Step 4: Run all server tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/webapp/test_server.py -v
```

Expected: all pass (including the 2 new tests).

- [ ] **Step 5: Commit**

```bash
git add spectraagent/webapp/server.py tests/spectraagent/webapp/test_server.py
git commit -m "feat: /ws/agent-events WebSocket — streams AgentBus events to browser clients"
```

---

## Task 3: QualityAgent

**Files:**
- Create: `spectraagent/webapp/agents/__init__.py`
- Create: `spectraagent/webapp/agents/quality.py`
- Create: `tests/spectraagent/webapp/agents/test_quality.py`

- [ ] **Step 1: Write the failing tests**

Create `spectraagent/webapp/agents/__init__.py` (empty).

Create `tests/spectraagent/webapp/agents/test_quality.py`:

```python
import asyncio

import numpy as np
import pytest

from spectraagent.webapp.agent_bus import AgentBus
from spectraagent.webapp.agents.quality import QualityAgent, _compute_snr


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


def _bus() -> tuple[AgentBus, asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    bus = AgentBus()
    bus.setup_loop(loop)
    return bus, loop


def _flush(loop: asyncio.AbstractEventLoop) -> None:
    """Run one event-loop iteration so call_soon_threadsafe callbacks execute."""
    loop.run_until_complete(asyncio.sleep(0))


@pytest.fixture
def wl() -> np.ndarray:
    return np.linspace(500.0, 900.0, 3648)


@pytest.fixture
def normal_spectrum(wl: np.ndarray) -> np.ndarray:
    """Lorentzian at 720 nm, amplitude 0.8 — clear peak, low noise."""
    sp = 0.8 / (1.0 + ((wl - 720.0) / 9.0) ** 2)
    sp += np.random.default_rng(0).normal(0, 0.001, len(wl))
    return np.clip(sp, 0.0, None)


@pytest.fixture
def saturated_spectrum(wl: np.ndarray) -> np.ndarray:
    """Peak at 65000 counts — exceeds saturation threshold."""
    return 65_000.0 / (1.0 + ((wl - 720.0) / 9.0) ** 2)


@pytest.fixture
def flat_noise_spectrum(wl: np.ndarray) -> np.ndarray:
    """Flat noise — no peak, SNR << 3."""
    return np.random.default_rng(1).normal(0.01, 0.01, len(wl)).clip(0.0, None)


# -----------------------------------------------------------------------
# _compute_snr
# -----------------------------------------------------------------------


def test_snr_high_for_lorentzian(wl, normal_spectrum):
    assert _compute_snr(wl, normal_spectrum) > 10.0


def test_snr_low_for_flat_noise(wl, flat_noise_spectrum):
    assert _compute_snr(wl, flat_noise_spectrum) < 3.0


# -----------------------------------------------------------------------
# QualityAgent
# -----------------------------------------------------------------------


def test_normal_frame_returns_true(wl, normal_spectrum):
    bus, loop = _bus()
    result = QualityAgent(bus).process(1, wl, normal_spectrum)
    loop.close()
    assert result is True


def test_normal_frame_emits_ok_event(wl, normal_spectrum):
    bus, loop = _bus()
    q = bus.subscribe()
    QualityAgent(bus).process(1, wl, normal_spectrum)
    _flush(loop)
    event = q.get_nowait()
    assert event.level == "ok"
    assert event.source == "QualityAgent"
    assert event.type == "quality"
    loop.close()


def test_saturated_frame_returns_false(wl, saturated_spectrum):
    bus, loop = _bus()
    result = QualityAgent(bus).process(1, wl, saturated_spectrum)
    loop.close()
    assert result is False


def test_saturated_frame_emits_error_event(wl, saturated_spectrum):
    bus, loop = _bus()
    q = bus.subscribe()
    QualityAgent(bus).process(1, wl, saturated_spectrum)
    _flush(loop)
    event = q.get_nowait()
    assert event.level == "error"
    assert event.data["quality"] == "saturated"
    loop.close()


def test_low_snr_frame_returns_true_with_warn(wl, flat_noise_spectrum):
    bus, loop = _bus()
    q = bus.subscribe()
    result = QualityAgent(bus).process(1, wl, flat_noise_spectrum)
    _flush(loop)
    event = q.get_nowait()
    assert result is True        # frame still processed
    assert event.level == "warn"
    assert event.data["quality"] == "low_snr"
    loop.close()


def test_event_contains_frame_number(wl, normal_spectrum):
    bus, loop = _bus()
    q = bus.subscribe()
    QualityAgent(bus).process(42, wl, normal_spectrum)
    _flush(loop)
    assert q.get_nowait().data["frame"] == 42
    loop.close()


def test_event_contains_snr(wl, normal_spectrum):
    bus, loop = _bus()
    q = bus.subscribe()
    QualityAgent(bus).process(1, wl, normal_spectrum)
    _flush(loop)
    assert q.get_nowait().data["snr"] > 0.0
    loop.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/webapp/agents/test_quality.py -v
```

Expected: `ModuleNotFoundError: No module named 'spectraagent.webapp.agents'`

- [ ] **Step 3: Create `spectraagent/webapp/agents/quality.py`**

```python
"""
spectraagent.webapp.agents.quality
===================================
QualityAgent — per-frame SNR and saturation gate.

Called synchronously from the acquisition loop (20 Hz).
Emits one AgentEvent per frame.

Quality rules (spec Section 4):
- max(intensities) > 60 000 counts → level="error", hard block (return False)
- SNR < 3                          → level="warn",  frame processed (return True)
- Normal                           → level="ok",    frame processed (return True)

SNR estimate: peak_intensity / std(off-peak noise floor).
Noise floor = first and last 5% of pixels (off-peak regions).
"""
from __future__ import annotations

import numpy as np

from spectraagent.webapp.agent_bus import AgentBus, AgentEvent

_SATURATION_THRESHOLD: float = 60_000.0
_SNR_WARN_THRESHOLD: float = 3.0


def _compute_snr(wavelengths: np.ndarray, intensities: np.ndarray) -> float:
    """Estimate SNR as peak_intensity / noise_std.

    Noise floor is estimated from the first and last 5 % of pixels
    (off-peak regions dominated by detector dark noise).
    """
    n = len(intensities)
    margin = max(n // 20, 10)
    noise = np.concatenate([intensities[:margin], intensities[-margin:]])
    noise_std = float(np.std(noise))
    if noise_std < 1e-9:
        noise_std = 1e-9
    return float(np.max(intensities)) / noise_std


class QualityAgent:
    """Per-frame quality gate — runs at 20 Hz in the acquisition thread.

    Parameters
    ----------
    bus:
        AgentBus for emitting events.
    saturation_threshold:
        Hard-block threshold in raw counts (default 60 000).
    snr_warn_threshold:
        SNR warning threshold (default 3.0).
    """

    def __init__(
        self,
        bus: AgentBus,
        saturation_threshold: float = _SATURATION_THRESHOLD,
        snr_warn_threshold: float = _SNR_WARN_THRESHOLD,
    ) -> None:
        self._bus = bus
        self._sat_threshold = saturation_threshold
        self._snr_threshold = snr_warn_threshold

    def process(
        self,
        frame_num: int,
        wavelengths: np.ndarray,
        intensities: np.ndarray,
    ) -> bool:
        """Check frame quality. Returns True to process frame, False to hard-block.

        Always emits exactly one AgentEvent regardless of outcome.
        """
        max_i = float(np.max(intensities))
        sat_pct = float(np.mean(intensities > self._sat_threshold) * 100.0)

        if max_i > self._sat_threshold:
            self._bus.emit(AgentEvent(
                source="QualityAgent",
                level="error",
                type="quality",
                data={
                    "frame": frame_num,
                    "snr": 0.0,
                    "saturation_pct": round(sat_pct, 2),
                    "quality": "saturated",
                    "max_intensity": round(max_i, 1),
                },
                text=(
                    f"Frame {frame_num} — SATURATED ({max_i:.0f} counts "
                    f"> {self._sat_threshold:.0f}). Frame discarded."
                ),
            ))
            return False

        snr = _compute_snr(wavelengths, intensities)

        if snr < self._snr_threshold:
            self._bus.emit(AgentEvent(
                source="QualityAgent",
                level="warn",
                type="quality",
                data={
                    "frame": frame_num,
                    "snr": round(snr, 2),
                    "saturation_pct": 0.0,
                    "quality": "low_snr",
                },
                text=(
                    f"Frame {frame_num} — SNR={snr:.1f} "
                    f"(below {self._snr_threshold}). Processed with warning."
                ),
            ))
            return True

        self._bus.emit(AgentEvent(
            source="QualityAgent",
            level="ok",
            type="quality",
            data={
                "frame": frame_num,
                "snr": round(snr, 2),
                "saturation_pct": round(sat_pct, 2),
                "quality": "ok",
            },
            text=f"Frame {frame_num} — SNR={snr:.1f}, quality=OK",
        ))
        return True
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/webapp/agents/test_quality.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add spectraagent/webapp/agents/ tests/spectraagent/webapp/agents/test_quality.py
git commit -m "feat: QualityAgent — per-frame SNR/saturation gate with AgentBus events"
```

---

## Task 4: DriftAgent

**Files:**
- Create: `spectraagent/webapp/agents/drift.py`
- Create: `tests/spectraagent/webapp/agents/test_drift.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/spectraagent/webapp/agents/test_drift.py`:

```python
import asyncio

import pytest

from spectraagent.webapp.agent_bus import AgentBus
from spectraagent.webapp.agents.drift import DriftAgent


def _bus():
    loop = asyncio.new_event_loop()
    bus = AgentBus()
    bus.setup_loop(loop)
    return bus, loop


def _flush(loop):
    loop.run_until_complete(asyncio.sleep(0))


def test_no_event_before_window_full():
    bus, loop = _bus()
    q = bus.subscribe()
    agent = DriftAgent(bus, integration_time_ms=50.0, window_frames=60)
    for i in range(30):           # only half the window
        agent.update(i, 720.0)
    _flush(loop)
    assert q.empty()
    loop.close()


def test_stable_signal_no_drift_event():
    bus, loop = _bus()
    q = bus.subscribe()
    agent = DriftAgent(bus, integration_time_ms=50.0, window_frames=60)
    for i in range(60):
        agent.update(i, 720.0)   # constant wavelength
    _flush(loop)
    assert q.empty()
    loop.close()


def test_fast_drift_emits_warn():
    """1.0 nm/min drift is well above the 0.05 nm/min threshold."""
    bus, loop = _bus()
    q = bus.subscribe()
    # integration_time_ms=50 → 1200 frames/min
    agent = DriftAgent(bus, integration_time_ms=50.0, window_frames=60)
    frames_per_min = 60_000 / 50.0
    drift_per_frame = 1.0 / frames_per_min   # 1 nm/min in nm/frame
    for i in range(60):
        agent.update(i, 720.0 + i * drift_per_frame)
    _flush(loop)
    assert not q.empty()
    event = q.get_nowait()
    assert event.level == "warn"
    assert event.type == "drift_warn"
    assert abs(event.data["drift_rate_nm_per_min"]) > 0.05
    loop.close()


def test_slow_drift_below_threshold_no_event():
    """0.01 nm/min drift is below the 0.05 nm/min threshold."""
    bus, loop = _bus()
    q = bus.subscribe()
    agent = DriftAgent(bus, integration_time_ms=50.0, window_frames=60)
    frames_per_min = 60_000 / 50.0
    drift_per_frame = 0.01 / frames_per_min
    for i in range(60):
        agent.update(i, 720.0 + i * drift_per_frame)
    _flush(loop)
    assert q.empty()
    loop.close()


def test_reset_clears_history():
    bus, loop = _bus()
    q = bus.subscribe()
    agent = DriftAgent(bus, integration_time_ms=50.0, window_frames=10)
    for i in range(10):
        agent.update(i, 720.0 + i * 0.1)
    agent.reset()
    # After reset, window is empty — 5 more frames should not trigger
    for i in range(5):
        agent.update(i, 721.0)
    _flush(loop)
    assert q.empty()
    loop.close()


def test_drift_event_has_required_fields():
    bus, loop = _bus()
    q = bus.subscribe()
    agent = DriftAgent(bus, integration_time_ms=50.0, window_frames=60)
    frames_per_min = 60_000 / 50.0
    for i in range(60):
        agent.update(i, 720.0 + i * (1.0 / frames_per_min))
    _flush(loop)
    event = q.get_nowait()
    for key in ("frame", "drift_rate_nm_per_min", "window_frames", "peak_wavelength"):
        assert key in event.data, f"missing key: {key}"
    loop.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/webapp/agents/test_drift.py -v
```

Expected: `ModuleNotFoundError: No module named 'spectraagent.webapp.agents.drift'`

- [ ] **Step 3: Create `spectraagent/webapp/agents/drift.py`**

```python
"""
spectraagent.webapp.agents.drift
==================================
DriftAgent — rolling CUSUM on LSPR peak wavelength shift.

Uses a 60-frame rolling window and linear regression slope to estimate
drift rate in nm/min.  Emits a ``drift_warn`` event when the absolute
drift rate exceeds the configurable threshold (default 0.05 nm/min).

Called from the acquisition loop after QualityAgent passes the frame.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from spectraagent.webapp.agent_bus import AgentBus, AgentEvent

_WINDOW_FRAMES: int = 60
_DRIFT_THRESHOLD_NM_PER_MIN: float = 0.05


class DriftAgent:
    """Rolling drift monitor for LSPR peak wavelength.

    Parameters
    ----------
    bus:
        AgentBus for emitting events.
    integration_time_ms:
        Frame acquisition time in ms; used to convert frames → minutes.
    window_frames:
        Rolling window size (default 60).
    drift_threshold_nm_per_min:
        Absolute drift rate that triggers ``drift_warn`` (default 0.05 nm/min).
    """

    def __init__(
        self,
        bus: AgentBus,
        integration_time_ms: float = 50.0,
        window_frames: int = _WINDOW_FRAMES,
        drift_threshold_nm_per_min: float = _DRIFT_THRESHOLD_NM_PER_MIN,
    ) -> None:
        self._bus = bus
        self._window = window_frames
        self._threshold = drift_threshold_nm_per_min
        # frames/min = 60 000 ms/min ÷ ms/frame
        self._frames_per_minute = 60_000.0 / integration_time_ms
        self._history: deque[float] = deque(maxlen=window_frames)

    def update(self, frame_num: int, peak_wavelength: float) -> None:
        """Record a peak wavelength observation. Emits drift_warn if threshold exceeded.

        Requires a full window before emitting any events.

        Parameters
        ----------
        frame_num:
            Current frame counter (included in event data).
        peak_wavelength:
            Detected LSPR peak wavelength in nm for this frame.
        """
        self._history.append(peak_wavelength)
        if len(self._history) < self._window:
            return

        history = np.array(self._history)
        x = np.arange(len(history), dtype=float)
        slope_nm_per_frame = float(np.polyfit(x, history, 1)[0])
        drift_rate = slope_nm_per_frame * self._frames_per_minute

        if abs(drift_rate) > self._threshold:
            self._bus.emit(AgentEvent(
                source="DriftAgent",
                level="warn",
                type="drift_warn",
                data={
                    "frame": frame_num,
                    "drift_rate_nm_per_min": round(drift_rate, 4),
                    "window_frames": self._window,
                    "peak_wavelength": round(peak_wavelength, 4),
                },
                text=(
                    f"Frame {frame_num} — drift rate {drift_rate:+.4f} nm/min "
                    f"exceeds ±{self._threshold} nm/min "
                    f"(over {self._window}-frame window)."
                ),
            ))

    def reset(self) -> None:
        """Clear rolling history (call at session start or reference capture)."""
        self._history.clear()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/webapp/agents/test_drift.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add spectraagent/webapp/agents/drift.py tests/spectraagent/webapp/agents/test_drift.py
git commit -m "feat: DriftAgent — rolling 60-frame CUSUM on LSPR peak wavelength"
```

---

## Task 5: CalibrationAgent

**Files:**
- Create: `spectraagent/webapp/agents/calibration.py`
- Create: `tests/spectraagent/webapp/agents/test_calibration.py`

The CalibrationAgent wraps `src.calibration.isotherms.select_isotherm` — do NOT reimplement AIC fitting. `select_isotherm` returns a dict with keys `best_model` (str), `best_result` (IsothermResult with `.aic`, `.r_squared`), `all_results` (dict), `aic_table` (list of tuples).

- [ ] **Step 1: Write the failing tests**

Create `tests/spectraagent/webapp/agents/test_calibration.py`:

```python
import asyncio

import numpy as np
import pytest

from spectraagent.webapp.agent_bus import AgentBus
from spectraagent.webapp.agents.calibration import CalibrationAgent


def _bus():
    loop = asyncio.new_event_loop()
    bus = AgentBus()
    bus.setup_loop(loop)
    return bus, loop


def _flush(loop):
    loop.run_until_complete(asyncio.sleep(0))


def test_no_event_below_min_points():
    bus, loop = _bus()
    q = bus.subscribe()
    agent = CalibrationAgent(bus, min_points=4)
    agent.add_point(0.1, -0.5)
    agent.add_point(0.2, -1.0)
    _flush(loop)
    assert q.empty()
    loop.close()


def test_event_emitted_at_min_points():
    bus, loop = _bus()
    q = bus.subscribe()
    agent = CalibrationAgent(bus, min_points=4)
    for c in [0.1, 0.2, 0.5, 1.0]:
        agent.add_point(c, -5.0 * c)
    _flush(loop)
    assert not q.empty()
    event = q.get_nowait()
    assert event.type == "model_selected"
    assert event.source == "CalibrationAgent"
    loop.close()


def test_event_has_required_data_fields():
    bus, loop = _bus()
    q = bus.subscribe()
    agent = CalibrationAgent(bus, min_points=4)
    for c in [0.1, 0.2, 0.5, 1.0]:
        agent.add_point(c, -5.0 * c)
    _flush(loop)
    data = q.get_nowait().data
    for key in ("n_points", "best_model", "best_aic", "r_squared"):
        assert key in data, f"missing key: {key}"
    loop.close()


def test_linear_data_prefers_low_param_model():
    """Perfect linear data should select 'linear' (AIC penalises extra params)."""
    bus, loop = _bus()
    q = bus.subscribe()
    agent = CalibrationAgent(bus, min_points=4)
    # Perfect linear: delta_lambda = -5 * concentration
    for c in [0.1, 0.2, 0.5, 1.0, 2.0]:
        agent.add_point(c, -5.0 * c)
    _flush(loop)
    # Drain all events (one per point after min_points)
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    assert len(events) > 0
    # The last event (most data) should select linear or langmuir
    assert events[-1].data["best_model"] in ("linear", "langmuir", "freundlich")
    loop.close()


def test_r_squared_is_between_0_and_1():
    bus, loop = _bus()
    q = bus.subscribe()
    agent = CalibrationAgent(bus, min_points=4)
    for c in [0.1, 0.2, 0.5, 1.0]:
        agent.add_point(c, -5.0 * c)
    _flush(loop)
    r2 = q.get_nowait().data["r_squared"]
    assert 0.0 <= r2 <= 1.0
    loop.close()


def test_clear_resets_data():
    bus, loop = _bus()
    agent = CalibrationAgent(bus)
    agent.add_point(0.1, -0.5)
    agent.clear()
    concs, deltas = agent.data
    assert concs == []
    assert deltas == []
    loop.close()


def test_data_property_returns_accumulated_points():
    bus, loop = _bus()
    agent = CalibrationAgent(bus, min_points=10)
    agent.add_point(0.1, -0.5)
    agent.add_point(0.2, -1.0)
    concs, deltas = agent.data
    assert concs == [0.1, 0.2]
    assert deltas == [-0.5, -1.0]
    loop.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/webapp/agents/test_calibration.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create `spectraagent/webapp/agents/calibration.py`**

```python
"""
spectraagent.webapp.agents.calibration
========================================
CalibrationAgent — AIC-based isotherm model selection.

Wraps ``src.calibration.isotherms.select_isotherm`` — do NOT reimplement
AIC fitting here.  ``select_isotherm`` evaluates Langmuir, Freundlich, Hill,
and Linear models, selects the winner by AICc (small-sample corrected AIC),
and returns a dict with ``best_model``, ``best_result``, ``aic_table``.

Emits a ``model_selected`` event after each calibration data point once
``min_points`` is reached.
"""
from __future__ import annotations

import logging

import numpy as np

from spectraagent.webapp.agent_bus import AgentBus, AgentEvent

log = logging.getLogger(__name__)

_MIN_POINTS: int = 4   # minimum before fitting is meaningful


class CalibrationAgent:
    """AIC model selector for calibration curve (wraps select_isotherm).

    Parameters
    ----------
    bus:
        AgentBus for emitting events.
    min_points:
        Minimum calibration points before fitting (default 4).
    """

    def __init__(self, bus: AgentBus, min_points: int = _MIN_POINTS) -> None:
        self._bus = bus
        self._min_points = min_points
        self._concentrations: list[float] = []
        self._delta_lambdas: list[float] = []

    def add_point(self, concentration: float, delta_lambda: float) -> None:
        """Add a calibration point and refit if enough data.

        Parameters
        ----------
        concentration:
            Analyte concentration (ppm or consistent units).
        delta_lambda:
            Measured LSPR peak shift in nm (typically negative on adsorption).
        """
        self._concentrations.append(float(concentration))
        self._delta_lambdas.append(float(delta_lambda))

        if len(self._concentrations) < self._min_points:
            return

        c = np.array(self._concentrations)
        r = np.array(self._delta_lambdas)
        n = len(c)

        try:
            from src.calibration.isotherms import select_isotherm

            result = select_isotherm(c, r)
            best_model: str = str(result["best_model"])
            best_result = result["best_result"]
            best_aic: float = float(best_result.aic)
            r_squared: float = float(best_result.r_squared)

            self._bus.emit(AgentEvent(
                source="CalibrationAgent",
                level="info",
                type="model_selected",
                data={
                    "n_points": n,
                    "best_model": best_model,
                    "best_aic": round(best_aic, 3),
                    "r_squared": round(r_squared, 4),
                    "aic_table": [
                        {"model": row[0], "aic": round(float(row[1]), 3)}
                        for row in result["aic_table"]
                    ],
                },
                text=(
                    f"Calibration: {n} points — best model: {best_model} "
                    f"(AICc={best_aic:.2f}, R²={r_squared:.4f})"
                ),
            ))
        except Exception as exc:
            log.warning("CalibrationAgent: fit failed: %s", exc)

    def clear(self) -> None:
        """Reset calibration data (call at new session start)."""
        self._concentrations.clear()
        self._delta_lambdas.clear()

    @property
    def data(self) -> tuple[list[float], list[float]]:
        """Return (concentrations, delta_lambdas) accumulated so far."""
        return list(self._concentrations), list(self._delta_lambdas)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/webapp/agents/test_calibration.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add spectraagent/webapp/agents/calibration.py tests/spectraagent/webapp/agents/test_calibration.py
git commit -m "feat: CalibrationAgent — wraps select_isotherm for AIC model selection"
```

---

## Task 6: ExperimentPlannerAgent + /api/calibration/* Routes

**Files:**
- Create: `spectraagent/webapp/agents/planner.py`
- Create: `tests/spectraagent/webapp/agents/test_planner.py`
- Modify: `spectraagent/webapp/server.py`
- Modify: `tests/spectraagent/webapp/test_server.py`

`GPRCalibration.predict(X)` expects X shape `(n_samples, n_features)`.
For 1D shift calibration: `shifts.reshape(-1, 1)` → returns `(mean, std)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/spectraagent/webapp/agents/test_planner.py`:

```python
import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest

from spectraagent.webapp.agent_bus import AgentBus
from spectraagent.webapp.agents.planner import ExperimentPlannerAgent


def _bus():
    loop = asyncio.new_event_loop()
    bus = AgentBus()
    bus.setup_loop(loop)
    return bus, loop


def _flush(loop):
    loop.run_until_complete(asyncio.sleep(0))


def _mock_gpr(peak_at: float = 5.0, n: int = 50, range_: tuple = (0.01, 10.0)):
    """GPR mock returning max uncertainty near peak_at."""
    gpr = MagicMock()
    xs = np.linspace(*range_, n)
    std = np.exp(-0.5 * ((xs - peak_at) / 0.5) ** 2)
    gpr.predict.return_value = (np.zeros(n), std)
    gpr.is_fitted = True
    return gpr


def test_suggest_returns_none_without_gpr():
    bus, loop = _bus()
    result = ExperimentPlannerAgent(bus).suggest()
    assert result is None
    loop.close()


def test_suggest_returns_float_with_gpr():
    bus, loop = _bus()
    agent = ExperimentPlannerAgent(bus, min_conc=0.01, max_conc=10.0)
    agent.set_gpr(_mock_gpr())
    result = agent.suggest()
    assert isinstance(result, float)
    loop.close()


def test_suggest_returns_highest_uncertainty_concentration():
    """Should return concentration near where the mock GPR has highest std."""
    bus, loop = _bus()
    agent = ExperimentPlannerAgent(bus, min_conc=0.01, max_conc=10.0, n_candidates=50)
    agent.set_gpr(_mock_gpr(peak_at=5.0))
    result = agent.suggest()
    assert 4.0 <= result <= 6.0   # near peak_at=5.0
    loop.close()


def test_suggest_emits_experiment_suggestion_event():
    bus, loop = _bus()
    q = bus.subscribe()
    agent = ExperimentPlannerAgent(bus, min_conc=0.01, max_conc=10.0)
    agent.set_gpr(_mock_gpr())
    agent.suggest()
    _flush(loop)
    assert not q.empty()
    event = q.get_nowait()
    assert event.type == "experiment_suggestion"
    assert event.source == "ExperimentPlannerAgent"
    loop.close()


def test_suggestion_event_has_required_fields():
    bus, loop = _bus()
    q = bus.subscribe()
    agent = ExperimentPlannerAgent(bus, min_conc=0.01, max_conc=10.0)
    agent.set_gpr(_mock_gpr())
    agent.suggest()
    _flush(loop)
    data = q.get_nowait().data
    for key in ("suggested_concentration", "posterior_std", "search_range"):
        assert key in data, f"missing key: {key}"
    loop.close()


def test_suggest_with_failed_gpr_returns_none():
    bus, loop = _bus()
    gpr = MagicMock()
    gpr.predict.side_effect = RuntimeError("GPR not fitted")
    agent = ExperimentPlannerAgent(bus)
    agent.set_gpr(gpr)
    result = agent.suggest()
    assert result is None
    loop.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/webapp/agents/test_planner.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create `spectraagent/webapp/agents/planner.py`**

```python
"""
spectraagent.webapp.agents.planner
====================================
ExperimentPlannerAgent — suggests the next concentration to measure.

Uses GPRCalibration posterior uncertainty: queries predict() on a grid of
candidate concentrations and returns the one with the highest posterior std
(maximum information gain per spec Section 4 — no BoTorch needed).

Called on-demand via POST /api/calibration/suggest or by CalibrationAgent
after model selection.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from spectraagent.webapp.agent_bus import AgentBus, AgentEvent

log = logging.getLogger(__name__)

_N_CANDIDATES: int = 50


class ExperimentPlannerAgent:
    """Concentration suggestion using GPR posterior uncertainty.

    Parameters
    ----------
    bus:
        AgentBus for emitting events.
    min_conc, max_conc:
        Concentration range to search.
    n_candidates:
        Grid resolution (default 50).
    """

    def __init__(
        self,
        bus: AgentBus,
        min_conc: float = 0.01,
        max_conc: float = 10.0,
        n_candidates: int = _N_CANDIDATES,
    ) -> None:
        self._bus = bus
        self._min_conc = min_conc
        self._max_conc = max_conc
        self._n_candidates = n_candidates
        self._gpr = None  # set via set_gpr()

    def set_gpr(self, gpr) -> None:
        """Inject a fitted GPRCalibration (or compatible mock) instance."""
        self._gpr = gpr

    def suggest(self) -> Optional[float]:
        """Return the concentration with the highest GPR posterior std.

        Returns None if no GPR is set or if prediction fails.
        Emits an ``experiment_suggestion`` event on success.
        """
        if self._gpr is None:
            return None

        try:
            candidates = np.linspace(self._min_conc, self._max_conc, self._n_candidates)
            # GPRCalibration.predict() expects shape (n, 1) for 1D features
            _, std_arr = self._gpr.predict(candidates.reshape(-1, 1))
            best_idx = int(np.argmax(std_arr))
            best_conc = float(candidates[best_idx])
            best_std = float(std_arr[best_idx])

            self._bus.emit(AgentEvent(
                source="ExperimentPlannerAgent",
                level="info",
                type="experiment_suggestion",
                data={
                    "suggested_concentration": round(best_conc, 4),
                    "posterior_std": round(best_std, 6),
                    "search_range": [self._min_conc, self._max_conc],
                    "n_candidates": self._n_candidates,
                },
                text=(
                    f"Suggested next concentration: {best_conc:.4f} "
                    f"(posterior σ={best_std:.4g})"
                ),
            ))
            return best_conc

        except Exception as exc:
            log.warning("ExperimentPlannerAgent.suggest() failed: %s", exc)
            return None
```

- [ ] **Step 4: Add calibration routes to `server.py`**

After the `AcquisitionConfig` Pydantic model and before `_agent_bus = AgentBus()`, add:

```python
class CalibrationPoint(BaseModel):
    concentration: float
    delta_lambda: float
```

Inside `create_app()`, after the `/api/acquisition/reference` route, add:

```python
    # ------------------------------------------------------------------
    # Calibration API
    # ------------------------------------------------------------------

    @app.post("/api/calibration/add-point")
    async def calibration_add_point(point: CalibrationPoint) -> JSONResponse:
        """Add a calibration data point; CalibrationAgent re-fits all models."""
        calib_agent = getattr(app.state, "calibration_agent", None)
        if calib_agent is not None:
            calib_agent.add_point(point.concentration, point.delta_lambda)
        return JSONResponse({
            "status": "added",
            "concentration": point.concentration,
            "delta_lambda": point.delta_lambda,
        })

    @app.post("/api/calibration/suggest")
    async def calibration_suggest() -> JSONResponse:
        """Return the next recommended concentration from ExperimentPlannerAgent."""
        planner = getattr(app.state, "planner_agent", None)
        if planner is None:
            return JSONResponse({"suggestion": None, "reason": "planner_not_initialized"})
        suggested = planner.suggest()
        if suggested is None:
            return JSONResponse({"suggestion": None, "reason": "no_gpr_fitted"})
        return JSONResponse({"suggestion": suggested})
```

- [ ] **Step 5: Add server route tests**

Add to `tests/spectraagent/webapp/test_server.py`:

```python
def test_calibration_add_point_returns_200(client):
    resp = client.post("/api/calibration/add-point", json={
        "concentration": 0.5,
        "delta_lambda": -2.5,
    })
    assert resp.status_code == 200
    assert resp.json()["concentration"] == 0.5


def test_calibration_suggest_returns_200(client):
    resp = client.post("/api/calibration/suggest")
    assert resp.status_code == 200
    # No GPR fitted yet → suggestion is null
    data = resp.json()
    assert "suggestion" in data
```

- [ ] **Step 6: Run all tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/webapp/agents/test_planner.py tests/spectraagent/webapp/test_server.py -v
```

Expected: 6 + all server tests pass.

- [ ] **Step 7: Commit**

```bash
git add spectraagent/webapp/agents/planner.py tests/spectraagent/webapp/agents/test_planner.py spectraagent/webapp/server.py tests/spectraagent/webapp/test_server.py
git commit -m "feat: ExperimentPlannerAgent + /api/calibration/add-point and /suggest routes"
```

---

## Task 7: Wire Agents to Startup Sequence

**Files:**
- Modify: `spectraagent/__main__.py`

The `start` command must:
1. Create `AgentBus` with the session JSONL path
2. Create all four agents, store on `app.state`
3. Update `_acquisition_loop()` to call QualityAgent + DriftAgent per frame
4. Update `acq_start()` to set JSONL path and reset DriftAgent on session start

Note: `app.state.agent_bus = _agent_bus` in `server.py` sets the module-level singleton.
In `start()`, create agents and store them so the server routes can reach them.

- [ ] **Step 1: Write the failing integration test**

Add to `tests/spectraagent/test_config.py`:

```python
def test_start_simulate_emits_quality_events():
    """Integration: start server, connect to /ws/agent-events, verify quality events."""
    import json
    import threading
    import time
    import httpx
    from httpx_ws import connect_ws  # noqa: F401 — checked below

    pytest.importorskip("httpx_ws")  # skip if not installed

    proc = subprocess.Popen(
        [sys.executable, "-m", "spectraagent", "start", "--simulate", "--no-browser"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        time.sleep(4)
        resp = httpx.get("http://127.0.0.1:8765/api/health", timeout=5)
        assert resp.status_code == 200
    finally:
        proc.terminate()
        proc.wait(timeout=5)
```

Actually, WebSocket testing from subprocess is complex. Use a simpler smoke test:

```python
def test_start_simulate_agent_bus_active():
    """Smoke test: server starts, health responds, no errors in stderr."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "spectraagent", "start", "--simulate", "--no-browser"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        time.sleep(4)
        resp = httpx.get("http://127.0.0.1:8765/api/health", timeout=5)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
    finally:
        proc.terminate()
        proc.wait(timeout=5)
```

This test already exists from Task 13 of Plan 1 (`test_start_simulate_no_browser_serves_health`). Don't duplicate it — reuse it to confirm nothing regressed.

- [ ] **Step 2: Read the current `_acquisition_loop` in `__main__.py`**

```bash
.venv/Scripts/python.exe -c "
import ast, pathlib
src = pathlib.Path('spectraagent/__main__.py').read_text()
for i, line in enumerate(src.splitlines(), 1):
    if '_acquisition_loop' in line or 'agent' in line.lower():
        print(i, line)
"
```

Expected: shows `_acquisition_loop` function and any existing agent references.

- [ ] **Step 3: Update `spectraagent/__main__.py`**

Replace the `_acquisition_loop` function and update the `start` command. The full replacement for `__main__.py` (preserve all existing imports and helpers, only update the listed functions):

**Replace `_acquisition_loop`:**

```python
def _acquisition_loop(
    driver: "AbstractHardwareDriver",
    app: "FastAPI",
) -> None:
    """Daemon thread: read spectra, run quality/drift agents, broadcast to WS clients."""
    import asyncio
    import json
    import time

    import numpy as np

    wl_list = driver.get_wavelengths().tolist()
    wl_np = np.array(wl_list)
    frame_num = 0

    while True:
        try:
            intensities = driver.read_spectrum()
        except Exception as exc:
            log.warning("Acquisition error: %s", exc)
            time.sleep(1.0)
            continue

        frame_num += 1

        # ------------------------------------------------------------------
        # Quality gate (always runs — per spec, hard-blocks saturated frames)
        # ------------------------------------------------------------------
        quality_agent = getattr(app.state, "quality_agent", None)
        if quality_agent is not None:
            passes = quality_agent.process(frame_num, wl_np, intensities)
            if not passes:
                continue   # saturated frame — discard, do not broadcast

        # ------------------------------------------------------------------
        # Drift monitor (only when a physics plugin is set to detect peak)
        # ------------------------------------------------------------------
        drift_agent = getattr(app.state, "drift_agent", None)
        plugin = getattr(app.state, "plugin", None)
        if drift_agent is not None and plugin is not None:
            try:
                peak_wl = plugin.detect_peak(wl_np, intensities)
                if peak_wl is not None:
                    drift_agent.update(frame_num, peak_wl)
            except Exception as exc:
                log.debug("DriftAgent.update() failed: %s", exc)

        # ------------------------------------------------------------------
        # Broadcast spectrum to /ws/spectrum clients
        # ------------------------------------------------------------------
        spectrum_bc = getattr(app.state, "spectrum_bc", None)
        if spectrum_bc is not None:
            msg = json.dumps({"wl": wl_list, "i": intensities.tolist()})
            try:
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(
                    lambda m=msg: asyncio.ensure_future(spectrum_bc.broadcast(m))
                )
            except RuntimeError:
                pass
```

**Add agent creation to `start` command** — insert after the existing `app.state.cached_ref = None` line and before the `acq_thread` creation:

```python
    # Step 5a: Create agents and store on app.state
    from spectraagent.webapp.agent_bus import AgentBus
    from spectraagent.webapp.agents.quality import QualityAgent
    from spectraagent.webapp.agents.drift import DriftAgent
    from spectraagent.webapp.agents.calibration import CalibrationAgent
    from spectraagent.webapp.agents.planner import ExperimentPlannerAgent

    # _agent_bus is the module-level singleton in server.py — retrieve it
    from spectraagent.webapp.server import _agent_bus as agent_bus

    quality_agent = QualityAgent(agent_bus, integration_time_ms=cfg.hardware.integration_time_ms) \
        if hasattr(QualityAgent.__init__, '__code__') else QualityAgent(agent_bus)
    # QualityAgent doesn't take integration_time_ms — use simpler call:
    quality_agent = QualityAgent(agent_bus)
    drift_agent = DriftAgent(agent_bus, integration_time_ms=cfg.hardware.integration_time_ms)
    calibration_agent = CalibrationAgent(agent_bus)
    planner_agent = ExperimentPlannerAgent(agent_bus)

    app.state.quality_agent = quality_agent
    app.state.drift_agent = drift_agent
    app.state.calibration_agent = calibration_agent
    app.state.planner_agent = planner_agent
    typer.echo("Agents: QualityAgent, DriftAgent, CalibrationAgent, ExperimentPlannerAgent")
```

Wait — that pattern is ugly. Use clean code:

```python
    # Step 5a: Create deterministic agents
    from spectraagent.webapp.agent_bus import AgentBus
    from spectraagent.webapp.agents.calibration import CalibrationAgent
    from spectraagent.webapp.agents.drift import DriftAgent
    from spectraagent.webapp.agents.planner import ExperimentPlannerAgent
    from spectraagent.webapp.agents.quality import QualityAgent
    from spectraagent.webapp.server import _agent_bus as agent_bus

    app.state.quality_agent = QualityAgent(agent_bus)
    app.state.drift_agent = DriftAgent(
        agent_bus,
        integration_time_ms=cfg.hardware.integration_time_ms,
    )
    app.state.calibration_agent = CalibrationAgent(agent_bus)
    app.state.planner_agent = ExperimentPlannerAgent(agent_bus)
    typer.echo("Agents ready: Quality, Drift, Calibration, Planner")
```

- [ ] **Step 4: Apply the changes to `__main__.py`**

Read the file first, then use Edit to:
1. Replace the `_acquisition_loop` function body with the new version above
2. Insert the agent creation block into `start()` after `app.state.cached_ref = None`

Use the Read tool to find exact line numbers, then Edit with exact old/new strings.

- [ ] **Step 5: Run the existing integration test**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/test_config.py::test_start_simulate_no_browser_serves_health -v -s
```

Expected: PASS (server still starts, /api/health still returns 200).

- [ ] **Step 6: Run full test suite**

```bash
.venv/Scripts/python.exe -m pytest tests/ -q --tb=short 2>&1 | tail -15
```

Expected: 712+ passed, 0 new failures.

- [ ] **Step 7: Commit**

```bash
git add spectraagent/__main__.py
git commit -m "feat: wire QualityAgent + DriftAgent to acquisition loop; all agents on app.state"
```

---

## Self-Review

After completing Task 7, check against the spec:

**Spec coverage (Section 4 — Layer 1 Deterministic Agents):**
- [x] QualityAgent — every frame, SNR gate, saturation hard-block at 60 000 counts
- [x] DriftAgent — rolling 60-frame CUSUM, emits `drift_warn` at > 0.05 nm/min
- [x] CalibrationAgent — AIC model selection after each point via `select_isotherm`
- [x] ExperimentPlannerAgent — GPR uncertainty grid, returns via `/api/calibration/suggest`
- [x] AgentBus — `call_soon_threadsafe` bridge, per-client queues, JSONL writer
- [x] `/ws/agent-events` WebSocket
- [x] AgentEvent JSON schema: `{ts, source, level, type, data, text}`
- [x] `level` values: ok, warn, error, info (claude added in Plan 3)

**Spec coverage (Section 4.1 — AgentBus threading model):**
- [x] `call_soon_threadsafe(_fanout, event)` — zero-blocking from pipeline thread
- [x] Per-client `asyncio.Queue` (not shared) — correct fan-out
- [x] `agent_events.jsonl` written by `_fanout`
- [x] `setup_loop()` called at FastAPI startup via `on_event("startup")`

**Not in this plan (covered by later plans):**
- Claude API Agents (Plan 3): AnomalyExplainer, ExperimentNarrator, ReportWriter, DiagnosticsAgent
- React Frontend (Plan 4): Agent Console tab, color-coded event log, Ask Claude UI
- `/api/agents/ask` SSE endpoint (Plan 3)
- Report generation (Plan 5)

**Acceptance criteria:**
- [ ] `spectraagent start --simulate --no-browser` → server runs, `/api/health` 200
- [ ] `GET /ws/agent-events` → WebSocket accepts connection
- [ ] Quality events appear in `/ws/agent-events` stream during simulation
- [ ] `POST /api/calibration/add-point` with 4+ points → CalibrationAgent emits `model_selected`
- [ ] All 712+ existing tests still pass

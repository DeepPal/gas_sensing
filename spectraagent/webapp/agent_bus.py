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
from typing import IO, Optional

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
        self._jsonl_file: Optional[IO[str]] = None  # opened lazily on first write

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

    def emit(self, event: AgentEvent) -> Optional["asyncio.Handle"]:
        """Emit an event from any thread. No-op if setup_loop() not yet called.

        Returns the asyncio Handle for the scheduled fanout callback, or None
        if the bus has no event loop attached.  Callers may call
        ``handle.cancel()`` to suppress delivery before the next loop tick.
        """
        if self._loop is None:
            return None
        return self._loop.call_soon_threadsafe(self._fanout, event)

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

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest

from spectraagent.webapp.agent_bus import AgentBus, AgentEvent


def _event(**kw) -> AgentEvent:
    defaults: dict[str, Any] = dict(source="Test", level="ok", type="test", data={}, text="t")
    defaults.update(kw)
    return AgentEvent(
        source=str(defaults["source"]),
        level=str(defaults["level"]),
        type=str(defaults["type"]),
        data=cast(dict, defaults["data"]),
        text=str(defaults["text"]),
    )


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

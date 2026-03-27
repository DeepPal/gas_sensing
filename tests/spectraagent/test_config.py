from pathlib import Path
import json
import shutil
from datetime import datetime, timezone

import pytest

from spectraagent.config import SpectraAgentConfig, load_config
from typer.testing import CliRunner
from spectraagent.__main__ import cli


def test_defaults_when_no_file(tmp_path):
    cfg = load_config(tmp_path / "spectraagent.toml")
    assert cfg.server.port == 8765
    assert cfg.hardware.integration_time_ms == 50.0
    assert cfg.agents.auto_explain is False
    assert cfg.claude.model == "claude-sonnet-4-6"


def test_file_created_when_missing(tmp_path):
    path = tmp_path / "spectraagent.toml"
    assert not path.exists()
    load_config(path)
    assert path.exists()


def test_override_from_file(tmp_path):
    path = tmp_path / "spectraagent.toml"
    path.write_text("[server]\nport = 9000\n")
    cfg = load_config(path)
    assert cfg.server.port == 9000
    assert cfg.server.host == "127.0.0.1"  # default preserved


def test_physics_defaults(tmp_path):
    cfg = load_config(tmp_path / "spectraagent.toml")
    assert cfg.physics.default_plugin == "lspr"
    assert cfg.physics.search_min_nm == 500.0
    assert cfg.physics.search_max_nm == 900.0


def test_plugins_list_shows_simulation():
    runner = CliRunner()
    result = runner.invoke(cli, ["plugins", "list"])
    assert result.exit_code == 0
    assert "simulation" in result.output.lower()


def test_plugins_list_shows_lspr():
    runner = CliRunner()
    result = runner.invoke(cli, ["plugins", "list"])
    assert result.exit_code == 0
    assert "lspr" in result.output.lower()


# ---------------------------------------------------------------------------
# Task 13: integration test for `spectraagent start --simulate --no-browser`
# ---------------------------------------------------------------------------

import subprocess
import sys
import time
import httpx


def _parse_event_ts(value: str) -> datetime:
    raw = value.strip()
    assert raw != ""
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def test_start_simulate_no_browser_serves_health():
    """Integration test: start server in subprocess, hit /api/health, then kill it."""
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
        assert resp.json()["simulate"] is True
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_start_simulate_persists_session_with_nonzero_frames():
    """Integration test: simulated run persists session metadata + event log."""
    port = 8766
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "spectraagent",
            "start",
            "--simulate",
            "--no-browser",
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    session_id = None
    session_dir = None
    try:
        health_url = f"http://127.0.0.1:{port}/api/health"
        for _ in range(25):
            try:
                resp = httpx.get(health_url, timeout=2)
                if resp.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.2)
        else:
            raise AssertionError("Server did not become healthy in time")

        start_resp = httpx.post(
            f"http://127.0.0.1:{port}/api/acquisition/start",
            timeout=5,
        )
        assert start_resp.status_code == 200
        session_id = start_resp.json()["session_id"]

        # Allow the acquisition thread to collect several simulated frames.
        time.sleep(0.5)

        stop_resp = httpx.post(
            f"http://127.0.0.1:{port}/api/acquisition/stop",
            timeout=5,
        )
        assert stop_resp.status_code == 200

        session_dir = Path("output") / "sessions" / session_id
        meta_path = session_dir / "session_meta.json"
        events_path = session_dir / "agent_events.jsonl"
        assert meta_path.exists()
        assert events_path.exists()

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert int(meta.get("frame_count", 0)) > 0
        assert meta.get("stopped_at") is not None

        lines = [ln for ln in events_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) > 0
        first_event = json.loads(lines[0])
        for key in ("ts", "source", "level", "type", "data", "text"):
            assert key in first_event
    finally:
        proc.terminate()
        proc.wait(timeout=5)
        if session_dir is not None and session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)


def test_start_simulate_event_log_grows_during_active_acquisition():
    """Integration test: active acquisition writes multiple well-formed events."""
    port = 8767
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "spectraagent",
            "start",
            "--simulate",
            "--no-browser",
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    session_id = None
    session_dir = None
    try:
        health_url = f"http://127.0.0.1:{port}/api/health"
        for _ in range(25):
            try:
                resp = httpx.get(health_url, timeout=2)
                if resp.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.2)
        else:
            raise AssertionError("Server did not become healthy in time")

        start_resp = httpx.post(
            f"http://127.0.0.1:{port}/api/acquisition/start",
            timeout=5,
        )
        assert start_resp.status_code == 200
        session_id = start_resp.json()["session_id"]

        # Let simulated acquisition run long enough to emit multiple quality events.
        time.sleep(1.0)

        stop_resp = httpx.post(
            f"http://127.0.0.1:{port}/api/acquisition/stop",
            timeout=5,
        )
        assert stop_resp.status_code == 200

        session_dir = Path("output") / "sessions" / session_id
        events_path = session_dir / "agent_events.jsonl"
        assert events_path.exists()

        lines = [ln for ln in events_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) >= 3

        events = [json.loads(line) for line in lines]
        for event in events[:3]:
            for key in ("ts", "source", "level", "type", "data", "text"):
                assert key in event

        assert any(event.get("source") == "QualityAgent" for event in events)
        assert any(event.get("type") == "quality" for event in events)
    finally:
        proc.terminate()
        proc.wait(timeout=5)
        if session_dir is not None and session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)


def test_start_simulate_soak_acquisition_consistency():
    """Integration test: longer simulated run keeps event and frame persistence consistent."""
    port = 8768
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "spectraagent",
            "start",
            "--simulate",
            "--no-browser",
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    session_id = None
    session_dir = None
    try:
        health_url = f"http://127.0.0.1:{port}/api/health"
        for _ in range(30):
            try:
                resp = httpx.get(health_url, timeout=2)
                if resp.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.2)
        else:
            raise AssertionError("Server did not become healthy in time")

        start_resp = httpx.post(
            f"http://127.0.0.1:{port}/api/acquisition/start",
            timeout=5,
        )
        assert start_resp.status_code == 200
        session_id = start_resp.json()["session_id"]

        # Soak the simulated loop briefly to validate persistence under sustained activity.
        time.sleep(2.5)

        stop_resp = httpx.post(
            f"http://127.0.0.1:{port}/api/acquisition/stop",
            timeout=5,
        )
        assert stop_resp.status_code == 200

        session_dir = Path("output") / "sessions" / session_id
        meta_path = session_dir / "session_meta.json"
        events_path = session_dir / "agent_events.jsonl"
        assert meta_path.exists()
        assert events_path.exists()

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        frame_count = int(meta.get("frame_count", 0))
        assert frame_count >= 10
        assert meta.get("stopped_at") is not None

        lines = [ln for ln in events_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) >= 5

        events = [json.loads(line) for line in lines]
        timestamps = [_parse_event_ts(str(event["ts"])) for event in events]
        assert timestamps == sorted(timestamps)

        quality_events = 0
        for event in events:
            for key in ("ts", "source", "level", "type", "data", "text"):
                assert key in event
            assert isinstance(event["data"], dict)
            assert isinstance(event["level"], str)
            assert event["level"].strip() != ""
            if event.get("type") == "quality":
                quality_events += 1

        assert quality_events >= 3
    finally:
        proc.terminate()
        proc.wait(timeout=5)
        if session_dir is not None and session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)


def test_start_simulate_repeated_start_stop_cycles_persist_clean_sessions():
    """Integration test: repeated acquisition cycles keep persistence stable."""
    port = 8769
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "spectraagent",
            "start",
            "--simulate",
            "--no-browser",
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    session_dirs: list[Path] = []
    try:
        health_url = f"http://127.0.0.1:{port}/api/health"
        for _ in range(30):
            try:
                resp = httpx.get(health_url, timeout=2)
                if resp.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.2)
        else:
            raise AssertionError("Server did not become healthy in time")

        seen_session_ids: set[str] = set()
        for _ in range(3):
            start_resp = httpx.post(
                f"http://127.0.0.1:{port}/api/acquisition/start",
                timeout=5,
            )
            assert start_resp.status_code == 200
            session_id = start_resp.json()["session_id"]
            assert session_id not in seen_session_ids
            seen_session_ids.add(session_id)

            time.sleep(0.8)

            stop_resp = httpx.post(
                f"http://127.0.0.1:{port}/api/acquisition/stop",
                timeout=5,
            )
            assert stop_resp.status_code == 200

            session_dir = Path("output") / "sessions" / session_id
            session_dirs.append(session_dir)
            meta_path = session_dir / "session_meta.json"
            events_path = session_dir / "agent_events.jsonl"
            assert meta_path.exists()
            assert events_path.exists()

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            assert int(meta.get("frame_count", 0)) > 0
            assert meta.get("stopped_at") is not None

            lines = [ln for ln in events_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            assert len(lines) >= 1
            events = [json.loads(line) for line in lines]
            timestamps = [_parse_event_ts(str(event["ts"])) for event in events]
            assert timestamps == sorted(timestamps)

        # Server remains responsive after repeated cycle transitions.
        final_health = httpx.get(health_url, timeout=2)
        assert final_health.status_code == 200
    finally:
        proc.terminate()
        proc.wait(timeout=5)
        for path in session_dirs:
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)

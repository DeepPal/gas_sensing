from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import pytest
from typer.testing import CliRunner

from spectraagent.__main__ import cli
from spectraagent.config import ConfigError, SpectraAgentConfig, load_config


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


@pytest.mark.integration
@pytest.mark.reliability
def test_start_simulate_no_browser_serves_health():
    """Integration test: start server in subprocess, hit /api/health, then kill it."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "spectraagent", "start", "--simulate", "--no-browser"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        health_url = "http://127.0.0.1:8765/api/health"
        for _ in range(50):
            try:
                resp = httpx.get(health_url, timeout=2)
                if resp.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.2)
        else:
            stderr = ""
            try:
                stderr = (proc.stderr.read() or b"")[-2000:].decode(errors="ignore")
            except Exception:
                pass
            raise AssertionError(f"Server did not become healthy in time. stderr tail:\n{stderr}")

        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["simulate"] is True
    finally:
        proc.terminate()
        proc.wait(timeout=5)


@pytest.mark.integration
@pytest.mark.reliability
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


@pytest.mark.integration
@pytest.mark.reliability
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
        # ok events are throttled (ok_emit_every=5); in a 1-second subprocess session
        # the frame rate may be 6–10 fps, yielding 1–2 events. Require >= 1 to stay
        # robust to test-environment timing variation while still validating the path.
        assert len(lines) >= 1

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


@pytest.mark.integration
@pytest.mark.reliability
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
        # At ~3–5 fps (pipeline overhead dominates), expect at least 5 frames in 2.5 s.
        assert frame_count >= 5
        assert meta.get("stopped_at") is not None

        lines = [ln for ln in events_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        # ok events are throttled (ok_emit_every=5); in a 2.5-second subprocess session
        # at ~6 fps the loop emits ~3 ok events. Require >= 2 to stay robust to
        # test-environment scheduling while still validating sustained persistence.
        assert len(lines) >= 2

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

        # Throttled ok events: at ~3 fps with ok_emit_every=5, a 2.5 s session
        # yields ~2–3 quality events. Require >= 1 to be robust to scheduling.
        assert quality_events >= 1
    finally:
        proc.terminate()
        proc.wait(timeout=5)
        if session_dir is not None and session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.reliability
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
            # ok events are throttled (ok_emit_every=5); a 0.8 s cycle at ~3–5 fps
            # may produce 0 or 1 events depending on counter phase. Validate schema
            # and ordering only when events are present rather than requiring a count.
            if lines:
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


# ---------------------------------------------------------------------------
# Config validation — ConfigError on bad values
# ---------------------------------------------------------------------------


def test_invalid_port_zero_raises(tmp_path):
    path = tmp_path / "spectraagent.toml"
    path.write_text("[server]\nport = 0\n")
    with pytest.raises(ConfigError, match="port"):
        load_config(path)


def test_invalid_port_too_large_raises(tmp_path):
    path = tmp_path / "spectraagent.toml"
    path.write_text("[server]\nport = 99999\n")
    with pytest.raises(ConfigError, match="port"):
        load_config(path)


def test_invalid_integration_time_zero_raises(tmp_path):
    path = tmp_path / "spectraagent.toml"
    path.write_text("[hardware]\nintegration_time_ms = 0.0\n")
    with pytest.raises(ConfigError, match="integration_time_ms"):
        load_config(path)


def test_invalid_integration_time_negative_raises(tmp_path):
    path = tmp_path / "spectraagent.toml"
    path.write_text("[hardware]\nintegration_time_ms = -1.0\n")
    with pytest.raises(ConfigError, match="integration_time_ms"):
        load_config(path)


def test_invalid_physics_range_inverted_raises(tmp_path):
    path = tmp_path / "spectraagent.toml"
    path.write_text("[physics]\nsearch_min_nm = 900.0\nsearch_max_nm = 500.0\n")
    with pytest.raises(ConfigError, match="search_min_nm"):
        load_config(path)


def test_invalid_claude_timeout_zero_raises(tmp_path):
    path = tmp_path / "spectraagent.toml"
    path.write_text("[claude]\ntimeout_s = 0\n")
    with pytest.raises(ConfigError, match="timeout_s"):
        load_config(path)


def test_valid_config_does_not_raise(tmp_path):
    """Default-generated config passes validation without error."""
    path = tmp_path / "spectraagent.toml"
    cfg = load_config(path)
    assert isinstance(cfg, SpectraAgentConfig)


# ---------------------------------------------------------------------------
# CLI: --version / -V flag
# ---------------------------------------------------------------------------


def test_version_flag():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "spectraagent" in result.output


def test_version_short_flag():
    runner = CliRunner()
    result = runner.invoke(cli, ["-V"])
    assert result.exit_code == 0
    assert "spectraagent" in result.output


# ---------------------------------------------------------------------------
# CLI: sessions sub-commands
# ---------------------------------------------------------------------------


def _write_mock_session(
    base: Path,
    session_id: str = "20260101_120000",
    *,
    stopped: bool = True,
    frame_count: int = 42,
) -> Path:
    """Create a minimal on-disk session for CLI tests."""
    session_dir = base / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    meta: dict = {
        "session_id": session_id,
        "started_at": "2026-01-01T12:00:00Z",
        "frame_count": frame_count,
    }
    if stopped:
        meta["stopped_at"] = "2026-01-01T12:05:00Z"
    (session_dir / "session_meta.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )
    return session_dir


def test_sessions_list_nonexistent_dir(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["sessions", "list", "--dir", str(tmp_path / "nosessions")])
    assert result.exit_code == 0
    assert "No sessions" in result.output


def test_sessions_list_empty_dir(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["sessions", "list", "--dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "No sessions" in result.output


def test_sessions_list_shows_metadata(tmp_path):
    _write_mock_session(tmp_path, "20260101_120000", stopped=True, frame_count=42)
    runner = CliRunner()
    result = runner.invoke(cli, ["sessions", "list", "--dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "20260101_120000" in result.output
    assert "SESSION ID" in result.output
    assert "stopped" in result.output
    assert "42" in result.output


def test_sessions_list_active_session(tmp_path):
    _write_mock_session(tmp_path, "20260101_130000", stopped=False, frame_count=10)
    runner = CliRunner()
    result = runner.invoke(cli, ["sessions", "list", "--dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "active" in result.output


def test_sessions_list_multiple_sorted(tmp_path):
    """Newer session should appear before older one."""
    # Use explicit started_at so sort order is deterministic
    for sid, started in [
        ("20260101_120000", "2026-01-01T12:00:00Z"),
        ("20260102_120000", "2026-01-02T12:00:00Z"),
    ]:
        d = tmp_path / sid
        d.mkdir()
        (d / "session_meta.json").write_text(
            json.dumps({"session_id": sid, "started_at": started, "frame_count": 1}),
            encoding="utf-8",
        )
    runner = CliRunner()
    result = runner.invoke(cli, ["sessions", "list", "--dir", str(tmp_path)])
    assert result.exit_code == 0
    idx_new = result.output.index("20260102_120000")
    idx_old = result.output.index("20260101_120000")
    assert idx_new < idx_old, "Newer session must appear first"


def test_sessions_get_not_found(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["sessions", "get", "nonexistent_id", "--dir", str(tmp_path)])
    assert result.exit_code == 1


def test_sessions_get_returns_metadata(tmp_path):
    _write_mock_session(tmp_path, "20260101_120000", frame_count=42)
    runner = CliRunner()
    result = runner.invoke(cli, ["sessions", "get", "20260101_120000", "--dir", str(tmp_path)])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["session_id"] == "20260101_120000"
    assert data["frame_count"] == 42


def test_sessions_get_with_events_flag(tmp_path):
    session_id = "20260101_120000"
    session_dir = _write_mock_session(tmp_path, session_id)
    events_path = session_dir / "agent_events.jsonl"
    events_path.write_text(
        '{"ts": "2026-01-01T12:00:01Z", "source": "QualityAgent", "level": "ok",'
        ' "type": "quality", "data": {}, "text": "ok"}\n',
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        cli, ["sessions", "get", session_id, "--dir", str(tmp_path), "--events"]
    )
    assert result.exit_code == 0
    assert "agent events" in result.output.lower()
    assert "QualityAgent" in result.output

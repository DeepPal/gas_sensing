"""
Tests for spectraagent.webapp.session_writer.SessionWriter

All tests use the tmp_path pytest fixture to avoid writing to the real filesystem.
"""
import json
from pathlib import Path

import pytest

from spectraagent.webapp.session_writer import SessionWriter


@pytest.fixture
def writer(tmp_path):
    return SessionWriter(sessions_dir=tmp_path / "sessions")


# ---------------------------------------------------------------------------
# start_session
# ---------------------------------------------------------------------------


def test_start_session_creates_directory(writer, tmp_path):
    """start_session creates the session directory and session_meta.json."""
    writer.start_session("20260327_120000", {"gas_label": "Ethanol", "target_concentration": 0.5})
    session_dir = tmp_path / "sessions" / "20260327_120000"
    assert session_dir.is_dir()
    assert (session_dir / "session_meta.json").exists()


def test_start_session_meta_has_required_fields(writer, tmp_path):
    """session_meta.json contains session_id, started_at=set, stopped_at=None, and passed meta."""
    writer.start_session("20260327_120001", {"gas_label": "CO2", "target_concentration": 1.0})
    meta_path = tmp_path / "sessions" / "20260327_120001" / "session_meta.json"
    meta = json.loads(meta_path.read_text())
    assert meta["session_id"] == "20260327_120001"
    assert meta["started_at"] is not None
    assert meta["stopped_at"] is None
    assert meta["gas_label"] == "CO2"
    assert meta["target_concentration"] == 1.0


# ---------------------------------------------------------------------------
# append_event
# ---------------------------------------------------------------------------


def test_append_event_writes_jsonl(writer, tmp_path):
    """append_event writes a JSON line to agent_events.jsonl."""
    writer.start_session("20260327_120002", {})
    writer.append_event({"source": "QualityAgent", "level": "ok", "type": "quality", "text": "ok"})
    events_path = tmp_path / "sessions" / "20260327_120002" / "agent_events.jsonl"
    lines = events_path.read_text().splitlines()
    assert len(lines) == 1
    evt = json.loads(lines[0])
    assert evt["source"] == "QualityAgent"


def test_append_event_noop_when_no_session(writer):
    """append_event is a no-op when no session is active — must not raise."""
    writer.append_event({"source": "X", "level": "ok", "type": "t", "text": "t"})  # no exception


# ---------------------------------------------------------------------------
# stop_session
# ---------------------------------------------------------------------------


def test_stop_session_updates_meta(writer, tmp_path):
    """stop_session writes stopped_at and frame_count to session_meta.json."""
    writer.start_session("20260327_120003", {})
    writer.stop_session(frame_count=42)
    meta = json.loads(
        (tmp_path / "sessions" / "20260327_120003" / "session_meta.json").read_text()
    )
    assert meta["stopped_at"] is not None
    assert meta["frame_count"] == 42


def test_stop_session_noop_when_no_session(writer):
    """stop_session is a no-op when no session is active — must not raise."""
    writer.stop_session()  # no exception


# ---------------------------------------------------------------------------
# list_sessions
# ---------------------------------------------------------------------------


def test_list_sessions_empty_dir(writer):
    """list_sessions returns [] when the sessions directory does not exist."""
    assert writer.list_sessions() == []


def test_list_sessions_returns_metadata(writer):
    """list_sessions returns one entry after a completed session."""
    writer.start_session("20260327_120004", {"gas_label": "Ethanol"})
    writer.stop_session()
    sessions = writer.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "20260327_120004"


def test_list_sessions_newest_first(writer):
    """list_sessions returns sessions sorted newest first by started_at."""
    import time
    writer.start_session("session_A", {"gas_label": "A"})
    writer.stop_session()
    time.sleep(0.01)  # ensure distinct started_at timestamps
    writer.start_session("session_B", {"gas_label": "B"})
    writer.stop_session()
    sessions = writer.list_sessions()
    assert len(sessions) == 2
    # B was started after A, so B should appear first
    assert sessions[0]["session_id"] == "session_B"
    assert sessions[1]["session_id"] == "session_A"


# ---------------------------------------------------------------------------
# get_session
# ---------------------------------------------------------------------------


def test_get_session_returns_meta_and_events(writer):
    """get_session returns meta dict with an 'events' list."""
    writer.start_session("20260327_120005", {"gas_label": "Ethanol"})
    writer.append_event({"source": "QA", "level": "ok", "type": "quality", "text": "ok"})
    writer.stop_session()
    result = writer.get_session("20260327_120005")
    assert result is not None
    assert result["session_id"] == "20260327_120005"
    assert isinstance(result["events"], list)
    assert len(result["events"]) == 1
    assert result["events"][0]["source"] == "QA"


def test_get_session_nonexistent_returns_none(writer):
    """get_session returns None for an unknown session_id."""
    assert writer.get_session("nonexistent_session") is None


# ---------------------------------------------------------------------------
# C4: Atomic JSON writes — meta file is always valid JSON
# ---------------------------------------------------------------------------


def test_start_session_meta_is_valid_json(writer, tmp_path):
    """C4: session_meta.json written by start_session is always valid JSON."""
    writer.start_session("atomic_test_start", {"gas_label": "Ethanol"})
    meta_path = tmp_path / "sessions" / "atomic_test_start" / "session_meta.json"
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    assert data["session_id"] == "atomic_test_start"


def test_stop_session_meta_is_valid_json(writer, tmp_path):
    """C4: session_meta.json updated by stop_session is always valid JSON."""
    writer.start_session("atomic_test_stop", {"gas_label": "CO2"})
    writer.stop_session(frame_count=100)
    meta_path = tmp_path / "sessions" / "atomic_test_stop" / "session_meta.json"
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    assert data["frame_count"] == 100
    assert data["stopped_at"] is not None


def test_no_tmp_file_left_after_start(writer, tmp_path):
    """C4: no .tmp_ files remain after start_session."""
    writer.start_session("atomic_test_tmp", {})
    session_dir = tmp_path / "sessions" / "atomic_test_tmp"
    tmp_files = list(session_dir.glob(".tmp_*"))
    assert tmp_files == []


def test_no_tmp_file_left_after_stop(writer, tmp_path):
    """C4: no .tmp_ files remain after stop_session."""
    writer.start_session("atomic_test_tmp2", {})
    writer.stop_session(frame_count=7)
    session_dir = tmp_path / "sessions" / "atomic_test_tmp2"
    tmp_files = list(session_dir.glob(".tmp_*"))
    assert tmp_files == []


def test_append_frame_result_writes_csv_row(writer, tmp_path):
    """append_frame_result writes a parseable CSV row."""
    import csv
    writer.start_session("csv_test_session", {})
    writer.append_frame_result({
        "frame": 1,
        "timestamp": "2026-01-01T00:00:00Z",
        "peak_wavelength": 717.5,
        "wavelength_shift": -0.3,
        "concentration_ppm": 12.4,
        "ci_low": 10.1,
        "ci_high": 14.7,
        "snr": 25.6,
        "gas_type": "Ethanol",
        "confidence_score": 0.97,
    })
    csv_path = tmp_path / "sessions" / "csv_test_session" / "pipeline_results.csv"
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert float(rows[0]["concentration_ppm"]) == pytest.approx(12.4)

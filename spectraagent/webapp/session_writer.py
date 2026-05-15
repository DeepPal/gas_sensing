"""
spectraagent.webapp.session_writer
=====================================
SessionWriter — persists session metadata, agent events, and pipeline results.

Each active session creates a directory under ``sessions_dir / session_id /``:
  session_meta.json    — metadata (session_id, gas_label, timestamps, frame_count)
  agent_events.jsonl   — one JSON line per AgentEvent (append-only)
  pipeline_results.csv — one row per frame: peak_wavelength, shift, concentration, SNR…

``append_event()`` and ``append_frame_result()`` are no-ops when no session is active.
``stop_session()`` is a no-op when no session is active.
All three are safe to call unconditionally from the asyncio event loop.
"""
from __future__ import annotations

import contextlib
import csv
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import IO, Optional

log = logging.getLogger(__name__)


def _write_json_atomic(path: Path, data: dict) -> None:
    """Write *data* as JSON to *path* using a temp-file-then-rename pattern.

    The rename is atomic on NTFS and POSIX: callers always see either the
    old file or the complete new file, never a half-written one.
    """
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    Path(tmp).replace(path)


_FRAME_RESULT_COLUMNS = [
    "frame", "timestamp", "peak_wavelength", "wavelength_shift",
    "concentration_ppm", "ci_low", "ci_high", "snr",
    "gas_type", "confidence_score",
]


class SessionWriter:
    """Persist session data to ``sessions_dir/{session_id}/``.

    Parameters
    ----------
    sessions_dir:
        Base directory for all session subdirectories.
        Created on demand when ``start_session()`` is called.
        Defaults to ``output/sessions`` relative to the working directory.
    """

    def __init__(self, sessions_dir: Path = Path("output/sessions")) -> None:
        self._dir = sessions_dir
        self._active_dir: Optional[Path] = None
        self._events_file: Optional[IO[str]] = None
        self._results_file: Optional[IO[str]] = None
        self._results_writer: Optional[csv.DictWriter] = None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(self, session_id: str, meta: dict) -> Path:
        """Create session directory and write initial session_meta.json.

        Opens ``agent_events.jsonl`` for appending.
        If a previous session was not stopped cleanly, it is closed first.
        """
        if self._events_file is not None:
            self.stop_session()

        session_dir = self._dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        full_meta: dict = {
            **meta,
            "session_id": session_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "stopped_at": None,
        }
        _write_json_atomic(session_dir / "session_meta.json", full_meta)

        self._events_file = open(  # noqa: SIM115
            session_dir / "agent_events.jsonl", "a", encoding="utf-8"
        )
        self._results_file = open(  # noqa: SIM115
            session_dir / "pipeline_results.csv", "a", newline="", encoding="utf-8"
        )
        self._results_writer = csv.DictWriter(
            self._results_file,
            fieldnames=_FRAME_RESULT_COLUMNS,
            extrasaction="ignore",
        )
        # Write header only when creating a new (empty) file.
        if (session_dir / "pipeline_results.csv").stat().st_size == 0:
            self._results_writer.writeheader()
            self._results_file.flush()
        self._active_dir = session_dir
        log.info("SessionWriter: started session %s", session_id)
        return session_dir

    def append_event(self, event_dict: dict) -> None:
        """Append an AgentEvent dict as a JSON line. No-op when no session active."""
        if self._events_file is None:
            return
        try:
            self._events_file.write(json.dumps(event_dict) + "\n")
            self._events_file.flush()
        except Exception as exc:
            log.warning("SessionWriter.append_event failed: %s", exc)

    def append_frame_result(self, row: dict) -> None:
        """Append one frame's pipeline result to pipeline_results.csv.

        No-op when no session is active. Never raises.

        Parameters
        ----------
        row:
            Dict with keys matching ``_FRAME_RESULT_COLUMNS``; extra keys are
            silently ignored, missing keys are written as empty strings.
        """
        if self._results_writer is None or self._results_file is None:
            return
        try:
            self._results_writer.writerow({k: row.get(k, "") for k in _FRAME_RESULT_COLUMNS})
            self._results_file.flush()
        except Exception as exc:
            log.warning("SessionWriter.append_frame_result failed: %s", exc)

    def stop_session(self, frame_count: int = 0) -> None:
        """Update session_meta.json with stopped_at + frame_count; close events file.

        No-op when no session is active. Never raises.
        """
        if self._active_dir is None:
            return

        meta_path = self._active_dir / "session_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["stopped_at"] = datetime.now(timezone.utc).isoformat()
                meta["frame_count"] = frame_count
                _write_json_atomic(meta_path, meta)
            except Exception as exc:
                log.warning("SessionWriter.stop_session: failed to update meta: %s", exc)

        if self._events_file is not None:
            with contextlib.suppress(Exception):
                self._events_file.close()
            self._events_file = None

        if self._results_file is not None:
            with contextlib.suppress(Exception):
                self._results_file.close()
            self._results_file = None
            self._results_writer = None

        log.info("SessionWriter: stopped session at %s", self._active_dir)
        self._active_dir = None

    # ------------------------------------------------------------------
    # Read access
    # ------------------------------------------------------------------

    def list_sessions(self) -> list:
        """Return all session metadata dicts, sorted newest first.

        Returns [] if sessions directory does not exist.
        """
        if not self._dir.exists():
            return []
        sessions = []
        for meta_path in self._dir.glob("*/session_meta.json"):
            try:
                sessions.append(json.loads(meta_path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("SessionWriter.list_sessions: skipping %s: %s", meta_path, exc)
        sessions.sort(key=lambda s: s.get("started_at", ""), reverse=True)
        return sessions

    def get_session(self, session_id: str) -> Optional[dict]:
        """Return session metadata + last 100 agent events, or None if not found."""
        meta_path = self._dir / session_id / "session_meta.json"
        if not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("SessionWriter.get_session: failed to read meta: %s", exc)
            return None

        events: list[dict] = []
        events_path = self._dir / session_id / "agent_events.jsonl"
        if events_path.exists():
            try:
                lines = events_path.read_text(encoding="utf-8").splitlines()
                for line in lines[-100:]:
                    with contextlib.suppress(json.JSONDecodeError):
                        events.append(json.loads(line))
            except OSError as exc:
                log.warning("SessionWriter.get_session: failed to read events: %s", exc)

        return {**meta, "events": events}

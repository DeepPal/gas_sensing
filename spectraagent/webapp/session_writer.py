"""
spectraagent.webapp.session_writer
=====================================
SessionWriter — persists session metadata and agent events to disk.

Each active session creates a directory under ``sessions_dir / session_id /``:
  session_meta.json    — metadata (session_id, gas_label, timestamps, frame_count)
  agent_events.jsonl   — one JSON line per AgentEvent (append-only)

``append_event()`` is a no-op when no session is active.
``stop_session()`` is a no-op when no session is active.
Both are safe to call unconditionally from the asyncio event loop.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Optional

log = logging.getLogger(__name__)


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
            "session_id": session_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "stopped_at": None,
        }
        full_meta.update(meta)
        (session_dir / "session_meta.json").write_text(
            json.dumps(full_meta, indent=2), encoding="utf-8"
        )

        self._events_file = open(
            session_dir / "agent_events.jsonl", "a", encoding="utf-8"
        )
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
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            except Exception as exc:
                log.warning("SessionWriter.stop_session: failed to update meta: %s", exc)

        if self._events_file is not None:
            try:
                self._events_file.close()
            except Exception:
                pass
            self._events_file = None

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
        for meta_path in sorted(self._dir.glob("*/session_meta.json"), reverse=True):
            try:
                sessions.append(json.loads(meta_path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("SessionWriter.list_sessions: skipping %s: %s", meta_path, exc)
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
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            except OSError as exc:
                log.warning("SessionWriter.get_session: failed to read events: %s", exc)

        return {**meta, "events": events}

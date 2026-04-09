from __future__ import annotations

from pathlib import Path
import time

from dashboard.backups import BackupManager, automatic_backup


def _make_session_dir(tmp_path: Path) -> Path:
    session_dir = tmp_path / "output" / "sessions" / "session_001"
    session_dir.mkdir(parents=True)
    (session_dir / "spectrum.csv").write_text("wavelength,intensity\n700,1.0\n", encoding="utf-8")
    (session_dir / "meta.json").write_text('{"gas":"ethanol"}', encoding="utf-8")
    return session_dir


def test_backup_session_creates_archive_and_manifest(tmp_path: Path) -> None:
    session_dir = _make_session_dir(tmp_path)

    manager = BackupManager(app_root=tmp_path, backup_dir=tmp_path / "backups")
    metadata = manager.backup_session(session_dir)

    backup_path = Path(metadata["backup_path"])
    manifest_path = manager.backup_dir / f"{metadata['backup_name']}.manifest.json"

    assert backup_path.exists()
    assert manifest_path.exists()
    assert metadata["sha256"]
    assert metadata["size_mb"] >= 0.0


def test_list_backups_returns_latest_first(tmp_path: Path) -> None:
    session_dir = _make_session_dir(tmp_path)
    manager = BackupManager(app_root=tmp_path, backup_dir=tmp_path / "backups")

    first = manager.backup_session(session_dir)
    time.sleep(1.1)
    second = manager.backup_session(session_dir)

    backups = manager.list_backups()

    assert backups
    names = [b["backup_name"] for b in backups]
    assert first["backup_name"] in names
    assert second["backup_name"] in names


def test_automatic_backup_returns_created_archive_path(tmp_path: Path) -> None:
    session_dir = _make_session_dir(tmp_path)

    backup_path = automatic_backup(session_dir=session_dir, backup_dir=tmp_path / "backups")

    assert backup_path.exists()
    assert backup_path.suffixes[-2:] == [".tar", ".gz"]

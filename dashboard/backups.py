"""
dashboard/backups.py - Automated backup management
===================================================

Provides reliable backup and archival of experimental sessions with:
- Timestamped snapshots
- Integrity hashing (SHA256)
- Compressed storage (tar.gz)
- Automatic cleanup of old backups
- Backup manifest with provenance

Suitable for research lab where data is precious and must not be lost.
"""

from __future__ import annotations

import datetime
import json
import logging
import shutil
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class BackupManager:
    """Manage backups of experimental sessions with integrity checking."""

    def __init__(
        self,
        app_root: Path | None = None,
        backup_dir: Path | None = None,
        retention_days: int = 90,
    ):
        self.app_root = app_root or Path(__file__).resolve().parents[1]
        self.backup_dir = backup_dir or self.app_root / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days

    def backup_session(self, session_dir: Path) -> dict[str, Any]:
        """
        Create a timestamped backup of a session directory.
        
        Parameters
        ----------
        session_dir : Path
            Path to session directory to backup (e.g., output/sessions/session_20260328_...)
        
        Returns
        -------
        dict
            Backup metadata including hash, size, timestamp
        """
        if not session_dir.exists():
            raise FileNotFoundError(f"Session directory not found: {session_dir}")

        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_name = f"{session_dir.name}_{timestamp}.tar.gz"
        backup_path = self.backup_dir / backup_name
        archive_root = self.backup_dir / f"{session_dir.name}_{timestamp}"

        log.info("💾 Creating backup: %s", backup_name)

        # Create compressed archive
        try:
            shutil.make_archive(
                str(archive_root),
                format="gztar",
                root_dir=session_dir.parent,
                base_dir=session_dir.name,
            )

            # Compute integrity hash
            sha256 = self._compute_file_hash(backup_path)
            size_mb = backup_path.stat().st_size / (1024 * 1024)

            metadata = {
                "backup_name": backup_name,
                "backup_path": str(backup_path),
                "source_session": str(session_dir),
                "timestamp": timestamp,
                "size_mb": round(size_mb, 2),
                "sha256": sha256,
                "retention_until": (
                    datetime.datetime.now(datetime.timezone.utc)
                    + datetime.timedelta(days=self.retention_days)
                ).isoformat(),
            }

            # Save metadata manifest
            manifest_path = self.backup_dir / f"{backup_name}.manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

            log.info(
                "✓ Backup created: %s (%s MB, SHA256: %s)",
                backup_name,
                metadata["size_mb"],
                sha256[:8],
            )

            return metadata

        except Exception as e:
            log.error("Backup failed: %s", e)
            raise

    def cleanup_old_backups(self, dryrun: bool = False) -> dict[str, Any]:
        """
        Delete backups older than retention_days.
        
        Parameters
        ----------
        dryrun : bool
            If True, report what would be deleted without deleting
        
        Returns
        -------
        dict
            Summary of deleted backups
        """
        log.info("🧹 Checking for old backups (retention: %d days)", self.retention_days)

        cutoff_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            days=self.retention_days
        )

        deleted = []
        freed_mb = 0.0

        for backup_file in self.backup_dir.glob("*.tar.gz"):
            # Check manifest for retention info
            manifest_path = backup_file.with_name(f"{backup_file.name}.manifest.json")

            if manifest_path.exists():
                try:
                    with open(manifest_path) as f:
                        manifest = json.load(f)
                    retention_until = datetime.datetime.fromisoformat(
                        manifest["retention_until"]
                    )

                    if retention_until < cutoff_date:
                        freed = backup_file.stat().st_size / (1024 * 1024)
                        if not dryrun:
                            backup_file.unlink()
                            manifest_path.unlink()
                            log.info("  🗑️  Deleted: %s", backup_file.name)
                        deleted.append(backup_file.name)
                        freed_mb += freed

                except Exception as e:
                    log.warning("Could not parse manifest %s: %s", manifest_path, e)

        result = {
            "deleted_count": len(deleted),
            "freed_mb": round(freed_mb, 2),
            "deleted_files": deleted,
            "dryrun": dryrun,
        }

        log.info("✓ Cleanup complete: Freed %s MB", result["freed_mb"])
        return result

    def list_backups(self) -> list[dict[str, Any]]:
        """List all available backups with metadata."""
        backups = []

        for manifest_path in sorted(self.backup_dir.glob("*.manifest.json")):
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                backups.append(manifest)
            except Exception as e:
                log.warning("Could not read manifest %s: %s", manifest_path, e)

        return sorted(backups, key=lambda x: x["timestamp"], reverse=True)

    def restore_backup(self, backup_name: str, restore_dir: Path) -> Path:
        """
        Restore a backup to a specified directory.
        
        Parameters
        ----------
        backup_name : str
            Name of backup file (e.g., session_20260328_123456.tar.gz)
        restore_dir : Path
            Where to restore the contents
        
        Returns
        -------
        Path
            Path to restored session directory
        """
        backup_path = self.backup_dir / backup_name

        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")

        log.info("♻️  Restoring backup: %s", backup_name)

        restore_dir.mkdir(parents=True, exist_ok=True)

        try:
            shutil.unpack_archive(str(backup_path), extract_dir=str(restore_dir))
            log.info("✓ Backup restored to: %s", restore_dir)
            return restore_dir
        except Exception as e:
            log.error("Restore failed: %s", e)
            raise

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """Compute SHA256 hash of a file for integrity verification."""
        import hashlib

        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


def automatic_backup(
    session_dir: Path,
    backup_dir: Path | None = None,
) -> Path:
    """
    Convenience function for one-off backup.
    
    Parameters
    ----------
    session_dir : Path
        Session directory to backup
    backup_dir : Path
        Optional backup directory (default: backups/)
    
    Returns
    -------
    Path
        Path to created backup file
    """
    manager = BackupManager(backup_dir=backup_dir)
    metadata = manager.backup_session(session_dir)
    return Path(metadata["backup_path"])

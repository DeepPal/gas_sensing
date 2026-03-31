"""
dashboard/reproducibility.py - Scientific reproducibility tracking
==================================================================

Captures metadata about every experiment run for scientific integrity:
- Code version (git commit hash)
- Configuration snapshot
- Dataset identity
- Hardware/environment info
- Quality metrics
- Timestamp and operator

This ensures every result can be traced back to exact code, config, and data.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class ReproducibilityManifest:
    """
    Scientific reproducibility manifest for a gas sensing session.
    
    Captures all information needed to reproduce an experiment exactly.
    """

    def __init__(
        self,
        session_id: str,
        app_root: Path | None = None,
        operator: str = "unknown",
    ):
        self.session_id = session_id
        self.app_root = app_root or Path(__file__).resolve().parents[1]
        self.operator = operator
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def get_git_info(self) -> dict[str, Any]:
        """Get version control information."""
        try:
            # Get current commit hash
            commit = subprocess.run(
                ["git", "-C", str(self.app_root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()

            # Get branch name
            branch = subprocess.run(
                ["git", "-C", str(self.app_root), "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()

            # Get short status (are there uncommitted changes?)
            status = subprocess.run(
                ["git", "-C", str(self.app_root), "status", "--porcelain"],
                capture_output=True,
                text=True,
            ).stdout.strip()

            return {
                "commit_hash": commit,
                "branch": branch,
                "uncommitted_changes": len(status) > 0,
                "status": "CLEAN" if not status else "MODIFIED",
            }
        except Exception as e:
            log.warning("Could not get git info: %s", e)
            return {
                "commit_hash": "unknown",
                "branch": "unknown",
                "uncommitted_changes": True,
                "status": "ERROR",
            }

    def get_environment_info(self) -> dict[str, Any]:
        """Get system and Python environment information."""
        import sys

        return {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": platform.platform(),
            "hostname": platform.node(),
            "processor": platform.processor() or "unknown",
        }

    def get_config_snapshot(self, config_path: Path | None = None) -> dict[str, Any]:
        """Get snapshot of active configuration."""
        config_path = config_path or self.app_root / "config" / "config.yaml"

        try:
            import yaml

            with open(config_path) as f:
                config = yaml.safe_load(f)
            return {
                "config_file": str(config_path),
                "snapshot": config,
                "hash": self._hash_dict(config),
            }
        except Exception as e:
            log.warning("Could not read config: %s", e)
            return {
                "config_file": str(config_path),
                "snapshot": None,
                "error": str(e),
            }

    def get_manifest(self) -> dict[str, Any]:
        """Get complete reproducibility manifest."""
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "operator": self.operator,
            "application": {
                "name": "Au-MIP LSPR Gas Sensing Platform",
                "version": "1.0.0",
                "purpose": "Research-grade optical gas sensing",
            },
            "version_control": self.get_git_info(),
            "environment": self.get_environment_info(),
            "configuration": self.get_config_snapshot(),
            "notes": (
                "This manifest uniquely identifies the code, configuration, "
                "and environment used for this experiment. Use it to reproduce "
                "the exact same analysis."
            ),
        }

    def save(self, output_dir: Path) -> Path:
        """
        Save manifest to a JSON file.
        
        Parameters
        ----------
        output_dir : Path
            Directory where manifest will be saved
        
        Returns
        -------
        Path
            Path to the saved manifest file
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / f"{self.session_id}_manifest.json"

        manifest = self.get_manifest()

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        log.info("✓ Reproducibility manifest saved: %s", manifest_path)
        return manifest_path

    @staticmethod
    def _hash_dict(d: dict) -> str:
        """Compute hash of a dictionary for integrity checking."""
        import hashlib

        json_str = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def create_session_manifest(session_id: str, output_dir: Path) -> Path:
    """
    Convenience function to create and save a manifest.
    
    Parameters
    ----------
    session_id : str
        Unique session identifier
    output_dir : Path
        Where to save the manifest
    
    Returns
    -------
    Path
        Path to saved manifest file
    """
    manifest = ReproducibilityManifest(session_id, operator="streamlit_user")
    return manifest.save(output_dir)

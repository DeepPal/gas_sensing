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

import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import subprocess
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

    def get_dependency_snapshot(self) -> dict[str, Any]:
        """Collect installed dependency metadata for stronger reproducibility."""
        snapshot: dict[str, Any] = {
            "status": "ok",
            "package_count": 0,
            "freeze_hash": "",
            "core_packages": {},
        }

        try:
            distributions = sorted(
                (
                    f"{dist.metadata['Name']}=={dist.version}"
                    for dist in importlib.metadata.distributions()
                    if dist.metadata.get("Name")
                ),
                key=str.lower,
            )
            freeze_blob = "\n".join(distributions)
            snapshot["package_count"] = len(distributions)
            snapshot["freeze_hash"] = hashlib.sha256(freeze_blob.encode("utf-8")).hexdigest()
        except Exception as exc:
            snapshot["status"] = "error"
            snapshot["error"] = f"distribution_scan_failed: {exc}"

        core_versions: dict[str, str] = {}
        for pkg in (
            "numpy",
            "pandas",
            "scipy",
            "scikit-learn",
            "fastapi",
            "uvicorn",
            "pydantic",
            "torch",
        ):
            with contextlib.suppress(Exception):
                core_versions[pkg] = importlib.metadata.version(pkg)
        snapshot["core_packages"] = core_versions
        return snapshot

    def get_config_snapshot(self, config_path: Path | None = None) -> dict[str, Any]:
        """Get snapshot of active configuration."""
        config_path = config_path or self.app_root / "config" / "config.yaml"

        try:
            import yaml

            with open(config_path, encoding="utf-8") as f:
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
                "name": "SpectraAgent — Spectrometer-Based Sensing Platform",
                "version": "1.0.0",
                "purpose": "Research-grade optical gas sensing",
            },
            "version_control": self.get_git_info(),
            "environment": self.get_environment_info(),
            "dependencies": self.get_dependency_snapshot(),
            "configuration": self.get_config_snapshot(),
            "notes": (
                "This manifest uniquely identifies the code, configuration, "
                "and environment used for this experiment. Use it to reproduce "
                "the exact same analysis."
            ),
        }

    def build_artifact_checksums(self, output_dir: Path) -> list[dict[str, str]]:
        """Return SHA256 checksums for files already present in output_dir.

        The list is relative to output_dir so it remains portable across machines.
        Manifest files are excluded to avoid self-referential hash churn.
        """
        checksums: list[dict[str, str]] = []
        if not output_dir.exists():
            return checksums

        for file_path in sorted(p for p in output_dir.rglob("*") if p.is_file()):
            if file_path.name.endswith("_manifest.json"):
                continue
            rel = file_path.relative_to(output_dir).as_posix()
            checksums.append({
                "path": rel,
                "sha256": self._hash_file(file_path),
            })
        return checksums

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
        manifest["artifact_checksums"] = self.build_artifact_checksums(output_dir)

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        log.info("✓ Reproducibility manifest saved: %s", manifest_path)
        return manifest_path

    @staticmethod
    def _hash_dict(d: dict) -> str:
        """Compute hash of a dictionary for integrity checking."""
        json_str = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    @staticmethod
    def _hash_file(path: Path) -> str:
        """Compute SHA256 hash of a file."""
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()


def verify_manifest_artifacts(manifest_path: Path) -> tuple[bool, list[str]]:
    """Validate checksummed artifacts listed in a reproducibility manifest."""
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, [f"Could not parse manifest {manifest_path}: {exc}"]

    checksums = payload.get("artifact_checksums", [])
    if not isinstance(checksums, list):
        return False, ["artifact_checksums must be a list"]

    base_dir = manifest_path.parent
    errors: list[str] = []
    for item in checksums:
        if not isinstance(item, dict):
            errors.append("artifact_checksums entries must be objects")
            continue
        rel = item.get("path")
        expected = item.get("sha256")
        if not isinstance(rel, str) or not isinstance(expected, str):
            errors.append("artifact_checksums entries require string path and sha256")
            continue

        path = base_dir / rel
        if not path.exists() or not path.is_file():
            errors.append(f"Missing artifact: {rel}")
            continue

        actual = ReproducibilityManifest._hash_file(path)
        if actual != expected:
            errors.append(f"Checksum mismatch: {rel}")

    return len(errors) == 0, errors


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

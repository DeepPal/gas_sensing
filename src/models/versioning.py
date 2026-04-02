"""Model versioning — timestamped saves, manifests, promotion, and rollback.

Each saved model lives in its own subdirectory::

    output/model_versions/
    ├── autoencoder_20260401_143022_abc1234/
    │   ├── model.pt          (PyTorch state dict or joblib artifact)
    │   ├── manifest.json     (metadata: metrics, config, git hash, timestamp)
    │   └── config.json       (model hyperparameters snapshot)
    └── autoencoder_latest -> autoencoder_20260401_143022_abc1234/   (symlink)

The ``ModelVersionStore`` class:

- ``save(model, name, metrics, config)`` — archive a new version, return version ID
- ``list_versions(name)`` — all saved versions, sorted newest-first
- ``load(name, version_id=None)`` — load a specific (or latest) version
- ``promote(name, version_id)`` — mark a version as production (updates ``_latest`` symlink)
- ``compare(name, metric)`` — rank versions by a scalar metric
- ``delete(name, version_id)`` — remove a specific version

Usage
-----
::

    from src.models.versioning import ModelVersionStore

    store = ModelVersionStore("output/model_versions")

    # Save after training
    version_id = store.save(
        model=autoencoder,
        name="autoencoder",
        metrics={"val_loss": 0.0032},
        config={"embed_dim": 32, "n_epochs": 100},
    )

    # Promote the best version
    best = store.compare("autoencoder", metric="val_loss", lower_is_better=True)[0]
    store.promote("autoencoder", best.version_id)

    # Load production model
    model = store.load("autoencoder")
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "ModelVersionStore",
    "VersionRecord",
]

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class VersionRecord:
    """Metadata for a single saved model version."""
    name: str
    version_id: str          # e.g. "20260401_143022_abc1234"
    timestamp: str           # ISO-8601 UTC
    git_commit: str          # short hash or "unknown"
    metrics: dict[str, float]
    config: dict[str, Any]
    artifact_path: str       # relative to store root
    is_promoted: bool = False
    notes: str = ""


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class ModelVersionStore:
    """Filesystem-backed model version store.

    All versions are stored under *store_root*.  Each version gets its own
    subdirectory named ``{name}_{version_id}``.

    Parameters
    ----------
    store_root :
        Base directory for all versioned models.  Created on first save.
    """

    _MANIFEST_FILE = "manifest.json"
    _MODEL_FILE_PT = "model.pt"
    _MODEL_FILE_JOBLIB = "model.joblib"
    _CONFIG_FILE = "config.json"
    _LATEST_MARKER = "_latest"

    def __init__(self, store_root: str | Path = "output/model_versions") -> None:
        self._root = Path(store_root)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        model: Any,
        name: str,
        metrics: dict[str, float] | None = None,
        config: dict[str, Any] | None = None,
        notes: str = "",
        promote: bool = False,
    ) -> str:
        """Serialise *model* and write a versioned checkpoint.

        Supports:
        - **PyTorch nn.Module** — saved with ``torch.save(state_dict)``
        - **sklearn / joblib** objects — saved with ``joblib.dump``
        - **ndarray** — saved with ``np.save``

        Parameters
        ----------
        model :
            Trained model object.
        name :
            Logical model name (e.g. ``"autoencoder"``).
        metrics :
            Scalar evaluation metrics to record (e.g. ``{"val_loss": 0.003}``).
        config :
            Hyperparameter snapshot.
        notes :
            Free-text annotation (e.g. analyte name, experiment ID).
        promote :
            If True, immediately promote this version as the active production
            version.

        Returns
        -------
        version_id : str
            The unique version identifier for this checkpoint.
        """
        version_id = self._make_version_id()
        version_dir = self._root / f"{name}_{version_id}"
        version_dir.mkdir(parents=True, exist_ok=True)

        # Serialise the model
        artifact_path = self._save_artifact(model, version_dir)

        # Write config snapshot
        if config:
            (version_dir / self._CONFIG_FILE).write_text(
                json.dumps(_serialise(config), indent=2), encoding="utf-8"
            )

        # Build and write manifest
        record = VersionRecord(
            name=name,
            version_id=version_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            git_commit=_git_short_hash(),
            metrics={k: float(v) for k, v in (metrics or {}).items()},
            config=config or {},
            artifact_path=str(artifact_path.relative_to(self._root)),
            notes=notes,
        )
        self._write_manifest(version_dir, record)

        log.info("Saved model version %s/%s → %s", name, version_id, version_dir)

        if promote:
            self.promote(name, version_id)

        return version_id

    def list_versions(self, name: str) -> list[VersionRecord]:
        """Return all saved versions for *name*, sorted newest-first."""
        records: list[VersionRecord] = []
        if not self._root.exists():
            return records
        for version_dir in sorted(self._root.glob(f"{name}_*"), reverse=True):
            if not version_dir.is_dir():
                continue
            manifest_path = version_dir / self._MANIFEST_FILE
            if not manifest_path.exists():
                continue
            try:
                record = self._read_manifest(manifest_path)
                records.append(record)
            except Exception as exc:
                log.warning("Could not read manifest %s: %s", manifest_path, exc)
        return records

    def load(
        self,
        name: str,
        version_id: str | None = None,
    ) -> Any:
        """Load and return the model object for *name*.

        Parameters
        ----------
        version_id :
            Specific version to load.  If None, loads the promoted version
            (``_latest`` marker), or the most recent version if no promotion
            exists.

        Returns
        -------
        Deserialised model object.

        Raises
        ------
        FileNotFoundError
            If no version exists for *name*.
        """
        version_dir = self._resolve_version_dir(name, version_id)
        return self._load_artifact(version_dir)

    def get_record(
        self,
        name: str,
        version_id: str | None = None,
    ) -> VersionRecord:
        """Return the manifest record without loading the model."""
        version_dir = self._resolve_version_dir(name, version_id)
        return self._read_manifest(version_dir / self._MANIFEST_FILE)

    def promote(self, name: str, version_id: str) -> None:
        """Mark *version_id* as the active production version.

        Updates the ``{name}_latest`` marker file so subsequent
        ``load(name)`` calls return this version.
        """
        target_dir = self._root / f"{name}_{version_id}"
        if not target_dir.exists():
            raise FileNotFoundError(
                f"Version {version_id!r} for {name!r} not found at {target_dir}"
            )

        # Demote current promoted version
        for existing in self.list_versions(name):
            if existing.is_promoted and existing.version_id != version_id:
                ex_dir = self._root / f"{name}_{existing.version_id}"
                ex_manifest = ex_dir / self._MANIFEST_FILE
                if ex_manifest.exists():
                    existing.is_promoted = False
                    self._write_manifest(ex_dir, existing)

        # Promote this version
        record = self._read_manifest(target_dir / self._MANIFEST_FILE)
        record.is_promoted = True
        self._write_manifest(target_dir, record)

        # Write latest-marker file (plain text, no symlinks — Windows-safe)
        marker = self._root / f"{name}{self._LATEST_MARKER}"
        marker.write_text(version_id, encoding="utf-8")

        log.info("Promoted %s/%s as production", name, version_id)

    def compare(
        self,
        name: str,
        metric: str,
        lower_is_better: bool = True,
    ) -> list[VersionRecord]:
        """Return all versions sorted by *metric* (best first).

        Versions missing the metric are placed last.
        """
        versions = self.list_versions(name)

        def sort_key(r: VersionRecord) -> float:
            val = r.metrics.get(metric, float("inf") if lower_is_better else float("-inf"))
            return val if lower_is_better else -val

        return sorted(versions, key=sort_key)

    def delete(self, name: str, version_id: str) -> None:
        """Permanently remove a specific version checkpoint.

        Will not delete the currently promoted version.
        """
        record = self.get_record(name, version_id)
        if record.is_promoted:
            raise ValueError(
                f"Cannot delete promoted version {version_id!r}. "
                "Promote a different version first."
            )
        version_dir = self._root / f"{name}_{version_id}"
        shutil.rmtree(version_dir)
        log.info("Deleted version %s/%s", name, version_id)

    def summary(self, name: str) -> str:
        """Human-readable summary of all versions for *name*."""
        versions = self.list_versions(name)
        if not versions:
            return f"No saved versions for {name!r}."
        lines = [f"Model: {name}  ({len(versions)} version(s))"]
        lines.append("-" * 60)
        for v in versions:
            promoted = " ★ PROMOTED" if v.is_promoted else ""
            metrics_str = "  ".join(
                f"{k}={val:.4g}" for k, val in sorted(v.metrics.items())
            )
            lines.append(
                f"  {v.version_id}  {v.timestamp[:19]}  {v.git_commit}"
                f"  [{metrics_str}]{promoted}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_version_id(self) -> str:
        """Create a version ID: timestamp (with milliseconds) + short git hash.

        Millisecond precision avoids collisions when multiple versions are
        saved within the same second (common in tests and rapid experiments).
        Format: ``YYYYMMDD_HHMMSS_mmm_<git>``
        """
        now = datetime.now(timezone.utc)
        ms = now.microsecond // 1000
        ts = now.strftime(f"%Y%m%d_%H%M%S_{ms:03d}")
        git = _git_short_hash()
        if git != "unknown":
            return f"{ts}_{git}"
        return f"{ts}_{uuid.uuid4().hex[:7]}"

    def _resolve_version_dir(self, name: str, version_id: str | None) -> Path:
        """Return the version directory, resolving None → latest/promoted."""
        if version_id is not None:
            d = self._root / f"{name}_{version_id}"
            if not d.exists():
                raise FileNotFoundError(
                    f"Version {version_id!r} for {name!r} not found at {d}"
                )
            return d

        # Try latest-marker file first
        marker = self._root / f"{name}{self._LATEST_MARKER}"
        if marker.exists():
            vid = marker.read_text(encoding="utf-8").strip()
            d = self._root / f"{name}_{vid}"
            if d.exists():
                return d

        # Fall back to most recent by directory name
        candidates = sorted(self._root.glob(f"{name}_*"), reverse=True)
        dirs = [c for c in candidates if c.is_dir() and (c / self._MANIFEST_FILE).exists()]
        if not dirs:
            raise FileNotFoundError(f"No saved versions found for model {name!r}.")
        return dirs[0]

    def _save_artifact(self, model: Any, version_dir: Path) -> Path:
        """Serialise model to the version directory; return the artifact path."""
        # PyTorch nn.Module
        try:
            import torch
            import torch.nn as nn
            if isinstance(model, nn.Module):
                path = version_dir / self._MODEL_FILE_PT
                torch.save(model.state_dict(), path)
                return path
        except ImportError:
            pass

        # numpy ndarray
        if isinstance(model, np.ndarray):
            path = version_dir / "model.npy"
            np.save(path, model)
            return path

        # Joblib-serialisable (sklearn, GPR, etc.)
        try:
            import joblib
            path = version_dir / self._MODEL_FILE_JOBLIB
            joblib.dump(model, path)
            return path
        except ImportError:
            pass

        # Last resort: JSON for simple dicts
        if isinstance(model, dict):
            path = version_dir / "model.json"
            path.write_text(json.dumps(_serialise(model), indent=2), encoding="utf-8")
            return path

        raise TypeError(
            f"Cannot serialise model of type {type(model).__name__}. "
            "Install torch or joblib to support this model type."
        )

    def _load_artifact(self, version_dir: Path) -> Any:
        """Deserialise from version directory."""
        pt_path = version_dir / self._MODEL_FILE_PT
        if pt_path.exists():
            import torch
            return torch.load(pt_path, map_location="cpu", weights_only=True)

        jl_path = version_dir / self._MODEL_FILE_JOBLIB
        if jl_path.exists():
            import joblib
            return joblib.load(jl_path)

        npy_path = version_dir / "model.npy"
        if npy_path.exists():
            return np.load(npy_path, allow_pickle=False)

        json_path = version_dir / "model.json"
        if json_path.exists():
            return json.loads(json_path.read_text(encoding="utf-8"))

        raise FileNotFoundError(f"No artifact found in {version_dir}")

    @staticmethod
    def _write_manifest(version_dir: Path, record: VersionRecord) -> None:
        manifest = version_dir / ModelVersionStore._MANIFEST_FILE
        manifest.write_text(
            json.dumps(asdict(record), indent=2), encoding="utf-8"
        )

    @staticmethod
    def _read_manifest(manifest_path: Path) -> VersionRecord:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return VersionRecord(**data)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _git_short_hash() -> str:
    """Return the current HEAD short hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _serialise(obj: Any) -> Any:
    """Recursively convert numpy scalars / arrays to JSON-serialisable types."""
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

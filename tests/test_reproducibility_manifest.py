from __future__ import annotations

import json
from pathlib import Path

from dashboard.reproducibility import ReproducibilityManifest, create_session_manifest


def test_create_session_manifest_writes_json(tmp_path: Path) -> None:
    manifest_path = create_session_manifest("session_001", tmp_path)

    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["session_id"] == "session_001"
    assert "timestamp" in data
    assert "version_control" in data


def test_reproducibility_manifest_contains_environment_and_config_hash(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "config.yaml"
    config_file.write_text("test_key: test_value\n", encoding="utf-8")

    manifest = ReproducibilityManifest(
        session_id="session_002",
        app_root=tmp_path,
        operator="tester",
    )

    payload = manifest.get_manifest()

    assert payload["operator"] == "tester"
    assert "environment" in payload
    assert "python_version" in payload["environment"]
    assert "configuration" in payload
    assert payload["configuration"]["hash"]

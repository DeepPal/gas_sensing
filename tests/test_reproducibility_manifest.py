from __future__ import annotations

import json
from pathlib import Path

from dashboard.reproducibility import (
    ReproducibilityManifest,
    create_session_manifest,
    verify_manifest_artifacts,
)


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


def test_manifest_contains_artifact_checksums(tmp_path: Path) -> None:
    session_dir = tmp_path / "session_003"
    session_dir.mkdir(parents=True)
    (session_dir / "results.csv").write_text("x,y\n1,2\n", encoding="utf-8")

    manifest_path = create_session_manifest("session_003", session_dir)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    checksums = payload.get("artifact_checksums", [])
    assert isinstance(checksums, list)
    assert any(item["path"] == "results.csv" for item in checksums)


def test_verify_manifest_artifacts_detects_tampering(tmp_path: Path) -> None:
    session_dir = tmp_path / "session_004"
    session_dir.mkdir(parents=True)
    data_path = session_dir / "results.csv"
    data_path.write_text("x,y\n1,2\n", encoding="utf-8")

    manifest_path = create_session_manifest("session_004", session_dir)
    ok_before, errors_before = verify_manifest_artifacts(manifest_path)
    assert ok_before is True
    assert errors_before == []

    data_path.write_text("x,y\n999,999\n", encoding="utf-8")
    ok_after, errors_after = verify_manifest_artifacts(manifest_path)
    assert ok_after is False
    assert any("Checksum mismatch" in err for err in errors_after)

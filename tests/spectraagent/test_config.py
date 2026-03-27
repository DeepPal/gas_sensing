from pathlib import Path

import pytest

from spectraagent.config import SpectraAgentConfig, load_config


def test_defaults_when_no_file(tmp_path):
    cfg = load_config(tmp_path / "spectraagent.toml")
    assert cfg.server.port == 8765
    assert cfg.hardware.integration_time_ms == 50.0
    assert cfg.agents.auto_explain is False
    assert cfg.claude.model == "claude-sonnet-4-6"


def test_file_created_when_missing(tmp_path):
    path = tmp_path / "spectraagent.toml"
    assert not path.exists()
    load_config(path)
    assert path.exists()


def test_override_from_file(tmp_path):
    path = tmp_path / "spectraagent.toml"
    path.write_text("[server]\nport = 9000\n")
    cfg = load_config(path)
    assert cfg.server.port == 9000
    assert cfg.server.host == "127.0.0.1"  # default preserved


def test_physics_defaults(tmp_path):
    cfg = load_config(tmp_path / "spectraagent.toml")
    assert cfg.physics.default_plugin == "lspr"
    assert cfg.physics.search_min_nm == 500.0
    assert cfg.physics.search_max_nm == 900.0

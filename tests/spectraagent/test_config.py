from pathlib import Path

import pytest

from spectraagent.config import SpectraAgentConfig, load_config
from typer.testing import CliRunner
from spectraagent.__main__ import cli


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


def test_plugins_list_shows_simulation():
    runner = CliRunner()
    result = runner.invoke(cli, ["plugins", "list"])
    assert result.exit_code == 0
    assert "simulation" in result.output.lower()


def test_plugins_list_shows_lspr():
    runner = CliRunner()
    result = runner.invoke(cli, ["plugins", "list"])
    assert result.exit_code == 0
    assert "lspr" in result.output.lower()


# ---------------------------------------------------------------------------
# Task 13: integration test for `spectraagent start --simulate --no-browser`
# ---------------------------------------------------------------------------

import subprocess
import sys
import time
import httpx


def test_start_simulate_no_browser_serves_health():
    """Integration test: start server in subprocess, hit /api/health, then kill it."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "spectraagent", "start", "--simulate", "--no-browser"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        time.sleep(4)
        resp = httpx.get("http://127.0.0.1:8765/api/health", timeout=5)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["simulate"] is True
    finally:
        proc.terminate()
        proc.wait(timeout=5)

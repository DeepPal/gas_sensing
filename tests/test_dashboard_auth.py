"""
tests/test_dashboard_auth.py - Unit tests for dashboard authentication
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch


class TestAuthModule:
    """Test the dashboard.auth module."""

    def test_password_hashing(self):
        """Verify PBKDF2 verifiers are stable for a fixed salt."""
        from dashboard.auth import _pbkdf2_hash_password

        verifier_1 = _pbkdf2_hash_password("test-password", salt=b"0123456789abcdef")
        verifier_2 = _pbkdf2_hash_password("test-password", salt=b"0123456789abcdef")

        assert verifier_1 == verifier_2
        assert verifier_1.startswith("pbkdf2_sha256$")

    def test_password_from_env(self):
        """Test reading password from environment variable."""
        from dashboard.auth import _get_stored_password

        with patch.dict("os.environ", {"DASHBOARD_PASSWORD": "env-secret"}):
            result = _get_stored_password()
            assert result == "env-secret"

    def test_password_hash_from_env(self):
        """Test reading PBKDF2 verifier from environment variable."""
        from dashboard.auth import _get_stored_password, _pbkdf2_hash_password

        password_hash = _pbkdf2_hash_password("env-secret", salt=b"0123456789abcdef")
        with patch.dict("os.environ", {"DASHBOARD_PASSWORD_HASH": password_hash}, clear=True):
            result = _get_stored_password()
            assert result == password_hash

    def test_password_from_file(self):
        """Test reading password hash from config file."""
        from dashboard.auth import PASSWORD_HASH_FILE, _get_stored_password, _pbkdf2_hash_password

        with tempfile.TemporaryDirectory() as tmpdir:
            pwd_file = Path(tmpdir) / PASSWORD_HASH_FILE
            password_hash = _pbkdf2_hash_password("file-password", salt=b"0123456789abcdef")
            pwd_file.write_text(password_hash, encoding="utf-8")

            # Mock Path.home() to return our temp directory
            with patch("dashboard.auth.Path.home", return_value=Path(tmpdir)):
                with patch.dict("os.environ", {}, clear=True):
                    result = _get_stored_password()
                    assert result == password_hash

    def test_password_fallback(self):
        """Test missing password configuration returns None."""
        from dashboard.auth import _get_stored_password

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock Path.home() to return a directory with no password file
            with patch("dashboard.auth.Path.home", return_value=Path(tmpdir)):
                with patch.dict("os.environ", {}, clear=True):
                    result = _get_stored_password()
                    assert result is None

    def test_verify_password_with_pbkdf2_hash(self):
        """Verify login validation against PBKDF2 verifier strings."""
        from dashboard.auth import _pbkdf2_hash_password, _verify_password

        password_hash = _pbkdf2_hash_password("correct-horse", salt=b"0123456789abcdef")

        assert _verify_password("correct-horse", password_hash) is True
        assert _verify_password("wrong-password", password_hash) is False


class TestHealthCheckModule:
    """Test the dashboard.health module."""

    def test_health_check_instantiation(self):
        """Verify HealthCheck can be instantiated."""
        from dashboard.health import HealthCheck

        hc = HealthCheck()
        assert hc.app_root is not None
        assert hc.timestamp is not None
        assert hc.hostname is not None

    def test_disk_space_check(self):
        """Test disk space check returns expected format."""
        from dashboard.health import HealthCheck

        hc = HealthCheck()
        result = hc.check_disk_space()

        assert isinstance(result, dict)
        assert "available_gb" in result
        assert "total_gb" in result
        assert "healthy" in result
        assert "status" in result
        assert isinstance(result["available_gb"], float)
        assert isinstance(result["healthy"], bool)

    def test_logs_check(self):
        """Test log file writability check."""
        from dashboard.health import HealthCheck

        hc = HealthCheck()
        result = hc.check_logs()

        assert isinstance(result, dict)
        assert "log_file" in result
        assert "healthy" in result
        assert "status" in result

    def test_hardware_check(self):
        """Test hardware availability check."""
        from dashboard.health import HealthCheck

        hc = HealthCheck()
        result = hc.check_hardware()

        assert isinstance(result, dict)
        assert "spectrometer" in result
        assert "live_server" in result

    def test_health_check_serialization(self):
        """Test health status can be serialized to JSON."""
        from dashboard.health import HealthCheck
        import json

        hc = HealthCheck()
        json_str = hc.to_json()

        # Should be valid JSON
        data = json.loads(json_str)
        assert isinstance(data, dict)
        assert "timestamp" in data
        assert "overall_healthy" in data

    def test_startup_check_returns_bool(self):
        """Test startup_check returns a boolean."""
        from dashboard.health import startup_check

        result = startup_check()
        assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

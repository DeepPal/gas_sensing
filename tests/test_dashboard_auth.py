"""
tests/test_dashboard_auth.py - Unit tests for dashboard authentication
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestAuthModule:
    """Test the dashboard.auth module."""

    def test_password_hashing(self):
        """Verify passwords are hashed consistently."""
        from dashboard.auth import _get_stored_password
        import hashlib

        with patch.dict("os.environ", {"DASHBOARD_PASSWORD": "test-password"}):
            hash1 = _get_stored_password()
            hash2 = _get_stored_password()

            assert hash1 == hash2, "Same password should produce same hash"
            assert hash1 == hashlib.sha256("test-password".encode()).hexdigest()

    def test_password_from_env(self):
        """Test reading password from environment variable."""
        from dashboard.auth import _get_stored_password
        import hashlib

        with patch.dict("os.environ", {"DASHBOARD_PASSWORD": "env-secret"}):
            result = _get_stored_password()
            expected = hashlib.sha256("env-secret".encode()).hexdigest()
            assert result == expected

    def test_password_from_file(self):
        """Test reading password from config file."""
        from dashboard.auth import _get_stored_password
        import hashlib

        with tempfile.TemporaryDirectory() as tmpdir:
            pwd_file = Path(tmpdir) / ".streamlit_au_mip_password"
            pwd_file.write_text("file-password")

            # Mock Path.home() to return our temp directory
            with patch("dashboard.auth.Path.home", return_value=Path(tmpdir)):
                with patch.dict("os.environ", {}, clear=True):
                    result = _get_stored_password()
                    expected = hashlib.sha256("file-password".encode()).hexdigest()
                    assert result == expected

    def test_password_fallback(self):
        """Test fallback to default password when none set."""
        from dashboard.auth import _get_stored_password
        import hashlib

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock Path.home() to return a directory with no password file
            with patch("dashboard.auth.Path.home", return_value=Path(tmpdir)):
                with patch.dict("os.environ", {}, clear=True):
                    result = _get_stored_password()
                    expected = hashlib.sha256("research-lab-default".encode()).hexdigest()
                    assert result == expected


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

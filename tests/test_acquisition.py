"""
tests.test_acquisition
=======================
Unit tests for src.acquisition:
  - Module-level re-exports (CCS200Spectrometer, RealtimeAcquisitionService)
  - Graceful None when hardware unavailable

Hardware tests are inherently integration tests that require the CCS200 DLL
and a physically connected spectrometer — those are skipped unconditionally in
CI.  The tests here only verify the import/export contract and the None-safe
degradation path.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Import contract
# ---------------------------------------------------------------------------


class TestAcquisitionImports:
    def test_module_importable(self):
        import src.acquisition  # noqa: F401

    def test_realtime_acquisition_service_exported(self):
        from src.acquisition import RealtimeAcquisitionService

        # Either a class or None (when gas_analysis not available)
        assert RealtimeAcquisitionService is None or callable(RealtimeAcquisitionService)

    def test_ccs200_spectrometer_exported(self):
        from src.acquisition import CCS200Spectrometer

        assert CCS200Spectrometer is None or callable(CCS200Spectrometer)

    def test_all_list_contains_expected_names(self):
        import src.acquisition as mod

        assert "RealtimeAcquisitionService" in mod.__all__
        assert "CCS200Spectrometer" in mod.__all__

    def test_realtime_service_is_class_or_none(self):
        from src.acquisition import RealtimeAcquisitionService

        if RealtimeAcquisitionService is not None:
            assert isinstance(RealtimeAcquisitionService, type)

    def test_ccs200_is_class_or_none(self):
        from src.acquisition import CCS200Spectrometer

        if CCS200Spectrometer is not None:
            assert isinstance(CCS200Spectrometer, type)


# ---------------------------------------------------------------------------
# RealtimeAcquisitionService interface contract (when available)
# ---------------------------------------------------------------------------


class TestRealtimeAcquisitionServiceContract:
    """Verify the public interface of RealtimeAcquisitionService.

    Skipped when the class is None (gas_analysis not installed).
    """

    @pytest.fixture
    def svc_cls(self):
        from src.acquisition import RealtimeAcquisitionService

        if RealtimeAcquisitionService is None:
            pytest.skip("RealtimeAcquisitionService not available")
        return RealtimeAcquisitionService

    def test_instantiates_without_connecting(self, svc_cls):
        """Constructor should not open hardware — only connect() does that."""
        svc = svc_cls(integration_time_ms=50)
        assert svc is not None

    def test_has_connect_method(self, svc_cls):
        svc = svc_cls(integration_time_ms=50)
        assert callable(getattr(svc, "connect", None))

    def test_has_start_method(self, svc_cls):
        svc = svc_cls(integration_time_ms=50)
        assert callable(getattr(svc, "start", None))

    def test_has_stop_method(self, svc_cls):
        svc = svc_cls(integration_time_ms=50)
        assert callable(getattr(svc, "stop", None))

    def test_has_register_callback_method(self, svc_cls):
        svc = svc_cls(integration_time_ms=50)
        assert callable(getattr(svc, "register_callback", None))


# ---------------------------------------------------------------------------
# CCS200Spectrometer interface contract (when available)
# ---------------------------------------------------------------------------


class TestCCS200SpectrometerContract:
    """Verify the public interface of CCS200Spectrometer.

    Skipped when the class is None or the DLL is not present (non-Windows,
    no VISA installation, etc.).
    """

    @pytest.fixture
    def spec_cls(self):
        from src.acquisition import CCS200Spectrometer

        if CCS200Spectrometer is None:
            pytest.skip("CCS200Spectrometer not available")
        return CCS200Spectrometer

    def test_class_has_num_pixels(self, spec_cls):
        assert hasattr(spec_cls, "NUM_PIXELS")
        assert spec_cls.NUM_PIXELS == 3648

    def test_class_has_default_integration(self, spec_cls):
        assert hasattr(spec_cls, "DEFAULT_INTEGRATION_S")
        assert spec_cls.DEFAULT_INTEGRATION_S > 0


# ---------------------------------------------------------------------------
# Task 12: _SessionWriter save_raw default changed to True
# ---------------------------------------------------------------------------


class TestSessionWriterSaveRawDefault:
    """Verify _SessionWriter defaults save_raw to True (Task 12)."""

    def test_save_raw_defaults_to_true(self, tmp_path):
        import inspect

        from src.inference.orchestrator import _SessionWriter

        sig = inspect.signature(_SessionWriter.__init__)
        assert sig.parameters["save_raw"].default is True

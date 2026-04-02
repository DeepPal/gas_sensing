"""
Tests for src.reporting.publication — journal figure export.

All figures are generated in a temporary directory; no display required
(matplotlib uses the Agg backend).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.reporting.publication import (
    JOURNAL_PRESETS,
    journal_style,
    list_presets,
    preset_info,
    save_calibration_figure,
    save_pls_diagnostics_figure,
    save_spectral_overlay_figure,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cal_data() -> dict:
    rng = np.random.default_rng(0)
    concs = np.linspace(0.5, 20.0, 10)
    responses = -0.05 * concs + rng.normal(0, 0.02, 10)
    fit = {
        "sensitivity": -0.05,
        "sensitivity_se": 0.002,
        "intercept": 0.0,
        "r_squared": 0.998,
        "rmse": 0.025,
        "noise_std": 0.015,
        "lod_ppm": 0.9,
        "loq_ppm": 3.0,
        "lob_ppm": 0.5,
        "lol_ppm": 18.0,
        "lod_ppm_ci_lower": 0.7,
        "lod_ppm_ci_upper": 1.1,
    }
    return {"concentrations": concs, "responses": responses, "fit_result": fit}


@pytest.fixture
def spectral_data() -> dict:
    rng = np.random.default_rng(1)
    wl = np.linspace(600.0, 900.0, 256)
    concs = np.array([1.0, 5.0, 10.0, 50.0, 100.0])
    peak = np.exp(-0.5 * ((wl - 750.0) / 30.0) ** 2)
    spectra = np.outer(concs, peak) + rng.normal(0, 0.005, (5, 256))
    return {"wavelengths": wl, "spectra": spectra, "concentrations": concs}


@pytest.fixture
def pls_result_stub():
    """Minimal duck-typed PLSFitResult for figure tests."""
    rng = np.random.default_rng(2)
    n_samples, n_feat, n_comp = 15, 100, 2

    class _Stub:
        x_scores = rng.normal(0, 1, (n_samples, n_comp))
        x_loadings = rng.normal(0, 0.1, (n_feat, n_comp))
        vip_scores = np.abs(rng.normal(1.0, 0.5, n_feat))
        rmsecv = 0.45
        q2 = 0.91
        n_components = n_comp

    return _Stub()


# ===========================================================================
# preset_info / list_presets
# ===========================================================================

class TestPresetInfo:
    def test_all_presets_have_required_keys(self) -> None:
        for key in JOURNAL_PRESETS:
            p = preset_info(key)
            for field in ("width_mm", "aspect", "dpi", "font_size",
                          "tick_size", "line_width", "marker_size"):
                assert field in p, f"Preset {key!r} missing field {field!r}"

    def test_unknown_preset_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            preset_info("unknown_journal_xyz")

    def test_list_presets_returns_all(self) -> None:
        entries = list_presets()
        assert len(entries) == len(JOURNAL_PRESETS)
        assert all("key" in e and "description" in e for e in entries)

    def test_acs_single_width(self) -> None:
        p = preset_info("acs_single")
        assert p["width_mm"] == 84

    def test_nature_dpi(self) -> None:
        p = preset_info("nature_s")
        assert p["dpi"] == 300

    def test_acs_dpi(self) -> None:
        assert preset_info("acs_single")["dpi"] == 600


# ===========================================================================
# journal_style context manager
# ===========================================================================

class TestJournalStyle:
    def test_yields_figsize_and_dpi(self) -> None:
        with journal_style("acs_single") as kw:
            assert "figsize" in kw
            assert "dpi" in kw

    def test_figsize_is_tuple_of_floats(self) -> None:
        with journal_style("acs_single") as kw:
            w, h = kw["figsize"]
            assert w > 0
            assert h > 0

    def test_width_matches_preset(self) -> None:
        import matplotlib
        _MM = 25.4
        with journal_style("acs_single") as kw:
            w_in, _ = kw["figsize"]
        # ACS single = 84 mm
        assert abs(w_in - 84.0 / _MM) < 0.01

    def test_rcparams_restored_after_context(self) -> None:
        import matplotlib.pyplot as plt
        before = plt.rcParams["font.size"]
        with journal_style("acs_single"):
            pass
        after = plt.rcParams["font.size"]
        assert before == after

    def test_dpi_matches_preset(self) -> None:
        with journal_style("nature_s") as kw:
            assert kw["dpi"] == 300


# ===========================================================================
# save_calibration_figure
# ===========================================================================

class TestSaveCalibrationFigure:
    def test_creates_file(self, tmp_path: Path, cal_data: dict) -> None:
        out = tmp_path / "fig_cal.png"
        result = save_calibration_figure(
            concentrations=cal_data["concentrations"],
            responses=cal_data["responses"],
            fit_result=cal_data["fit_result"],
            out_path=out,
        )
        assert result.exists()
        assert result.stat().st_size > 0

    def test_returns_correct_path(self, tmp_path: Path, cal_data: dict) -> None:
        out = tmp_path / "fig_cal.png"
        result = save_calibration_figure(
            concentrations=cal_data["concentrations"],
            responses=cal_data["responses"],
            fit_result=cal_data["fit_result"],
            out_path=out,
        )
        assert result == out

    def test_creates_parent_directory(self, tmp_path: Path, cal_data: dict) -> None:
        out = tmp_path / "sub" / "nested" / "fig.png"
        save_calibration_figure(
            concentrations=cal_data["concentrations"],
            responses=cal_data["responses"],
            fit_result=cal_data["fit_result"],
            out_path=out,
        )
        assert out.exists()

    def test_all_journal_presets(self, tmp_path: Path, cal_data: dict) -> None:
        for preset in JOURNAL_PRESETS:
            out = tmp_path / f"cal_{preset}.png"
            save_calibration_figure(
                concentrations=cal_data["concentrations"],
                responses=cal_data["responses"],
                fit_result=cal_data["fit_result"],
                out_path=out,
                preset=preset,
            )
            assert out.exists(), f"Figure not created for preset {preset!r}"

    def test_pdf_format(self, tmp_path: Path, cal_data: dict) -> None:
        out = tmp_path / "fig_cal.pdf"
        save_calibration_figure(
            concentrations=cal_data["concentrations"],
            responses=cal_data["responses"],
            fit_result=cal_data["fit_result"],
            out_path=out,
        )
        assert out.exists()

    def test_no_lod_annotations(self, tmp_path: Path, cal_data: dict) -> None:
        """Should not raise when LOD annotation is disabled."""
        out = tmp_path / "fig_no_lod.png"
        save_calibration_figure(
            concentrations=cal_data["concentrations"],
            responses=cal_data["responses"],
            fit_result=cal_data["fit_result"],
            out_path=out,
            annotate_lod=False,
            annotate_loq=False,
            annotate_lol=False,
        )
        assert out.exists()

    def test_nan_lod_handled_gracefully(self, tmp_path: Path) -> None:
        concs = np.linspace(1, 10, 5)
        resps = -0.05 * concs
        fit = {"sensitivity": -0.05, "intercept": 0.0,
               "lod_ppm": float("nan"), "loq_ppm": float("nan")}
        out = tmp_path / "fig_nan.png"
        save_calibration_figure(concs, resps, fit, out)
        assert out.exists()


# ===========================================================================
# save_spectral_overlay_figure
# ===========================================================================

class TestSaveSpectralOverlayFigure:
    def test_creates_file(self, tmp_path: Path, spectral_data: dict) -> None:
        out = tmp_path / "overlay.png"
        save_spectral_overlay_figure(
            wavelengths=spectral_data["wavelengths"],
            spectra=spectral_data["spectra"],
            concentrations=spectral_data["concentrations"],
            out_path=out,
        )
        assert out.exists()

    def test_with_reference_spectrum(self, tmp_path: Path, spectral_data: dict) -> None:
        rng = np.random.default_rng(5)
        ref = rng.uniform(0.0, 0.1, len(spectral_data["wavelengths"]))
        out = tmp_path / "overlay_ref.png"
        save_spectral_overlay_figure(
            wavelengths=spectral_data["wavelengths"],
            spectra=spectral_data["spectra"],
            concentrations=spectral_data["concentrations"],
            out_path=out,
            reference_spectrum=ref,
        )
        assert out.exists()

    def test_wavelength_range_zoom(self, tmp_path: Path, spectral_data: dict) -> None:
        out = tmp_path / "overlay_zoom.png"
        save_spectral_overlay_figure(
            wavelengths=spectral_data["wavelengths"],
            spectra=spectral_data["spectra"],
            concentrations=spectral_data["concentrations"],
            out_path=out,
            wl_range=(680.0, 820.0),
        )
        assert out.exists()

    def test_no_colorbar(self, tmp_path: Path, spectral_data: dict) -> None:
        out = tmp_path / "overlay_nocb.png"
        save_spectral_overlay_figure(
            wavelengths=spectral_data["wavelengths"],
            spectra=spectral_data["spectra"],
            concentrations=spectral_data["concentrations"],
            out_path=out,
            show_colorbar=False,
        )
        assert out.exists()

    def test_single_spectrum_input(self, tmp_path: Path) -> None:
        """Single 1D spectrum should not crash."""
        wl = np.linspace(600, 900, 100)
        sp = np.ones(100) * 500.0
        out = tmp_path / "overlay_single.png"
        save_spectral_overlay_figure(
            wavelengths=wl, spectra=sp, concentrations=np.array([5.0]),
            out_path=out,
        )
        assert out.exists()


# ===========================================================================
# save_pls_diagnostics_figure
# ===========================================================================

class TestSavePLSDiagnosticsFigure:
    def test_creates_file(self, tmp_path: Path, pls_result_stub) -> None:
        rng = np.random.default_rng(3)
        concs = np.linspace(1.0, 20.0, 15)
        preds = concs + rng.normal(0, 0.5, 15)
        out = tmp_path / "pls_diag.png"
        save_pls_diagnostics_figure(
            pls_result=pls_result_stub,
            concentrations=concs,
            predicted=preds,
            out_path=out,
        )
        assert out.exists()

    def test_with_wavelengths(self, tmp_path: Path, pls_result_stub) -> None:
        rng = np.random.default_rng(4)
        concs = np.linspace(1.0, 20.0, 15)
        preds = concs + rng.normal(0, 0.5, 15)
        wl = np.linspace(600.0, 900.0, 100)
        out = tmp_path / "pls_wl.png"
        save_pls_diagnostics_figure(
            pls_result=pls_result_stub,
            concentrations=concs,
            predicted=preds,
            out_path=out,
            wavelengths=wl,
        )
        assert out.exists()

    def test_none_loadings_handled(self, tmp_path: Path) -> None:
        """Figure should not raise when optional arrays are None."""
        class _Minimal:
            x_scores = None
            x_loadings = None
            vip_scores = None
            rmsecv = float("nan")
            q2 = float("nan")
            n_components = 1

        concs = np.array([1.0, 5.0, 10.0])
        preds = np.array([1.1, 4.9, 10.2])
        out = tmp_path / "pls_minimal.png"
        save_pls_diagnostics_figure(
            pls_result=_Minimal(),
            concentrations=concs,
            predicted=preds,
            out_path=out,
        )
        assert out.exists()

    def test_acs_double_preset(self, tmp_path: Path, pls_result_stub) -> None:
        concs = np.linspace(1.0, 20.0, 15)
        preds = concs + 0.3
        out = tmp_path / "pls_acs_double.png"
        save_pls_diagnostics_figure(
            pls_result=pls_result_stub,
            concentrations=concs,
            predicted=preds,
            out_path=out,
            preset="acs_double",
        )
        assert out.exists()

    def test_top_n_vip_truncation(self, tmp_path: Path, pls_result_stub) -> None:
        """top_n_vip=5 should not crash even when more VIPs are available."""
        concs = np.linspace(1.0, 20.0, 15)
        preds = concs + 0.3
        out = tmp_path / "pls_vip5.png"
        save_pls_diagnostics_figure(
            pls_result=pls_result_stub,
            concentrations=concs,
            predicted=preds,
            out_path=out,
            top_n_vip=5,
        )
        assert out.exists()

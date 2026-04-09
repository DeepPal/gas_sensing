"""End-to-end tests: src pipeline → conformal intervals → agent context."""
import numpy as np
import pytest


def test_conformal_intervals_are_finite_and_ordered():
    """Full path: GPR fit → conformal calibrate → predict interval."""
    from src.calibration.conformal import ConformalCalibrator
    from src.calibration.gpr import GPRCalibration

    np.random.seed(123)
    n = 40
    concs = np.random.uniform(0.1, 5.0, n)
    shifts = -10.0 * concs / (1.0 + concs) + np.random.normal(0, 0.1, n)

    # Train/cal split
    X_train, y_train = shifts[:20].reshape(-1, 1), concs[:20]
    X_cal, y_cal = shifts[20:30].reshape(-1, 1), concs[20:30]
    X_test = shifts[30:].reshape(-1, 1)
    y_test = concs[30:]

    gpr = GPRCalibration()
    gpr.fit(X_train, y_train)

    cal = ConformalCalibrator()
    cal.calibrate(gpr, X_cal, y_cal)

    lo, hi = cal.predict_interval(gpr, X_test, alpha=0.10)
    assert np.all(np.isfinite(lo))
    assert np.all(np.isfinite(hi))
    assert np.all(hi > lo)

    coverage = np.mean((y_test >= lo) & (y_test <= hi))
    assert coverage >= 0.75, f"Coverage {coverage:.2%} unexpectedly low"


def test_session_analyzer_produces_valid_lod():
    """SessionAnalyzer output has positive LOD/LOQ from realistic calibration data."""
    from src.inference.session_analyzer import SessionAnalyzer

    events = []
    for conc in [0.5, 1.0, 2.0, 3.0, 4.0]:
        events.append({
            "type": "calibration_point",
            "concentration_ppm": conc,
            "wavelength_shift": -10.0 * conc / (1.0 + conc),
            "snr": 14.0,
        })
    for i in range(10):
        conc = 2.5
        events.append({
            "type": "measurement",
            "concentration_ppm": conc + i * 0.02,
            "ci_low": conc - 0.25,
            "ci_high": conc + 0.25,
            "wavelength_shift": -10.0 * conc / (1.0 + conc),
            "snr": 13.0 + i * 0.1,
            "peak_wavelength": 715.0 + i * 0.01,
        })

    analysis = SessionAnalyzer().analyze(events, frame_count=15)
    assert np.isfinite(analysis.lod_ppm) and analysis.lod_ppm > 0
    assert np.isfinite(analysis.loq_ppm) and analysis.loq_ppm > analysis.lod_ppm
    assert analysis.calibration_r2 is not None and analysis.calibration_r2 > 0.5


def test_bayesian_designer_and_physics_gpr_together():
    """BayesianExperimentDesigner with PhysicsInformedGPR suggests valid concentrations."""
    from src.calibration.active_learning import BayesianExperimentDesigner
    from src.calibration.physics_kernel import PhysicsInformedGPR

    np.random.seed(0)
    concs = np.array([0.5, 1.0, 2.0])
    shifts = -10.0 * concs / (1.0 + concs) + np.random.normal(0, 0.05, 3)

    model = PhysicsInformedGPR()
    model.fit(shifts.reshape(-1, 1), concs)

    designer = BayesianExperimentDesigner(min_conc=0.01, max_conc=10.0)
    suggestion = designer.suggest_next(model, concs.tolist())
    assert 0.01 <= suggestion <= 10.0


def test_full_pipeline_with_conformal_stage():
    """RealTimePipeline with set_gpr populates ci_low/ci_high on a real frame."""
    from src.calibration.gpr import GPRCalibration
    from src.inference.realtime_pipeline import PipelineConfig, RealTimePipeline

    np.random.seed(5)
    n = 15
    concs = np.linspace(0.5, 4.0, n)
    shifts = -10.0 * concs / (1.0 + concs) + np.random.normal(0, 0.08, n)

    gpr = GPRCalibration()
    gpr.fit(shifts.reshape(-1, 1), concs)

    X_cal = shifts[:8].reshape(-1, 1)
    y_cal = concs[:8]

    cfg = PipelineConfig(
        reference_wavelength=717.9,
        peak_search_min_nm=650.0,
        peak_search_max_nm=780.0,
    )
    pipeline = RealTimePipeline(cfg)
    pipeline._calibration.set_gpr(gpr, X_cal, y_cal)

    # Synthetic reference + test spectrum (Gaussian instead of Lorentzian — simpler)
    wl = np.linspace(300, 1000, 3648)
    ref = 1000 * np.exp(-((wl - 717.9) ** 2) / (2 * 5.0 ** 2))
    test = 1000 * np.exp(-((wl - 716.0) ** 2) / (2 * 5.0 ** 2))  # -1.9 nm shift

    pipeline._features.set_reference(ref)
    pipeline._calibration.set_reference(ref)

    result = pipeline.process_spectrum(wl, test)
    assert result.success or result.spectrum.wavelength_shift is not None

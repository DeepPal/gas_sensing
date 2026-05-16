from src.reporting.scientific_summary import build_deterministic_scientific_report


def test_build_deterministic_scientific_report_includes_readiness_actions():
    report = build_deterministic_scientific_report(
        {
            "session_id": "sess-001",
            "gas_label": "Ethanol",
            "hardware": "CCS200",
            "analysis": {
                "frame_count": 120,
                "calibration_n_points": 3,
                "calibration_r2": 0.982,
                "lod_ppm": 0.12,
                "lod_ci_lower": 0.09,
                "lod_ci_upper": 0.16,
                "loq_ppm": 0.40,
                "lod_used_blanks": False,
                "mean_snr": 2.4,
                "summary_text": "Signal increased with concentration but calibration remains underpowered.",
            },
        }
    )

    assert "Deterministic Scientific Summary" in report
    assert "Ethanol" in report
    assert "Calibration density: ACTION NEEDED" in report
    assert "Blank-backed LOD: ACTION NEEDED" in report
    assert "Run dedicated blank replicates" in report


def test_build_deterministic_scientific_report_includes_audit_and_noise_sections():
    report = build_deterministic_scientific_report(
        {
            "session_id": "sess-002",
            "gas_label": "Methanol",
            "temperature_c": 25.0,
            "humidity_pct": 40.0,
            "analysis": {
                "frame_count": 200,
                "calibration_n_points": 6,
                "calibration_r2": 0.998,
                "lod_ppm": 0.03,
                "loq_ppm": 0.10,
                "lod_used_blanks": True,
                "mean_snr": 5.2,
                "tau_63_s": 12.0,
                "kinetics_fit_r2": 0.95,
                "lol_ppm": 50.0,
                "audit": {
                    "method": "IUPAC_2012_Eurachem",
                    "lod_formula": "3·σ_blank_nm / m",
                    "sigma_source": "blank_events",
                    "n_bootstrap": 1000,
                    "framework_version": "1.0.0",
                },
                "allan_deviation": {
                    "tau_opt_s": 3.0,
                    "sigma_min": 0.002,
                    "noise_type": "white",
                    "drift_onset_tau_s": 30.0,
                },
            },
        }
    )

    assert "## Audit trail" in report
    assert "IUPAC_2012_Eurachem" in report
    assert "## Noise-floor characterization" in report
    assert "tau_opt" in report

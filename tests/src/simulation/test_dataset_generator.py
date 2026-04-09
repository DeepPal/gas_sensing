"""Tests for src.simulation.dataset_generator — batch dataset generation."""
import numpy as np
import pytest

from src.simulation.dataset_generator import (
    DatasetConfig,
    DatasetGenerator,
    make_ethanol_acetone_dataset,
)
from src.simulation.gas_response import make_analyte, make_single_peak_sensor


@pytest.fixture
def simple_sensor():
    sensor = make_single_peak_sensor(700.0)
    sensor.analytes = [make_analyte("Ethanol", 1, -0.5, tau_s=30.0)]
    return sensor


@pytest.fixture
def two_analyte_sensor():
    sensor = make_single_peak_sensor(700.0)
    sensor.analytes = [
        make_analyte("Ethanol", 1, -0.5, tau_s=30.0),
        make_analyte("Acetone", 1, -0.35, tau_s=18.0),
    ]
    return sensor


class TestDatasetConfig:
    def test_default_concentration_levels(self, simple_sensor):
        cfg = DatasetConfig(sensor_config=simple_sensor, analyte_names=["Ethanol"])
        assert len(cfg.concentration_levels) > 0

    def test_n_sessions_respected(self, simple_sensor):
        cfg = DatasetConfig(
            sensor_config=simple_sensor,
            analyte_names=["Ethanol"],
            n_sessions=3,
            concentration_levels=[0.5, 1.0],
        )
        gen = DatasetGenerator(cfg)
        df = gen.generate_calibration_dataset()
        # 3 sessions × 2 concentrations
        assert len(df) == 6


class TestDatasetGenerator:
    def test_calibration_dataset_shape(self, simple_sensor):
        cfg = DatasetConfig(
            sensor_config=simple_sensor,
            analyte_names=["Ethanol"],
            n_sessions=2,
            concentration_levels=[0.5, 1.0, 2.0],
        )
        df = DatasetGenerator(cfg).generate_calibration_dataset()
        assert len(df) == 6  # 2 sessions × 3 concentrations
        assert "concentration_ppm" in df.columns
        assert "peak_shift_0" in df.columns

    def test_calibration_dataset_has_correct_analyte(self, simple_sensor):
        cfg = DatasetConfig(
            sensor_config=simple_sensor,
            analyte_names=["Ethanol"],
            n_sessions=2,
            concentration_levels=[1.0],
        )
        df = DatasetGenerator(cfg).generate_calibration_dataset()
        assert (df["analyte"] == "Ethanol").all()

    def test_peak_shift_has_correct_sign(self, simple_sensor):
        """Ethanol has negative sensitivity → should produce negative mean shift."""
        cfg = DatasetConfig(
            sensor_config=simple_sensor,
            analyte_names=["Ethanol"],
            n_sessions=1,
            concentration_levels=[5.0],  # high conc → clear signal
            domain_randomize=False,
        )
        df = DatasetGenerator(cfg).generate_calibration_dataset()
        assert df["peak_shift_0"].mean() < 0

    def test_higher_concentration_larger_shift(self, simple_sensor):
        cfg = DatasetConfig(
            sensor_config=simple_sensor,
            analyte_names=["Ethanol"],
            n_sessions=3,
            concentration_levels=[0.5, 5.0],
            domain_randomize=False,
        )
        df = DatasetGenerator(cfg).generate_calibration_dataset()
        low = df[df["concentration_ppm"] == 0.5]["peak_shift_0"].mean()
        high = df[df["concentration_ppm"] == 5.0]["peak_shift_0"].mean()
        assert abs(high) > abs(low)

    def test_reference_spectrum_in_output(self, simple_sensor):
        cfg = DatasetConfig(
            sensor_config=simple_sensor,
            analyte_names=["Ethanol"],
            n_sessions=1,
            concentration_levels=[1.0],
        )
        df = DatasetGenerator(cfg).generate_calibration_dataset()
        assert "reference_spectrum" in df.columns
        ref = df["reference_spectrum"].iloc[0]
        assert len(ref) == 3648

    def test_mixture_dataset_two_analyte(self, two_analyte_sensor):
        cfg = DatasetConfig(
            sensor_config=two_analyte_sensor,
            analyte_names=["Ethanol", "Acetone"],
            n_sessions=1,
            concentration_levels=[0.5, 1.0],
        )
        df = DatasetGenerator(cfg).generate_mixture_dataset()
        assert "conc_Ethanol_ppm" in df.columns
        assert "conc_Acetone_ppm" in df.columns
        assert len(df) > 0

    def test_mixture_raises_for_single_analyte(self, simple_sensor):
        cfg = DatasetConfig(
            sensor_config=simple_sensor,
            analyte_names=["Ethanol"],
            n_sessions=1,
        )
        with pytest.raises(ValueError, match="at least 2 analytes"):
            DatasetGenerator(cfg).generate_mixture_dataset()

    def test_kinetic_dataset_returns_sessions(self, simple_sensor):
        cfg = DatasetConfig(
            sensor_config=simple_sensor,
            analyte_names=["Ethanol"],
            n_sessions=2,
        )
        sessions = DatasetGenerator(cfg).generate_kinetic_dataset("Ethanol", 1.0, n_sessions=2)
        assert len(sessions) == 2
        assert sessions[0].n_frames > 0

    def test_to_numpy_correct_shapes(self, simple_sensor):
        cfg = DatasetConfig(
            sensor_config=simple_sensor,
            analyte_names=["Ethanol"],
            n_sessions=2,
            concentration_levels=[0.5, 1.0, 2.0],
        )
        df = DatasetGenerator(cfg).generate_calibration_dataset()
        X, y = DatasetGenerator.to_numpy(df, "Ethanol")
        assert X.shape == (6, 1)  # n_samples × n_peaks
        assert y.shape == (6,)

    def test_to_spectrum_numpy_shapes(self, simple_sensor):
        cfg = DatasetConfig(
            sensor_config=simple_sensor,
            analyte_names=["Ethanol"],
            n_sessions=1,
            concentration_levels=[1.0],
        )
        df = DatasetGenerator(cfg).generate_calibration_dataset()
        spectra, concs = DatasetGenerator.to_spectrum_numpy(df)
        assert spectra.shape[0] == len(df)
        assert spectra.shape[1] == 3648
        assert concs.shape == (len(df),)


class TestMakeEthanolAcetone:
    def test_factory_returns_config_and_generator(self):
        cfg, gen = make_ethanol_acetone_dataset(n_sessions=2, random_seed=0)
        assert "Ethanol" in cfg.analyte_names
        assert "Acetone" in cfg.analyte_names
        assert isinstance(gen, DatasetGenerator)

    def test_calibration_dataset_runs(self):
        cfg, gen = make_ethanol_acetone_dataset(n_sessions=2, random_seed=42)
        df = gen.generate_calibration_dataset("Ethanol")
        assert len(df) > 0

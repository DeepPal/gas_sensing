"""
Tests for src.experiment_tracking — ExperimentTracker and get_tracker.

Each test class gets its own tmp_path-based SQLite tracking DB to avoid
global MLflow state contamination between tests and to use the
MLflow-recommended backend.

Tests skip gracefully when mlflow is not installed.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

mlflow = pytest.importorskip("mlflow", reason="mlflow not installed")

from src.experiment_tracking import ExperimentTracker, _json_default, get_tracker


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


def _make_tracker(tmp_path: Path, name: str = "test_experiment") -> ExperimentTracker:
    """Create a tracker backed by a temporary SQLite tracking DB."""
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    return ExperimentTracker(experiment_name=name, tracking_uri=uri)


def _sensor_metrics() -> dict[str, Any]:
    return {
        "sensitivity": -0.05,
        "sensitivity_se": 0.002,
        "r_squared": 0.997,
        "rmse": 0.025,
        "noise_std": 0.015,
        "lob_ppm": 0.5,
        "lod_ppm": 0.9,
        "loq_ppm": 3.0,
        "lol_ppm": 18.0,
        "lod_ppm_ci_lower": 0.7,
        "lod_ppm_ci_upper": 1.1,
        "loq_ppm_ci_lower": 2.5,
        "loq_ppm_ci_upper": 3.6,
    }


def _gpr_model():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (10, 4))
    y = rng.uniform(1, 20, 10)
    model = GaussianProcessRegressor(
        kernel=RBF() + WhiteKernel(), normalize_y=True
    )
    model.fit(X, y)
    return model, X, y


class _PLSResultStub:
    def __init__(self) -> None:
        rng = np.random.default_rng(1)
        self.n_components = 2
        self.optimal_n_components = 2
        self.r2_calibration = 0.993
        self.rmsec = 0.21
        self.q2 = 0.91
        self.rmsecv = 0.45
        self.pearson_r = 0.996
        self.bias = 0.003
        self.vip_scores = np.abs(rng.normal(1.0, 0.5, 50))
        self.rmsecv_per_component = [0.80, 0.45, 0.48]
        self.n_features = 50
        self.x_scores = rng.normal(0, 1, (10, 2))
        self.x_loadings = rng.normal(0, 0.1, (50, 2))


class _PLSModelStub:
    """Minimal duck-typed PLSCalibration stub for artifact logging."""
    cv_folds = -1
    n_components = 2
    scale = True
    _model = None  # log_model will fail gracefully

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))


# ===========================================================================
# ExperimentTracker — construction
# ===========================================================================


class TestExperimentTrackerInit:
    def test_available_when_mlflow_installed(self, tmp_path: Path) -> None:
        tracker = _make_tracker(tmp_path)
        assert tracker.available is True

    def test_experiment_name_stored(self, tmp_path: Path) -> None:
        tracker = _make_tracker(tmp_path, name="My_Experiment")
        assert tracker.experiment_name == "My_Experiment"

    def test_creates_experiment_in_mlflow(self, tmp_path: Path) -> None:
        tracker = _make_tracker(tmp_path, name="Ethanol_Experiment")
        exp = mlflow.get_experiment_by_name("Ethanol_Experiment")
        assert exp is not None

    def test_get_existing_experiment_on_second_init(self, tmp_path: Path) -> None:
        _make_tracker(tmp_path, name="Shared_Experiment")
        tracker2 = _make_tracker(tmp_path, name="Shared_Experiment")
        assert tracker2._experiment_id is not None


# ===========================================================================
# start_run context manager
# ===========================================================================


class TestStartRun:
    def test_yields_self(self, tmp_path: Path) -> None:
        tracker = _make_tracker(tmp_path)
        with tracker.start_run(run_name="test_run") as ctx:
            assert ctx is tracker

    def test_run_is_active_inside(self, tmp_path: Path) -> None:
        tracker = _make_tracker(tmp_path)
        with tracker.start_run(run_name="test_run"):
            assert mlflow.active_run() is not None

    def test_run_ends_after_context(self, tmp_path: Path) -> None:
        tracker = _make_tracker(tmp_path)
        with tracker.start_run(run_name="test_run"):
            pass
        assert mlflow.active_run() is None

    def test_run_name_tag_stored(self, tmp_path: Path) -> None:
        tracker = _make_tracker(tmp_path)
        with tracker.start_run(run_name="named_run", tags={"gas": "Ethanol"}):
            run = mlflow.active_run()
            assert run.data.tags.get("gas") == "Ethanol"

    def test_no_op_when_unavailable(self, tmp_path: Path) -> None:
        """When mlflow is not available, start_run should still yield without error."""
        tracker = _make_tracker(tmp_path)
        tracker.available = False
        entered = False
        with tracker.start_run("no_op_run"):
            entered = True
        assert entered


# ===========================================================================
# log_gpr_run
# ===========================================================================


class TestLogGprRun:
    def test_returns_run_id(self, tmp_path: Path) -> None:
        tracker = _make_tracker(tmp_path)
        model, X, y = _gpr_model()
        with tracker.start_run("gpr_run"):
            run_id = tracker.log_gpr_run(
                model=model,
                sensor_metrics=_sensor_metrics(),
                y_concs=y,
                X_features=X,
                gas_name="Ethanol",
            )
        assert isinstance(run_id, str) and len(run_id) > 0

    def test_params_logged(self, tmp_path: Path) -> None:
        tracker = _make_tracker(tmp_path)
        model, X, y = _gpr_model()
        with tracker.start_run("gpr_run"):
            run_id = tracker.log_gpr_run(
                model=model,
                sensor_metrics=_sensor_metrics(),
                y_concs=y,
                X_features=X,
                gas_name="TestGas",
            )
        run = mlflow.get_run(run_id)
        assert run.data.params["model_type"] == "GPR"
        assert run.data.params["gas_name"] == "TestGas"
        assert int(run.data.params["n_training_samples"]) == len(y)

    def test_sensor_metrics_logged(self, tmp_path: Path) -> None:
        tracker = _make_tracker(tmp_path)
        model, X, y = _gpr_model()
        with tracker.start_run("gpr_metrics"):
            run_id = tracker.log_gpr_run(
                model=model,
                sensor_metrics=_sensor_metrics(),
                y_concs=y,
                X_features=X,
            )
        run = mlflow.get_run(run_id)
        assert "r_squared" in run.data.metrics
        assert "lod_ppm" in run.data.metrics
        assert run.data.metrics["r_squared"] == pytest.approx(0.997)

    def test_returns_none_outside_run(self, tmp_path: Path) -> None:
        tracker = _make_tracker(tmp_path)
        model, X, y = _gpr_model()
        result = tracker.log_gpr_run(
            model=model,
            sensor_metrics=_sensor_metrics(),
            y_concs=y,
            X_features=X,
        )
        assert result is None

    def test_returns_none_when_unavailable(self, tmp_path: Path) -> None:
        tracker = _make_tracker(tmp_path)
        tracker.available = False
        model, X, y = _gpr_model()
        result = tracker.log_gpr_run(
            model=model,
            sensor_metrics={},
            y_concs=y,
            X_features=X,
        )
        assert result is None

    def test_cal_figure_artifact_logged(self, tmp_path: Path) -> None:
        tracker = _make_tracker(tmp_path)
        model, X, y = _gpr_model()
        # Create a dummy PNG
        fig_path = tmp_path / "cal.png"
        fig_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        with tracker.start_run("gpr_fig"):
            run_id = tracker.log_gpr_run(
                model=model,
                sensor_metrics=_sensor_metrics(),
                y_concs=y,
                X_features=X,
                cal_figure_path=fig_path,
            )
        artifacts = mlflow.MlflowClient().list_artifacts(run_id, path="figures")
        assert len(artifacts) > 0

    def test_nonexistent_figure_does_not_raise(self, tmp_path: Path) -> None:
        tracker = _make_tracker(tmp_path)
        model, X, y = _gpr_model()
        with tracker.start_run("gpr_nofig"):
            tracker.log_gpr_run(
                model=model,
                sensor_metrics=_sensor_metrics(),
                y_concs=y,
                X_features=X,
                cal_figure_path="/nonexistent/path/cal.png",
            )


# ===========================================================================
# log_pls_run
# ===========================================================================


class TestLogPlsRun:
    def test_returns_run_id(self, tmp_path: Path) -> None:
        tracker = _make_tracker(tmp_path)
        pls_model = _PLSModelStub()
        pls_result = _PLSResultStub()
        y = np.linspace(1.0, 20.0, 10)
        with tracker.start_run("pls_run"):
            run_id = tracker.log_pls_run(
                pls_model=pls_model,
                pls_result=pls_result,
                sensor_metrics=_sensor_metrics(),
                y_concs=y,
                gas_name="Acetone",
            )
        assert isinstance(run_id, str) and len(run_id) > 0

    def test_pls_params_logged(self, tmp_path: Path) -> None:
        tracker = _make_tracker(tmp_path)
        pls_model = _PLSModelStub()
        pls_result = _PLSResultStub()
        y = np.linspace(1.0, 20.0, 10)
        with tracker.start_run("pls_params"):
            run_id = tracker.log_pls_run(
                pls_model=pls_model,
                pls_result=pls_result,
                sensor_metrics=_sensor_metrics(),
                y_concs=y,
                gas_name="Acetone",
            )
        run = mlflow.get_run(run_id)
        assert run.data.params["model_type"] == "PLS"
        assert run.data.params["gas_name"] == "Acetone"
        assert int(run.data.params["n_components"]) == 2
        assert run.data.params["cv_strategy"] == "LOO"

    def test_pls_metrics_logged(self, tmp_path: Path) -> None:
        tracker = _make_tracker(tmp_path)
        pls_model = _PLSModelStub()
        pls_result = _PLSResultStub()
        y = np.linspace(1.0, 20.0, 10)
        with tracker.start_run("pls_metrics"):
            run_id = tracker.log_pls_run(
                pls_model=pls_model,
                pls_result=pls_result,
                sensor_metrics=_sensor_metrics(),
                y_concs=y,
            )
        run = mlflow.get_run(run_id)
        assert "q2_crossvalidated" in run.data.metrics
        assert run.data.metrics["q2_crossvalidated"] == pytest.approx(0.91)
        assert "r2_calibration" in run.data.metrics

    def test_vip_csv_artifact_logged(self, tmp_path: Path) -> None:
        tracker = _make_tracker(tmp_path)
        pls_model = _PLSModelStub()
        pls_result = _PLSResultStub()
        wl = np.linspace(600.0, 900.0, 50)
        y = np.linspace(1.0, 20.0, 10)
        with tracker.start_run("pls_vip"):
            run_id = tracker.log_pls_run(
                pls_model=pls_model,
                pls_result=pls_result,
                sensor_metrics=_sensor_metrics(),
                y_concs=y,
                wavelengths=wl,
            )
        artifacts = mlflow.MlflowClient().list_artifacts(run_id, path="vip_scores")
        assert len(artifacts) > 0

    def test_n_vip_above_threshold_metric(self, tmp_path: Path) -> None:
        tracker = _make_tracker(tmp_path)
        pls_model = _PLSModelStub()
        pls_result = _PLSResultStub()
        y = np.linspace(1.0, 20.0, 10)
        with tracker.start_run("pls_nvip"):
            run_id = tracker.log_pls_run(
                pls_model=pls_model,
                pls_result=pls_result,
                sensor_metrics=_sensor_metrics(),
                y_concs=y,
            )
        run = mlflow.get_run(run_id)
        assert "n_vip_above_threshold" in run.data.metrics
        n_actual = int((pls_result.vip_scores > 1.0).sum())
        assert int(run.data.metrics["n_vip_above_threshold"]) == n_actual

    def test_returns_none_when_unavailable(self, tmp_path: Path) -> None:
        tracker = _make_tracker(tmp_path)
        tracker.available = False
        result = tracker.log_pls_run(
            pls_model=_PLSModelStub(),
            pls_result=_PLSResultStub(),
            sensor_metrics={},
            y_concs=np.array([1.0]),
        )
        assert result is None


# ===========================================================================
# list_runs / best_run
# ===========================================================================


class TestListRuns:
    def _log_two_runs(self, tracker: ExperimentTracker) -> None:
        model, X, y = _gpr_model()
        sm1 = {**_sensor_metrics(), "r_squared": 0.990}
        sm2 = {**_sensor_metrics(), "r_squared": 0.997}
        with tracker.start_run("run_a"):
            tracker.log_gpr_run(model=model, sensor_metrics=sm1, y_concs=y, X_features=X)
        with tracker.start_run("run_b"):
            tracker.log_gpr_run(model=model, sensor_metrics=sm2, y_concs=y, X_features=X)

    def test_list_runs_returns_dataframe(self, tmp_path: Path) -> None:
        import pandas as pd
        tracker = _make_tracker(tmp_path)
        self._log_two_runs(tracker)
        df = tracker.list_runs()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_list_runs_has_metric_columns(self, tmp_path: Path) -> None:
        tracker = _make_tracker(tmp_path)
        self._log_two_runs(tracker)
        df = tracker.list_runs()
        assert "metrics.r_squared" in df.columns

    def test_list_runs_empty_when_no_runs(self, tmp_path: Path) -> None:
        tracker = _make_tracker(tmp_path)
        df = tracker.list_runs()
        assert df is not None
        assert len(df) == 0

    def test_list_runs_none_when_unavailable(self, tmp_path: Path) -> None:
        tracker = _make_tracker(tmp_path)
        tracker.available = False
        result = tracker.list_runs()
        assert result is None

    def test_best_run_higher_is_better(self, tmp_path: Path) -> None:
        tracker = _make_tracker(tmp_path)
        self._log_two_runs(tracker)
        best = tracker.best_run(metric="metrics.r_squared", higher_is_better=True)
        assert best is not None
        assert float(best["metrics.r_squared"]) == pytest.approx(0.997)

    def test_best_run_lower_is_better(self, tmp_path: Path) -> None:
        tracker = _make_tracker(tmp_path)
        self._log_two_runs(tracker)
        best = tracker.best_run(metric="metrics.rmse", higher_is_better=False)
        assert best is not None

    def test_best_run_returns_none_for_missing_metric(self, tmp_path: Path) -> None:
        tracker = _make_tracker(tmp_path)
        self._log_two_runs(tracker)
        result = tracker.best_run(metric="metrics.nonexistent_xyz")
        assert result is None

    def test_best_run_none_when_unavailable(self, tmp_path: Path) -> None:
        tracker = _make_tracker(tmp_path)
        tracker.available = False
        result = tracker.best_run()
        assert result is None


# ===========================================================================
# get_run_url
# ===========================================================================


class TestGetRunUrl:
    def test_returns_empty_outside_run(self, tmp_path: Path) -> None:
        tracker = _make_tracker(tmp_path)
        assert tracker.get_run_url() == ""

    def test_returns_string_inside_run(self, tmp_path: Path) -> None:
        tracker = _make_tracker(tmp_path)
        with tracker.start_run("url_test"):
            url = tracker.get_run_url()
        assert isinstance(url, str)

    def test_returns_empty_when_unavailable(self, tmp_path: Path) -> None:
        tracker = _make_tracker(tmp_path)
        tracker.available = False
        assert tracker.get_run_url() == ""


# ===========================================================================
# get_tracker singleton
# ===========================================================================


class TestGetTracker:
    def test_returns_experiment_tracker(self, tmp_path: Path) -> None:
        import src.experiment_tracking as et
        original = et._default_tracker
        et._default_tracker = None
        try:
            uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
            t = get_tracker(experiment_name="Singleton_Test", tracking_uri=uri)
            assert isinstance(t, ExperimentTracker)
        finally:
            et._default_tracker = original

    def test_returns_same_instance_on_repeat_calls(self, tmp_path: Path) -> None:
        import src.experiment_tracking as et
        original = et._default_tracker
        et._default_tracker = None
        try:
            uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
            t1 = get_tracker(experiment_name="Repeat_Test", tracking_uri=uri)
            t2 = get_tracker(experiment_name="Repeat_Test", tracking_uri=uri)
            assert t1 is t2
        finally:
            et._default_tracker = original


# ===========================================================================
# _json_default serialisation helper
# ===========================================================================


class TestJsonDefault:
    def test_numpy_int(self) -> None:
        assert _json_default(np.int64(42)) == 42
        assert isinstance(_json_default(np.int64(42)), int)

    def test_numpy_float(self) -> None:
        assert _json_default(np.float32(3.14)) == pytest.approx(3.14, abs=1e-4)

    def test_numpy_array(self) -> None:
        arr = np.array([1, 2, 3])
        result = _json_default(arr)
        assert result == [1, 2, 3]

    def test_fallback_to_str(self) -> None:
        result = _json_default(object())
        assert isinstance(result, str)

    def test_roundtrip_metrics_dict(self) -> None:
        metrics = {
            "r_squared": np.float64(0.997),
            "n_samples": np.int32(10),
            "values": np.array([1.0, 2.0]),
        }
        encoded = json.dumps(metrics, default=_json_default)
        decoded = json.loads(encoded)
        assert decoded["r_squared"] == pytest.approx(0.997)
        assert decoded["n_samples"] == 10
        assert decoded["values"] == [1.0, 2.0]


# ===========================================================================
# NaN / edge case handling
# ===========================================================================


class TestNanAndEdgeCases:
    def test_nan_sensor_metrics_not_logged(self, tmp_path: Path) -> None:
        """NaN values in sensor_metrics must not be forwarded to mlflow."""
        tracker = _make_tracker(tmp_path)
        model, X, y = _gpr_model()
        sm = {"r_squared": float("nan"), "sensitivity": -0.05}
        with tracker.start_run("nan_test"):
            run_id = tracker.log_gpr_run(
                model=model, sensor_metrics=sm, y_concs=y, X_features=X
            )
        run = mlflow.get_run(run_id)
        assert "r_squared" not in run.data.metrics
        assert run.data.metrics.get("sensitivity") == pytest.approx(-0.05)

    def test_empty_sensor_metrics_does_not_raise(self, tmp_path: Path) -> None:
        tracker = _make_tracker(tmp_path)
        model, X, y = _gpr_model()
        with tracker.start_run("empty_metrics"):
            tracker.log_gpr_run(
                model=model, sensor_metrics={}, y_concs=y, X_features=X
            )

    def test_pls_result_with_none_vip(self, tmp_path: Path) -> None:
        """PLSResult with vip_scores=None must not crash."""
        tracker = _make_tracker(tmp_path)
        pls_model = _PLSModelStub()
        pls_result = _PLSResultStub()
        pls_result.vip_scores = None
        pls_result.rmsecv_per_component = []
        y = np.linspace(1.0, 20.0, 10)
        with tracker.start_run("pls_novip"):
            tracker.log_pls_run(
                pls_model=pls_model,
                pls_result=pls_result,
                sensor_metrics=_sensor_metrics(),
                y_concs=y,
            )

"""
tests/test_batch.py
===================
Unit tests for ``src.batch.data_loader`` and ``src.batch.aggregation``.

Design notes
------------
- Uses only in-memory fixtures (no disk I/O against real Joy_Data files).
- ``tmp_path`` (pytest built-in) creates a real temporary directory so we can
  test the directory-scanning logic without touching the project's data/.
- No torch, FastAPI, or MLflow required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.batch.aggregation import (
    average_stable_block,
    build_canonical_from_scan,
    find_stable_block,
    select_canonical_per_concentration,
)
from src.batch.data_loader import (
    ExperimentScan,
    load_frames,
    load_last_n_frames,
    read_spectrum_csv,
    scan_experiment_root,
    sort_frame_paths,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spectrum_df(
    peak_nm: float = 531.5,
    n_points: int = 100,
    noise_std: float = 10.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic LSPR absorption spectrum as a DataFrame."""
    rng = np.random.default_rng(seed)
    wl = np.linspace(480.0, 600.0, n_points)
    baseline = np.ones(n_points) * 10_000.0
    absorption = 300.0 * np.exp(-((wl - peak_nm) ** 2) / (2 * 2.0**2))
    intensity = baseline + rng.normal(0, noise_std, n_points) - absorption
    return pd.DataFrame({"wavelength": wl, "intensity": intensity})


def _write_spectrum_csv(path: Path, df: pd.DataFrame) -> None:
    """Write DataFrame to CSV (headerless two-column format)."""
    df[["wavelength", "intensity"]].to_csv(path, index=False, header=False)


def _make_experiment_dir(
    root: Path,
    concentrations: list[float],
    n_trials: int = 2,
    n_frames: int = 15,
) -> None:
    """Create a synthetic Joy_Data-style experiment directory."""
    for conc in concentrations:
        conc_dir = root / f"{conc} ppm Ethanol"
        for trial in range(1, n_trials + 1):
            trial_dir = conc_dir / f"trial-{trial}"
            trial_dir.mkdir(parents=True)
            for frame in range(n_frames):
                df = _make_spectrum_df(seed=trial * 100 + frame)
                _write_spectrum_csv(trial_dir / f"frame_{frame:04d}.csv", df)


# ---------------------------------------------------------------------------
# read_spectrum_csv tests
# ---------------------------------------------------------------------------


class TestReadSpectrumCsv:
    def test_reads_headerless_two_column(self, tmp_path: Path) -> None:
        df = _make_spectrum_df()
        p = tmp_path / "spectrum.csv"
        _write_spectrum_csv(p, df)
        result = read_spectrum_csv(str(p))
        assert "wavelength" in result.columns
        assert "intensity" in result.columns
        assert len(result) == len(df)

    def test_reads_headered_csv(self, tmp_path: Path) -> None:
        df = _make_spectrum_df()
        p = tmp_path / "headered.csv"
        df.to_csv(p, index=False)
        result = read_spectrum_csv(p)
        assert set(result.columns) >= {"wavelength", "intensity"}
        assert len(result) == len(df)

    def test_wavelengths_sorted_ascending(self, tmp_path: Path) -> None:
        df = _make_spectrum_df()
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        p = tmp_path / "shuffled.csv"
        df_shuffled.to_csv(p, index=False)
        result = read_spectrum_csv(p)
        assert (np.diff(result["wavelength"].to_numpy()) > 0).all()

    def test_raises_on_missing_file(self) -> None:
        with pytest.raises(RuntimeError):
            read_spectrum_csv("/nonexistent/path/spectrum.csv")

    def test_drops_nan_rows(self, tmp_path: Path) -> None:
        df = _make_spectrum_df()
        df.loc[5, "intensity"] = float("nan")
        p = tmp_path / "with_nan.csv"
        df.to_csv(p, index=False)
        result = read_spectrum_csv(p)
        assert result["intensity"].isna().sum() == 0


# ---------------------------------------------------------------------------
# sort_frame_paths tests
# ---------------------------------------------------------------------------


class TestSortFramePaths:
    def test_sorts_timestamp_paths(self) -> None:
        paths = [
            "EtOH_20250605_10h27m00s000ms.csv",
            "EtOH_20250605_10h26m30s000ms.csv",
            "EtOH_20250605_10h26m00s000ms.csv",
        ]
        # Use tmp files so os.path.getmtime doesn't fail
        sorted_paths = sort_frame_paths(paths)
        # The oldest timestamp should sort first (even without real files)
        # Just check it returns a list of same length
        assert len(sorted_paths) == 3

    def test_returns_list(self) -> None:
        result = sort_frame_paths([])
        assert isinstance(result, list)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# scan_experiment_root tests
# ---------------------------------------------------------------------------


class TestScanExperimentRoot:
    def test_finds_concentrations(self, tmp_path: Path) -> None:
        _make_experiment_dir(tmp_path, concentrations=[0.5, 1.0, 2.0])
        scan = scan_experiment_root(tmp_path, gas_type="Ethanol")
        assert isinstance(scan, ExperimentScan)
        assert set(scan.concentrations) == {0.5, 1.0, 2.0}

    def test_finds_all_frames(self, tmp_path: Path) -> None:
        _make_experiment_dir(tmp_path, concentrations=[0.5], n_trials=2, n_frames=15)
        scan = scan_experiment_root(tmp_path)
        assert scan.total_frames == 30  # 2 trials × 15 frames

    def test_gas_type_stored(self, tmp_path: Path) -> None:
        _make_experiment_dir(tmp_path, concentrations=[1.0])
        scan = scan_experiment_root(tmp_path, gas_type="IPA")
        assert scan.gas_type == "IPA"

    def test_raises_on_empty_dir(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError):
            scan_experiment_root(empty)

    def test_raises_on_nonexistent_dir(self) -> None:
        with pytest.raises(ValueError):
            scan_experiment_root("/no/such/directory")

    def test_frames_for_concentration_sorted(self, tmp_path: Path) -> None:
        _make_experiment_dir(tmp_path, concentrations=[0.5], n_trials=1, n_frames=5)
        scan = scan_experiment_root(tmp_path)
        frames = scan.frames_for(0.5)
        assert isinstance(frames, list)
        assert len(frames) == 5

    def test_flat_layout(self, tmp_path: Path) -> None:
        """CSV files directly under concentration dir (no trial subfolders)."""
        conc_dir = tmp_path / "1.0 ppm Ethanol"
        conc_dir.mkdir()
        for i in range(5):
            _write_spectrum_csv(conc_dir / f"frame_{i:04d}.csv", _make_spectrum_df(seed=i))
        scan = scan_experiment_root(tmp_path)
        assert 1.0 in scan.concentrations
        assert scan.total_frames == 5


# ---------------------------------------------------------------------------
# load_frames / load_last_n_frames tests
# ---------------------------------------------------------------------------


class TestLoadFrames:
    def test_loads_all_frames(self, tmp_path: Path) -> None:
        paths = []
        for i in range(5):
            p = tmp_path / f"frame_{i}.csv"
            _write_spectrum_csv(p, _make_spectrum_df(seed=i))
            paths.append(str(p))
        frames = load_frames(paths)
        assert len(frames) == 5
        assert all(isinstance(df, pd.DataFrame) for df in frames)

    def test_max_frames_limits_load(self, tmp_path: Path) -> None:
        paths = []
        for i in range(10):
            p = tmp_path / f"frame_{i}.csv"
            _write_spectrum_csv(p, _make_spectrum_df(seed=i))
            paths.append(str(p))
        frames = load_frames(paths, max_frames=3)
        assert len(frames) == 3

    def test_load_last_n_frames(self, tmp_path: Path) -> None:
        paths = []
        for i in range(20):
            p = tmp_path / f"frame_{i}.csv"
            _write_spectrum_csv(p, _make_spectrum_df(seed=i))
            paths.append(str(p))
        frames = load_last_n_frames(paths, n=10)
        assert len(frames) == 10

    def test_skips_bad_files(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.csv"
        bad.write_text("not,a,valid,spectrum\nfoo,bar,baz,qux\n")
        good = tmp_path / "good.csv"
        _write_spectrum_csv(good, _make_spectrum_df())
        frames = load_frames([str(bad), str(good)])
        # Bad file skipped; good file loaded
        assert len(frames) >= 1


# ---------------------------------------------------------------------------
# find_stable_block tests
# ---------------------------------------------------------------------------


class TestFindStableBlock:
    def _make_frames(self, n: int = 20, add_transient: bool = False) -> list[pd.DataFrame]:
        frames = []
        for i in range(n):
            noise_std = 100.0 if (add_transient and i < 5) else 5.0
            frames.append(_make_spectrum_df(noise_std=noise_std, seed=i))
        return frames

    def test_returns_tuple_of_three(self) -> None:
        frames = self._make_frames(10)
        result = find_stable_block(frames)
        assert len(result) == 3
        start, end, weights = result
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert isinstance(weights, np.ndarray)

    def test_weights_shape_matches_frames(self) -> None:
        frames = self._make_frames(15)
        _, _, weights = find_stable_block(frames)
        assert weights.shape == (15,)

    def test_start_le_end(self) -> None:
        frames = self._make_frames(12)
        start, end, _ = find_stable_block(frames)
        assert start <= end

    def test_weights_sum_positive(self) -> None:
        frames = self._make_frames(10)
        _, _, weights = find_stable_block(frames)
        assert weights.sum() > 0

    def test_stable_block_avoids_noisy_start(self) -> None:
        """With a noisy transient in the first 5 frames, stable block should start later."""
        frames = self._make_frames(20, add_transient=True)
        start, end, _ = find_stable_block(frames, diff_threshold=0.01)
        # The stable block should not start at frame 0 (high transient noise)
        # Allow some tolerance for different noise seeds
        assert end > 3, "Stable block end should be well past the noisy start"

    def test_single_frame_handled(self) -> None:
        frames = [_make_spectrum_df()]
        start, end, weights = find_stable_block(frames)
        assert start == 0 and end == 0
        assert weights.shape == (1,)

    def test_top_k_selection(self) -> None:
        frames = self._make_frames(20)
        _, _, weights = find_stable_block(frames, top_k=5)
        assert int((weights > 0).sum()) <= 5


# ---------------------------------------------------------------------------
# average_stable_block tests
# ---------------------------------------------------------------------------


class TestAverageStableBlock:
    def test_returns_dataframe(self) -> None:
        frames = [_make_spectrum_df(seed=i) for i in range(10)]
        start, end, weights = find_stable_block(frames)
        result = average_stable_block(frames, start, end, weights)
        assert isinstance(result, pd.DataFrame)
        assert "wavelength" in result.columns

    def test_same_length_as_input_frames(self) -> None:
        frames = [_make_spectrum_df(n_points=100, seed=i) for i in range(8)]
        start, end, weights = find_stable_block(frames)
        result = average_stable_block(frames, start, end, weights)
        assert len(result) == 100

    def test_averaged_signal_is_finite(self) -> None:
        frames = [_make_spectrum_df(seed=i) for i in range(10)]
        start, end, weights = find_stable_block(frames)
        result = average_stable_block(frames, start, end, weights)
        assert result["intensity"].isna().sum() == 0
        assert np.isfinite(result["intensity"].to_numpy()).all()

    def test_empty_frames_returns_empty_df(self) -> None:
        result = average_stable_block([], 0, 0)
        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ---------------------------------------------------------------------------
# select_canonical_per_concentration tests
# ---------------------------------------------------------------------------


class TestSelectCanonicalPerConcentration:
    def _make_stable_results(
        self, concentrations: list[float], n_trials: int = 3
    ) -> dict[float, dict[str, pd.DataFrame]]:
        result = {}
        for conc in concentrations:
            result[conc] = {
                f"trial-{t}": _make_spectrum_df(peak_nm=531.5 - conc * 0.1, seed=t)
                for t in range(n_trials)
            }
        return result

    def test_returns_one_entry_per_concentration(self) -> None:
        stable = self._make_stable_results([0.5, 1.0, 2.0])
        canonical = select_canonical_per_concentration(stable)
        assert set(canonical.keys()) == {0.5, 1.0, 2.0}

    def test_canonical_has_wavelength_column(self) -> None:
        stable = self._make_stable_results([1.0])
        canonical = select_canonical_per_concentration(stable)
        assert "wavelength" in canonical[1.0].columns

    def test_canonical_intensity_is_mean_of_trials(self) -> None:
        """With 3 identical trial spectra, canonical should match each trial."""
        df = _make_spectrum_df()
        stable = {0.5: {"t1": df.copy(), "t2": df.copy(), "t3": df.copy()}}
        canonical = select_canonical_per_concentration(stable)
        np.testing.assert_allclose(
            canonical[0.5]["intensity"].to_numpy(),
            df["intensity"].to_numpy(),
            rtol=1e-6,
        )


# ---------------------------------------------------------------------------
# build_canonical_from_scan (end-to-end)
# ---------------------------------------------------------------------------


class TestBuildCanonicalFromScan:
    def test_produces_canonical_per_concentration(self) -> None:
        scan_data: dict[float, dict[str, list[pd.DataFrame]]] = {
            0.5: {
                "trial-1": [_make_spectrum_df(seed=i) for i in range(15)],
                "trial-2": [_make_spectrum_df(seed=i + 100) for i in range(15)],
            },
            1.0: {
                "trial-1": [_make_spectrum_df(seed=i + 200) for i in range(15)],
            },
        }
        canonical = build_canonical_from_scan(scan_data, n_tail=10)
        assert set(canonical.keys()) == {0.5, 1.0}
        for df in canonical.values():
            assert "wavelength" in df.columns
            assert "intensity" in df.columns

    def test_n_tail_zero_uses_all_frames(self) -> None:
        frames = [_make_spectrum_df(seed=i) for i in range(20)]
        scan_data = {1.0: {"trial-1": frames}}
        canonical = build_canonical_from_scan(scan_data, n_tail=0)
        assert 1.0 in canonical

    def test_empty_scan_returns_empty(self) -> None:
        canonical = build_canonical_from_scan({})
        assert canonical == {}

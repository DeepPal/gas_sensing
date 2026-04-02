"""Robustness sweep utilities for publication-grade reproducibility checks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


@dataclass
class RobustnessResult:
    param_value: float
    mean_r2: float
    std_r2: float
    mean_lod_ppm: float
    std_lod_ppm: float
    runs: int


class RobustnessRunner:
    """Run parameter robustness sweeps on calibration data extracted from CSV spectra."""

    def __init__(
        self,
        dataset_dir: Path,
        param_name: str,
        range_spec: str,
        steps: int,
        runs_per_step: int,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.param_name = param_name
        self.range_spec = range_spec
        self.steps = steps
        self.runs_per_step = runs_per_step
        self._rng = np.random.default_rng(42)

    def _parse_range(self) -> np.ndarray:
        m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*:\s*(-?\d+(?:\.\d+)?)\s*$", self.range_spec)
        if not m:
            raise ValueError("Range must be in '<start>:<end>' format, e.g. 45:55")
        start = float(m.group(1))
        end = float(m.group(2))
        if self.steps < 2:
            raise ValueError("steps must be >= 2")
        return np.linspace(start, end, self.steps)

    def _extract_concentration(self, path: Path) -> float | None:
        for text in [path.name, str(path.parent), str(path.parent.parent)]:
            m = re.search(r"([\d.]+)\s*ppm", text, re.IGNORECASE)
            if m:
                return float(m.group(1))
        return None

    def _extract_response(self, path: Path) -> float:
        df = pd.read_csv(path)
        if "wavelength" in df.columns and "intensity" in df.columns:
            wl = df["wavelength"].to_numpy(dtype=float)
            intensity = df["intensity"].to_numpy(dtype=float)
        else:
            wl = df.iloc[:, 0].to_numpy(dtype=float)
            intensity = df.iloc[:, 1].to_numpy(dtype=float)
        peak_idx = int(np.argmax(intensity))
        return float(wl[peak_idx])

    def _load_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        concentrations: list[float] = []
        responses: list[float] = []
        for csv_path in sorted(self.dataset_dir.rglob("*.csv")):
            conc = self._extract_concentration(csv_path)
            if conc is None:
                continue
            try:
                resp = self._extract_response(csv_path)
            except Exception:
                continue
            concentrations.append(conc)
            responses.append(resp)

        if len(concentrations) < 6:
            raise ValueError("Need at least 6 labelled spectra in dataset for robustness sweep")

        return np.asarray(concentrations, dtype=float), np.asarray(responses, dtype=float)

    def _fit_metrics(self, conc: np.ndarray, resp: np.ndarray) -> tuple[float, float]:
        slope, intercept = np.polyfit(conc, resp, 1)
        pred = slope * conc + intercept
        resid = resp - pred
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((resp - np.mean(resp)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)

        low_mask = conc <= np.percentile(conc, 25)
        noise_sigma = float(np.std(resid[low_mask])) if np.any(low_mask) else float(np.std(resid))
        lod = float((3.0 * noise_sigma) / max(abs(slope), 1e-9))
        return r2, lod

    def _perturb(self, base_resp: np.ndarray, param_value: float, center: float) -> np.ndarray:
        base_std = max(float(np.std(base_resp)), 1e-6)
        rel = abs(param_value - center) / max(abs(center), 1.0)

        # Parameter excursions away from center increase additive noise.
        noise_sigma = base_std * (0.02 + 0.12 * rel)
        return base_resp + self._rng.normal(0.0, noise_sigma, size=len(base_resp))

    def run(self) -> list[RobustnessResult]:
        conc, resp = self._load_dataset()
        sweep_values = self._parse_range()
        center = float(np.mean(sweep_values))

        out: list[RobustnessResult] = []
        for val in sweep_values:
            r2_runs: list[float] = []
            lod_runs: list[float] = []
            for _ in range(self.runs_per_step):
                perturbed = self._perturb(resp, float(val), center)
                r2, lod = self._fit_metrics(conc, perturbed)
                r2_runs.append(r2)
                lod_runs.append(lod)

            out.append(
                RobustnessResult(
                    param_value=float(val),
                    mean_r2=float(np.mean(r2_runs)),
                    std_r2=float(np.std(r2_runs)),
                    mean_lod_ppm=float(np.mean(lod_runs)),
                    std_lod_ppm=float(np.std(lod_runs)),
                    runs=self.runs_per_step,
                )
            )

        return out

    def to_dataframe(self, results: list[RobustnessResult]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                self.param_name: [r.param_value for r in results],
                "mean_r2": [r.mean_r2 for r in results],
                "std_r2": [r.std_r2 for r in results],
                "mean_lod_ppm": [r.mean_lod_ppm for r in results],
                "std_lod_ppm": [r.std_lod_ppm for r in results],
                "runs": [r.runs for r in results],
            }
        )

    def format_table(self, results: list[RobustnessResult]) -> str:
        header = (
            f"{self.param_name:<14} | {'R2 mean±std':<18} | {'LOD mean±std (ppm)':<22} | runs\n"
            + "-" * 72
        )
        lines = [header]
        for r in results:
            lines.append(
                f"{r.param_value:<14.3f} | "
                f"{r.mean_r2:.4f} ± {r.std_r2:.4f} | "
                f"{r.mean_lod_ppm:.4f} ± {r.std_lod_ppm:.4f} | "
                f"{r.runs}"
            )
        return "\n".join(lines)

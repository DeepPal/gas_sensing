"""Environment compensation utilities for LSPR calibration reporting.

All functions are CONFIG-free — temperature/humidity reference values,
correction coefficients, and feature flags are passed explicitly.

Typical usage::

    info = compute_environment_summary(
        stable_by_conc, T_ref=25.0, H_ref=50.0, cT=-0.002
    )
    coeffs = compute_environment_coefficients(stable_by_conc, calib_result)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def compute_environment_summary(
    stable_by_conc: dict[float, dict[str, pd.DataFrame]],
    *,
    T_ref: float = 25.0,
    H_ref: float = 50.0,
    cT: Optional[float] = None,
    cH: Optional[float] = None,
    env_enabled: bool = False,
    apply_to_frames: bool = False,
    apply_to_transmittance: bool = True,
    override_temp: Optional[float] = None,
    override_humid: Optional[float] = None,
) -> dict[str, object]:
    """Summarise environment-compensation metadata and per-trial correction offsets.

    Reads ``"temperature"`` and ``"humidity"`` columns from each trial DataFrame
    (if present) and computes the linear correction offset
    ``Δsignal = c_T·(T−T_ref) + c_H·(H−H_ref)`` for each trial.

    Args:
        stable_by_conc: Nested ``{concentration_ppm: {trial_name: DataFrame}}``.
            DataFrames may contain ``"temperature"`` and ``"humidity"`` columns.
        T_ref: Reference temperature in °C (default 25).
        H_ref: Reference relative humidity in % (default 50).
        cT: First-order temperature coefficient (signal per °C).
            ``None`` disables temperature correction.
        cH: First-order humidity coefficient (signal per %).
            ``None`` disables humidity correction.
        env_enabled: Stored in result; indicates whether compensation is active.
        apply_to_frames: Stored in result; whether per-frame correction is applied.
        apply_to_transmittance: Stored in result; whether correction targets transmittance.
        override_temp: Fallback temperature when the column is absent in a trial.
        override_humid: Fallback humidity when the column is absent in a trial.

    Returns:
        Dict with keys ``"enabled"``, ``"apply_to_frames"``,
        ``"apply_to_transmittance"``, ``"reference"``, ``"coefficients"``,
        ``"override"``, ``"temperature_mean"``, ``"humidity_mean"``,
        ``"offset_mean"``, ``"offset_std"``, ``"offset_count"``.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"wavelength": [700.0], "transmittance": [0.9]})
        >>> info = compute_environment_summary({0.5: {"t0": df}})
        >>> info["offset_count"]
        0
    """
    t_vals: list[float] = []
    h_vals: list[float] = []
    offsets: list[float] = []

    for _conc, trials in stable_by_conc.items():
        for _trial, df in trials.items():
            T_val: Optional[float] = None
            H_val: Optional[float] = None
            if "temperature" in df.columns:
                try:
                    T_val = float(pd.to_numeric(df["temperature"], errors="coerce").dropna().mean())
                except Exception:
                    T_val = None
            if "humidity" in df.columns:
                try:
                    H_val = float(pd.to_numeric(df["humidity"], errors="coerce").dropna().mean())
                except Exception:
                    H_val = None
            if T_val is None and override_temp is not None:
                T_val = float(override_temp)
            if H_val is None and override_humid is not None:
                H_val = float(override_humid)
            if T_val is not None:
                t_vals.append(T_val)
            if H_val is not None:
                h_vals.append(H_val)
            off = 0.0
            if cT is not None and T_val is not None:
                off += float(cT) * (T_val - T_ref)
            if cH is not None and H_val is not None:
                off += float(cH) * (H_val - H_ref)
            if off != 0.0 and np.isfinite(off):
                offsets.append(float(off))

    return {
        "enabled": bool(env_enabled),
        "apply_to_frames": bool(apply_to_frames),
        "apply_to_transmittance": bool(apply_to_transmittance),
        "reference": {"temperature": float(T_ref), "humidity": float(H_ref)},
        "coefficients": {
            "temperature": float(cT) if cT is not None else None,
            "humidity": float(cH) if cH is not None else None,
        },
        "override": {
            "temperature": float(override_temp) if override_temp is not None else None,
            "humidity": float(override_humid) if override_humid is not None else None,
        },
        "temperature_mean": float(np.mean(t_vals)) if t_vals else None,
        "humidity_mean": float(np.mean(h_vals)) if h_vals else None,
        "offset_mean": float(np.mean(offsets)) if offsets else 0.0,
        "offset_std": float(np.std(offsets, ddof=1)) if len(offsets) > 1 else 0.0,
        "offset_count": int(len(offsets)),
    }


def compute_environment_coefficients(
    stable_by_conc: dict[float, dict[str, pd.DataFrame]],
    calib: dict[str, object],
    *,
    T_ref: float = 25.0,
    H_ref: float = 50.0,
) -> dict[str, object]:
    """Estimate first-order temperature/humidity correction coefficients via OLS.

    Fits ``y ≈ β₀ + β_c·x + c_T·(T−T_ref) + c_H·(H−H_ref)`` and compares
    R² and RMSE against a concentration-only baseline.

    Args:
        stable_by_conc: Nested dataset used to extract per-concentration T/H means.
        calib: Calibration result dict containing:

            - ``"concentrations"`` – list[float] of concentration levels
            - ``"transformed_concentrations"`` – (optional) pre-transformed x values
            - ``"peak_wavelengths"`` – list[float] of observed peak positions (y)

        T_ref: Reference temperature in °C.
        H_ref: Reference relative humidity in %.

    Returns:
        Dict with ``"estimated_coefficients"`` (sub-keys ``"temperature"``,
        ``"humidity"``, ``"beta_c"``, ``"intercept"``), ``"r2_conc_only"``,
        ``"r2_full"``, ``"delta_r2"``, ``"rmse_conc_only"``, ``"rmse_full"``,
        ``"delta_rmse"``, ``"n_points"``.
        Returns empty dict when no environmental data or insufficient data
        is available.

    Example:
        >>> calib = {"concentrations": [0.5, 1.0], "peak_wavelengths": [717.0, 716.5]}
        >>> result = compute_environment_coefficients({}, calib)
        >>> result  # no env data → empty
        {}
    """
    conc_seq = np.asarray(calib.get("concentrations", []), dtype=float)
    x_seq = np.asarray(calib.get("transformed_concentrations", conc_seq), dtype=float)
    y_seq = np.asarray(calib.get("peak_wavelengths", []), dtype=float)
    if not (conc_seq.size and y_seq.size and conc_seq.size == y_seq.size):
        return {}

    t_by_conc: dict[float, float] = {}
    h_by_conc: dict[float, float] = {}
    for conc, trials in stable_by_conc.items():
        t_vals: list[float] = []
        h_vals: list[float] = []
        for df in trials.values():
            if "temperature" in df.columns:
                tv = pd.to_numeric(df["temperature"], errors="coerce").dropna()
                if not tv.empty:
                    t_vals.append(float(tv.mean()))
            if "humidity" in df.columns:
                hv = pd.to_numeric(df["humidity"], errors="coerce").dropna()
                if not hv.empty:
                    h_vals.append(float(hv.mean()))
        if t_vals:
            t_by_conc[float(conc)] = float(np.mean(t_vals))
        if h_vals:
            h_by_conc[float(conc)] = float(np.mean(h_vals))

    T_arr = np.full_like(conc_seq, np.nan, dtype=float)
    H_arr = np.full_like(conc_seq, np.nan, dtype=float)
    for i, c in enumerate(conc_seq):
        if float(c) in t_by_conc:
            T_arr[i] = t_by_conc[float(c)]
        if float(c) in h_by_conc:
            H_arr[i] = h_by_conc[float(c)]

    have_T = bool(np.isfinite(T_arr).any())
    have_H = bool(np.isfinite(H_arr).any())
    if not (have_T or have_H):
        return {}

    dT = np.where(np.isfinite(T_arr), T_arr - T_ref, 0.0)
    dH = np.where(np.isfinite(H_arr), H_arr - H_ref, 0.0)

    # Concentration-only baseline
    Xc = np.column_stack([np.ones_like(x_seq), x_seq])
    beta_c, *_ = np.linalg.lstsq(Xc, y_seq, rcond=None)
    yhat_c = Xc @ beta_c
    ss_tot = float(np.sum((y_seq - np.mean(y_seq)) ** 2))
    ss_res_c = float(np.sum((y_seq - yhat_c) ** 2))
    r2_c = 1.0 - ss_res_c / ss_tot if ss_tot > 0 else float("nan")
    rmse_c = float(np.sqrt(np.mean((y_seq - yhat_c) ** 2)))

    # Full model with available env vars
    cols = [np.ones_like(x_seq), x_seq]
    names = ["intercept", "beta_c"]
    if have_T and float(np.nanstd(T_arr)) > 0:
        cols.append(dT)
        names.append("cT")
    if have_H and float(np.nanstd(H_arr)) > 0:
        cols.append(dH)
        names.append("cH")
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y_seq, rcond=None)
    yhat = X @ beta
    ss_res = float(np.sum((y_seq - yhat) ** 2))
    r2_full = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse_full = float(np.sqrt(np.mean((y_seq - yhat) ** 2)))

    coeff_map: dict[str, float] = dict(zip(names, beta.tolist()))

    return {
        "estimated_coefficients": {
            "temperature": float(coeff_map["cT"]) if "cT" in coeff_map else None,
            "humidity": float(coeff_map["cH"]) if "cH" in coeff_map else None,
            "beta_c": float(coeff_map["beta_c"]),
            "intercept": float(coeff_map["intercept"]),
        },
        "r2_conc_only": float(r2_c),
        "r2_full": float(r2_full),
        "delta_r2": float(r2_full - r2_c) if np.isfinite(r2_full) and np.isfinite(r2_c) else float("nan"),
        "rmse_conc_only": float(rmse_c),
        "rmse_full": float(rmse_full),
        "delta_rmse": float(rmse_c - rmse_full) if np.isfinite(rmse_c) and np.isfinite(rmse_full) else float("nan"),
        "n_points": int(len(y_seq)),
    }

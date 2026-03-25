"""
src.batch.aggregation
=====================
Temporal-gating and signal aggregation for batch analysis.

The core problem: each trial produces dozens of 5-Hz frames; we need a single
representative spectrum per (concentration, trial) pair.  The strategy is:

1. **Find the stable plateau** — the longest run of consecutive frames where
   the mean-absolute-deviation of the normalised intensity change is below a
   threshold.  This avoids transient artefacts at the start of gas exposure.

2. **Average the plateau frames** — weighted by per-frame intensity (optional)
   to down-weight noisy or low-signal frames.

3. **Aggregate across trials** — weighted average of per-trial representatives
   to produce a single canonical spectrum per concentration.

Public API
----------
- ``align_on_grid(frames)``                → (wl, Y_matrix, weights)
- ``find_stable_block(frames, ...)``       → (start_idx, end_idx, weights)
- ``average_stable_block(frames, ...)``   → representative pd.DataFrame
- ``select_canonical_per_concentration``  → concentration → pd.DataFrame
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grid alignment
# ---------------------------------------------------------------------------


def _signal_column(df: pd.DataFrame) -> str:
    """Return the primary signal column name (transmittance > intensity)."""
    for col in ("transmittance", "intensity", "absorbance"):
        if col in df.columns:
            return col
    # Fallback: first non-wavelength column
    for col in df.columns:
        if col != "wavelength":
            return str(col)
    return "intensity"


def _common_signal_columns(frames: Sequence[pd.DataFrame]) -> list[str]:
    """Columns shared by all frames, excluding 'wavelength'."""
    if not frames:
        return []
    common = set(frames[0].columns) - {"wavelength"}
    for df in frames[1:]:
        common &= set(df.columns) - {"wavelength"}
    return sorted(common)


def align_on_grid(
    frames: Sequence[pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate all *frames* onto the first frame's wavelength grid.

    Returns
    -------
    wl : ndarray, shape (P,)
        Shared wavelength axis.
    Y : ndarray, shape (N, P)
        Intensity matrix (one row per frame).
    weights : ndarray, shape (N,)
        Uniform weights (all 1.0); callers may replace with custom weights.
    """
    if not frames:
        return np.array([]), np.empty((0, 0)), np.array([])

    base_wl = frames[0]["wavelength"].to_numpy(dtype=float)

    rows: list[np.ndarray] = []
    for df in frames:
        col = _signal_column(df)
        vec = df[col].to_numpy(dtype=float)
        wl = df["wavelength"].to_numpy(dtype=float)
        if not np.array_equal(wl, base_wl):
            vec = np.interp(base_wl, wl, vec)
        rows.append(vec)

    Y = np.vstack(rows)
    weights: np.ndarray = np.ones(len(frames), dtype=float)
    return base_wl, Y, weights


# ---------------------------------------------------------------------------
# Stable-block detection
# ---------------------------------------------------------------------------


def find_stable_block(
    frames: Sequence[pd.DataFrame],
    diff_threshold: float = 0.01,
    weight_mode: str = "uniform",
    top_k: int | None = None,
    min_block: int | None = None,
) -> tuple[int, int, np.ndarray]:
    """Find the longest stable run in *frames* and compute per-frame weights.

    Stability is measured by the mean-absolute-deviation of the normalised
    frame-to-frame difference:

    .. math::

        \\text{MAD}(i) = \\frac{\\text{mean}|I_i - I_{i-1}|}{\\text{range}(I_{i-1})}

    A frame is *stable* if ``MAD < diff_threshold``.

    Parameters
    ----------
    frames:
        Chronologically ordered list of DataFrames, each with at least
        ``wavelength`` and a signal column.
    diff_threshold:
        Maximum normalised frame-to-frame difference to be considered stable.
        Typical value: 0.01 (1 % of the signal range).
    weight_mode:
        ``"uniform"`` — equal weights for all plateau frames.
        ``"intensity"`` — weight proportional to mean signal level.
        ``"max"`` — weight proportional to peak signal level.
    top_k:
        If given, keep only the *top_k* highest-signal frames within the
        stable block (further noise reduction).
    min_block:
        Minimum plateau length.  If the detected block is shorter, pad it
        symmetrically toward the centre of the frame sequence.

    Returns
    -------
    start_idx : int
    end_idx : int
        Inclusive indices into *frames* for the stable block.
    weights : ndarray, shape (len(frames),)
        Per-frame weights; zero outside the stable block.
    """
    wl, Y, _ = align_on_grid(frames)

    n = Y.shape[0]
    if n == 0:
        return 0, 0, np.array([1.0])
    if n == 1:
        return 0, 0, np.array([1.0])

    # Frame-to-frame normalised MAD
    eps = 1e-9
    stable_flags = np.ones(n, dtype=bool)
    for i in range(1, n):
        prev, curr = Y[i - 1], Y[i]
        rng = max(float(prev.max() - prev.min()), eps)
        mad = float(np.mean(np.abs(curr - prev))) / rng
        stable_flags[i] = mad < diff_threshold

    # Longest stable run
    best_len = best_start = 0
    curr_len = curr_start = 0
    for i, stable in enumerate(stable_flags):
        if stable:
            if curr_len == 0:
                curr_start = i
            curr_len += 1
            if curr_len > best_len:
                best_len, best_start = curr_len, curr_start
        else:
            curr_len = 0

    # Fall back to 3-frame window centred on middle.
    # This means no stable plateau was found — log a warning so downstream
    # callers know the "stable block" is actually an unconverged fallback.
    if best_len <= 1:
        mid = n // 2
        best_start = max(0, mid - 1)
        best_len = min(3, n - best_start)
        log.warning(
            "find_stable_block: no stable plateau found in %d frames "
            "(diff_threshold=%.4f). Falling back to 3-frame window at "
            "frame %d–%d. Downstream quality metrics may be unreliable.",
            n, diff_threshold, best_start, best_start + best_len - 1,
        )

    start_idx = best_start
    end_idx = best_start + best_len - 1

    # Pad to min_block if needed
    if min_block and min_block > 0:
        desired = min(int(min_block), n)
        current = end_idx - start_idx + 1
        if current < desired:
            pad = desired - current
            start_idx = max(0, start_idx - pad // 2)
            end_idx = min(n - 1, end_idx + (pad - pad // 2))
            if end_idx - start_idx + 1 < desired:
                start_idx = max(0, end_idx - desired + 1)

    # Build weight vector
    mask = np.zeros(n, dtype=bool)
    mask[start_idx : end_idx + 1] = True
    weights = np.zeros(n, dtype=float)
    weights[mask] = 1.0

    mode = (weight_mode or "uniform").lower()
    if mode in {"intensity", "max"}:
        block = Y[mask]
        if block.size > 0:
            scores = block.mean(axis=1) if mode == "intensity" else block.max(axis=1)
            scores = np.clip(scores, 1e-9, None)
            weights[mask] = scores

    # top_k selection within the block
    if top_k and top_k > 0:
        block_indices = np.flatnonzero(mask)
        if block_indices.size > top_k:
            scores = Y[mask].mean(axis=1)
            top_in_block = np.argsort(scores)[-top_k:]
            new_mask = np.zeros(n, dtype=bool)
            new_mask[block_indices[top_in_block]] = True
            mask = new_mask
            weights = np.zeros(n, dtype=float)
            weights[mask] = 1.0
            start_idx = int(block_indices[top_in_block].min())
            end_idx = int(block_indices[top_in_block].max())

    # Ensure weights sum to something positive
    if weights.sum() <= 0:
        weights[mask] = 1.0

    return start_idx, end_idx, weights


# ---------------------------------------------------------------------------
# Weighted averaging
# ---------------------------------------------------------------------------


def average_stable_block(
    frames: Sequence[pd.DataFrame],
    start_idx: int,
    end_idx: int,
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute a weighted-average spectrum over the stable block.

    Parameters
    ----------
    frames:
        Full chronological list of frames for a trial.
    start_idx, end_idx:
        Inclusive indices from :func:`find_stable_block`.
    weights:
        Per-frame weights from :func:`find_stable_block`.  If ``None``,
        uniform weights are used.

    Returns
    -------
    pd.DataFrame
        Single representative spectrum with ``wavelength`` + signal columns.
    """
    if not frames:
        return pd.DataFrame()

    n = len(frames)
    base_wl = frames[0]["wavelength"].to_numpy(dtype=float)

    start_idx = max(0, min(start_idx, n - 1))
    end_idx = max(start_idx, min(end_idx, n - 1))
    indices = list(range(start_idx, end_idx + 1))

    if weights is not None and len(weights) == n:
        frame_weights = weights[indices]
    else:
        frame_weights = np.ones(len(indices), dtype=float)

    common_cols = _common_signal_columns(frames)
    if not common_cols:
        common_cols = [_signal_column(frames[0])]

    out = pd.DataFrame({"wavelength": base_wl})
    for col in common_cols:
        accum = np.zeros_like(base_wl, dtype=float)
        total_w = 0.0
        for rank, fi in enumerate(indices):
            df = frames[fi]
            if col not in df.columns:
                continue
            w = float(frame_weights[rank])
            if w <= 0:
                continue
            vec = df[col].to_numpy(dtype=float)
            wl = df["wavelength"].to_numpy(dtype=float)
            if not np.array_equal(wl, base_wl):
                vec = np.interp(base_wl, wl, vec)
            accum += w * vec
            total_w += w
        if total_w > 0:
            out[col] = accum / total_w

    return out


# ---------------------------------------------------------------------------
# Canonical spectrum selection
# ---------------------------------------------------------------------------


def select_canonical_per_concentration(
    stable_results: dict[float, dict[str, pd.DataFrame]],
    trial_weights: dict[float, dict[str, float]] | None = None,
) -> dict[float, pd.DataFrame]:
    """Weighted average of per-trial representative spectra per concentration.

    Parameters
    ----------
    stable_results:
        ``concentration → trial_label → representative_DataFrame``.
        Each DataFrame must have ``wavelength`` + at least one signal column.
    trial_weights:
        Optional ``concentration → trial_label → weight`` mapping.
        Defaults to uniform weighting if ``None``.

    Returns
    -------
    Dict[float, pd.DataFrame]
        ``concentration → canonical_spectrum``.
    """
    canonical: dict[float, pd.DataFrame] = {}

    for conc, trials in stable_results.items():
        if not trials:
            continue

        base_wl: np.ndarray | None = None
        accum: dict[str, np.ndarray] = {}
        weight_sums: dict[str, float] = {}

        for trial_name, df in trials.items():
            wl = df["wavelength"].to_numpy(dtype=float)
            if base_wl is None:
                base_wl = wl

            # Resolve per-trial weight
            weight = 1.0
            if trial_weights is not None:
                conc_tw = trial_weights.get(conc, {})
                try:
                    weight = float(conc_tw.get(trial_name, 1.0))
                except (TypeError, ValueError):
                    weight = 1.0
            if not np.isfinite(weight) or weight <= 0:
                continue

            for col in df.columns:
                if col == "wavelength":
                    continue
                vec = df[col].to_numpy(dtype=float)
                if not np.array_equal(wl, base_wl):
                    vec = np.interp(base_wl, wl, vec)
                if col not in accum:
                    accum[col] = weight * vec
                    weight_sums[col] = weight
                else:
                    accum[col] += weight * vec
                    weight_sums[col] += weight

        if base_wl is None or not accum:
            continue

        canonical_df = pd.DataFrame({"wavelength": base_wl})
        for col, summed in accum.items():
            w_total = weight_sums.get(col, 0.0)
            if w_total > 0:
                canonical_df[col] = summed / w_total
        canonical[conc] = canonical_df

    return canonical


# ---------------------------------------------------------------------------
# High-level convenience
# ---------------------------------------------------------------------------


def average_top_frames(frames: list[pd.DataFrame], top_k: int = 5) -> pd.DataFrame:
    """Average the ``top_k`` frames with the highest mean signal intensity.

    This is an alternative to :func:`find_stable_block` + :func:`average_stable_block`
    when temporal ordering is not informative (e.g., random-order batch).

    Parameters
    ----------
    frames:
        List of DataFrames, each with ``wavelength`` + at least one signal column.
    top_k:
        Number of highest-scoring frames to include in the average.

    Returns
    -------
    pd.DataFrame
        Averaged spectrum on the wavelength grid of the first frame, or an
        empty DataFrame if ``frames`` is empty.
    """
    if not frames:
        return pd.DataFrame()

    priority = ("intensity", "transmittance", "absorbance")
    signal_col = _signal_column(frames[0])
    for col in priority:
        if all(col in df.columns for df in frames):
            signal_col = col
            break

    scores = [
        (float(df[signal_col].mean()) if signal_col in df.columns else -1e9, i)
        for i, df in enumerate(frames)
    ]
    scores.sort(key=lambda t: t[0], reverse=True)

    top_k = max(1, min(len(frames), top_k))
    top_indices = [idx for _, idx in scores[:top_k]]
    selected = [frames[i] for i in top_indices]

    base_wl = selected[0]["wavelength"].values
    accum = np.zeros_like(base_wl, dtype=float)
    count = 0
    for df in selected:
        if signal_col not in df.columns:
            continue
        wl = df["wavelength"].values
        vec = df[signal_col].values
        if not np.array_equal(wl, base_wl):
            vec = np.interp(base_wl, wl, vec)
        accum += vec
        count += 1

    result = pd.DataFrame({"wavelength": base_wl})
    if count > 0:
        result[signal_col] = accum / count
        if signal_col != "intensity":
            result["intensity"] = result[signal_col]
    return result


def build_canonical_from_scan(
    scan_data: dict[float, dict[str, list[pd.DataFrame]]],
    diff_threshold: float = 0.01,
    n_tail: int = 10,
    weight_mode: str = "uniform",
) -> dict[float, pd.DataFrame]:
    """End-to-end: frames → canonical spectra.

    Parameters
    ----------
    scan_data:
        ``concentration → trial_label → [list of loaded DataFrames]``.
        Frames must already be chronologically sorted.
    diff_threshold:
        Passed to :func:`find_stable_block`.
    n_tail:
        If provided, only the last *n_tail* frames of each trial are used
        (temporal tail gating — assumes the gas response has plateaued).
        Set to 0 to disable.
    weight_mode:
        Passed to :func:`find_stable_block`.

    Returns
    -------
    Dict[float, pd.DataFrame]
        Canonical (representative) spectrum per concentration.
    """
    stable_results: dict[float, dict[str, pd.DataFrame]] = {}

    for conc, trials in scan_data.items():
        stable_results[conc] = {}
        for trial_label, frames in trials.items():
            if not frames:
                continue
            # Temporal tail gating
            tail = frames[-n_tail:] if n_tail > 0 else frames
            if not tail:
                continue
            start, end, weights = find_stable_block(
                tail, diff_threshold=diff_threshold, weight_mode=weight_mode
            )
            rep = average_stable_block(tail, start, end, weights)
            if not rep.empty:
                stable_results[conc][trial_label] = rep

    return select_canonical_per_concentration(stable_results)

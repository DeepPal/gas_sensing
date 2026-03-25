import numpy as np


def calculate_t90_t10(
    time: np.ndarray, signal: np.ndarray, on_time: float, off_time: float, baseline_window: int = 10
) -> dict[str, float]:
    """
    Calculate T90 (Response Time) and T10 (Recovery Time).

    Args:
        time: Time vector (seconds)
        signal: Signal vector (intensity/absorbance)
        on_time: Timestamp when gas was turned ON
        off_time: Timestamp when gas was turned OFF
        baseline_window: Number of points to average for baseline/steady-state

    Returns:
        Dictionary with 't90', 't10', 'response_amplitude'
    """
    # 1. Define Regions
    # Baseline: just before ON
    idx_on = np.searchsorted(time, on_time)
    idx_off = np.searchsorted(time, off_time)

    if idx_on == 0 or idx_off >= len(time):
        return {"t90": np.nan, "t10": np.nan, "status": "invalid_indices"}

    baseline = np.mean(signal[max(0, idx_on - baseline_window) : idx_on])

    # steady state: just before OFF (assuming equilibrium reached)
    steady_state = np.mean(signal[max(0, idx_off - baseline_window) : idx_off])

    delta = steady_state - baseline
    target_90 = baseline + 0.9 * delta
    target_10 = baseline + 0.1 * delta  # For recovery

    # 2. Calculate T90 (Response)
    # Search in window [ON, OFF]
    response_slice = signal[idx_on:idx_off]
    time_slice = time[idx_on:idx_off]

    # Find first crossing of 90%
    if delta > 0:
        # Increasing signal
        crossings_90 = np.where(response_slice >= target_90)[0]
    else:
        # Decreasing signal (if dipping)
        crossings_90 = np.where(response_slice <= target_90)[0]

    t90_val = np.nan
    if len(crossings_90) > 0:
        t90_time = time_slice[crossings_90[0]]
        t90_val = t90_time - on_time

    # 3. Calculate T10 (Recovery)
    # Search in window [OFF, END]
    recovery_slice = signal[idx_off:]
    recovery_time_slice = time[idx_off:]

    # Find first return to 10%
    if delta > 0:
        # Recovering downwards
        crossings_10 = np.where(recovery_slice <= target_10)[0]
    else:
        # Recovering upwards
        crossings_10 = np.where(recovery_slice >= target_10)[0]

    t10_val = np.nan
    if len(crossings_10) > 0:
        t10_time = recovery_time_slice[crossings_10[0]]
        t10_val = t10_time - off_time

    return {
        "t90": t90_val,
        "t10": t10_val,
        "amplitude": delta,
        "baseline": baseline,
        "steady_state": steady_state,
    }


def estimate_response_time(series: np.ndarray, sampling_rate: float = 1.0) -> float:
    """Wrapper for simple series without explicit timestamps."""
    # Placeholder for more complex auto-detection logic
    pass

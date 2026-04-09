# Agent 01 — Connect Spectrometer & Live Data View
**Stage A · Step 1 of 6 · Dashboard: Step 1 (top section)**

---

## Purpose
Establish a real-time connection to the ThorLabs CCS200 spectrometer and display live spectral data on the dashboard. This is the entry point for the entire pipeline. All downstream agents depend on a valid spectrum frame.

---

## Hardware Spec
| Parameter | Value |
|-----------|-------|
| Instrument | ThorLabs CCS200 |
| Interface | USB 2.0/3.0 via TLCCS DLL |
| Wavelength range | 200–1000 nm (3648 pixels) |
| Integration time range | 10 ms – 5000 ms |
| Acquisition rate | ~2.4 Hz at 50 ms integration |
| Driver module | `gas_analysis/acquisition/ccs200_native.py` |

---

## Source Files
| File | Role |
|------|------|
| `gas_analysis/acquisition/ccs200_native.py` | `CCS200Spectrometer` — DLL wrapper |
| `dashboard/agentic_pipeline_tab.py` | `_acquire_frames()`, Step 1 Live Preview section |

---

## Dashboard Location
`dashboard/agentic_pipeline_tab.py` → Step 1 → **"📡 Live Data View"** section (above the metadata form)

---

## Behaviour

### 1. Hardware Discovery & Connection
```python
from gas_analysis.acquisition.ccs200_native import CCS200Spectrometer
spec = CCS200Spectrometer(integration_time_s=0.030)
wl = spec.get_wavelengths()          # np.ndarray, shape (3648,), units: nm
```
- If the DLL is not found or no device is connected, fall through to **Simulation Mode**.
- Simulation mode generates a Gaussian peak + noise:
  ```python
  wl = np.linspace(200, 1000, 1000)
  frame = np.exp(-0.5 * ((wl - 532) / 18)**2) * 4.5 + np.random.normal(0, 0.04, 1000)
  ```

### 2. Integration Time Configuration
- UI: `st.number_input("Integration (ms)", 10, 5000, 30, key="ap_prev_int")`
- Passed directly to `CCS200Spectrometer(integration_time_s=ms/1000.0)`.

### 3. Live Preview (read-only, not recorded)
```python
wl_p, fr_p = _acquire_frames(integration_ms, n_frames=5, concentration_ppm=0.0, chart_ph)
```
- Renders in `chart_preview_ph = prev_left.empty()` — updated in-place.
- Computes and displays SNR and RMS noise as live quality indicators.

### 4. Quality Gates (per frame)
```python
from gas_analysis.core.preprocessing import compute_snr, estimate_noise_metrics
snr = compute_snr(frame)             # target: ≥ 10.0
metrics = estimate_noise_metrics(wl, frame)
noise_ok = metrics.rms < 0.1        # target: < 0.1
saturated = frame.max() > 60000     # flag saturation
```

### 5. VISA Error Handling (Confirmed Patterns)
| Error code | Meaning | Resolution |
|-----------|---------|-----------|
| `-1073807339` | `VI_ERROR_TMO` — stale handle | Close + reopen device |
| `-1074001152` | `TLCCS_ERROR_SCAN_PENDING` | Wait and retry |
| `-1073807343` | Device connected but not powered | Check power |

Always wrap in `try/finally` to call `spec.close()` on teardown.

### 6. Warm-Up
`_acquire_frames()` uses `_warmup_scan()` internally: fires a throwaway scan before the first real read to drain stale state.

---

## Session State Outputs
| Key | Type | Consumed By |
|-----|------|------------|
| `ap_preview_wl` | `np.ndarray` | UI display, Step 1 |
| `ap_preview_frame` | `np.ndarray` | UI display, Step 1 |
| `ap_wl` | `np.ndarray` | Agents 02–05 |
| `ap_buffer[*].wl` | `np.ndarray` | Agents 03–05 |

---

## Done When
- Live preview chart renders at ≥1 Hz with SNR ≥ 10.
- Hardware/simulation status clearly shown in UI.
- → Pass control to **Agent 02** (Pre-Acquisition Logging).

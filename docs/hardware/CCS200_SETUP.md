# Thorlabs CCS200 Hardware Setup

This guide covers installing the CCS200 driver, confirming the device is recognised,
and troubleshooting common errors. The steps are Windows-specific (the native DLL path
is Windows-only; Linux/macOS users must use the VISA fallback).

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Thorlabs TLCCS driver | Download from Thorlabs website → Software section for CCS200 |
| Python venv | `.venv/Scripts/python.exe` — must have `ctypes` (stdlib, always present) |
| USB cable | Type-A to Mini-B; use the cable supplied with the CCS200 |
| Optional: NI-VISA or Thorlabs VISA | Required only for VISA fallback path |

Install order matters: **install the Thorlabs driver before plugging in the device**.

---

## 2. Driver Installation

1. Download **Thorlabs CCS Series Instrument Driver** from the Thorlabs website.
2. Run the installer as Administrator.
3. Reboot when prompted.
4. Plug the CCS200 into a USB port.
5. Open **Device Manager** → *Test and Measurement Devices* → confirm `CCS200` appears.

The installer places the DLL at:

```
C:\Program Files\IVI Foundation\VISA\Win64\Bin\TLCCS_64.dll
```

SpectraAgent searches this path automatically. No additional configuration is needed.

---

## 3. Verify Connection

Run the built-in probe:

```bash
python -m spectraagent plugins list
```

Expected output when connected:

```
Hardware Drivers:
  [thorlabs_ccs200]  spectraagent.drivers.thorlabs:ThorlabsDriver  --  [OK] loadable

Sensor Physics Plugins:
  [lspr]  spectraagent.physics.lspr:LSPRPhysicsPlugin  --  [OK] loadable
```

Then start with hardware:

```bash
python -m spectraagent start
# Hardware badge in browser should show: "Live - CCS200"
```

---

## 4. Driver Selection Priority

On `spectraagent start`, the driver is selected in this order:

1. `--simulate` flag → `SimulationDriver` (forced)
2. `cfg.hardware.default_driver` from `spectraagent.toml` → loaded via entry-point
3. If entry-point load fails → `SimulationDriver` (fallback, warning printed to stderr)

The default in `spectraagent.toml` is `thorlabs_ccs200`. The VISA and serial paths
in `gas_analysis/acquisition/` are available as last-resort fallbacks for the legacy
`run.py` pipeline but are not used by the SpectraAgent runtime.

---

## 5. Integration Time

Default integration time is **50 ms** (~2.4 Hz acquisition rate).

Change it in `spectraagent.toml`:

```toml
[hardware]
integration_time_ms = 100.0   # 100 ms → ~1.2 Hz
```

Or at runtime via the React frontend calibration panel, or via the API:

```bash
curl -X POST http://localhost:8765/acquisition/config \
     -H "Content-Type: application/json" \
     -d '{"integration_time_ms": 100.0, "gas_label": "Ethanol"}'
```

Valid range: **10 ms – 60 000 ms** (hardware limit).

---

## 6. Warm-Up

The CCS200 requires a **warm-up scan** after power-on to drain stale state from any
previous session that closed uncleanly. `ThorlabsDriver.connect()` fires a throwaway
`startScan → getScanData` cycle automatically — you do not need to wait manually.

Allow **2–5 minutes** after powering on the light source before taking a reference
spectrum. The light source output is thermally unstable until the lamp reaches
operating temperature.

---

## 7. Error Codes

| Code | Hex | Meaning | Fix |
|---|---|---|---|
| `-1073807339` | `0xBFFF0015` | `VI_ERROR_TMO` — timeout on first read | Device was not closed cleanly in last session. Unplug and replug USB, restart SpectraAgent. |
| `-1074001152` | `0xBFFA8000` | `TLCCS_ERROR_SCAN_PENDING` | Previous scan still running. SpectraAgent adds a 250 ms post-scan cooldown to prevent this. |
| `-1073807343` | `0xBFFF0011` | Device not ready | Device connected via VISA but not powered. Check USB power, try a different port. |
| `-1073807346` | `0xBFFF000E` | Resource not found | DLL not found or device not recognised by OS. Reinstall Thorlabs driver. |

If `plugins list` shows `[FAIL]` with a DLL path error, the Thorlabs driver is not
installed or the 64-bit DLL is missing. Reinstall using the 64-bit installer.

---

## 8. Acquisition Calling Convention

For reference when writing custom drivers or debugging:

```python
from ctypes import cdll, c_double, c_uint32

dll = cdll.LoadLibrary(r"C:\Program Files\IVI Foundation\VISA\Win64\Bin\TLCCS_64.dll")
# getScanData: pass buffer directly, NOT byref()
buf = (c_double * 3648)()
dll.tlccs_startScan(handle)
time.sleep(integration_ms / 1000.0 + 0.25)   # integration + 250 ms margin
dll.tlccs_getScanData(handle, buf)             # no argtypes set
time.sleep(0.10)                               # 100 ms cooldown between scans
intensities = list(buf)                        # 3648 floats
```

Key points:
- Pass `buf` **directly** (not `byref(buf)`).
- Do **not** set `argtypes` for `getScanData` — ctypes default calling convention works.
- Total cycle time at 50 ms integration: ~410 ms (~2.4 Hz).

---

## 9. Dark Noise Characterisation

With the light source off and lens cap on:

| Metric | Expected value |
|---|---|
| Max intensity | ~0.008 (dimensionless) |
| Min intensity | ~−0.004 |
| RSD at dark noise | ~12% (normal — low signal amplifies relative noise) |

Dark noise RSD > 20% suggests the device needs recalibration or the integration time
is too short. An `ERROR: saturation` message from `QualityAgent` at dark noise indicates
the saturation threshold in `spectraagent.toml` is set too low.

---

## 10. Using the Research-Facing Registry

Scripts and notebooks that do not use the full SpectraAgent server can access the
CCS200 through the `SpectrometerRegistry`:

```python
from src.spectrometer.registry import SpectrometerRegistry

with SpectrometerRegistry.create("ccs200") as spec:
    spec.open()
    dark = spec.acquire_dark()          # cap on
    ref  = spec.acquire_reference()     # reference gas / blank
    sample = spec.acquire()             # sample gas
    print(f"Peak: {sample.peak_wavelength:.3f} nm")
```

If the CCS200 DLL is not present, `create("ccs200")` raises `KeyError`.
Use `create("simulated")` for development without hardware.

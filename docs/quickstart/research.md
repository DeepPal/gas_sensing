# Getting Started — Research Labs

## Prerequisites

- Python 3.10 or 3.11
- 8 GB RAM (16 GB recommended for CNN inference)
- ThorLabs CCS200 spectrometer (optional — simulated mode works without hardware)

## Installation (two options)

**Option A — Docker (recommended, no Python setup)**
```bash
docker compose up
```

**Option B — pip install**
```bash
git clone https://github.com/DeepPal/spectraagent
cd spectraagent
python -m venv .venv
.venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -e ".[all]"
```

## Running a calibration session (Streamlit)

```bash
# Simulated data (no hardware needed)
streamlit run dashboard/app.py
```

Navigate to **Tab 1 — Agentic Pipeline**:

1. **Step 1 — Load data:** Upload CSV files from `data/Joy_Data/` or connect live hardware
2. **Step 2 — Reference spectrum:** Load a blank measurement to establish Δλ = 0
3. **Step 3 — Calibration:** Fit GPR (Matérn ν=5/2) to concentration series; inspect R² and LOD
4. **Step 4 — Analysis:** Allan deviation, FOM, residual diagnostics, publication tables

## Running live acquisition (SpectraAgent)

```bash
# Simulated hardware
spectraagent start --simulate

# Real ThorLabs CCS200
spectraagent start --hw
```

Then open http://localhost:8765/app

## Exporting results

From the Streamlit dashboard:
- **CSV:** Session → Export → CSV
- **HDF5:** Session → Export → HDF5 archive (reproducible)
- **HTML report:** Session → Generate Report

## Scientific definitions

All outputs are governed by `CHARTER.md` and `docs/adr/science/`:

| Output | Method | Reference |
|--------|--------|-----------|
| LOD | IUPAC 2011, 3σ blank / sensitivity | `docs/adr/science/001-lod-definition.md` |
| Calibration | GPR, Matérn ν=5/2 kernel | `docs/adr/science/002-gpr-kernel.md` |
| Primary signal | Δλ (peak wavelength shift, nm) | `docs/adr/science/003-lspr-signal-convention.md` |
| FOM | \|S\|/FWHM (Willets & Van Duyne 2007) | `docs/adr/science/004-fom-definition.md` |
| Prediction interval | Conformal prediction, 95% coverage | `docs/adr/science/005-conformal-prediction.md` |

## Citing this software

```bibtex
@software{spectraagent2026,
  title  = {SpectraAgent: Universal Agentic Spectroscopy Platform},
  author = {Chula Research Team},
  year   = {2026},
  url    = {https://github.com/DeepPal/spectraagent}
}
```

See also `CITATION.cff` for machine-readable citation metadata.

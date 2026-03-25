# API Reference

Auto-generated from the typed `src/` package (mypy-clean, 46 source files).

All public functions and classes include:

- Full type signatures (from annotations)
- Docstrings (Google style)
- Source links

## Package layout

```
src/
├── preprocessing/     Spectrum preprocessing (baseline, denoising, normalisation)
├── calibration/       Nonlinear isotherm fitting (Langmuir, Freundlich, Hill)
├── features/          LSPR feature extraction
├── models/            CNN classifier + ONNX export
├── scientific/        ICH Q2 LoD/LoQ, selectivity matrix
├── batch/             Multi-frame aggregation
├── inference/         Real-time orchestrator
├── api/               FastAPI REST server (also at /docs when running)
├── agents/            Quality agent (spectrum QC)
└── training/          MLflow tracking, ablation studies
```

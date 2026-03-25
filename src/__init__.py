"""
src
===
Au-MIP LSPR Gas Sensing Platform — modular source package.

Layer order (lower number = fewer dependencies):
  Layer 0: configs/          — YAML loader, no project imports
  Layer 1: src/schemas/      — Pydantic contracts
  Layer 2: src/preprocessing/— pure signal-processing functions
  Layer 3: src/features/     — feature extraction
  Layer 4: src/models/       — ML model definitions
  Layer 5: src/calibration/  — calibration methods
  Layer 6: src/inference/    — real-time pipeline + orchestration
  Layer 7: src/batch/        — offline analysis pipeline
  Layer 8: src/training/     — MLflow training wrappers
  Layer 9: src/api/          — FastAPI inference server
  Layer 10: src/agents/      — drift/quality/training agents
"""

__version__ = "3.0.0"

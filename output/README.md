# Output Directory Layout

This folder contains all generated analysis results and presentation assets.

## Structure

- `scientific/` – Per‑gas scientific pipeline outputs (plots, metrics, reports)
  - `Acetone/`, `Ethanol/`, `Methanol/`, `Isopropanol/`, `Toluene/`, `Xylene/`, `MixVOC/`
- `world_class/` – Multi‑gas comparative analysis and world‑class mode outputs
- `publication_figures/` – Publication‑ready figures and tables (Figure 1–5)
- `dist/` – Exported presentation bundles for downstream consumption
  - `presentation_assets/` – Multi‑gas bundles with manifests and SHA checksums

## How each folder is generated

- `scientific/{Gas}/` – Created by `run_scientific_pipeline.py --gas <Gas>` (or via the unified CLI)
- `world_class/` – Created by `run_world_class_analysis.py` (or `pipeline run world-class`)
- `publication_figures/` – Created by the world‑class or publication‑generation scripts
- `dist/presentation_assets/` – Created by `export_presentation_assets.py --gases …` (or `pipeline export`)

## Usage notes

- Do not edit files in `scientific/` or `world_class/` manually; they are regenerated on each pipeline run.
- Use `export_presentation_assets.py` to bundle the latest results for the `Kevin_acetone_ppt` presentation repo.
- `dist/` is ignored by git (see top‑level `.gitignore`).

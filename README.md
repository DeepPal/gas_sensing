# Gas Sensing Pipeline

A scientifically rigorous, minimal-output pipeline for optical fiber gas sensing analysis with automated presentation asset generation.

## Quick Start

```bash
# Run the full end-to-end refresh (scientific + world-class + export + PPT)
python pipeline.py refresh

# Run only the scientific pipeline for a specific gas
python pipeline.py run scientific --gas Acetone

# Export presentation assets for all gases
python pipeline.py export

# Validate project health
python pipeline.py check --require-scientific --require-export
```

## Project Structure

```
├── pipeline.py                 # Unified CLI (run, export, refresh, check)
├── run_scientific_pipeline.py  # Scientific analysis (legacy, called by pipeline.py)
├── export_presentation_assets.py
├── config/
│   └── config.yaml             # Central configuration (paths, ROI defaults, etc.)
├── output/
│   ├── scientific/              # Per‑gas pipeline outputs
│   │   ├── Acetone/
│   │   ├── Ethanol/
│   │   └── …
│   ├── world_class/            # Multi‑gas comparative analysis
│   ├── publication_figures/    # Publication‑ready figures (Figure 1–5)
│   └── dist/
│       └── presentation_assets/  # Multi‑gas export bundles (ignored by git)
├── Kevin_acetone_ppt/          # Presentation repo (pure consumer)
│   ├── config/
│   │   └── presentation_scientific.yaml
│   ├── generated_assets/       # Auto‑synced from dist/presentation_assets
│   └── slides_automation/      # PPT generation library
└── Kevin_Data/                  # Raw experimental CSVs
    ├── Acetone/
    ├── Ethanol/
    └── …
```

## Commands Reference

### `pipeline.py run <mode>`
Run a specific pipeline mode.

- `scientific` – Validated scientific pipeline (publication‑ready)
- `world-class` – Multi‑gas comparative analysis
- `ml-enhanced` – ML‑enhanced analysis with feature engineering
- `comparative` – Comparative analysis across all gases
- `debug` – Debug mode with detailed logging
- `validation` – Run comprehensive system validation

```bash
python pipeline.py run scientific --gas Acetone
python pipeline.py run world-class
```

### `pipeline.py export`
Bundle canonical figures, metrics, and narrative text for presentation consumption.

```bash
python pipeline.py export --gases Acetone,Ethanol --dest dist/presentation_assets
```

### `pipeline.py refresh`
End‑to‑end automation: run pipelines, export assets, optionally generate PPTs.

```bash
python pipeline.py refresh --skip-ppt
```

### `pipeline.py check`
Validate project health (data paths, outputs, export bundles).

```bash
python pipeline.py check --require-scientific --require-export
```

## Adding a New Gas

1. Add CSV data under `Kevin_Data/<GasName>/`.
2. Add the gas name to `DEFAULT_GASES` in `pipeline.py`.
3. Run `python pipeline.py run scientific --gas <GasName>` to generate outputs.
4. The export and refresh commands automatically include the new gas.

## Presentation Workflow

1. Run a full refresh or export to populate `dist/presentation_assets/{Gas}/`.
2. Assets are automatically synced into `Kevin_acetone_ppt/generated_assets/exported/{Gas}/`.
3. Update `Kevin_acetone_ppt/config/presentation_scientific.yaml` to reference the new assets.
4. Generate slides with `python pipeline.py refresh` (or manually via slides_automation CLI).

## Configuration

Central configuration lives in `config/config.yaml`. You can override paths, ROI defaults, and other parameters there or via CLI flags (`--config`).

## Reproducibility

- All pipelines log provenance (git commit, config SHA256, environment versions).
- Export bundles include SHA256 checksums per file.
- `dist/` is git‑ignored; regenerate assets via `pipeline.py export`.

## Troubleshooting

- If a gas is missing from export, ensure `output/scientific/<Gas>/` exists and contains `plots/`, `metrics/`, `reports/`.
- Use `pipeline.py check` to validate data paths and required outputs.
- Logs for each run are stored in `pipeline_logs/` (created by the orchestration scripts).

## Legacy Scripts

The following scripts remain for compatibility but are superseded by `pipeline.py`:
- `run_scientific_pipeline.py`
- `run_world_class_analysis.py`
- `export_presentation_assets.py`
- `run_full_refresh.py`

Prefer `pipeline.py` for all new work.

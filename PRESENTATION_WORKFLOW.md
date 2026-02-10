# Presentation Asset Workflow

## Goals
1. Maintain this repository (`Code_Acetone_paper_3`) as the *only* source of truth for canonical metrics, figures, and narratives.
2. Treat `Kevin_acetone_ppt/` as a lightweight presentation/manuscript workspace that consumes exported assets without duplicating scientific logic.
3. Provide a repeatable handoff so that PPT/Docx generation always tracks the latest validated pipeline outputs.

## Repository Roles
| Repo | Purpose | Key Directories |
|------|---------|-----------------|
| `Code_Acetone_paper_3/` | Data, pipelines, canonical Markdown docs, publication figures | `output/{gas}_scientific/`, `UNIFIED_RESULTS.md`, `CODEMAP.md` |
| `Code_Acetone_paper_3/Kevin_acetone_ppt/` | Presentation automation toolkit | `generated_assets/`, `config/presentation_scientific.yaml`, `data/` |

## Asset Flow
```
            ┌──────────────────────────────┐
            │  Code_Acetone_paper_3/       │
            │  (source of truth)           │
            └──────────────┬───────────────┘
                           │ export_presentation_assets.py
                           ▼
            ┌──────────────────────────────┐
            │  dist/presentation_assets/   │  (staging)
            └──────────────┬───────────────┘
                           │ sync (robocopy/rsync/manual)
                           ▼
            ┌──────────────────────────────┐
            │  Kevin_acetone_ppt/          │
            │  generated_assets/ & data/   │
            └──────────────────────────────┘
```

### Responsibilities
1. **Source repo**
   - Regenerate plots via `run_scientific_pipeline.py` or `unified_pipeline.py`.
   - Keep Markdown narratives (UNIFIED_RESULTS, CODEMAP, REPORT) updated.
   - Run `python export_presentation_assets.py --dest <...>` to package figures, metrics, and summary text.

2. **Presentation repo** (`Kevin_acetone_ppt/`)
   - Consume artifacts under `generated_assets/` and `data/` as declared in `config/presentation_scientific.yaml`.
   - Run `python -m slides_automation.cli ...` to build PPTX / Google Slides decks or manuscripts.

## Export Package Contents
`dist/presentation_assets/` is the canonical staging area. Each export run rewrites:
| File/Folder | Source | Description |
|-------------|--------|-------------|
| `plots/*.png` | `output/acetone_scientific/plots/` | Calibration curve, ROI heatmap, etc. |
| `metrics/calibration_metrics.json` | `output/acetone_scientific/metrics/` | Machine-readable results |
| `reports/summary.md` | `output/acetone_scientific/reports/` | Scientific narrative snippet |
| `text/unified_results.md` | `UNIFIED_RESULTS.md` | Canonical highlights |
| `text/key_figures.json` | Script-generated digest (ROI, sensitivity, LOD, etc.) |

## Sync Options
1. **Manual copy** – drag `dist/presentation_assets/*` into `Kevin_acetone_ppt/generated_assets/` and `data/`.
2. **Robocopy (Windows)**
   ```powershell
   robocopy dist\presentation_assets Kevin_acetone_ppt\generated_assets /E
   ```
3. **Rsync (WSL/Linux/macOS)**
   ```bash
   rsync -av dist/presentation_assets/ Kevin_acetone_ppt/generated_assets/
   ```

## Best Practices
- Always run the export script *after* regenerating figures to avoid stale metrics.
- Keep `dist/presentation_assets/manifest.json` (auto-generated) in version control to audit which source files fed each PPT build.
- Do **not** edit exported files inside `Kevin_acetone_ppt/`; re-run the export instead.
- If the presentation repo needs custom diagrams, store them under `Kevin_acetone_ppt/generated_assets/custom/` so they are not overwritten by exports.

## Next Steps
1. Implement `export_presentation_assets.py` (see accompanying script) to automate package creation.
2. Update `Kevin_acetone_ppt/config/presentation_scientific.yaml` to reference files under `generated_assets/`/`data/` populated by the exporter.
3. Optional: add a simple PowerShell or Bash helper in `Kevin_acetone_ppt/scripts/` to pull the latest `dist/presentation_assets/`.

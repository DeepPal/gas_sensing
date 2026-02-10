# Presentation Assets (Consumer)

This repo consumes generated assets from the main gas‑sensing pipeline. It should **not** contain any manually edited figures or stale results.

## How assets are populated

- `generated_assets/exported/{Gas}/` – Auto‑synced from `dist/presentation_assets/{Gas}/` by the main repo’s export step.
- `config/presentation_scientific.yaml` – References assets under `../generated_assets/` and contains canonical placeholder values.
- `slides_automation/` – PPT generation library; uses the config to build Google Slides or PowerPoint files.

## Usage

1. In the main repo, run `python pipeline.py refresh` (or `python pipeline.py export`).
2. Assets are automatically synced into `generated_assets/exported/`.
3. Generate slides with `python -m slides_automation.cli --config config/presentation_scientific.yaml`.

## What NOT to do

- Do not edit files under `generated_assets/`; they are overwritten on each export.
- Do not store legacy figures or results in this repo.
- Do not manually edit placeholder values in the config; they should match the canonical metrics from the pipeline.

## Folder layout

```
Kevin_acetone_ppt/
├── config/
│   └── presentation_scientific.yaml
├── generated_assets/
│   └── exported/
│       ├── Acetone/
│       ├── Ethanol/
│       └── …
├── slides_automation/
└── dist/  (generated PPTX/Google Slides)
```

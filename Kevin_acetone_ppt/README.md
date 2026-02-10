# Presentation Automation Toolkit · Primary Command → `python -m slides_automation.cli --config config/presentation_scientific.yaml --no-google`

A flexible hybrid workflow that clones your master Google Slides template, swaps in experiment data, and optionally builds a fully editable local PPTX deck. You keep complete manual control over the design while the Python tooling handles repetitive slide generation.

## Features
- **Template-driven automation** – Google Slides + Drive APIs clone your template, replace placeholders, embed images, and share the draft deck automatically.
- **Offline PPTX generation** – `python-pptx` fallback builds local decks for cases where Google Slides is not desired or offline work is required.
- **Config-first orchestration** – `config/presentation_scientific.yaml` declares credentials, placeholders, slide definitions, data sources, and every generated visual asset.
- **Data plumbing** – CSV/TSV/Excel/JSON/Markdown sources hydrate tables or narrative sections directly inside slides.
- **CLI + Python API** – run from the command line (`presentation-generator`) or import `PresentationProject` inside notebooks/scripts.

## Repository Layout
```
├── config/
│   ├── presentation_scientific.yaml   # Active config (full deck)
│   └── archive/…                      # Older configs kept for reference
├── generated_assets/                  # Auto-generated PNG visuals referenced by the config
├── slides_automation/                 # Core automation library (config loader, CLI, pptx builder, etc.)
├── scripts/generate_diagrams*.py      # Regenerate assets in generated_assets/
├── data/                              # CSV tables injected into slides
├── dist/                              # Generated PPTX exports
├── core.py                            # Simple script runner (optional)
├── pyproject.toml                     # Dependencies + console entrypoint
└── README.md
```

## Prerequisites
1. **Python 3.10+**
2. **Virtual environment** (recommended)
3. **Google Cloud project** with Slides + Drive APIs enabled and a **Service Account JSON** downloaded locally.

Install dependencies:
```bash
pip install -e .
```

## Google Slides Setup
1. Visit <https://console.cloud.google.com/> and create a project.
2. Enable **Google Slides API** and **Google Drive API**.
3. Create a **Service Account**, generate a JSON key file, and place it in the repo (e.g., `credentials.json`).
4. Share your master Google Slides template with the service account email (viewer is enough since the Drive copy API is used).
5. Update `config/presentation_scientific.yaml` (or your cloned config):
   - `credentials_file`: path to the JSON key.
   - `google_slides.template_id`: the ID portion of your Slides template URL.
   - Optionally set `folder_id` (Drive folder to hold generated decks) and `share_emails` for automatic sharing.

## Local Template / Assets
- Drop any supporting images under `assets/` and point `image_path` entries to them.
- If you maintain a PowerPoint template, set `pptx.template_path` to the `.pptx` file.
- `data/sensor_results.csv` demonstrates how a CSV table is injected; add your own data sources and reference them via `table_source` keys.

## Running the Toolkit
### CLI (recommended)
```bash
python -m slides_automation.cli \
  --config config/presentation_scientific.yaml \
  --no-google \
  --output-name full_scientific
```

Useful flags:
- `--no-google` / `--no-pptx` – skip either output.
- `--placeholders-json overrides.json` – bulk placeholder overrides.
- `--output-name spr_dec_2025` – custom filename for PPTX export.
- `--dry-run` – validate config without generating artifacts.
- Remove `--no-google` once Google Drive storage quota is available to generate Slides decks automatically.

### Python entrypoint
```bash
python core.py
```
This script loads the config you point it to (defaulting to `presentation_scientific.yaml` once you edit `core.py`), runs the orchestrator, and logs resulting URLs/paths.

### Programmatic usage
```python
from pathlib import Path
from slides_automation.config import load_config
from slides_automation.orchestrator import PresentationProject

config = load_config(Path("config/presentation_scientific.yaml"))
project = PresentationProject(config)
result = project.run(title="Experiment SPR_042")
print(result)
```

## Customizing Slides
- **Placeholders**: define `{SOME_TAG}` in your Google Slides template or local slide definitions, then provide values under `placeholders` or via CLI overrides.
- **Images**: `google_slides.image_uploads` uploads local files to Drive and positions them using inch-based coordinates.
- **Tables**: specify `table_source` with a key pointing to a CSV/Excel/JSON dataset loaded via `data_sources`.
- **Speaker notes**: `notes` fields in slide definitions populate the Notes pane for presenter guidance.

## Manuscript Generation
The toolkit also generates publication-ready DOCX manuscripts from Markdown templates:

```bash
python -m slides_automation.cli --manuscript --autogen-manuscript --no-google --no-pptx
```

This will:
1. Auto-generate key Results/Discussion sections from CSV data sources
2. Convert Markdown to DOCX via Pandoc
3. Post-process the DOCX for single-column, compact journal-like styling (A4, 0.75" margins, Times New Roman 10pt)

Output: `help_files/MANUSCRIPT_DRAFT.docx`

## PPTX Audit
Audit generated PPTX files for picture overflow and tight margins:

```bash
python -m slides_automation.audit_pptx dist/your-deck.pptx --min-margin-in 0.25
```

Optionally write a JSON report:
```bash
python -m slides_automation.audit_pptx dist/your-deck.pptx --out audit_report.json --fail-on-issues
```

## Next Steps
1. Replace template IDs, credential paths, and placeholder values with your actual research assets.
2. Expand `config/presentation_scientific.yaml` or duplicate it per study (store older variants under `config/archive/`).
3. Hook the CLI into your data pipeline or lab notebook for one-command presentation drafts.
4. Add regression tests (e.g., `pytest`) if you plan to extend business logic further.

For troubleshooting, enable verbose logs via `--log-level DEBUG`. For Google Slides runs, ensure your Drive storage quota has free space—`storageQuotaExceeded` will halt remote deck creation until space is cleared.

```bash
# Generate PPTX only
python -m slides_automation.cli --no-google

# Generate manuscript DOCX
python -m slides_automation.cli --manuscript --autogen-manuscript --no-google --no-pptx

# Audit PPTX for overflow
python -m slides_automation.audit_pptx dist/your-deck.pptx --min-margin-in 0.25

# Validate config without generating
python -m slides_automation.cli --validate-only --strict
```
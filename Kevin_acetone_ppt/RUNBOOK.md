# Gas Analysis & Presentation Runbook

**Project:** Data-Driven Spectral Feature Engineering for Sub-ppm Acetone Detection

---

## Step 1: Run Gas Analysis Pipeline (Required First)

### Run Scientific Pipeline (Primary)
```powershell
cd "c:\Users\deepp\Desktop\Chula_Work\PRojects\kevin gas sensing_AI improve\Code_Acetone_paper_3"
python run_scientific_pipeline.py --gas Acetone
```

### Run All Pipeline Modes
```powershell
# Scientific mode (default)
python pipeline.py run scientific --gas Acetone

# World-class analysis
python pipeline.py run world-class --gas Acetone

# ML-enhanced analysis
python pipeline.py run ml-enhanced --gas Acetone

# Comparative analysis (all gases)
python pipeline.py run comparative

# Export presentation assets
python pipeline.py export --gases Acetone,Ethanol,Methanol
```

### Pipeline Outputs
- **Scientific results:** `output/scientific/Acetone/`
- **Canonical spectra:** `output/scientific/Acetone/canonical_spectra/`
- **Calibration curves:** `output/scientific/Acetone/calibration/`
- **Validation metrics:** `output/scientific/Acetone/validation/`
- **Reports:** `output/scientific/Acetone/reports/`

---

## Step 2: Generate Presentation (After Pipeline)

### Generate All Diagrams
```powershell
cd "c:\Users\deepp\Desktop\Chula_Work\PRojects\kevin gas sensing_AI improve\Code_Acetone_paper_3\Kevin_acetone_ppt"
python scripts\generate_diagrams.py
```
- Outputs: `generated_assets/*.png` and `generated_assets/*.pdf`

### Build the PowerPoint Presentation
```powershell
python -m slides_automation.cli --config config\presentation_scientific.yaml --no-google
```
- Output: `dist/data-driven-spectral-feature-engineering-for-sub-ppm-acetone-detection.pptx`

---

## Configuration Files

| File | Purpose |
|------|---------|
| `config/presentation_scientific.yaml` | Slide content, titles, notes, image references |
| `scripts/generate_diagrams.py` | Diagram generation code |
| `data/` | Input data files |
| `output/scientific/` | Canonical spectra, metrics, processed data |

---

## Common Tasks

### Edit Slide Text
1. Open `config/presentation_scientific.yaml`
2. Find the slide section (search by title)
3. Edit `title`, `subtitle`, `notes`, or `content`
4. Rerun Step 2

### Edit a Diagram
1. Open `scripts/generate_diagrams.py`
2. Find the corresponding function (e.g., `create_feature_engineering_progression`)
3. Make changes
4. Rerun Step 1, then Step 2

### Add New Diagram
1. Add new function in `scripts/generate_diagrams.py`
2. Call it from `main()` function
3. Add reference in `config/presentation_scientific.yaml`
4. Rerun Step 1, then Step 2

---

## Output Locations

- **Diagrams:** `generated_assets/`
- **Final PPT:** `dist/data-driven-spectral-feature-engineering-for-sub-ppm-acetone-detection.pptx`
- **Build Manifest:** `dist/*.manifest.json`

---

## Troubleshooting

### Font/Glyph Warnings
- Avoid special Unicode characters (icons, emojis) in matplotlib text
- Use standard ASCII labels

### Tight Layout Warnings
- Complex multi-panel figures may show warnings
- Usually safe to ignore; check output PNG visually

### Missing Data Files
- Verify paths in `output/scientific/Acetone/canonical_spectra/`
- Check `data/` directory for required inputs

---

## Complete Workflow (One Command)

```powershell
cd "c:\Users\deepp\Desktop\Chula_Work\PRojects\kevin gas sensing_AI improve\Code_Acetone_paper_3" && python run_scientific_pipeline.py --gas Acetone && cd "Kevin_acetone_ppt" && python scripts\generate_diagrams.py && python -m slides_automation.cli --config config\presentation_scientific.yaml --no-google
```

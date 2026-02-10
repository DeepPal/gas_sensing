# Deprecation Notice

The following scripts are superseded by the unified `pipeline.py` CLI. They remain for compatibility but should not be used in new workflows.

## Superseded Scripts

- `run_scientific_pipeline.py` → Use `python pipeline.py run scientific --gas <Gas>`
- `run_world_class_analysis.py` → Use `python pipeline.py run world-class`
- `export_presentation_assets.py` → Use `python pipeline.py export`
- `run_full_refresh.py` → Use `python pipeline.py refresh`
- `comparative_analysis.py` → Use `python pipeline.py run comparative`
- `validate_installation.py` → Use `python pipeline.py run validation`
- `run_debug.py` → Use `python pipeline.py run debug`

## Migration Examples

```bash
# Old
python run_scientific_pipeline.py --gas Acetone
# New
python pipeline.py run scientific --gas Acetone

# Old
python run_world_class_analysis.py
# New
python pipeline.py run world-class

# Old
python export_presentation_assets.py --gases Acetone,Ethanol
# New
python pipeline.py export --gases Acetone,Ethanol

# Old
python run_full_refresh.py --skip-ppt
# New
python pipeline.py refresh --skip-ppt
```

The unified CLI provides:
- Consistent argument parsing and help
- Centralized configuration via `config/config.yaml`
- Health checks via `pipeline.py check`
- Standardized output layout under `output/`

## Timeline

- These legacy scripts will be removed in a future release.
- New features will only be added to `pipeline.py`.
- Documentation and examples will reference the unified CLI.

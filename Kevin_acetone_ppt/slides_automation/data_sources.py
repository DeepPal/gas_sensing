"""Data source loading utilities for the slides_automation package.

Supports loading CSV, TSV, Excel, JSON, and text/Markdown files into
memory for use in slide generation and manuscript automation.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd

LOGGER = logging.getLogger(__name__)


def load_data_sources(paths: Dict[str, str]) -> Dict[str, Any]:
    """Load the configured data sources into memory.

    Supported formats:
        - .csv → pandas.DataFrame
        - .tsv → pandas.DataFrame
        - .xlsx/.xls → pandas.DataFrame
        - .json → dict/list
        - .txt/.md → raw string
    """

    loaded: Dict[str, Any] = {}
    for name, location in paths.items():
        path = Path(location)
        if not path.exists():
            LOGGER.warning("Data source %s does not exist at %s", name, path)
            continue

        suffix = path.suffix.lower()
        try:
            if suffix in {".csv", ".tsv"}:
                sep = "\t" if suffix == ".tsv" else ","
                df = pd.read_csv(path, sep=sep)
                
                # Validation checks
                if df.empty:
                    LOGGER.warning("Data source %s is empty", name)
                    
                if df.isnull().values.any():
                    null_cols = df.columns[df.isnull().any()].tolist()
                    LOGGER.warning("Data source %s has missing values in columns: %s", name, null_cols)
                
                # Check for object columns that could be numeric
                for col in df.select_dtypes(include=['object']).columns:
                    try:
                        pd.to_numeric(df[col], errors='raise')
                        LOGGER.info("Column '%s' in %s could be converted to numeric type", col, name)
                    except (ValueError, TypeError):
                        pass
                
                loaded[name] = df
                
            elif suffix in {".xlsx", ".xls"}:
                loaded[name] = pd.read_excel(path)
            elif suffix == ".json":
                loaded[name] = json.loads(path.read_text(encoding="utf-8"))
            elif suffix in {".md", ".txt"}:
                loaded[name] = path.read_text(encoding="utf-8")
            else:
                LOGGER.warning("Unsupported data format %s for %s", suffix, name)
                continue
            LOGGER.info("Loaded data source '%s' from %s", name, path)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.error("Failed to load %s from %s: %s", name, path, exc)
    return loaded

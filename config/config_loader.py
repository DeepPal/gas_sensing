import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the YAML configuration file.

    Args:
        config_path: Optional explicit path to the configuration file. If None,
                     load `config.yaml` from the `config` package directory.

    Returns:
        A dictionary containing the configuration settings.
    """
    if config_path is None:
        # Resolve to the config.yaml located alongside this module
        path = Path(__file__).resolve().parent / "config.yaml"
    else:
        path = Path(config_path)

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}

    return config

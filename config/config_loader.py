"""
config.config_loader
====================
YAML configuration loader for the Au-MIP LSPR gas sensing platform.

Key features
------------
- **Duplicate-key detection**: raises an error if the same key appears twice
  in a mapping, which YAML itself allows but usually indicates a mistake.
- **Structural validation**: checks required numeric and boolean fields in
  the ``roi``, ``response_series``, and ``preprocessing`` sections.
- **Module-level cache**: :data:`CONFIG` is updated in-place by
  :func:`load_config` so callers that imported it before the first
  ``load_config()`` call still see the current values.

Typical usage
-------------
::

    from config.config_loader import load_config

    cfg = load_config("config/config.yaml")
    smoothing_lam = cfg["preprocessing"]["als_lambda"]

Notes
-----
- All values are returned as plain Python objects (dicts, lists, scalars).
- No environment-variable substitution is performed; use a secrets manager
  or ``.env`` file for values that should not be committed to source control.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal: YAML loader that rejects duplicate mapping keys
# ---------------------------------------------------------------------------


class _UniqueKeyLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` extended to raise on duplicate mapping keys."""


def _construct_mapping(
    loader: yaml.SafeLoader,
    node: yaml.Node,
    deep: bool = False,
) -> dict[object, object]:
    """Build a dict from a YAML mapping node, rejecting duplicate keys."""
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key: {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


# ---------------------------------------------------------------------------
# Module-level config cache (updated in-place by load_config)
# ---------------------------------------------------------------------------

CONFIG: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------


def _as_float(value: object, default: float) -> float:
    """Safely coerce *value* to float; return *default* on failure."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _require_positive_float(section: str, key: str, value: object) -> None:
    """Raise ``ValueError`` if *value* is not a positive float."""
    fv = _as_float(value, -1.0)
    if fv <= 0:
        raise ValueError(f"{section}.{key} must be a positive number, got {value!r}")


def _require_positive_float_list(section: str, key: str, value: object) -> None:
    """Raise ``ValueError`` if *value* is not a list of positive floats."""
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"{section}.{key} must not be an empty list")
        bad = [v for v in value if _as_float(v, -1.0) <= 0]
        if bad:
            raise ValueError(f"{section}.{key} values must be positive, got {bad!r}")
    elif value is not None:
        _require_positive_float(section, key, value)


def _validate_config(config: dict[str, Any]) -> None:
    """Validate the loaded configuration dict.

    Raises
    ------
    TypeError
        If *config* is not a dict.
    ValueError
        If a required numeric or boolean field has an invalid value.
    """
    if not isinstance(config, dict):
        raise TypeError(
            f"Configuration must be a YAML mapping (dict), got {type(config).__name__!r}"
        )

    # ------------------------------------------------------------------
    # roi.shift
    # ------------------------------------------------------------------
    roi_cfg = config.get("roi") or {}
    if isinstance(roi_cfg, dict):
        shift_cfg = roi_cfg.get("shift") or {}
        if isinstance(shift_cfg, dict) and shift_cfg:
            step_nm = shift_cfg.get("step_nm")
            if step_nm is not None:
                _require_positive_float("roi.shift", "step_nm", step_nm)

            window_nm = shift_cfg.get("window_nm")
            _require_positive_float_list("roi.shift", "window_nm", window_nm)

        # ------------------------------------------------------------------
        # roi.discovery
        # ------------------------------------------------------------------
        discovery_cfg = roi_cfg.get("discovery") or {}
        if isinstance(discovery_cfg, dict) and discovery_cfg.get("enabled"):
            step_nm = discovery_cfg.get("step_nm")
            if step_nm is not None:
                _require_positive_float("roi.discovery", "step_nm", step_nm)

            window_nm = discovery_cfg.get("window_nm")
            if window_nm is not None:
                _require_positive_float("roi.discovery", "window_nm", window_nm)

    # ------------------------------------------------------------------
    # response_series
    # ------------------------------------------------------------------
    response_cfg = config.get("response_series") or {}
    if isinstance(response_cfg, dict) and response_cfg.get("enabled"):
        min_frames = response_cfg.get("min_activation_frames")
        if min_frames is not None:
            try:
                min_frames_i = int(min_frames)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"response_series.min_activation_frames must be an integer, got {min_frames!r}"
                ) from exc
            if min_frames_i < 1:
                raise ValueError(
                    f"response_series.min_activation_frames must be >= 1, got {min_frames_i}"
                )

    # ------------------------------------------------------------------
    # preprocessing
    # ------------------------------------------------------------------
    preproc_cfg = config.get("preprocessing") or {}
    if isinstance(preproc_cfg, dict):
        enabled = preproc_cfg.get("enabled")
        if enabled is not None and not isinstance(enabled, bool):
            raise ValueError(
                f"preprocessing.enabled must be a boolean (true/false), "
                f"got {type(enabled).__name__!r}"
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load, parse, and validate the YAML configuration file.

    Parameters
    ----------
    config_path:
        Path to the YAML file.  If ``None``, defaults to
        ``config/config.yaml`` relative to this module's directory.

    Returns
    -------
    dict
        The validated configuration as a plain Python dict.  The module-level
        :data:`CONFIG` dict is also updated in-place.

    Raises
    ------
    FileNotFoundError
        If *config_path* does not point to an existing file.
    TypeError
        If the YAML root is not a mapping.
    ValueError
        If a required configuration value has an invalid value.
    yaml.YAMLError
        If the YAML is malformed or contains duplicate keys.
    """
    if config_path is None:
        path = Path(__file__).resolve().parent / "config.yaml"
    else:
        path = Path(config_path)

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    log.debug("Loading configuration from %s", path)

    with path.open(encoding="utf-8") as fh:
        loaded: Any = yaml.load(fh, Loader=_UniqueKeyLoader) or {}

    if not isinstance(loaded, dict):
        raise TypeError(f"Configuration root must be a YAML mapping, got {type(loaded).__name__!r}")

    _validate_config(loaded)

    CONFIG.clear()
    CONFIG.update(loaded)

    log.info("Configuration loaded (%d top-level keys) from %s", len(CONFIG), path)
    return CONFIG

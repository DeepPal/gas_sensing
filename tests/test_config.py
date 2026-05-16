"""
tests/test_config.py
====================
Unit tests for ``config.config_loader``.

Coverage targets
----------------
- ``load_config`` happy path (minimal valid YAML)
- ``load_config`` with the real ``config/config.yaml``
- Duplicate-key detection
- Negative / zero ``step_nm`` rejection
- Empty ``window_nm`` list rejection
- Non-boolean ``preprocessing.enabled`` rejection
- Non-positive ``min_activation_frames`` rejection
- Missing file raises ``FileNotFoundError``
- CONFIG module-level cache is updated after load
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from config.config_loader import CONFIG, load_config

# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestLoadConfigHappyPath:
    """Tests that valid configurations load without error."""

    def test_minimal_config_returns_dict(self, minimal_config_yaml: Path) -> None:
        result = load_config(minimal_config_yaml)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_config_cache_updated(self, minimal_config_yaml: Path) -> None:
        """The module-level CONFIG dict must be updated in-place after load."""
        CONFIG.clear()
        load_config(minimal_config_yaml)
        assert len(CONFIG) > 0

    def test_returns_same_object_as_cache(self, minimal_config_yaml: Path) -> None:
        result = load_config(minimal_config_yaml)
        # load_config returns CONFIG directly
        assert result is CONFIG

    def test_real_config_yaml_loads(self, full_config_path: Path) -> None:
        """The production config/config.yaml must be valid and loadable."""
        result = load_config(full_config_path)
        assert isinstance(result, dict)
        # Basic structural check — sections expected in a proper config
        expected_sections = {"roi", "preprocessing", "sensor"}
        present = expected_sections & set(result.keys())
        assert present, (
            f"Expected at least one of {expected_sections} in config, "
            f"found top-level keys: {list(result.keys())}"
        )

    def test_preprocessing_enabled_bool_true(self, tmp_path: Path) -> None:
        cfg_text = "preprocessing:\n  enabled: true\n"
        p = tmp_path / "cfg.yaml"
        p.write_text(cfg_text, encoding="utf-8")
        result = load_config(p)
        assert result["preprocessing"]["enabled"] is True

    def test_preprocessing_enabled_bool_false(self, tmp_path: Path) -> None:
        cfg_text = "preprocessing:\n  enabled: false\n"
        p = tmp_path / "cfg.yaml"
        p.write_text(cfg_text, encoding="utf-8")
        result = load_config(p)
        assert result["preprocessing"]["enabled"] is False

    def test_default_path_used_when_none_given(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When config_path=None, the loader resolves to config/config.yaml."""
        import config.config_loader as _mod

        config_dir = Path(_mod.__file__).resolve().parent
        expected = config_dir / "config.yaml"
        if not expected.exists():
            pytest.skip("Real config.yaml not present — skipping default-path test")
        result = load_config(None)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Error-path tests
# ---------------------------------------------------------------------------


class TestLoadConfigErrors:
    """Tests that invalid configurations raise appropriate exceptions."""

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_config(tmp_path / "does_not_exist.yaml")

    def test_duplicate_key_raises_yaml_error(self, invalid_config_yaml_duplicate_key: Path) -> None:
        with pytest.raises(Exception):  # yaml.YAMLError or ConstructorError
            load_config(invalid_config_yaml_duplicate_key)

    def test_negative_roi_shift_step_nm_raises(
        self, invalid_config_yaml_negative_step: Path
    ) -> None:
        with pytest.raises(ValueError, match="step_nm"):
            load_config(invalid_config_yaml_negative_step)

    def test_zero_roi_shift_step_nm_raises(self, tmp_path: Path) -> None:
        cfg = {"roi": {"shift": {"step_nm": 0.0, "window_nm": 5.0}}}
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.dump(cfg), encoding="utf-8")
        with pytest.raises(ValueError, match="step_nm"):
            load_config(p)

    def test_empty_window_nm_list_raises(self, tmp_path: Path) -> None:
        cfg = {"roi": {"shift": {"step_nm": 0.1, "window_nm": []}}}
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.dump(cfg), encoding="utf-8")
        with pytest.raises(ValueError, match="window_nm"):
            load_config(p)

    def test_negative_window_nm_in_list_raises(self, tmp_path: Path) -> None:
        cfg = {"roi": {"shift": {"step_nm": 0.1, "window_nm": [5.0, -1.0]}}}
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.dump(cfg), encoding="utf-8")
        with pytest.raises(ValueError, match="window_nm"):
            load_config(p)

    def test_non_boolean_preprocessing_enabled_raises(self, tmp_path: Path) -> None:
        cfg = {"preprocessing": {"enabled": "yes"}}
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.dump(cfg), encoding="utf-8")
        with pytest.raises(ValueError, match="enabled"):
            load_config(p)

    def test_response_series_min_frames_zero_raises(self, tmp_path: Path) -> None:
        cfg = {"response_series": {"enabled": True, "min_activation_frames": 0}}
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.dump(cfg), encoding="utf-8")
        with pytest.raises(ValueError, match="min_activation_frames"):
            load_config(p)

    def test_response_series_min_frames_non_int_raises(self, tmp_path: Path) -> None:
        cfg = {"response_series": {"enabled": True, "min_activation_frames": "abc"}}
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.dump(cfg), encoding="utf-8")
        with pytest.raises(ValueError, match="min_activation_frames"):
            load_config(p)

    def test_non_mapping_root_raises_type_error(self, tmp_path: Path) -> None:
        p = tmp_path / "list_root.yaml"
        p.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(TypeError, match="mapping"):
            load_config(p)

    def test_roi_discovery_negative_step_nm_raises(self, tmp_path: Path) -> None:
        cfg = {
            "roi": {
                "shift": {"step_nm": 0.1},
                "discovery": {"enabled": True, "step_nm": -0.5, "window_nm": 10.0},
            }
        }
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.dump(cfg), encoding="utf-8")
        with pytest.raises(ValueError, match="step_nm"):
            load_config(p)


# ---------------------------------------------------------------------------
# Content / value tests
# ---------------------------------------------------------------------------


class TestConfigValues:
    """Tests that parsed values match expected types and values."""

    def test_roi_shift_step_nm_is_float(self, minimal_config_yaml: Path) -> None:
        cfg = load_config(minimal_config_yaml)
        step = cfg["roi"]["shift"]["step_nm"]
        assert isinstance(step, float)
        assert step > 0

    def test_roi_shift_window_nm_is_list(self, minimal_config_yaml: Path) -> None:
        cfg = load_config(minimal_config_yaml)
        window = cfg["roi"]["shift"]["window_nm"]
        assert isinstance(window, list)
        assert all(isinstance(w, (int, float)) and w > 0 for w in window)

    def test_response_series_min_frames_int(self, minimal_config_yaml: Path) -> None:
        cfg = load_config(minimal_config_yaml)
        val = cfg["response_series"]["min_activation_frames"]
        assert isinstance(val, int) and val >= 1

    def test_quality_section_present(self, minimal_config_yaml: Path) -> None:
        cfg = load_config(minimal_config_yaml)
        assert "quality" in cfg
        assert cfg["quality"]["min_snr"] == pytest.approx(4.0)

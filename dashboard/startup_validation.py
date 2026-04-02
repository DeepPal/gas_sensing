"""
dashboard/startup_validation.py - Pre-flight checks for research reliability
============================================================================

Validates all critical systems before allowing dashboard to run.
Ensures data integrity, hardware availability, and configuration correctness.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class StartupValidator:
    """Validate all systems before dashboard startup."""

    def __init__(self, app_root: Path | None = None):
        self.app_root = app_root or Path(__file__).resolve().parents[1]
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings: list[str] = []

    def validate_all(self) -> bool:
        """
        Run all startup checks.

        Returns
        -------
        bool
            True if all critical checks pass, False otherwise
        """
        log.info("🔍 Running startup validation checks...")

        # Critical checks (must pass)
        self._check_data_directory()
        self._check_config_file()
        self._check_log_directory()
        self._check_output_directory()

        # non-critical checks (warnings only)
        self._check_git_status()
        self._check_dependencies()
        self._check_disk_space()

        # Summary
        log.info("")
        log.info("=" * 70)
        log.info("Startup Validation Summary")
        log.info("=" * 70)
        log.info("✓ Checks passed:  %d", self.checks_passed)
        log.info("✗ Checks failed:  %d", self.checks_failed)
        log.info("⚠️  Warnings:      %d", len(self.warnings))

        if self.warnings:
            log.warning("Non-critical warnings detected:")
            for warning in self.warnings:
                log.warning("  • %s", warning)

        if self.checks_failed == 0:
            log.info("")
            log.info("✓ STARTUP VALIDATION PASSED - Dashboard is ready")
            log.info("=" * 70)
            return True
        else:
            log.error("")
            log.error("✗ STARTUP VALIDATION FAILED - Please fix errors above")
            log.error("=" * 70)
            return False

    def _check_data_directory(self) -> None:
        """Check if data directory exists and is writable."""
        data_dir = self.app_root / "data"
        if not data_dir.exists():
            log.error("✗ Data directory missing: %s", data_dir)
            self.checks_failed += 1
            return

        try:
            test_file = data_dir / ".startup_check"
            test_file.write_text("test")
            test_file.unlink()
            log.info("✓ Data directory is writable")
            self.checks_passed += 1
        except Exception as e:
            log.error("✗ Data directory is not writable: %s", e)
            self.checks_failed += 1

    def _check_config_file(self) -> None:
        """Check if configuration files exist."""
        config_file = self.app_root / "config" / "config.yaml"
        if not config_file.exists():
            log.error("✗ Configuration file missing: %s", config_file)
            self.checks_failed += 1
            return

        try:
            import yaml

            with open(config_file, encoding="utf-8") as f:
                config = yaml.safe_load(f)
            if config is None or not isinstance(config, dict):
                raise ValueError("Config is empty or invalid")
            log.info("✓ Configuration file is valid")
            self.checks_passed += 1
        except Exception as e:
            log.error("✗ Configuration file is invalid: %s", e)
            self.checks_failed += 1

    def _check_log_directory(self) -> None:
        """Check if logs directory exists and is writable."""
        log_dir = self.app_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        try:
            test_file = log_dir / ".startup_check"
            test_file.write_text("test")
            test_file.unlink()
            log.info("✓ Logs directory is writable")
            self.checks_passed += 1
        except Exception as e:
            log.error("✗ Logs directory is not writable: %s", e)
            self.checks_failed += 1

    def _check_output_directory(self) -> None:
        """Check if output directory exists and is writable."""
        output_dir = self.app_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            test_file = output_dir / ".startup_check"
            test_file.write_text("test")
            test_file.unlink()
            log.info("✓ Output directory is writable")
            self.checks_passed += 1
        except Exception as e:
            log.error("✗ Output directory is not writable: %s", e)
            self.checks_failed += 1

    def _check_git_status(self) -> None:
        """Check if code is in a clean git state (warning only)."""
        try:
            import subprocess

            result = subprocess.run(
                ["git", "-C", str(self.app_root), "status", "--porcelain"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                if result.stdout.strip():
                    self.warnings.append(
                        "Git repository has uncommitted changes (not a blocker, but note for reproducibility)"
                    )
                    log.warning("⚠️  Git repository has uncommitted changes")
                else:
                    log.info("✓ Git repository is clean")
                    self.checks_passed += 1
            else:
                self.warnings.append("Could not check git status (not in a git repo)")
        except FileNotFoundError:
            self.warnings.append("Git not installed (reproducibility tracking will be limited)")

    def _check_dependencies(self) -> None:
        """Check if critical Python dependencies are installed."""
        required_packages = [
            "streamlit",
            "numpy",
            "pandas",
            "scipy",
            "sklearn",   # pip: scikit-learn
            "yaml",      # pip: pyyaml
        ]

        missing = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)

        if missing:
            log.error("✗ Missing required packages: %s", ", ".join(missing))
            self.checks_failed += 1
        else:
            log.info("✓ All required Python packages installed")
            self.checks_passed += 1

    def _check_disk_space(self) -> None:
        """Check if sufficient disk space is available."""
        import shutil

        total, used, free = shutil.disk_usage(self.app_root)
        free_gb = free / (1024 ** 3)

        if free_gb < 1.0:
            log.error("✗ Low disk space: only %.1f GB available", free_gb)
            self.checks_failed += 1
        elif free_gb < 5.0:
            self.warnings.append(f"Low disk space: only {free_gb:.1f} GB available")
            log.warning("⚠️  Low disk space: %.1f GB available", free_gb)
        else:
            log.info("✓ Sufficient disk space: %.1f GB available", free_gb)
            self.checks_passed += 1


def run_startup_validation(app_root: Path | None = None) -> bool:
    """
    Convenience function to run startup validation.

    Parameters
    ----------
    app_root : Path
        Application root directory

    Returns
    -------
    bool
        True if all critical checks pass
    """
    validator = StartupValidator(app_root)
    return validator.validate_all()

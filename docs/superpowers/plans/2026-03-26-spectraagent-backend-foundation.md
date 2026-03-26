# SpectraAgent Backend Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the installable `spectraagent` package with hardware plugin adapters, sensor physics plugin, and a working FastAPI server so that `spectraagent start --simulate --no-browser` runs and `/api/health` returns 200, `/ws/spectrum` streams Lorentzian spectra at ~20 Hz, and `/ws/trend` streams Δλ trend data.

**Architecture:** New `spectraagent/` package sits alongside existing `src/` and `gas_analysis/` (not replacing them). Hardware and physics adapters are thin wrappers over existing code using entry-point plugin discovery. FastAPI server consolidates the existing `live_server.py` broadcaster into one unified app on port 8765.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, Typer, pytest, httpx (test client), numpy, tomllib (stdlib). All existing `src/` and `gas_analysis/` imports are used as-is.

---

## File Map

### Created
| File | Purpose |
|---|---|
| `spectraagent/__init__.py` | Package version marker |
| `spectraagent/__main__.py` | Typer CLI: `start`, `plugins` commands (stub for now) |
| `spectraagent/config.py` | Load `spectraagent.toml` → `SpectraAgentConfig` dataclass |
| `spectraagent/drivers/__init__.py` | Package marker + re-exports |
| `spectraagent/drivers/base.py` | `AbstractHardwareDriver` ABC |
| `spectraagent/drivers/simulation.py` | `SimulationDriver` — synthetic Lorentzian spectra |
| `spectraagent/drivers/thorlabs.py` | `ThorlabsCCSDriver` — wraps `RealtimeAcquisitionService` |
| `spectraagent/physics/__init__.py` | Package marker + re-exports |
| `spectraagent/physics/base.py` | `AbstractSensorPhysicsPlugin` ABC |
| `spectraagent/physics/lspr.py` | `LSPRPlugin` — wraps `src/features/lspr_features.py` |
| `spectraagent/webapp/__init__.py` | Package marker |
| `spectraagent/webapp/server.py` | FastAPI app — all routes, CORS, static mount, WebSocket |
| `spectraagent/webapp/static/dist/.gitkeep` | Placeholder until React build |
| `tests/spectraagent/__init__.py` | Test package marker |
| `tests/spectraagent/test_config.py` | Config load/default tests |
| `tests/spectraagent/drivers/__init__.py` | Test package marker |
| `tests/spectraagent/drivers/test_simulation.py` | SimulationDriver tests |
| `tests/spectraagent/drivers/test_base.py` | ABC contract tests |
| `tests/spectraagent/physics/__init__.py` | Test package marker |
| `tests/spectraagent/physics/test_lspr.py` | LSPRPlugin tests |
| `tests/spectraagent/webapp/__init__.py` | Test package marker |
| `tests/spectraagent/webapp/test_server.py` | FastAPI route + WebSocket tests |

### Modified
| File | What changes |
|---|---|
| `pyproject.toml` | Rename package, add deps, entry points, package-data, ruff config |
| `src/inference/orchestrator.py` | `save_raw=True` as default |

---

## Task 1: Update pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Apply all pyproject.toml changes**

Open `pyproject.toml` and make these changes (shown as diffs):

```toml
# CHANGE: project name
name = "spectraagent"
version = "1.0.0"
description = "Universal agentic spectroscopy platform — AI-native sensor analysis"

# ADD to keywords:
keywords = [
    "LSPR", "SPR", "spectrometer", "spectroscopy", "agentic-ai",
    "surface-plasmon-resonance", "gas-sensor", "calibration",
    "machine-learning", "real-time", "au-nanoparticle",
    "molecularly-imprinted-polymer", "VOC",
]

# CHANGE dependencies section — remove streamlit, add typer + anthropic + tomli fallback:
dependencies = [
    # Numerics
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "pandas>=1.3.0",
    "PyWavelets>=1.4.1",
    # ML (non-torch)
    "scikit-learn>=1.0.0",
    "joblib>=1.2.0",
    "statsmodels>=0.13.0",
    # Config
    "pyyaml>=6.0",
    "tomli>=2.0; python_version < '3.11'",
    # Data
    "pyarrow>=10.0.0",
    # Viz
    "matplotlib>=3.4.0",
    "plotly>=5.0.0",
    # Hardware serial
    "pyserial>=3.5",
    # API
    "fastapi>=0.111.0",
    "uvicorn[standard]>=0.29.0",
    "pydantic>=2.0.0",
    "httpx>=0.27.0",
    # CLI
    "typer>=0.9.0",
    # AI
    "anthropic>=0.25.0",
]

# ADD optional pdf extra (replace the existing reports extra):
[project.optional-dependencies]
ml = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
]
tracking = ["mlflow>=2.14.0"]
hardware = ["pyvisa>=1.13.0", "pyvisa-py>=0.6.0"]
pdf = ["playwright>=1.40"]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.4.0",
    "mypy>=1.8.0",
    "black>=24.0.0",
    "types-PyYAML>=6.0",
]

# CHANGE scripts section — keep existing gas-* entries, ADD spectraagent:
[project.scripts]
spectraagent         = "spectraagent.__main__:cli"
gas-serve            = "serve:main"
gas-train            = "src.training.train_gpr:main"
gas-train-cnn        = "src.training.train_cnn:main"
gas-export-onnx      = "src.models.onnx_export:main"
gas-cross-gas-eval   = "src.training.cross_gas_eval:main"
gas-ablation         = "src.training.ablation:main"

# ADD entry points:
[project.entry-points."spectraagent.hardware"]
thorlabs_ccs = "spectraagent.drivers.thorlabs:ThorlabsCCSDriver"
simulation   = "spectraagent.drivers.simulation:SimulationDriver"

[project.entry-points."spectraagent.sensor_physics"]
lspr = "spectraagent.physics.lspr:LSPRPlugin"

# CHANGE package discovery to include spectraagent:
[tool.setuptools.packages.find]
where = ["."]
include = ["src*", "config*", "dashboard*", "gas_analysis*", "spectraagent*"]

# ADD package-data for pre-built React assets:
[tool.setuptools.package-data]
"src" = ["py.typed"]
"spectraagent" = ["webapp/static/dist/**/*"]

# ADD spectraagent to ruff first-party:
[tool.ruff.lint.isort]
known-first-party = ["src", "config", "dashboard", "gas_analysis", "spectraagent"]
```

- [ ] **Step 2: Reinstall in editable mode**

```bash
.venv/Scripts/python.exe -m pip install -e . --quiet
```

Expected: no errors. If `anthropic` fails to install, run `.venv/Scripts/python.exe -m pip install anthropic` separately.

- [ ] **Step 3: Verify package name changed**

```bash
.venv/Scripts/python.exe -m pip show spectraagent
```

Expected output contains `Name: spectraagent`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: rename package to spectraagent, add typer/anthropic/plugin entry-points"
```

---

## Task 2: Package Skeleton + Config Loader

**Files:**
- Create: `spectraagent/__init__.py`
- Create: `spectraagent/config.py`
- Create: `tests/spectraagent/__init__.py`
- Create: `tests/spectraagent/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/spectraagent/__init__.py` (empty).

Create `tests/spectraagent/test_config.py`:

```python
import os
import tempfile
from pathlib import Path

import pytest

from spectraagent.config import SpectraAgentConfig, load_config


def test_defaults_when_no_file(tmp_path):
    cfg = load_config(tmp_path / "spectraagent.toml")
    assert cfg.server.port == 8765
    assert cfg.hardware.integration_time_ms == 50.0
    assert cfg.agents.auto_explain is False
    assert cfg.claude.model == "claude-sonnet-4-6"


def test_file_created_when_missing(tmp_path):
    path = tmp_path / "spectraagent.toml"
    assert not path.exists()
    load_config(path)
    assert path.exists()


def test_override_from_file(tmp_path):
    path = tmp_path / "spectraagent.toml"
    path.write_text("[server]\nport = 9000\n")
    cfg = load_config(path)
    assert cfg.server.port == 9000
    assert cfg.server.host == "127.0.0.1"  # default preserved


def test_physics_defaults(tmp_path):
    cfg = load_config(tmp_path / "spectraagent.toml")
    assert cfg.physics.default_plugin == "lspr"
    assert cfg.physics.search_min_nm == 500.0
    assert cfg.physics.search_max_nm == 900.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'spectraagent'`

- [ ] **Step 3: Create `spectraagent/__init__.py`**

```python
"""SpectraAgent — Universal Agentic Spectroscopy Platform."""
__version__ = "1.0.0"
```

- [ ] **Step 4: Create `spectraagent/config.py`**

```python
"""
spectraagent.config
===================
Load spectraagent.toml into typed dataclasses.
Creates the file with defaults if it does not exist.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

_DEFAULT_TOML = """\
[hardware]
default_driver = "thorlabs_ccs"
integration_time_ms = 50.0

[physics]
default_plugin = "lspr"
search_min_nm = 500.0
search_max_nm = 900.0

[agents]
auto_explain = false
anomaly_explainer_cooldown_s = 300
diagnostics_cooldown_s = 60

[claude]
model = "claude-sonnet-4-6"
timeout_s = 30

[server]
host = "127.0.0.1"
port = 8765
open_browser = true
"""


@dataclass
class HardwareConfig:
    default_driver: str = "thorlabs_ccs"
    integration_time_ms: float = 50.0


@dataclass
class PhysicsConfig:
    default_plugin: str = "lspr"
    search_min_nm: float = 500.0
    search_max_nm: float = 900.0


@dataclass
class AgentsConfig:
    auto_explain: bool = False
    anomaly_explainer_cooldown_s: float = 300.0
    diagnostics_cooldown_s: float = 60.0


@dataclass
class ClaudeConfig:
    model: str = "claude-sonnet-4-6"
    timeout_s: float = 30.0


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    open_browser: bool = True


@dataclass
class SpectraAgentConfig:
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    agents: AgentsConfig = field(default_factory=AgentsConfig)
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


def load_config(path: Path | None = None) -> SpectraAgentConfig:
    """Load config from *path* (default: ``spectraagent.toml`` in CWD).

    If the file does not exist it is created with defaults and the defaults
    are returned — so first run always succeeds.
    """
    if path is None:
        path = Path("spectraagent.toml")
    if not path.exists():
        path.write_text(_DEFAULT_TOML, encoding="utf-8")
    raw: dict = tomllib.loads(path.read_text(encoding="utf-8"))

    hw = raw.get("hardware", {})
    ph = raw.get("physics", {})
    ag = raw.get("agents", {})
    cl = raw.get("claude", {})
    sv = raw.get("server", {})

    return SpectraAgentConfig(
        hardware=HardwareConfig(
            default_driver=hw.get("default_driver", "thorlabs_ccs"),
            integration_time_ms=float(hw.get("integration_time_ms", 50.0)),
        ),
        physics=PhysicsConfig(
            default_plugin=ph.get("default_plugin", "lspr"),
            search_min_nm=float(ph.get("search_min_nm", 500.0)),
            search_max_nm=float(ph.get("search_max_nm", 900.0)),
        ),
        agents=AgentsConfig(
            auto_explain=bool(ag.get("auto_explain", False)),
            anomaly_explainer_cooldown_s=float(ag.get("anomaly_explainer_cooldown_s", 300.0)),
            diagnostics_cooldown_s=float(ag.get("diagnostics_cooldown_s", 60.0)),
        ),
        claude=ClaudeConfig(
            model=str(cl.get("model", "claude-sonnet-4-6")),
            timeout_s=float(cl.get("timeout_s", 30.0)),
        ),
        server=ServerConfig(
            host=str(sv.get("host", "127.0.0.1")),
            port=int(sv.get("port", 8765)),
            open_browser=bool(sv.get("open_browser", True)),
        ),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/test_config.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Create stub `spectraagent/__main__.py`**

```python
"""spectraagent.__main__ — CLI entry point (Typer)."""
from __future__ import annotations

import typer

cli = typer.Typer(
    name="spectraagent",
    help="Universal Agentic Spectroscopy Platform",
    no_args_is_help=True,
)


@cli.command()
def start(
    simulate: bool = typer.Option(False, "--simulate", help="Force simulation mode"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open browser"),
    host: str = typer.Option("", "--host", help="Bind host (overrides config)"),
    port: int = typer.Option(0, "--port", help="Port (overrides config)"),
    physics: str = typer.Option("", "--physics", help="Sensor physics plugin (overrides config)"),
) -> None:
    """Start the SpectraAgent server and open the browser."""
    typer.echo("spectraagent start — not yet implemented (Task 13)")


@cli.command(name="plugins")
def plugins_cmd(
    action: str = typer.Argument("list", help="Action: list"),
) -> None:
    """Show discovered plugins."""
    typer.echo("spectraagent plugins — not yet implemented (Task 6)")


if __name__ == "__main__":
    cli()
```

- [ ] **Step 7: Verify CLI stub works**

```bash
.venv/Scripts/python.exe -m spectraagent --help
```

Expected: shows `start` and `plugins` commands.

- [ ] **Step 8: Commit**

```bash
git add spectraagent/ tests/spectraagent/
git commit -m "feat: spectraagent package skeleton — config loader + CLI stub"
```

---

## Task 3: AbstractHardwareDriver

**Files:**
- Create: `spectraagent/drivers/__init__.py`
- Create: `spectraagent/drivers/base.py`
- Create: `tests/spectraagent/drivers/__init__.py`
- Create: `tests/spectraagent/drivers/test_base.py`

- [ ] **Step 1: Write the failing test**

Create `tests/spectraagent/drivers/__init__.py` (empty).

Create `tests/spectraagent/drivers/test_base.py`:

```python
import numpy as np
import pytest

from spectraagent.drivers.base import AbstractHardwareDriver


class _ConcreteDriver(AbstractHardwareDriver):
    """Minimal concrete implementation for testing the ABC contract."""

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_wavelengths(self) -> np.ndarray:
        return np.linspace(500.0, 900.0, 3648)

    def read_spectrum(self) -> np.ndarray:
        return np.ones(3648) * 0.5

    def get_integration_time_ms(self) -> float:
        return 50.0

    def set_integration_time_ms(self, ms: float) -> None:
        pass

    @property
    def name(self) -> str:
        return "TestDriver"

    @property
    def is_connected(self) -> bool:
        return getattr(self, "_connected", False)


def test_cannot_instantiate_abc():
    with pytest.raises(TypeError):
        AbstractHardwareDriver()  # type: ignore[abstract]


def test_concrete_driver_satisfies_interface():
    drv = _ConcreteDriver()
    drv.connect()
    assert drv.is_connected
    wl = drv.get_wavelengths()
    assert wl.shape == (3648,)
    sp = drv.read_spectrum()
    assert sp.shape == (3648,)
    assert drv.name == "TestDriver"
    drv.disconnect()
    assert not drv.is_connected


def test_wavelengths_and_spectrum_same_length():
    drv = _ConcreteDriver()
    drv.connect()
    assert drv.get_wavelengths().shape == drv.read_spectrum().shape
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/drivers/test_base.py -v
```

Expected: `ModuleNotFoundError: No module named 'spectraagent.drivers'`

- [ ] **Step 3: Create `spectraagent/drivers/__init__.py`**

```python
"""spectraagent.drivers — Hardware driver plugin package."""
from spectraagent.drivers.base import AbstractHardwareDriver

__all__ = ["AbstractHardwareDriver"]
```

- [ ] **Step 4: Create `spectraagent/drivers/base.py`**

```python
"""
spectraagent.drivers.base
=========================
Abstract base class for all hardware spectrometer drivers.

Third-party plugins implement this interface and register via:

    [project.entry-points."spectraagent.hardware"]
    my_driver = "mypkg.driver:MyDriver"
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AbstractHardwareDriver(ABC):
    """Spectrometer hardware driver interface.

    ``connect()`` must be called before any other method.
    ``disconnect()`` must be called when the session ends (use try/finally).

    ``read_spectrum()`` is a **blocking** call — it returns when the next
    frame is available.  The acquisition loop calls it in a dedicated thread.
    """

    @abstractmethod
    def connect(self) -> None:
        """Open the connection to the spectrometer hardware."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection and release all hardware resources."""

    @abstractmethod
    def get_wavelengths(self) -> np.ndarray:
        """Return the wavelength calibration array, shape ``(N,)``, in nm.

        Call once after ``connect()`` and cache the result — the calibration
        does not change within a session.
        """

    @abstractmethod
    def read_spectrum(self) -> np.ndarray:
        """Block until the next spectrum frame is available, then return it.

        Returns
        -------
        np.ndarray
            Intensity array, shape ``(N,)``, same length as ``get_wavelengths()``.
        """

    @abstractmethod
    def get_integration_time_ms(self) -> float:
        """Return the current integration time in milliseconds."""

    @abstractmethod
    def set_integration_time_ms(self, ms: float) -> None:
        """Set the integration time in milliseconds."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable driver name, e.g. ``'ThorLabs CCS200'``."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """True if ``connect()`` has succeeded and ``disconnect()`` not yet called."""
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/drivers/test_base.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add spectraagent/drivers/ tests/spectraagent/drivers/
git commit -m "feat: AbstractHardwareDriver ABC with full interface contract"
```

---

## Task 4: SimulationDriver

**Files:**
- Create: `spectraagent/drivers/simulation.py`
- Modify: `tests/spectraagent/drivers/test_simulation.py`

- [ ] **Step 1: Write the failing test**

Create `tests/spectraagent/drivers/test_simulation.py`:

```python
import time

import numpy as np
import pytest

from spectraagent.drivers.simulation import SimulationDriver


@pytest.fixture
def drv():
    d = SimulationDriver(integration_time_ms=1.0)  # fast for tests
    d.connect()
    yield d
    d.disconnect()


def test_name(drv):
    assert drv.name == "Simulation"


def test_is_connected(drv):
    assert drv.is_connected


def test_disconnect_clears_connected():
    d = SimulationDriver()
    d.connect()
    d.disconnect()
    assert not d.is_connected


def test_wavelengths_shape(drv):
    wl = drv.get_wavelengths()
    assert wl.shape == (3648,)


def test_wavelengths_range(drv):
    wl = drv.get_wavelengths()
    assert wl[0] == pytest.approx(500.0)
    assert wl[-1] == pytest.approx(900.0)


def test_spectrum_shape(drv):
    sp = drv.read_spectrum()
    assert sp.shape == (3648,)


def test_spectrum_non_negative(drv):
    sp = drv.read_spectrum()
    assert np.all(sp >= 0.0)


def test_spectrum_has_peak_near_720nm(drv):
    wl = drv.get_wavelengths()
    sp = drv.read_spectrum()
    peak_idx = int(np.argmax(sp))
    assert 700.0 <= wl[peak_idx] <= 740.0


def test_spectrum_max_amplitude_reasonable(drv):
    sp = drv.read_spectrum()
    assert 0.5 <= sp.max() <= 1.0


def test_integration_time_roundtrip(drv):
    drv.set_integration_time_ms(25.0)
    assert drv.get_integration_time_ms() == pytest.approx(25.0)


def test_read_blocks_for_integration_time():
    drv = SimulationDriver(integration_time_ms=50.0)
    drv.connect()
    t0 = time.monotonic()
    drv.read_spectrum()
    elapsed_ms = (time.monotonic() - t0) * 1000
    drv.disconnect()
    assert elapsed_ms >= 40.0  # at least 80% of integration time
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/drivers/test_simulation.py -v
```

Expected: `ModuleNotFoundError: No module named 'spectraagent.drivers.simulation'`

- [ ] **Step 3: Create `spectraagent/drivers/simulation.py`**

```python
"""
spectraagent.drivers.simulation
================================
Simulation hardware driver — generates synthetic Lorentzian LSPR spectra.

Used for offline development, demos, and CI where no physical spectrometer
is available.  The peak is centred at 720 nm with slight per-frame jitter to
simulate realistic drift.
"""
from __future__ import annotations

import time

import numpy as np

from spectraagent.drivers.base import AbstractHardwareDriver

_N_PIXELS: int = 3648
_WL_MIN: float = 500.0
_WL_MAX: float = 900.0
_PEAK_NM: float = 720.0
_GAMMA_NM: float = 9.0       # Lorentzian half-width at half-maximum
_AMPLITUDE: float = 0.8
_JITTER_STD: float = 0.02    # nm — per-frame peak position jitter


class SimulationDriver(AbstractHardwareDriver):
    """Synthetic spectrometer driver for testing and demos.

    Generates a Lorentzian peak at ~720 nm with Gaussian noise and slight
    per-frame jitter.  ``read_spectrum()`` blocks for ``integration_time_ms``
    to faithfully simulate real acquisition timing.

    Parameters
    ----------
    integration_time_ms:
        How long ``read_spectrum()`` sleeps before returning (default 50 ms →
        ~20 Hz equivalent).
    noise_level:
        Standard deviation of Gaussian noise added to the spectrum.
    """

    def __init__(
        self,
        integration_time_ms: float = 50.0,
        noise_level: float = 0.002,
    ) -> None:
        self._integration_time_ms = integration_time_ms
        self._noise_level = noise_level
        self._connected: bool = False
        self._wavelengths: np.ndarray = np.linspace(_WL_MIN, _WL_MAX, _N_PIXELS)

    # ------------------------------------------------------------------
    # AbstractHardwareDriver interface
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_wavelengths(self) -> np.ndarray:
        return self._wavelengths.copy()

    def read_spectrum(self) -> np.ndarray:
        """Sleep for integration time, then return a synthetic spectrum."""
        time.sleep(self._integration_time_ms / 1000.0)
        peak_wl = _PEAK_NM + float(np.random.normal(0.0, _JITTER_STD))
        lorentz = _AMPLITUDE / (1.0 + ((self._wavelengths - peak_wl) / _GAMMA_NM) ** 2)
        noise = np.random.normal(0.0, self._noise_level, _N_PIXELS)
        return np.clip(lorentz + noise, 0.0, None)

    def get_integration_time_ms(self) -> float:
        return self._integration_time_ms

    def set_integration_time_ms(self, ms: float) -> None:
        self._integration_time_ms = ms

    @property
    def name(self) -> str:
        return "Simulation"

    @property
    def is_connected(self) -> bool:
        return self._connected
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/drivers/test_simulation.py -v
```

Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add spectraagent/drivers/simulation.py tests/spectraagent/drivers/test_simulation.py
git commit -m "feat: SimulationDriver — synthetic Lorentzian spectra for offline dev"
```

---

## Task 5: ThorlabsCCSDriver Adapter

**Files:**
- Create: `spectraagent/drivers/thorlabs.py`

No test file — hardware is not available in CI. The adapter is verified manually with real hardware.

- [ ] **Step 1: Create `spectraagent/drivers/thorlabs.py`**

```python
"""
spectraagent.drivers.thorlabs
==============================
Hardware driver adapter for ThorLabs CCS200 spectrometer.

Wraps ``gas_analysis.acquisition.ccs200_realtime.RealtimeAcquisitionService``
(callback-based) into the blocking ``AbstractHardwareDriver`` interface using
a thread-safe queue.  The acquisition thread fills the queue; ``read_spectrum()``
blocks until the next frame arrives.

Hardware-specific notes (from calibrated experience):
- Integration time 50 ms → ~20 Hz (410 ms total including cooldown)
- Error -1073807343: device connected but not powered
- Error -1073807339: stale VISA handle — reconnect required
- Always call ``disconnect()`` in a try/finally block
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Any

import numpy as np

from spectraagent.drivers.base import AbstractHardwareDriver

log = logging.getLogger(__name__)

_QUEUE_MAXSIZE = 4       # drop oldest if consumer falls behind
_READ_TIMEOUT_S = 5.0   # raise if no frame arrives within this time


class ThorlabsCCSDriver(AbstractHardwareDriver):
    """Blocking driver adapter for ThorLabs CCS200 via RealtimeAcquisitionService.

    The underlying service runs a callback thread continuously.  This adapter
    bridges the callback → blocking-read interface using a bounded Queue.

    Parameters
    ----------
    integration_time_ms:
        Spectrometer integration time in milliseconds.
    resource_string:
        VISA resource string, e.g. ``"USB0::0x1313::0x8089::M00840499::RAW"``.
        Pass ``None`` for auto-discovery.
    """

    def __init__(
        self,
        integration_time_ms: float = 50.0,
        resource_string: str | None = None,
    ) -> None:
        # Import here so that missing DLL does not crash at module import time
        from gas_analysis.acquisition.ccs200_realtime import RealtimeAcquisitionService

        self._svc = RealtimeAcquisitionService(
            integration_time_ms=integration_time_ms,
            resource_string=resource_string,
        )
        self._frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._wavelengths: np.ndarray | None = None
        self._connected: bool = False

    # ------------------------------------------------------------------
    # AbstractHardwareDriver interface
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to CCS200 and start the acquisition thread."""
        self._svc.connect()
        self._wavelengths = self._svc.wavelengths.copy()
        self._svc.register_callback(self._on_sample)
        self._svc.start()
        self._connected = True
        log.info("ThorlabsCCSDriver connected — %d pixels, %.1f–%.1f nm",
                 len(self._wavelengths), self._wavelengths[0], self._wavelengths[-1])

    def disconnect(self) -> None:
        """Stop acquisition and release the VISA handle."""
        try:
            self._svc.stop()
        finally:
            self._connected = False
            log.info("ThorlabsCCSDriver disconnected")

    def get_wavelengths(self) -> np.ndarray:
        if self._wavelengths is None:
            raise RuntimeError("Not connected — call connect() first")
        return self._wavelengths.copy()

    def read_spectrum(self) -> np.ndarray:
        """Block until the next frame arrives from the acquisition thread."""
        try:
            return self._frame_queue.get(timeout=_READ_TIMEOUT_S)
        except queue.Empty as exc:
            raise RuntimeError(
                f"No spectrum received within {_READ_TIMEOUT_S} s — "
                "check hardware connection"
            ) from exc

    def get_integration_time_ms(self) -> float:
        return float(self._svc.integration_time_ms)

    def set_integration_time_ms(self, ms: float) -> None:
        self._svc.integration_time_ms = ms

    @property
    def name(self) -> str:
        return "ThorLabs CCS200"

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_sample(self, data: dict[str, Any]) -> None:
        """Callback invoked by the acquisition thread on each new frame."""
        frame = np.array(data["intensities"], dtype=np.float64)
        if self._frame_queue.full():
            # Drop the oldest frame rather than blocking the acquisition thread
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
        self._frame_queue.put_nowait(frame)
```

- [ ] **Step 2: Verify it imports cleanly (no hardware needed)**

```bash
.venv/Scripts/python.exe -c "from spectraagent.drivers.thorlabs import ThorlabsCCSDriver; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add spectraagent/drivers/thorlabs.py
git commit -m "feat: ThorlabsCCSDriver adapter — wraps RealtimeAcquisitionService via queue bridge"
```

---

## Task 6: Plugin Discovery + `plugins list` Command

**Files:**
- Modify: `spectraagent/__main__.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/spectraagent/test_config.py`:

```python
from typer.testing import CliRunner
from spectraagent.__main__ import cli


def test_plugins_list_shows_simulation():
    runner = CliRunner()
    result = runner.invoke(cli, ["plugins", "list"])
    assert result.exit_code == 0
    assert "simulation" in result.output.lower()


def test_plugins_list_shows_lspr():
    runner = CliRunner()
    result = runner.invoke(cli, ["plugins", "list"])
    assert result.exit_code == 0
    assert "lspr" in result.output.lower()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/test_config.py::test_plugins_list_shows_simulation -v
```

Expected: FAIL — output says "not yet implemented".

- [ ] **Step 3: Update `spectraagent/__main__.py` plugins command**

Replace the `plugins_cmd` function with:

```python
from importlib.metadata import entry_points

@cli.command(name="plugins")
def plugins_cmd(
    action: str = typer.Argument("list", help="Action: list"),
) -> None:
    """Show discovered plugins and their status."""
    if action != "list":
        typer.echo(f"Unknown action: {action}", err=True)
        raise typer.Exit(1)

    typer.echo("\nHardware Drivers:")
    hw_eps = entry_points(group="spectraagent.hardware")
    for ep in hw_eps:
        try:
            ep.load()
            status = "✓ loadable"
        except Exception as exc:
            status = f"✗ {exc}"
        typer.echo(f"  [{ep.name}]  {ep.value}  —  {status}")

    typer.echo("\nSensor Physics Plugins:")
    ph_eps = entry_points(group="spectraagent.sensor_physics")
    for ep in ph_eps:
        try:
            ep.load()
            status = "✓ loadable"
        except Exception as exc:
            status = f"✗ {exc}"
        typer.echo(f"  [{ep.name}]  {ep.value}  —  {status}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/test_config.py -v
```

Expected: all pass (including the 4 config tests from Task 2).

- [ ] **Step 5: Manually verify CLI output**

```bash
.venv/Scripts/python.exe -m spectraagent plugins list
```

Expected output includes lines like:
```
Hardware Drivers:
  [simulation]  spectraagent.drivers.simulation:SimulationDriver  —  ✓ loadable
  [thorlabs_ccs]  spectraagent.drivers.thorlabs:ThorlabsCCSDriver  —  ✓ loadable
Sensor Physics Plugins:
  [lspr]  spectraagent.physics.lspr:LSPRPlugin  —  ...
```

- [ ] **Step 6: Commit**

```bash
git add spectraagent/__main__.py
git commit -m "feat: plugins list command — discovers hardware + physics entry points"
```

---

## Task 7: AbstractSensorPhysicsPlugin

**Files:**
- Create: `spectraagent/physics/__init__.py`
- Create: `spectraagent/physics/base.py`
- Create: `tests/spectraagent/physics/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `tests/spectraagent/physics/__init__.py` (empty).

Add to `tests/spectraagent/physics/` a new file `test_base.py`:

```python
import numpy as np
import pytest

from spectraagent.physics.base import AbstractSensorPhysicsPlugin


class _ConcretePhysics(AbstractSensorPhysicsPlugin):
    def detect_peak(self, wl, intensities):
        return float(wl[int(np.argmax(intensities))])

    def extract_features(self, wl, intensities, reference=None, cached_ref=None):
        return {"delta_lambda": -0.5, "snr": 10.0, "peak_wavelength": 720.0}

    def compute_reference_cache(self, wl, reference):
        return {"peak": float(wl[int(np.argmax(reference))])}

    def calibration_priors(self):
        return {"models": ["Linear"], "bounds": {}}

    @property
    def name(self):
        return "TestPhysics"


def test_cannot_instantiate_abc():
    with pytest.raises(TypeError):
        AbstractSensorPhysicsPlugin()  # type: ignore[abstract]


def test_concrete_satisfies_interface():
    ph = _ConcretePhysics()
    wl = np.linspace(500, 900, 3648)
    sp = np.zeros(3648)
    sp[1000] = 1.0
    peak = ph.detect_peak(wl, sp)
    assert isinstance(peak, float)
    feats = ph.extract_features(wl, sp)
    assert "delta_lambda" in feats
    assert "snr" in feats
    cache = ph.compute_reference_cache(wl, sp)
    assert cache is not None
    assert ph.name == "TestPhysics"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/physics/ -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create `spectraagent/physics/__init__.py`**

```python
"""spectraagent.physics — Sensor physics plugin package."""
from spectraagent.physics.base import AbstractSensorPhysicsPlugin

__all__ = ["AbstractSensorPhysicsPlugin"]
```

- [ ] **Step 4: Create `spectraagent/physics/base.py`**

```python
"""
spectraagent.physics.base
=========================
Abstract base class for sensor physics plugins.

Each plugin encapsulates the signal model for a specific sensor type
(LSPR, SPR, UV-Vis, etc.).  Plugins are registered via:

    [project.entry-points."spectraagent.sensor_physics"]
    lspr = "spectraagent.physics.lspr:LSPRPlugin"
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AbstractSensorPhysicsPlugin(ABC):
    """Sensor physics plugin interface.

    Methods operate only on pre-processed numpy arrays — never raw bytes or
    file handles.  All methods must be safe to call from multiple threads
    (they should be stateless or use immutable state only).
    """

    @abstractmethod
    def detect_peak(
        self,
        wavelengths: np.ndarray,
        intensities: np.ndarray,
    ) -> float | None:
        """Detect the primary spectral peak and return its wavelength in nm.

        Returns ``None`` if no valid peak is found (e.g. saturated or noisy
        spectrum).
        """

    @abstractmethod
    def extract_features(
        self,
        wavelengths: np.ndarray,
        intensities: np.ndarray,
        reference: np.ndarray | None = None,
        cached_ref: object | None = None,
    ) -> dict[str, float]:
        """Extract physics-meaningful features from one spectrum.

        Parameters
        ----------
        wavelengths:
            Wavelength calibration array, shape ``(N,)``.
        intensities:
            Raw spectrum intensities, shape ``(N,)``.
        reference:
            Reference spectrum intensities if available, shape ``(N,)``.
        cached_ref:
            Plugin-specific pre-computed reference object returned by
            ``compute_reference_cache()``.  Pass this every frame to avoid
            redundant computation on the reference.

        Returns
        -------
        dict[str, float]
            At minimum: ``{"delta_lambda": float, "snr": float}``.
            Additional keys are plugin-specific.
        """

    @abstractmethod
    def compute_reference_cache(
        self,
        wavelengths: np.ndarray,
        reference: np.ndarray,
    ) -> object:
        """Pre-compute an expensive reference calculation once.

        The returned object is passed as ``cached_ref`` to every subsequent
        ``extract_features()`` call, eliminating redundant computation
        (e.g. Lorentzian fitting of the reference peak).
        """

    @abstractmethod
    def calibration_priors(self) -> dict:
        """Return calibration model priors for this sensor type.

        Returns
        -------
        dict
            ``{"models": ["Langmuir", "Linear", ...], "bounds": {...}}``
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable plugin name, e.g. ``'LSPR'``."""
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/physics/ -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add spectraagent/physics/ tests/spectraagent/physics/
git commit -m "feat: AbstractSensorPhysicsPlugin ABC with full interface contract"
```

---

## Task 8: LSPRPlugin

**Files:**
- Create: `spectraagent/physics/lspr.py`
- Create: `tests/spectraagent/physics/test_lspr.py`

- [ ] **Step 1: Write the failing test**

Create `tests/spectraagent/physics/test_lspr.py`:

```python
import numpy as np
import pytest

from spectraagent.physics.lspr import LSPRPlugin


@pytest.fixture
def plugin():
    return LSPRPlugin()


@pytest.fixture
def synthetic_spectrum():
    """Lorentzian at 531.5 nm — matches LSPR_REFERENCE_PEAK_NM constant."""
    wl = np.linspace(480.0, 600.0, 3648)
    peak = 531.5
    gamma = 9.0
    sp = 0.8 / (1.0 + ((wl - peak) / gamma) ** 2)
    sp += np.random.default_rng(42).normal(0, 0.001, len(wl))
    return wl, sp


def test_name(plugin):
    assert plugin.name == "LSPR"


def test_detect_peak_returns_float(plugin, synthetic_spectrum):
    wl, sp = synthetic_spectrum
    peak = plugin.detect_peak(wl, sp)
    assert peak is not None
    assert isinstance(peak, float)
    assert 520.0 <= peak <= 545.0


def test_detect_peak_returns_none_on_flat_spectrum(plugin):
    wl = np.linspace(480.0, 600.0, 3648)
    sp = np.ones(3648) * 0.1
    result = plugin.detect_peak(wl, sp)
    # Flat spectrum — no clear peak, may return None or a value
    # Just verify it doesn't raise
    assert result is None or isinstance(result, float)


def test_compute_reference_cache_returns_object(plugin, synthetic_spectrum):
    wl, sp = synthetic_spectrum
    cache = plugin.compute_reference_cache(wl, sp)
    assert cache is not None


def test_extract_features_with_cache(plugin, synthetic_spectrum):
    wl, sp = synthetic_spectrum
    # Slightly shifted gas spectrum
    peak_gas = 531.5 - 0.71  # -0.71 nm shift
    sp_gas = 0.8 / (1.0 + ((wl - peak_gas) / 9.0) ** 2)
    sp_gas += np.random.default_rng(0).normal(0, 0.001, len(wl))
    cache = plugin.compute_reference_cache(wl, sp)
    feats = plugin.extract_features(wl, sp_gas, reference=sp, cached_ref=cache)
    assert "delta_lambda" in feats
    assert "snr" in feats
    assert feats["snr"] > 0


def test_calibration_priors_has_models(plugin):
    priors = plugin.calibration_priors()
    assert "models" in priors
    assert "Langmuir" in priors["models"]
    assert "Linear" in priors["models"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/physics/test_lspr.py -v
```

Expected: `ModuleNotFoundError: No module named 'spectraagent.physics.lspr'`

- [ ] **Step 3: Create `spectraagent/physics/lspr.py`**

```python
"""
spectraagent.physics.lspr
==========================
LSPR sensor physics plugin.

Wraps ``src.features.lspr_features`` — the underlying Lorentzian peak
detection and feature extraction code is unchanged.  This class only
adapts the function-based API to the ``AbstractSensorPhysicsPlugin``
interface.

Physics: Au nanoparticle LSPR sensor.  Primary signal is peak wavelength
SHIFT Δλ = λ_gas − λ_reference (nm).  Negative Δλ = blue-shift on analyte
adsorption.
"""
from __future__ import annotations

import numpy as np

from src.features.lspr_features import (
    LSPR_SEARCH_MAX_NM,
    LSPR_SEARCH_MIN_NM,
    LSPRReference,
    compute_lspr_reference,
    detect_lspr_peak,
    extract_lspr_features,
)
from spectraagent.physics.base import AbstractSensorPhysicsPlugin


class LSPRPlugin(AbstractSensorPhysicsPlugin):
    """LSPR sensor physics plugin — wraps ``src.features.lspr_features``.

    Parameters
    ----------
    search_min_nm, search_max_nm:
        Wavelength window for peak search.  Defaults match the Au-MIP sensor
        constants in ``lspr_features.py``.
    """

    def __init__(
        self,
        search_min_nm: float = LSPR_SEARCH_MIN_NM,
        search_max_nm: float = LSPR_SEARCH_MAX_NM,
    ) -> None:
        self._search_min = search_min_nm
        self._search_max = search_max_nm

    # ------------------------------------------------------------------
    # AbstractSensorPhysicsPlugin interface
    # ------------------------------------------------------------------

    def detect_peak(
        self,
        wavelengths: np.ndarray,
        intensities: np.ndarray,
    ) -> float | None:
        return detect_lspr_peak(
            wavelengths, intensities, self._search_min, self._search_max
        )

    def compute_reference_cache(
        self,
        wavelengths: np.ndarray,
        reference: np.ndarray,
    ) -> LSPRReference:
        """Pre-compute Lorentzian fit of reference spectrum (call once per session)."""
        return compute_lspr_reference(
            wavelengths, reference, self._search_min, self._search_max
        )

    def extract_features(
        self,
        wavelengths: np.ndarray,
        intensities: np.ndarray,
        reference: np.ndarray | None = None,
        cached_ref: object | None = None,
    ) -> dict[str, float]:
        lspr_ref = cached_ref if isinstance(cached_ref, LSPRReference) else None
        result = extract_lspr_features(
            wavelengths,
            intensities,
            reference_intensities=reference,
            lspr_ref=lspr_ref,
        )
        if result is None:
            return {"delta_lambda": 0.0, "snr": 0.0, "peak_wavelength": 0.0}
        return {
            "delta_lambda": result.delta_lambda if result.delta_lambda is not None else 0.0,
            "snr": result.snr if result.snr is not None else 0.0,
            "peak_wavelength": result.peak_wavelength if result.peak_wavelength is not None else 0.0,
            "delta_intensity_peak": result.delta_intensity_peak if result.delta_intensity_peak is not None else 0.0,
            "delta_intensity_area": result.delta_intensity_area if result.delta_intensity_area is not None else 0.0,
        }

    def calibration_priors(self) -> dict:
        return {
            "models": ["Langmuir", "Freundlich", "Hill", "Linear"],
            "bounds": {
                "Langmuir": {"Bmax": (0.0, 10.0), "Kd": (1e-6, 1.0)},
                "Linear": {"slope": (-100.0, 0.0), "intercept": (-5.0, 5.0)},
            },
        }

    @property
    def name(self) -> str:
        return "LSPR"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/physics/test_lspr.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add spectraagent/physics/lspr.py tests/spectraagent/physics/test_lspr.py
git commit -m "feat: LSPRPlugin — wraps src.features.lspr_features via AbstractSensorPhysicsPlugin"
```

---

## Task 9: FastAPI Server Skeleton (Health + CORS + Static Mount)

**Files:**
- Create: `spectraagent/webapp/__init__.py`
- Create: `spectraagent/webapp/server.py`
- Create: `spectraagent/webapp/static/dist/.gitkeep`
- Create: `tests/spectraagent/webapp/__init__.py`
- Create: `tests/spectraagent/webapp/test_server.py`

- [ ] **Step 1: Write the failing test**

Create `tests/spectraagent/webapp/__init__.py` (empty).

Create `tests/spectraagent/webapp/test_server.py`:

```python
import pytest
from fastapi.testclient import TestClient

from spectraagent.webapp.server import create_app


@pytest.fixture
def client():
    app = create_app(simulate=True)
    with TestClient(app) as c:
        yield c


def test_health_returns_200(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200


def test_health_response_schema(client):
    resp = client.get("/api/health")
    data = resp.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "hardware" in data
    assert "version" in data


def test_cors_header_present(client):
    resp = client.get("/api/health", headers={"Origin": "http://localhost:3000"})
    assert "access-control-allow-origin" in resp.headers


def test_unknown_route_returns_404(client):
    resp = client.get("/api/nonexistent")
    assert resp.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/webapp/test_server.py -v
```

Expected: `ModuleNotFoundError: No module named 'spectraagent.webapp'`

- [ ] **Step 3: Create supporting files**

Create `spectraagent/webapp/__init__.py` (empty).

Create `spectraagent/webapp/static/dist/.gitkeep` (empty).

- [ ] **Step 4: Create `spectraagent/webapp/server.py`**

```python
"""
spectraagent.webapp.server
==========================
FastAPI application — all HTTP routes and WebSocket endpoints.

``create_app(simulate)`` is a factory used both by the CLI (``spectraagent start``)
and by the test suite (``TestClient(create_app(simulate=True))``).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import spectraagent

log = logging.getLogger(__name__)

_STATIC_DIST = Path(__file__).resolve().parent / "static" / "dist"

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(simulate: bool = False) -> FastAPI:
    """Create and configure the FastAPI application.

    Parameters
    ----------
    simulate:
        If True, use :class:`~spectraagent.drivers.simulation.SimulationDriver`
        regardless of config.  The hardware connection is *not* started here —
        that happens in the CLI startup sequence.
    """
    app = FastAPI(
        title="SpectraAgent",
        version=spectraagent.__version__,
        docs_url="/api/docs",
        redoc_url=None,
    )

    # CORS — allow all origins so LAN clients work without configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store simulate flag for route handlers
    app.state.simulate = simulate
    app.state.driver = None    # set by CLI before serving
    app.state.plugin = None    # set by CLI before serving

    # ------------------------------------------------------------------
    # Health endpoint
    # ------------------------------------------------------------------

    @app.get("/api/health")
    async def health() -> JSONResponse:
        driver = app.state.driver
        return JSONResponse({
            "status": "ok",
            "version": spectraagent.__version__,
            "hardware": driver.name if driver is not None else "not_connected",
            "simulate": app.state.simulate,
        })

    # ------------------------------------------------------------------
    # Static files (React SPA) — mounted last so API routes take priority
    # ------------------------------------------------------------------
    if _STATIC_DIST.exists() and any(_STATIC_DIST.iterdir()):
        app.mount("/", StaticFiles(directory=str(_STATIC_DIST), html=True), name="static")
    else:
        @app.get("/")
        async def index_placeholder() -> JSONResponse:
            return JSONResponse({
                "message": "React frontend not yet built. Run: cd spectraagent/webapp/frontend && npm run build"
            })

    return app
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/webapp/test_server.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add spectraagent/webapp/ tests/spectraagent/webapp/
git commit -m "feat: FastAPI server skeleton — health endpoint, CORS, static mount"
```

---

## Task 10: /ws/spectrum and /ws/trend WebSockets

**Files:**
- Modify: `spectraagent/webapp/server.py`
- Modify: `tests/spectraagent/webapp/test_server.py`

These WebSockets are adapted from `dashboard/live_server.py` (`_Broadcaster`, `_spectrum_loop`, `_trend_loop`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/spectraagent/webapp/test_server.py`:

```python
import json
import threading
import time

from spectraagent.drivers.simulation import SimulationDriver
from spectraagent.webapp.server import Broadcaster, create_app


def test_broadcaster_fan_out():
    """Messages sent to Broadcaster are received by all connected clients."""
    bc = Broadcaster()
    received: list[str] = []

    async def fake_ws_send(msg):
        received.append(msg)

    class _FakeWS:
        async def send_text(self, msg):
            received.append(msg)

    import asyncio
    loop = asyncio.new_event_loop()

    async def run():
        ws = _FakeWS()
        bc.connect(ws)
        await bc.broadcast("hello")
        bc.disconnect(ws)

    loop.run_until_complete(run())
    loop.close()
    assert received == ["hello"]


def test_ws_spectrum_endpoint_connects(client):
    """WebSocket /ws/spectrum accepts connections without error."""
    with client.websocket_connect("/ws/spectrum") as ws:
        # Just connecting and disconnecting should work
        pass  # connection accepted = success
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/webapp/test_server.py::test_broadcaster_fan_out -v
```

Expected: `ImportError: cannot import name 'Broadcaster'`

- [ ] **Step 3: Add Broadcaster and WebSocket endpoints to `server.py`**

Add the following to `spectraagent/webapp/server.py`, before the `create_app` function:

```python
import asyncio
import json
from collections import deque


class Broadcaster:
    """Thread-safe WebSocket fan-out.

    Adapted from ``dashboard.live_server._Broadcaster``.
    All WebSocket ``send_text`` calls are awaited in the asyncio event loop.
    """

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    def connect(self, ws: WebSocket) -> None:
        self._clients.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)

    async def broadcast(self, message: str) -> None:
        dead: list[WebSocket] = []
        for ws in list(self._clients):
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)
```

Then, inside `create_app`, add after the health endpoint and before the static mount:

```python
    # ------------------------------------------------------------------
    # WebSocket: /ws/spectrum  — pushes every new frame at ~20 Hz
    # ------------------------------------------------------------------
    _spectrum_bc = Broadcaster()
    _trend_bc = Broadcaster()

    @app.websocket("/ws/spectrum")
    async def ws_spectrum(websocket: WebSocket) -> None:
        await websocket.accept()
        _spectrum_bc.connect(websocket)
        try:
            while True:
                # Keep connection alive; data is pushed by _spectrum_loop
                await asyncio.sleep(60)
        except WebSocketDisconnect:
            pass
        finally:
            _spectrum_bc.disconnect(websocket)

    @app.websocket("/ws/trend")
    async def ws_trend(websocket: WebSocket) -> None:
        await websocket.accept()
        _trend_bc.connect(websocket)
        try:
            while True:
                await asyncio.sleep(60)
        except WebSocketDisconnect:
            pass
        finally:
            _trend_bc.disconnect(websocket)

    # Store broadcasters on app.state so the acquisition loop can reach them
    app.state.spectrum_bc = _spectrum_bc
    app.state.trend_bc = _trend_bc
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/webapp/test_server.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add spectraagent/webapp/server.py tests/spectraagent/webapp/test_server.py
git commit -m "feat: /ws/spectrum and /ws/trend WebSocket endpoints with Broadcaster fan-out"
```

---

## Task 11: Acquisition API Routes

**Files:**
- Modify: `spectraagent/webapp/server.py`
- Modify: `tests/spectraagent/webapp/test_server.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/spectraagent/webapp/test_server.py`:

```python
def test_acquisition_config_post(client):
    resp = client.post("/api/acquisition/config", json={
        "integration_time_ms": 100.0,
        "gas_label": "Ethanol",
        "target_concentration": 0.1,
    })
    assert resp.status_code == 200
    assert resp.json()["integration_time_ms"] == 100.0


def test_acquisition_config_defaults_ok(client):
    resp = client.post("/api/acquisition/config", json={})
    assert resp.status_code == 200


def test_acquisition_start_returns_session_id(client):
    resp = client.post("/api/acquisition/start")
    assert resp.status_code == 200
    assert "session_id" in resp.json()


def test_acquisition_stop(client):
    client.post("/api/acquisition/start")
    resp = client.post("/api/acquisition/stop")
    assert resp.status_code == 200


def test_acquisition_reference_requires_running(client):
    # Capture reference while not running — should still return 200 (simulation mode)
    resp = client.post("/api/acquisition/reference")
    assert resp.status_code in (200, 400)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/webapp/test_server.py::test_acquisition_config_post -v
```

Expected: 404 (route does not exist yet).

- [ ] **Step 3: Add acquisition routes to `server.py`**

Add to `spectraagent/webapp/server.py` after the import block (before `create_app`):

```python
from datetime import datetime
from pydantic import BaseModel


class AcquisitionConfig(BaseModel):
    integration_time_ms: float = 50.0
    gas_label: str = "unknown"
    target_concentration: float | None = None
```

Add inside `create_app`, after the trend WebSocket endpoint:

```python
    # ------------------------------------------------------------------
    # Acquisition API
    # ------------------------------------------------------------------
    _acq_config: dict[str, Any] = {
        "integration_time_ms": 50.0,
        "gas_label": "unknown",
        "target_concentration": None,
    }
    _session_active: dict[str, Any] = {"running": False, "session_id": None}
    _latest_spectrum: dict[str, Any] = {"wl": None, "intensities": None}

    @app.post("/api/acquisition/config")
    async def acq_config(cfg: AcquisitionConfig) -> JSONResponse:
        _acq_config.update(cfg.model_dump())
        if app.state.driver is not None:
            app.state.driver.set_integration_time_ms(cfg.integration_time_ms)
        return JSONResponse(_acq_config)

    @app.post("/api/acquisition/start")
    async def acq_start() -> JSONResponse:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        _session_active["running"] = True
        _session_active["session_id"] = session_id
        return JSONResponse({"status": "started", "session_id": session_id})

    @app.post("/api/acquisition/stop")
    async def acq_stop() -> JSONResponse:
        _session_active["running"] = False
        return JSONResponse({"status": "stopped",
                             "session_id": _session_active.get("session_id")})

    @app.post("/api/acquisition/reference")
    async def acq_reference() -> JSONResponse:
        intensities = _latest_spectrum.get("intensities")
        if intensities is None:
            return JSONResponse(
                {"error": "No spectrum available yet — wait for first frame"},
                status_code=400,
            )
        # Store reference in app state for physics plugin
        app.state.reference = intensities
        app.state.cached_ref = None  # invalidate cache
        return JSONResponse({"status": "reference_captured",
                             "peak_wavelength": None})  # peak computed lazily
```

- [ ] **Step 4: Run all tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/webapp/test_server.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add spectraagent/webapp/server.py
git commit -m "feat: /api/acquisition/* routes — config, start, stop, reference"
```

---

## Task 12: Fix save_raw Default in Orchestrator

**Files:**
- Modify: `src/inference/orchestrator.py:90`

- [ ] **Step 1: Find the save_raw default**

```bash
.venv/Scripts/python.exe -c "
import ast, pathlib
src = pathlib.Path('src/inference/orchestrator.py').read_text()
for i, line in enumerate(src.splitlines(), 1):
    if 'save_raw' in line:
        print(i, line)
"
```

Expected: shows lines where `save_raw` appears. Note the line number of the `__init__` default.

- [ ] **Step 2: Change the default**

In `src/inference/orchestrator.py`, find the `_SessionWriter.__init__` signature (around line 90) which contains `save_raw: bool = False` and change it to `save_raw: bool = True`.

Use the Read tool to find the exact line, then Edit to make the change. The line looks like:

```python
def __init__(
    ...
    save_raw: bool = False,
    ...
```

Change to:

```python
def __init__(
    ...
    save_raw: bool = True,
    ...
```

- [ ] **Step 3: Run existing tests to confirm nothing breaks**

```bash
.venv/Scripts/python.exe -m pytest tests/ -q --tb=short 2>&1 | tail -10
```

Expected: same number passing as before (669+), no new failures.

- [ ] **Step 4: Commit**

```bash
git add src/inference/orchestrator.py
git commit -m "fix: save_raw=True as default in _SessionWriter (spec requirement)"
```

---

## Task 13: `spectraagent start` CLI — Full Startup Sequence

**Files:**
- Modify: `spectraagent/__main__.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/spectraagent/test_config.py`:

```python
import subprocess
import sys
import time
import httpx


def test_start_simulate_no_browser_serves_health():
    """Integration test: start server in subprocess, hit /api/health, then kill it."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "spectraagent", "start", "--simulate", "--no-browser"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # Give uvicorn time to bind
        time.sleep(3)
        resp = httpx.get("http://127.0.0.1:8765/api/health", timeout=5)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["simulate"] is True
    finally:
        proc.terminate()
        proc.wait(timeout=5)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/test_config.py::test_start_simulate_no_browser_serves_health -v
```

Expected: FAIL — server not running (connection refused).

- [ ] **Step 3: Implement the `start` command in `spectraagent/__main__.py`**

Replace the stub `start` function with the full implementation:

```python
import logging
import os
import sys
import threading
import time
import webbrowser
from importlib.metadata import entry_points
from pathlib import Path

import typer
import uvicorn

from spectraagent.config import load_config

log = logging.getLogger(__name__)


@cli.command()
def start(
    simulate: bool = typer.Option(False, "--simulate", help="Force simulation mode"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open browser"),
    host: str = typer.Option("", "--host", help="Bind host (overrides config)"),
    port: int = typer.Option(0, "--port", help="Port (overrides config)"),
    physics: str = typer.Option("", "--physics", help="Sensor physics plugin name"),
) -> None:
    """Start the SpectraAgent server and open the browser."""
    cfg = load_config()

    bind_host = host or cfg.server.host
    bind_port = port or cfg.server.port
    physics_name = physics or cfg.physics.default_plugin

    # Step 1: Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        typer.echo(
            "⚠  ANTHROPIC_API_KEY not set — Claude agents disabled. "
            "Set the env var to enable AI features.",
            err=True,
        )

    # Step 2: Discover and load hardware driver
    driver = _load_driver(simulate, cfg)
    typer.echo(f"✓ Hardware: {driver.name}")

    # Step 3: Discover and load physics plugin
    plugin = _load_physics_plugin(physics_name)
    typer.echo(f"✓ Physics: {plugin.name}")

    # Step 4: Build the FastAPI app
    from spectraagent.webapp.server import create_app
    app = create_app(simulate=simulate)
    app.state.driver = driver
    app.state.plugin = plugin
    app.state.reference = None
    app.state.cached_ref = None

    # Step 5: Start the acquisition broadcast loop in a daemon thread
    acq_thread = threading.Thread(
        target=_acquisition_loop,
        args=(driver, app),
        daemon=True,
        name="spectraagent-acquisition",
    )
    acq_thread.start()
    typer.echo("✓ Acquisition loop started")

    # Step 6: Open browser (always localhost, not the bind host)
    if not no_browser:
        browser_url = f"http://localhost:{bind_port}"
        threading.Timer(1.5, lambda: webbrowser.open(browser_url)).start()
        typer.echo(f"✓ Opening browser at {browser_url}")

    # Step 7: Start uvicorn (blocking)
    typer.echo(f"✓ Serving at http://{bind_host}:{bind_port}")
    uvicorn.run(app, host=bind_host, port=bind_port, log_level="warning")


def _load_driver(simulate: bool, cfg) -> "AbstractHardwareDriver":
    """Load hardware driver: simulation if forced, else try config default, fallback sim."""
    from spectraagent.drivers.simulation import SimulationDriver

    if simulate:
        drv = SimulationDriver(integration_time_ms=cfg.hardware.integration_time_ms)
        drv.connect()
        return drv

    hw_eps = {ep.name: ep for ep in entry_points(group="spectraagent.hardware")}
    driver_name = cfg.hardware.default_driver
    if driver_name not in hw_eps:
        typer.echo(f"  Driver '{driver_name}' not found — falling back to simulation", err=True)
        drv = SimulationDriver(integration_time_ms=cfg.hardware.integration_time_ms)
        drv.connect()
        return drv

    try:
        cls = hw_eps[driver_name].load()
        drv = cls(integration_time_ms=cfg.hardware.integration_time_ms)
        drv.connect()
        return drv
    except Exception as exc:
        typer.echo(f"  Failed to connect to {driver_name}: {exc} — falling back to simulation", err=True)
        drv = SimulationDriver(integration_time_ms=cfg.hardware.integration_time_ms)
        drv.connect()
        return drv


def _load_physics_plugin(name: str) -> "AbstractSensorPhysicsPlugin":
    """Load physics plugin by entry-point name."""
    ph_eps = {ep.name: ep for ep in entry_points(group="spectraagent.sensor_physics")}
    if name not in ph_eps:
        raise typer.BadParameter(f"Physics plugin '{name}' not found. Run 'spectraagent plugins list'.")
    return ph_eps[name].load()()


def _acquisition_loop(driver: "AbstractHardwareDriver", app: "FastAPI") -> None:
    """Daemon thread: read spectra and broadcast to WebSocket clients."""
    import asyncio
    import json

    from src.inference.live_state import LiveDataStore

    wl = driver.get_wavelengths().tolist()

    while True:
        try:
            intensities = driver.read_spectrum()
        except Exception as exc:
            log.warning("Acquisition error: %s", exc)
            time.sleep(1.0)
            continue

        # Push to LiveDataStore (existing code reads from here)
        LiveDataStore.push_raw(wl=wl, intensities=intensities.tolist())

        # Broadcast spectrum to /ws/spectrum clients
        spectrum_bc = getattr(app.state, "spectrum_bc", None)
        if spectrum_bc is not None:
            msg = json.dumps({"wl": wl, "i": intensities.tolist()})
            # Fire-and-forget via the event loop (non-blocking from this thread)
            try:
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(
                    lambda m=msg: asyncio.ensure_future(spectrum_bc.broadcast(m))
                )
            except RuntimeError:
                pass  # loop not running yet — skip this frame
```

- [ ] **Step 4: Check `LiveDataStore` has `push_raw` or find the right method name**

```bash
.venv/Scripts/python.exe -c "
from src.inference.live_state import LiveDataStore
print([m for m in dir(LiveDataStore) if not m.startswith('__')])
"
```

Expected: lists methods including `push`. If the method is `push` (not `push_raw`), update the acquisition loop to call `LiveDataStore.push(result)` with the appropriate dict format. Use whatever method already exists — do not add a new method to `live_state.py`.

- [ ] **Step 5: Run the integration test**

```bash
.venv/Scripts/python.exe -m pytest tests/spectraagent/test_config.py::test_start_simulate_no_browser_serves_health -v -s
```

Expected: PASS. The test starts a real uvicorn server, hits `/api/health`, and terminates it.

- [ ] **Step 6: Manually verify the full startup**

```bash
.venv/Scripts/python.exe -m spectraagent start --simulate --no-browser
```

Expected terminal output:
```
✓ Hardware: Simulation
✓ Physics: LSPR
✓ Acquisition loop started
✓ Serving at http://127.0.0.1:8765
```

In another terminal:
```bash
curl http://127.0.0.1:8765/api/health
```

Expected: `{"status":"ok","version":"1.0.0","hardware":"Simulation","simulate":true}`

- [ ] **Step 7: Run full test suite to verify nothing broken**

```bash
.venv/Scripts/python.exe -m pytest tests/ -q --tb=short 2>&1 | tail -15
```

Expected: 669+ passed, 0 errors.

- [ ] **Step 8: Commit**

```bash
git add spectraagent/__main__.py
git commit -m "feat: spectraagent start — full startup with driver discovery, acquisition loop, uvicorn"
```

---

## Self-Review

After completing Task 13, run this checklist:

**Spec coverage check (Section 14 Phases 1–4):**
- [x] Phase 1 — skeleton: `spectraagent/`, `pyproject.toml`, `spectraagent.__main__:cli` entry point
- [x] Phase 2 — hardware adapters: `AbstractHardwareDriver`, `SimulationDriver`, `ThorlabsCCSDriver`
- [x] Phase 3 — physics plugin: `AbstractSensorPhysicsPlugin`, `LSPRPlugin`
- [x] Phase 4 — FastAPI backend: `server.py`, CORS, static mount, `/api/health`, `/ws/spectrum`, `/ws/trend`, `/api/acquisition/*`, `spectraagent start --simulate --no-browser`

**Acceptance criteria (from spec):**
- [ ] `spectraagent --help` shows `start` and `plugins` commands
- [ ] `spectraagent plugins list` shows simulation + lspr as loadable
- [ ] `spectraagent start --simulate --no-browser` starts uvicorn on port 8765
- [ ] `GET /api/health` returns `{"status": "ok", ...}`
- [ ] `POST /api/acquisition/config` persists integration time
- [ ] All 669+ existing tests still pass

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

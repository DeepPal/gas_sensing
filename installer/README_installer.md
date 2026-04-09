# SpectraAgent — Windows Installer

## What this produces

A single `.exe` file (~60 MB) that a researcher double-clicks to install SpectraAgent. No terminal, no Python knowledge needed.

## Installer flow (what the researcher sees)

1. **Welcome screen** — brief description
2. **License agreement** — MIT
3. **Install location** — defaults to `C:\Users\<name>\AppData\Local\SpectraAgent\`
4. **Component selection**:
   - Core platform (required, ~150 MB download)
   - CNN classifier via PyTorch (optional, ~2.5 GB download, requires internet)
   - Hardware VISA support (optional, for ThorLabs CCS200)
5. **Shortcuts** — Desktop and Start Menu options
6. **Progress window** — creates venv, installs packages, builds React frontend
7. **First-run wizard** — set dashboard password + Anthropic API key
8. **Finish**

Total install time: 5–20 min depending on internet speed and whether PyTorch is selected.

## How to build the installer

### Prerequisites

1. Install **Inno Setup 6** (free): https://jrsoftware.org/isdl.php
2. Install **Node.js** (for React pre-build, optional): https://nodejs.org
3. Have the project in a clean state (`git status` shows no untracked junk)

### Build

```bat
installer\build_installer.bat
```

The `.exe` appears at `installer\dist\SpectraAgent_1.0.0_Setup.exe`.

### Or manually in Inno Setup IDE

1. Open Inno Setup IDE (installed with Inno Setup)
2. File → Open → `installer\spectraagent_setup.iss`
3. Press F9 (Build)

## File structure

```
installer/
├── spectraagent_setup.iss      ← Main Inno Setup script (edit this for version bumps)
├── install_deps.bat            ← Called during install: creates venv, pip install
├── first_run_wizard.py         ← GUI wizard: password + API key setup (tkinter)
├── build_installer.bat         ← One-click build script
├── README_installer.md         ← This file
├── resources/
│   ├── icon.ico                ← App icon (replace with your lab logo)
│   ├── installer_banner.bmp    ← Left-panel banner (164×314 px, 24-bit BMP)
│   ├── installer_icon.bmp      ← Small icon (55×58 px, 24-bit BMP)
│   ├── run_spectraagent_installed.bat  ← Location-aware launcher (copied to install dir)
│   └── run_dashboard_installed.bat    ← Location-aware dashboard launcher
└── dist/                       ← Build output (gitignored)
    └── SpectraAgent_1.0.0_Setup.exe
```

## Adding a custom icon and banner

- **icon.ico**: 256×256 + 48×48 + 32×32 + 16×16 multi-size .ico file
- **installer_banner.bmp**: 164×314 px, 24-bit BMP — shown on the left side of every page
- **installer_icon.bmp**: 55×58 px, 24-bit BMP — shown top-right

Generate a placeholder:
```python
from PIL import Image
Image.new("RGB", (164, 314), color=(26, 26, 46)).save("installer/resources/installer_banner.bmp")
Image.new("RGB", (55, 58),   color=(15, 138, 191)).save("installer/resources/installer_icon.bmp")
```

## Updating the version

1. Update `AppVersion` in `spectraagent_setup.iss` (line 10)
2. Update `version` in `pyproject.toml`
3. Rebuild

## Known limitations

- **No offline install**: PyTorch and other packages are downloaded from PyPI/pytorch.org during install. Offline bundle possible but would require shipping a ~3 GB wheel cache.
- **Python must be pre-installed OR user accepts the Python download prompt**. We do not bundle an embedded Python (adds ~70 MB, complex to maintain).
- **VISA drivers not bundled**: ThorLabs IVI/VISA drivers must be installed separately from the ThorLabs website. The installer checks for `pyvisa` but cannot install the native DLL.
- **macOS/Linux**: Use `run_spectraagent.bat`/`run_dashboard_secure.sh` directly; no installer for non-Windows platforms yet.

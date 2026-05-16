@echo off
:: ============================================================
:: install_deps.bat
:: Called by Inno Setup during installation.
:: Args: %1 = install directory   %2 = install_ml (0 or 1)
:: ============================================================
setlocal EnableDelayedExpansion

set INSTALL_DIR=%~1
set INSTALL_ML=%~2

cd /d "%INSTALL_DIR%"

echo.
echo ============================================================
echo   SpectraAgent — Dependency Installation
echo ============================================================
echo   Install directory: %INSTALL_DIR%
echo.

:: ── Find Python ──────────────────────────────────────────────
set PYTHON_EXE=
if exist "%INSTALL_DIR%\installer\python_path.txt" (
    set /p PYTHON_EXE=<"%INSTALL_DIR%\installer\python_path.txt"
)
if not defined PYTHON_EXE (
    :: Fallback: try PATH
    where python >nul 2>&1
    if not errorlevel 1 (
        set PYTHON_EXE=python
    ) else (
        echo ERROR: Python not found. Please install Python 3.9+ from python.org
        exit /b 1
    )
)
echo   Python: %PYTHON_EXE%
echo.

:: ── Create virtual environment ───────────────────────────────
echo [1/5] Creating virtual environment...
if exist ".venv\Scripts\python.exe" (
    echo   Virtual environment already exists — skipping creation.
) else (
    "%PYTHON_EXE%" -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        exit /b 1
    )
    echo   Done.
)

:: ── Upgrade pip and core build tools ─────────────────────────
echo.
echo [2/5] Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel --quiet
if errorlevel 1 (
    echo WARNING: pip upgrade failed — continuing with existing version.
)
echo   Done.

:: ── Install core dependencies ─────────────────────────────────
echo.
echo [3/5] Installing core dependencies (~150 MB, please wait)...
echo   This includes: numpy, scipy, pandas, scikit-learn, FastAPI,
echo   Streamlit, Plotly, anthropic, and all other required packages.
echo.
.venv\Scripts\pip.exe install -e ".[all,tracking]" --quiet --no-warn-script-location
if errorlevel 1 (
    echo.
    echo ERROR: Core dependency installation failed.
    echo   Check your internet connection and try running:
    echo     %INSTALL_DIR%\.venv\Scripts\pip install -e ".[all]"
    exit /b 1
)
echo   Core packages installed.

:: ── Install hardware VISA support ────────────────────────────
echo.
echo [4/5] Installing hardware VISA support (ThorLabs CCS200)...
.venv\Scripts\pip.exe install pyvisa pyvisa-py --quiet --no-warn-script-location
if errorlevel 1 (
    echo   WARNING: VISA hardware support failed to install.
    echo   Simulation mode will still work. For hardware use, run:
    echo     %INSTALL_DIR%\.venv\Scripts\pip install pyvisa pyvisa-py
) else (
    echo   VISA support installed.
)

:: ── Install ML/PyTorch (optional) ────────────────────────────
echo.
if "%INSTALL_ML%"=="1" (
    echo [5/5] Installing PyTorch CNN classifier (~2.5 GB download)...
    echo   This may take 10-30 minutes depending on your internet speed.
    echo.
    .venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet --no-warn-script-location
    if errorlevel 1 (
        echo   WARNING: PyTorch installation failed.
        echo   The platform will work without CNN classification.
        echo   To install later, run:
        echo     %INSTALL_DIR%\.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    ) else (
        echo   PyTorch installed. CNN gas classification enabled.
    )
) else (
    echo [5/5] Skipping PyTorch (CNN classifier not selected).
    echo   To install later: %INSTALL_DIR%\.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

:: ── Build React frontend ──────────────────────────────────────
echo.
echo Building React frontend...
if exist "spectraagent\webapp\frontend\package.json" (
    where npm >nul 2>&1
    if not errorlevel 1 (
        cd spectraagent\webapp\frontend
        call npm install --silent 2>nul
        call npm run build --silent 2>nul
        cd /d "%INSTALL_DIR%"
        echo   React frontend built.
    ) else (
        echo   Node.js not found — skipping React build.
        echo   The API and Streamlit dashboard will work normally.
        echo   For the live React UI: install Node.js from nodejs.org and run:
        echo     cd %INSTALL_DIR%\spectraagent\webapp\frontend
        echo     npm install ^&^& npm run build
    )
) else (
    echo   Frontend package.json not found — skipping.
)

:: ── Done ─────────────────────────────────────────────────────
echo.
echo ============================================================
echo   Installation complete!
echo ============================================================
echo.
echo   To start SpectraAgent:  run_spectraagent.bat --simulate
echo   To open the dashboard:  run_dashboard_secure.bat
echo.

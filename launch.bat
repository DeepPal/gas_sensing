@echo off
:: ============================================================
::  SpectraAgent — Single-command launcher
::  Starts BOTH the live acquisition server AND the dashboard.
::
::  Usage:
::    launch.bat              — simulation mode (default, no hardware needed)
::    launch.bat --hardware   — real ThorLabs CCS200 spectrometer
::    launch.bat --help       — show this help
:: ============================================================
setlocal EnableDelayedExpansion

cd /d "%~dp0"

REM ── Parse arguments ──────────────────────────────────────────
set MODE_FLAG=--simulate
set MODE_NAME=Simulation
for %%A in (%*) do (
    if /I "%%A"=="--hardware" ( set MODE_FLAG=--hardware & set MODE_NAME=Hardware )
    if /I "%%A"=="-h"         ( set MODE_FLAG=--hardware & set MODE_NAME=Hardware )
    if /I "%%A"=="--help"     ( goto :show_help )
)

REM ── Check virtual environment ─────────────────────────────────
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo  ERROR: Python environment not found.
    echo  Run the installer or:  python -m venv .venv ^&^& .venv\Scripts\pip install -e ".[dev]"
    echo.
    pause & exit /b 1
)

REM ── Check password is configured ─────────────────────────────
set PW_OK=0
if defined DASHBOARD_PASSWORD     set PW_OK=1
if defined DASHBOARD_PASSWORD_HASH set PW_OK=1
if exist "%USERPROFILE%\.streamlit_au_mip_password_hash" set PW_OK=1
if exist "%USERPROFILE%\.streamlit_au_mip_password"      set PW_OK=1
if "%PW_OK%"=="0" (
    echo.
    echo  No dashboard password set. Running first-time setup...
    echo.
    .venv\Scripts\python.exe -m dashboard.auth --set-password
    if errorlevel 1 (
        echo  Password setup failed. Exiting.
        pause & exit /b 1
    )
)

cls
echo.
echo  ================================================================
echo    SpectraAgent  ^|  %MODE_NAME% Mode
echo  ================================================================
echo.
echo    SpectraAgent  -^>  http://localhost:8765/app
echo    Dashboard     -^>  http://localhost:8501
echo.
echo    Press Ctrl+C (in this window) to stop both services.
echo  ================================================================
echo.

REM ── Kill any stale processes on our ports ────────────────────
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8765 " 2^>nul') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8501 " 2^>nul') do taskkill /F /PID %%a >nul 2>&1
timeout /t 1 /nobreak >nul

REM ── Launch everything through the Python launcher ────────────
.venv\Scripts\python.exe launcher.py %MODE_FLAG%

echo.
echo  Both services have stopped.
pause
exit /b 0

:show_help
echo.
echo  SpectraAgent Launcher
echo  ---------------------
echo  launch.bat              Start in simulation mode (default)
echo  launch.bat --hardware   Start with ThorLabs CCS200 spectrometer
echo  launch.bat --help       Show this help
echo.
echo  Starts both:
echo    - SpectraAgent live acquisition server (http://localhost:8765/app)
echo    - Streamlit analysis dashboard         (http://localhost:8501)
echo.
echo  Press Ctrl+C to stop both services cleanly.
echo.
pause

@echo off
setlocal EnableDelayedExpansion
title SpectraAgent
echo ============================================================
echo   SpectraAgent  —  Universal Agentic Spectroscopy Platform
echo ============================================================
echo.

REM ── Parse arguments ──────────────────────────────────────────
set SIMULATE_FLAG=
set PORT=8765
for %%A in (%*) do (
    if /I "%%A"=="--simulate" set SIMULATE_FLAG=--simulate
    if /I "%%A"=="-s"         set SIMULATE_FLAG=--simulate
)

REM ── Check virtual environment ─────────────────────────────────
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv not found.
    echo Run:  python -m venv .venv ^&^& .venv\Scripts\pip install -e ".[dev]"
    echo.
    pause
    exit /b 1
)

REM ── Kill any existing server on the configured port ───────────
echo [1/3] Checking for existing process on port %PORT%...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":%PORT% " 2^>nul') do (
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 1 /nobreak >nul

REM ── Activate virtual environment ─────────────────────────────
echo [2/3] Activating Python environment...
call .venv\Scripts\activate.bat

REM ── Determine run mode ────────────────────────────────────────
echo [3/3] Starting SpectraAgent...
if defined SIMULATE_FLAG (
    echo   Mode:  Simulation  [no hardware required]
) else (
    echo   Mode:  Hardware    [ThorLabs CCS200 via plugin registry]
    echo   Tip:   Pass --simulate if no spectrometer is connected.
)
echo.
echo   Web app:   http://localhost:%PORT%/app
echo   API docs:  http://localhost:%PORT%/docs
echo   WebSocket: ws://localhost:%PORT%/ws/live
echo.
echo   Press Ctrl+C to stop.
echo ============================================================
echo.

REM ── Launch ───────────────────────────────────────────────────
.venv\Scripts\python.exe -m spectraagent start %SIMULATE_FLAG% --port %PORT%

echo.
echo SpectraAgent stopped.
pause

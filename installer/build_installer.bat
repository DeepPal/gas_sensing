@echo off
:: ============================================================
:: build_installer.bat
:: Builds the SpectraAgent Windows installer using Inno Setup 6.
::
:: Prerequisites:
::   1. Inno Setup 6  — https://jrsoftware.org/isdl.php
::   2. Run this from the project root OR from the installer/ directory
:: ============================================================
setlocal EnableDelayedExpansion

echo.
echo ============================================================
echo   SpectraAgent Installer Builder
echo ============================================================
echo.

REM ── Locate the project root ───────────────────────────────────
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
cd /d "%PROJECT_ROOT%"
echo   Project root: %PROJECT_ROOT%

REM ── Check Inno Setup is installed ────────────────────────────
set ISCC=
for %%D in (
    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
    "C:\Program Files\Inno Setup 6\ISCC.exe"
    "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
) do (
    if exist %%D (
        set ISCC=%%~D
        goto :found_iscc
    )
)
echo ERROR: Inno Setup 6 not found.
echo.
echo Please install it from: https://jrsoftware.org/isdl.php
echo Then re-run this script.
echo.
pause
exit /b 1

:found_iscc
echo   Inno Setup: %ISCC%
echo.

REM ── Check LICENSE file exists ─────────────────────────────────
if not exist "LICENSE" (
    echo WARNING: LICENSE file not found. Creating a placeholder...
    echo MIT License > LICENSE
    echo Copyright ^(c^) 2026 Chulalongkorn University LSPR Sensing Lab >> LICENSE
)

REM ── Create installer icon placeholder if not present ─────────
if not exist "installer\resources\icon.ico" (
    echo WARNING: installer\resources\icon.ico not found.
    echo   Using Inno Setup default icon. Add a custom .ico for branded installer.
)

REM ── Create output directory ───────────────────────────────────
if not exist "installer\dist" mkdir "installer\dist"

REM ── Pre-build React frontend ──────────────────────────────────
echo [1/3] Building React frontend...
if exist "spectraagent\webapp\frontend\package.json" (
    where npm >nul 2>&1
    if not errorlevel 1 (
        cd spectraagent\webapp\frontend
        echo   Running npm install...
        call npm install --silent
        echo   Running npm build...
        call npm run build --silent
        cd /d "%PROJECT_ROOT%"
        echo   React frontend built at spectraagent\webapp\frontend\dist\
    ) else (
        echo   Node.js not found — skipping React pre-build.
        echo   The installer will attempt to build it on the target machine.
    )
) else (
    echo   No package.json found — skipping.
)

REM ── Run Inno Setup compiler ───────────────────────────────────
echo.
echo [2/3] Compiling installer...
cd /d "%PROJECT_ROOT%"
"%ISCC%" "installer\spectraagent_setup.iss" /O"installer\dist" /Q
if errorlevel 1 (
    echo.
    echo ERROR: Inno Setup compilation failed.
    echo Check the output above for details.
    pause
    exit /b 1
)

REM ── Report ────────────────────────────────────────────────────
echo.
echo [3/3] Done!
echo.
echo   Installer created at:
dir /b "installer\dist\SpectraAgent_*.exe" 2>nul | findstr /I ".exe" && (
    for %%F in ("installer\dist\SpectraAgent_*.exe") do (
        echo     %%F
        echo     Size: %%~zF bytes
    )
)
echo.
echo   Share this .exe file with researchers — no other files needed.
echo   They run it and the wizard handles everything.
echo.
pause

@echo off
REM run_dashboard_secure.bat (installed version)
REM Installed location-aware wrapper — adapts to wherever SpectraAgent was installed.

setlocal enabledelayedexpansion

REM ── Switch to install directory ───────────────────────────────
set INSTALL_DIR=%~dp0
cd /d "%INSTALL_DIR%"

REM ── Activate virtual environment ─────────────────────────────
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found at %INSTALL_DIR%.venv
    echo Please re-run the SpectraAgent installer to repair the installation.
    pause
    exit /b 1
)

REM Print startup info
echo.
echo ==========================================
echo   SpectraAgent — Streamlit Dashboard
echo ==========================================
echo   Dashboard URL: http://localhost:8501
echo ---

REM Show password status / fail closed when no secret is configured.
if defined DASHBOARD_PASSWORD (
    echo [OK] Using DASHBOARD_PASSWORD from environment
) else if defined DASHBOARD_PASSWORD_HASH (
    echo [OK] Using DASHBOARD_PASSWORD_HASH from environment
) else if exist "%USERPROFILE%\.streamlit_au_mip_password_hash" (
    echo [OK] Password file found
) else if exist "%USERPROFILE%\.streamlit_au_mip_password" (
    echo [WARN] Using legacy plaintext password file
) else (
    echo [ERROR] No dashboard password configured.
    echo.
    echo  To set a password, open Start Menu - SpectraAgent - Set Dashboard Password
    echo  Or run:  .venv\Scripts\python.exe -m dashboard.auth --set-password
    echo.
    pause
    exit /b 1
)

echo ---
echo Press Ctrl+C to stop
echo ==========================================
echo.

REM Try to generate self-signed certs (non-fatal if OpenSSL unavailable)
.venv\Scripts\python.exe -m dashboard.security >nul 2>nul

set "CERT_FILE=.streamlit\certs\server.crt"
set "KEY_FILE=.streamlit\certs\server.key"
set "HTTPS_ARGS="
if exist "%CERT_FILE%" if exist "%KEY_FILE%" (
    echo HTTPS mode: enabled
    echo Dashboard URL: https://localhost:8501
    set "HTTPS_ARGS=--server.sslCertFile=%CERT_FILE% --server.sslKeyFile=%KEY_FILE%"
)

.venv\Scripts\python.exe -m streamlit run dashboard/app.py ^
    --logger.level=info ^
    --client.showErrorDetails=false ^
    --server.headless=true ^
    --server.enableXsrfProtection=true ^
    --server.enableCORS=false ^
    !HTTPS_ARGS!

pause

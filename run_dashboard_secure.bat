@echo off
REM run_dashboard_secure.bat - Launch the dashboard with authentication
REM
REM Usage:
REM   run_dashboard_secure.bat
REM
REM To set password for this session:
REM   set DASHBOARD_PASSWORD=my-lab-password
REM   run_dashboard_secure.bat

setlocal enabledelayedexpansion

cd /d "%~dp0"

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    echo.
    echo 📦 Activating virtual environment...
    call .venv\Scripts\activate.bat
)

REM Print startup info
echo.
echo ==========================================
echo 🔬 SpectraAgent — Spectrometer-Based Sensing Platform
echo ==========================================
echo Dashboard URL: http://localhost:8501
echo ---

REM Show password status / fail closed when no secret is configured.
if defined DASHBOARD_PASSWORD (
    echo [OK] Using DASHBOARD_PASSWORD from environment
) else if defined DASHBOARD_PASSWORD_HASH (
    echo [OK] Using DASHBOARD_PASSWORD_HASH from environment
) else if exist "%USERPROFILE%\.streamlit_au_mip_password_hash" (
    echo [OK] Using hashed password file %USERPROFILE%\.streamlit_au_mip_password_hash
) else if exist "%USERPROFILE%\.streamlit_au_mip_password" (
    echo [WARN] Using legacy plaintext password file %USERPROFILE%\.streamlit_au_mip_password
) else (
    echo [ERROR] No dashboard password configured.
    echo         Run: python -m dashboard.auth --set-password
    echo         Or set DASHBOARD_PASSWORD for this session.
    exit /b 1
)

echo ---
echo Health check: python -m dashboard.health
echo Press Ctrl+C to stop
echo ==========================================
echo.

REM Try to generate self-signed certs (non-fatal if OpenSSL is unavailable)
python -m dashboard.security >nul 2>nul

set "CERT_FILE=.streamlit\certs\server.crt"
set "KEY_FILE=.streamlit\certs\server.key"
set "HTTPS_ARGS="
if exist "%CERT_FILE%" if exist "%KEY_FILE%" (
    echo HTTPS mode: enabled (self-signed cert)
    echo Dashboard URL: https://localhost:8501
    set "HTTPS_ARGS=--server.sslCertFile=%CERT_FILE% --server.sslKeyFile=%KEY_FILE%"
) else (
    echo HTTPS mode: unavailable (running HTTP)
)
echo.

REM Launch Streamlit
streamlit run dashboard/app.py ^
    --logger.level=info ^
    --client.showErrorDetails=false ^
    --server.headless=true ^
    --server.enableXsrfProtection=true ^
    --server.enableCORS=false ^
    !HTTPS_ARGS!

pause

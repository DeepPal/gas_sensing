@echo off
REM run_dashboard_secure.bat - Launch the dashboard with authentication
REM
REM Usage:
REM   run_dashboard_secure.bat
REM
REM To set password:
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
echo 🔬 Au-MIP LSPR Gas Sensing Platform
echo ==========================================
echo Dashboard URL: http://localhost:8501
echo ---

REM Show password status
if defined DASHBOARD_PASSWORD (
    echo ✓ Using DASHBOARD_PASSWORD from environment
) else (
    echo ⚠️  Using default password 'research-lab-default'
    echo    Set DASHBOARD_PASSWORD env var to customize
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

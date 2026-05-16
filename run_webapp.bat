@echo off
REM Run SpectroSense web app from project root
REM .venv is at Main_Research_Chula/.venv
REM After cd, we are 2 levels deep → use ..\..\.venv\...

cd /d "%~dp0spectrometer_webapp\backend"
echo.
echo  SpectroSense ^| http://localhost:8080
echo  Press Ctrl+C to stop.
echo.

REM Two levels up from backend/ reaches the project root where .venv lives
"..\..\.venv\Scripts\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload
pause

@echo off
cd /d "%~dp0"
REM Streamlit dashboard (port 8501) + FastAPI live server (port 5006) start together.
REM The live server is launched automatically by app.py in a background thread.
REM Open http://localhost:8501 for the full dashboard.
REM Open http://localhost:5006 for the standalone live spectrum view.
.venv\Scripts\python.exe -m streamlit run dashboard/app.py
pause

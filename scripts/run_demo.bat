@echo off
rem Launches the Swing Analyzer demo and opens it in your browser.
rem Close this window (or press Ctrl+C) to stop the demo.
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Could not find .venv\Scripts\python.exe. Create the environment first; see README.md.
  pause
  exit /b 1
)
echo Starting Swing Analyzer. First load of a clip takes about a minute while frames decode.
start "" /min powershell -NoProfile -Command "Start-Sleep 6; Start-Process http://127.0.0.1:8501"
".venv\Scripts\python.exe" -m streamlit run app\streamlit_app.py --server.port 8501
echo.
echo The demo stopped.
pause

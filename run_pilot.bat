@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First create a virtual environment and install requirements, as shown in README.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m schlieren_qml run configs\pilot.json
set "SCHLIEREN_EXIT_CODE=%errorlevel%"
pause
exit /b %SCHLIEREN_EXIT_CODE%

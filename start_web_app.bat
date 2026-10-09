@echo off
setlocal
cd /d "%~dp0"

rem Use the same venv as train/test to avoid mixed dependency versions.
set "PROJECT_VENV=%~dp0..\venv\Scripts\python.exe"
if not exist "%PROJECT_VENV%" (
  echo Cannot find project venv Python:
  echo   %PROJECT_VENV%
  echo.
  echo Run setup_new_machine.bat first.
  exit /b 1
)

rem Startup must never select/repair model paths implicitly. Recipes use immutable local IDs.
rem Use the maintenance repair tool explicitly after moving the legacy workspace.

"%PROJECT_VENV%" "%~dp007_web_app\app.py"

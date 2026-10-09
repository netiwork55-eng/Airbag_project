@echo off
setlocal
cd /d "%~dp0"

echo === Airbag Project new-machine setup ===
echo Project: %CD%
echo.

set "PROJECT_VENV=%~dp0..\venv\Scripts\python.exe"
set "PROJECT_PIP=%~dp0..\venv\Scripts\pip.exe"

if not exist "%PROJECT_VENV%" (
  echo Creating virtual environment:
  echo   %~dp0..\venv
  py -3.12 -m venv "%~dp0..\venv"
  if errorlevel 1 (
    echo py launcher failed. Trying python...
    python -m venv "%~dp0..\venv"
  )
)

if not exist "%PROJECT_VENV%" (
  echo.
  echo Cannot create venv. Install Python 3.12 first, then run this file again.
  exit /b 1
)

echo.
echo Checking the tested Python version...
"%PROJECT_VENV%" -c "import sys; sys.exit(0 if sys.version_info[:2] == (3,12) else 1)"
if errorlevel 1 exit /b 1

echo.
echo Installing project requirements...
"%PROJECT_PIP%" install --index-url https://pypi.org/simple -r "%~dp0requirements.txt"
if errorlevel 1 exit /b 1

echo.
echo Installing the separate inspection worker from the pinned official packages...
"%PROJECT_VENV%" "%~dp0tools\maintenance\setup_inspection.py" --fetch-backbone
if errorlevel 1 exit /b 1

echo.
echo Setup complete.
echo Next: run start_web_app.bat and open http://127.0.0.1:8000
pause

@echo off
REM WaferTools V3 Windows Launcher
REM Double-click this file to start WaferTools

title WaferTools V3 Launcher

echo ================================================================
echo          WaferTools V3 - Windows Launcher
echo          Harvard University Lichtman Lab
echo ================================================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo [INFO] Please install Python 3.10+ from https://www.python.org
    echo.
    pause
    exit /b 1
)

echo [OK] Python found
echo.

REM Check if virtual environment exists
if not exist ".venv\" (
    if not exist "venv\" (
        echo [INFO] Virtual environment not found
        echo [INFO] Creating virtual environment...
        python -m venv .venv
        
        echo [INFO] Installing dependencies...
        call .venv\Scripts\activate.bat
        python -m pip install --upgrade pip
        pip install -r requirements.txt 2>nul || pip install -r offline\requirements_offline.txt
    ) else (
        call venv\Scripts\activate.bat
    )
) else (
    call .venv\Scripts\activate.bat
)

echo [OK] Python environment ready
echo.

REM Run the launcher
python launcher.py

REM Keep window open on error
if errorlevel 1 (
    echo.
    echo [ERROR] WaferTools exited with an error
    pause
)


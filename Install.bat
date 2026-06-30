@echo off
REM ============================================================
REM  WaferTools - One-click FIRST-TIME INSTALL
REM  For a computer that does NOT have WaferTools yet.
REM  Save this single file where you want WaferTools to live
REM  (e.g. D:\), then double-click it. It downloads the program
REM  into a new "WaferTools" folder that is already update-ready.
REM ============================================================
title WaferTools - Install
cd /d "%~dp0"

echo ================================================================
echo            WaferTools - First-time Install
echo ================================================================
echo.

REM --- Git installed? ---
where git >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git is not installed on this computer.
    echo [INFO]  Install it once from https://git-scm.com/download/win
    echo         then run this file again.
    echo.
    pause
    exit /b 1
)

REM --- Already installed here? ---
if exist "WaferTools\.git\" (
    echo [OK]   WaferTools is already installed in this location.
    echo [INFO] Open the "WaferTools" folder, then use:
    echo          Update.bat      to get the latest version
    echo          WaferTools.bat  to start the program
    echo.
    pause
    exit /b 0
)

echo [INFO] Downloading WaferTools from GitHub...
echo        (first time only - this can take a few minutes)
echo.
git clone https://github.com/FumingYang-Felix/WaferTools.git
if errorlevel 1 (
    echo.
    echo [ERROR] Download failed. Check the internet connection and try again.
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================================
echo  [OK] Installed into the "WaferTools" folder next to this file.
echo
echo  Next:
echo    1. Open the new "WaferTools" folder
echo    2. Double-click  WaferTools.bat   to start the program
echo    3. To update later: double-click  Update.bat
echo ================================================================
echo.
pause

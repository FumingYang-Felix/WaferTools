@echo off
REM ============================================================
REM  WaferTools - One-click Update (sync the latest version)
REM  Just double-click this file. No typing needed.
REM ============================================================
title WaferTools Update
cd /d "%~dp0"

echo ================================================================
echo            WaferTools - Update / Sync
echo ================================================================
echo.

REM --- Is Git available? ---
where git >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git is not installed on this computer.
    echo [INFO]  Install it once from https://git-scm.com/download/win
    echo         then run this file again. Or ask Fuming for help.
    echo.
    pause
    exit /b 1
)

REM --- Is this folder a Git checkout? ---
if not exist ".git\" (
    echo [ERROR] This folder is not set up for one-click updates yet.
    echo [INFO]  Ask Fuming to run the one-time Git setup here.
    echo.
    pause
    exit /b 1
)

echo [INFO] Downloading the latest version from GitHub...
echo.
git pull
if errorlevel 1 (
    echo.
    echo [WARN] The update did not finish cleanly.
    echo [INFO] Your current version still works fine.
    echo        If this keeps happening, send a photo of this window to Fuming.
) else (
    echo.
    echo [OK]   You now have the latest version.
)

echo.
echo ----------------------------------------------------------------
echo  Next: double-click  WaferTools.bat  to start the program.
echo ----------------------------------------------------------------
echo.
pause

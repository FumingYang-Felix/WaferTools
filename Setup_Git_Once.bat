@echo off
REM ============================================================
REM  WaferTools - ONE-TIME Git setup
REM  Run this ONCE on a machine where WaferTools was installed
REM  from a downloaded ZIP (folder has no Git yet). After this,
REM  Update.bat will work.  Best run with Fuming present.
REM
REM  Your data (results, uploads, settings) is NOT touched -
REM  only the program source files are linked to GitHub.
REM ============================================================
title WaferTools - One-time Git Setup
cd /d "%~dp0"

echo ================================================================
echo            WaferTools - One-time Git setup
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

REM --- Already set up? ---
if exist ".git\" (
    echo [OK]   This folder is already linked to GitHub.
    echo [INFO] You don't need this file again - just use Update.bat.
    echo.
    pause
    exit /b 0
)

echo [INFO] Linking this folder to the WaferTools GitHub repository...
git init >nul 2>&1
git remote remove origin >nul 2>&1
git remote add origin https://github.com/FumingYang-Felix/WaferTools.git

echo [INFO] Downloading the latest version (this can take a minute)...
git fetch origin
if errorlevel 1 (
    echo.
    echo [ERROR] Could not reach GitHub. Check the internet connection
    echo         and run this file again.
    echo.
    pause
    exit /b 1
)

echo [INFO] Applying the latest version (your data is kept)...
git checkout -f -B main origin/main
if errorlevel 1 (
    echo.
    echo [ERROR] Setup could not finish. Send a photo of this window to Fuming.
    echo.
    pause
    exit /b 1
)
git branch --set-upstream-to=origin/main main >nul 2>&1

echo.
echo ================================================================
echo  [OK] Setup complete!  From now on:
echo        1. Double-click  Update.bat       to get the latest version
echo        2. Double-click  WaferTools.bat   to start the program
echo ================================================================
echo.
pause

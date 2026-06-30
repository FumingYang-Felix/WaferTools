#!/bin/bash
# WaferTools Direct Launcher - No permission needed
# This script runs directly in the current terminal

# Get script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Set terminal window size (wider)
# Format: columns x rows (default is usually 80x24, we make it wider)
printf '\e[8;30;120t'

# Clear screen and show header
clear
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         WaferTools V3 - Direct Launcher                   ║"
echo "║         Harvard University Lichtman Lab                    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[!] Python 3 not found"
    echo "    Please install Python from https://www.python.org"
    read -p "Press Enter to exit..."
    exit 1
fi

# Activate venv if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run launcher
python3 launcher.py

# Keep window open on error
if [ $? -ne 0 ]; then
    echo ""
    echo "[!] WaferTools exited with an error"
    read -p "Press Enter to close..."
fi


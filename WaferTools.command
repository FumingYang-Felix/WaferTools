#!/bin/bash
# WaferTools V3 macOS Launcher
# Double-click this file to start WaferTools

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Print header
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         WaferTools V3 - macOS Launcher                     ║"
echo "║         Harvard University Lichtman Lab                    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    echo "📥 Please install Python 3 from https://www.python.org"
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ] && [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found"
    echo "🔧 Creating virtual environment..."
    python3 -m venv .venv
    
    echo "📦 Installing dependencies..."
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt 2>/dev/null || pip install -r offline/requirements_offline.txt
else
    # Activate existing venv
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    else
        source venv/bin/activate
    fi
fi

echo "✅ Python environment ready"
echo ""

# Run the launcher
python3 launcher.py

# Keep terminal open on error
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ WaferTools exited with an error"
    read -p "Press Enter to close..."
fi


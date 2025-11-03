#!/bin/bash
# Create a macOS application with custom icon for WaferTools

APP_NAME="WaferTools"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
APP_PATH="$SCRIPT_DIR/${APP_NAME}.app"
ICON_PATH="$SCRIPT_DIR/WaferTools.icns"

echo "Creating WaferTools macOS Application..."
echo "=========================================="

# Remove existing app if exists
if [ -d "$APP_PATH" ]; then
    echo "[*] Removing existing app..."
    rm -rf "$APP_PATH"
fi

# Create AppleScript that launches the Python launcher
cat > /tmp/wafertools_launcher.applescript << 'APPLESCRIPT_END'
on run
    tell application "Terminal"
        activate
        do script "cd " & quoted form of POSIX path of (path to me) & "/../.. && python3 launcher.py"
    end tell
end run
APPLESCRIPT_END

# Compile AppleScript to application
echo "[*] Compiling AppleScript..."
osacompile -o "$APP_PATH" /tmp/wafertools_launcher.applescript

if [ $? -eq 0 ]; then
    echo "[+] Application created successfully"
    
    # Add custom icon if it exists
    if [ -f "$ICON_PATH" ]; then
        echo "[*] Adding custom icon..."
        
        # Copy icon to Resources folder
        RESOURCES_DIR="$APP_PATH/Contents/Resources"
        if [ -d "$RESOURCES_DIR" ]; then
            cp "$ICON_PATH" "$RESOURCES_DIR/applet.icns"
            echo "[+] Custom icon added"
        fi
        
        # Update Info.plist to use custom icon
        PLIST_PATH="$APP_PATH/Contents/Info.plist"
        if [ -f "$PLIST_PATH" ]; then
            /usr/libexec/PlistBuddy -c "Set :CFBundleIconFile applet.icns" "$PLIST_PATH" 2>/dev/null
            /usr/libexec/PlistBuddy -c "Set :CFBundleName WaferTools" "$PLIST_PATH" 2>/dev/null
            /usr/libexec/PlistBuddy -c "Set :CFBundleDisplayName WaferTools V3" "$PLIST_PATH" 2>/dev/null
        fi
        
        # Clear icon cache
        touch "$APP_PATH"
        
    else
        echo "[!] Icon file not found: $ICON_PATH"
        echo "    Run: python3 create_app_icon.py first"
    fi
    
    echo ""
    echo "=========================================="
    echo "[+] Done! You can now:"
    echo "    1. Double-click WaferTools.app to launch"
    echo "    2. Drag it to /Applications"
    echo "    3. Add it to Dock"
    echo "=========================================="
else
    echo "[!] Failed to create application"
    exit 1
fi

# Cleanup
rm -f /tmp/wafertools_launcher.applescript

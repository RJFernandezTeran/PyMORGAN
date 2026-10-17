#!/usr/bin/env bash
set -e

# ============================================================================ #
# PyMORGAN Suite macOS DMG and App Builder                                     #
# ============================================================================ #

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$REPO_ROOT/dist"
OUTPUT_DIR="$REPO_ROOT/dist_installer"
STAGING_DIR="$REPO_ROOT/build_macos_staging"

echo "======================================================================"
echo "  Building PyMORGAN Suite for macOS"
echo "======================================================================"

mkdir -p "$OUTPUT_DIR"
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"

PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
fi
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

VERSION=$("$PYTHON_BIN" -c "import pymorgan; print(getattr(pymorgan, '__version__', '1.0.0'))")
echo "Building version: v$VERSION"

# 1. Convert PNG to ICNS if iconutil is available
ICON_PNG="$REPO_ROOT/src/pymorgan/gui/icons/pirate-hat.png"
ICNS_FILE="$REPO_ROOT/src/pymorgan/gui/icons/pymorgan.icns"
if command -v iconutil >/dev/null 2>&1 && [ -f "$ICON_PNG" ] && [ ! -f "$ICNS_FILE" ]; then
    echo "Creating macOS .icns icon..."
    ICONSET="$REPO_ROOT/src/pymorgan/gui/icons/pymorgan.iconset"
    rm -rf "$ICONSET"
    mkdir -p "$ICONSET"
    sips -z 16 16     "$ICON_PNG" --out "$ICONSET/icon_16x16.png" >/dev/null 2>&1 || true
    sips -z 32 32     "$ICON_PNG" --out "$ICONSET/icon_16x16@2x.png" >/dev/null 2>&1 || true
    sips -z 32 32     "$ICON_PNG" --out "$ICONSET/icon_32x32.png" >/dev/null 2>&1 || true
    sips -z 64 64     "$ICON_PNG" --out "$ICONSET/icon_32x32@2x.png" >/dev/null 2>&1 || true
    sips -z 128 128   "$ICON_PNG" --out "$ICONSET/icon_128x128.png" >/dev/null 2>&1 || true
    sips -z 256 256   "$ICON_PNG" --out "$ICONSET/icon_128x128@2x.png" >/dev/null 2>&1 || true
    sips -z 256 256   "$ICON_PNG" --out "$ICONSET/icon_256x256.png" >/dev/null 2>&1 || true
    sips -z 512 512   "$ICON_PNG" --out "$ICONSET/icon_256x256@2x.png" >/dev/null 2>&1 || true
    sips -z 512 512   "$ICON_PNG" --out "$ICONSET/icon_512x512.png" >/dev/null 2>&1 || true
    sips -z 1024 1024 "$ICON_PNG" --out "$ICONSET/icon_512x512@2x.png" >/dev/null 2>&1 || true
    iconutil -c icns "$ICONSET" -o "$ICNS_FILE" >/dev/null 2>&1 || echo "Warning: iconutil skipped"
    rm -rf "$ICONSET"
fi

# 2. Run PyInstaller if dist/PyMORGAN_Suite is not already built
if [ ! -d "$DIST_DIR/PyMORGAN_Suite" ] || [ "$1" == "--rebuild" ]; then
    echo "Running PyInstaller on macOS..."
    cd "$REPO_ROOT"
    "$PYTHON_BIN" -m PyInstaller --noconfirm "$REPO_ROOT/installer/PyMORGAN_Suite.spec"
else
    echo "Reusing existing PyInstaller bundle at $DIST_DIR/PyMORGAN_Suite..."
fi

# 3. Create .app bundle wrapper preserving framework symlinks
APP_BUNDLE="$DIST_DIR/PyMORGAN Suite.app"
rm -rf "$APP_BUNDLE"
if [ -d "$DIST_DIR/PyMORGAN_Suite" ]; then
    mkdir -p "$APP_BUNDLE/Contents/MacOS"
    mkdir -p "$APP_BUNDLE/Contents/Resources"
    ditto "$DIST_DIR/PyMORGAN_Suite" "$APP_BUNDLE/Contents/MacOS"
    chmod +x "$APP_BUNDLE/Contents/MacOS/PyMORGAN_Suite" "$APP_BUNDLE/Contents/MacOS/PyMORGAN" 2>/dev/null || true
    if [ -f "$APP_BUNDLE/Contents/MacOS/PyRATE-TA" ]; then
        chmod +x "$APP_BUNDLE/Contents/MacOS/PyRATE-TA"
    fi
    ln -sf "PyMORGAN_Suite" "$APP_BUNDLE/Contents/MacOS/launcher"
    if [ -f "$ICNS_FILE" ]; then
        cp "$ICNS_FILE" "$APP_BUNDLE/Contents/Resources/app.icns"
    fi
    cat <<EOF > "$APP_BUNDLE/Contents/Info.plist"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>PyMORGAN_Suite</string>
    <key>CFBundleIconFile</key>
    <string>app.icns</string>
    <key>CFBundleIdentifier</key>
    <string>org.unige.pymorgan.suite</string>
    <key>CFBundleName</key>
    <string>PyMORGAN Suite</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>${VERSION}</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF
fi

# 4. Prepare DMG Staging
echo "Preparing DMG disk image..."
ditto "$APP_BUNDLE" "$STAGING_DIR/PyMORGAN Suite.app"
ln -s /Applications "$STAGING_DIR/Applications"

DMG_NAME="PyMORGAN_Suite_v${VERSION}_macOS.dmg"
DMG_PATH="$OUTPUT_DIR/$DMG_NAME"
rm -f "$DMG_PATH"

for attempt in 1 2 3 4 5; do
    if hdiutil create -volname "PyMORGAN Suite" -srcfolder "$STAGING_DIR" -ov -format UDZO "$DMG_PATH"; then
        break
    fi
    echo "hdiutil create attempt $attempt failed (Resource busy), retrying in 3s..."
    sleep 3
done

if [ ! -f "$DMG_PATH" ]; then
    echo "Error: Failed to create DMG at $DMG_PATH"
    exit 1
fi

# Also create zip for users who prefer direct unzipping
ZIP_NAME="PyMORGAN_Suite_v${VERSION}_macOS.zip"
echo "Creating $ZIP_NAME..."
ditto -c -k --keepParent "$STAGING_DIR/PyMORGAN Suite.app" "$OUTPUT_DIR/$ZIP_NAME"

rm -rf "$STAGING_DIR"

echo "======================================================================"
echo "  macOS packages built successfully:"
echo "  * $DMG_PATH"
echo "  * $OUTPUT_DIR/$ZIP_NAME"
echo "======================================================================"

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

VERSION=$(uv run python -c "import pymorgan; print(getattr(pymorgan, '__version__', '1.0.0'))")
echo "Building version: v$VERSION"

# 1. Convert PNG to ICNS if iconutil is available
ICON_PNG="$REPO_ROOT/src/pymorgan/gui/icons/pirate-hat.png"
ICNS_FILE="$REPO_ROOT/src/pymorgan/gui/icons/pymorgan.icns"
if command -v iconutil >/dev/null 2>&1 && [ -f "$ICON_PNG" ] && [ ! -f "$ICNS_FILE" ]; then
    echo "Creating macOS .icns icon..."
    ICONSET="$REPO_ROOT/src/pymorgan/gui/icons/pymorgan.iconset"
    mkdir -p "$ICONSET"
    sips -z 16 16     "$ICON_PNG" --out "$ICONSET/icon_16x16.png" >/dev/null
    sips -z 32 32     "$ICON_PNG" --out "$ICONSET/icon_16x16@2x.png" >/dev/null
    sips -z 32 32     "$ICON_PNG" --out "$ICONSET/icon_32x32.png" >/dev/null
    sips -z 64 64     "$ICON_PNG" --out "$ICONSET/icon_32x32@2x.png" >/dev/null
    sips -z 128 128   "$ICON_PNG" --out "$ICONSET/icon_128x128.png" >/dev/null
    sips -z 256 256   "$ICON_PNG" --out "$ICONSET/icon_128x128@2x.png" >/dev/null
    sips -z 256 256   "$ICON_PNG" --out "$ICONSET/icon_256x256.png" >/dev/null
    sips -z 512 512   "$ICON_PNG" --out "$ICONSET/icon_256x256@2x.png" >/dev/null
    sips -z 512 512   "$ICON_PNG" --out "$ICONSET/icon_512x512.png" >/dev/null
    iconutil -c icns "$ICONSET" -o "$ICNS_FILE"
    rm -rf "$ICONSET"
fi

# 2. Run PyInstaller
echo "Running PyInstaller on macOS..."
cd "$REPO_ROOT"
uv run --with pyinstaller pyinstaller --noconfirm "$REPO_ROOT/installer/PyMORGAN_Suite.spec"

# 3. Locate the produced .app
APP_BUNDLE="$DIST_DIR/PyMORGAN Suite.app"
if [ ! -d "$APP_BUNDLE" ]; then
    # If built into directory, create standard .app wrapper
    if [ -d "$DIST_DIR/PyMORGAN_Suite" ]; then
        mkdir -p "$APP_BUNDLE/Contents/MacOS"
        mkdir -p "$APP_BUNDLE/Contents/Resources"
        cp -r "$DIST_DIR/PyMORGAN_Suite/"* "$APP_BUNDLE/Contents/MacOS/"
        ln -sf "$APP_BUNDLE/Contents/MacOS/PyMORGAN_Suite" "$APP_BUNDLE/Contents/MacOS/launcher"
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
fi

# 4. Prepare DMG Staging
echo "Preparing DMG disk image..."
cp -r "$APP_BUNDLE" "$STAGING_DIR/"
ln -s /Applications "$STAGING_DIR/Applications"

DMG_NAME="PyMORGAN_Suite_v${VERSION}_macOS.dmg"
DMG_PATH="$OUTPUT_DIR/$DMG_NAME"
rm -f "$DMG_PATH"

hdiutil create -volname "PyMORGAN Suite" -srcfolder "$STAGING_DIR" -ov -format UDZO "$DMG_PATH"

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

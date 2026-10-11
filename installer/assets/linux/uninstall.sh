#!/usr/bin/env bash
set -e

# ============================================================================ #
# PyMORGAN Suite Linux Uninstaller                                             #
# ============================================================================ #

APP_NAME="PyMORGAN Suite"
INSTALL_DIR="${1:-$HOME/.local/share/PyMORGAN_Suite}"
BIN_DIR="$HOME/.local/bin"
DESKTOP_DIR="$HOME/.local/share/applications"
ICON_DIR="$HOME/.local/share/icons/hicolor/256x256/apps"

echo "Uninstalling $APP_NAME..."

# Remove symlinks
rm -f "$BIN_DIR/pymorgan-suite"
rm -f "$BIN_DIR/pymorgan"
rm -f "$BIN_DIR/pyrate-ta"

# Remove desktop files
rm -f "$DESKTOP_DIR/pymorgan-suite.desktop"
rm -f "$DESKTOP_DIR/pymorgan.desktop"
rm -f "$DESKTOP_DIR/pyrate-ta.desktop"

# Remove icons
rm -f "$ICON_DIR/pymorgan.png"
rm -f "$ICON_DIR/pyrate_ta.png"

# Remove directory
if [ -d "$INSTALL_DIR" ]; then
    rm -rf "$INSTALL_DIR"
fi

if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$DESKTOP_DIR" || true
fi

echo "$APP_NAME successfully uninstalled."

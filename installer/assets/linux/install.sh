#!/usr/bin/env bash
set -e

# ============================================================================ #
# PyMORGAN Suite Linux Installer                                               #
# ============================================================================ #

APP_NAME="PyMORGAN Suite"
DEFAULT_INSTALL_DIR="$HOME/.local/share/PyMORGAN_Suite"
BIN_DIR="$HOME/.local/bin"
DESKTOP_DIR="$HOME/.local/share/applications"
ICON_DIR="$HOME/.local/share/icons/hicolor/256x256/apps"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================================================"
echo "  $APP_NAME — Linux Installer"
echo "======================================================================"
echo ""

INSTALL_DIR="${1:-$DEFAULT_INSTALL_DIR}"
echo "Installing to: $INSTALL_DIR"

mkdir -p "$INSTALL_DIR"
mkdir -p "$BIN_DIR"
mkdir -p "$DESKTOP_DIR"
mkdir -p "$ICON_DIR"

# Copy binary payload
echo "Copying application binaries..."
cp -r "$SCRIPT_DIR/bin/"* "$INSTALL_DIR/"

# Install icons
echo "Installing application icons..."
if [ -f "$INSTALL_DIR/icons/pirate-hat.png" ]; then
    cp "$INSTALL_DIR/icons/pirate-hat.png" "$ICON_DIR/pymorgan.png"
fi
if [ -f "$INSTALL_DIR/icons/pirate_ship.png" ]; then
    cp "$INSTALL_DIR/icons/pirate_ship.png" "$ICON_DIR/pyrate_ta.png"
fi

# Install desktop menu launchers
echo "Creating application launchers..."
for desktop_file in "$SCRIPT_DIR/desktop/"*.desktop; do
    if [ -f "$desktop_file" ]; then
        dest="$DESKTOP_DIR/$(basename "$desktop_file")"
        sed "s|DIR_PLACEHOLDER|$INSTALL_DIR|g" "$desktop_file" > "$dest"
        sed -i "s|ICON_PYMORGAN|$ICON_DIR/pymorgan.png|g" "$dest" 2>/dev/null || true
        sed -i "s|ICON_PYRATE|$ICON_DIR/pyrate_ta.png|g" "$dest" 2>/dev/null || true
        chmod +x "$dest"
    fi
done

# Create command-line symlinks
echo "Creating command-line symlinks in $BIN_DIR..."
ln -sf "$INSTALL_DIR/PyMORGAN_Suite" "$BIN_DIR/pymorgan-suite"
ln -sf "$INSTALL_DIR/PyMORGAN" "$BIN_DIR/pymorgan"
if [ -f "$INSTALL_DIR/PyRATE-TA" ]; then
    ln -sf "$INSTALL_DIR/PyRATE-TA" "$BIN_DIR/pyrate-ta"
fi

# Refresh desktop database
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$DESKTOP_DIR" || true
fi
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
    gtk-update-icon-cache -f -t "$HOME/.local/share/icons/hicolor" >/dev/null 2>&1 || true
fi

echo ""
echo "======================================================================"
echo "  Installation complete!"
echo "  You can launch the applications from your desktop menu or running:"
echo "    pymorgan-suite    (Unified Suite Launcher)"
echo "    pymorgan          (PyMORGAN 1D & 2D Processing)"
echo "    pyrate-ta         (PyRATE-TA Global Fitting)"
echo "======================================================================"

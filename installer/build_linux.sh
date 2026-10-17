#!/usr/bin/env bash
set -e

# ============================================================================ #
# PyMORGAN Suite Linux Package Builder                                         #
# ============================================================================ #

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$REPO_ROOT/dist/PyMORGAN_Suite"
OUTPUT_DIR="$REPO_ROOT/dist_installer"
BUILD_TMP="$REPO_ROOT/build_linux_pkg"

echo "======================================================================"
echo "  Building PyMORGAN Suite for Linux (x86_64)"
echo "======================================================================"

# 1. Clean previous build temp
rm -rf "$BUILD_TMP"
mkdir -p "$BUILD_TMP"
mkdir -p "$OUTPUT_DIR"

# 2. Extract version
PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
fi
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

VERSION=$("$PYTHON_BIN" -c "import pymorgan; print(getattr(pymorgan, '__version__', '1.0.0'))")
echo "Building version: v$VERSION"

# 3. Run PyInstaller if dist not already present or --rebuild requested
if [ ! -d "$DIST_DIR" ] || [ "$1" == "--rebuild" ]; then
    echo "Running PyInstaller..."
    cd "$REPO_ROOT"
    "$PYTHON_BIN" -m PyInstaller --noconfirm "$REPO_ROOT/installer/PyMORGAN_Suite.spec"
fi

# 4. Assemble distribution layout
PKG_DIR="$BUILD_TMP/PyMORGAN_Suite_v${VERSION}_Linux_x86_64"
mkdir -p "$PKG_DIR/bin"
mkdir -p "$PKG_DIR/desktop"

echo "Assembling package files..."
cp -r "$DIST_DIR/"* "$PKG_DIR/bin/"
cp "$REPO_ROOT/installer/assets/linux/install.sh" "$PKG_DIR/"
cp "$REPO_ROOT/installer/assets/linux/uninstall.sh" "$PKG_DIR/"
cp "$REPO_ROOT/installer/assets/linux/"*.desktop "$PKG_DIR/desktop/"
chmod +x "$PKG_DIR/install.sh" "$PKG_DIR/uninstall.sh"
chmod +x "$PKG_DIR/bin/PyMORGAN_Suite" "$PKG_DIR/bin/PyMORGAN" 2>/dev/null || true
if [ -f "$PKG_DIR/bin/PyRATE-TA" ]; then
    chmod +x "$PKG_DIR/bin/PyRATE-TA"
fi

# 5. Create tar.gz distribution archive
ARCHIVE_NAME="PyMORGAN_Suite_v${VERSION}_Linux_x86_64.tar.gz"
echo "Creating $ARCHIVE_NAME..."
cd "$BUILD_TMP"
tar -czf "$OUTPUT_DIR/$ARCHIVE_NAME" "PyMORGAN_Suite_v${VERSION}_Linux_x86_64"

# Clean up
rm -rf "$BUILD_TMP"

echo "======================================================================"
echo "  Linux package built successfully:"
echo "  $OUTPUT_DIR/$ARCHIVE_NAME"
echo "======================================================================"

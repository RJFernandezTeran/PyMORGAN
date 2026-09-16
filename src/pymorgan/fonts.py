"""Font installation helper for PyMORGAN.

Locates TeX Gyre Heros fonts from a MiKTeX / TeX Live installation or the
rinoh-typeface-texgyreheros package and copies them into Matplotlib's font
directory, then clears the font cache.

Usage
-----
After installing PyMORGAN::

    pymorgan-install-fonts

Or from Python::

    from pymorgan.fonts import install_fonts
    install_fonts()
"""

from __future__ import annotations

import shutil
from functools import cache
from pathlib import Path

# Preferred fonts, in priority order.
PREFERRED_FONTS = ["TeX Gyre Heros", "Helvetica"]


@cache
def available_preferred_fonts() -> list[str]:
    """Return which of PREFERRED_FONTS Matplotlib can currently resolve."""
    import matplotlib.font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    return [name for name in PREFERRED_FONTS if name in available]


def _find_miktex_heros() -> list[Path]:
    """Search common MiKTeX / TeX Live locations for TeX Gyre Heros font files."""
    home = Path.home()
    search_roots = [
        # MiKTeX — system-wide and per-user
        Path(r"C:\Program Files\MiKTeX"),
        Path(r"C:\Program Files (x86)\MiKTeX"),
        home / "AppData" / "Local" / "Programs" / "MiKTeX",
        home / "AppData" / "Roaming" / "MiKTeX",
        # TeX Live — common install paths
        Path(r"C:\texlive"),
        Path("/usr/share/texmf"),
        Path("/usr/local/texlive"),
        home / "texlive",
    ]
    patterns = ["**/texgyreheros-*.otf", "**/texgyreheros-*.ttf"]
    found: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            found.extend(root.glob(pattern))
    return found


def _find_rinoh_heros() -> list[Path]:
    """Search rinoh-typeface-texgyreheros package location for TeX Gyre Heros font files."""
    try:
        import rinoh_typeface_texgyreheros as tgh
    except ImportError:
        return []

    pkg_dir = Path(tgh.__file__).parent
    patterns = ["**/texgyreheros-*.otf", "**/texgyreheros-*.ttf"]
    found: list[Path] = []
    for pattern in patterns:
        found.extend(pkg_dir.glob(pattern))
    return found


def find_heros_fonts() -> list[Path]:
    """Locate TeX Gyre Heros font files.

    First searches system LaTeX installations (MiKTeX, TeX Live).
    If none are found, falls back to the rinoh-typeface-texgyreheros package.
    """
    fonts = _find_miktex_heros()
    if fonts:
        return fonts
    return _find_rinoh_heros()


def install_fonts(quiet: bool = False) -> None:
    """Copy TeX Gyre Heros fonts into Matplotlib's font directory and clear the cache."""
    import matplotlib
    import matplotlib.font_manager as fm

    fonts_miktex = _find_miktex_heros()
    if fonts_miktex:
        fonts = fonts_miktex
        source = "MiKTeX / TeX Live installation"
    else:
        fonts = _find_rinoh_heros()
        source = "rinoh-typeface-texgyreheros package"

    if not fonts:
        if not quiet:
            print(
                "No TeX Gyre Heros fonts found.\n"
                "Check that MiKTeX / TeX Live or `rinoh-typeface-texgyreheros` is installed, then re-run:\n"
                "    pymorgan-install-fonts"
            )
        return

    if not quiet:
        print(f"Found TeX Gyre Heros fonts from {source}.")
    mpl_font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    try:
        mpl_font_dir.mkdir(parents=True, exist_ok=True)
        if not quiet:
            print(f"Installing to: {mpl_font_dir}")

        for src in fonts:
            dst = mpl_font_dir / src.name
            shutil.copy2(src, dst)
            if not quiet:
                print(f"  Copied: {src.name}")
    except (OSError, PermissionError) as exc:
        if not quiet:
            print(f"Could not copy font files to {mpl_font_dir}: {exc}")

    # Register directly with Matplotlib's font manager in this session
    for font_file in fonts:
        try:
            fm.fontManager.addfont(str(font_file))
        except Exception:
            pass

    # Clear font cache so Matplotlib picks up the new files.
    try:
        cache_dir = Path(matplotlib.get_cachedir())
        for f in cache_dir.glob("fontlist-*.json"):
            try:
                f.unlink()
                if not quiet:
                    print(f"  Deleted cache: {f.name}")
            except OSError:
                pass
    except Exception:
        pass

    # Rebuild in this session.
    try:
        fm._load_fontmanager(try_read_cache=False)
    except Exception:
        pass
    available_preferred_fonts.cache_clear()

    # Record receipt flag in cache dir for fast subsequent checks
    try:
        flag = Path(matplotlib.get_cachedir()) / "pymorgan_fonts_installed"
        flag.touch()
    except Exception:
        pass

    found = [f.name for f in fm.fontManager.ttflist if "Heros" in f.name]
    if not quiet:
        if found:
            print(f"\nSuccess — Matplotlib now sees: {found}")
        else:
            print("\nFonts copied but not yet detected. Restart your Python session.")


def are_fonts_installed() -> bool:
    """Check whether TeX Gyre Heros fonts have already been installed for Matplotlib.

    Performs a fast multi-stage check:
    1. Checks for a receipt flag in Matplotlib's cache directory (< 0.1 ms).
    2. Checks whether TeX Gyre Heros files exist in Matplotlib's fonts directory (< 1 ms).
    3. Checks whether TeX Gyre Heros is loaded in Matplotlib's active font manager.
    """
    try:
        import matplotlib

        flag = Path(matplotlib.get_cachedir()) / "pymorgan_fonts_installed"
        if flag.exists():
            return True

        mpl_font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        if mpl_font_dir.exists():
            if any(mpl_font_dir.glob("*heros*.otf")) or any(mpl_font_dir.glob("*heros*.ttf")):
                try:
                    flag.touch()
                except Exception:
                    pass
                return True
    except Exception:
        pass

    try:
        import matplotlib.font_manager as fm

        if any("Heros" in f.name for f in fm.fontManager.ttflist):
            return True
    except Exception:
        pass

    return False


def ensure_fonts_installed(quiet: bool = True) -> bool:
    """Ensure TeX Gyre Heros fonts are available in Matplotlib.

    If fonts are already verified, does nothing. Otherwise, installs them from
    the bundled `rinoh-typeface-texgyreheros` dependency or system TeX
    installations.

    Returns True if at least one preferred font is available, False otherwise.
    """
    if are_fonts_installed():
        return True

    try:
        install_fonts(quiet=quiet)
    except Exception:
        pass

    return are_fonts_installed()


def main() -> None:
    install_fonts(quiet=False)


if __name__ == "__main__":
    main()

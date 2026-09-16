"""PyMORGAN startup launcher.

Provides the smart entrypoint that checks whether the TeX Gyre Heros fonts
have been installed in Matplotlib. On the very first run, it installs them
automatically without requiring manual user setup; on all subsequent runs,
the check completes in fractions of a millisecond and launches the GUI directly.
"""

from __future__ import annotations

import sys

from .fonts import are_fonts_installed, install_fonts


def main(argv: list[str] | None = None) -> int:
    """Run font check-and-install on first run, then launch the PyMORGAN GUI."""
    if not are_fonts_installed():
        print("[PyMORGAN] First-time startup: TeX Gyre Heros fonts not detected.")
        print("[PyMORGAN] Automatically configuring fonts into Matplotlib...")
        try:
            install_fonts(quiet=False)
            print("[PyMORGAN] Font configuration complete.\n")
        except Exception as exc:
            print(f"[PyMORGAN] Warning: font installation failed ({exc}). Continuing...\n")

    from .gui.app import main as gui_main

    return gui_main(argv)


if __name__ == "__main__":
    sys.exit(main())

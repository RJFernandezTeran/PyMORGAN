"""Entrypoint for standalone PyMORGAN Suite Launcher executable."""

import sys

from pymorgan.gui.suite_launcher import main

if __name__ == "__main__":
    sys.exit(main())

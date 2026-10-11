"""Entrypoint for standalone PyMORGAN executable."""

import sys

from pymorgan.launcher import main

if __name__ == "__main__":
    sys.exit(main())

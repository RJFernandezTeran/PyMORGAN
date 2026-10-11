"""Entrypoint for standalone PyRATE-TA executable."""

import sys

from pyrate_ta.launcher import main

if __name__ == "__main__":
    sys.exit(main())

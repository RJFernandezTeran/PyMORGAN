"""Package execution entry point: python -m pymorgan launches the GUI via launcher."""

from __future__ import annotations

import sys

from pymorgan.launcher import main

if __name__ == "__main__":
    sys.exit(main())

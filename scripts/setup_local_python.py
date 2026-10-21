#!/usr/bin/env python3
"""Standalone setup script to enable ``import pymorgan`` and ``import pyrate_ta`` in any Python >= 3.11 environment.

Usage:
    # Configure the current Python environment (Conda, virtualenv, or system Python):
    python scripts/setup_local_python.py --python current

    # Auto-detect and configure the primary Python environment on this machine:
    python scripts/setup_local_python.py

    # List all detected Python >= 3.11 environments:
    python scripts/setup_local_python.py --list
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> int:
    from pymorgan.gui.python_env_setup import cli_main

    argv = list(sys.argv[1:])
    if "--python" in argv:
        idx = argv.index("--python")
        if idx + 1 < len(argv) and argv[idx + 1].lower() in ("current", "self"):
            argv[idx + 1] = sys.executable
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

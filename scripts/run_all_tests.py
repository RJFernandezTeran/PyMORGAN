"""Unified test runner script for PyMORGAN and PyRATE-TA.

Usage:
    uv run python scripts/run_all_tests.py
    uv run python scripts/run_all_tests.py --pymorgan
    uv run python scripts/run_all_tests.py --pyrate
    uv run python scripts/run_all_tests.py -k "stitch"
"""

import sys
from pathlib import Path

# Ensure in-tree pymorgan is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pymorgan.test_runner import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

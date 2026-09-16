#!/usr/bin/env bash
set -e

# Change to script directory
cd "$(dirname "$0")"

# Check for existing Python environment (.venv, uv, python3, python)
if [ -f ".venv/bin/python" ]; then
    exec .venv/bin/python -m pymorgan "$@"
elif command -v uv >/dev/null 2>&1; then
    exec uv run python -m pymorgan "$@"
elif command -v python3 >/dev/null 2>&1; then
    exec python3 -m pymorgan "$@"
elif command -v python >/dev/null 2>&1; then
    exec python -m pymorgan "$@"
else
    echo "[ERROR] No Python interpreter was found on your system." >&2
    echo "Please install Python (>= 3.12) or uv." >&2
    exit 1
fi

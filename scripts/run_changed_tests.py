#!/usr/bin/env python3
"""Adaptive test runner for PyMORGAN.

Detects modified files in the git working tree and runs only the relevant subset
of tests to speed up verification. Runs all tests if --all is passed or if
global/configuration files were modified.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

# Map source directories or files to specific test suites
RULES: list[tuple[str, list[str]]] = [
    ("src/pymorgan/gui/", ["tests/test_gui.py", "tests/test_theme.py", "tests/test_picker.py"]),
    ("src/pymorgan/oneD/chirp.py", ["tests/test_chirp.py"]),
    ("src/pymorgan/oneD/process.py", ["tests/test_process.py"]),
    ("src/pymorgan/oneD/load.py", ["tests/test_load.py", "tests/test_mess.py"]),
    ("src/pymorgan/oneD/plot.py", ["tests/test_plot.py"]),
    ("src/pymorgan/oneD/dataset.py", ["tests/test_process.py", "tests/test_load.py"]),
    ("src/pymorgan/twoD/", ["tests/test_twoD.py"]),
    ("src/pymorgan/steadyState/", ["tests/test_steadystate.py"]),
    ("src/pymorgan/settings.py", ["tests/test_settings.py", "tests/test_theme.py"]),
    ("src/pymorgan/display.py", ["tests/test_display.py"]),
    ("src/pymorgan/helpers.py", ["tests/test_settings.py", "tests/test_theme.py", "tests/test_legend_pruning.py"]),
]

GLOBAL_FILES = {
    "pyproject.toml",
    "src/pymorgan/__init__.py",
    "tests/conftest.py",
    "tests/synthetic.py",
}


def get_changed_files() -> list[str] | None:
    """Retrieve lists of files modified in index or working tree relative to HEAD.

    Returns None if git command fails (e.g. not a git repository).
    """
    try:
        # Get list of modified files relative to HEAD (covers staged, unstaged, and untracked)
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        files = []
        for line in res.stdout.splitlines():
            if not line.strip():
                continue
            # status is the first two characters (e.g. " M", "M ", "??")
            file_path = line[3:].strip()
            # If renamed, take the destination path
            if " -> " in file_path:
                file_path = file_path.split(" -> ")[1].strip()
            files.append(file_path)
        return files
    except Exception as exc:
        print(f"Warning: Failed to run git status (not a git repo?): {exc}")
        return None


def resolve_tests(changed_files: list[str]) -> set[str]:
    """Map changed file paths to test files."""
    tests_to_run: set[str] = set()

    for f in changed_files:
        # Normalise to forward slashes for cross-platform matches
        f_norm = f.replace("\\", "/")

        # Global/Core files trigger the full suite
        if f_norm in GLOBAL_FILES:
            print(f"Global configuration/utility file modified: {f_norm}. Running full suite.")
            return set()

        # If a test file itself is changed, run it directly
        if f_norm.startswith("tests/test_") and f_norm.endswith(".py"):
            tests_to_run.add(f_norm)
            continue

        # Match against mapped components
        matched = False
        for prefix, test_list in RULES:
            if f_norm.startswith(prefix):
                tests_to_run.update(test_list)
                matched = True

        if not matched and (f_norm.startswith("src/") or f_norm.startswith("tests/")):
            # Fallback if inside src or tests but not matching a rule: run full suite
            print(f"Unmapped change in source/test tree: {f_norm}. Falling back to full suite.")
            return set()

    return tests_to_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Adaptive test runner for PyMORGAN.")
    parser.add_argument(
        "--all", action="store_true", help="Force running the full test suite."
    )
    args, pytest_args = parser.parse_known_args()

    # If --all is passed, run everything
    if args.all:
        print("Forcing full test suite run.")
        cmd = [sys.executable, "-m", "pytest"] + pytest_args
        return subprocess.run(cmd).returncode

    changed_files = get_changed_files()
    if changed_files is None:
        print("Not a git repository. Running full test suite.")
        cmd = [sys.executable, "-m", "pytest"] + pytest_args
        return subprocess.run(cmd).returncode

    if not changed_files:
        print("No changed files detected. Running quick checks (run_checks.py).")
        cmd = [sys.executable, "scripts/run_checks.py"]
        return subprocess.run(cmd).returncode

    targets = resolve_tests(changed_files)
    if not targets:
        print("Changes require running the full test suite.")
        cmd = [sys.executable, "-m", "pytest"] + pytest_args
    else:
        print(f"Running tests for changed files: {sorted(targets)}")
        cmd = [sys.executable, "-m", "pytest"] + sorted(targets) + pytest_args

    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())

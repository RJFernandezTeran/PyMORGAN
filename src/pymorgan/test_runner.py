"""Unified test runner for PyMORGAN and PyRATE-TA test suites.

Executes test suites for both packages, collects statistics, and outputs
a unified summary report.

Usage:
    uv run pymorgan-run-tests
    python scripts/run_all_tests.py [--pymorgan | --pyrate] [pytest options]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Match pytest summary line like: "280 passed, 6 skipped, 4 warnings in 19.22s"
_SUMMARY_RE = re.compile(
    r"(?:(?P<passed>\d+)\s+passed)?"
    r"(?:[,\s]+(?P<skipped>\d+)\s+skipped)?"
    r"(?:[,\s]+(?P<failed>\d+)\s+failed)?"
    r"(?:[,\s]+(?P<errors>\d+)\s+error)?"
    r".*?in\s+(?P<duration>[\d\.]+)s",
    re.IGNORECASE,
)


def _find_repo_roots() -> tuple[Path, Path | None]:
    """Locate PyMORGAN root and PyRATE-TA sibling repository root."""
    pymorgan_root = Path(__file__).resolve().parent.parent.parent
    # Fallback if installed in site-packages
    if not (pymorgan_root / "tests").is_dir():
        pymorgan_root = Path.cwd()

    pyrate_root = pymorgan_root.parent / "PyRATE-TA"
    if not (pyrate_root / "tests").is_dir():
        # Check current directory sibling
        cwd_sibling = Path.cwd().parent / "PyRATE-TA"
        if (cwd_sibling / "tests").is_dir():
            pyrate_root = cwd_sibling
        else:
            pyrate_root = None

    return pymorgan_root, pyrate_root


def _run_suite(
    suite_name: str,
    cwd: Path,
    extra_args: list[str] | None = None,
) -> tuple[int, int, int, int, float, str]:
    """Run pytest for a single package and extract test statistics.

    Returns:
        (passed, skipped, failed, errors, duration, stdout_text)
    """
    cmd = ["uv", "run", "pytest", "tests"]
    if extra_args:
        cmd.extend(extra_args)

    print("=" * 72)
    print(f"  Running {suite_name} Test Suite")
    print(f"  Directory: {cwd}")
    print(f"  Command:   {' '.join(cmd)}")
    print("=" * 72)

    t0 = time.perf_counter()
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    # Remove VIRTUAL_ENV from inherited env so uv targets the directory's own .venv
    env.pop("VIRTUAL_ENV", None)

    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    elapsed = time.perf_counter() - t0

    # Stream output to console
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)

    passed = 0
    skipped = 0
    failed = 0
    errors = 0
    duration = elapsed

    # Parse pytest summary
    for line in proc.stdout.splitlines():
        line_clean = line.strip(" =")
        m = _SUMMARY_RE.search(line_clean)
        if m:
            groups = m.groupdict()
            if groups.get("passed"):
                passed = int(groups["passed"])
            if groups.get("skipped"):
                skipped = int(groups["skipped"])
            if groups.get("failed"):
                failed = int(groups["failed"])
            if groups.get("errors"):
                errors = int(groups["errors"])
            if groups.get("duration"):
                try:
                    duration = float(groups["duration"])
                except ValueError:
                    pass

    # If pytest failed to exit 0 but parsed no failures (e.g. collection error)
    if proc.returncode != 0 and failed == 0 and errors == 0:
        failed = 1

    return passed, skipped, failed, errors, duration, proc.stdout


def main(argv: list[str] | None = None) -> int:
    """Unified test runner entry point."""
    parser = argparse.ArgumentParser(
        description="Unified Test Runner for PyMORGAN and PyRATE-TA",
        add_help=True,
    )
    parser.add_argument(
        "--pymorgan",
        action="store_true",
        help="Run only PyMORGAN tests",
    )
    parser.add_argument(
        "--pyrate",
        action="store_true",
        help="Run only PyRATE-TA tests",
    )
    args, unknown = parser.parse_known_args(argv)

    pymorgan_root, pyrate_root = _find_repo_roots()

    suites_to_run: list[tuple[str, Path]] = []
    if args.pymorgan:
        suites_to_run.append(("PyMORGAN", pymorgan_root))
    elif args.pyrate:
        if pyrate_root is None:
            print("Error: PyRATE-TA repository not found.", file=sys.stderr)
            return 1
        suites_to_run.append(("PyRATE-TA", pyrate_root))
    else:
        suites_to_run.append(("PyMORGAN", pymorgan_root))
        if pyrate_root is not None:
            suites_to_run.append(("PyRATE-TA", pyrate_root))

    results = []
    total_passed = 0
    total_skipped = 0
    total_failed = 0
    total_errors = 0
    t_start_all = time.perf_counter()

    for name, root in suites_to_run:
        p, s, f, e, dur, _ = _run_suite(name, root, unknown)
        status = "PASS" if (f == 0 and e == 0) else "FAIL"
        results.append({
            "name": name,
            "passed": p,
            "skipped": s,
            "failed": f,
            "errors": e,
            "duration": dur,
            "status": status,
        })
        total_passed += p
        total_skipped += s
        total_failed += f
        total_errors += e

    total_time = time.perf_counter() - t_start_all

    # Print summary box
    print("\n" + "=" * 72)
    print("                    PyMORGAN & PyRATE-TA Unified Test Summary")
    print("=" * 72)
    print(f"{'Package':<14} {'Passed':<8} {'Skipped':<9} {'Failed':<8} {'Time':<10} {'Status'}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['name']:<14} {r['passed']:<8} {r['skipped']:<9} {r['failed'] + r['errors']:<8} "
            f"{r['duration']:>5.2f}s     {r['status']}"
        )
    print("-" * 72)
    overall_status = "ALL PASSED" if (total_failed == 0 and total_errors == 0) else "SOME FAILED"
    print(
        f"{'TOTAL':<14} {total_passed:<8} {total_skipped:<9} {total_failed + total_errors:<8} "
        f"{total_time:>5.2f}s     {overall_status}"
    )
    print("=" * 72 + "\n")

    return 0 if (total_failed == 0 and total_errors == 0) else 1


if __name__ == "__main__":
    sys.exit(main())

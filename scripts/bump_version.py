"""Bump the PyMORGAN version in ``src/pymorgan/__about__.py``.

The scheme is ``0.<minor>.<yymmdd>.dev<N>`` (PEP 440): ``yymmdd`` is the build
date and ``N`` the iteration within that day, starting at 1.

Run::

    python scripts/bump_version.py            # next build today  -> .dev(N+1)
    python scripts/bump_version.py --minor    # 0.5.x -> 0.6.<today>.dev1
    python scripts/bump_version.py --set 0.6.260801.dev1
    python scripts/bump_version.py --check    # verify only, exit 1 if stale
    python scripts/bump_version.py --dry-run  # print the next version, write nothing

The counter restarts at ``dev1`` whenever the date changes, so the first build
of a day is ``0.5.<today>.dev1``. ``--check`` is meant for a pre-commit hook: it
fails when the recorded version was not produced today, i.e. when an edit went
in without a bump.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

ABOUT = Path(__file__).resolve().parents[1] / "src" / "pymorgan" / "__about__.py"

# 0.<minor>.<yymmdd>.dev<N>
_VERSION_RE = re.compile(r'^__version__\s*=\s*"(?P<version>[^"]+)"', re.M)
_PARTS_RE = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<stamp>\d{6})\.dev(?P<dev>\d+)$")


def read_version(path: Path = ABOUT) -> str:
    """Current version string, as written in ``__about__.py``."""
    match = _VERSION_RE.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise SystemExit(f"no __version__ found in {path}")
    return match.group("version")


def parse(version: str) -> tuple[int, int, str, int]:
    """Split ``0.5.260727.dev3`` into ``(0, 5, "260727", 3)``."""
    match = _PARTS_RE.match(version)
    if match is None:
        raise SystemExit(
            f"version {version!r} does not match 0.<minor>.<yymmdd>.dev<N>; "
            "pass --set to fix it explicitly"
        )

    return (
        int(match.group("major")),
        int(match.group("minor")),
        match.group("stamp"),
        int(match.group("dev")),
    )


def next_version(current: str, *, minor_bump: bool = False, today: str | None = None) -> str:
    """Next version after ``current``: ``dev+1`` today, ``dev1`` on a new day."""
    major, minor, stamp, dev = parse(current)
    today = today or date.today().strftime("%y%m%d")
    if minor_bump:
        return f"{major}.{minor + 1}.{today}.dev1"
    if stamp != today:
        return f"{major}.{minor}.{today}.dev1"
    return f"{major}.{minor}.{stamp}.dev{dev + 1}"


def write_version(version: str, path: Path = ABOUT) -> None:
    """Replace the ``__version__`` line, leaving the rest of the file alone."""
    text = path.read_text(encoding="utf-8")
    new_text, count = _VERSION_RE.subn(f'__version__ = "{version}"', text, count=1)
    if count != 1:
        raise SystemExit(f"could not rewrite __version__ in {path}")
    path.write_text(new_text, encoding="utf-8", newline="")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--minor", action="store_true", help="bump 0.x -> 0.(x+1), restart at dev1")
    parser.add_argument("--set", dest="explicit", metavar="VERSION", help="set this version exactly")
    parser.add_argument("--check", action="store_true", help="only verify the version is from today")
    parser.add_argument("--dry-run", action="store_true", help="print the next version, write nothing")
    args = parser.parse_args(argv)

    current = read_version()
    today = date.today().strftime("%y%m%d")

    if args.check:
        _, _, stamp, _ = parse(current)
        if stamp != today:
            print(f"version {current} is from {stamp}, not today ({today}) - bump it")
            return 1
        print(f"version {current} is up to date")
        return 0

    if args.explicit:
        parse(args.explicit)  # validate the scheme before writing
        new = args.explicit
    else:
        new = next_version(current, minor_bump=args.minor, today=today)

    if args.dry_run:
        print(f"{current} -> {new} (dry run)")
        return 0

    write_version(new)
    print(f"{current} -> {new}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

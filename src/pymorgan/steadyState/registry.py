"""Loader registry for steady-state spectra (absorption / emission).

A *loader* maps a file path to ``(x, y, x_units)`` where ``x`` and ``y`` are
1-D arrays (spectral axis and signal) and ``x_units`` is a best-guess unit
string for the spectral axis (``"nm"``, ``"cm^{-1}"``, ``"eV"``) or ``None``.
New formats are added with :func:`register_spectrum_loader`.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np

SpectrumLoaderResult = tuple  # (x, y, x_units | None)
SpectrumLoader = Callable[[str], SpectrumLoaderResult]

_LOADERS: dict[str, SpectrumLoader] = {}


def register_spectrum_loader(*names: str) -> Callable[[SpectrumLoader], SpectrumLoader]:
    """Register a steady-state loader under one or more data-type names."""

    def decorator(fn: SpectrumLoader) -> SpectrumLoader:
        for name in names:
            _LOADERS[name] = fn
        return fn

    return decorator


def get_spectrum_loader(name: str) -> SpectrumLoader:
    """Return the loader registered for ``name`` or raise ``KeyError``."""
    try:
        return _LOADERS[name]
    except KeyError:
        raise KeyError(
            f"No steady-state loader registered for {name!r}. "
            f"Available: {available_spectrum_loaders()}"
        ) from None


def available_spectrum_loaders() -> list[str]:
    """Return the sorted list of registered data-type names."""
    return sorted(_LOADERS)


# --------------------------------------------------------------------------- #
#                              Generic X-Y reader                             #
# --------------------------------------------------------------------------- #
_DELIMITER = re.compile(r"[\s,;]+")
_UNIT_HINTS = (
    ("wavenumber", "cm^{-1}"),
    ("cm-1", "cm^{-1}"),
    ("wavelength", "nm"),
    ("energy", "eV"),
    (" ev", "eV"),
)


def _guess_units(header: str | None) -> str | None:
    """Infer the spectral-axis unit from a header line, if recognisable."""
    if not header:
        return None
    low = header.lower()
    for needle, unit in _UNIT_HINTS:
        if needle in low:
            return unit
    return None


@register_spectrum_loader("csv", "txt", "xy")
def read_xy(path: str) -> SpectrumLoaderResult:
    """Read a two-column X-Y text file with an auto-detected delimiter.

    The delimiter is detected per line among whitespace, tab, comma and
    semicolon, so space-, tab- and comma-separated files all parse. Lines that
    are blank, start with ``#`` or do not parse as numbers (e.g. a header) are
    skipped; the last such header line is used to guess the spectral-axis unit.
    """
    rows: list[list[float]] = []
    last_header: str | None = None
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = [p for p in _DELIMITER.split(stripped) if p]
            try:
                values = [float(p) for p in parts]
            except ValueError:
                last_header = stripped
                continue
            if len(values) >= 2:
                rows.append(values[:2])

    if not rows:
        raise ValueError(f"No numeric X-Y data found in {path!r}.")

    data = np.asarray(rows, dtype=float)
    return data[:, 0], data[:, 1], _guess_units(last_header)


@register_spectrum_loader("opus", "ftir")
def read_opus_absorbance(path: str) -> SpectrumLoaderResult:
    """Read absorbance from a Bruker OPUS file via the ``brukeropus`` package."""
    try:
        from brukeropus import read_opus
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Reading OPUS files requires the 'brukeropus' package (pip install brukeropus)."
        ) from exc

    opus = read_opus(path)
    if "a" not in opus.data_keys:
        raise ValueError(f"No absorbance block ('a') found in OPUS file {path!r}.")
    return np.asarray(opus.a.x, dtype=float), np.asarray(opus.a.y, dtype=float), "cm^{-1}"


def _stub_spectrum_loader(name: str) -> SpectrumLoader:
    """Build a placeholder loader that fails loudly until implemented."""

    def loader(path: str) -> SpectrumLoaderResult:
        raise NotImplementedError(
            f"The {name!r} steady-state loader is not implemented yet (requested file: {path})."
        )

    loader.__name__ = f"load_{name}"
    return loader


# Instrument-specific text exports: recognised but not implemented yet.
for _name in ("uvvis", "fluorimeter"):
    register_spectrum_loader(_name)(_stub_spectrum_loader(_name))

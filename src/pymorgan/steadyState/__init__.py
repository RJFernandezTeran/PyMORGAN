"""Unified interface for steady-state spectra (absorption / emission).

Mirrors :mod:`pymorgan.oneD`: a :class:`Spectrum` (and :class:`SpectrumSeries`)
over the raw arrays, with formats resolved through a loader registry and
presentation driven by the shared :class:`pymorgan.settings.Settings`.
"""

from .dataset import Spectrum, SpectrumKind, SpectrumSeries, load_spectrum
from .registry import (
    available_spectrum_loaders,
    get_spectrum_loader,
    register_spectrum_loader,
)

__all__ = [
    "Spectrum",
    "SpectrumSeries",
    "SpectrumKind",
    "load_spectrum",
    "register_spectrum_loader",
    "get_spectrum_loader",
    "available_spectrum_loaders",
]

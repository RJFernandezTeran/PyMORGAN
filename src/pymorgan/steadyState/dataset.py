"""Steady-state spectra: :class:`Spectrum` and :class:`SpectrumSeries`.

These mirror the design of :class:`pymorgan.oneD.Dataset1D`: a thin object over
the raw arrays, formats resolved through a loader registry, and presentation
driven by the shared :class:`pymorgan.settings.Settings`. A :class:`Spectrum`
can be plotted on its own, overlaid as part of a :class:`SpectrumSeries`, or
fed into :meth:`pymorgan.oneD.Dataset1D.plot_spectra` as a steady-state overlay.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from enum import StrEnum
from os import PathLike

import numpy as np

from ..settings import Settings, get_settings
from .registry import get_spectrum_loader

# Spectral-axis unit -> (label, mathtext units) for axis labelling.
_XLABEL: dict[str, tuple[str, str]] = {
    "nm": ("Wavelength", "nm"),
    "cm^{-1}": ("Wavenumber", r"cm$^{-1}$"),
    "cm-1": ("Wavenumber", r"cm$^{-1}$"),
    "eV": ("Energy", "eV"),
}


class SpectrumKind(StrEnum):
    """Kind of steady-state measurement: absorption, emission or excitation."""

    ABSORPTION = "absorption"
    EMISSION = "emission"
    EXCITATION = "excitation"


class Spectrum:
    """A single steady-state spectrum (one X-Y trace plus metadata)."""

    def __init__(
        self,
        x,
        y,
        kind: SpectrumKind | str = SpectrumKind.ABSORPTION,
        *,
        x_units: str = "nm",
        label: str | None = None,
        source: str | None = None,
    ):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.kind = SpectrumKind(kind)
        self.x_units = x_units
        self.label = label
        self.source = source

    # ----------------------------------------------------------------- #
    #                          Construction                             #
    # ----------------------------------------------------------------- #
    @classmethod
    def from_file(
        cls,
        path: str | PathLike,
        kind: SpectrumKind | str = SpectrumKind.ABSORPTION,
        *,
        data_type: str = "csv",
        x_units: str | None = None,
        label: str | None = None,
    ) -> Spectrum:
        """Load a spectrum from ``path`` using the loader for ``data_type``."""
        loader = get_spectrum_loader(data_type)
        x, y, guessed = loader(str(path))
        return cls(
            x,
            y,
            kind,
            x_units=x_units or guessed or "nm",
            label=label,
            source=str(path),
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Spectrum(kind={self.kind.value!r}, n={self.x.size}, "
            f"x_units={self.x_units!r}, label={self.label!r})"
        )

    # ----------------------------------------------------------------- #
    #                           Transformations                         #
    # ----------------------------------------------------------------- #
    def normalised(self) -> Spectrum:
        """Return a copy scaled so that ``max(|y|) == 1``."""
        peak = np.nanmax(np.abs(self.y))
        scale = peak if peak else 1.0
        return Spectrum(
            self.x,
            self.y / scale,
            self.kind,
            x_units=self.x_units,
            label=self.label,
            source=self.source,
        )

    def normalized(self) -> Spectrum:
        """Alias for :meth:`normalised`."""
        return self.normalised()

    def to_stimulated_emission(self) -> Spectrum:
        """Return the stimulated-emission spectrum ``F_stim = F_spont * x^4``.

        Defined for emission spectra on a wavelength axis. The ``x^4`` factor
        converts a spontaneous-emission line shape to the stimulated-emission
        cross-section used when overlaying on transient spectra.
        """
        if self.kind is not SpectrumKind.EMISSION:
            raise ValueError("to_stimulated_emission is only defined for emission.")
        return Spectrum(
            self.x,
            self.y * self.x**4,
            self.kind,
            x_units=self.x_units,
            label=self.label,
            source=self.source,
        )

    def as_overlay_dict(self) -> dict:
        """Return ``{"X": x, "Y": y}`` for :meth:`Dataset1D.plot_spectra`.

        The raw (uncorrected) data is returned: ``plot_spectra`` applies its own
        smoothing, ``x^4`` emission correction and normalisation to the overlay.
        """
        return {"X": self.x, "Y": self.y}

    # ----------------------------------------------------------------- #
    #                              Plotting                             #
    # ----------------------------------------------------------------- #
    def _xy_units(self) -> tuple[dict, dict]:
        xl = _XLABEL.get(self.x_units, (self.x_units, self.x_units))
        if self.kind is SpectrumKind.ABSORPTION:
            yl = ("Absorbance", "a.u.")
        else:  # emission and excitation
            yl = ("Intensity", "a.u.")
        return {"lbl": xl[0], "ltx": xl[1]}, {"lbl": yl[0], "ltx": yl[1]}

    def plot(
        self,
        ax=None,
        *,
        settings: Settings | None = None,
        label_style: str | None = None,
        normalise: bool = False,
        normalize: bool | None = None,
        **kwargs,
    ):
        """Plot the spectrum on its own. Extra kwargs go to ``Axes.plot``."""
        import matplotlib.pyplot as plt

        from pymorgan import helpers as hlp

        if normalize is not None:
            normalise = normalize
        s = settings or get_settings()
        style = label_style if label_style is not None else s.label_style.value
        spec = self.normalised() if normalise else self

        if ax is None:
            _, ax = plt.subplots()
        kwargs.setdefault("label", spec.label)
        ax.plot(spec.x, spec.y, **kwargs)

        x_units, y_units = spec._xy_units()
        hlp.setXYlabels(ax, style, x_units, y_units, normY=normalise)
        ax.autoscale(enable=True, axis="x", tight=True)
        return ax


class SpectrumSeries:
    """An ordered collection of :class:`Spectrum` objects, plotted as an overlay."""

    def __init__(self, spectra: Iterable[Spectrum] | None = None):
        self.spectra: list[Spectrum] = list(spectra) if spectra is not None else []

    @classmethod
    def from_files(
        cls,
        paths: Sequence[str | PathLike],
        kind: SpectrumKind | str = SpectrumKind.ABSORPTION,
        *,
        data_type: str = "csv",
        x_units: str | None = None,
        labels: Sequence[str] | None = None,
    ) -> SpectrumSeries:
        """Load several spectra into a series."""
        spectra = []
        for i, path in enumerate(paths):
            label = labels[i] if labels is not None else None
            spectra.append(
                Spectrum.from_file(path, kind, data_type=data_type, x_units=x_units, label=label)
            )
        return cls(spectra)

    def add(self, spectrum: Spectrum) -> SpectrumSeries:
        """Append a spectrum and return ``self`` (chainable)."""
        self.spectra.append(spectrum)
        return self

    def __iter__(self) -> Iterator[Spectrum]:
        return iter(self.spectra)

    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(self, index) -> Spectrum:
        return self.spectra[index]

    def normalised(self) -> SpectrumSeries:
        """Return a copy in which every spectrum is normalised."""
        return SpectrumSeries(sp.normalised() for sp in self.spectra)

    def normalized(self) -> SpectrumSeries:
        """Alias for :meth:`normalised`."""
        return self.normalised()

    def plot(
        self,
        ax=None,
        *,
        settings: Settings | None = None,
        label_style: str | None = None,
        normalise: bool = False,
        normalize: bool | None = None,
        cmap: str = "rainbow",
        **kwargs,
    ):
        """Overlay every spectrum, coloured along ``cmap`` with a legend."""
        import matplotlib.pyplot as plt

        from pymorgan import helpers as hlp

        if normalize is not None:
            normalise = normalize
        if not self.spectra:
            raise ValueError("SpectrumSeries is empty.")

        s = settings or get_settings()
        style = label_style if label_style is not None else s.label_style.value
        series = self.normalised() if normalise else self

        if ax is None:
            _, ax = plt.subplots()
        colours = plt.get_cmap(cmap)(np.linspace(0, 1, len(series)))
        for colour, spec in zip(colours, series, strict=False):
            ax.plot(spec.x, spec.y, color=colour, label=spec.label, **kwargs)

        x_units, y_units = series.spectra[0]._xy_units()
        hlp.setXYlabels(ax, style, x_units, y_units, normY=normalise)
        ax.autoscale(enable=True, axis="x", tight=True)
        if any(sp.label for sp in series):
            ax.legend()
        return ax


def load_spectrum(
    path: str | PathLike,
    kind: SpectrumKind | str = SpectrumKind.ABSORPTION,
    *,
    data_type: str = "csv",
    x_units: str | None = None,
    label: str | None = None,
) -> Spectrum:
    """Convenience wrapper for :meth:`Spectrum.from_file`."""
    return Spectrum.from_file(path, kind, data_type=data_type, x_units=x_units, label=label)

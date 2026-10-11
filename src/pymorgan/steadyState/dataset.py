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

from ..settings import Settings, SteadyStatePlotMode, get_settings
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

    def _repr_html_(self) -> str:
        """Rich HTML summary card for Jupyter notebook display."""
        x_min = f"{float(np.nanmin(self.x)):.1f}" if self.x.size else "n/a"
        x_max = f"{float(np.nanmax(self.x)):.1f}" if self.x.size else "n/a"
        y_min = f"{float(np.nanmin(self.y)):.3g}" if self.y.size else "n/a"
        y_max = f"{float(np.nanmax(self.y)):.3g}" if self.y.size else "n/a"
        kind_name = self.kind.value.capitalize() if hasattr(self.kind, "value") else str(self.kind).capitalize()

        source_row = (
            f'<tr><td style="color:#64748b; padding: 3px 0;">Source</td>'
            f'<td style="word-break:break-all; font-size:11px; color:#64748b;">{self.source}</td></tr>'
            if self.source
            else ""
        )

        return (
            '<div style="font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, Helvetica, Arial, sans-serif; '
            'border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px 18px; max-width: 500px; '
            'background: #ffffff; color: #1e293b; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin: 8px 0;">'
            '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px;">'
            '<span style="font-weight: 700; font-size: 15px; color: #0f172a;">PyMORGAN Spectrum</span>'
            f'<span style="background:#0284c7;color:#fff;padding:2px 8px;border-radius:12px;font-size:11px;">{kind_name}</span>'
            '</div>'
            '<table style="width: 100%; border-collapse: collapse; font-size: 13px;">'
            f'<tr><td style="color:#64748b; padding: 3px 0;">Label</td><td style="font-weight:600;">{self.label or "(unlabelled)"}</td></tr>'
            f'<tr><td style="color:#64748b; padding: 3px 0;">Points</td><td style="font-weight:600;">{self.x.size}</td></tr>'
            f'<tr><td style="color:#64748b; padding: 3px 0;">X range</td><td>[{x_min}, {x_max}] {self.x_units}</td></tr>'
            f'<tr><td style="color:#64748b; padding: 3px 0;">Y range</td><td>[{y_min}, {y_max}]</td></tr>'
            f'{source_row}'
            '</table>'
            '</div>'
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

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"SpectrumSeries(n={len(self.spectra)})"

    def _repr_html_(self) -> str:
        """Rich HTML summary card for Jupyter notebook display."""
        n_spectra = len(self.spectra)
        labels = [sp.label or f"#{i+1}" for i, sp in enumerate(self.spectra[:5])]
        if n_spectra > 5:
            labels_str = ", ".join(labels) + f", ... (+{n_spectra - 5} more)"
        else:
            labels_str = ", ".join(labels) if labels else "(empty)"

        kinds = {sp.kind.value if hasattr(sp.kind, "value") else str(sp.kind) for sp in self.spectra}
        kinds_str = ", ".join(k.capitalize() for k in kinds) if kinds else "n/a"

        return (
            '<div style="font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, Helvetica, Arial, sans-serif; '
            'border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px 18px; max-width: 500px; '
            'background: #ffffff; color: #1e293b; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin: 8px 0;">'
            '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px;">'
            '<span style="font-weight: 700; font-size: 15px; color: #0f172a;">PyMORGAN SpectrumSeries</span>'
            f'<span style="background:#6366f1;color:#fff;padding:2px 8px;border-radius:12px;font-size:11px;">{n_spectra} spectra</span>'
            '</div>'
            '<table style="width: 100%; border-collapse: collapse; font-size: 13px;">'
            f'<tr><td style="color:#64748b; padding: 3px 0;">Kind(s)</td><td style="font-weight:600;">{kinds_str}</td></tr>'
            f'<tr><td style="color:#64748b; padding: 3px 0;">Spectra</td><td style="font-size:12px;">{labels_str}</td></tr>'
            '</table>'
            '</div>'
        )

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


def plot_steady_state(
    abs_spectrum: Spectrum | None = None,
    em_spectrum: Spectrum | None = None,
    ax=None,
    *,
    settings: Settings | None = None,
    label_style: str | None = None,
    plot_mode: SteadyStatePlotMode | str | None = None,
    normalise: bool = False,
    normalize: bool | None = None,
    **kwargs,
):
    """Plot steady-state absorption and/or emission spectra.

    When both absorption and emission are provided, `plot_mode` determines the layout:
    - ``SteadyStatePlotMode.DOUBLE_Y`` (default): Absorption on the primary (left) Y-axis,
      Emission on a twin (right) Y-axis (via ``ax.twinx()``) sharing the same X-axis.
    - ``SteadyStatePlotMode.NORMALISED``: Both spectra normalised to their maximum (1.0)
      on a single shared Y-axis.

    If only one spectrum is provided, it is drawn on a single axis.

    Returns
    -------
    matplotlib.axes.Axes or tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]
        The axis (or tuple of (primary_ax, twin_ax)) the spectra were drawn into.
    """
    import matplotlib.pyplot as plt

    from pymorgan import helpers as hlp

    if normalize is not None:
        normalise = normalize
    s = settings or get_settings()
    style = label_style if label_style is not None else s.label_style.value
    mode = SteadyStatePlotMode(plot_mode) if plot_mode is not None else s.ss_plot_mode

    if ax is None:
        _, ax = plt.subplots(figsize=s.spectra_figsize)

    if abs_spectrum is not None and em_spectrum is not None:
        if mode == SteadyStatePlotMode.DOUBLE_Y and not normalise:
            color_abs = getattr(s, "ss_abs_color", "b")
            color_em = getattr(s, "ss_em_color", "r")
            lbl_abs = abs_spectrum.label or "Absorption"
            lbl_em = em_spectrum.label or "Emission"

            lines_abs = ax.plot(
                abs_spectrum.x, abs_spectrum.y, color=color_abs, label=lbl_abs, **kwargs
            )
            x_units, y_units_abs = abs_spectrum._xy_units()
            hlp.setXYlabels(ax, style, x_units, y_units_abs, normY=False)
            ax.tick_params(axis="y", labelcolor=color_abs)
            ax.yaxis.label.set_color(color_abs)

            ax_em = ax.twinx()
            lines_em = ax_em.plot(
                em_spectrum.x, em_spectrum.y, color=color_em, label=lbl_em, **kwargs
            )
            _, y_units_em = em_spectrum._xy_units()
            hlp.setXYlabels(ax_em, style, x_units, y_units_em, setXLabel=False, setYLabel=True)
            ax_em.tick_params(axis="y", labelcolor=color_em)
            ax_em.yaxis.label.set_color(color_em)

            all_lines = lines_abs + lines_em
            all_labels = [l.get_label() for l in all_lines]
            ax.legend(all_lines, all_labels, loc="best")
            ax.autoscale(enable=True, axis="x", tight=True)
            return ax, ax_em
        else:
            abs_n = abs_spectrum.normalised()
            em_n = em_spectrum.normalised()
            color_abs = getattr(s, "ss_abs_color", "b")
            color_em = getattr(s, "ss_em_color", "r")
            lbl_abs = abs_spectrum.label or "Absorption"
            lbl_em = em_spectrum.label or "Emission"

            ax.plot(abs_n.x, abs_n.y, color=color_abs, label=lbl_abs, **kwargs)
            ax.plot(em_n.x, em_n.y, color=color_em, label=lbl_em, **kwargs)
            x_units, _ = abs_spectrum._xy_units()
            y_units = {"lbl": "Absorbance / Intensity", "ltx": "a.u."}
            hlp.setXYlabels(ax, style, x_units, y_units, normY=True)
            ax.legend(loc="best")
            ax.autoscale(enable=True, axis="x", tight=True)
            return ax
    elif abs_spectrum is not None:
        spec = abs_spectrum.normalised() if normalise else abs_spectrum
        color = getattr(s, "ss_abs_color", "b")
        ax.plot(spec.x, spec.y, color=color, label=spec.label or "Absorption", **kwargs)
        x_units, y_units = spec._xy_units()
        hlp.setXYlabels(ax, style, x_units, y_units, normY=normalise)
        if spec.label:
            ax.legend(loc="best")
        ax.autoscale(enable=True, axis="x", tight=True)
        return ax
    elif em_spectrum is not None:
        spec = em_spectrum.normalised() if normalise else em_spectrum
        color = getattr(s, "ss_em_color", "r")
        ax.plot(spec.x, spec.y, color=color, label=spec.label or "Emission", **kwargs)
        x_units, y_units = spec._xy_units()
        hlp.setXYlabels(ax, style, x_units, y_units, normY=normalise)
        if spec.label:
            ax.legend(loc="best")
        ax.autoscale(enable=True, axis="x", tight=True)
        return ax
    else:
        raise ValueError("At least one of abs_spectrum or em_spectrum must be provided.")

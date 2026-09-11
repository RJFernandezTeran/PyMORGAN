"""Plotting stage of the 1-D pipeline.

Every plotter takes a :class:`~pymorgan.oneD.Dataset1D` as its first argument
and reads the delay/probe axes, signal and unit dictionary from it. Label
style, colourmap, time-axis scale and the delta-A unit convention default to the
active :class:`pymorgan.settings.Settings` and can be overridden per call. This
makes the plotting layer extensible: a new view only needs ``(data, ax, ...)``
and gains the whole pipeline's context.
"""

from __future__ import annotations

import pathlib
import string

import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.colors as mcolors
import mpl_axes_aligner
import numpy as np
from matplotlib.legend_handler import HandlerLine2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import uniform_filter

from pymorgan import helpers as hlp
from pymorgan.log import get_logger
from pymorgan.settings import LEGEND_LOCATIONS

logger = get_logger(__name__)

# Match the legacy behaviour of the plotting routines.
np.seterr(divide="ignore")
np.seterr(invalid="ignore")


def _new_axes(ax, fig=None, layout=None, **subplot_kw):
    """Return ``(fig, ax)``, creating them if ``ax`` is ``None``.

    ``layout`` is forwarded to :func:`matplotlib.pyplot.subplots` only when
    a new figure is created (``ax is None``).  Pass ``'constrained'`` for
    kinetics/spectra figures so that the outside-right legend is automatically
    accounted for by the layout engine and survives the toolbar "Tight layout"
    button.
    """
    if ax is None:
        if layout is not None:
            subplot_kw.setdefault("layout", layout)
        fig, ax = plt.subplots(**subplot_kw)
    elif fig is None:
        fig = ax.figure
    return fig, ax


def _get_probe(data, detector: int = 0) -> np.ndarray:
    """Return 1-D probe array for detector (0-based)."""
    if hasattr(data, "_detector_probe"):
        return data._detector_probe(detector)
    probe = np.asarray(data.probe, dtype=float)
    if probe.ndim > 1:
        d = int(np.clip(detector, 0, probe.shape[1] - 1))
        return probe[:, d]
    return probe


def _z_label_style(label_style, units):
    """Delimiter style for the signal (Delta A) label.

    The 'x1E3' convention is rendered without a delimiter (Delta A x10^3);
    any other unit (e.g. mOD) honours the active label style, so mOD with
    "/" yields "Delta A / mOD".
    """
    if units.get("unitsZ_ltx") == r"$\times 10^{3}$":
        return ""
    return label_style


# Colorbar geometry, fixed in inches so the bar keeps the same physical size
# whatever the figure size is, and the tight_layout margins below leave a
# little (not excessive) room on the right for the tick labels.
_CBAR_WIDTH_IN = 0.15
_CBAR_PAD_IN = 0.1
# tight_layout bounding box: reserve room on the right (18%) and top (8%) for colorbar labels and ticks.
_LAYOUT_RECT = (0.0, 0.0, 0.82, 0.92)
# tight_layout rect when the legend is placed outside-right: reserve enough
# room for it so it is never clipped.
_LAYOUT_RECT_LEGEND = (0.0, 0.0, 0.75, 0.92)
# Plain decimals are kept inside this power-of-ten window; outside it the
# ScalarFormatter factors out a common '10**n' shown as a compact offset at
# the top of the bar, so extreme magnitudes (e.g. 1e-11) never overflow.
_CBAR_POWERLIMITS = (-3, 4)


# Wavenumber axes above this magnitude (cm-1) are shown divided by 1000 with a
# "x10^3 cm-1" label, so UV/Vis wavenumbers stay legible.
_WN_SCALE_THRESHOLD = 5000.0
_WN_SCALED_LTX = r"$\times 10^{3}$ cm$^{-1}$"


def _wn_div1000_formatter():
    """Tick formatter dividing wavenumber values by 1000 (for the x10^3 label)."""
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda v, _pos=None: f"{v / 1000:g}")


def _axis_label_text(labelStyle, lbl, ltx):
    """Axis-label string for ``lbl``/``ltx`` honouring the delimiter style."""
    match labelStyle:
        case "()":
            return r"%s (%s)" % (lbl, ltx)
        case "[]":
            return r"%s [%s]" % (lbl, ltx)
        case "/":
            return r"%s / %s" % (lbl, ltx)
        case _:
            return r"%s %s" % (lbl, ltx)


def _resolve_x_axis(X, Units, settings, *, x_axis_unit=None, secondary_axis=None):
    """Resolve the spectral X axis for a display unit + optional secondary axis.

    Returns ``(xplot, XUnits, sec, conv)``:

    * ``xplot``   -- ``X`` converted from the dataset's native spectral unit to
      the display unit (``Settings.x_axis_unit``, or ``x_axis_unit`` override).
    * ``XUnits``  -- ``{"lbl", "ltx"}`` axis-label dict for the display unit.
    * ``sec``     -- ``None`` or ``{"functions", "lbl", "ltx"}`` describing the
      complementary secondary axis (when ``secondary_axis`` is on and the
      complementary unit differs from the display unit).
    * ``conv``    -- callable mapping native-unit values to the display unit
      (for auxiliary overlays such as steady-state Abs/Em spectra).
    """
    xu = x_axis_unit if x_axis_unit is not None else settings.x_axis_unit.value
    sec_on = secondary_axis if secondary_axis is not None else settings.secondary_axis
    native = hlp.native_x_unit(Units)
    display = hlp.resolve_x_unit(xu, native)
    xplot = hlp.convert_spectral(X, native, display)
    lbl, ltx = hlp.x_unit_label(display)
    XUnits = {"lbl": lbl, "ltx": ltx, "wn_scaled": False}
    _xmax = float(np.nanmax(np.abs(xplot))) if np.asarray(xplot).size else 0.0
    if display == "cm-1" and _xmax > _WN_SCALE_THRESHOLD:
        XUnits["ltx"] = _WN_SCALED_LTX
        XUnits["wn_scaled"] = True

    def conv(v, _a=native, _b=display):
        return hlp.convert_spectral(v, _a, _b)

    sec = None
    if sec_on:
        comp = hlp.complementary_unit(display)
        if comp != display:

            def _fwd(v, _a=display, _b=comp):
                return hlp.convert_spectral(v, _a, _b)

            def _inv(v, _a=comp, _b=display):
                return hlp.convert_spectral(v, _a, _b)

            clbl, cltx = hlp.x_unit_label(comp)
            sec = {"functions": (_fwd, _inv), "lbl": clbl, "ltx": cltx, "scaled": False}
            if comp == "cm-1":
                _smax = float(np.nanmax(np.abs(_fwd(xplot)))) if np.asarray(xplot).size else 0.0
                if _smax > _WN_SCALE_THRESHOLD:
                    sec["ltx"] = _WN_SCALED_LTX
                    sec["scaled"] = True
    return xplot, XUnits, sec, conv


def _add_complementary_axis(where, sec, labelStyle, side="top"):
    """Attach the complementary secondary axis described by ``sec`` (or no-op)."""
    if sec is None:
        return None
    if side in ("top", "bottom"):
        ax2 = where.secondary_xaxis(side, functions=sec["functions"])
        ax2.set_xlabel(_axis_label_text(labelStyle, sec["lbl"], sec["ltx"]), labelpad=8)
        sec_axis = ax2.xaxis
    else:
        ax2 = where.secondary_yaxis(side, functions=sec["functions"])
        ax2.set_ylabel(_axis_label_text(labelStyle, sec["lbl"], sec["ltx"]), labelpad=8)
        sec_axis = ax2.yaxis
    if sec.get("scaled"):
        sec_axis.set_major_formatter(_wn_div1000_formatter())
    return ax2


def _asinh_colorbar_ticks(vmin, vmax, linear_width):
    """Generate clean, rounded tick values for an asinh colorbar without crowding near 0."""
    z_max = max(abs(vmin), abs(vmax))
    if z_max <= 0 or not np.isfinite(z_max):
        return [0.0]
    s = max(1e-9, float(linear_width))
    exp_min = int(np.floor(np.log10(s * 0.7)))
    exp_max = int(np.ceil(np.log10(z_max * 1.05)))
    candidates = []
    for exp in range(exp_min, exp_max + 1):
        base = 10.0 ** exp
        for mult in (1.0, 2.0, 5.0):
            val = mult * base
            if val >= s * 0.75 and val <= z_max * 1.05:
                candidates.append(val)
    candidates = sorted(set(candidates))
    if len(candidates) == 0:
        pos_ticks = [z_max]
    elif len(candidates) <= 3:
        pos_ticks = list(candidates)
    elif len(candidates) == 4:
        pos_ticks = [candidates[0], candidates[1], candidates[3]]
    else:
        mid_target = np.sqrt(candidates[0] * candidates[-1])
        mid_idx = int(np.argmin([abs(np.log(c) - np.log(mid_target)) for c in candidates[1:-1]])) + 1
        pos_ticks = [candidates[0], candidates[mid_idx], candidates[-1]]
    ticks = []
    if vmin < 0:
        ticks += [-x for x in reversed(pos_ticks) if x <= abs(vmin) * 1.05]
    ticks.append(0.0)
    if vmax > 0:
        ticks += [x for x in pos_ticks if x <= vmax * 1.05]
    return ticks


def _attach_colorbar(
    where,
    mappable,
    *,
    labelStyle=None,
    Units=None,
    Asinh: bool = False,
    asinh_linear_width: float | None = None,
    cbarLbl: str = "top",
    show_label=True,
):
    """Attach a divider-based colorbar with smart (overflow-safe) tick labels.

    A :func:`make_axes_locatable` divider appends a colorbar axis of fixed
    physical size (``_CBAR_WIDTH_IN``/``_CBAR_PAD_IN`` inches) to the right of
    ``where``; ``_finalize_layout`` then leaves a little room for its tick
    labels. Because the bar is a real sibling axes, ``tight_layout`` accounts
    for it and its labels at any figure size, so changing the figure-size
    defaults keeps everything on screen automatically.

    Tick numbers use a :class:`~matplotlib.ticker.ScalarFormatter` whose
    scientific mode is bounded by ``_CBAR_POWERLIMITS``: ordinary magnitudes
    print as plain decimals, while values outside that window are rendered with
    a common ``10**n`` factor shown as a compact offset above the bar -- this
    prevents labels such as ``0.00000000001`` from overflowing the axis. The
    unit label (e.g. the ``Delta A x10^3`` convention) is left untouched, so the
    offset and the unit label remain independent.
    """
    divider = make_axes_locatable(where)
    cax = divider.append_axes("right", size=_CBAR_WIDTH_IN, pad=_CBAR_PAD_IN)
    if Asinh and asinh_linear_width is not None and getattr(mappable, "norm", None) is not None:
        norm = mappable.norm
        ticks = _asinh_colorbar_ticks(norm.vmin, norm.vmax, asinh_linear_width)
    else:
        ticks = tkr.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10])
    cbar = where.figure.colorbar(mappable, cax=cax, ticks=ticks)
    _fmt = tkr.ScalarFormatter(useMathText=True)
    _fmt.set_powerlimits(_CBAR_POWERLIMITS)
    cbar.ax.yaxis.set_major_formatter(_fmt)
    zstyle = _z_label_style(labelStyle, Units)
    if show_label:
        if cbarLbl == "top":
            cbar.ax.set_title(
                hlp.fmtZlabel(zstyle, Units["unitsZ_lbl"], Units["unitsZ_ltx"], twoLines=True),
                fontsize=14,
                loc="left",
            )
        elif cbarLbl == "side":
            cbar.set_label(
                hlp.fmtZlabel(zstyle, Units["unitsZ_lbl"], Units["unitsZ_ltx"], twoLines=False),
                fontsize=14,
            )
    cbar.ax.tick_params(axis="y", direction="out", labelsize=14)
    cbar.ax.yaxis.offsetText.set_fontsize(14)
    return cbar


def _finalize_layout(fig, created: bool = True):
    """Lay the figure out deterministically around the divider colorbar.

    The divider geometry is manual, so any managed engine is switched off (the
    embedded GUI canvas is created with ``constrained_layout``, which conflicts
    with ``make_axes_locatable``); ``tight_layout`` is then run to size the
    margins cleanly to the current labels.

    ``created`` is ``False`` when the caller supplied its own axes (composite
    figures built from a ``GridSpec``, or the embedded GUI canvas). In that case
    the figure-wide ``tight_layout`` would override the caller's geometry, so
    only the conflicting layout engine is cleared and the caller stays
    responsible for the final layout.
    """
    fig.set_layout_engine(None)
    if created:
        fig.tight_layout()


def _tight_layout_for_legend(fig, s):
    """Run tight_layout with the correct rect given the legend location.

    When the legend is outside the axes (``legend_location = "outside right"``
    in Settings) the standard tight_layout clips it because it has no idea
    about the off-axis legend box.  We reserve extra room on the right in
    that case via ``_LAYOUT_RECT_LEGEND`` so the legend is always visible.
    For any other placement the legend is inside the axes bounding box and
    the default tight_layout rect is fine.
    """
    loc = str(getattr(s, "legend_location", "best") or "best").strip().lower()
    rect = _LAYOUT_RECT_LEGEND if loc.startswith("outside") else None
    if rect is not None:
        fig.tight_layout(rect=rect)
    else:
        fig.tight_layout()


_FIGURE_KIND_LABELS = {
    "contour": "Contour plot",
    "spectra": "Spectral traces",
    "kinetics": "Kinetic traces",
    "scan spectra": "Per-scan spectral traces",
    "scan kinetics": "Per-scan kinetic traces",
}

# Minor-tick subdivisions for the symlog kinetic time axis (shared by the
# averaged and per-scan kinetic plotters).
_SYMLOG_SUBS = (
    -0.9,
    -0.8,
    -0.7,
    -0.6,
    -0.5,
    -0.4,
    -0.3,
    -0.2,
    -0.1,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
)


def _set_figure_window_title(fig, kind, data):
    """Name a plotter-created figure window.

    Format: ``Figure N (<kind label>) [<dataset name>]`` -- N is the matplotlib
    figure number and the dataset name omits any file extension. No-op when the
    figure has no window manager (headless run / embedded GUI canvas).
    """
    mgr = getattr(fig.canvas, "manager", None)
    if mgr is None or not hasattr(mgr, "set_window_title"):
        return
    label = _FIGURE_KIND_LABELS.get(kind, kind)
    src = getattr(data, "source", None)
    name = pathlib.Path(src).stem if src else ""
    num = getattr(fig, "number", None)
    prefix = "Figure %d" % num if num is not None else "Figure"
    title = "%s (%s) [%s]" % (prefix, label, name) if name else "%s (%s)" % (prefix, label)
    mgr.set_window_title(title)


# --------------------------------------------------------------------------- #
#                                  Contour                                     #
# --------------------------------------------------------------------------- #
def _time_axis_formatter(mode):
    """Return a matplotlib major-tick formatter for the log/symlog time axis.

    ``mode`` is a :class:`pymorgan.settings.TimeAxisLabel` value:
    ``"power"`` -> 10^n, ``"decimal"`` -> plain numbers (0.1, 1, 10),
    ``"mixed"`` -> powers of ten except 10^0, which is shown as "1".
    Non-power tick positions fall back to a plain ``%g`` rendering.
    """

    def _decompose(value):
        sign = "-" if value < 0 else ""
        mag = abs(value)
        exp = int(round(np.log10(mag)))
        is_pow = np.isclose(mag, 10.0**exp)
        return sign, exp, is_pow

    def _fmt(value, _pos=None):
        if value == 0:
            return "0"
        sign, exp, is_pow = _decompose(value)
        if not is_pow:
            return "%g" % value
        if mode == "decimal":
            return "%g" % value
        if mode == "mixed" and exp == 0:
            return "%s1" % sign
        return r"$%s10^{%d}$" % (sign, exp)

    return tkr.FuncFormatter(_fmt)


def plot_contour(
    data,
    ax=None,
    *,
    detector: int = 0,
    settings=None,
    label_style: str | None = None,
    delta_a_units=None,
    x_axis_unit=None,
    secondary_axis=None,
    ShowLines=False,
    smooth=0,
    filled=True,
    Nlevels=None,
    Nskip=None,
    white_levels=None,
    Asinh=False,
    asinh_pct=None,
    cbarLbl="top",
    cmap_ID=None,
    anisotropy=False,
    Zmax=None,
    Zmin=None,
    Zscale=100,
    ProbeDir="X",
    Yscale=None,
    time_axis_label=None,
    quick=None,
    show_xlabel=True,
    show_ylabel=True,
    show_colorbar=True,
    show_colorbar_label=True,
    aspect="auto",
):
    """Delay-vs-probe contour map of ``data``.

    Draws the transient signal as a filled contour map (probe on X, delay on Y
    by default) with a symmetric, zero-centred colour scale and a divider
    colorbar. Every keyword that is not given falls back to the active
    :class:`~pymorgan.settings.Settings`, so the defaults follow the settings
    file and only per-figure choices need to be passed.

    Parameters
    ----------
    data : Dataset1D
        Dataset to plot. Bound as ``Dataset1D.plot_contour``, so this argument
        is supplied automatically when called as a method.
    ax : matplotlib.axes.Axes, optional
        Axis to draw into. ``None`` creates a new figure sized from
        ``Settings.contour_figsize`` and lays it out with ``tight_layout``. When
        an axis *is* supplied the figure geometry is left untouched, which is
        what makes the plotter usable inside a ``GridSpec`` composite (see
        ``examples/composite_1D.py``).
    detector : int, default 0
        Detector index for multi-detector datasets.
    settings : Settings, optional
        Settings object to resolve defaults from; defaults to the active one.
    label_style : {"()", "[]", "/", ""}, optional
        Delimiter convention for axis labels, e.g. ``"Delay (ps)"`` vs
        ``"Delay / ps"``. Defaults to ``Settings.label_style``.
    delta_a_units : str, optional
        Signal-unit convention for the colorbar label ("mOD", "x1E3", ...).
        Defaults to ``Settings.delta_a_units``. Data are always stored in mOD.
    x_axis_unit : {"nm", "cm-1", "eV", "THz"}, optional
        Convert the probe axis to this unit for display. Defaults to
        ``Settings.x_axis_unit`` (no conversion when unset).
    secondary_axis : bool, optional
        Add a complementary-unit axis opposite the probe axis (top for
        ``ProbeDir="X"``, right for ``"Y"``). Defaults to the active setting.
    ShowLines : bool, default False
        ``True`` overlays black contour lines on the filled map; ``False``
        colours the contour lines with the colourmap instead.
    smooth : int, default 0
        Box-filter width (points) applied to the data used for the *contour-line*
        overlay only, when ``ShowLines`` is set. The filled background always
        uses the raw data. Values <= 1 disable smoothing.
    filled : bool, default True
        Draw the filled ``contourf`` background. ``False`` gives a line-only map.
    Nlevels : int, optional
        Number of contour levels (rounded up to an even number so the scale stays
        symmetric about zero). Default 30.
    Nskip : int, default 2
        Keep every ``Nskip``-th contour *line*, counting outwards from zero, so
        the line overlay stays legible when ``Nlevels`` is large.
    white_levels : float, optional
        Number of central levels forced to white, which suppresses noise around
        the zero crossing. Defaults to ``Settings.white_levels``.
    Asinh : bool, default False
        Apply arcsinh color scaling and non-linear contour levels, compressing
        large amplitudes so weak and strong bands can share one scale while
        preserving native signal units on the data and colorbar.
    asinh_pct : float, optional
        Linear threshold for arcsinh color scaling as a percentage of the active
        amplitude limit (Zscale / Zmax). Defaults to ``Settings.asinh_pct`` (5.0%).
    cbarLbl : {"side", "top"}, default "side"
        Place the colorbar unit label beside or above the bar.
    cmap_ID : str, optional
        Colourmap specification (e.g. ``"DkRd/Wh/DkBu"``). Defaults to
        ``Settings.cmap``.
    anisotropy : bool, default False
        Treat ``Z`` as an anisotropy map and fix the colour scale to +/-0.7.
    Zmin, Zmax : float, optional
        Explicit colour-scale limits, in the plotted signal units. Given
        together they override ``Zscale``; this is how several panels are put on
        a common scale.
    Zscale : float, default 100
        Colour-scale limit as a percentage of the peak absolute signal, used
        when ``Zmax`` is not given. Values below 100 saturate the map and bring
        out weak features.
    ProbeDir : {"X", "Y"}, default "X"
        Axis carrying the probe; ``"Y"`` transposes the map so delay runs along
        X.
    Yscale : {"symlog", "log", "lin"}, optional
        Scale of the delay axis. Defaults to ``Settings.time_axis_scale``.
        ``"symlog"`` is linear within +/-1 (time unit) and logarithmic outside.
    time_axis_label : str, optional
        Tick-label convention for a log/symlog delay axis. Defaults to
        ``Settings.time_axis_label``.
    quick : bool, optional
        Draw a single rasterised ``pcolormesh`` instead of contour sets and skip
        the line overlay. Much faster for large maps and for interactive use.
        Defaults to ``Settings.quick_plots``.
    show_xlabel, show_ylabel : bool, default True
        Draw the X / Y axis labels. Set to ``False`` on inner panels of a
        composite figure that share an axis with a neighbour.
    show_colorbar : bool, default True
        Attach the divider colorbar. Set ``False`` when several panels share one
        externally drawn colorbar.
    show_colorbar_label : bool, default True
        Draw the unit label on the colorbar (ignored when ``show_colorbar`` is
        ``False``).

    Returns
    -------
    matplotlib.axes.Axes
        The axis the map was drawn into.

    Notes
    -----
    Detector gaps (jumps in the probe axis larger than 1.5x the median spacing)
    are masked so no contour is interpolated across them.

    Examples
    --------
    >>> ax = data.plot_contour(ShowLines=True, Zscale=30, Yscale="symlog")
    >>> data.plot_contour(ax=ax_panel, show_colorbar=False, Zmin=-2, Zmax=2)
    """
    s, labelStyle, Units = data._resolve(settings, label_style, delta_a_units)
    if cmap_ID is None:
        cmap_ID = s.cmap
    if Yscale is None:
        Yscale = s.time_axis_scale.value
    if time_axis_label is None:
        time_axis_label = s.time_axis_label.value
    if white_levels is None:
        white_levels = s.white_levels
    if asinh_pct is None:
        asinh_pct = getattr(s, "asinh_pct", 5.0)

    created = ax is None
    pFig, where = _new_axes(ax, figsize=s.contour_figsize)
    if aspect:
        where.set_aspect(aspect)
    Zavg_C = data._detector_slice(detector)
    Y_t = data.delays
    X_l = _get_probe(data, detector)

    if ProbeDir == "Y":
        Zavg_C = Zavg_C.T
        X_l, Y_t = Y_t, X_l

    if Zmax is None:
        Zmax = np.nanmax(np.abs(Zavg_C)) * Zscale / 100
    if Zmin is None:
        Zmin = -Zmax
    if Zmin == Zmax or abs(Zmax - Zmin) < 1e-12 or np.isnan(Zmax):
        Zmin, Zmax = -1.0, 1.0
    if Nlevels is None:
        Nlevels = getattr(s, "n_contours", 40)

    NctrL = int(Nlevels)
    if NctrL % 2 != 0:
        NctrL += 1
    NctrF = int(Nlevels)
    if NctrF % 2 != 0:
        NctrF += 1
    if Nskip is None:
        Nskip = 2

    if anisotropy:
        Zmin = -0.7
        Zmax = 0.7

    cm_obj, _ = hlp.CalcCMAP(cmap_ID, NctrF)
    if white_levels:
        cm_obj = hlp.zero_center_cmap(cm_obj, NctrF, int(white_levels))

    LinThres = 1
    LinSize = 0.5
    LCol = "k"

    if ProbeDir == "X":
        grad = np.abs(np.gradient(X_l))
        mask2D = np.tile(
            grad > 1.5 * np.nanmedian(grad), (Zavg_C.shape[0], 1)
        )
    else:
        grad = np.abs(np.gradient(Y_t))
        mask2D = np.tile(
            grad > 1.5 * np.nanmedian(grad), (Zavg_C.shape[1], 1)
        ).T
    if np.ma.is_masked(Zavg_C):
        mask2D = mask2D | np.ma.getmaskarray(Zavg_C)
    if np.any(np.isnan(Zavg_C)):
        mask2D = mask2D | np.isnan(np.asarray(Zavg_C))
    Zavg_C = np.ma.masked_where(mask2D, Zavg_C)

    # Convert the probe axis to the chosen display unit (X for ProbeDir 'X',
    # Y for 'Y'); the detector-gap mask above used the native axis.
    _probe_native = X_l if ProbeDir == "X" else Y_t
    _xplot, _xunits, _sec, _ = _resolve_x_axis(
        _probe_native, Units, s, x_axis_unit=x_axis_unit, secondary_axis=secondary_axis
    )
    if ProbeDir == "X":
        X_l = _xplot
    else:
        Y_t = _xplot

    asinh_linear_width = None
    if Asinh:
        zspan = max(abs(Zmin), abs(Zmax))
        if zspan == 0 or np.isnan(zspan):
            zspan = 1.0
        s_val = max(1e-9, (float(asinh_pct) / 100.0) * zspan)
        asinh_linear_width = s_val
        norm = mcolors.AsinhNorm(linear_width=s_val, vmin=Zmin, vmax=Zmax)

        u_min = -np.arcsinh(abs(Zmin) / s_val) if Zmin < 0 else np.arcsinh(Zmin / s_val)
        u_max = np.arcsinh(Zmax / s_val) if Zmax > 0 else -np.arcsinh(abs(Zmax) / s_val)
        ctrLvl_F = np.unique(s_val * np.sinh(np.linspace(u_min, u_max, NctrF)))
        ctrLvl_L = s_val * np.sinh(np.linspace(u_min, u_max, NctrL))
    else:
        norm = mcolors.Normalize(vmin=Zmin, vmax=Zmax)
        ctrLvl_F = np.unique(np.linspace(Zmin, Zmax, NctrF))
        ctrLvl_L = np.linspace(Zmin, Zmax, NctrL)

    mappable = None
    quick = bool(s.quick_plots if quick is None else quick)
    if quick:
        # Fast path: a single image instead of a filled contour set, and no
        # contour-line overlay (see Settings.quick_plots).
        mappable = where.pcolormesh(
            X_l,
            Y_t,
            Zavg_C,
            norm=norm,
            cmap=cm_obj,
            shading="nearest",
            rasterized=True,
        )
    elif filled:
        mappable = where.contourf(
            X_l, Y_t, Zavg_C, norm=norm, levels=ctrLvl_F, cmap=cm_obj, extend="neither"
        )

    if Nskip > 1:
        ctrLvl_LP = ctrLvl_L[ctrLvl_L >= 0]
        ctrLvl_LN = ctrLvl_L[ctrLvl_L <= 0]
        ctrLvl_L = np.append(np.flipud(np.flipud(ctrLvl_LN)[::Nskip]), ctrLvl_LP[::Nskip])
    # The +/- partitions both contain an exact 0 when the level count is odd and
    # the limits straddle zero, which duplicates that level and trips
    # matplotlib's "contour levels must be increasing" check. Collapse to the
    # sorted unique set so the levels are always strictly increasing.
    ctrLvl_L = np.unique(ctrLvl_L)

    if quick:
        pass  # the contour-line passes below are what quick mode skips
    elif ShowLines:
        # Optionally smooth the data used for the contour-line overlay only;
        # the filled background (contourf above) always uses the raw data.
        _smooth_pts = abs(int(smooth))
        if _smooth_pts > 1:
            if np.ma.is_masked(Zavg_C):
                _mask = np.ma.getmaskarray(Zavg_C)
                _valid = (~_mask).astype(float)
                _filled_Z = np.nan_to_num(np.ma.filled(Zavg_C, 0.0), nan=0.0)
                _sum_val = uniform_filter(_filled_Z, size=_smooth_pts)
                _sum_cnt = uniform_filter(_valid, size=_smooth_pts)
                with np.errstate(divide="ignore", invalid="ignore"):
                    _Z_for_lines = np.where(_sum_cnt > 0, _sum_val / _sum_cnt, 0.0)
                _Z_for_lines = np.ma.masked_where(_mask, _Z_for_lines)
            else:
                _Z_for_lines = uniform_filter(Zavg_C, size=_smooth_pts)
        else:
            _Z_for_lines = Zavg_C
        where.contour(
            X_l,
            Y_t,
            _Z_for_lines,
            norm=norm,
            levels=ctrLvl_L,
            linewidths=0.5,
            colors=LCol,
        )
    else:
        where.contour(
            X_l,
            Y_t,
            Zavg_C,
            norm=norm,
            levels=ctrLvl_L,
            linewidths=0.5,
            cmap=cm_obj,
            extend="neither",
        )

    if ProbeDir == "X":
        XUnits = _xunits
        YUnits = {"lbl": "Delay", "ltx": Units["unitsT_ltx"]}
    else:
        XUnits = {"lbl": "Delay", "ltx": Units["unitsT_ltx"]}
        YUnits = _xunits
    hlp.setXYlabels(where, labelStyle, XUnits, YUnits, setXLabel=show_xlabel, setYLabel=show_ylabel)
    _add_complementary_axis(where, _sec, labelStyle, side="top" if ProbeDir == "X" else "right")
    if _xunits.get("wn_scaled"):
        probe_axis = where.xaxis if ProbeDir == "X" else where.yaxis
        probe_axis.set_major_formatter(_wn_div1000_formatter())

    if ProbeDir == "X":
        if Yscale == "symlog":
            where.set_yscale(
                "symlog",
                linthresh=LinThres,
                linscale=LinSize,
                subs=(
                    -0.9,
                    -0.8,
                    -0.7,
                    -0.6,
                    -0.5,
                    -0.4,
                    -0.3,
                    -0.2,
                    -0.1,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                ),
            )
            where.axhline(y=LinThres, color="w", linewidth=1, linestyle=":")
            tmin_SL = Y_t[0]
            where.set_ylim([tmin_SL, Y_t[-1]])
        elif Yscale == "log":
            where.set_yscale("log", subs=(1, 2, 3, 4, 5, 6, 7, 8, 9))
            where.set_ylim([0.01, Y_t[-1]])
        elif Yscale == "lin":
            where.set_yscale("linear")
            where.set_ylim([Y_t[0], Y_t[-1]])
        where.axhline(y=0, color="0.5", linewidth=1)
    else:
        if Yscale == "symlog":
            where.set_xscale(
                "symlog",
                linthresh=LinThres,
                linscale=LinSize,
                subs=(
                    -0.9,
                    -0.8,
                    -0.7,
                    -0.6,
                    -0.5,
                    -0.4,
                    -0.3,
                    -0.2,
                    -0.1,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                ),
            )
            where.axvline(x=LinThres, color="w", linewidth=1, linestyle=":")
            tmin_SL = Y_t[0]
            where.set_xlim([tmin_SL, X_l[-1]])
        elif Yscale == "log":
            where.set_xscale("log", subs=(1, 2, 3, 4, 5, 6, 7, 8, 9))
            where.set_xlim([0.01, X_l[-1]])
        elif Yscale == "lin":
            where.set_xscale("linear")
            where.set_xlim([X_l[0], X_l[-1]])
        where.axvline(x=0, color="0.5", linewidth=1)
        where.invert_yaxis()

    if Yscale in ("symlog", "log"):
        taxis = where.yaxis if ProbeDir == "X" else where.xaxis
        taxis.set_major_formatter(_time_axis_formatter(time_axis_label))

    if show_colorbar:
        if mappable is None:
            from matplotlib.cm import ScalarMappable

            mappable = ScalarMappable(norm=norm, cmap=cm_obj)
            mappable.set_array([])
        _attach_colorbar(
            where,
            mappable,
            labelStyle=labelStyle,
            Units=Units,
            Asinh=Asinh,
            asinh_linear_width=asinh_linear_width,
            cbarLbl=cbarLbl,
            show_label=show_colorbar_label,
        )

    _x_arr = np.asarray(X_l, dtype=float)
    _y_arr = np.asarray(Y_t, dtype=float)
    _z_grid = np.asarray(Zavg_C)

    def _format_coord(x, y):
        try:
            if (
                _x_arr.size == 0
                or _y_arr.size == 0
                or x < np.nanmin(_x_arr)
                or x > np.nanmax(_x_arr)
                or y < np.nanmin(_y_arr)
                or y > np.nanmax(_y_arr)
            ):
                return f"x={x:.4g}, y={y:.4g}"
            ix = int(np.nanargmin(np.abs(_x_arr - x)))
            iy = int(np.nanargmin(np.abs(_y_arr - y)))
            val = _z_grid[iy, ix]
            if np.ma.is_masked(val) or np.isnan(val):
                return f"x={x:.4g}, y={y:.4g}, z=NaN"
            return f"x={x:.4g}, y={y:.4g}, z={val:.4g}"
        except Exception:
            return f"x={x:.4g}, y={y:.4g}"

    where.format_coord = _format_coord

    _finalize_layout(where.figure, created)
    if created:
        _set_figure_window_title(pFig, "contour", data)
    return where


# --------------------------------------------------------------------------- #
#                                  Surface                                     #
# --------------------------------------------------------------------------- #
def apply_delay_axis(where, axis_name, scale, Y_t, time_axis_label=None, settings=None):
    """Scale and format a delay axis the way ``plot_contour`` does.

    ``axis_name`` is ``"x"``, ``"y"`` or ``"z"``; ``scale`` is ``"symlog"``,
    ``"log"`` or ``"lin"``. Keeps every delay axis consistent with the embedded
    contour (same scale, limits and tick-label format), which is what the
    plot-controls panel configures.

    Public so that a host application drawing a view PyMORGAN has no concept of
    -- PyRATE-TA's concentration profiles, for instance -- still gets the delay
    axis this package defines, rather than a second convention beside it.
    """
    if settings is None:
        from pymorgan import get_settings

        settings = get_settings()
    if scale is None:
        scale = settings.time_axis_scale
    scale = scale.value if hasattr(scale, "value") else scale

    set_scale = getattr(where, f"set_{axis_name}scale")
    set_lim = getattr(where, f"set_{axis_name}lim")
    axis = getattr(where, f"{axis_name}axis")

    if scale == "symlog":
        set_scale("symlog", linthresh=1, linscale=0.5, subs=_SYMLOG_SUBS)
        set_lim([Y_t[0], Y_t[-1]])
    elif scale == "log":
        # Same lower bound as plot_contour. The positive range is applied before
        # switching so matplotlib does not warn about the pre-zero delays.
        set_lim([0.01, Y_t[-1]])
        set_scale("log", subs=(1, 2, 3, 4, 5, 6, 7, 8, 9))
        set_lim([0.01, Y_t[-1]])
    else:
        set_scale("linear")
        set_lim([Y_t[0], Y_t[-1]])

    if scale in ("symlog", "log"):
        axis.set_major_formatter(_time_axis_formatter(time_axis_label))
    return scale


def plot_surface(
    data,
    ax=None,
    *,
    detector: int = 0,
    settings=None,
    label_style: str | None = None,
    delta_a_units=None,
    ShowLines=False,
    Asinh=False,
    cbarLbl="top",
    cmap_ID=None,
    show_xlabel=True,
    show_ylabel=True,
    show_colorbar=True,
    show_colorbar_label=True,
    rstride=1,
    cstride=1,
    elev=30,
    azim=-60,
    Yscale=None,
    time_axis_label=None,
):
    """Render delay-vs-probe 3D surface plot with colourmap shading.

    The delay (Y) axis follows the same configuration as the contour plots --
    ``Yscale`` defaults to ``Settings.time_axis_scale`` and the tick labels to
    ``Settings.time_axis_label`` -- so the surface matches the embedded preview
    and the plot-controls panel.

    Parameters
    ----------
    data : Dataset1D
        Dataset to plot; supplied automatically via ``Dataset1D.plot_surface``.
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional
        3-D axis to draw into. ``None`` creates one; a supplied axis **must**
        already have ``projection="3d"``.
    detector : int, default 0
        Detector index for multi-detector datasets.
    settings : Settings, optional
        Settings object to resolve defaults from; defaults to the active one.
    label_style : {"()", "[]", "/", ""}, optional
        Delimiter convention for the axis labels.
    delta_a_units : str, optional
        Signal-unit convention for the Z label.
    ShowLines : bool, default False
        Draw the surface with visible wireframe edges.
    Asinh : bool, default False
        Plot ``arcsinh(Z)``, compressing large amplitudes.
    cbarLbl : {"side", "top"}, default "side"
        Placement of the colorbar unit label.
    cmap_ID : str, optional
        Colourmap specification; defaults to ``Settings.cmap``.
    show_xlabel, show_ylabel, show_colorbar, show_colorbar_label : bool
        Element visibility, all ``True`` by default.
    rstride, cstride : int, default 1
        Row / column downsampling of the surface mesh. Increase for large maps.
    elev, azim : float, default 30 and -60
        Initial camera elevation and azimuth, in degrees.
    Yscale : {"symlog", "log", "lin"}, optional
        Scale of the delay axis; defaults to ``Settings.time_axis_scale``.
    time_axis_label : str, optional
        Tick-label convention for a log/symlog delay axis.

    Returns
    -------
    mpl_toolkits.mplot3d.axes3d.Axes3D
        The axis the surface was drawn into.
    """
    s, labelStyle, Units = data._resolve(settings, label_style, delta_a_units)
    if cmap_ID is None:
        cmap_ID = s.cmap

    pFig, where = _new_axes(ax, subplot_kw={"projection": "3d"})
    Zavg_C = data._detector_slice(detector)
    Y_t = data.delays
    X_l = _get_probe(data, detector)

    if Asinh:
        Zavg_C = np.arcsinh(Zavg_C)
        Units["unitsZ_lbl"] = r"arcsinh (%s/%s)" % (Units["unitsZ_lbl"], Units["unitsZ_ltx"])

    Zmax = np.max(np.abs(Zavg_C))
    Zmin = -Zmax
    NctrF = 100

    [cm_obj, _] = hlp.CalcCMAP(cmap_ID, NctrF)

    grad = np.abs(np.gradient(X_l))
    mask2D = np.tile(grad > 1.5 * np.nanmedian(grad), (Zavg_C.shape[0], 1))
    if np.ma.is_masked(Zavg_C):
        mask2D = mask2D | np.ma.getmaskarray(Zavg_C)
    if np.any(np.isnan(Zavg_C)):
        mask2D = mask2D | np.isnan(np.asarray(Zavg_C))
    Zavg_C = np.ma.masked_where(mask2D, Zavg_C)

    X_grid, Y_grid = np.meshgrid(X_l, Y_t)

    surf = where.plot_surface(
        X_grid,
        Y_grid,
        Zavg_C,
        cmap=cm_obj,
        vmin=Zmin,
        vmax=Zmax,
        rstride=rstride,
        cstride=cstride,
        linewidth=0.15,
        edgecolor=(0.2, 0.2, 0.2, 0.15),
        antialiased=True,
    )

    # Make background 3D panes and gridlines semitransparent so surface colourmap is unobscured
    where.xaxis.pane.set_alpha(0.05)
    where.yaxis.pane.set_alpha(0.05)
    where.zaxis.pane.set_alpha(0.05)
    where.xaxis.pane.set_edgecolor((0.7, 0.7, 0.7, 0.3))
    where.yaxis.pane.set_edgecolor((0.7, 0.7, 0.7, 0.3))
    where.zaxis.pane.set_edgecolor((0.7, 0.7, 0.7, 0.3))

    if hasattr(where, "mouse_init"):
        where.mouse_init()

    apply_delay_axis(where, "y", Yscale, Y_t, time_axis_label=time_axis_label, settings=s)

    XUnits = {"lbl": Units["unitsL_lbl"], "ltx": Units["unitsL_ltx"]}
    YUnits = {"lbl": "Delay", "ltx": Units["unitsT_ltx"]}
    hlp.setXYlabels(where, labelStyle, XUnits, YUnits, setXLabel=show_xlabel, setYLabel=show_ylabel)
    where.set_zlabel(r"$\Delta$A", fontweight="bold")

    where.view_init(elev=elev, azim=azim)

    if show_colorbar:
        cbar = pFig.colorbar(surf, ax=where, shrink=0.6, pad=0.1, aspect=15)
        if show_colorbar_label:
            cbar.set_label(
                r"%s (%s)" % (Units["unitsZ_lbl"], Units["unitsZ_ltx"]), fontweight="bold"
            )

    return where


def plot_noise_3d(
    data,
    ax=None,
    *,
    detector: int = 0,
    settings=None,
    label_style: str | None = None,
    delta_a_units=None,
    x_axis_unit=None,
    secondary_axis=None,
    show_xlabel=True,
    show_ylabel=True,
):
    """Plot the noise array as a rotatable 3D surface plot using the 'hot_r' colourmap.

    The noise surface makes it easy to spot pixels or delay ranges where the
    standard deviation is anomalously high (bad detector pixels, a drifting
    scan) before those regions are trusted in a fit.

    Parameters
    ----------
    data : Dataset1D
        Dataset to plot; supplied automatically via ``Dataset1D.plot_noise_3d``.
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional
        3-D axis to draw into (``projection="3d"`` required when supplied).
    detector : int, default 0
        Detector index for multi-detector datasets.
    settings : Settings, optional
        Settings object to resolve defaults from; defaults to the active one.
    label_style : {"()", "[]", "/", ""}, optional
        Delimiter convention for the axis labels.
    delta_a_units : str, optional
        Signal-unit convention for the noise (Z) label.
    x_axis_unit : {"nm", "cm-1", "eV", "THz"}, optional
        Convert the probe axis to this unit for display.
    secondary_axis : bool, optional
        Add a complementary-unit axis.
    show_xlabel, show_ylabel : bool, default True
        Draw the X / Y axis labels.

    Returns
    -------
    mpl_toolkits.mplot3d.axes3d.Axes3D
        The axis the surface was drawn into.

    Raises
    ------
    ValueError
        If the dataset carries neither a noise array nor single-scan data.

    See Also
    --------
    pymorgan.oneD.Dataset1D.noise_array : the array this plots.
    """
    s, labelStyle, Units = data._resolve(settings, label_style, delta_a_units)

    pFig, where = _new_axes(ax, subplot_kw={"projection": "3d"})

    noise = data.noise_array()
    if noise is None:
        raise ValueError("No noise or single-scan data available to plot.")

    if noise.ndim == 3:
        noise_2d = noise[:, :, detector]
    else:
        noise_2d = noise

    Y_t = data.delays
    X_l = _get_probe(data, detector)

    # Convert the probe axis to the chosen display unit
    _xplot, _xunits, _sec, _ = _resolve_x_axis(
        X_l, Units, s, x_axis_unit=x_axis_unit, secondary_axis=secondary_axis
    )
    X_l = _xplot

    # Mask out detector gaps (similar to plot_contour)
    probe_arr = _get_probe(data, detector)
    grad = np.abs(np.gradient(probe_arr))
    mask2D = np.tile(
        grad > 1.5 * np.nanmedian(grad),
        (noise_2d.shape[0], 1),
    )
    if np.ma.is_masked(noise_2d):
        mask2D = mask2D | np.ma.getmaskarray(noise_2d)
    if np.any(np.isnan(noise_2d)):
        mask2D = mask2D | np.isnan(np.asarray(noise_2d))
    noise_2d = np.ma.masked_where(mask2D, noise_2d)

    # Construct 2D grid for plot_surface
    X_grid, Y_grid = np.meshgrid(X_l, Y_t)

    # Plot 3D surface. Use 'hot_r' colourmap so that white = low noise (0).
    surf = where.plot_surface(
        X_grid,
        Y_grid,
        noise_2d,
        cmap="hot_r",
        edgecolor="none",
        antialiased=True,
    )

    # Label the axes
    XUnits = _xunits
    YUnits = {"lbl": "Delay", "ltx": Units["unitsT_ltx"]}
    hlp.setXYlabels(where, labelStyle, XUnits, YUnits, setXLabel=show_xlabel, setYLabel=show_ylabel)
    where.set_zlabel("Noise")

    # Add a colorbar
    pFig.colorbar(surf, ax=where, pad=0.1, shrink=0.7, aspect=15, label="Noise")

    _finalize_layout(pFig, ax is None)
    return where


def plot_counts(
    data,
    ax=None,
    *,
    settings=None,
    label_style: str | None = None,
    Yscale=None,
    time_axis_label=None,
):
    """Plot accumulation counts as a function of probe delay.

    Parameters
    ----------
    data : Dataset1D
        Dataset to plot.
    ax : matplotlib.axes.Axes, optional
        Axis to draw into. If None, creates a new figure.
    settings : Settings, optional
        Active settings.
    label_style : str, optional
        Label delimiter style.
    Yscale : {"symlog", "log", "lin"}, optional
        Delay axis scale (defaults to time_axis_scale from settings).
    time_axis_label : str, optional
        Format string for delay axis tick labels.

    Returns
    -------
    matplotlib.axes.Axes
    """
    counts = getattr(data, "counts", None)
    if counts is None and hasattr(data, "units"):
        counts = data.units.get("counts")

    if counts is None:
        raise ValueError("No accumulation counts available for this dataset.")

    s, labelStyle, Units = data._resolve(settings, label_style, None)
    if Yscale is None:
        Yscale = s.time_axis_scale.value
    if time_axis_label is None:
        time_axis_label = s.time_axis_label.value

    created = ax is None
    pFig, where = _new_axes(ax, figsize=s.kinetics_figsize)

    cts_arr = np.asarray(counts).ravel()
    delays_arr = np.asarray(data.delays).ravel()

    where.plot(delays_arr, cts_arr, "o-", color="navy", linewidth=1.5, markersize=3, label="Counts")
    where.set_yscale("log")

    _apply_time_xscale(
        where, Yscale, delays_arr, time_axis_label=time_axis_label, settings=s, axis="x"
    )

    XUnits = {"lbl": "Delay", "ltx": Units["unitsT_ltx"]}
    YUnits = {"lbl": "Counts", "ltx": ""}
    hlp.setXYlabels(where, labelStyle, XUnits, YUnits)
    where.set_ylabel("Counts")

    Ntotal = float(np.sum(cts_arr))
    if Ntotal.is_integer():
        ntotal_str = f"{int(Ntotal):,}"
    else:
        ntotal_str = f"{Ntotal:.4g}"

    where.set_title(f"Accumulation Counts ($N_{{total}}$ = {ntotal_str})")
    where.grid(True, which="major", linestyle=":", alpha=0.5)

    if created:
        _set_figure_window_title(pFig, "counts", data)

    pFig.tight_layout()
    return where


def legend_placement(settings=None) -> dict:
    """``loc`` (and anchor) for a cut legend, as ``**kwargs`` for ``ax.legend``.

    ``"outside right"`` (the default) parks the legend beside the axis, which
    suits a stand-alone figure with many traces; every other value places it
    inside. A host drawing into a small embedded panel passes a settings copy
    that forces it inside, since an outside legend there is drawn over the
    neighbouring panel or clipped away entirely.

    Public so that a host application redrawing a legend of its own (PyRATE-TA's
    embedded panels do, after overlaying a fit) places it where the user asked
    rather than guessing.
    """
    from pymorgan import get_settings

    s = settings if settings is not None else get_settings()
    where = str(getattr(s, "legend_location", "best") or "best").strip().lower()
    if where.startswith("outside"):
        return {"loc": "center left", "bbox_to_anchor": (1, 0.5)}
    if where not in LEGEND_LOCATIONS:
        logger.debug("unknown legend_location %r; using 'best'", where)
        return {"loc": "best"}
    return {"loc": where}


def _create_legend(where, s, upd=None):
    """Create a draggable cut legend, pruned and placed as the settings ask."""
    handles, labels = where.get_legend_handles_labels()
    valid = [(h, l) for h, l in zip(handles, labels, strict=True) if l and not l.startswith("_")]

    if getattr(s, "prune_legend", False) and len(valid) > int(getattr(s, "max_legend_entries", 15)):
        K = int(s.max_legend_entries)
        if K < 1:
            K = 1
        idx = np.round(np.linspace(0, len(valid) - 1, K)).astype(int)
        idx = np.unique(idx)
        valid = [valid[i] for i in idx]

    handles = [v[0] for v in valid]
    labels = [v[1] for v in valid]

    kwargs = {
        "handlelength": 0.75,
        "labelspacing": s.legend_label_spacing,
        **legend_placement(s),
    }
    if upd is not None:
        kwargs["handler_map"] = {plt.Line2D: HandlerLine2D(update_func=upd)}

    leg = where.legend(handles, labels, **kwargs)
    leg.set_draggable(True)
    return leg


# --------------------------------------------------------------------------- #
#                              Transient spectra                              #
# --------------------------------------------------------------------------- #
def plot_spectra(
    data,
    plot_delays,
    ax=None,
    fig=None,
    *,
    detector: int = 0,
    settings=None,
    label_style: str | None = None,
    delta_a_units=None,
    x_axis_unit=None,
    secondary_axis=None,
    dualScale=None,  # deprecated: a truthy value maps to secondary_axis=True
    normY=False,
    anisotropy=False,
    roundT=None,
    flipColor=False,
    Abs=None,
    Em=None,
    doSmooth=0,
    show_xlabel=True,
    show_ylabel=True,
    **kwargs,
):
    """Transient spectra at the requested ``plot_delays``.

    Draws one spectrum per requested delay, coloured along a rainbow scale from
    early to late, with a draggable legend giving each delay in the most
    readable time unit (fs / ps / ns). Unset keywords fall back to the active
    :class:`~pymorgan.settings.Settings`.

    Parameters
    ----------
    data : Dataset1D
        Dataset to plot; supplied automatically via ``Dataset1D.plot_spectra``.
    plot_delays : sequence of float
        Delays to cut at, in the dataset's own time unit. Each is snapped to the
        nearest measured delay and duplicates are dropped, so approximate values
        are fine.
    ax : matplotlib.axes.Axes, optional
        Axis to draw into; ``None`` creates a figure sized from
        ``Settings.spectra_figsize``.
    fig : matplotlib.figure.Figure, optional
        Figure owning ``ax``; inferred from ``ax`` when omitted.
    detector : int, default 0
        Detector index for multi-detector datasets.
    settings : Settings, optional
        Settings object to resolve defaults from; defaults to the active one.
    label_style : {"()", "[]", "/", ""}, optional
        Delimiter convention for the axis labels.
    delta_a_units : str, optional
        Signal-unit convention for the Y label ("mOD", "x1E3", ...).
    x_axis_unit : {"nm", "cm-1", "eV", "THz"}, optional
        Convert the probe axis to this unit for display.
    secondary_axis : bool, optional
        Add a top axis in the complementary spectral unit.
    dualScale : bool, optional
        Deprecated alias for ``secondary_axis``.
    normY : bool, default False
        Normalise each spectrum to its own peak absolute amplitude, which
        compares band shapes independently of intensity.
    anisotropy : bool, default False
        Label the Y axis ``r(t)`` and fix its limits to [-0.3, 0.9].
    roundT : bool, optional
        Round the delays shown in the legend. Defaults to
        ``Settings.round_labels``.
    flipColor : bool, default False
        Reverse the rainbow colour order (late delays coloured as early ones).
    Abs, Em : dict or Spectrum, optional
        Steady-state absorption / emission overlay, as ``{"X": ..., "Y": ...}``
        or a :class:`~pymorgan.steadyState.Spectrum`. Drawn inverted and
        normalised on a twin axis whose zero is aligned with the transient one.
        ``Em`` is multiplied by ``X**4`` first (wavelength-to-wavenumber
        Jacobian for a fluorescence spectrum).
    doSmooth : int, default 0
        Box-filter width in points. ``> 0`` plots the smoothed trace only;
        ``< 0`` plots the smoothed trace over a faint copy of the raw one, which
        is the honest way to show how much smoothing was applied.
    show_xlabel, show_ylabel : bool, default True
        Draw the X / Y axis labels.
    **kwargs
        Forwarded to :meth:`matplotlib.axes.Axes.plot` (``linewidth``,
        ``linestyle``, ...).

    Returns
    -------
    matplotlib.axes.Axes
        The axis the spectra were drawn into.

    Notes
    -----
    Detector gaps are masked, so traces are broken rather than interpolated
    across them. When ``Settings.error_shading`` is on and the dataset carries a
    noise array, each trace is drawn with a shaded +/-1 sigma band.

    The legend is placed by ``Settings.legend_location`` -- beside the axis by
    default, which suits a single-panel figure; any Matplotlib ``loc`` puts it
    inside instead.

    Examples
    --------
    >>> data.plot_spectra([0.5, 1, 5, 50, 500], doSmooth=3, roundT=True)
    >>> data.plot_spectra([1, 10], ax=panel, x_axis_unit="nm", secondary_axis=True)
    """
    s, labelStyle, Units = data._resolve(settings, label_style, delta_a_units)
    if roundT is None:
        roundT = s.round_labels
    created = ax is None
    pFig, where = _new_axes(ax, fig, layout="constrained", figsize=s.spectra_figsize)
    Zavg_C = data._detector_slice(detector)
    Y_t = data.delays
    X_l = _get_probe(data, detector)
    SelTraces = plot_delays

    cond = np.abs(np.gradient(X_l)) > 1.5 * np.nanmedian(np.abs(np.gradient(X_l)))
    mask2D = np.tile(cond, (Zavg_C.shape[0], 1))
    Zavg_C = np.ma.masked_where(mask2D, Zavg_C)

    _, Plt_ID = hlp.find_nearest(Y_t, SelTraces, unique=True)
    Nplots = len(Plt_ID)

    if created:
        pFig.set_size_inches(*s.spectra_figsize)
    YUnits = {"lbl": Units["unitsZ_lbl"], "ltx": Units["unitsZ_ltx"]}

    if secondary_axis is None and dualScale:
        secondary_axis = True
    xplot, XUnits, _sec, _xconv = _resolve_x_axis(
        X_l, Units, s, x_axis_unit=x_axis_unit, secondary_axis=secondary_axis
    )

    if Abs is not None:
        Abs_Y = Abs["Y"][(Abs["X"] >= np.nanmin(X_l)) & (Abs["X"] <= np.nanmax(X_l))]
        Abs_X = Abs["X"][(Abs["X"] >= np.nanmin(X_l)) & (Abs["X"] <= np.nanmax(X_l))]
        Abs_Y = hlp.uniform_smooth(Abs_Y, 20)
        Abs_Y = -Abs_Y / np.nanmax(np.abs(Abs_Y))
        Abs_X = _xconv(Abs_X)

    if Em is not None:
        Em_Y = Em["Y"] * Em["X"] ** 4
        Em_Y = Em_Y[(Em["X"] >= np.nanmin(X_l)) & (Em["X"] <= np.nanmax(X_l))]
        Em_X = Em["X"][(Em["X"] >= np.nanmin(X_l)) & (Em["X"] <= np.nanmax(X_l))]
        Em_Y = hlp.uniform_smooth(Em_Y, 20)
        Em_Y = -Em_Y / np.nanmax(np.abs(Em_Y))
        Em_X = _xconv(Em_X)

    traces_cmap = kwargs.pop("traces_cmap", None)
    if traces_cmap is None:
        traces_cmap = s.traces_cmap
    cm = hlp.get_trace_cmap(traces_cmap, Nplots, reverse=flipColor)

    noise_slice = None
    if s.error_shading:
        noise_all = data.noise_array()
        if noise_all is not None:
            if noise_all.ndim == 3:
                noise_slice = noise_all[:, :, detector]
            elif noise_all.ndim == 2:
                noise_slice = noise_all

    for i in range(Nplots):
        yplot = Zavg_C[Plt_ID[i], :]
        norm_factor = 1.0
        if normY:
            norm_factor = np.nanmax(np.abs(yplot))
            if norm_factor != 0 and np.isfinite(norm_factor):
                yplot = yplot / norm_factor

        if doSmooth:
            yplot_s = hlp.uniform_smooth(yplot, np.abs(doSmooth))
            yplot_s = np.ma.masked_where(cond, yplot_s)

        [t_lbl, str_lbl] = hlp.ConvertTimeUnits(Y_t[Plt_ID[i]], Units["unitsT_ltx"], roundT=roundT)
        lbl = "%.3g %s" % (t_lbl, str_lbl)

        if noise_slice is not None:
            y_noise = noise_slice[Plt_ID[i], :]
            if normY:
                if norm_factor != 0 and np.isfinite(norm_factor):
                    y_noise = y_noise / norm_factor
            y_noise = np.ma.masked_where(cond, y_noise)
            y_center = yplot_s if doSmooth != 0 else yplot
            where.fill_between(
                xplot,
                y_center - y_noise,
                y_center + y_noise,
                color=cm[i],
                alpha=s.error_shading_alpha,
                linewidth=0,
            )

        # The line width (and style) are defaults, not fixed: an overlay -- a
        # fit drawn over its data, say -- needs to be able to thin the line or
        # replace it with markers. Passing ``linewidth``/``lw`` used to collide
        # with the hard-coded value and raise inside Matplotlib.
        line_kw = dict(kwargs)
        width = line_kw.pop("linewidth", line_kw.pop("lw", 1.5))
        style = line_kw.pop("linestyle", line_kw.pop("ls", "-"))
        if doSmooth > 0:
            where.plot(
                xplot, yplot_s, linestyle=style, linewidth=width, color=cm[i], label=lbl, **line_kw
            )
        if doSmooth < 0:
            where.plot(
                xplot, yplot_s, linestyle=style, linewidth=width, color=cm[i], label=lbl, **line_kw
            )
            where.plot(
                xplot, yplot, linestyle=style, linewidth=width, color=cm[i], alpha=0.25, **line_kw
            )
        if doSmooth == 0:
            where.plot(
                xplot, yplot, linestyle=style, linewidth=width, color=cm[i], label=lbl, **line_kw
            )

    if (Abs is not None) or (Em is not None):
        ax3 = where.twinx()
        ss_dashes = list(s.ss_line_dashes)
        if Abs is not None:
            ax3.plot(
                Abs_X,
                Abs_Y,
                color=s.ss_abs_color,
                dashes=ss_dashes,
                alpha=s.ss_line_alpha,
                linewidth=s.ss_line_width,
            )
            ax3.fill_between(Abs_X, Abs_Y, color=s.ss_abs_color, alpha=s.ss_fill_alpha)
            ax3.fill(
                np.nan,
                np.nan,
                color=s.ss_abs_color,
                alpha=s.ss_line_alpha,
                linewidth=s.ss_line_width,
                label="Abs.",
            )
        if Em is not None:
            ax3.plot(
                Em_X,
                Em_Y,
                color=s.ss_em_color,
                dashes=ss_dashes,
                alpha=s.ss_line_alpha,
                linewidth=s.ss_line_width,
            )
            ax3.fill_between(Em_X, Em_Y, color=s.ss_em_color, alpha=s.ss_fill_alpha)
            ax3.fill(
                np.nan,
                np.nan,
                color=s.ss_em_color,
                alpha=s.ss_line_alpha,
                linewidth=s.ss_line_width,
                label="Em.",
            )
        ax3.yaxis.set_visible(False)
        ax3.tick_params(right=False, left=False)

    _add_complementary_axis(where, _sec, labelStyle, side="top")

    hlp.setXYlabels(
        where, labelStyle, XUnits, YUnits, normY=normY, setXLabel=show_xlabel, setYLabel=show_ylabel
    )
    if XUnits.get("wn_scaled"):
        where.xaxis.set_major_formatter(_wn_div1000_formatter())
    where.autoscale(enable=True, axis="x", tight=True)
    where.axhline(y=0, color="0.75", linewidth=0.75)

    def upd(handle, orig):
        handle.update_from(orig)
        handle.set(linewidth=4, solid_capstyle="round")

    if anisotropy:
        where.set_ylabel(r"r(t)")
        where.set_ylim([-0.3, 0.9])

    _create_legend(where, s, upd)
    where.set_xlim(xplot[0], xplot[-1])

    if (Abs is not None) or (Em is not None):
        ax3.legend(loc="best", handlelength=0.75)
        yl = where.get_ylim()
        c_pos = (0 - yl[0]) / (yl[1] - yl[0])
        mpl_axes_aligner.shift.yaxis(ax3, 0, c_pos, expand=True)

    if created:
        _set_figure_window_title(pFig, "spectra", data)
    return where


def plot_background(
    data,
    ax=None,
    fig=None,
    *,
    detector: int = 0,
    settings=None,
    label_style: str | None = None,
    delta_a_units=None,
    x_axis_unit=None,
    secondary_axis=None,
    dualScale=None,  # deprecated: a truthy value maps to secondary_axis=True
    normY=False,
    doSmooth=0,
    color="r",
    title=None,
    show_xlabel=True,
    show_ylabel=True,
    **kwargs,
):
    """Pre-zero background spectrum, drawn with the transient-spectra rules.

    Mirrors :func:`plot_spectra` for the single averaged background trace stored
    on ``data`` (``bkg_avg``, populated by :meth:`Dataset1D.background_correct`):
    same detector-gap masking, dual-scale option, axis labels, zero line and
    autoscaling. There is no legend (a single trace).

    A flat background at zero indicates a clean pre-zero region; structure in it
    that resembles the transient spectrum usually means the background window
    overlaps the rise, or that scattered pump light reaches the detector.

    Parameters
    ----------
    data : Dataset1D
        Dataset to plot; supplied automatically via
        ``Dataset1D.plot_background``.
    ax : matplotlib.axes.Axes, optional
        Axis to draw into; ``None`` creates a figure sized from
        ``Settings.spectra_figsize``.
    fig : matplotlib.figure.Figure, optional
        Figure owning ``ax``; inferred from ``ax`` when omitted.
    detector : int, default 0
        Detector index for multi-detector datasets.
    settings : Settings, optional
        Settings object to resolve defaults from; defaults to the active one.
    label_style : {"()", "[]", "/", ""}, optional
        Delimiter convention for the axis labels.
    delta_a_units : str, optional
        Signal-unit convention for the Y label.
    x_axis_unit : {"nm", "cm-1", "eV", "THz"}, optional
        Convert the probe axis to this unit for display.
    secondary_axis : bool, optional
        Add a top axis in the complementary spectral unit.
    dualScale : bool, optional
        Deprecated alias for ``secondary_axis``.
    normY : bool, default False
        Normalise the trace to its peak absolute amplitude.
    doSmooth : int, default 0
        Box-filter width in points (``< 0`` overlays the raw trace faintly).
    colour : str, default "r"
        Line colour.
    title : str, optional
        Axis title.
    show_xlabel, show_ylabel : bool, default True
        Draw the X / Y axis labels.
    **kwargs
        Forwarded to :meth:`matplotlib.axes.Axes.plot`.

    Returns
    -------
    matplotlib.axes.Axes
        The axis the background was drawn into.

    Raises
    ------
    ValueError
        If no background has been computed; call
        :meth:`Dataset1D.background_correct` first (``do_correct=False`` stores
        the background without subtracting it).
    """
    if getattr(data, "bkg_avg", None) is None:
        raise ValueError("No background available; run background_correct first.")
    s, labelStyle, Units = data._resolve(settings, label_style, delta_a_units)
    created = ax is None
    pFig, where = _new_axes(ax, fig, layout="constrained", figsize=s.spectra_figsize)
    X_l = _get_probe(data, detector)
    yplot = np.asarray(data.bkg_avg)[:, detector]

    with np.errstate(divide="ignore", invalid="ignore"):
        grad = np.abs(np.gradient(X_l))
        cond = grad > 1.5 * np.nanmedian(grad)
        cond = np.nan_to_num(cond, nan=True)
    yplot = np.ma.masked_where(cond | np.isnan(X_l) | np.isnan(yplot), yplot)

    if created:
        pFig.set_size_inches(*s.spectra_figsize)
    YUnits = {"lbl": Units["unitsZ_lbl"], "ltx": Units["unitsZ_ltx"]}

    if secondary_axis is None and dualScale:
        secondary_axis = True
    xplot, XUnits, _sec, _ = _resolve_x_axis(
        X_l, Units, s, x_axis_unit=x_axis_unit, secondary_axis=secondary_axis
    )

    if normY:
        max_y = np.nanmax(np.abs(yplot))
        if max_y > 0 and np.isfinite(max_y):
            yplot = yplot / max_y

    if doSmooth:
        yplot_s = hlp.uniform_smooth(yplot, np.abs(doSmooth))
        yplot_s = np.ma.masked_where(cond | np.isnan(X_l), yplot_s)

    if doSmooth > 0:
        where.plot(xplot, yplot_s, "-", linewidth=1.5, color=color, **kwargs)
    elif doSmooth < 0:
        where.plot(xplot, yplot_s, "-", linewidth=1.5, color=color, **kwargs)
        where.plot(xplot, yplot, "-", linewidth=1.5, color=color, alpha=0.25, **kwargs)
    else:
        where.plot(xplot, yplot, "-", linewidth=1.5, color=color, **kwargs)

    _add_complementary_axis(where, _sec, labelStyle, side="top")

    hlp.setXYlabels(
        where, labelStyle, XUnits, YUnits, normY=normY, setXLabel=show_xlabel, setYLabel=show_ylabel
    )
    if XUnits.get("wn_scaled"):
        where.xaxis.set_major_formatter(_wn_div1000_formatter())
    where.autoscale(enable=True, axis="x", tight=True)
    where.axhline(y=0, color="0.75", linewidth=0.75)

    fin_x = xplot[np.isfinite(xplot)]
    if fin_x.size >= 2:
        where.set_xlim(fin_x[0], fin_x[-1])

    fin_y = (
        yplot.compressed()
        if isinstance(yplot, np.ma.MaskedArray)
        else yplot[np.isfinite(yplot)]
    )
    if fin_y.size == 0 or np.all(fin_y == 0) or np.nanmax(fin_y) == np.nanmin(fin_y):
        where.set_ylim(-0.1, 0.1) if not normY else where.set_ylim(-1.0, 1.0)

    if title is None:
        src = getattr(data, "source", None)
        name = pathlib.Path(src).stem if src else ""
        title = f"Background \u2014 {name}" if name else "Background"
    where.set_title(title)
    mgr = getattr(pFig.canvas, "manager", None)
    if mgr is not None and hasattr(mgr, "set_window_title"):
        mgr.set_window_title(title)

    return where


# --------------------------------------------------------------------------- #
#                               Kinetic traces                               #
# --------------------------------------------------------------------------- #
def plot_kinetics(
    data,
    plot_wavelengths,
    ax=None,
    fig=None,
    *,
    detector: int = 0,
    settings=None,
    label_style: str | None = None,
    delta_a_units=None,
    normY=False,
    Xscale=None,
    anisotropy=False,
    plotStyle="o-",
    doSmooth=0,
    round_labels=None,
    show_xlabel=True,
    show_ylabel=True,
    swap_axes=False,
    **kwargs,
):
    """Kinetic traces at the requested ``plot_wavelengths``.

    Draws the signal versus delay at one or more probe positions, on the time
    axis convention of the active :class:`~pymorgan.settings.Settings`
    (``"symlog"`` by default: linear within +/-1 time unit, logarithmic outside,
    so the rise around time zero and decades of decay share one panel).

    Parameters
    ----------
    data : Dataset1D
        Dataset to plot; supplied automatically via ``Dataset1D.plot_kinetics``.
    plot_wavelengths : sequence of float
        Probe positions to cut at, in the dataset's native probe unit (nm,
        cm-1, ...). Each is snapped to the nearest pixel and duplicates are
        dropped.
    ax : matplotlib.axes.Axes, optional
        Axis to draw into; ``None`` creates a figure sized from
        ``Settings.kinetics_figsize``.
    fig : matplotlib.figure.Figure, optional
        Figure owning ``ax``; inferred from ``ax`` when omitted.
    detector : int, default 0
        Detector index for multi-detector datasets.
    settings : Settings, optional
        Settings object to resolve defaults from; defaults to the active one.
    label_style : {"()", "[]", "/", ""}, optional
        Delimiter convention for the axis labels.
    delta_a_units : str, optional
        Signal-unit convention for the Y label ("mOD", "x1E3", ...).
    normY : bool, default False
        Normalise each trace to its own peak absolute amplitude, which compares
        decay kinetics independently of amplitude.
    Xscale : {"symlog", "log", "lin"}, optional
        Scale of the delay axis. Defaults to ``Settings.time_axis_scale``.
    anisotropy : bool, default False
        Label the Y axis ``r(t)`` and fix its limits to [-0.3, 0.5].
    doSmooth : int, default 0
        Box-filter width in points applied to each trace; ``< 0`` also draws
        the raw trace faintly underneath, so the smoothing can be judged.
    plotStyle : str, default "o-"
        Matplotlib format string for the traces, e.g. ``"-"`` for lines only.
    round_labels : bool, optional
        Round the probe positions shown in the legend. Defaults to
        ``Settings.round_labels``.
    show_xlabel, show_ylabel : bool, default True
        Draw the X / Y axis labels.
    **kwargs
        Forwarded to :meth:`matplotlib.axes.Axes.plot`. ``markersize`` and
        ``linewidth`` default to ``Settings.kinetics_marker_size`` and
        ``Settings.kinetics_line_width`` unless given explicitly (``ms`` / ``lw``
        also count as given).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis the traces were drawn into.
    t : numpy.ndarray
        Delay axis, shape ``[Ndelays]``.
    Y : numpy.ndarray
        Extracted traces, shape ``[Ndelays x Ncuts]``, in the order of
        ``plot_wavelengths`` after snapping. Useful for exporting or fitting the
        same data that was plotted.

    Notes
    -----
    When ``Settings.error_shading`` is on and the dataset carries a noise array,
    each trace is drawn with a shaded +/-1 sigma band. The legend is placed by
    ``Settings.legend_location``, beside the axis by default.

    The returned ``Y`` is the extracted data itself, so this is also the
    supported way to get traces out of PyMORGAN for fitting elsewhere (kinetic
    fitting lives in the companion project **PyRATE-TA**).

    Examples
    --------
    >>> ax, t, Y = data.plot_kinetics([2132, 2218], plotStyle="-", lw=1.5)
    >>> Y.shape
    (45, 2)
    """
    s, labelStyle, Units = data._resolve(settings, label_style, delta_a_units)
    if Xscale is None:
        Xscale = s.time_axis_scale.value
    if round_labels is None:
        round_labels = s.round_labels
    # Pop pymorgan-specific kwargs that must not reach matplotlib.plot
    time_axis_label = kwargs.pop("time_axis_label", None)

    created = ax is None
    pFig, where = _new_axes(ax, fig, layout="constrained")
    Zavg_C = data._detector_slice(detector)
    Y_t = data.delays
    X_l = _get_probe(data, detector)
    SelTraces = plot_wavelengths

    LinThres = 1
    LinSize = 0.5

    [Plt_val, Plt_ID] = hlp.find_nearest(X_l, SelTraces, unique=True)
    Nplots = len(Plt_ID)

    if created:
        pFig.set_size_inches(*s.kinetics_figsize)
    YUnits = {"lbl": Units["unitsZ_lbl"], "ltx": Units["unitsZ_ltx"]}
    traces_cmap = kwargs.pop("traces_cmap", None)
    flipColor = kwargs.pop("flipColor", False)
    if traces_cmap is None:
        traces_cmap = s.traces_cmap
    cm = hlp.get_trace_cmap(traces_cmap, Nplots, reverse=flipColor)

    # Marker size / line width default to the active settings; explicit
    # kwargs (markersize/ms, linewidth/lw) still win.
    if "markersize" not in kwargs and "ms" not in kwargs:
        kwargs["markersize"] = s.kinetics_marker_size
    if "linewidth" not in kwargs and "lw" not in kwargs:
        kwargs["linewidth"] = s.kinetics_line_width

    noise_slice = None
    if s.error_shading:
        noise_all = data.noise_array()
        if noise_all is not None:
            if noise_all.ndim == 3:
                noise_slice = noise_all[:, :, detector]
            elif noise_all.ndim == 2:
                noise_slice = noise_all

    yplot = np.zeros((len(Y_t), Nplots))
    for i in range(Nplots):
        norm_factor = 1.0
        if normY:
            norm_factor = np.max(np.abs(Zavg_C[:, Plt_ID[i]]))
            if norm_factor != 0 and np.isfinite(norm_factor):
                yplot[:, i] = Zavg_C[:, Plt_ID[i]] / norm_factor
            else:
                yplot[:, i] = Zavg_C[:, Plt_ID[i]]
        else:
            yplot[:, i] = Zavg_C[:, Plt_ID[i]]
        # Inherit the probe (X-axis) unit; round subject to the setting.
        value = round(Plt_val[i]) if round_labels else Plt_val[i]
        lbl = f"{value:g} {Units['unitsL_ltx']}"

        if noise_slice is not None:
            y_noise = noise_slice[:, Plt_ID[i]]
            if normY:
                if norm_factor != 0 and np.isfinite(norm_factor):
                    y_noise = y_noise / norm_factor
            y_low = yplot[:, i] - y_noise
            y_high = yplot[:, i] + y_noise
            filler = where.fill_betweenx if swap_axes else where.fill_between
            filler(
                Y_t,
                y_low,
                y_high,
                color=cm[i],
                alpha=s.error_shading_alpha,
                linewidth=0,
            )

        def draw(values, colour, label, **extra):
            """One trace, in whichever orientation was asked for."""
            if swap_axes:
                return where.plot(values, Y_t, plotStyle, color=colour, label=label, **extra)
            return where.plot(Y_t, values, plotStyle, color=colour, label=label, **extra)

        if doSmooth:
            smoothed = hlp.uniform_smooth(yplot[:, i], int(np.abs(doSmooth)))
            draw(smoothed, cm[i], lbl, **kwargs)
            if doSmooth < 0:
                # Negative width: the smoothed trace *over* the raw one, so the
                # smoothing can be judged rather than trusted. The caller may
                # have set alpha itself, so override rather than pass twice.
                draw(yplot[:, i], cm[i], "_nolegend_", **{**kwargs, "alpha": 0.25})
        else:
            draw(yplot[:, i], cm[i], lbl, **kwargs)

    XUnits = {"lbl": "Delay", "ltx": Units["unitsT_ltx"]}
    if swap_axes:
        # Rotated: the signal runs horizontally and the delays vertically, so
        # the panel can sit beside a contour map and share its delay axis.
        hlp.setXYlabels(
            where,
            labelStyle,
            YUnits,
            XUnits,
            normX=normY,
            setXLabel=show_ylabel,
            setYLabel=show_xlabel,
        )
        where.autoscale(enable=True, axis="y", tight=True)
        where.axvline(x=0, color="0.75", linewidth=0.75)
    else:
        hlp.setXYlabels(
            where,
            labelStyle,
            XUnits,
            YUnits,
            normY=normY,
            setXLabel=show_xlabel,
            setYLabel=show_ylabel,
        )
        where.autoscale(enable=True, axis="x", tight=True)
        where.axhline(y=0, color="0.75", linewidth=0.75)

    def upd(handle, orig):
        handle.update_from(orig)
        handle.set(linewidth=4, solid_capstyle="round")

    _create_legend(where, s, upd)

    delay_axis = "y" if swap_axes else "x"
    set_delay_scale = where.set_yscale if swap_axes else where.set_xscale
    set_delay_lim = where.set_ylim if swap_axes else where.set_xlim
    delay_zero_line = where.axhline if swap_axes else where.axvline
    delay_axis_obj = where.yaxis if swap_axes else where.xaxis

    if Xscale == "symlog":
        set_delay_scale(
            "symlog",
            linthresh=LinThres,
            linscale=LinSize,
            subs=(
                -0.9,
                -0.8,
                -0.7,
                -0.6,
                -0.5,
                -0.4,
                -0.3,
                -0.2,
                -0.1,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
            ),
        )
        set_delay_lim([Y_t[0], Y_t[-1]])
        delay_zero_line(0, color="0.25", linewidth=1.0)
        delay_axis_obj.set_major_formatter(tkr.ScalarFormatter())
    elif Xscale == "log":
        set_delay_scale("log", subs=(1, 2, 3, 4, 5, 6, 7, 8, 9))
        set_delay_lim([0.01, Y_t[-1]])
    elif Xscale == "lin":
        set_delay_scale("linear")
        set_delay_lim([Y_t[0], Y_t[-1]])
        delay_zero_line(0, color="0.25", linewidth=1.0)
    if anisotropy:
        if swap_axes:
            where.set_xlabel(r"r(t)")
            where.set_xlim([-0.3, 0.5])
        else:
            where.set_ylabel(r"r(t)")
            where.set_ylim([-0.3, 0.5])

    _apply_time_xscale(
        where, Xscale, Y_t, time_axis_label=time_axis_label, settings=s, axis=delay_axis
    )

    if created:
        _set_figure_window_title(pFig, "kinetics", data)
    return where, Y_t, yplot


# --------------------------------------------------------------------------- #
#                          Per-scan kinetics / spectra                        #
# --------------------------------------------------------------------------- #
def _bin_scan_indices(nscans, binsize, scan_ids=None):
    """Group scan indices into bins of ``binsize`` and build their labels.

    Returns a list of ``(label, index_array)`` tuples.
    """
    groups = []
    nbins = int(np.ceil(nscans / binsize))
    for b in range(nbins):
        start = b * binsize
        stop = min((b + 1) * binsize, nscans)
        idx = np.arange(start, stop)
        if scan_ids is not None and len(scan_ids) >= stop:
            if len(idx) > 1:
                lbl = f"Scans {scan_ids[start] + 1} to {scan_ids[stop-1] + 1}"
            else:
                lbl = f"Scan {scan_ids[start] + 1}"
        else:
            lbl = f"Scans {start + 1} to {stop}" if len(idx) > 1 else f"Scan {start + 1}"
        groups.append((lbl, idx))
    return groups


def _bin_scan_matrix(Z, binsize, scan_ids=None):
    """Average ``Z`` along its scan axis (axis 2) in bins of ``binsize``.

    Returns ``(Z_binned, labels)`` where ``Z_binned`` has shape
    ``(n_delays, n_pixels, n_bins)``.
    """
    groups = _bin_scan_indices(Z.shape[2], binsize, scan_ids=scan_ids)
    Zb = np.stack([np.nanmean(Z[:, :, idx], axis=2) for _, idx in groups], axis=2)
    return Zb, [label for label, _ in groups]


def _apply_time_xscale(where, Xscale, Y_t, time_axis_label=None, settings=None, axis="x"):
    """Apply the symlog/log/linear delay-axis scaling and time label formatting.

    ``axis`` selects which axis carries the delays: ``"x"`` for the usual
    orientation, ``"y"`` when the plot has been rotated so the delays run
    vertically (see ``swap_axes`` in :func:`plot_kinetics`), which is what lets
    a kinetics panel share its delay axis with a contour map beside it.
    """
    if settings is None:
        from pymorgan import get_settings

        settings = get_settings()
    if time_axis_label is None:
        time_axis_label = (
            settings.time_axis_label.value
            if hasattr(settings.time_axis_label, "value")
            else settings.time_axis_label
        )

    vertical = axis == "y"
    set_scale = where.set_yscale if vertical else where.set_xscale
    set_lim = where.set_ylim if vertical else where.set_xlim
    zero_line = where.axhline if vertical else where.axvline
    time_axis = where.yaxis if vertical else where.xaxis

    if Xscale == "symlog":
        set_scale("symlog", linthresh=1, linscale=0.5, subs=_SYMLOG_SUBS)
        set_lim([Y_t[0], Y_t[-1]])
        zero_line(0, color="0.25", linewidth=1.0)
        time_axis.set_major_formatter(_time_axis_formatter(time_axis_label))
    elif Xscale == "log":
        set_scale("log", subs=(1, 2, 3, 4, 5, 6, 7, 8, 9))
        set_lim([0.01, Y_t[-1]])
        time_axis.set_major_formatter(_time_axis_formatter(time_axis_label))
    elif Xscale == "lin":
        set_scale("linear")
        set_lim([Y_t[0], Y_t[-1]])
        zero_line(0, color="0.25", linewidth=1.0)


def _scan_legend(where, s):
    """Attach the standard draggable, right-hand cut legend."""

    def upd(handle, orig):
        handle.update_from(orig)
        handle.set(linewidth=4, solid_capstyle="round")

    return _create_legend(where, s, upd)


def plot_scan_kinetics(
    data,
    plot_wavelengths,
    ax=None,
    fig=None,
    *,
    detector: int = 0,
    binsize: int = 1,
    settings=None,
    label_style: str | None = None,
    delta_a_units=None,
    normY=False,
    Xscale=None,
    plotStyle="o-",
    round_labels=None,
    show_xlabel=True,
    show_ylabel=True,
    **kwargs,
):
    """Per-scan kinetic traces at requested ``plot_wavelengths``.

    Overlays one trace per scan group at each probe position. When more than one
    probe position is requested, the position is prefixed to each label. This is
    the diagnostic for scan-to-scan reproducibility: systematic drift between
    groups points to sample degradation or a moving alignment, whereas random
    scatter is just shot noise.

    Requires single-scan data, i.e. a directory format loaded with
    ``Settings.load_single_scans`` enabled (check
    :attr:`Dataset1D.has_single_scans`).

    Parameters
    ----------
    data : Dataset1D
        Dataset to plot; supplied automatically via
        ``Dataset1D.plot_scan_kinetics``.
    plot_wavelengths : sequence of float
        Probe positions to cut at, in the native probe unit.
    ax : matplotlib.axes.Axes, optional
        Axis to draw into; ``None`` creates a figure sized from
        ``Settings.kinetics_figsize``.
    fig : matplotlib.figure.Figure, optional
        Figure owning ``ax``; inferred from ``ax`` when omitted.
    detector : int, default 0
        Detector index for multi-detector datasets.
    binsize : int, default 1
        Number of consecutive scans averaged into each trace. ``1`` plots every
        scan; larger values trade scan resolution for signal-to-noise.
    settings : Settings, optional
        Settings object to resolve defaults from; defaults to the active one.
    label_style : {"()", "[]", "/", ""}, optional
        Delimiter convention for the axis labels.
    delta_a_units : str, optional
        Signal-unit convention for the Y label.
    normY : bool, default False
        Normalise each trace to its own peak absolute amplitude.
    Xscale : {"symlog", "log", "lin"}, optional
        Scale of the delay axis; defaults to ``Settings.time_axis_scale``.
    plotStyle : str, default "o-"
        Matplotlib format string for the traces.
    round_labels : bool, optional
        Round the probe positions in the legend; defaults to
        ``Settings.round_labels``.
    show_xlabel, show_ylabel : bool, default True
        Draw the X / Y axis labels.
    **kwargs
        Forwarded to :meth:`matplotlib.axes.Axes.plot`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis the traces were drawn into.
    t : numpy.ndarray
        Delay axis, shape ``[Ndelays]``.
    Y : numpy.ndarray
        Binned traces, shape ``[Ndelays x Npixels x Nbins]``.
    """
    s, labelStyle, Units = data._resolve(settings, label_style, delta_a_units)
    if Xscale is None:
        Xscale = s.time_axis_scale.value
    if round_labels is None:
        round_labels = s.round_labels
    # Pop pymorgan-specific kwargs that must not reach matplotlib.plot
    time_axis_label = kwargs.pop("time_axis_label", None)

    created = ax is None
    pFig, where = _new_axes(ax, fig, layout="constrained")
    if created:
        pFig.set_size_inches(*s.kinetics_figsize)

    Z = data.scan_signal(detector)  # [Ndelays x Npixels x Nscans]
    Y_t = data.delays
    X_l = _get_probe(data, detector)
    Zb, labels = _bin_scan_matrix(Z, binsize, scan_ids=data.scan_ids)
    n_bin = len(labels)

    Plt_val, Plt_ID = hlp.find_nearest(X_l, plot_wavelengths, unique=True)
    n_pos = len(Plt_ID)
    traces_cmap = kwargs.pop("traces_cmap", None)
    flipColor = kwargs.pop("flipColor", False)
    if traces_cmap is None:
        traces_cmap = s.traces_cmap
    cm = hlp.get_trace_cmap(traces_cmap, n_bin, reverse=flipColor)
    YUnits = {"lbl": Units["unitsZ_lbl"], "ltx": Units["unitsZ_ltx"]}

    if "markersize" not in kwargs and "ms" not in kwargs:
        kwargs["markersize"] = s.kinetics_marker_size
    if "linewidth" not in kwargs and "lw" not in kwargs:
        kwargs["linewidth"] = s.kinetics_line_width

    for p in range(n_pos):
        pos_val = round(Plt_val[p]) if round_labels else Plt_val[p]
        pos_lbl = "%g %s" % (pos_val, Units["unitsL_ltx"])
        for b in range(n_bin):
            y = Zb[:, Plt_ID[p], b]
            if normY:
                y = y / np.nanmax(np.abs(y))
            lbl = labels[b] if n_pos == 1 else "%s, %s" % (pos_lbl, labels[b])
            where.plot(Y_t, y, plotStyle, color=cm[b], label=lbl, **kwargs)

    XUnits = {"lbl": "Delay", "ltx": Units["unitsT_ltx"]}
    hlp.setXYlabels(
        where, labelStyle, XUnits, YUnits, normY=normY, setXLabel=show_xlabel, setYLabel=show_ylabel
    )
    where.autoscale(enable=True, axis="x", tight=True)
    where.axhline(y=0, color="0.75", linewidth=0.75)
    _scan_legend(where, s)
    _apply_time_xscale(where, Xscale, Y_t, time_axis_label=time_axis_label, settings=s)
    if created:
        _set_figure_window_title(pFig, "scan kinetics", data)
    return where, Y_t, Zb


def plot_scan_spectra(
    data,
    plot_delays,
    ax=None,
    fig=None,
    *,
    detector: int = 0,
    binsize: int = 1,
    settings=None,
    label_style: str | None = None,
    delta_a_units=None,
    x_axis_unit=None,
    secondary_axis=None,
    normY=False,
    doSmooth=0,
    roundT=None,
    show_xlabel=True,
    show_ylabel=True,
    **kwargs,
):
    """Per-scan transient spectra at the requested delay(s).

    Overlays one spectrum per scan -- or per group of ``binsize`` scans,
    averaged -- at each requested delay. Traces are coloured by scan group and
    the legend labels the groups ("Scan 3", "Scans 4 to 6"); the delay is
    prefixed when more than one delay is requested. ``doSmooth`` matches
    :func:`plot_spectra` (>0 smoothed, <0 smoothed + faint original).

    Requires single-scan data (see :attr:`Dataset1D.has_single_scans`); the
    spectral counterpart of :func:`plot_scan_kinetics`.

    Parameters
    ----------
    data : Dataset1D
        Dataset to plot; supplied automatically via
        ``Dataset1D.plot_scan_spectra``.
    plot_delays : sequence of float
        Delays to cut at, in the dataset's own time unit.
    ax : matplotlib.axes.Axes, optional
        Axis to draw into; ``None`` creates a figure sized from
        ``Settings.spectra_figsize``.
    fig : matplotlib.figure.Figure, optional
        Figure owning ``ax``; inferred from ``ax`` when omitted.
    detector : int, default 0
        Detector index for multi-detector datasets.
    binsize : int, default 1
        Number of consecutive scans averaged into each spectrum.
    settings : Settings, optional
        Settings object to resolve defaults from; defaults to the active one.
    label_style : {"()", "[]", "/", ""}, optional
        Delimiter convention for the axis labels.
    delta_a_units : str, optional
        Signal-unit convention for the Y label.
    x_axis_unit : {"nm", "cm-1", "eV", "THz"}, optional
        Convert the probe axis to this unit for display.
    secondary_axis : bool, optional
        Add a top axis in the complementary spectral unit.
    normY : bool, default False
        Normalise each spectrum to its own peak absolute amplitude.
    doSmooth : int, default 0
        Box-filter width in points (``< 0`` overlays the raw trace faintly).
    roundT : bool, optional
        Round the delays in the legend; defaults to ``Settings.round_labels``.
    show_xlabel, show_ylabel : bool, default True
        Draw the X / Y axis labels.
    **kwargs
        Forwarded to :meth:`matplotlib.axes.Axes.plot`.

    Returns
    -------
    matplotlib.axes.Axes
        The axis the spectra were drawn into.
    """
    s, labelStyle, Units = data._resolve(settings, label_style, delta_a_units)
    if roundT is None:
        roundT = s.round_labels

    created = ax is None
    pFig, where = _new_axes(ax, fig, layout="constrained", figsize=s.spectra_figsize)
    if created:
        pFig.set_size_inches(*s.spectra_figsize)

    Z = data.scan_signal(detector)  # [Ndelays x Npixels x Nscans]
    Y_t = data.delays
    X_l = _get_probe(data, detector)
    Zb, labels = _bin_scan_matrix(Z, binsize, scan_ids=data.scan_ids)
    n_bin = len(labels)

    # Mask large probe-axis gaps (e.g. detector seams), as in plot_spectra.
    cond = np.abs(np.gradient(X_l)) > 1.5 * np.nanmedian(np.abs(np.gradient(X_l)))

    _, Plt_ID = hlp.find_nearest(Y_t, plot_delays, unique=True)
    n_pos = len(Plt_ID)
    traces_cmap = kwargs.pop("traces_cmap", None)
    flipColor = kwargs.pop("flipColor", False)
    if traces_cmap is None:
        traces_cmap = s.traces_cmap
    cm = hlp.get_trace_cmap(traces_cmap, n_bin, reverse=flipColor)
    YUnits = {"lbl": Units["unitsZ_lbl"], "ltx": Units["unitsZ_ltx"]}
    xplot, XUnits, _sec, _ = _resolve_x_axis(
        X_l, Units, s, x_axis_unit=x_axis_unit, secondary_axis=secondary_axis
    )

    for p in range(n_pos):
        [t_lbl, str_lbl] = hlp.ConvertTimeUnits(Y_t[Plt_ID[p]], Units["unitsT_ltx"], roundT=roundT)
        pos_lbl = "%.3g %s" % (t_lbl, str_lbl)
        for b in range(n_bin):
            raw = Zb[Plt_ID[p], :, b]
            yplot = np.ma.masked_where(cond, raw)
            if normY:
                yplot = yplot / np.nanmax(np.abs(yplot))
            lbl = labels[b] if n_pos == 1 else "%s, %s" % (pos_lbl, labels[b])
            if doSmooth:
                ys = hlp.uniform_smooth(raw, np.abs(doSmooth))
                ys = np.ma.masked_where(cond, ys)
                if normY:
                    ys = ys / np.nanmax(np.abs(ys))
                where.plot(xplot, ys, "-", linewidth=1.5, color=cm[b], label=lbl, **kwargs)
                if doSmooth < 0:
                    where.plot(xplot, yplot, "-", linewidth=1.5, color=cm[b], alpha=0.25, **kwargs)
            else:
                where.plot(xplot, yplot, "-", linewidth=1.5, color=cm[b], label=lbl, **kwargs)

    hlp.setXYlabels(
        where, labelStyle, XUnits, YUnits, normY=normY, setXLabel=show_xlabel, setYLabel=show_ylabel
    )
    if XUnits.get("wn_scaled"):
        where.xaxis.set_major_formatter(_wn_div1000_formatter())
    _add_complementary_axis(where, _sec, labelStyle, side="top")
    where.autoscale(enable=True, axis="x", tight=True)
    where.axhline(y=0, color="0.75", linewidth=0.75)
    _scan_legend(where, s)
    where.set_xlim(xplot[0], xplot[-1])
    if created:
        _set_figure_window_title(pFig, "scan spectra", data)
    return where


# --------------------------------------------------------------------------- #
#                       Species-associated spectra (EAS/SAS)                  #
# --------------------------------------------------------------------------- #
def plot_species_spectra(
    data,
    Sfit,
    Taus,
    TauErr,
    isFixTau,
    modelType,
    ax=None,
    fig=None,
    *,
    detector: int = 0,
    settings=None,
    label_style: str | None = None,
    delta_a_units=None,
    x_axis_unit=None,
    secondary_axis=None,
    dualScale=None,  # deprecated: a truthy value maps to secondary_axis=True
    normY=False,
    anisotropy=False,
    roundT=None,
    flipColor=False,
    Abs=None,
    Em=None,
    doSmooth=0,
    printErrors=None,
    pair_round=None,
    species_labels=None,
    show_xlabel=True,
    show_ylabel=True,
    **kwargs,
):
    """Plot evolution-/species-associated spectra from a global-analysis fit.

    Draws the amplitude spectrum of each component returned by a global fit, one
    trace per component, labelled with its lifetime and (optionally) the fitted
    uncertainty. Whether the traces are EAS or SAS depends on ``modelType``: a
    parallel (sum-of-exponentials) model gives decay-associated spectra, a
    sequential model gives evolution-associated spectra.

    Parameters
    ----------
    data : Dataset1D
        Dataset the fit was performed on; supplies the probe axis. Supplied
        automatically via ``Dataset1D.plot_species_spectra``.
    Sfit : numpy.ndarray
        Component amplitude spectra, shape ``[Npixels x Ncomponents]``. Probe
        pixels that are all-NaN are dropped.
    Taus : sequence of float
        Fitted lifetime of each component, in the dataset's time unit.
    TauErr : sequence of float
        Uncertainty on each lifetime; used only when ``printErrors`` is set.
    isFixTau : sequence of bool
        Per-component flag marking lifetimes that were held fixed in the fit;
        those are labelled without an uncertainty.
    modelType : str
        Kinetic model the fit used, which decides the EAS / SAS wording in the
        legend.
    ax : matplotlib.axes.Axes, optional
        Axis to draw into; ``None`` creates a new figure.
    fig : matplotlib.figure.Figure, optional
        Figure owning ``ax``; inferred from ``ax`` when omitted.
    settings : Settings, optional
        Settings object to resolve defaults from; defaults to the active one.
    label_style : {"()", "[]", "/", ""}, optional
        Delimiter convention for the axis labels.
    delta_a_units : str, optional
        Signal-unit convention for the Y label.
    x_axis_unit : {"nm", "cm-1", "eV", "THz"}, optional
        Convert the probe axis to this unit for display.
    secondary_axis : bool, optional
        Add a top axis in the complementary spectral unit.
    dualScale : bool, optional
        Deprecated alias for ``secondary_axis``.
    normY : bool, default False
        Normalise each component spectrum to its peak absolute amplitude.
    anisotropy : bool, default False
        Label the Y axis ``r(t)`` and fix its limits.
    roundT : bool, optional
        Round the lifetimes shown in the legend to two significant figures.
        Defaults to ``Settings.round_labels``; the remaining digits are capped
        by ``Settings.label_digits``.
    flipColor : bool, default False
        Reverse the rainbow colour order.
    Abs, Em : dict or Spectrum, optional
        Steady-state absorption / emission overlay, as in :func:`plot_spectra`.
    doSmooth : int, default 0
        Box-filter width in points (``< 0`` overlays the raw trace faintly).
    printErrors : bool, default True
        Include ``TauErr`` in the legend labels.
    pair_round : bool, default False
        Round each lifetime together with its uncertainty (see
        :func:`pymorgan.helpers.round_to_uncertainty`), so neither is quoted
        more precisely than the other supports.
    show_xlabel, show_ylabel : bool, default True
        Draw the X / Y axis labels.
    **kwargs
        Forwarded to :meth:`matplotlib.axes.Axes.plot`.

    Returns
    -------
    matplotlib.axes.Axes
        The axis the spectra were drawn into.

    Notes
    -----
    The ``Sfit`` / ``Taus`` / ``TauErr`` inputs come from a global-analysis fit,
    which PyMORGAN does not perform — see the companion project **PyRATE-TA**. This
    function is the rendering half of that seam and takes plain arrays, so it is
    independent of how the fit was obtained.
    """
    s, labelStyle, Units = data._resolve(settings, label_style, delta_a_units)
    created = ax is None
    pFig, where = _new_axes(ax, fig, layout="constrained")
    X_l = _get_probe(data, detector)
    Sfit = np.asarray(Sfit)
    if Sfit.shape[0] != X_l.size:
        # A fit restricted to part of the spectral range returns spectra on that
        # window only. Silently broadcasting them onto the full probe axis would
        # mislabel every point, so say what is wrong instead: the caller should
        # pass the axis the fit used (PyRATE-TA wraps its result for exactly this).
        raise ValueError(
            f"Sfit has {Sfit.shape[0]} probe points but the dataset has {X_l.size}. "
            "Pass a dataset whose probe axis matches the fitted window."
        )

    keep = ~np.isnan(Sfit).all(axis=1)
    X_l = X_l[keep]
    Sfit = Sfit[keep, :]

    cond = np.abs(np.gradient(X_l)) > 1.5 * np.nanmedian(np.abs(np.gradient(X_l)))
    mask2D = np.tile(cond, (Sfit.shape[1], 1)).T
    Sfit = np.ma.masked_where(mask2D, Sfit)

    Nplots = Sfit.shape[1]
    if printErrors is None:
        printErrors = bool(getattr(s, "show_uncertainties", True))
    if pair_round is None:
        pair_round = bool(getattr(s, "round_uncertainties", True))
    if roundT is None:
        roundT = bool(getattr(s, "round_labels", True))

    if created:
        pFig.set_size_inches(7.5, 4.375)
    YUnits = {"lbl": Units["unitsZ_lbl"], "ltx": Units["unitsZ_ltx"]}

    if secondary_axis is None and dualScale:
        secondary_axis = True
    xplot, XUnits, _sec, _ = _resolve_x_axis(
        X_l, Units, s, x_axis_unit=x_axis_unit, secondary_axis=secondary_axis
    )

    if flipColor:
        cm = plt.cm.turbo(np.linspace(1, 0, Nplots + 1))
    else:
        cm = plt.cm.turbo(np.linspace(0, 1, Nplots + 1))
    cm = hlp.adjust_cmap(cm, 0.9)

    X_l = np.ma.masked_where(cond, X_l)

    for i in range(Nplots):
        yplot = Sfit[:, i]
        if normY:
            yplot = yplot / np.nanmax(np.abs(yplot))
        if doSmooth:
            yplot_s = hlp.uniform_smooth(yplot, np.abs(doSmooth))
            yplot_s = np.ma.masked_where(cond, yplot_s)

        name = (
            species_labels[i]
            if species_labels and i < len(species_labels)
            else string.ascii_uppercase[i]
        )
        if modelType == "Target":
            # For Target models, species spectra represent individual chemical species
            # rather than single unbranched lifetimes. Label with the species name.
            lbl = name
        elif modelType == "Sequential":
            tau_text = (
                hlp.format_lifetime_label(
                    Taus[i],
                    Units["unitsT_ltx"],
                    err=TauErr[i] if (printErrors and TauErr is not None and i < len(TauErr)) else None,
                    roundT=roundT,
                    fixed=bool(isFixTau[i]) if isFixTau is not None and i < len(isFixTau) else False,
                    pair_round=bool(pair_round),
                    settings=s,
                )
                if (Taus is not None and i < len(Taus))
                else ""
            )
            lbl = f"{name} ({tau_text})" if tau_text else name
        elif modelType == "Parallel":
            tau_text = (
                hlp.format_lifetime_label(
                    Taus[i],
                    Units["unitsT_ltx"],
                    err=TauErr[i] if (printErrors and TauErr is not None and i < len(TauErr)) else None,
                    roundT=roundT,
                    fixed=bool(isFixTau[i]) if isFixTau is not None and i < len(isFixTau) else False,
                    pair_round=bool(pair_round),
                    settings=s,
                )
                if (Taus is not None and i < len(Taus))
                else ""
            )
            if species_labels and i < len(species_labels):
                lbl = f"{name} ({tau_text})" if tau_text else name
            else:
                lbl = r"$\tau_{%i}$ = %s" % (i + 1, tau_text) if tau_text else f"Component {i + 1}"
        else:
            tau_text = (
                hlp.format_lifetime_label(
                    Taus[i],
                    Units["unitsT_ltx"],
                    err=TauErr[i] if (printErrors and TauErr is not None and i < len(TauErr)) else None,
                    roundT=roundT,
                    fixed=bool(isFixTau[i]) if isFixTau is not None and i < len(isFixTau) else False,
                    pair_round=bool(pair_round),
                    settings=s,
                )
                if (Taus is not None and i < len(Taus))
                else ""
            )
            lbl = f"{name} ({tau_text})" if tau_text else name


        if doSmooth > 0:
            where.plot(xplot, yplot_s, linewidth=1.5, color=cm[i], label=lbl, **kwargs)
        elif doSmooth < 0:
            where.plot(xplot, yplot_s, linewidth=1.5, color=cm[i], label=lbl, **kwargs)
            where.plot(xplot, yplot, linewidth=1.5, color=cm[i], alpha=0.25, **kwargs)
        else:
            where.plot(xplot, yplot, linewidth=1.5, color=cm[i], label=lbl, **kwargs)

    _add_complementary_axis(where, _sec, labelStyle, side="top")

    hlp.setXYlabels(
        where, labelStyle, XUnits, YUnits, normY=normY, setXLabel=show_xlabel, setYLabel=show_ylabel
    )
    if XUnits.get("wn_scaled"):
        where.xaxis.set_major_formatter(_wn_div1000_formatter())
    where.autoscale(enable=True, axis="x", tight=True)
    where.axhline(y=0, color="0.75", linewidth=0.75)

    def upd(handle, orig):
        handle.update_from(orig)
        handle.set(linewidth=4, solid_capstyle="round")

    where.set_xlim(xplot[0], xplot[-1])
    # Species spectra keep their legend *inside*, at "best", and draggable:
    # there are only a handful of components, the spectra rarely fill the whole
    # axis, and where the free space is depends on the data -- so Matplotlib
    # places it and the reader moves it if it still overlaps a band.
    legend = where.legend(
        loc="best", handlelength=0.75, handler_map={plt.Line2D: HandlerLine2D(update_func=upd)}
    )
    if legend is not None:
        legend.set_draggable(True)
    return where

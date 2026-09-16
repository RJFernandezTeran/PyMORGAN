"""Plotting stage of the 2-D pipeline.

The main view, :func:`plot_map`, renders a single population-time (t2) map as a
pump-vs-probe contour. It is :class:`~pymorgan.twoD.Dataset2D`-aware and builds
*from a given axis*: a colorbar and an optional steady-state / pump-probe top
panel are appended to the supplied (or freshly created) axis with a divider, so
the whole composite slots into a larger layout. A diagonal (omega_1 = omega_3)
line is drawn only where the pump and probe axes actually overlap.
"""

from __future__ import annotations

from collections import namedtuple

import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pymorgan import helpers as hlp

# Handles to the (main, top, colorbar) axes, so callers can compose further.
Map2DAxes = namedtuple("Map2DAxes", ["ax", "top", "cbar"])

# fmt2Dlabel only understands these delimiters; fall back to "()" otherwise.
_FREQ_DELIMS = ("()", "[]", "/")


def _display_axes(data, units, freq_unit, settings):
    """Conversion of the spectral axes into the display unit.

    Returns ``(convert, unit_ltx)`` where ``convert`` maps native pump/probe
    values (and anything sharing those axes: limits, overlay points, top
    spectra) into the display unit, applying the 1e3 rescaling of
    :func:`helpers.to_display_axis` when the axis is large enough to need it.
    """
    native = _freq_unit_token(data, units)
    target = freq_unit if freq_unit is not None else getattr(settings, "twoD_freq_unit", native)
    target = target.value if hasattr(target, "value") else str(target)
    if target in (None, "", "native"):
        target = native

    span = np.concatenate([np.asarray(data.pump, dtype=float), np.asarray(data.probe, dtype=float)])
    _, factor, unit_ltx = hlp.to_display_axis(span, native, target)

    def convert(values):
        if values is None:
            return None
        arr = hlp.convert_spectral(np.asarray(values, dtype=float), native, target) / factor
        return arr

    return convert, unit_ltx


def display_converter(data, freq_unit=None, settings=None):
    """Callable mapping native pump/probe values onto the displayed axes.

    The GUI keeps its limits in the dataset's own units; use this to place them
    on axes drawn by :func:`plot_map` / :func:`plot_surface` in another unit.
    """
    from ..settings import get_settings

    s = settings or get_settings()
    convert, _ = _display_axes(data, getattr(data, "units", None), freq_unit, s)
    return convert


def _freq_unit_token(data, units, default: str = "cm-1") -> str:
    """Spectral unit for the axis labels.

    ``Dataset2D.units`` is normally the unit dictionary built by the loaders,
    but a dataset may also be constructed with a bare string (``units="mOD"``),
    in which case the spectral unit comes from ``freq_units``.
    """
    if isinstance(units, dict):
        return units.get("unitsL", default)
    return str(getattr(data, "freq_units", None) or default)


def _spectrum_xy(spectrum):
    """Coerce a steady-state/PP overlay into ``(x, y, label)``.

    Accepts a ``pymorgan.Spectrum`` (uses ``.x``/``.y``/``.label``), a mapping
    with ``"X"``/``"Y"`` keys (as produced by ``Spectrum.as_overlay_dict`` /
    ``Dataset1D`` cuts), or an ``(x, y)`` pair.
    """
    if spectrum is None:
        return None, None, None
    if hasattr(spectrum, "x") and hasattr(spectrum, "y"):
        return np.asarray(spectrum.x), np.asarray(spectrum.y), getattr(spectrum, "label", None)
    if isinstance(spectrum, dict):
        return np.asarray(spectrum["X"]), np.asarray(spectrum["Y"]), spectrum.get("label")
    x, y = spectrum
    return np.asarray(x), np.asarray(y), None


def plot_map(
    data,
    t2,
    ax=None,
    *,
    settings=None,
    label_style: str | None = None,
    freq_label: str | None = None,
    cmap_ID=None,
    ShowLines=True,
    Nskip=2,
    diagonal=True,
    t2_label=True,
    aspect="equal",
    show_xlabel=True,
    show_ylabel=True,
    show_colorbar=True,
    show_colorbar_label=True,
    top_spectrum=None,
    top_label=None,
    top_height=1.2,
    white_levels=None,
    vmin: float | None = None,
    vmax: float | None = None,
    text_white_bg=True,
    cut_plot=False,
    pump_lim=None,
    probe_lim=None,
    cls_points=None,
    ivcls_points=None,
    nls_points=None,
    Nlevels=None,
    quick=None,
    freq_unit=None,
    detector: int = 0,
    filled: bool = True,
):
    """Plot the 2-D map nearest population time ``t2``.

    Draws one pump-probe correlation map as a filled contour plot with a
    symmetric, zero-centred colour scale. The pump axis runs along X by default
    and along Y when ``Settings.pump_axis`` is ``"Vertical"``; both spectral
    axes are shown in ``freq_unit`` (default ``Settings.twoD_freq_unit``), and
    limits and overlays given in the dataset's own units are converted with
    them.

    The colorbar and the optional top spectrum are appended to ``ax`` with
    :func:`~mpl_toolkits.axes_grid1.make_axes_locatable`, so the call composes
    into a ``GridSpec`` layout (see ``examples/composite_2D.py``).

    Parameters
    ----------
    data : Dataset2D
        Dataset to plot; supplied automatically via ``Dataset2D.plot_map``.
    t2 : float
        Requested population time. Snapped to the nearest measured t2, so an
        approximate value is fine; use :meth:`Dataset2D.map_index` to find out
        which map was chosen.
    ax : matplotlib.axes.Axes, optional
        Axis to draw into; ``None`` creates a 6x6-inch figure.
    settings : Settings, optional
        Settings object to resolve defaults from; defaults to the active one.
    label_style : {"()", "[]", "/", ""}, optional
        Delimiter convention for the axis labels.
    freq_label : str, optional
        Naming convention for the spectral axes (e.g. omega_1 / omega_3 vs
        pump / probe). Defaults to ``Settings.freq_label``.
    cmap_ID : str, optional
        Colourmap specification (e.g. ``"DkRd/Wh/DkBu"``, ``"vik"``, ``"berlin"``). Defaults to
        ``Settings.cmap``.
    ShowLines : bool, default True
        Overlay black contour lines on the filled map.
    Nskip : int, default 2
        Keep every ``Nskip``-th contour line, counting outwards from zero.
    Nlevels : int, optional
        Number of contour levels (rounded up to an even number). Default 30.
        Pass the same value when building a shared colorbar so the bar is
        discretised like the panels.
    diagonal : bool, default True
        Draw the ``omega_1 = omega_3`` diagonal.
    t2_label : bool, default True
        Print the population time in the upper-left corner, formatted according
        to ``Settings.t2_label_style``.
    aspect : str or float, default "equal"
        Axis aspect ratio. ``"equal"`` keeps the diagonal at 45 degrees, which
        is what makes lineshape tilt readable; pass ``None`` to let the panel
        fill its cell instead.
    show_xlabel, show_ylabel : bool, default True
        Draw the X / Y axis labels. Set ``False`` on inner panels of a row or
        column that share an axis.
    show_colorbar : bool, default True
        Append the divider colorbar. Set ``False`` when several panels share one
        externally drawn colorbar.
    show_colorbar_label : bool, default True
        Draw the unit label on the colorbar.
    top_spectrum : Spectrum or dict, optional
        Steady-state (e.g. FTIR) spectrum drawn in a panel above the map,
        sharing the X axis. Accepts a :class:`~pymorgan.steadyState.Spectrum` or
        ``{"X": ..., "Y": ..., "label": ...}``.
    top_label : str, optional
        Legend label for ``top_spectrum``; falls back to the spectrum's own.
    top_height : float, default 1.2
        Height of the top panel, in inches.
    white_levels : float, optional
        Number of central levels forced to white; those levels are also excluded
        from the black contour lines. Defaults to ``Settings.white_levels``.
    vmin, vmax : float, optional
        Explicit colour-scale limits. Given together they override the
        per-map symmetric autoscale — this is how a t2 series is put on one
        common scale so panel-to-panel intensity changes are real.
    text_white_bg : bool, default True
        Draw the t2 label on an opaque rounded box rather than directly on the
        map.
    cut_plot : bool, default False
        Crop the data to ``pump_lim`` / ``probe_lim`` before plotting (rather
        than only setting the view limits), so the colour autoscale and the
        contour levels are computed from the cropped region.
    pump_lim, probe_lim : tuple of float, optional
        ``(min, max)`` limits in the dataset's own spectral unit.
    cls_points, ivcls_points, nls_points : optional
        Centre-line-slope, inverse-CLS and nodal-line-slope point sets to
        overlay, as returned by :meth:`Dataset2D.center_line_slope`,
        :meth:`Dataset2D.doCLS` and :meth:`Dataset2D.nodal_line_slope`. Colours
        and draw style follow the ``cls_*`` / ``overlay_*`` settings.
    filled : bool, default True
        Draw the filled ``contourf`` background. ``False`` gives a line-only map.
    quick : bool, optional
        Draw a single rasterised ``pcolormesh`` and skip the contour-line
        overlay. Defaults to ``Settings.quick_plots``.
    freq_unit : str, optional
        Spectral unit for display. Defaults to ``Settings.twoD_freq_unit``.

    Returns
    -------
    Map2DAxes
        Namedtuple ``(ax, top, cbar)`` holding the main axis, the top-spectrum
        axis and the colorbar axis. ``top`` and ``cbar`` are ``None`` when not
        drawn.

    Examples
    --------
    >>> out = data.plot_map(0.5, ShowLines=True)
    >>> out.ax.set_title("t2 = 0.5 ps")

    A t2 series on one common colour scale, with the colorbar drawn separately:

    >>> vmax = float(np.nanmax(np.abs(data.Z)))
    >>> for ax, t2 in zip(axes, [0.2, 1, 5, 20]):
    ...     data.plot_map(t2, ax=ax, vmin=-vmax, vmax=vmax, show_colorbar=False)
    """
    s, lstyle, flabel, units = data._resolve(settings, label_style, freq_label)
    if cmap_ID is None:
        cmap_ID = s.cmap
    if white_levels is None:
        white_levels = s.white_levels

    idx = data.map_index(t2)
    pump = np.asarray(data.pump, dtype=float)
    probe = np.asarray(data.probe, dtype=float)
    if probe.ndim > 1:
        d = int(np.clip(detector, 0, probe.shape[1] - 1))
        probe = probe[:, d]

    Z_all = np.asarray(data.Z, dtype=float)
    if Z_all.ndim == 4:
        d = int(np.clip(detector, 0, Z_all.shape[3] - 1))
        Zmap = Z_all[:, :, idx, d]
    else:
        Zmap = Z_all[:, :, idx]

    # Everything below works in display units; the cropping just above uses the
    # native axes, so convert once the crop has been applied.
    to_display, freq_unit_ltx = _display_axes(data, units, freq_unit, s)

    if cut_plot and pump_lim is not None and probe_lim is not None:
        p_min_v, p_max_v = min(pump_lim), max(pump_lim)
        pr_min_v, pr_max_v = min(probe_lim), max(probe_lim)

        pump_indices = np.nonzero((pump >= p_min_v) & (pump <= p_max_v))[0]
        probe_indices = np.nonzero((probe >= pr_min_v) & (probe <= pr_max_v))[0]

        if len(pump_indices) > 0 and len(probe_indices) > 0:
            pump = pump[pump_indices]
            probe = probe[probe_indices]
            Zmap = Zmap[np.ix_(pump_indices, probe_indices)]

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    if aspect:
        ax.set_aspect(aspect)

    # Build the colorbar and top panel from this axis so it composes into layouts.
    divider = make_axes_locatable(ax)
    ax_top = None
    cax = None
    if top_spectrum is not None:
        ax_top = divider.append_axes("top", top_height, pad=0.2, sharex=ax)
        ax_top.xaxis.set_tick_params(labelbottom=False)
    if show_colorbar:
        cax = divider.append_axes("right", 0.15, pad=0.1)

    if vmin is not None and vmax is not None:
        Zmin = vmin
        Zmax = vmax
    else:
        Zmax = np.nanmax(np.abs(Zmap))
        if not np.isfinite(Zmax) or Zmax == 0:
            Zmax = 1.0  # degenerate (flat/empty) map: avoid non-increasing contour levels
        Zmin = -Zmax

    # Check if pump axis is vertical
    pump_axis_val = s.pump_axis
    if hasattr(pump_axis_val, "value"):
        pump_axis_val = pump_axis_val.value
    is_vertical = pump_axis_val == "Vertical"

    # Convert the axes (and everything plotted against them) for display.
    pump = to_display(pump)
    probe = to_display(probe)
    pump_lim = None if pump_lim is None else to_display(pump_lim)
    probe_lim = None if probe_lim is None else to_display(probe_lim)

    flabel_delim = lstyle if lstyle in _FREQ_DELIMS else "()"
    pumpAll, probeAll = hlp.fmt2Dlabel(flabel_delim, flabel, freq_unit_ltx)

    if is_vertical:
        X = probe
        Y = pump
        Z_contour = Zmap
        xlabel_str = probeAll
        ylabel_str = pumpAll
        default_xlim = (probe[0], probe[-1])
        default_ylim = (pump[0], pump[-1])
    else:
        X = pump
        Y = probe
        Z_contour = Zmap.T
        xlabel_str = pumpAll
        ylabel_str = probeAll
        default_xlim = (pump[0], pump[-1])
        default_ylim = (probe[0], probe[-1])

    NctrF = Nlevels if Nlevels is not None else getattr(s, "n_contours", 40)
    if NctrF % 2 != 0:
        NctrF += 1
    NctrL = Nlevels if Nlevels is not None else getattr(s, "n_contours", 40)

    if NctrL % 2 != 0:
        NctrL += 1

    nw = int(white_levels) if white_levels else 2
    cm_obj, _ = hlp.CalcCMAP(cmap_ID, NctrF, Nwhite=nw)
    if white_levels:
        cm_obj = hlp.zero_center_cmap(cm_obj, NctrF, int(white_levels))

    ctrLvl_F = np.linspace(Zmin, Zmax, NctrF)
    ctrLvl_L = np.linspace(Zmin, Zmax, NctrL)

    # Exclude white levels from black contour lines
    is_scientific = str(cmap_ID).lower() in (
        "vik",
        "cmc.vik",
        "berlin",
        "cmc.berlin",
        "vik_r",
        "cmc.vik_r",
        "berlin_r",
        "cmc.berlin_r",
    )
    if white_levels and not is_scientific:
        k = int(white_levels)
        if k > 0:
            mid = NctrL // 2
            lo = max(0, mid - k // 2)
            hi = min(NctrL, lo + k)
            mask = np.ones(len(ctrLvl_L), dtype=bool)
            mask[lo:hi] = False
            ctrLvl_L = ctrLvl_L[mask]

    if Nskip > 1:
        lp = ctrLvl_L[ctrLvl_L >= 0]
        ln = ctrLvl_L[ctrLvl_L <= 0]
        # In case mask removal emptied one side, protect slicing
        if len(ln) > 0 and len(lp) > 0:
            ctrLvl_L = np.append(np.flipud(np.flipud(ln)[::Nskip]), lp[::Nskip])

    quick = bool(s.quick_plots if quick is None else quick)
    if quick:
        # Fast path: a single image, no contour-line overlay (Settings.quick_plots).
        Ctr = ax.pcolormesh(
            X,
            Y,
            Z_contour,
            vmin=Zmin,
            vmax=Zmax,
            cmap=cm_obj,
            shading="nearest",
            rasterized=True,
        )
    elif filled:
        Ctr = ax.contourf(
            X, Y, Z_contour, vmin=Zmin, vmax=Zmax, levels=ctrLvl_F, cmap=cm_obj, extend="neither"
        )
        if ShowLines:
            ax.contour(
                X,
                Y,
                Z_contour,
                vmin=Zmin,
                vmax=Zmax,
                levels=ctrLvl_L,
                locator=tkr.LinearLocator(),
                linewidths=0.5,
                colors="k",
            )
    else:
        Ctr = ax.contour(
            X,
            Y,
            Z_contour,
            vmin=Zmin,
            vmax=Zmax,
            levels=ctrLvl_L,
            locator=tkr.LinearLocator(),
            linewidths=0.75,
            cmap=cm_obj if not ShowLines else None,
            colors="k" if ShowLines else None,
            extend="neither",
        )

    # Diagonal only where the pump and probe ranges overlap.
    if diagonal:
        lo = max(float(np.min(pump)), float(np.min(probe)))
        hi = min(float(np.max(pump)), float(np.max(probe)))
        if lo < hi:
            ax.plot([lo, hi], [lo, hi], "k", linewidth=1)

    if show_colorbar and cax is not None:
        cbar = ax.figure.colorbar(Ctr, cax=cax, ticks=tkr.AutoLocator())
        if show_colorbar_label:
            zstyle = lstyle if lstyle in _FREQ_DELIMS else "()"
            cbar.set_label(
                hlp.fmtZlabel(zstyle, units["unitsZ_lbl"], units["unitsZ_ltx"], twoLines=False)
            )
        cbar.ax.tick_params(axis="y", direction="out")

    if t2_label:
        if text_white_bg:
            props = dict(
                boxstyle="round", edgecolor="0.5", facecolor="w", alpha=0.85, linewidth=0.5
            )
        else:
            props = dict(boxstyle="round", edgecolor="none", facecolor="none", alpha=0.0)
        t2_style = getattr(s, "t2_label_style", None)
        t2_style_val = t2_style.value if hasattr(t2_style, "value") else (t2_style or "t2=")
        only_numbers = t2_style_val == "plain"
        ax.text(
            0.05,
            0.95,
            hlp.print2Ddelay(data.delays[idx], onlyNumbers=only_numbers),
            transform=ax.transAxes,
            fontsize=16,
            verticalalignment="top",
            bbox=props,
        )

    if show_xlabel:
        ax.set_xlabel(xlabel_str)
    if show_ylabel:
        ax.set_ylabel(ylabel_str)
    ax.set_xlim(default_xlim[0], default_xlim[1])
    ax.set_ylim(default_ylim[0], default_ylim[1])

    # Draw CLS, IvCLS, NLS overlays if provided. Their coordinates and fitted
    # centre lines are in the dataset's own units, so both are converted for
    # display -- the fit itself cannot simply be rescaled, because nm/eV/THz are
    # not linear in wavenumber.
    def _overlay(points, is_inverse):
        if points is None:
            return None
        fit = points[2] if len(points) == 3 else None
        xs = np.asarray(points[0], dtype=float)
        ys = np.asarray(points[1], dtype=float)
        line = None
        if fit is not None and xs.size and ys.size:
            m, c = fit
            if is_inverse:  # probe is the independent variable
                y_line = np.linspace(np.min(ys), np.max(ys), 100)
                x_line = m * y_line + c
            else:
                x_line = np.linspace(np.min(xs), np.max(xs), 100)
                y_line = m * x_line + c
            line = (to_display(x_line), to_display(y_line))
        return to_display(xs), to_display(ys), line

    for points, is_inverse, colour in (
        (cls_points, False, s.cls_color),
        (ivcls_points, True, s.ivcls_color),
        (nls_points, False, s.nls_color),
    ):
        overlay = _overlay(points, is_inverse)
        if overlay is None:
            continue
        xs, ys, line = overlay
        _draw_overlay_helper(
            ax,
            xs,
            ys,
            None,
            colour,
            s.overlay_draw_style,
            s.overlay_linewidth,
            is_vertical=is_vertical,
            is_inverse=is_inverse,
            line=line,
        )

    if ax_top is not None:
        sx, sy, slabel = _spectrum_xy(top_spectrum)
        sx = to_display(sx)
        ax_top.plot(sx, sy, color="k", linewidth=2, label=top_label or slabel)
        ax_top.axhline(y=0, linestyle="-", color="0.5", linewidth=0.5)
        ax_top.set_xlim(pump[0], pump[-1])
        if top_label or slabel:
            ax_top.legend(
                loc="upper left",
                bbox_to_anchor=(1, 1),
                fontsize=12,
                frameon=True,
                handlelength=0.75,
            )

    _x_arr = np.asarray(X, dtype=float)
    _y_arr = np.asarray(Y, dtype=float)
    _z_grid = np.asarray(Z_contour)

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

    ax.format_coord = _format_coord

    return Map2DAxes(ax=ax, top=ax_top, cbar=cax)


def _draw_overlay_helper(
    ax,
    x_raw,
    y_raw,
    fit_params,
    colour,
    style,
    lw,
    is_vertical=False,
    is_inverse=False,
    line=None,
):
    from ..settings import OverlayDrawStyle

    style_val = style.value if hasattr(style, "value") else style

    # 1. Plot the points if style is Points or Both
    if style_val in (OverlayDrawStyle.POINTS, OverlayDrawStyle.BOTH, "Points", "Both"):
        x_plot = y_raw if is_vertical else x_raw
        y_plot = x_raw if is_vertical else y_raw
        ax.plot(x_plot, y_plot, "o", color=colour, markersize=lw * 3, zorder=5)

    # 2. Plot the line if style is Lines or Both (using idealized best-fit line)
    if style_val in (OverlayDrawStyle.LINES, OverlayDrawStyle.BOTH, "Lines", "Both"):
        if line is not None:
            # Pre-computed centre line (already in display units).
            x_line, y_line = line
            x_plot = y_line if is_vertical else x_line
            y_plot = x_line if is_vertical else y_line
            ax.plot(x_plot, y_plot, "-", color=colour, linewidth=lw, zorder=5)
        elif fit_params is not None and len(x_raw) > 0:
            m, c = fit_params
            if is_inverse:
                # independent variable is probe (y_raw)
                y_line = np.linspace(np.min(y_raw), np.max(y_raw), 100)
                x_line = m * y_line + c
            else:
                # independent variable is pump (x_raw)
                x_line = np.linspace(np.min(x_raw), np.max(x_raw), 100)
                y_line = m * x_line + c

            x_plot = y_line if is_vertical else x_line
            y_plot = x_line if is_vertical else y_line
            ax.plot(x_plot, y_plot, "-", color=colour, linewidth=lw, zorder=5)
        elif len(x_raw) > 0:
            x_plot = y_raw if is_vertical else x_raw
            y_plot = x_raw if is_vertical else y_raw
            ax.plot(x_plot, y_plot, "-", color=colour, linewidth=lw, zorder=5)


def plot_surface(
    data,
    t2: float,
    ax=None,
    *,
    settings=None,
    label_style: str | None = None,
    freq_label: str | None = None,
    cmap_ID=None,
    rstride: int = 1,
    cstride: int = 1,
    alpha: float = 1.0,
    elev: float = 30.0,
    azim: float = -60.0,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    show_colorbar: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    pump_lim=None,
    probe_lim=None,
    zlim=None,
    freq_unit=None,
):
    """Plot the 2-D spectrum at population time ``t2`` as a true 3D surface plot.

    ``pump_lim`` / ``probe_lim`` (``(min, max)`` in cm-1) and ``zlim`` restrict
    the displayed axes, so the surface can be shown with the same view as the
    contour preview; ``vmin`` / ``vmax`` set the colour scale in the same way as
    :func:`plot_map`. The spectral axes follow ``Settings.pump_axis``, matching
    :func:`plot_map`: pump along X by default, swapped when it is "Vertical", and
    are shown in ``freq_unit`` (default ``Settings.twoD_freq_unit``).

    Parameters
    ----------
    data : Dataset2D
        Dataset to plot; supplied automatically via ``Dataset2D.plot_surface``.
    t2 : float
        Requested population time, snapped to the nearest measured t2.
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional
        3-D axis to draw into (``projection="3d"`` required when supplied).
    settings : Settings, optional
        Settings object to resolve defaults from; defaults to the active one.
    label_style : {"()", "[]", "/", ""}, optional
        Delimiter convention for the axis labels.
    freq_label : str, optional
        Naming convention for the spectral axes; defaults to
        ``Settings.freq_label``.
    cmap_ID : str, optional
        Colourmap specification; defaults to ``Settings.cmap``.
    rstride, cstride : int, default 1
        Row / column downsampling of the surface mesh.
    alpha : float, default 1.0
        Surface opacity.
    elev, azim : float, default 30.0 and -60.0
        Initial camera elevation and azimuth, in degrees.
    show_xlabel, show_ylabel, show_colorbar : bool, default True
        Element visibility.
    vmin, vmax : float, optional
        Explicit colour-scale limits; given together they override the symmetric
        autoscale.
    pump_lim, probe_lim : tuple of float, optional
        ``(min, max)`` view limits in the dataset's own spectral unit.
    zlim : tuple of float, optional
        ``(min, max)`` limits of the signal (Z) axis.
    freq_unit : str, optional
        Spectral unit for display; defaults to ``Settings.twoD_freq_unit``.

    Returns
    -------
    mpl_toolkits.mplot3d.axes3d.Axes3D
        The axis the surface was drawn into.
    """
    s, lstyle, flabel, units = data._resolve(settings, label_style, freq_label)
    if cmap_ID is None:
        cmap_ID = s.cmap

    idx = data.map_index(t2)
    t2_actual = data.delays[idx]
    to_display, freq_unit_ltx = _display_axes(data, units, freq_unit, s)
    pump = to_display(data.pump)
    probe = to_display(data.probe)
    pump_lim = None if pump_lim is None else to_display(pump_lim)
    probe_lim = None if probe_lim is None else to_display(probe_lim)
    Zmap = data.Z[:, :, idx]

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    # Axis orientation follows Settings.pump_axis, as in plot_map: by default the
    # pump runs along X, and "Vertical" swaps the two spectral axes.
    pump_axis_val = s.pump_axis
    if hasattr(pump_axis_val, "value"):
        pump_axis_val = pump_axis_val.value
    is_vertical = pump_axis_val == "Vertical"

    if is_vertical:
        X, Y = np.meshgrid(probe, pump)
        Z_surface = Zmap
    else:
        X, Y = np.meshgrid(pump, probe)
        Z_surface = Zmap.T

    if vmin is None or vmax is None:
        vmax_abs = np.max(np.abs(Zmap))
        if vmin is None:
            vmin = -vmax_abs
        if vmax is None:
            vmax = vmax_abs

    NctrF = 100
    nw = int(getattr(s, "white_levels", 2) or 2)
    [cm_obj, _] = hlp.CalcCMAP(cmap_ID, NctrF, Nwhite=nw)

    surf = ax.plot_surface(
        X,
        Y,
        Z_surface,
        cmap=cm_obj,
        vmin=vmin,
        vmax=vmax,
        rstride=rstride,
        cstride=cstride,
        alpha=alpha,
        linewidth=0.15,
        edgecolor=(0.2, 0.2, 0.2, 0.15),
        antialiased=True,
    )

    # Make background 3D panes and gridlines semitransparent so surface colourmap is unobscured
    ax.xaxis.pane.set_alpha(0.05)
    ax.yaxis.pane.set_alpha(0.05)
    ax.zaxis.pane.set_alpha(0.05)
    ax.xaxis.pane.set_edgecolor((0.7, 0.7, 0.7, 0.3))
    ax.yaxis.pane.set_edgecolor((0.7, 0.7, 0.7, 0.3))
    ax.zaxis.pane.set_edgecolor((0.7, 0.7, 0.7, 0.3))

    if hasattr(ax, "mouse_init"):
        ax.mouse_init()

    # Same labelling convention as plot_map (Settings.freq_label), so the surface
    # and the contour read identically.
    flabel_delim = lstyle if lstyle in _FREQ_DELIMS else "()"
    pumpAll, probeAll = hlp.fmt2Dlabel(flabel_delim, flabel, freq_unit_ltx)
    if show_xlabel:
        ax.set_xlabel(probeAll if is_vertical else pumpAll, labelpad=8, fontweight="bold")
    if show_ylabel:
        ax.set_ylabel(pumpAll if is_vertical else probeAll, labelpad=8, fontweight="bold")
    ax.set_zlabel(r"$\Delta$A", labelpad=8, fontweight="bold")
    ax.set_title(
        f"2D Surface Plot ($t_2$ = {t2_actual:.2f} ps)", fontsize=11, fontweight="bold", pad=12
    )

    # Axis limits, mapped through the same orientation, so the surface shows the
    # same window as the embedded contour.
    x_lim = probe_lim if is_vertical else pump_lim
    y_lim = pump_lim if is_vertical else probe_lim
    if x_lim is not None and None not in x_lim:
        ax.set_xlim(sorted(float(v) for v in x_lim))
    if y_lim is not None and None not in y_lim:
        ax.set_ylim(sorted(float(v) for v in y_lim))
    if zlim is not None and None not in zlim:
        ax.set_zlim(sorted(float(v) for v in zlim))

    ax.view_init(elev=elev, azim=azim)

    if show_colorbar:
        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, aspect=15)
        cbar.set_label(r"$\Delta$A", fontweight="bold")

    return ax

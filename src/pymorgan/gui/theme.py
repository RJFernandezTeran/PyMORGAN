"""Dark-/light-theme helpers for the PyQt6 GUI.

The GUI may be started with a *dark* Qt theme (``gui_theme = "Fusion Dark"`` in
``settings.toml``; see :mod:`pymorgan.gui.app`). When it is, the **embedded**
preview canvas should use a dark background so it blends with the surrounding
widgets. Figures opened in their own windows by the *Plots and Cuts* buttons are
left untouched and therefore keep the white background defined by the active
``.mplstyle`` profile.

This module is the single place that
  * builds the dark :class:`~PyQt6.QtGui.QPalette` used for the window, and
  * detects whether the running application is dark (from that palette), and
  * recolours an embedded :class:`~matplotlib.figure.Figure` accordingly.

Only :mod:`pymorgan.gui` imports it; the framework-agnostic ``pymorgan`` core
never depends on Qt.
"""

from __future__ import annotations

# Embedded-canvas colours. Light mode matches the .mplstyle default (white
# figure, black foreground); dark mode is a neutral charcoal with light ink.
LIGHT_BG = "white"
LIGHT_FG = "black"
DARK_BG = "#2b2b2b"
DARK_FG = "#e6e6e6"

# Dark-mode substitutes for the fixed trace colours used by plots that are drawn
# with hard-coded colours (the spectrometer-calibration panels: black reference,
# red convolved reference, purple experimental, orange/green fit overlays...).
# Keys are matplotlib colour specs as written at the call site; values are the
# brighter equivalents used on the charcoal background. Grey guide lines
# (``"0.5"``..``"0.75"``) are already light and are left untouched.
DARK_TRACE_MAP = {
    "black": "#e6e6e6",
    "k": "#e6e6e6",
    "red": "#ff6f6f",
    "r": "#ff6f6f",
    "blue": "#6cb6ff",
    "b": "#6cb6ff",
    "cyan": "#67e8f0",
    "green": "#5fd98a",
    "magenta": "#ff8ad8",
    "tab:blue": "#6cb6ff",
    "tab:red": "#ff6f6f",
    "tab:purple": "#c9a2ff",
    "tab:orange": "#ffb35c",
    "tab:green": "#5fd98a",
    "tab:brown": "#d9a06b",
    "#000000": "#e6e6e6",
}


def _dark_rgb_map():
    """``{(r, g, b): (r, g, b)}`` version of :data:`DARK_TRACE_MAP` (cached)."""
    global _DARK_RGB_MAP
    if _DARK_RGB_MAP is None:
        from matplotlib.colors import to_rgb

        _DARK_RGB_MAP = {
            tuple(round(c, 4) for c in to_rgb(k)): to_rgb(v) for k, v in DARK_TRACE_MAP.items()
        }
    return _DARK_RGB_MAP


_DARK_RGB_MAP = None


def dark_trace_color(colour):
    """Map a hard-coded light-theme trace colour to its dark-theme equivalent.

    Accepts colour names, hex strings and RGB(A) tuples. Unknown colours
    (including the grey guide lines) are returned unchanged, so this is safe to
    apply to any artist.
    """
    if isinstance(colour, str):
        hit = DARK_TRACE_MAP.get(colour, DARK_TRACE_MAP.get(colour.lower()))
        return hit if hit is not None else colour
    try:
        from matplotlib.colors import to_rgb

        rgb = tuple(round(c, 4) for c in to_rgb(colour))
    except (ValueError, TypeError):
        return colour
    hit = _dark_rgb_map().get(rgb)
    if hit is None:
        return colour
    if hasattr(colour, "__len__") and len(colour) == 4:
        return (*hit, colour[3])  # keep the original alpha
    return hit


def _base_color(artist, getter):
    """Return an artist's original (light-theme) colour, remembering it once."""
    if not hasattr(artist, "_pm_base_color"):
        artist._pm_base_color = getter()
    return artist._pm_base_color


def base_line_color(line):
    """Return the colour a line was plotted with, before any dark-mode remap."""
    return getattr(line, "_pm_base_color", line.get_color())


def adapt_axes_traces(ax, dark: bool):
    """Recolour hard-coded artist colours in ``ax`` for the dark theme.

    Applied to the *embedded* calibration panels so their traces stay legible on
    the dark background; the original colours are cached on each artist, so the
    function is idempotent and switching back to the light theme restores them.
    Pop-out windows keep the light figure and the original colours.
    """
    from matplotlib.text import Text

    def convert(value):
        return dark_trace_color(value) if dark else value

    for line in ax.get_lines():
        line.set_color(convert(_base_color(line, line.get_color)))

    for coll in list(ax.collections):
        if not hasattr(coll, "_pm_base_color"):
            coll._pm_base_color = (coll.get_facecolor().copy(), coll.get_edgecolor().copy())
        face, edge = coll._pm_base_color
        # ``fill_between`` stores RGBA rows; map each row through the table.
        coll.set_facecolor([convert(tuple(c)) for c in face] if len(face) else face)
        coll.set_edgecolor([convert(tuple(c)) for c in edge] if len(edge) else edge)

    for text in (ax.title, ax.xaxis.label, ax.yaxis.label):
        text.set_color(convert(_base_color(text, text.get_color)))

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        if isinstance(label, Text):
            label.set_color(convert(_base_color(label, label.get_color)))

    leg = ax.get_legend()
    if leg is not None:
        for handle in leg.legend_handles:
            if hasattr(handle, "get_color") and hasattr(handle, "set_color"):
                handle.set_color(convert(_base_color(handle, handle.get_color)))
        for text in leg.get_texts():
            text.set_color(convert(_base_color(text, text.get_color)))
        if leg.get_title() is not None:
            title = leg.get_title()
            title.set_color(convert(_base_color(title, title.get_color)))
    return ax


def is_dark_palette(app=None) -> bool:
    """Return ``True`` if the application's window colour is dark.

    The decision is taken from the active :class:`~PyQt6.QtGui.QPalette` so it
    follows whatever style/palette :mod:`pymorgan.gui.app` installed, rather
    than re-parsing the theme string. Returns ``False`` when no application
    exists yet.
    """
    from PyQt6.QtGui import QPalette
    from PyQt6.QtWidgets import QApplication

    app = app or QApplication.instance()
    if app is None:
        return False
    c = app.palette().color(QPalette.ColorRole.Window)
    # Perceived luminance (ITU-R BT.601).
    luminance = 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
    return luminance < 128


def dark_palette():
    """Build a dark :class:`~PyQt6.QtGui.QPalette` for the Fusion style."""
    from PyQt6.QtGui import QColor, QPalette

    bg = QColor(43, 43, 43)
    base = QColor(30, 30, 30)
    text = QColor(230, 230, 230)
    disabled = QColor(127, 127, 127)
    highlight = QColor(53, 132, 228)

    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, bg)
    p.setColor(QPalette.ColorRole.WindowText, text)
    p.setColor(QPalette.ColorRole.Base, base)
    p.setColor(QPalette.ColorRole.AlternateBase, bg)
    p.setColor(QPalette.ColorRole.ToolTipBase, base)
    p.setColor(QPalette.ColorRole.ToolTipText, text)
    p.setColor(QPalette.ColorRole.Text, text)
    p.setColor(QPalette.ColorRole.Button, bg)
    p.setColor(QPalette.ColorRole.ButtonText, text)
    p.setColor(QPalette.ColorRole.BrightText, QColor(255, 80, 80))
    p.setColor(QPalette.ColorRole.Link, highlight)
    p.setColor(QPalette.ColorRole.Highlight, highlight)
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))

    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        p.setColor(QPalette.ColorGroup.Disabled, role, disabled)
    return p


def dark_tab_stylesheet() -> str:
    """Qt style sheet that recolours ``MainTabs`` for the dark theme.

    The ``.ui`` ships a light tab gradient with no explicit text colour, so in
    dark mode the tab labels render dark-on-light and become unreadable. This
    mirrors the light style sheet's geometry (rounded top corners, padding) with
    dark gradients and light text, and is applied at runtime only when the
    palette is dark; the light ``.ui`` style sheet is left untouched otherwise.
    """
    return """
QTabWidget#MainTabs::pane { border-top: 1px solid #555555; }
QTabWidget#MainTabs::tab-bar { left: 5px; }
QTabWidget#MainTabs > QTabBar::tab {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #3c3c3c, stop: 0.5 #333333, stop: 1.0 #2b2b2b);
    color: #e6e6e6;
    border: 1px solid #555555;
    border-bottom-color: #555555;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    min-width: 30ex;
    padding: 4px;
}
QTabWidget#MainTabs > QTabBar::tab:selected, QTabWidget#MainTabs > QTabBar::tab:hover {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #5a5a5a, stop: 0.5 #4a4a4a, stop: 1.0 #3c3c3c);
    color: #ffffff;
}
QTabWidget#MainTabs > QTabBar::tab:selected { border-color: #777777; border-bottom-color: #555555; }
QTabWidget#MainTabs > QTabBar::tab:!selected { margin-top: 2px; }
"""


def style_figure(
    fig,
    dark: bool,
    *,
    axes_bg: str | None = None,
    axes_fg: str | None = None,
    style_text: bool = True,
):
    """Recolour every axis of an embedded ``fig`` for the given mode.

    Sets the figure and axes face colours plus the title, axis-label, spine and
    tick colours so the plot is legible on a dark (or light) background. Data
    artists (lines, contours, images, colorbar mappables) are left untouched.
    Colorbar axes are covered because they appear in ``fig.axes``.

    ``axes_bg`` / ``axes_fg`` override the colours used *inside* the axes while
    the figure surround still follows the theme (e.g. white panels on a dark
    window). With ``style_text=False`` the title, axis-label and tick-label
    colours are left alone: callers that give those texts a *semantic* colour
    (such as the calibration panels' purple experimental axis) recolour them
    themselves via :func:`adapt_axes_traces`.

    Intended for the **embedded** preview canvas only. New figures opened by the
    plot buttons must not be passed here so they keep the white ``.mplstyle``
    background.
    """
    bg = DARK_BG if dark else LIGHT_BG
    fg = DARK_FG if dark else LIGHT_FG
    a_bg = bg if axes_bg is None else axes_bg
    a_fg = fg if axes_fg is None else axes_fg

    fig.set_facecolor(bg)
    # Include child axes (the colorbar is an inset of the main axes and is
    # therefore stored in ``ax.child_axes``, not ``fig.axes``).
    axes = list(fig.axes)
    for ax in fig.axes:
        axes.extend(getattr(ax, "child_axes", ()))
    for ax in axes:
        ax.set_facecolor(a_bg)
        for spine in ax.spines.values():
            spine.set_color(a_fg)
        leg = ax.get_legend()
        if leg is not None:
            leg.get_frame().set_facecolor(a_bg)
            leg.get_frame().set_edgecolor(a_fg)
        if not style_text:
            ax.tick_params(color=a_fg, which="both")  # tick marks only, not labels
            continue
        for title_attr in ("title", "_left_title", "_right_title"):
            t_obj = getattr(ax, title_attr, None)
            if t_obj is not None:
                t_obj.set_color(a_fg)
        if hasattr(ax, "_titles"):
            for t_obj in ax._titles.values():
                if t_obj is not None:
                    t_obj.set_color(a_fg)
        ax.xaxis.label.set_color(a_fg)
        ax.yaxis.label.set_color(a_fg)
        ax.xaxis.get_offset_text().set_color(a_fg)
        ax.yaxis.get_offset_text().set_color(a_fg)
        ax.tick_params(colors=a_fg, which="both")
        if leg is not None:
            for text in leg.get_texts():
                text.set_color(a_fg)
            if leg.get_title() is not None:
                leg.get_title().set_color(a_fg)
    return fig

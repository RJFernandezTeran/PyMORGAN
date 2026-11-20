"""Embeddable matplotlib canvas with a navigation toolbar."""

from __future__ import annotations

import os

# Ensure matplotlib selects the same Qt binding we use (PyQt6).
os.environ.setdefault("QT_API", "pyqt6")

try:
    from pymorgan.gui._mpl_export_shim import ensure_mpl_export_backends

    ensure_mpl_export_backends()
except Exception:
    pass
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

try:  # toolbar moved across matplotlib versions
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as _NavToolbar
except ImportError:  # pragma: no cover
    from matplotlib.backends.backend_qt import NavigationToolbar2QT as _NavToolbar

import numpy as np
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget


def _scroll_step(event) -> float:
    """Return scroll step magnitude (+up / -down) for a Matplotlib scroll event."""
    step = float(getattr(event, "step", 0) or 0)
    if step != 0.0:
        return step
    button = getattr(event, "button", None)
    if button == "up":
        return 1.0
    if button == "down":
        return -1.0
    return 0.0


def _scroll_modifiers(event) -> tuple[bool, bool]:
    """Return ``(has_shift, has_ctrl)`` for a Matplotlib scroll event."""
    has_shift = False
    has_ctrl = False

    mods = getattr(event, "modifiers", None)
    if isinstance(mods, (list, tuple, set, frozenset)):
        tokens = {str(m).lower() for m in mods}
        has_shift |= "shift" in tokens
        has_ctrl |= bool(tokens & {"ctrl", "control", "cmd", "super", "meta"})

    key = getattr(event, "key", None)
    if isinstance(key, str) and key:
        tokens = {part.strip().lower() for part in key.split("+")}
        has_shift |= "shift" in tokens
        has_ctrl |= bool(tokens & {"ctrl", "control", "cmd", "super", "meta"})

    gui_event = getattr(event, "guiEvent", None)
    if gui_event is not None and hasattr(gui_event, "modifiers"):
        qmods = gui_event.modifiers()
        has_shift |= bool(qmods & Qt.KeyboardModifier.ShiftModifier)
        has_ctrl |= bool(
            qmods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)
        )

    if gui_event is None and key is None and mods is None:
        app = QApplication.instance()
        if app is not None:
            qmods = app.keyboardModifiers()
            has_shift |= bool(qmods & Qt.KeyboardModifier.ShiftModifier)
            has_ctrl |= bool(
                qmods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)
            )

    return has_shift, has_ctrl


def _zoom_axis_limits(
    axis, lim: tuple[float, float], center: float | None, step: float, *, base: float = 1.2
) -> tuple[float, float] | None:
    """Compute cursor-centered zoomed limits using ``axis``'s display transform."""
    lo, hi = float(lim[0]), float(lim[1])
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo == hi:
        return None

    tr = axis.get_transform()
    t_lo, t_hi = (float(v) for v in tr.transform([lo, hi]))
    if not (np.isfinite(t_lo) and np.isfinite(t_hi)) or t_lo == t_hi:
        return None

    if center is not None and np.isfinite(center):
        t_c = float(tr.transform([float(center)])[0])
        if not np.isfinite(t_c):
            t_c = 0.5 * (t_lo + t_hi)
    else:
        t_c = 0.5 * (t_lo + t_hi)

    scale = float(base) ** (-float(np.clip(step, -10.0, 10.0)))
    new_t_lo = t_c + (t_lo - t_c) * scale
    new_t_hi = t_c + (t_hi - t_c) * scale

    new_lo, new_hi = (float(v) for v in tr.inverted().transform([new_t_lo, new_t_hi]))
    if not (np.isfinite(new_lo) and np.isfinite(new_hi)) or new_lo == new_hi:
        return None
    return new_lo, new_hi


def handle_contour_scroll_zoom(event, ax, *, base: float = 1.2) -> bool:
    """Handle Shift+Scroll (X), Ctrl+Scroll (Y), and Ctrl+Shift+Scroll (both).

    Returns ``True`` when a zoom modifier was active (whether or not the pointer
    was inside ``ax``), so callers can consume the event without altering Z scale.
    """
    has_shift, has_ctrl = _scroll_modifiers(event)
    if not (has_shift or has_ctrl):
        return False

    if ax is None or getattr(event, "inaxes", None) is not ax:
        return True

    step = _scroll_step(event)
    if step == 0.0:
        return True

    canvas = getattr(ax.figure, "canvas", None)
    toolbar = getattr(canvas, "toolbar", None)
    if toolbar is not None and hasattr(toolbar, "_nav_stack") and hasattr(toolbar, "push_current"):
        try:
            if toolbar._nav_stack() is None:
                toolbar.push_current()
        except Exception:
            pass

    changed = False
    if has_shift:
        new_xlim = _zoom_axis_limits(ax.xaxis, ax.get_xlim(), getattr(event, "xdata", None), step, base=base)
        if new_xlim is not None:
            ax.set_xlim(new_xlim)
            changed = True

    if has_ctrl:
        new_ylim = _zoom_axis_limits(ax.yaxis, ax.get_ylim(), getattr(event, "ydata", None), step, base=base)
        if new_ylim is not None:
            ax.set_ylim(new_ylim)
            changed = True

    if changed and canvas is not None:
        canvas.draw_idle()
    return True


def attach_contour_scroll_zoom(ax, *, base: float = 1.2):
    """Attach Shift/Ctrl/Ctrl+Shift scroll-wheel axis zooming to ``ax``."""
    if ax is None or ax.figure is None or getattr(ax.figure, "canvas", None) is None:
        return None
    canvas = ax.figure.canvas
    prev_cid = getattr(ax.figure, "_pymorgan_contour_scroll_cid", None)
    if prev_cid is not None:
        return prev_cid
    cid = canvas.mpl_connect(
        "scroll_event", lambda ev, _ax=ax, _b=base: handle_contour_scroll_zoom(ev, _ax, base=_b)
    )
    ax.figure._pymorgan_contour_scroll_cid = cid
    return cid


class _ZoomableFigureCanvas(FigureCanvasQTAgg):
    """Qt figure canvas forwarding horizontal Shift+Wheel deltas on macOS/trackpads."""

    def wheelEvent(self, event):
        if (
            event.angleDelta().y() == 0
            and event.pixelDelta().y() == 0
            and bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            and self.figure is not None
        ):
            app = QApplication.instance()
            if event.pixelDelta().isNull() or (app is not None and app.platformName() == "xcb"):
                steps = event.angleDelta().x() / 120.0
            else:
                steps = float(event.pixelDelta().x())
                if steps == 0.0 and event.angleDelta().x() != 0:
                    steps = event.angleDelta().x() / 120.0
            if steps:
                from matplotlib.backend_bases import MouseEvent

                MouseEvent(
                    "scroll_event",
                    self,
                    *self.mouseEventCoords(event),
                    step=steps,
                    modifiers=self._mpl_modifiers(event.modifiers()),
                    guiEvent=event,
                )._process()
                return
        super().wheelEvent(event)


class MplCanvas(QWidget):
    """A widget bundling a standalone :class:`~matplotlib.figure.Figure`,
    its Qt canvas and a navigation toolbar.
    """

    def __init__(self, parent: QWidget | None = None, show_toolbar: bool = True):
        super().__init__(parent)
        self.figure = Figure(figsize=(6, 5), constrained_layout=True)
        self.canvas = _ZoomableFigureCanvas(self.figure)
        self.toolbar = _NavToolbar(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if show_toolbar:
            layout.addWidget(self.toolbar)
        else:
            self.toolbar.hide()
        layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)

    def reset(self):
        """Clear the figure and return a single fresh axis."""
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        return self.ax

    def draw(self):
        """Schedule a redraw of the canvas."""
        self.canvas.draw_idle()

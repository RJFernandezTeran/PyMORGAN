"""Interactive point-picking on the embedded contour for kinetic / spectral cuts.

When the GUI's "Interactive" toggle (``PP_Interactive_TickBox``) is checked, the
kinetic- and spectral-cut buttons hand control to a :class:`ContourPicker`
instead of a numeric-entry dialog. The user left-clicks positions on the contour
map -- kinetic cuts read the probe (X) coordinate, spectral cuts read the delay
(Y) coordinate -- right-clicks to remove the last point, and presses Enter to
finish (Escape to cancel). The picked values are returned **sorted** to a
completion callback, which opens the cut in a new figure.

A third mode, ``axis="xy"``, reads *both* coordinates of each click (used by the
Manual chirp-fit point selection, see :func:`pymorgan.oneD.chirp.fit_chirp_manual`):
points are drawn as markers rather than guide-lines, and are returned in click
order (not sorted) since the (wavelength, delay) pairing must be preserved
without sorting.

The picker is framework-agnostic: it talks to a matplotlib canvas through the
``mpl_connect`` / ``mpl_disconnect`` event API only, so it is unit-testable with a
headless Agg canvas and carries no Qt dependency. Marker guide-lines/points are
drawn as points are added and removed when picking ends.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

_LINE_KW = {"color": "0.15", "lw": 1.0, "ls": "--", "alpha": 0.9}
_POINT_KW = {"color": "0.15", "marker": "o", "ms": 7, "mfc": "none", "mew": 1.5, "alpha": 0.9}


class ContourPicker:
    """Collect axis positions by clicking a matplotlib axis.

    Parameters
    ----------
    canvas:
        The matplotlib canvas hosting ``ax`` (``mpl_connect`` / ``draw_idle``).
    ax:
        The contour axis to pick on.
    axis:
        ``"x"`` to read the probe (X) coordinate for kinetic cuts, ``"y"`` to read
        the delay (Y) coordinate for spectral cuts, or ``"xy"`` to read both
        coordinates as ``(x, y)`` pairs (manual chirp-fit point selection).
    on_done:
        Called once with the picked values when the user presses Enter (sorted
        ascending for ``"x"``/``"y"``; in click order for ``"xy"``), or with
        ``None`` if the selection is cancelled (Escape / clearing).
    line_kw:
        Optional overrides for the guide-line style (``"x"``/``"y"`` modes).
    point_kw:
        Optional overrides for the marker style (``"xy"`` mode).
    """

    def __init__(
        self,
        canvas,
        ax,
        axis: str,
        on_done: Callable[[list[float] | list[tuple[float, float]] | None], None],
        *,
        line_kw: dict | None = None,
        point_kw: dict | None = None,
    ):
        if axis not in ("x", "y", "xy"):
            raise ValueError(f"axis must be 'x', 'y' or 'xy', got {axis!r}")
        self.canvas = canvas
        self.ax = ax
        self.axis = axis
        self._on_done = on_done
        self._line_kw = {**_LINE_KW, **(line_kw or {})}
        self._point_kw = {**_POINT_KW, **(point_kw or {})}
        self.points: list = []
        self._artists: list = []
        self._cids: list[int] = []
        self._active = False

    # ------------------------------------------------------------------ #
    #                              Lifecycle                             #
    # ------------------------------------------------------------------ #
    def start(self) -> ContourPicker:
        """Connect event handlers and begin picking. Returns ``self``."""
        if self._active:
            return self
        self._active = True
        self._cids = [
            self.canvas.mpl_connect("button_press_event", self._on_click),
            self.canvas.mpl_connect("key_press_event", self._on_key),
        ]
        # Key events only reach the matplotlib canvas when it has focus. Give it
        # on the next event-loop turn: the caller may still be inside a
        # ``busy_guard`` (which disables the tab widget), and a disabled parent
        # silently rejects setFocus() -- leaving ENTER dead afterwards.
        set_focus = getattr(self.canvas, "setFocus", None)
        if callable(set_focus):
            try:
                from PyQt6.QtCore import QTimer

                QTimer.singleShot(0, set_focus)
            except Exception:  # pragma: no cover - non-Qt canvas (tests)
                set_focus()
        return self

    def cancel(self):
        """Abort picking, remove guide-lines, and report ``None``."""
        self._teardown()
        self._emit(None)

    @property
    def active(self) -> bool:
        return self._active

    # ------------------------------------------------------------------ #
    #                            Event handlers                          #
    # ------------------------------------------------------------------ #
    def _on_click(self, event):
        if not self._active or event.inaxes is not self.ax:
            return
        if event.button == 1:  # left-click: add a point
            if self.axis == "xy":
                if event.xdata is None or event.ydata is None:
                    return
                self._add((float(event.xdata), float(event.ydata)))
            else:
                value = event.xdata if self.axis == "x" else event.ydata
                if value is not None:
                    self._add(float(value))
        elif event.button == 3:  # right-click: remove the last point
            self._remove_last()

    def _on_key(self, event):
        if not self._active:
            return
        if event.key in ("enter", "return"):
            self._finish()
        elif event.key == "escape":
            self.cancel()

    # ------------------------------------------------------------------ #
    #                          Point bookkeeping                         #
    # ------------------------------------------------------------------ #
    def _add(self, value):
        self.points.append(value)
        if self.axis == "xy":
            x, y = value
            (marker,) = self.ax.plot([x], [y], **self._point_kw)
            self._artists.append(marker)
        else:
            draw_line = self.ax.axvline if self.axis == "x" else self.ax.axhline
            self._artists.append(draw_line(value, **self._line_kw))
        self._refresh()

    def _remove_last(self):
        if not self.points:
            return
        self.points.pop()
        self._artists.pop().remove()
        self._refresh()

    def _finish(self):
        # "xy" pairs must keep their pairing -- return click order, not sorted.
        values = list(self.points) if self.axis == "xy" else sorted(self.points)
        self._teardown()
        self._emit(values if values else None)

    # ------------------------------------------------------------------ #
    #                              Helpers                               #
    # ------------------------------------------------------------------ #
    def _teardown(self):
        for cid in self._cids:
            self.canvas.mpl_disconnect(cid)
        self._cids = []
        for art in self._artists:
            art.remove()
        self._artists = []
        self._active = False
        self._refresh()

    def _emit(self, values: Sequence[float] | None):
        callback, self._on_done = self._on_done, None
        if callback is not None:
            callback(list(values) if values is not None else None)

    def _refresh(self):
        draw = getattr(self.canvas, "draw_idle", None) or self.canvas.draw
        draw()

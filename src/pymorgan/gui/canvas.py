"""Embeddable matplotlib canvas with a navigation toolbar."""

from __future__ import annotations

import os

# Ensure matplotlib selects the same Qt binding we use (PyQt6).
os.environ.setdefault("QT_API", "pyqt6")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

try:  # toolbar moved across matplotlib versions
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as _NavToolbar
except ImportError:  # pragma: no cover
    from matplotlib.backends.backend_qt import NavigationToolbar2QT as _NavToolbar

from matplotlib.figure import Figure
from PyQt6.QtWidgets import QVBoxLayout, QWidget


class MplCanvas(QWidget):
    """A widget bundling a standalone :class:`~matplotlib.figure.Figure`,
    its Qt canvas and a navigation toolbar.
    """

    def __init__(self, parent: QWidget | None = None, show_toolbar: bool = True):
        super().__init__(parent)
        self.figure = Figure(figsize=(6, 5), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
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

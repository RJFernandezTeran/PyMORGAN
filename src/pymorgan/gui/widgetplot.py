"""The ``WidgetPlot`` promoted widget used by the Qt Designer ``.ui``.

The original layout promotes the plot area (``PPaxes``) to a custom
``WidgetPlot`` class. We provide it here as the embeddable matplotlib canvas
(:class:`pymorgan.gui.canvas.MplCanvas`), exposing the same ``figure`` /
``canvas`` / ``ax`` attributes the rest of the GUI uses.
"""

from __future__ import annotations

from .canvas import MplCanvas


class WidgetPlot(MplCanvas):
    """Promoted matplotlib plot widget (figure + toolbar + axis)."""

    def __init__(self, parent=None, show_toolbar: bool = True):
        super().__init__(parent, show_toolbar=show_toolbar)

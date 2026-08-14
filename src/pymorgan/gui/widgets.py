"""Custom Qt widgets promoted by ``main_window.ui``.

Kept separate from the controllers so Qt Designer can resolve the promoted
classes by header. Only ``pymorgan.gui`` depends on Qt (core stays Qt-free).
"""

from __future__ import annotations

from PyQt6.QtGui import QValidator
from PyQt6.QtWidgets import QDoubleSpinBox


class SciDoubleSpinBox(QDoubleSpinBox):
    """Double spin box displaying its value with a ``%g``-style format.

    Used for the plot-controls limit and colour-scale fields: the value is a
    plain double (the stored value keeps full precision at the box's
    ``decimals``), and free-form float input -- including scientific notation --
    is accepted. The default display format is ``%.3g``; call
    :meth:`setDisplayFormat` to override it per widget.

    A Qt ``prefix``/``suffix`` (e.g. a unit set in Qt Designer) is honoured: it
    is appended to the displayed text and stripped again when parsing input, so
    typing either ``1850`` or ``1850 cm-1`` works.
    """

    _FMT = "%.3g"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fmt = self._FMT
        self.setDecimals(6)  # internal precision; display is handled by textFromValue

    def setDisplayFormat(self, fmt) -> None:
        """Set the display format and refresh.

        ``fmt`` is either a printf-style string (e.g. ``"%.2g"``) or a callable
        ``float -> str`` for cases needing more control.
        """
        self._fmt = fmt
        self.setValue(self.value())  # re-render with the new format

    def displayFormat(self):
        return self._fmt

    def textFromValue(self, value: float) -> str:
        if callable(self._fmt):
            return self._fmt(value)
        return self._fmt % value

    def _strip_affixes(self, text: str) -> str:
        t = text.strip()
        prefix, suffix = self.prefix().strip(), self.suffix().strip()
        if prefix and t.startswith(prefix):
            t = t[len(prefix) :]
        if suffix and t.endswith(suffix):
            t = t[: -len(suffix)]
        return t.strip()

    def valueFromText(self, text: str) -> float:
        try:
            return float(self._strip_affixes(text))
        except ValueError:
            return self.value()

    def validate(self, text: str, pos: int):
        t = self._strip_affixes(text)
        if t in ("", "+", "-", ".", "+.", "-.", "e", "E", "e+", "e-"):
            return (QValidator.State.Intermediate, text, pos)
        try:
            float(t)
        except ValueError:
            return (QValidator.State.Invalid, text, pos)
        return (QValidator.State.Acceptable, text, pos)


class LEDIndicator(QDoubleSpinBox if False else __import__("PyQt6.QtWidgets", fromlist=["QWidget"]).QWidget):
    """Circular LED status indicator widget rendered with QPainter.

    States:
      - "off" / "gray": Medium metallic gray circular bulb (#a0a0a0)
      - "loading" / "yellow": In-progress glowing yellow bulb
      - "loaded" / "green": Successfully loaded glowing green bulb
      - "error" / "red": Glowing red bulb
    """

    def __init__(self, parent=None, state: str = "off", diameter: int = 16):
        super().__init__(parent)
        from PyQt6.QtCore import QSize, Qt

        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._diameter = diameter
        self._state = "off"
        self.setFixedSize(QSize(diameter, diameter))
        self.setMinimumSize(QSize(diameter, diameter))
        self.setMaximumSize(QSize(diameter, diameter))
        self.set_state(state)

    def sizeHint(self):
        from PyQt6.QtCore import QSize

        return QSize(self._diameter, self._diameter)

    def set_state(self, state: str):
        self._state = state.lower()
        self.update()  # Request repaint

    def state(self) -> str:
        return self._state

    def paintEvent(self, event):
        from PyQt6.QtGui import QBrush, QColor, QPainter, QPen, QRadialGradient

        from pymorgan.gui.theme import is_dark_palette

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        d = min(self.width(), self.height()) - 2
        if d <= 0:
            return

        r = d / 2.0
        cx = self.width() / 2.0
        cy = self.height() / 2.0

        dark = is_dark_palette()

        # Theme-aware off state: soft, low-contrast neutral gray
        if dark:
            off_colors = (QColor(110, 110, 110), QColor(60, 60, 60), QColor(80, 80, 80))
        else:
            off_colors = (QColor(235, 235, 235), QColor(170, 170, 170), QColor(150, 150, 150))

        color_map = {
            "off": off_colors,
            "gray": off_colors,
            "loading": (QColor(255, 255, 190), QColor(230, 180, 0), QColor(160, 120, 0)),
            "yellow": (QColor(255, 255, 190), QColor(230, 180, 0), QColor(160, 120, 0)),
            "loaded": (QColor(210, 255, 220), QColor(0, 200, 50), QColor(0, 120, 30)),
            "green": (QColor(210, 255, 220), QColor(0, 200, 50), QColor(0, 120, 30)),
            "error": (QColor(255, 210, 210), QColor(220, 20, 20), QColor(140, 10, 10)),
            "red": (QColor(255, 210, 210), QColor(220, 20, 20), QColor(140, 10, 10)),
        }

        c_highlight, c_base, c_border = color_map.get(self._state, off_colors)

        grad = QRadialGradient(cx - r * 0.35, cy - r * 0.35, r * 1.3)
        grad.setColorAt(0.0, c_highlight)
        grad.setColorAt(0.6, c_base)
        grad.setColorAt(1.0, c_border)

        painter.setPen(QPen(c_border, 1.2))
        painter.setBrush(QBrush(grad))
        painter.drawEllipse(int(cx - r), int(cy - r), int(d), int(d))
        painter.end()


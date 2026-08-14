"""Settings panel auto-generated from ``Settings.field_specs()``.

Each field becomes the appropriate widget (combo / spin / line edit); edits are
written back to the active settings via ``pymorgan.update_settings`` and a
``changed`` signal is emitted so the view can re-render.

The panel uses a :class:`QTabWidget` with two pages:

* **Aesthetics** – visual / presentation options (style profile, colourmap,
  font scale, axis labels, steady-state overlay style, …).
* **Data** – data-loading and treatment options (chirp boundary handling,
  single-scan loading, cut-limit inheritance, …).

Fields are routed to the correct tab via the ``"tab"`` key in
:meth:`pymorgan.Settings.field_specs`.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QWidget,
)

import pymorgan as pm
from pymorgan.log import get_logger

logger = get_logger(__name__)


def _available_qt_styles() -> list[str]:
    """Qt style names installed on this machine, with their dark variants."""
    from PyQt6.QtWidgets import QStyleFactory

    styles = list(QStyleFactory.keys())
    dark = [f"{name} Dark" for name in styles if name.lower() == "fusion"]
    return styles + dark


def _as_text(value) -> str:
    """Render a settings value for a line edit (tuples as "8.0, 6.0")."""
    if value is None:
        return ""
    if isinstance(value, (tuple, list)):
        return ", ".join(f"{v:g}" if isinstance(v, (int, float)) else str(v) for v in value)
    return str(value)

# Map the "tab" key in field_specs to a human-readable tab title.
_TAB_TITLES: dict[str, str] = {
    "common": "Common",
    "oneD": "1D Settings",
    "twoD": "2D Settings",
    "gui": "GUI && Defaults",
}


class SettingsPanel(QTabWidget):
    """Editable view of the active :class:`pymorgan.Settings`.

    Displays three tabs whose contents are auto-generated from
    :meth:`pymorgan.Settings.field_specs`:

    * **Common** – settings shared by all plots.
    * **1D Settings** – options for 1D spectroscopy.
    * **2D Settings** – options for 2D spectroscopy.
    """

    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._widgets: dict[str, object] = {}

        # Build one QFormLayout per tab category, in declaration order.
        forms: dict[str, QFormLayout] = {}
        for tab_key, title in _TAB_TITLES.items():
            page = QWidget()
            form = QFormLayout(page)
            form.setContentsMargins(8, 8, 8, 8)
            form.setVerticalSpacing(6)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self.addTab(scroll, title)
            forms[tab_key] = form

        # Offer the Qt styles this machine actually provides (plus a dark
        # variant); pymorgan.settings cannot ask Qt itself.
        pm.settings.register_gui_themes(_available_qt_styles())

        settings = pm.get_settings()
        overlay_group_box = None
        overlay_form = None
        overlay_fields = {
            "cls_color", "ivcls_color", "nls_color", "overlay_draw_style", "overlay_linewidth",
            "sd_intensity_threshold", "sd_peak_range", "sd_peak_type", "sd_interpolation"
        }


        for name, spec in pm.Settings.field_specs().items():
            tab_key = spec.get("tab", "common")
            form = forms.get(tab_key, forms["common"])

            current = getattr(settings, name)
            value = current.value if hasattr(current, "value") else current
            widget = self._make_widget(name, spec, value)
            tooltip = spec.get("tooltip")
            if tooltip:
                widget.setToolTip(tooltip)
            self._widgets[name] = widget

            if name in overlay_fields:
                if overlay_group_box is None:
                    overlay_group_box = QGroupBox("Spectral Diffusion Overlay Settings")
                    overlay_form = QFormLayout(overlay_group_box)
                    overlay_form.setContentsMargins(8, 8, 8, 8)
                    overlay_form.setVerticalSpacing(6)
                    form.addRow(overlay_group_box)
                overlay_form.addRow(spec["label"], widget)
            else:
                form.addRow(spec["label"], widget)


    def _make_widget(self, name: str, spec: dict, value):
        kind = spec["kind"]
        if kind == "choice":
            widget = QComboBox()
            widget.addItems([str(c) for c in spec["choices"]])
            widget.setCurrentText(str(value))
            widget.currentTextChanged.connect(lambda v, n=name: self._apply(n, v))
        elif kind == "float":
            widget = QDoubleSpinBox()
            widget.setRange(float(spec.get("min", 0.0)), float(spec.get("max", 10.0)))
            widget.setSingleStep(float(spec.get("step", 0.1)))
            widget.setValue(float(value))
            widget.valueChanged.connect(lambda v, n=name: self._apply(n, v))
        elif kind == "int":
            widget = QSpinBox()
            widget.setRange(int(spec.get("min", 0)), int(spec.get("max", 100)))
            widget.setSingleStep(int(spec.get("step", 1)))
            widget.setValue(int(value))
            widget.valueChanged.connect(lambda v, n=name: self._apply(n, v))
        elif kind == "bool":
            widget = QCheckBox()
            widget.setChecked(bool(value))
            widget.toggled.connect(lambda v, n=name: self._apply(n, v))
        else:  # text
            widget = QLineEdit(_as_text(value))
            widget.editingFinished.connect(lambda n=name, w=widget: self._apply(n, w.text()))
        return widget

    def _apply(self, name: str, value):
        """Write one field back to the active settings, ignoring invalid input.

        Free-text fields (paths, figure sizes) can hold anything the user types;
        a value the setting cannot accept is reported and the widget is reset to
        the value still in force, rather than raising inside the Qt slot.
        """
        try:
            pm.update_settings(**{name: value})
        except (TypeError, ValueError) as err:
            logger.warning("Ignoring invalid value for %s: %s", name, err)
            widget = self._widgets.get(name)
            current = getattr(pm.get_settings(), name, None)
            if isinstance(widget, QLineEdit):
                widget.setText(_as_text(current))
            return
        self.changed.emit()

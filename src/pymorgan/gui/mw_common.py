"""Shared constants and helpers for the main-window modules.

Item roles for the dataset-browser model, the widget groups shown only while a
dataset is loaded, the data-type display-name mapping and the small pure helpers
used by :class:`~pymorgan.gui.main_window.MainWindow` and its per-tab mixins.
Kept in a module of its own so the mixins in :mod:`pymorgan.gui.tabs` can import
them without importing the main window (which imports the mixins).
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyqt6")


from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
)

# Item roles for the dataset-browser model.
_PATH_ROLE = Qt.ItemDataRole.UserRole
_ISDIR_ROLE = Qt.ItemDataRole.UserRole + 1
# Coalescing window for embedded-contour re-renders (slider drags, spin boxes).
_RENDER_DEBOUNCE_MS = 60

# Default delays (ps) suggested in the transient-spectra prompt.
_DEFAULT_SPEC_DELAYS = [0.25, 0.5, 1, 2, 3, 4, 5, 10, 20, 50, 100, 500, 1500, 5000]

# Group boxes (and the plot area) shown only while a dataset is loaded:
#   oneD_PlotsCuts_box (Plots and Cuts), oneD_PreProcessing_box (Pre-processing),
#   oneD_singleScan_box (Explore Individual Scans), PPaxes (embedded plot).
_DATASET_WIDGETS = [
    "oneD_PlotsCuts_box",
    "oneD_PreProcessing_box",
    "oneD_singleScan_box",
    "PP_plotCounts_btn",
    "PPaxes",
    "PC_box",
]

_DATASET_WIDGETS_2D = [
    "twoD_PlotsCuts_box",
    "twoDaxes",
    "twoD_PC_box",
    "twoD_subtabs",
]

# Display-to-shortname & shortname-to-display mappings for 1D and 2D data selection
DATA_TYPE_DISPLAY_NAMES: dict[str, str] = {
    # 1D Datatypes
    "PDAT": "Processed 1D Spectrum (.pdat)",
    "HARPIA_TA": "Light Conversion HARPIA-TA (.dat)",
    "UniGE_fsTA": "UniGE fsTA",
    "UniGE_nsTA": "UniGE nsTA",
    "MESS_TRIR": "UniGE Transient IR (MESS)",
    "MESS_TRUVIS": "UniGE Transient UV-Vis (MESS)",
    "UniGE_FLUPSold": "UniGE FLUPS (Legacy)",
    "UniGE_FLUPSnew": "UniGE FLUPS (Modern)",
    "Helios_TA": "Helios TA (Ultrafast Systems)",
    "UoS_IRpp": "U. of Sheffield IR Pump-Probe",
    "Exported_TXT": "Exported TXT",
    # 2D Datatypes
    "P2DAT": "Processed 2D Spectrum (.p2dat)",
    "MESS_2DIR": "UniGE 2D-IR (MESS Raw)",
    "UoS_2DIR": "U. of Sheffield 2D-IR",
    "RAL_RAW": "RAL 2D-IR (Raw)",
    "RAL_Proc": "RAL 2D-IR (Processed)",
}


def get_combo_datatype(combo: QComboBox | None, default: str) -> str:
    """Return the internal short data_type string for a QComboBox selection."""
    if combo is None:
        return default
    data = combo.currentData()
    if data is not None and str(data):
        return str(data)
    text = combo.currentText().strip()
    for short_name, disp_name in DATA_TYPE_DISPLAY_NAMES.items():
        if text in (short_name, disp_name):
            return short_name
    return text


def set_combo_datatype(combo: QComboBox | None, datatype: str) -> None:
    """Set the QComboBox selection using the internal short data_type string or display text."""
    if combo is None or not datatype:
        return
    idx = combo.findData(datatype)
    if idx < 0:
        disp = DATA_TYPE_DISPLAY_NAMES.get(datatype, datatype)
        idx = combo.findText(disp)
    if idx < 0:
        idx = combo.findText(datatype)
    if idx >= 0:
        if combo.currentIndex() == idx:
            combo.currentTextChanged.emit(combo.itemText(idx))
        else:
            combo.setCurrentIndex(idx)


def populate_datatype_combo(combo: QComboBox, loaders: list[str], default_type: str) -> None:
    """Populate a QComboBox with friendly display names and userData=short_name pairs."""
    combo.blockSignals(True)
    combo.clear()
    for short_name in loaders:
        disp_name = DATA_TYPE_DISPLAY_NAMES.get(short_name, short_name)
        combo.addItem(disp_name, userData=short_name)
    combo.blockSignals(False)
    set_combo_datatype(combo, default_type)


def _safe_set_limits(ax, xlim, ylim):
    """Apply axis limits, skipping a non-positive lower bound on a log axis.

    Avoids matplotlib's "non-positive ylim on a log-scaled axis" warning when
    the stored delay limits include the pre-zero (negative) region.
    """
    xmin, xmax = xlim
    if xmin == xmax:
        xmin, xmax = xmin - 1.0, xmax + 1.0
    if ax.get_xscale() == "log" and xmin <= 0:
        ax.set_xlim(right=xmax)
    else:
        ax.set_xlim(xmin, xmax)
    ymin, ymax = ylim
    if ymin == ymax:
        ymin, ymax = ymin - 1.0, ymax + 1.0
    if ax.get_yscale() == "log" and ymin <= 0:
        ax.set_ylim(top=ymax)
    else:
        ax.set_ylim(ymin, ymax)

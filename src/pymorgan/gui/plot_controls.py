"""Plot-controls controller for the embedded Pump-Probe contour preview.

The control *widgets* (X / Y axis-limit spin boxes, the colour-scale range with a
percentage slider and a symmetric-scale lock, the dataset Z read-out, a Restore
Limits button, and the contour-rendering controls -- colour scheme, number of
contours, white-level band, filled vs. line contours, line spacing, plus the
time-axis scale and label format) are now declared in ``main_window.ui`` inside
the ``PC_box`` group box, so the panel is editable in Qt Designer. This module no
longer *builds* those widgets: :class:`PlotControlsPanel` is a controller that
binds to the existing widgets by objectName, populates the choice combos from
:class:`pymorgan.Settings`, wires the interactions, and exposes the same public
API the rest of the GUI relies on.

The time-axis scale, label and colour scheme write to the active
:class:`pymorgan.Settings` (single source of truth); the remaining controls are
per-view and are passed to :func:`pymorgan.oneD.plot.plot_contour` via
:meth:`contour_kwargs`.

Limit edits emit :attr:`limitsChanged` (re-apply X/Y limits only); every other
change emits :attr:`renderRequested` (a full re-plot is needed).
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QObject, pyqtSignal
from PyQt6.QtWidgets import QLabel, QSpinBox

import pymorgan as pm
from pymorgan import helpers as hlp

# Diverging colourmaps understood by helpers.CalcCMAP.
_CMAPS = ["DkRd/Wh/DkBu", "Rd/Wh/Bu v2", "Seismic", "Jet"]
# Short probe-axis tokens for the X-limit labels, keyed by units2dic quantity.
_X_TOKEN = {"Wavelength": "WL", "Wavenumber": "WN", "Energy": "E"}
# Short axis tokens keyed by the (possibly converted) display unit.
_X_UNIT_TOKEN = {"nm": "WL", "cm-1": "WN", "eV": "E", "THz": "f"}

# Colour-scale percentage range shared by the 1D and 2D "Scale %" spin boxes.
_PCT_MIN, _PCT_MAX = float(np.finfo(float).eps), 100.0


def _fmt_pct(value: float) -> str:
    """Format a colour-scale percentage as ``%.2g``.

    Plain decimal notation is kept for values of order 1-100 (``100`` rather than
    ``1e+02``), where ``%.2g`` would otherwise switch to an exponent.
    """
    text = "%.2g" % value
    if "e" in text and 1.0 <= abs(value) <= 100.0:
        return "%g" % float("%.3g" % value)
    return text


def _fmt_freq(value: float) -> str:
    """Format a spectral-axis limit in plain decimal notation.

    Wavenumbers are 3-4 digit numbers, where the default ``%.3g`` would switch to
    an exponent (``2.11e+03``); one decimal is kept instead (``2105.2``). Smaller
    units (eV, and probe axes in nm) fall back to ``%.4g`` so that values such as
    1.55 eV keep their significant digits.
    """
    if abs(value) >= 100.0:
        return "%.1f" % value
    return "%.4g" % value


def _configure_pct_spin(spin) -> None:
    """Let a ``Scale %`` spin box take any float double in ``[eps, 100]``.

    The range and format also live in ``main_window.ui``; this enforces them for
    widgets bound at runtime (and for spin boxes that predate the change).
    """
    if spin is None:
        return
    spin.blockSignals(True)
    try:
        spin.setRange(_PCT_MIN, _PCT_MAX)
        spin.setDecimals(16)
        spin.setSingleStep(0.1)
        if hasattr(spin, "setDisplayFormat"):
            spin.setDisplayFormat(_fmt_pct)
        spin.setToolTip(f"Colour-scale percentage of max |ΔA| ({_PCT_MIN:g} to {_PCT_MAX:g} %)")
    except (TypeError, AttributeError):
        pass  # an integer QSpinBox cannot take the float range
    finally:
        spin.blockSignals(False)


# Controller-attribute -> objectName of the widget declared in main_window.ui.
_WIDGETS = {
    "x_min": "PC_xMin",
    "x_max": "PC_xMax",
    "y_min": "PC_yMin",
    "y_max": "PC_yMax",
    "z_min": "PC_zMin",
    "z_max": "PC_zMax",
    "_x_lbl_min": "PC_xMinLabel",
    "_x_lbl_max": "PC_xMaxLabel",
    "white_spin": "PC_whiteLevels",
    "cmb_cmap": "PC_cmap",
    "sym_chk": "PC_symmetric",
    "filled_chk": "PC_filled",
    "z_range": "PC_zReadout",
    "restore_btn": "PC_restore",
    "cmb_tlabel": "PC_timeLabel",
    "cmb_scale": "PC_timeScale",
    "n_contours": "PC_nContours",
    "showlines_chk": "PC_showLines",
    "arcsinh_chk": "PC_arcsinh",
    "arcsinh_pct": "PC_arcsinhPct",
    "n_skip": "PC_nSkip",
    "z_slider": "PC_zSlider",
    "z_pct": "PC_zPct",
    "smooth_spin": "PC_smooth",
    "restrict_z_chk": "PC_restrictZ",
    "autoy_chk": "PC_autoYcuts",
    "cmb_xunit": "PC_xUnit",
    "secondary_chk": "PC_secondaryAxis",
}


class PlotControlsPanel(QObject):
    """Binds and drives the ``PC_box`` plot-controls widgets from the .ui.

    ``host`` is the loaded main window (the ``PC_*`` widgets are its children).
    """

    renderRequested = pyqtSignal()  # needs a full re-plot
    limitsChanged = pyqtSignal()  # only re-apply X/Y limits

    def __init__(self, host, parent=None):
        super().__init__(parent)
        self._zabs = 1.0  # |Z| max used as the 100% colour-scale reference
        self._zabs_full = 1.0  # |Z| max over the whole dataset (restrict-off ref)
        self._zfull_lo = -1.0  # full-dataset signal min (read-out, restrict off)
        self._zfull_hi = 1.0  # full-dataset signal max (read-out, restrict off)
        self._dataset = None
        self._x_display = "nm"  # current spectral display unit of the X spin boxes
        self._x_scaled = False

        _OPTIONAL_WIDGETS = {"arcsinh_pct"}
        missing = [
            n for attr, n in _WIDGETS.items()
            if attr not in _OPTIONAL_WIDGETS and getattr(host, n, None) is None
        ]
        if missing:
            raise RuntimeError(f"PC_box widgets missing from main_window.ui: {missing}")
        for attr, name in _WIDGETS.items():
            setattr(self, attr, getattr(host, name, None))

        # The choice combos are authoritative from Settings, not the .ui items.
        self._populate_choice_combo(self.cmb_scale, "time_axis_scale")
        self._populate_choice_combo(self.cmb_tlabel, "time_axis_label")
        self._set_combo_items(self.cmb_cmap, _CMAPS)
        self._populate_choice_combo(self.cmb_xunit, "x_axis_unit")

        self.sync_from_settings()

        _configure_pct_spin(self.z_pct)

        # Configure n_contours default (from settings, default 40), step (2), min (2), and coerce odd numbers
        if self.n_contours is not None:
            s = pm.get_settings()
            default_n = int(getattr(s, "n_contours", 40))
            self.n_contours.blockSignals(True)
            self.n_contours.setRange(2, 400)
            self.n_contours.setSingleStep(2)
            self.n_contours.setValue(default_n)
            self.n_contours.blockSignals(False)
            self._prev_n_contours = default_n



            def on_n_contours_changed(val):
                if val % 2 != 0:
                    if val < self._prev_n_contours:
                        even_val = val - 1
                    else:
                        even_val = val + 1
                    if even_val < self.n_contours.minimum():
                        even_val = self.n_contours.minimum()
                    self.n_contours.blockSignals(True)
                    self.n_contours.setValue(even_val)
                    self.n_contours.blockSignals(False)
                    self._prev_n_contours = even_val
                else:
                    self._prev_n_contours = val

            self.n_contours.valueChanged.connect(on_n_contours_changed)

        # Detector selection spinner (only visible for datasets with >1 detector)
        self.detector_lbl = getattr(host, "PC_detectorLabel", None)
        self.detector_spin = getattr(host, "PC_detector", None)
        if self.detector_spin is None:
            box = getattr(host, "PC_box", None)
            if box is not None:
                layout = box.layout()
                self.detector_lbl = QLabel("Detector:", box)
                self.detector_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.detector_spin = QSpinBox(box)
                self.detector_spin.setRange(1, 1)
                self.detector_spin.setFixedWidth(56)
                self.detector_spin.setToolTip("Select active detector (1-based)")
                if layout is not None:
                    layout.addWidget(self.detector_lbl, 6, 5)
                    layout.addWidget(self.detector_spin, 6, 6)

        if self.detector_lbl is not None and self.detector_spin is not None:
            self.detector_lbl.setVisible(False)
            self.detector_spin.setVisible(False)
            self.detector_spin.valueChanged.connect(self._on_detector_changed)

        self._connect_signals()

    # ------------------------------------------------------------------ #
    #                            Construction                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _set_combo_items(combo, items):
        combo.blockSignals(True)
        combo.clear()
        combo.addItems([str(c) for c in items])
        combo.blockSignals(False)

    def _populate_choice_combo(self, combo, field: str):
        spec = pm.Settings.field_specs()[field]
        self._set_combo_items(combo, spec["choices"])

    def _connect_signals(self):
        for w in (self.x_min, self.x_max, self.y_min, self.y_max):
            w.valueChanged.connect(self._on_xy_changed)
        self.restrict_z_chk.toggled.connect(lambda *_: self._refresh_zref())
        self.z_min.valueChanged.connect(lambda *_: self._on_z_edited(self.z_min, self.z_max))
        self.z_max.valueChanged.connect(lambda *_: self._on_z_edited(self.z_max, self.z_min))
        self.z_slider.valueChanged.connect(self._on_slider)
        self.z_pct.valueChanged.connect(self._on_pct_spin)
        for w in (self.n_contours, self.n_skip):
            w.valueChanged.connect(lambda *_: self.renderRequested.emit())
        self.white_spin.valueChanged.connect(lambda v: self._on_setting("white_levels", float(v)))
        for chk in (self.filled_chk, self.showlines_chk):
            chk.toggled.connect(lambda *_: self.renderRequested.emit())
        self.arcsinh_chk.toggled.connect(self._on_arcsinh_toggled)
        if hasattr(self, "arcsinh_pct") and self.arcsinh_pct is not None:
            self.arcsinh_pct.valueChanged.connect(lambda v: self._on_setting("asinh_pct", float(v)))
        self.sym_chk.toggled.connect(self._on_symmetric)
        self.restore_btn.clicked.connect(self.restore_limits)
        self.cmb_scale.currentTextChanged.connect(lambda v: self._on_setting("time_axis_scale", v))
        self.cmb_tlabel.currentTextChanged.connect(lambda v: self._on_setting("time_axis_label", v))
        self.cmb_cmap.currentTextChanged.connect(lambda v: self._on_setting("cmap", v))
        self.cmb_xunit.currentTextChanged.connect(self._on_xunit_changed)
        self.secondary_chk.toggled.connect(self._on_secondary_toggled)

    # ------------------------------------------------------------------ #
    #                                Slots                               #
    # ------------------------------------------------------------------ #
    def _on_arcsinh_toggled(self, checked: bool):
        if hasattr(self, "arcsinh_pct") and self.arcsinh_pct is not None:
            self.arcsinh_pct.setEnabled(checked)
        self.renderRequested.emit()

    def _on_z_edited(self, edited, other):
        """Mirror the partner field when the symmetric lock is active."""
        if self.sym_chk.isChecked():
            other.blockSignals(True)
            other.setValue(-edited.value())
            other.blockSignals(False)
        self.renderRequested.emit()

    def _on_slider(self, pct: int):
        self._apply_pct(float(pct))

    def _on_pct_spin(self, pct: float):
        self._apply_pct(float(pct))

    def _apply_pct(self, pct: float):
        """Apply a colour-scale percentage from the slider or the spin box.

        The spin box accepts any float percentage in ``[1e-6, 100]`` (shown as
        ``%.2g``); the integer slider is the coarse control and mirrors it at the
        nearest whole percent, clamped to its own range.
        """
        self.z_slider.blockSignals(True)
        self.z_slider.setValue(max(self.z_slider.minimum(), int(round(pct))))
        self.z_slider.blockSignals(False)
        self.z_pct.blockSignals(True)
        self.z_pct.setValue(pct)
        self.z_pct.blockSignals(False)
        zmax = self._zabs * pct / 100.0
        for w, v in ((self.z_min, -zmax), (self.z_max, zmax)):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)
        self.renderRequested.emit()

    def _on_xy_changed(self, *_):
        """X/Y limit edit: refresh the Z reference if restricted to the view."""
        if self.restrict_z_chk.isChecked():
            self._refresh_zref()
        self.limitsChanged.emit()

    def _compute_zref(self) -> tuple[float, float, float]:
        """Return ``(zabs, zlo, zhi)`` for the colour-scale reference / read-out.

        Uses the whole dataset by default; when the "Z: current range" box is
        checked, ``max|Z|`` (and the read-out span) are taken only over the data
        inside the current X (probe) and Y (delay) limits.
        """
        full = (self._zabs_full, self._zfull_lo, self._zfull_hi)
        if self._dataset is None or not self.restrict_z_chk.isChecked():
            return full
        Z = np.asarray(self._dataset._detector_slice(self.detector), dtype=float)
        probe = np.asarray(self._dataset._detector_probe(self.detector), dtype=float)
        delays = np.asarray(self._dataset.delays, dtype=float)
        native = hlp.native_x_unit(self._dataset.units)
        
        xlim_vals = list(self.xlim())
        if getattr(self, "_x_scaled", False):
            xlim_vals = [v * 1000.0 for v in xlim_vals]

        xl = hlp.convert_spectral(np.asarray(xlim_vals, dtype=float), self._x_display, native)
        xlo, xhi = sorted((float(xl[0]), float(xl[1])))
        ylo, yhi = sorted(self.ylim())
        pmask = (probe >= xlo) & (probe <= xhi)
        dmask = (delays >= ylo) & (delays <= yhi)
        if not pmask.any() or not dmask.any():
            return full
        sub = Z[dmask][:, pmask]  # mask delays (axis 0) then probe (axis 1)
        finite = sub[np.isfinite(sub)]
        if finite.size == 0:
            return full
        zabs = float(np.nanmax(np.abs(finite))) or self._zabs_full
        return (zabs, float(np.nanmin(finite)), float(np.nanmax(finite)))

    def _refresh_zref(self):
        """Recompute the Z reference + read-out and rescale Z at the current %."""
        zabs, zlo, zhi = self._compute_zref()
        self._zabs = zabs or 1.0
        self.z_range.setText(f"Max ΔAbs (mOD): {zlo:.3g}, {zhi:.3g}")
        self._apply_pct(float(self.z_pct.value()))

    def _on_symmetric(self, on: bool):
        if on:
            self.z_min.blockSignals(True)
            self.z_min.setValue(-self.z_max.value())
            self.z_min.blockSignals(False)
        self.renderRequested.emit()

    def _on_setting(self, field: str, value: float | str):
        pm.update_settings(**{field: value})
        self.renderRequested.emit()

    def _display_unit(self) -> str:
        """Effective spectral display unit (resolves "native" against the data)."""
        native = hlp.native_x_unit(self._dataset.units) if self._dataset is not None else "nm"
        return hlp.resolve_x_unit(pm.get_settings().x_axis_unit.value, native)

    def _update_x_token(self, unit: str):
        """Update the Min/Max X spin-box labels and suffixes for the current display unit."""
        self._x_lbl_min.setText("Min probe")
        self._x_lbl_max.setText("Max probe")

        if unit == "nm":
            suffix = " nm"
        elif unit == "cm-1":
            suffix = " 10³ cm⁻¹" if getattr(self, "_x_scaled", False) else " cm⁻¹"
        elif unit == "eV":
            suffix = " eV"
        elif unit == "THz":
            suffix = " THz"
        else:
            suffix = f" {unit}"

        for w in (self.x_min, self.x_max):
            w.setSuffix(suffix)

    def _on_xunit_changed(self, value: str):
        """Switch the spectral display unit, converting the X spin-box limits."""
        old = self._x_display
        pm.update_settings(x_axis_unit=value)
        new = self._display_unit()
        if self._dataset is not None and old != new:
            native = hlp.native_x_unit(self._dataset.units)
            probe_new = hlp.convert_spectral(
                np.asarray(self._dataset._detector_probe(self.detector), dtype=float), native, new
            )
            new_scaled = new == "cm-1" and float(np.nanmax(np.abs(probe_new))) > 5000.0

            converted_vals = []
            for w in (self.x_min, self.x_max):
                val_unscaled = w.value()
                if getattr(self, "_x_scaled", False):
                    val_unscaled *= 1000.0
                new_val_unscaled = float(hlp.convert_spectral(val_unscaled, old, new))
                new_val = new_val_unscaled / 1000.0 if new_scaled else new_val_unscaled
                converted_vals.append(new_val)

            v_min, v_max = min(converted_vals), max(converted_vals)
            for w, v in ((self.x_min, v_min), (self.x_max, v_max)):
                w.blockSignals(True)
                w.setValue(v)
                w.blockSignals(False)

            self._x_scaled = new_scaled
            self._x_display = new
            self._update_x_token(new)
            if self.restrict_z_chk.isChecked():
                self._refresh_zref()
        else:
            self._x_display = new
            self._update_x_token(new)
        self.renderRequested.emit()

    def _on_secondary_toggled(self, on: bool):
        pm.update_settings(secondary_axis=bool(on))
        self.renderRequested.emit()

    # ------------------------------------------------------------------ #
    #                              Public API                            #
    # ------------------------------------------------------------------ #
    def sync_from_settings(self):
        """Reflect the active settings in the dropdowns (no signals emitted)."""
        s = pm.get_settings()
        for cb, val in (
            (self.cmb_scale, s.time_axis_scale.value),
            (self.cmb_tlabel, s.time_axis_label.value),
            (self.cmb_cmap, s.cmap),
            (self.cmb_xunit, s.x_axis_unit.value),
        ):
            cb.blockSignals(True)
            cb.setCurrentText(str(val))
            cb.blockSignals(False)
        self.secondary_chk.blockSignals(True)
        self.secondary_chk.setChecked(bool(s.secondary_axis))
        self.secondary_chk.blockSignals(False)
        self.white_spin.blockSignals(True)
        self.white_spin.setValue(int(s.white_levels))
        self.white_spin.blockSignals(False)
        if hasattr(self, "arcsinh_pct") and self.arcsinh_pct is not None:
            self.arcsinh_pct.blockSignals(True)
            self.arcsinh_pct.setValue(float(getattr(s, "asinh_pct", 5.0)))
            self.arcsinh_pct.blockSignals(False)
        if hasattr(self, "n_contours") and self.n_contours is not None:
            self.n_contours.blockSignals(True)
            self.n_contours.setValue(int(getattr(s, "n_contours", 40)))
            self.n_contours.blockSignals(False)
        mw = getattr(self, "mw", None)
        chk_quick = getattr(mw, "PP_QuickPlots_chk", None) if mw is not None else None
        if chk_quick is not None:
            chk_quick.blockSignals(True)
            chk_quick.setChecked(bool(s.quick_plots))
            chk_quick.blockSignals(False)

        if hasattr(self, "_x_display") and self._x_display != self._display_unit():
            self._on_xunit_changed(s.x_axis_unit.value)


    @property
    def detector(self) -> int:
        """Currently selected detector index (0-based for array indexing)."""
        if getattr(self, "detector_spin", None) is not None:
            if self.detector_spin.isVisible():
                return int(self.detector_spin.value()) - 1
        return 0

    def _on_detector_changed(self, val: int):
        if self._dataset is not None:
            det = self.detector
            det_probe = self._dataset._detector_probe(det)
            native = hlp.native_x_unit(self._dataset.units)
            probe_disp = hlp.convert_spectral(
                np.asarray(det_probe, dtype=float), native, self._x_display
            )
            xmin_val = float(np.nanmin(probe_disp))
            xmax_val = float(np.nanmax(probe_disp))
            if getattr(self, "_x_scaled", False):
                xmin_val /= 1000.0
                xmax_val /= 1000.0

            for w, v in ((self.x_min, xmin_val), (self.x_max, xmax_val)):
                w.blockSignals(True)
                w.setValue(v)
                w.blockSignals(False)

            Z_slice = self._dataset._detector_slice(det)
            finite = Z_slice[np.isfinite(Z_slice)]
            self._zabs_full = (float(np.nanmax(np.abs(finite))) if finite.size else 1.0) or 1.0
            self._zfull_lo = float(np.nanmin(finite)) if finite.size else -1.0
            self._zfull_hi = float(np.nanmax(finite)) if finite.size else 1.0

            zabs, zlo, zhi = self._compute_zref()
            self._zabs = zabs or 1.0
            for w, v in ((self.z_min, -self._zabs), (self.z_max, self._zabs)):
                w.blockSignals(True)
                w.setValue(v)
                w.blockSignals(False)
            self.z_range.setText(f"Max ΔAbs (mOD): {zlo:.3g}, {zhi:.3g}")
            if self.restrict_z_chk.isChecked():
                self._refresh_zref()

        self.renderRequested.emit()

    def set_dataset(self, dataset):
        """Bind layout controls to a new :class:`pymorgan.oneD.Dataset1D`."""
        self._dataset = dataset
        if dataset is None:
            return

        self._x_display = self._display_unit()

        n_det = getattr(dataset, "n_detectors", 1)
        if hasattr(self, "detector_spin") and self.detector_spin is not None:
            if n_det > 1:
                self.detector_lbl.setVisible(True)
                self.detector_spin.setVisible(True)
                self.detector_spin.blockSignals(True)
                self.detector_spin.setRange(1, n_det)
                if self.detector_spin.value() > n_det or self.detector_spin.value() < 1:
                    self.detector_spin.setValue(1)
                self.detector_spin.blockSignals(False)
            else:
                self.detector_lbl.setVisible(False)
                self.detector_spin.setVisible(False)
                self.detector_spin.blockSignals(True)
                self.detector_spin.setValue(1)
                self.detector_spin.blockSignals(False)

        Z = np.asarray(dataset._detector_slice(self.detector), dtype=float)
        finite = Z[np.isfinite(Z)]
        self._zabs_full = (float(np.nanmax(np.abs(finite))) if finite.size else 1.0) or 1.0
        self._zfull_lo = float(np.nanmin(finite)) if finite.size else -1.0
        self._zfull_hi = float(np.nanmax(finite)) if finite.size else 1.0
        native = hlp.native_x_unit(dataset.units)
        probe = hlp.convert_spectral(
            np.asarray(dataset._detector_probe(self.detector), dtype=float), native, self._x_display
        )
        delays = np.asarray(dataset.delays, dtype=float)

        self._x_scaled = self._x_display == "cm-1" and float(np.nanmax(np.abs(probe))) > 5000.0
        self._update_x_token(self._x_display)

        t_unit = dataset.units.get("unitsT_ltx", "ps") if hasattr(dataset, "units") and isinstance(dataset.units, dict) else "ps"
        self.y_min.setSuffix(f" {t_unit}")
        self.y_max.setSuffix(f" {t_unit}")

        # X/Y limits first so a windowed Z reference (restrict-Z) can read them.
        xmin_val = float(np.nanmin(probe))
        xmax_val = float(np.nanmax(probe))
        if self._x_scaled:
            xmin_val /= 1000.0
            xmax_val /= 1000.0

        for w, v in (
            (self.x_min, xmin_val),
            (self.x_max, xmax_val),
            (self.y_min, max(-0.5, float(np.nanmin(delays)))),
            (self.y_max, float(np.nanmax(delays))),
        ):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)

        zabs, zlo, zhi = self._compute_zref()
        self._zabs = zabs or 1.0
        for w, v in ((self.z_min, -self._zabs), (self.z_max, self._zabs)):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)
        self.z_slider.blockSignals(True)
        self.z_slider.setValue(100)
        self.z_slider.blockSignals(False)
        self.z_pct.blockSignals(True)
        self.z_pct.setValue(100)
        self.z_pct.blockSignals(False)
        self.z_range.setText(f"Max ΔAbs (mOD): {zlo:.3g}, {zhi:.3g}")

    def restore_limits(self):
        """Reset every limit/scale to the current dataset's defaults."""
        if self._dataset is not None:
            self.set_dataset(self._dataset)
            self.renderRequested.emit()

    def xlim(self) -> tuple[float, float]:
        val_min = self.x_min.value()
        val_max = self.x_max.value()
        if getattr(self, "_x_scaled", False):
            val_min *= 1000.0
            val_max *= 1000.0
        return (val_min, val_max)

    def ylim(self) -> tuple[float, float]:
        return (self.y_min.value(), self.y_max.value())

    def zlimits(self) -> tuple[float, float]:
        return (self.z_min.value(), self.z_max.value())

    def time_axis_kwargs(self) -> dict:
        """Delay-axis configuration of the panel, for plots drawn in new figures.

        The scale and label combos write straight to the active settings, so the
        values are read back from there; passing them explicitly keeps a plot
        consistent with the panel even if a call overrides the settings.
        """
        s = pm.get_settings()
        scale = s.time_axis_scale.value if hasattr(s.time_axis_scale, "value") else s.time_axis_scale
        label = s.time_axis_label.value if hasattr(s.time_axis_label, "value") else s.time_axis_label
        return {"Yscale": scale, "time_axis_label": label}

    def smooth(self) -> int:
        """Smoothing for transient-spectra / kinetic cuts (``doSmooth``).

        0 = none, >0 = smoothed only, <0 = smoothed plus the original
        semitransparent; the magnitude sets the smoothing-window width.
        """
        return int(self.smooth_spin.value())

    def probe_display_unit(self) -> str:
        """Spectral unit currently shown on the X axis (resolved display unit)."""
        return self._display_unit()

    def probe_to_native(self, values):
        """Convert probe positions from the X display unit to the dataset's native unit."""
        if self._dataset is None:
            return values
        native = hlp.native_x_unit(self._dataset.units)
        values_list = [float(v) for v in np.atleast_1d(values)]
        if getattr(self, "_x_scaled", False):
            values_list = [v * 1000.0 for v in values_list]
        if native == self._x_display:
            return values_list
        arr = hlp.convert_spectral(np.asarray(values_list, dtype=float), self._x_display, native)
        return [float(v) for v in np.atleast_1d(arr)]

    def probe_from_native(self, values):
        """Convert probe positions from the dataset's native unit to the X display unit."""
        if self._dataset is None:
            return values
        native = hlp.native_x_unit(self._dataset.units)
        if native == self._x_display:
            arr = [float(v) for v in np.atleast_1d(values)]
        else:
            arr = hlp.convert_spectral(np.asarray(values, dtype=float), native, self._x_display)
            arr = [float(v) for v in np.atleast_1d(arr)]
        if getattr(self, "_x_scaled", False):
            arr = [v / 1000.0 for v in arr]
        return arr

    def autoscale_cut_y(self) -> bool:
        """Whether kinetic/spectral cuts should autoscale their signal (Y) axis.

        When True, the inherited-limits feature copies only the shared (X) axis
        from the contour and leaves the cut's Y axis to autoscale to its data.
        """
        return bool(self.autoy_chk.isChecked())

    def contour_kwargs(self) -> dict:
        """Per-view keyword arguments for ``plot_contour``."""
        zmin, zmax = self.zlimits()
        asinh = self.arcsinh_chk.isChecked()
        asinh_pct = (
            float(self.arcsinh_pct.value())
            if hasattr(self, "arcsinh_pct") and self.arcsinh_pct is not None
            else 5.0
        )
        return {
            "detector": self.detector,
            "Zmin": zmin,
            "Zmax": zmax,
            "Asinh": asinh,
            "asinh_pct": asinh_pct,
            "filled": self.filled_chk.isChecked(),
            "ShowLines": self.showlines_chk.isChecked(),
            "smooth": self.smooth(),
            "Nlevels": int(self.n_contours.value()),
            "Nskip": int(self.n_skip.value()),
            "white_levels": int(self.white_spin.value()),
            "cbarLbl": "top",
            "aspect": "auto",
        }


_WIDGETS_2D = {
    "pump_min": "twoD_PC_pumpMin",
    "pump_max": "twoD_PC_pumpMax",
    "probe_min": "twoD_PC_probeMin",
    "probe_max": "twoD_PC_probeMax",
    "cmb_cmap": "twoD_PC_cmap",
    "n_contours": "twoD_PC_nContours",
    "sym_chk": "twoD_PC_symmetric",
    "filled_chk": "twoD_PC_filled",
    "showlines_chk": "twoD_PC_showLines",
    "white_spin": "twoD_PC_whiteLevels",
    "z_slider": "twoD_PC_zSlider",
    "z_pct": "twoD_PC_zPct",
    "da_range": "twoD_PC_daRange",
    "max_da": "twoD_PC_maxDa",
    "restore_btn": "twoD_PC_restore",
    "lines_every": "twoD_PC_contourLinesEvery",
    "lines_fmt": "twoD_PC_contourLinesFormat",
    "square_chk": "twoD_PC_square",
    "text_white_bg_chk": "twoD_PC_textWhiteBg",
    "cut_plot_chk": "twoD_PC_cutPlot",
}



class PlotControlsPanel2D(QObject):
    """Binds and drives the ``twoD_PC_box`` plot-controls widgets from the .ui."""

    renderRequested = pyqtSignal()  # needs a full re-plot
    limitsChanged = pyqtSignal()  # only re-apply limits

    def __init__(self, host, parent=None):
        super().__init__(parent)
        self._dataset = None
        self._active_delay_index = 0
        self._zabs = 1.0  # max |Z| at current delay

        for attr, name in _WIDGETS_2D.items():
            setattr(self, attr, getattr(host, name, None))

        # Populate combos
        if self.cmb_cmap is not None:
            PlotControlsPanel._set_combo_items(self.cmb_cmap, _CMAPS)
            s = pm.get_settings()
            self.cmb_cmap.setCurrentText(str(s.cmap))
            self.cmb_cmap.currentTextChanged.connect(self._on_cmap_changed)

        if self.white_spin is not None:
            s = pm.get_settings()
            self.white_spin.setValue(int(s.white_levels))
            self.white_spin.valueChanged.connect(self._on_white_changed)

        if self.lines_every is not None:
            self.lines_every.setValue(2)

        if self.showlines_chk is not None:
            self.showlines_chk.setChecked(True)

        if self.text_white_bg_chk is not None:
            self.text_white_bg_chk.setChecked(True)

        # Configure n_contours default (from settings, default 40), step (2), min (2), and coerce odd numbers
        if self.n_contours is not None:
            s = pm.get_settings()
            default_n = int(getattr(s, "n_contours", 40))
            self.n_contours.blockSignals(True)
            self.n_contours.setRange(2, 400)
            self.n_contours.setSingleStep(2)
            self.n_contours.setValue(default_n)
            self.n_contours.blockSignals(False)
            self._prev_n_contours = default_n



            def on_twoD_n_contours_changed(val):
                if val % 2 != 0:
                    if val < self._prev_n_contours:
                        even_val = val - 1
                    else:
                        even_val = val + 1
                    if even_val < self.n_contours.minimum():
                        even_val = self.n_contours.minimum()
                    self.n_contours.blockSignals(True)
                    self.n_contours.setValue(even_val)
                    self.n_contours.blockSignals(False)
                    self._prev_n_contours = even_val
                else:
                    self._prev_n_contours = val

            self.n_contours.valueChanged.connect(on_twoD_n_contours_changed)

        for w in (self.pump_min, self.pump_max, self.probe_min, self.probe_max):
            if w is not None:
                w._FMT = "%.1f"

        _configure_pct_spin(self.z_pct)

        # Spectral limits: plain decimal display (e.g. "2105.2 cm-1").
        for w in (self.pump_min, self.pump_max, self.probe_min, self.probe_max):
            if w is not None and hasattr(w, "setDisplayFormat"):
                w.setDisplayFormat(_fmt_freq)

        # Detector selection spinner (only visible for datasets with >1 detector)
        self.detector_lbl = getattr(host, "twoD_PC_detectorLabel", None)
        self.detector_spin = getattr(host, "twoD_PC_detector", None)
        if self.detector_spin is None:
            box = getattr(host, "twoD_PC_box", None)
            if box is not None:
                layout = box.layout()
                self.detector_lbl = QLabel("Detector:", box)
                self.detector_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.detector_spin = QSpinBox(box)
                self.detector_spin.setRange(1, 1)
                self.detector_spin.setFixedWidth(56)
                self.detector_spin.setToolTip("Select active detector (1-based)")
                if layout is not None:
                    layout.addWidget(self.detector_lbl, 3, 4)
                    layout.addWidget(self.detector_spin, 3, 5)

        if self.detector_lbl is not None and self.detector_spin is not None:
            self.detector_lbl.setVisible(False)
            self.detector_spin.setVisible(False)
            self.detector_spin.valueChanged.connect(self._on_detector_changed)

        self._connect_signals()

    def _connect_signals(self):
        for w in (self.pump_min, self.pump_max, self.probe_min, self.probe_max):
            if w is not None:
                w.valueChanged.connect(lambda *_: self.limitsChanged.emit())

        if self.z_slider is not None:
            self.z_slider.valueChanged.connect(self._on_slider)
        if self.z_pct is not None:
            self.z_pct.valueChanged.connect(self._on_pct_spin)

        for w in (self.n_contours, self.lines_every):
            if w is not None:
                w.valueChanged.connect(lambda *_: self.renderRequested.emit())

        for chk in (self.filled_chk, self.showlines_chk, self.sym_chk, self.square_chk, self.text_white_bg_chk, self.cut_plot_chk):
            if chk is not None:
                chk.toggled.connect(lambda *_: self.renderRequested.emit())


        if self.lines_fmt is not None:
            self.lines_fmt.textChanged.connect(lambda *_: self.renderRequested.emit())

        if self.restore_btn is not None:
            self.restore_btn.clicked.connect(self.restore_limits)

    def _on_cmap_changed(self, val):
        pm.update_settings(cmap=val)
        self.renderRequested.emit()

    def _on_white_changed(self, val):
        pm.update_settings(white_levels=float(val))
        self.renderRequested.emit()

    def _on_slider(self, pct: int):
        self._apply_pct(float(pct))

    def _on_pct_spin(self, pct: float):
        self._apply_pct(float(pct))

    def _apply_pct(self, pct: float):
        if self.z_slider is not None:
            self.z_slider.blockSignals(True)
            self.z_slider.setValue(max(self.z_slider.minimum(), int(round(pct))))
            self.z_slider.blockSignals(False)
        if self.z_pct is not None:
            self.z_pct.blockSignals(True)
            self.z_pct.setValue(float(pct))
            self.z_pct.blockSignals(False)
        self._update_z_displays()
        self.renderRequested.emit()


    def _update_z_displays(self):
        """Update read-only displays (da_range, max_da) based on Zabs and current pct."""
        pct = self.z_pct.value() if self.z_pct is not None else 100.0
        scaled_max = self._zabs * pct / 100.0

        if self.max_da is not None:
            self.max_da.setText(f"{scaled_max:.3f}")

        if self.da_range is not None:
            if self.sym_chk is not None and self.sym_chk.isChecked():
                self.da_range.setText(f"[-{scaled_max:.3f}, {scaled_max:.3f}]")
            else:
                if self._dataset is not None:
                    Zmap = self._dataset.Z[:, :, self._active_delay_index]
                    zmin = float(np.nanmin(Zmap)) * pct / 100.0
                    zmax = float(np.nanmax(Zmap)) * pct / 100.0
                    self.da_range.setText(f"[{zmin:.3f}, {zmax:.3f}]")
                else:
                    self.da_range.setText(f"[-{scaled_max:.3f}, {scaled_max:.3f}]")

    def set_dataset(self, dataset, active_delay_index=0):
        self._dataset = dataset
        self._active_delay_index = active_delay_index

        n_det = int(getattr(dataset, "n_detectors", 1))
        if getattr(self, "detector_spin", None) is not None and getattr(self, "detector_lbl", None) is not None:
            if n_det > 1:
                self.detector_lbl.setVisible(True)
                self.detector_spin.setVisible(True)
                self.detector_spin.blockSignals(True)
                self.detector_spin.setRange(1, n_det)
                if self.detector_spin.value() > n_det or self.detector_spin.value() < 1:
                    self.detector_spin.setValue(1)
                self.detector_spin.blockSignals(False)
            else:
                self.detector_lbl.setVisible(False)
                self.detector_spin.setVisible(False)
                self.detector_spin.blockSignals(True)
                self.detector_spin.setValue(1)
                self.detector_spin.blockSignals(False)

        pump = np.asarray(dataset.pump, dtype=float)
        probe = np.asarray(dataset.probe, dtype=float)

        # Show the spectral unit on the limit spin boxes (defaults to cm-1 in the
        # .ui; follow the dataset if it reports something else).
        unit = str(getattr(dataset, "freq_units", "") or "").strip()
        if unit:
            for w in (self.pump_min, self.pump_max, self.probe_min, self.probe_max):
                if w is not None and w.suffix().strip() != unit:
                    w.setSuffix(f" {unit}")

        # Set default pump limits dynamically
        if dataset.datatype == "shaper":
            pump_min_val = float(probe.min())
            pump_max_val = float(probe.max())
        elif dataset.datatype == "interferometer" and hasattr(dataset, "proc_absFFT_ZPint") and dataset.proc_absFFT_ZPint is not None:
            spec = np.abs(dataset.proc_absFFT_ZPint[:len(dataset.pump), active_delay_index])
            threshold = 0.20 * np.max(spec)
            above_indices = np.nonzero(spec >= threshold)[0]
            if len(above_indices) > 0:
                pump_min_val = float(dataset.pump[above_indices[0]])
                pump_max_val = float(dataset.pump[above_indices[-1]])
            else:
                pump_min_val = float(pump.min())
                pump_max_val = float(pump.max())
        else:
            pump_min_val = float(pump.min())
            pump_max_val = float(pump.max())

        # Clip defaults to actual pump bounds to be safe
        pump_min_val = max(float(pump.min()), min(float(pump.max()), pump_min_val))
        pump_max_val = max(float(pump.min()), min(float(pump.max()), pump_max_val))

        # Set limits
        for w, v in (
            (self.pump_min, pump_min_val),
            (self.pump_max, pump_max_val),
            (self.probe_min, float(probe.min())),
            (self.probe_max, float(probe.max())),
        ):
            if w is not None:
                w.blockSignals(True)
                w.setValue(v)
                w.blockSignals(False)

        self.update_delay_index(active_delay_index, force_reset_pct=True)

    def update_delay_index(self, active_delay_index, force_reset_pct=False):
        self._active_delay_index = active_delay_index
        if self._dataset is not None:
            Zmap = self._dataset.Z[:, :, active_delay_index]

            # If cut plot is checked, slice Zmap
            if self.cut_plot_chk is not None and self.cut_plot_chk.isChecked():
                pump = np.asarray(self._dataset.pump, dtype=float)
                probe = np.asarray(self._dataset.probe, dtype=float)

                pmin = self.pump_min.value() if self.pump_min is not None else pump.min()
                pmax = self.pump_max.value() if self.pump_max is not None else pump.max()
                p_min_v, p_max_v = min(pmin, pmax), max(pmin, pmax)

                prmin = self.probe_min.value() if self.probe_min is not None else probe.min()
                prmax = self.probe_max.value() if self.probe_max is not None else probe.max()
                pr_min_v, pr_max_v = min(prmin, prmax), max(prmin, prmax)

                pump_indices = np.nonzero((pump >= p_min_v) & (pump <= p_max_v))[0]
                probe_indices = np.nonzero((probe >= pr_min_v) & (probe <= pr_max_v))[0]

                if len(pump_indices) > 0 and len(probe_indices) > 0:
                    Zmap = Zmap[np.ix_(pump_indices, probe_indices)]

            finite = Zmap[np.isfinite(Zmap)]
            self._zabs = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
            if self._zabs == 0:
                self._zabs = 1.0
        else:
            self._zabs = 1.0

        if force_reset_pct:
            if self.z_slider is not None:
                self.z_slider.blockSignals(True)
                self.z_slider.setValue(100)
                self.z_slider.blockSignals(False)
            if self.z_pct is not None:
                self.z_pct.blockSignals(True)
                self.z_pct.setValue(100)
                self.z_pct.blockSignals(False)

        self._update_z_displays()

    @property
    def detector(self) -> int:
        """Currently selected detector index (0-based for array indexing)."""
        if getattr(self, "detector_spin", None) is not None:
            if self.detector_spin.isVisible():
                return int(self.detector_spin.value()) - 1
        return 0

    def _on_detector_changed(self, val: int):
        if self._dataset is not None:
            det = self.detector
            probe = np.asarray(self._dataset.probe, dtype=float)
            if probe.ndim > 1:
                det_probe = probe[:, min(det, probe.shape[1] - 1)]
                for w, v in (
                    (self.probe_min, float(det_probe.min())),
                    (self.probe_max, float(det_probe.max())),
                ):
                    if w is not None:
                        w.blockSignals(True)
                        w.setValue(v)
                        w.blockSignals(False)
            self.update_delay_index(self._active_delay_index, force_reset_pct=True)
        self.renderRequested.emit()

    def restore_limits(self):
        """Reset every limit/scale to the current dataset's defaults."""
        if self._dataset is not None:
            self.set_dataset(self._dataset, self._active_delay_index)
            self.renderRequested.emit()

    def pump_lim(self) -> tuple[float, float]:
        pmin = self.pump_min.value() if self.pump_min is not None else 0.0
        pmax = self.pump_max.value() if self.pump_max is not None else 1.0
        return (pmin, pmax)

    def probe_lim(self) -> tuple[float, float]:
        pmin = self.probe_min.value() if self.probe_min is not None else 0.0
        pmax = self.probe_max.value() if self.probe_max is not None else 1.0
        return (pmin, pmax)

    def zlimits(self) -> tuple[float, float]:
        pct = self.z_pct.value() if self.z_pct is not None else 100.0
        scaled_max = self._zabs * pct / 100.0
        if self.sym_chk is not None and self.sym_chk.isChecked():
            return (-scaled_max, scaled_max)
        else:
            if self._dataset is not None:
                Zmap = self._dataset.Z[:, :, self._active_delay_index]
                zmin = float(np.nanmin(Zmap)) * pct / 100.0
                zmax = float(np.nanmax(Zmap)) * pct / 100.0
                return (zmin, zmax)
            else:
                return (-scaled_max, scaled_max)

    def contour_kwargs(self) -> dict:
        zmin, zmax = self.zlimits()
        cmap_ID = self.cmb_cmap.currentText() if self.cmb_cmap is not None else "DkRd/Wh/DkBu"
        return {
            "detector": self.detector,
            "cmap_ID": cmap_ID,
            "filled": self.filled_chk.isChecked() if self.filled_chk is not None else True,
            "ShowLines": self.showlines_chk.isChecked() if self.showlines_chk is not None else True,
            "Nskip": int(self.lines_every.value()) if self.lines_every is not None else 2,
            "white_levels": int(self.white_spin.value()) if self.white_spin is not None else 0,
            "text_white_bg": self.text_white_bg_chk.isChecked() if self.text_white_bg_chk is not None else False,
            "cut_plot": self.cut_plot_chk.isChecked() if self.cut_plot_chk is not None else False,
            "pump_lim": self.pump_lim(),
            "probe_lim": self.probe_lim(),
            "Nlevels": int(self.n_contours.value()) if self.n_contours is not None else 40,
            "symmetric": self.sym_chk.isChecked() if self.sym_chk is not None else False,
        }

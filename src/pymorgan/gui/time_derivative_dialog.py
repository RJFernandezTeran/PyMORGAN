"""Dialog for configuring and previewing time derivative d(Delta A)/dt calculations."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from pymorgan.gui.canvas import MplCanvas
from pymorgan.oneD.process import time_derivative


class TimeDerivativeDialog(QDialog):
    """Dialog for time derivative of spectra with live kinetic preview."""

    def __init__(self, parent, dataset):
        super().__init__(parent)
        self.setWindowTitle("Time Derivative of Spectra")
        self.resize(720, 640)

        self.dataset = dataset
        self.delays_out: np.ndarray | None = None
        self.Zavg_dt: np.ndarray | None = None
        self.Zss_dt: np.ndarray | None = None

        layout = QVBoxLayout(self)

        # Header description
        hdr = QLabel(
            "Calculate the time derivative d(\u0394A)/dt along the delay axis with optional\n"
            "early-delay cutoff, time-domain interpolation, and smoothing."
        )
        hdr.setStyleSheet("font-weight: bold; margin-bottom: 4px;")
        layout.addWidget(hdr)

        # Settings Container Layout
        settings_layout = QHBoxLayout()

        # Group 1: Delay Cutoff & Interpolation
        grp_time = QGroupBox("Delay Window & Interpolation")
        vbox_time = QVBoxLayout(grp_time)

        # Cutoff
        h_cut = QHBoxLayout()
        self.cut_early_chk = QCheckBox("Cut early delays (t < t_min):")
        self.t_min_spin = QDoubleSpinBox()
        self.t_min_spin.setRange(-1000.0, 1e7)
        self.t_min_spin.setDecimals(3)
        self.t_min_spin.setSingleStep(0.1)
        u_t = dataset.units.get("unitsT_lbl", "ps") if dataset.units else "ps"
        self.t_min_spin.setSuffix(f" {u_t}")
        # Default t_min to 0.2 if data spans > 1.0, otherwise 0.0
        default_tmin = 0.2 if float(np.max(dataset.delays)) > 1.0 and float(np.min(dataset.delays)) < 0.2 else 0.0
        self.t_min_spin.setValue(default_tmin)
        self.cut_early_chk.setChecked(default_tmin > 0.0)
        self.t_min_spin.setEnabled(self.cut_early_chk.isChecked())
        h_cut.addWidget(self.cut_early_chk)
        h_cut.addWidget(self.t_min_spin)
        vbox_time.addLayout(h_cut)

        # Interpolation
        self.interp_chk = QCheckBox("Interpolate in time domain")
        self.interp_chk.setChecked(False)
        vbox_time.addWidget(self.interp_chk)

        h_interp = QHBoxLayout()
        h_interp.addWidget(QLabel("Points:"))
        self.interp_pts_spin = QSpinBox()
        self.interp_pts_spin.setRange(10, 100000)
        self.interp_pts_spin.setValue(max(100, int(2 * len(dataset.delays))))
        self.interp_pts_spin.setEnabled(False)
        h_interp.addWidget(self.interp_pts_spin)

        h_interp.addWidget(QLabel("Method:"))
        self.interp_kind_combo = QComboBox()
        self.interp_kind_combo.addItems(["Cubic Spline", "PCHIP (Monotonic)", "Linear"])
        self.interp_kind_combo.setEnabled(False)
        h_interp.addWidget(self.interp_kind_combo)
        vbox_time.addLayout(h_interp)

        settings_layout.addWidget(grp_time)

        # Group 2: Smoothing
        grp_smooth = QGroupBox("Time-Domain Smoothing")
        vbox_smooth = QVBoxLayout(grp_smooth)

        self.smooth_chk = QCheckBox("Smooth in time domain")
        self.smooth_chk.setChecked(False)
        vbox_smooth.addWidget(self.smooth_chk)

        h_sm_method = QHBoxLayout()
        h_sm_method.addWidget(QLabel("Filter:"))
        self.smooth_method_combo = QComboBox()
        self.smooth_method_combo.addItems(["Savitzky-Golay", "Gaussian", "Moving Average"])
        self.smooth_method_combo.setEnabled(False)
        h_sm_method.addWidget(self.smooth_method_combo)
        vbox_smooth.addLayout(h_sm_method)

        h_sm_params = QHBoxLayout()
        h_sm_params.addWidget(QLabel("Window:"))
        self.smooth_win_spin = QSpinBox()
        self.smooth_win_spin.setRange(5, 99)
        self.smooth_win_spin.setSingleStep(2)
        self.smooth_win_spin.setValue(7)
        self.smooth_win_spin.setEnabled(False)
        h_sm_params.addWidget(self.smooth_win_spin)

        h_sm_params.addWidget(QLabel("Poly order:"))
        self.smooth_poly_spin = QSpinBox()
        self.smooth_poly_spin.setRange(1, 5)
        self.smooth_poly_spin.setValue(2)
        self.smooth_poly_spin.setEnabled(False)
        h_sm_params.addWidget(self.smooth_poly_spin)
        vbox_smooth.addLayout(h_sm_params)

        settings_layout.addWidget(grp_smooth)
        layout.addLayout(settings_layout)

        # Probe Selector for Preview
        h_probe = QHBoxLayout()
        h_probe.addWidget(QLabel("Preview probe wavelength / pixel:"))
        self.probe_spin = QDoubleSpinBox()
        probe = dataset.probe
        p_min, p_max = float(np.min(probe)), float(np.max(probe))
        u_l = dataset.units.get("unitsL_lbl", "nm") if dataset.units else "nm"
        self.probe_spin.setRange(min(p_min, p_max), max(p_min, p_max))
        self.probe_spin.setDecimals(2)
        self.probe_spin.setSingleStep(1.0)
        self.probe_spin.setSuffix(f" {u_l}")
        # Default to middle wavelength
        self.probe_spin.setValue(float(probe[len(probe) // 2]))
        h_probe.addWidget(self.probe_spin)
        h_probe.addStretch()
        layout.addLayout(h_probe)

        # Live Preview MplCanvas
        self.canvas = MplCanvas(self)
        layout.addWidget(self.canvas)

        # Standard Dialog Buttons
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Apply Time Derivative")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        # Connect Events
        self.cut_early_chk.toggled.connect(self.t_min_spin.setEnabled)
        self.cut_early_chk.toggled.connect(self._update_preview)
        self.t_min_spin.valueChanged.connect(self._update_preview)

        self.interp_chk.toggled.connect(self.interp_pts_spin.setEnabled)
        self.interp_chk.toggled.connect(self.interp_kind_combo.setEnabled)
        self.interp_chk.toggled.connect(self._update_preview)
        self.interp_pts_spin.valueChanged.connect(self._update_preview)
        self.interp_kind_combo.currentIndexChanged.connect(self._update_preview)

        self.smooth_chk.toggled.connect(self.smooth_method_combo.setEnabled)
        self.smooth_chk.toggled.connect(self.smooth_win_spin.setEnabled)
        self.smooth_chk.toggled.connect(self.smooth_poly_spin.setEnabled)
        self.smooth_chk.toggled.connect(self._update_preview)
        self.smooth_method_combo.currentIndexChanged.connect(self._on_smooth_method_changed)
        self.smooth_win_spin.valueChanged.connect(self._update_preview)
        self.smooth_poly_spin.valueChanged.connect(self._update_preview)

        self.probe_spin.valueChanged.connect(self._update_preview)

        # Initial Preview
        self._update_preview()

    def _on_smooth_method_changed(self):
        method = self.smooth_method_combo.currentText()
        is_savgol = "Savitzky" in method
        self.smooth_poly_spin.setEnabled(is_savgol and self.smooth_chk.isChecked())
        self._update_preview()

    def get_options(self) -> dict:
        """Return the dictionary of configured parameters."""
        t_min = float(self.t_min_spin.value()) if self.cut_early_chk.isChecked() else None
        interp = self.interp_chk.isChecked()
        n_interp = int(self.interp_pts_spin.value()) if interp else None

        interp_text = self.interp_kind_combo.currentText().lower()
        if "pchip" in interp_text:
            interp_kind = "pchip"
        elif "linear" in interp_text:
            interp_kind = "linear"
        else:
            interp_kind = "cubic"

        smooth = self.smooth_chk.isChecked()
        sm_text = self.smooth_method_combo.currentText().lower()
        if "gaussian" in sm_text:
            smooth_method = "gaussian"
        elif "moving" in sm_text:
            smooth_method = "moving_average"
        else:
            smooth_method = "savgol"

        return {
            "t_min": t_min,
            "interpolate": interp,
            "n_interp": n_interp,
            "interp_kind": interp_kind,
            "smooth": smooth,
            "smooth_method": smooth_method,
            "smooth_window": int(self.smooth_win_spin.value()),
            "smooth_poly": int(self.smooth_poly_spin.value()),
        }

    def _update_preview(self):
        """Recompute derivative on the fly and update the 2-panel preview plot."""
        if self.dataset is None:
            return

        opts = self.get_options()
        try:
            delays_out, Zavg_dt, Zss_dt = time_derivative(
                self.dataset.delays,
                self.dataset.Z,
                self.dataset.Zss_R,
                **opts,
            )
            self.delays_out = delays_out
            self.Zavg_dt = Zavg_dt
            self.Zss_dt = Zss_dt
        except Exception as exc:
            self.canvas.figure.clear()
            ax = self.canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, f"Error: {exc}", ha="center", va="center", color="red", transform=ax.transAxes)
            self.canvas.draw()
            return

        # Find closest probe index
        target_probe = float(self.probe_spin.value())
        pix_idx = int(np.argmin(np.abs(self.dataset.probe - target_probe)))
        actual_probe = float(self.dataset.probe[pix_idx])

        # Plot raw vs derivative
        for ax in self.canvas.figure.axes:
            try:
                ax.set_xscale("linear")
                ax.set_yscale("linear")
            except Exception:
                pass
        self.canvas.figure.clear()
        ax1 = self.canvas.figure.add_subplot(211)
        ax2 = self.canvas.figure.add_subplot(212, sharex=ax1)

        raw_trace = self.dataset.Z[:, pix_idx, 0] if self.dataset.Z.ndim == 3 else self.dataset.Z[:, pix_idx]
        dt_trace = Zavg_dt[:, pix_idx, 0] if Zavg_dt.ndim == 3 else Zavg_dt[:, pix_idx]

        u_t = self.dataset.units.get("unitsT_lbl", "ps") if self.dataset.units else "ps"
        u_z = self.dataset.units.get("unitsZ_lbl", "\u0394A") if self.dataset.units else "\u0394A"

        # Raw Trace
        ax1.plot(self.dataset.delays, raw_trace, "o-", color="#3b82f6", markersize=3, label="Raw \u0394A(t)")
        if opts["t_min"] is not None:
            ax1.axvline(opts["t_min"], color="#ef4444", linestyle="--", alpha=0.7, label=f"t_min = {opts['t_min']:.2f}")
        ax1.set_ylabel(f"Signal ({u_z})", fontsize=8.5, fontweight="bold")
        ax1.set_title(f"Kinetic Trace at \u03bb = {actual_probe:.2f} {self.dataset.units.get('unitsL_lbl', 'nm')}", fontsize=9, fontweight="bold")
        ax1.legend(loc="best", fontsize=7.5)
        ax1.grid(True, alpha=0.3)

        # Time Derivative Trace
        ax2.plot(delays_out, dt_trace, ".-", color="#e11d48", markersize=3, label="d(\u0394A)/dt")
        ax2.axhline(0, color="gray", linestyle=":", alpha=0.6)
        ax2.set_xlabel(f"Delay ({u_t})", fontsize=8.5, fontweight="bold")
        ax2.set_ylabel(f"d(\u0394A)/dt ({u_z}/{u_t})", fontsize=8.5, fontweight="bold")
        ax2.legend(loc="best", fontsize=7.5)
        ax2.grid(True, alpha=0.3)

        # Log delay scaling if all delays > 0
        if np.all(delays_out > 0) and (delays_out[-1] / delays_out[0]) > 20.0:
            ax1.set_xscale("log")
            ax2.set_xscale("log")

        self.canvas.draw()

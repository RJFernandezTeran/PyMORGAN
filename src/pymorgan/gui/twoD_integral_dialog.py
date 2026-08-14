"""Dialog for interactive 2D ROI integral dynamics calculation and exponential fitting."""

from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyqt6")

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector

import pymorgan as pm
from pymorgan import helpers as hlp
from pymorgan.twoD.dataset import Dataset2D


class TwoDIntegralDynamicsDialog(QDialog):
    """Interactive dialog for 2D ROI volume integration and kinetic dynamics."""

    def __init__(self, parent, dataset: Dataset2D):
        super().__init__(parent)
        self.dataset = dataset
        self.delays = dataset.delays
        self.I_t2: np.ndarray | None = None
        self.fit_t2: np.ndarray | None = None
        self.fit_curve: np.ndarray | None = None
        self.rect_selector: RectangleSelector | None = None
        self._updating_from_selector = False

        self.setWindowTitle("2D Integral Dynamics")

        screen = QApplication.primaryScreen()
        screen_w = screen.availableGeometry().width() if screen else 1200
        dlg_w = min(1200, screen_w)
        self.resize(dlg_w, 660)

        self._init_ui()
        self._calculate_integral()

    def _get_axis_config(self):
        s = pm.get_settings()
        pump_axis_val = s.pump_axis
        if hasattr(pump_axis_val, "value"):
            pump_axis_val = pump_axis_val.value
        is_vertical = (pump_axis_val == "Vertical")

        flabel = s.freq_label.value if hasattr(s.freq_label, "value") else s.freq_label
        lstyle = s.label_style.value if hasattr(s.label_style, "value") else s.label_style
        flabel_delim = lstyle if lstyle in ("()", "[]", "/") else "()"
        units_dict = self.dataset.units if self.dataset is not None else {"unitsL": "cm-1"}
        unitsL = units_dict.get("unitsL", "cm-1")

        pump_label, probe_label = hlp.fmt2Dlabel(flabel_delim, flabel, unitsL)
        return is_vertical, pump_label, probe_label

    def _init_ui(self):
        layout = QHBoxLayout(self)

        # Left panel: ROI selection & 2D preview contour
        left_layout = QVBoxLayout()
        roi_group = QGroupBox("2D ROI Integration Box")
        roi_layout = QVBoxLayout(roi_group)

        # Pump min/max
        pump_span_layout = QHBoxLayout()
        pump_span_layout.addWidget(QLabel("Pump Min (cm⁻¹):"))
        self.spn_pump_min = QDoubleSpinBox()
        self.spn_pump_min.setRange(float(np.min(self.dataset.pump)), float(np.max(self.dataset.pump)))
        self.spn_pump_min.setValue(float(np.percentile(self.dataset.pump, 30)))
        pump_span_layout.addWidget(self.spn_pump_min)

        pump_span_layout.addWidget(QLabel("Max:"))
        self.spn_pump_max = QDoubleSpinBox()
        self.spn_pump_max.setRange(float(np.min(self.dataset.pump)), float(np.max(self.dataset.pump)))
        self.spn_pump_max.setValue(float(np.percentile(self.dataset.pump, 70)))
        pump_span_layout.addWidget(self.spn_pump_max)
        roi_layout.addLayout(pump_span_layout)

        # Probe min/max
        probe_span_layout = QHBoxLayout()
        probe_span_layout.addWidget(QLabel("Probe Min (cm⁻¹):"))
        self.spn_probe_min = QDoubleSpinBox()
        self.spn_probe_min.setRange(float(np.min(self.dataset.probe)), float(np.max(self.dataset.probe)))
        self.spn_probe_min.setValue(float(np.percentile(self.dataset.probe, 30)))
        probe_span_layout.addWidget(self.spn_probe_min)

        probe_span_layout.addWidget(QLabel("Max:"))
        self.spn_probe_max = QDoubleSpinBox()
        self.spn_probe_max.setRange(float(np.min(self.dataset.probe)), float(np.max(self.dataset.probe)))
        self.spn_probe_max.setValue(float(np.percentile(self.dataset.probe, 70)))
        probe_span_layout.addWidget(self.spn_probe_max)
        roi_layout.addLayout(probe_span_layout)

        self.spn_pump_min.valueChanged.connect(self._on_spinbox_changed)
        self.spn_pump_max.valueChanged.connect(self._on_spinbox_changed)
        self.spn_probe_min.valueChanged.connect(self._on_spinbox_changed)
        self.spn_probe_max.valueChanged.connect(self._on_spinbox_changed)

        # Preview t2 delay combo box (matching main window)
        t2_layout = QHBoxLayout()
        t2_layout.addWidget(QLabel("Preview t₂ delay:"))
        self.cb_t2_delay = QComboBox()
        for idx, t2_val in enumerate(self.dataset.delays):
            self.cb_t2_delay.addItem(f"{idx + 1}: {t2_val:.2f} ps")
        self.cb_t2_delay.currentIndexChanged.connect(self._update_plot_2d)
        t2_layout.addWidget(self.cb_t2_delay)
        roi_layout.addLayout(t2_layout)

        lbl_drag_info = QLabel("<b>Tip:</b> Click and drag on the 2D plot to interactively set the ROI box.")
        lbl_drag_info.setWordWrap(True)
        roi_layout.addWidget(lbl_drag_info)

        left_layout.addWidget(roi_group)

        # 2D contour canvas
        self.fig_2d, self.ax_2d = plt.subplots(figsize=(4.5, 4))
        self.canvas_2d = FigureCanvas(self.fig_2d)
        left_layout.addWidget(self.canvas_2d, stretch=1)

        # Right panel: Kinetic plot & fitting
        right_layout = QVBoxLayout()
        fit_group = QGroupBox("Kinetic Trace I(t₂) & Exponential Fit")
        fit_layout = QVBoxLayout(fit_group)

        # Time axis mode
        axis_layout = QHBoxLayout()
        axis_layout.addWidget(QLabel("Time axis:"))
        self.cb_time_axis = QComboBox()
        self.cb_time_axis.addItems(["Linear", "Symlog"])
        self.cb_time_axis.currentTextChanged.connect(self._update_plot_1d)
        axis_layout.addWidget(self.cb_time_axis)

        # Exponential fit model
        axis_layout.addWidget(QLabel("Fit Model:"))
        self.cb_fit_model = QComboBox()
        self.cb_fit_model.addItems(["Single Exp", "Double Exp"])
        axis_layout.addWidget(self.cb_fit_model)

        self.btn_fit_exp = QPushButton("Fit Kinetics")
        self.btn_fit_exp.setStyleSheet("font-weight: bold; background-color: #059669; color: white;")
        self.btn_fit_exp.clicked.connect(self._fit_exponential)
        axis_layout.addWidget(self.btn_fit_exp)

        fit_layout.addLayout(axis_layout)

        self.lbl_fit_results = QLabel("Fit results: (Click 'Fit Kinetics')")
        self.lbl_fit_results.setWordWrap(True)
        fit_layout.addWidget(self.lbl_fit_results)

        # 1D matplotlib canvas
        self.fig_1d, self.ax_1d = plt.subplots(figsize=(5.5, 4))
        self.canvas_1d = FigureCanvas(self.fig_1d)
        fit_layout.addWidget(self.canvas_1d, stretch=1)

        right_layout.addWidget(fit_group)

        # Actions & Export
        action_layout = QHBoxLayout()
        self.btn_export = QPushButton("Export Trace (CSV)...")
        self.btn_export.clicked.connect(self._export_csv)
        action_layout.addWidget(self.btn_export)

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)
        action_layout.addWidget(self.btn_close)

        right_layout.addLayout(action_layout)

        layout.addLayout(left_layout, stretch=1)
        layout.addLayout(right_layout, stretch=1)

    def _on_spinbox_changed(self):
        if not self._updating_from_selector:
            self._calculate_integral()

    def _on_rect_select(self, eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            return
        if eclick.ydata is None or erelease.ydata is None:
            return

        is_vertical, _, _ = self._get_axis_config()
        x1, x2 = sorted([float(eclick.xdata), float(erelease.xdata)])
        y1, y2 = sorted([float(eclick.ydata), float(erelease.ydata)])

        if is_vertical:
            # Y = Pump, X = Probe
            pump_min, pump_max = y1, y2
            probe_min, probe_max = x1, x2
        else:
            # X = Pump, Y = Probe
            pump_min, pump_max = x1, x2
            probe_min, probe_max = y1, y2

        p_lo, p_hi = float(np.min(self.dataset.pump)), float(np.max(self.dataset.pump))
        r_lo, r_hi = float(np.min(self.dataset.probe)), float(np.max(self.dataset.probe))

        pump_min = np.clip(pump_min, p_lo, p_hi)
        pump_max = np.clip(pump_max, p_lo, p_hi)
        probe_min = np.clip(probe_min, r_lo, r_hi)
        probe_max = np.clip(probe_max, r_lo, r_hi)

        if pump_min >= pump_max or probe_min >= probe_max:
            return

        self._updating_from_selector = True
        self.spn_pump_min.setValue(pump_min)
        self.spn_pump_max.setValue(pump_max)
        self.spn_probe_min.setValue(probe_min)
        self.spn_probe_max.setValue(probe_max)
        self._updating_from_selector = False

        self._calculate_integral()

    def _calculate_integral(self):
        p_min = self.spn_pump_min.value()
        p_max = self.spn_pump_max.value()
        r_min = self.spn_probe_min.value()
        r_max = self.spn_probe_max.value()

        if p_min >= p_max or r_min >= r_max:
            return

        try:
            delays, I_t2 = self.dataset.integral_dynamics((p_min, p_max), (r_min, r_max))
            self.delays = delays
            self.I_t2 = I_t2
            self.fit_curve = None
            self.lbl_fit_results.setText("Fit results: (Click 'Fit Kinetics')")

            self._update_plot_2d()
            self._update_plot_1d()
        except Exception as exc:
            pass

    def _draw_diagonal(self, ax):
        lo = max(float(np.min(self.dataset.pump)), float(np.min(self.dataset.probe)))
        hi = min(float(np.max(self.dataset.pump)), float(np.max(self.dataset.probe)))
        if lo < hi:
            diag_pts = np.linspace(lo, hi, 100)
            ax.plot(diag_pts, diag_pts, "k--", alpha=0.5)

    def _update_plot_2d(self):
        i_t2 = self.cb_t2_delay.currentIndex()
        i_t2 = int(np.clip(i_t2, 0, len(self.dataset.delays) - 1))
        t2_val = self.dataset.delays[i_t2]

        self.ax_2d.clear()

        from pymorgan.twoD.plot import plot_map
        plot_map(
            self.dataset,
            t2=t2_val,
            ax=self.ax_2d,
            show_colorbar=True,
            t2_label=False,
        )

        is_vertical, pump_label, probe_label = self._get_axis_config()
        if is_vertical:
            x_lbl, y_lbl = probe_label, pump_label
        else:
            x_lbl, y_lbl = pump_label, probe_label

        # Highlight ROI Box
        p_min = self.spn_pump_min.value()
        p_max = self.spn_pump_max.value()
        r_min = self.spn_probe_min.value()
        r_max = self.spn_probe_max.value()

        if is_vertical:
            rx, ry = r_min, p_min
            rw, rh = r_max - r_min, p_max - p_min
        else:
            rx, ry = p_min, r_min
            rw, rh = p_max - p_min, r_max - r_min

        rect = patches.Rectangle(
            (rx, ry),
            rw,
            rh,
            linewidth=2,
            edgecolor="yellow",
            facecolor="yellow",
            alpha=0.3,
        )
        self.ax_2d.add_patch(rect)

        self.ax_2d.set_title(f"2D Map & ROI (t$_2$ = {t2_val:.2f} ps)")
        self.ax_2d.set_xlabel(x_lbl)
        self.ax_2d.set_ylabel(y_lbl)

        # Re-attach RectangleSelector for interactive dragging
        self.rect_selector = RectangleSelector(
            self.ax_2d,
            self._on_rect_select,
            useblit=True,
            button=[1],
            minspanx=2,
            minspany=2,
            interactive=True,
        )

        self.fig_2d.tight_layout()
        self.canvas_2d.draw()

    def _update_plot_1d(self):
        if self.I_t2 is None:
            return

        self.ax_1d.clear()
        self.ax_1d.plot(self.delays, self.I_t2, "bo-", label="Integral I(t$_2$)")

        if self.fit_curve is not None and self.fit_t2 is not None:
            self.ax_1d.plot(self.fit_t2, self.fit_curve, "r-", linewidth=2.0, label="Exp Fit")

        if self.cb_time_axis.currentText() == "Symlog":
            self.ax_1d.set_xscale("symlog", linthresh=1.0)

        self.ax_1d.set_title("Integral Dynamics Trace")
        self.ax_1d.set_xlabel("Population time t$_2$ (ps)")
        self.ax_1d.set_ylabel("Integral Intensity I(t$_2$)")
        self.ax_1d.grid(True, linestyle=":", alpha=0.6)
        self.ax_1d.legend(loc="best")

        self.fig_1d.tight_layout()
        self.canvas_1d.draw()

    def _fit_exponential(self):
        if self.I_t2 is None:
            return

        from scipy.optimize import curve_fit

        model_type = self.cb_fit_model.currentText()
        pos_delays = self.delays >= 0
        t_fit = self.delays[pos_delays]
        y_fit = self.I_t2[pos_delays]

        if len(t_fit) < 3:
            QMessageBox.warning(self, "Fit Error", "Not enough positive t2 delay points for exponential fitting.")
            return

        try:
            if model_type == "Single Exp":
                def single_exp(t, y0, A1, tau1):
                    return y0 + A1 * np.exp(-t / max(tau1, 1e-3))

                p0 = [y_fit[-1], y_fit[0] - y_fit[-1], (t_fit[-1] - t_fit[0]) / 3.0]
                popt, _ = curve_fit(single_exp, t_fit, y_fit, p0=p0, maxfev=2000)

                t_grid = np.linspace(np.min(t_fit), np.max(t_fit), 200)
                self.fit_t2 = t_grid
                self.fit_curve = single_exp(t_grid, *popt)

                self.lbl_fit_results.setText(
                    f"Single Exp Fit: y₀ = {popt[0]:.4g}, A₁ = {popt[1]:.4g}, τ₁ = {popt[2]:.3f} ps"
                )
            else:
                def double_exp(t, y0, A1, tau1, A2, tau2):
                    return y0 + A1 * np.exp(-t / max(tau1, 1e-3)) + A2 * np.exp(-t / max(tau2, 1e-3))

                p0 = [y_fit[-1], (y_fit[0] - y_fit[-1]) / 2, 1.0, (y_fit[0] - y_fit[-1]) / 2, 10.0]
                popt, _ = curve_fit(double_exp, t_fit, y_fit, p0=p0, maxfev=3000)

                t_grid = np.linspace(np.min(t_fit), np.max(t_fit), 200)
                self.fit_t2 = t_grid
                self.fit_curve = double_exp(t_grid, *popt)

                self.lbl_fit_results.setText(
                    f"Double Exp Fit: y₀ = {popt[0]:.4g}, A₁ = {popt[1]:.4g}, τ₁ = {popt[2]:.3f} ps | A₂ = {popt[3]:.4g}, τ₂ = {popt[4]:.3f} ps"
                )

            self._update_plot_1d()
        except Exception as exc:
            QMessageBox.critical(self, "Fit Failed", f"Exponential fit error:\n{exc}")

    def _export_csv(self):
        if self.I_t2 is None:
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export Integral Dynamics CSV", "integral_dynamics.csv", "CSV Files (*.csv);;Text files (*.txt)")
        if not path:
            return

        try:
            data = np.column_stack([self.delays, self.I_t2])
            header = "t2_delay_ps,integral_intensity"
            np.savetxt(path, data, delimiter=",", header=header, comments="")
            QMessageBox.information(self, "Export Successful", f"Kinetic trace saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", f"Could not save file:\n{exc}")

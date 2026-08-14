"""Dialog for interactive click-to-pick 2D Gaussian spectral fitting."""

from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyqt6")

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

import pymorgan as pm
from pymorgan import helpers as hlp
from pymorgan.twoD.analyse import (
    evaluate_2d_gaussian_map,
    fit_2d_gaussian_global,
    fit_2d_gaussian_map,
)
from pymorgan.twoD.dataset import Dataset2D


class TwoDGaussianFitDialog(QDialog):
    """Interactive dialog for 2D Gaussian spectral fitting with click-to-pick."""

    def __init__(self, parent, dataset: Dataset2D):
        super().__init__(parent)
        self.dataset = dataset
        self.modes: list[dict] = []
        self.fitted_modes: list[dict] | None = None
        self.fit_map: np.ndarray | None = None
        self.residual_map: np.ndarray | None = None

        self.setWindowTitle("2D Gaussian Spectral Fitting")

        screen = QApplication.primaryScreen()
        screen_w = screen.availableGeometry().width() if screen else 1900
        dlg_w = min(1900, screen_w)
        self.resize(dlg_w, 720)

        self._init_ui()
        self._update_plot()

    def _init_ui(self):
        layout = QHBoxLayout(self)

        # Left panel: controls & mode table
        controls_layout = QVBoxLayout()
        controls_group = QGroupBox("Fitting Options & Modes")
        controls_group.setMaximumWidth(360)
        group_layout = QVBoxLayout(controls_group)

        # Checkboxes
        self.chk_correlated = QCheckBox("Correlated (tilted) 2D Gaussians")
        self.chk_correlated.setChecked(True)
        self.chk_correlated.toggled.connect(self._on_options_changed)
        group_layout.addWidget(self.chk_correlated)

        self.chk_global = QCheckBox("Global t₂ fit across all delays")
        self.chk_global.setChecked(False)
        group_layout.addWidget(self.chk_global)

        self.chk_norm_t2 = QCheckBox("Normalise each t₂ delay by max amplitude")
        self.chk_norm_t2.setToolTip("Rescales each t₂ map by its peak amplitude so weak cross-peaks and long delays contribute equally to the fit")
        self.chk_norm_t2.setChecked(False)
        group_layout.addWidget(self.chk_norm_t2)

        # Initial Anharmonicity guess
        anharm_layout = QHBoxLayout()
        anharm_layout.addWidget(QLabel("Default Anharm (cm⁻¹):"))
        self.spn_default_anharm = QDoubleSpinBox()
        self.spn_default_anharm.setRange(0.5, 200.0)
        self.spn_default_anharm.setValue(15.0)
        anharm_layout.addWidget(self.spn_default_anharm)
        group_layout.addLayout(anharm_layout)

        # Preview t2 delay combo box (matching main window)
        prev_t2_layout = QHBoxLayout()
        prev_t2_layout.addWidget(QLabel("Preview t₂ delay:"))
        self.cb_t2_delay = QComboBox()
        for idx, t2_val in enumerate(self.dataset.delays):
            self.cb_t2_delay.addItem(f"{idx + 1}: {t2_val:.2f} ps")
        self.cb_t2_delay.currentIndexChanged.connect(self._on_t2_delay_changed)
        prev_t2_layout.addWidget(self.cb_t2_delay)
        group_layout.addLayout(prev_t2_layout)

        # Parameter Bounds Group Box
        bounds_group = QGroupBox("Fit Parameter Bounds")
        bounds_layout = QVBoxLayout(bounds_group)

        anharm_bnd_layout = QHBoxLayout()
        anharm_bnd_layout.addWidget(QLabel("Anharm Δ:"))
        self.spn_anharm_min = QDoubleSpinBox()
        self.spn_anharm_min.setRange(0.0, 300.0)
        self.spn_anharm_min.setValue(2.0)
        self.spn_anharm_min.setPrefix("Min: ")
        anharm_bnd_layout.addWidget(self.spn_anharm_min)

        self.spn_anharm_max = QDoubleSpinBox()
        self.spn_anharm_max.setRange(0.5, 500.0)
        self.spn_anharm_max.setValue(50.0)
        self.spn_anharm_max.setPrefix("Max: ")
        anharm_bnd_layout.addWidget(self.spn_anharm_max)
        bounds_layout.addLayout(anharm_bnd_layout)

        sigma_bnd_layout = QHBoxLayout()
        sigma_bnd_layout.addWidget(QLabel("Width σ:"))
        self.spn_sigma_min = QDoubleSpinBox()
        self.spn_sigma_min.setRange(0.1, 100.0)
        self.spn_sigma_min.setValue(1.0)
        self.spn_sigma_min.setPrefix("Min: ")
        sigma_bnd_layout.addWidget(self.spn_sigma_min)

        self.spn_sigma_max = QDoubleSpinBox()
        self.spn_sigma_max.setRange(1.0, 500.0)
        self.spn_sigma_max.setValue(60.0)
        self.spn_sigma_max.setPrefix("Max: ")
        sigma_bnd_layout.addWidget(self.spn_sigma_max)
        bounds_layout.addLayout(sigma_bnd_layout)

        group_layout.addWidget(bounds_group)

        # Instructions
        lbl_info = QLabel("<b>Add Peak:</b> Click button below, then click peak on Data plot:<br/>• Blue circle (o) = GSB<br/>• Red cross (x) = ESA")
        lbl_info.setWordWrap(True)
        group_layout.addWidget(lbl_info)

        # Add Peak Mode Button
        self.btn_add_peak = QPushButton("Add Peak (Click Plot)")
        self.btn_add_peak.setCheckable(True)
        self.btn_add_peak.setChecked(False)
        self.btn_add_peak.setStyleSheet(
            "QPushButton:checked { background-color: #2563eb; color: white; font-weight: bold; }"
        )
        self.btn_add_peak.toggled.connect(self._on_add_peak_toggled)
        group_layout.addWidget(self.btn_add_peak)

        # Mode Table (narrow)
        self.table_modes = QTableWidget(0, 7)
        self.table_modes.setMaximumWidth(340)
        self.table_modes.setHorizontalHeaderLabels([
            "w1", "w3", "Δ", "GSB Amp", "ESA Amp", "σ1", "σ3"
        ])
        self.table_modes.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        for col, width in enumerate([48, 48, 38, 55, 55, 40, 40]):
            self.table_modes.setColumnWidth(col, width)
        group_layout.addWidget(self.table_modes)

        # Mode Action Buttons
        btn_layout1 = QHBoxLayout()
        self.btn_delete_mode = QPushButton("Delete Selected Mode")
        self.btn_delete_mode.clicked.connect(self._delete_selected_mode)
        btn_layout1.addWidget(self.btn_delete_mode)

        self.btn_clear_modes = QPushButton("Clear All")
        self.btn_clear_modes.clicked.connect(self._clear_modes)
        btn_layout1.addWidget(self.btn_clear_modes)
        group_layout.addLayout(btn_layout1)

        self.btn_run_fit = QPushButton("Run 2D Fit")
        self.btn_run_fit.setStyleSheet("font-weight: bold; background-color: #059669; color: white; padding: 6px;")
        self.btn_run_fit.clicked.connect(self._run_fit)
        group_layout.addWidget(self.btn_run_fit)

        controls_layout.addWidget(controls_group)

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)
        controls_layout.addWidget(self.btn_close)

        # Right panel: 3 Matplotlib axes (wide plotting canvas with linked axes)
        try:
            from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
        except ImportError:
            from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar

        self.fig, (self.ax_data, self.ax_fit, self.ax_res) = plt.subplots(
            1, 3, figsize=(12, 5), sharex=True, sharey=True
        )
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self._drag_start = None
        self.canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self.canvas.mpl_connect("button_release_event", self._on_canvas_release)

        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        layout.addLayout(controls_layout, stretch=1)
        layout.addLayout(plot_layout, stretch=4)

    def _on_add_peak_toggled(self, checked):
        if checked:
            self.btn_add_peak.setText("Click GSB & Drag to ESA...")
        else:
            self.btn_add_peak.setText("Add Peak (Click/Drag Plot)")

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

    def _on_canvas_press(self, event):
        if not self.btn_add_peak.isChecked():
            return
        if getattr(self.toolbar, "mode", "") != "":
            return
        if event.inaxes != self.ax_data:
            return
        if event.xdata is None or event.ydata is None:
            return

        self._drag_start = (float(event.xdata), float(event.ydata))

    def _on_canvas_release(self, event):
        if not self.btn_add_peak.isChecked():
            return
        if getattr(self.toolbar, "mode", "") != "":
            return
        if event.inaxes != self.ax_data or self._drag_start is None:
            self._drag_start = None
            return
        if event.xdata is None or event.ydata is None:
            self._drag_start = None
            return

        start_x, start_y = self._drag_start
        end_x, end_y = float(event.xdata), float(event.ydata)
        self._drag_start = None

        is_vertical, _, _ = self._get_axis_config()
        if is_vertical:
            w1_gsb = start_y
            w3_gsb = start_x
            w3_esa_click = end_x
        else:
            w1_gsb = start_x
            w3_gsb = start_y
            w3_esa_click = end_y

        anharm_drag = abs(w3_gsb - w3_esa_click)
        def_anharm = self.spn_default_anharm.value()

        if anharm_drag >= 1.0:
            anharm = float(anharm_drag)
            w3_esa = float(w3_esa_click)
        else:
            anharm = float(def_anharm)
            w3_esa = float(w3_gsb - anharm)

        # Autocalculate best initial amplitudes from experimental 2D data
        i_t2 = self.cb_t2_delay.currentIndex()
        i_t2 = int(np.clip(i_t2, 0, len(self.dataset.delays) - 1))
        map_2d = self.dataset.Z[:, :, i_t2]

        p_idx = int(np.argmin(np.abs(self.dataset.pump - w1_gsb)))
        r_gsb_idx = int(np.argmin(np.abs(self.dataset.probe - w3_gsb)))
        r_esa_idx = int(np.argmin(np.abs(self.dataset.probe - w3_esa)))

        amp_gsb_raw = float(map_2d[p_idx, r_gsb_idx])
        amp_esa_raw = float(map_2d[p_idx, r_esa_idx])

        amp_gsb = amp_gsb_raw if amp_gsb_raw != 0 else -1.0
        amp_esa = amp_esa_raw if amp_esa_raw != 0 else 0.8

        new_mode = {
            "w1": float(w1_gsb),
            "w3": float(w3_gsb),
            "anharm": float(anharm),
            "amp_gsb": float(amp_gsb),
            "amp_esa": float(amp_esa),
            "sigma_w1": 10.0,
            "sigma_w3": 10.0,
        }
        self.modes.append(new_mode)
        self.fitted_modes = None
        self.fit_map = None
        self.residual_map = None
        self.btn_add_peak.setChecked(False)
        self._populate_table()
        self._update_plot()

    def _populate_table(self):
        display_modes = self.fitted_modes if self.fitted_modes is not None else self.modes
        self.table_modes.setRowCount(len(display_modes))
        for row, m in enumerate(display_modes):
            w1_val = m.get("w1", 0.0)
            w3_val = m.get("w3", 0.0)
            anharm_val = m.get("anharm", 15.0)
            amp_g_val = m.get("amp_gsb", -1.0)
            amp_e_val = m.get("amp_esa", 0.8)
            sig1_val = m.get("sigma_w1", 10.0)
            sig3_val = m.get("sigma_w3", 10.0)

            self.table_modes.setItem(row, 0, QTableWidgetItem(f"{w1_val:.1f}"))
            self.table_modes.setItem(row, 1, QTableWidgetItem(f"{w3_val:.1f}"))
            self.table_modes.setItem(row, 2, QTableWidgetItem(f"{anharm_val:.1f}"))
            self.table_modes.setItem(row, 3, QTableWidgetItem(f"{amp_g_val:.2f}"))
            self.table_modes.setItem(row, 4, QTableWidgetItem(f"{amp_e_val:.2f}"))
            self.table_modes.setItem(row, 5, QTableWidgetItem(f"{sig1_val:.1f}"))
            self.table_modes.setItem(row, 6, QTableWidgetItem(f"{sig3_val:.1f}"))

    def _delete_selected_mode(self):
        selected_rows = self.table_modes.selectionModel().selectedRows()
        if not selected_rows:
            row = self.table_modes.currentRow()
            if row >= 0:
                selected_rows = [self.table_modes.model().index(row, 0)]
        if not selected_rows:
            QMessageBox.information(self, "No Selection", "Please select a mode row in the table to delete.")
            return

        rows = sorted([r.row() for r in selected_rows], reverse=True)
        for r in rows:
            if 0 <= r < len(self.modes):
                self.modes.pop(r)

        self.fitted_modes = None
        self.fit_map = None
        self.residual_map = None
        self._populate_table()
        self._update_plot()

    def _clear_modes(self):
        self.modes.clear()
        self.fitted_modes = None
        self.fit_map = None
        self.residual_map = None
        self._populate_table()
        self._update_plot()

    def _on_options_changed(self):
        self.fitted_modes = None
        self.fit_map = None
        self.residual_map = None
        self._update_plot()

    def _on_t2_delay_changed(self):
        self.fitted_modes = None
        self.fit_map = None
        self.residual_map = None
        self._update_plot()

    def _run_fit(self):
        if not self.modes:
            QMessageBox.warning(self, "No Modes", "Click on the left plot to pick at least one mode centre.")
            return

        i_t2 = self.cb_t2_delay.currentIndex()
        correlated = self.chk_correlated.isChecked()
        is_global = self.chk_global.isChecked()

        bounds_config = {
            "anharm_min": self.spn_anharm_min.value(),
            "anharm_max": self.spn_anharm_max.value(),
            "sigma_min": self.spn_sigma_min.value(),
            "sigma_max": self.spn_sigma_max.value(),
        }

        norm_t2 = self.chk_norm_t2.isChecked()

        try:
            if is_global:
                shared, all_delay_modes, fit_cube, res_cube = fit_2d_gaussian_global(
                    self.dataset.pump,
                    self.dataset.probe,
                    self.dataset.delays,
                    self.dataset.Z,
                    self.modes,
                    correlated=correlated,
                    bounds_config=bounds_config,
                    normalize_t2=norm_t2,
                    show_progress=True,
                )
                self.fitted_modes = all_delay_modes[i_t2]
                self.fit_map = fit_cube[:, :, i_t2]
                self.residual_map = res_cube[:, :, i_t2]
                QMessageBox.information(
                    self, "Global Fit Completed", f"Successfully fitted {len(self.modes)} mode(s) globally across {self.dataset.n_maps} delays."
                )
            else:
                map_2d = self.dataset.Z[:, :, i_t2]
                f_modes, f_map, r_map = fit_2d_gaussian_map(
                    self.dataset.pump,
                    self.dataset.probe,
                    map_2d,
                    self.modes,
                    correlated=correlated,
                    bounds_config=bounds_config,
                    normalize_t2=norm_t2,
                    show_progress=True,
                )
                self.fitted_modes = f_modes
                self.fit_map = f_map
                self.residual_map = r_map
                QMessageBox.information(
                    self, "Fit Completed", f"Successfully fitted {len(self.modes)} mode(s) for delay {self.dataset.delays[i_t2]:.2f} ps."
                )

            self._populate_table()
            self._update_plot()
        except Exception as exc:
            QMessageBox.critical(self, "Fit Failed", f"2D Gaussian fitting error:\n{exc}")

    def _update_plot(self):
        i_t2 = self.cb_t2_delay.currentIndex()
        i_t2 = int(np.clip(i_t2, 0, len(self.dataset.delays) - 1))
        t2_val = self.dataset.delays[i_t2]

        map_2d = self.dataset.Z[:, :, i_t2]

        # Calculate Data/Fit vmax and Residual vmax (10x sensitivity)
        vmax_data = float(np.max(np.abs(map_2d))) or 1.0
        vmax_fit = float(np.max(np.abs(self.fit_map))) if self.fit_map is not None else 0.0
        vmax_data_fit = float(max(vmax_data, vmax_fit))
        vmax_res = vmax_data_fit / 10.0

        self.ax_data.clear()
        self.ax_fit.clear()
        self.ax_res.clear()

        from pymorgan.twoD.plot import plot_map

        # 1. Plot Data
        plot_map(
            self.dataset,
            t2=t2_val,
            ax=self.ax_data,
            vmin=-vmax_data_fit,
            vmax=vmax_data_fit,
            show_colorbar=False,
            t2_label=False,
        )
        self.ax_data.set_title(f"Data ($t_2$ = {t2_val:.2f} ps)")

        # Draw GSB/ESA markers on Data plot
        active_modes = self.fitted_modes if self.fitted_modes is not None else self.modes
        is_vertical, _, _ = self._get_axis_config()
        for m in active_modes:
            w1 = m["w1"]
            w3_gsb = m["w3"]
            anharm = m.get("anharm", 15.0)
            w3_esa = w3_gsb - anharm

            if is_vertical:
                mx_gsb, my_gsb = w3_gsb, w1
                mx_esa, my_esa = w3_esa, w1
            else:
                mx_gsb, my_gsb = w1, w3_gsb
                mx_esa, my_esa = w1, w3_esa

            self.ax_data.plot(mx_gsb, my_gsb, "bo", markersize=7, markeredgecolor="white", markeredgewidth=1.5)
            self.ax_data.plot(mx_esa, my_esa, "rx", markersize=8, markeredgewidth=2.0)
            self.ax_data.plot([mx_gsb, mx_esa], [my_gsb, my_esa], "k:", alpha=0.6)

        # 2. Plot Fit (no Y labels/ticks)
        if self.fit_map is not None:
            fit_ds = Dataset2D(
                delays=[t2_val],
                pump=self.dataset.pump,
                probe=self.dataset.probe,
                Z=self.fit_map[:, :, np.newaxis],
                units=self.dataset.units,
                freq_units=self.dataset.freq_units,
            )
            plot_map(
                fit_ds,
                t2=t2_val,
                ax=self.ax_fit,
                vmin=-vmax_data_fit,
                vmax=vmax_data_fit,
                show_colorbar=False,
                show_ylabel=False,
                t2_label=False,
            )
            self.ax_fit.set_title("2D Gaussian Fit")
        else:
            self.ax_fit.set_title("2D Fit (Not run)")
        self.ax_fit.tick_params(labelleft=False)

        # 3. Plot Residual (10x Z scale, no Y labels/ticks)
        if self.residual_map is not None:
            res_ds = Dataset2D(
                delays=[t2_val],
                pump=self.dataset.pump,
                probe=self.dataset.probe,
                Z=self.residual_map[:, :, np.newaxis],
                units=self.dataset.units,
                freq_units=self.dataset.freq_units,
            )
            plot_map(
                res_ds,
                t2=t2_val,
                ax=self.ax_res,
                vmin=-vmax_res,
                vmax=vmax_res,
                show_colorbar=False,
                show_ylabel=False,
                t2_label=False,
            )
            self.ax_res.set_title("Residual (Data - Fit) [10x Z-scale]")
        else:
            self.ax_res.set_title("Residual [10x Z-scale]")
        self.ax_res.tick_params(labelleft=False)

        # Remove previous colorbar axes if present
        for cbar_attr in ("_cbar_fit_ax", "_cbar_res_ax", "_cbar_ax"):
            if hasattr(self, cbar_attr) and getattr(self, cbar_attr) is not None:
                try:
                    getattr(self, cbar_attr).remove()
                except Exception:
                    pass
                setattr(self, cbar_attr, None)

        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        s = pm.get_settings()
        cm_obj, _ = hlp.CalcCMAP(s.cmap, 30)
        cm_obj = hlp.zero_center_cmap(cm_obj, 30, int(s.white_levels))
        units_z = self.dataset.units.get("unitsZ", "mOD") if hasattr(self.dataset, "units") and isinstance(self.dataset.units, dict) else "mOD"

        # Colorbar 1: Data & Fit
        divider_fit = make_axes_locatable(self.ax_fit)
        self._cbar_fit_ax = divider_fit.append_axes("right", size="5%", pad=0.1)
        sm_fit = cm.ScalarMappable(norm=mcolors.Normalize(vmin=-vmax_data_fit, vmax=vmax_data_fit), cmap=cm_obj)
        sm_fit.set_array([])
        cbar_fit = self.fig.colorbar(sm_fit, cax=self._cbar_fit_ax)
        cbar_fit.set_label(r"$\Delta$A" + f" ({units_z})")

        # Colorbar 2: Residual (10x)
        divider_res = make_axes_locatable(self.ax_res)
        self._cbar_res_ax = divider_res.append_axes("right", size="5%", pad=0.1)
        sm_res = cm.ScalarMappable(norm=mcolors.Normalize(vmin=-vmax_res, vmax=vmax_res), cmap=cm_obj)
        sm_res.set_array([])
        cbar_res = self.fig.colorbar(sm_res, cax=self._cbar_res_ax)
        cbar_res.set_label(r"$\Delta$A" + f" ({units_z}) [10x]")

        self.fig.tight_layout()
        self.canvas.draw()

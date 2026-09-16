import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import pymorgan as pm

from ..log import get_logger
from ..twoD.kubo_fit import run_kubo_fit, simulate_2d_spectrum

logger = get_logger(__name__)


class KuboDialog(QDialog):
    def __init__(self, parent, dataset, pump_range=None, probe_range=None, t2_range=None):
        super().__init__(parent)
        self.setWindowTitle("Direct Kubo Model Fitting Setup")
        self.resize(1200, 850)
        self.dataset = dataset
        self.pump_range = pump_range
        self.probe_range = probe_range
        self.t2_range = t2_range
        self.fid_data = None
        self.fit_history = []
        self.last_fit_params = None

        self.num_bands = 0
        self.num_decays = 0

        # Setup t2 range filtering indices
        if t2_range is not None:
            t2_min, t2_max = t2_range
            self.delay_indices = [i for i, d in enumerate(dataset.delays) if t2_min <= d <= t2_max]
        else:
            self.delay_indices = list(range(len(dataset.delays)))

        if not self.delay_indices:
            self.delay_indices = list(range(len(dataset.delays)))

        # Setup slices coordinates
        self.pump_axis = dataset.pump
        if pump_range is not None:
            self.pump_axis = self.pump_axis[(self.pump_axis >= pump_range[0]) & (self.pump_axis <= pump_range[1])]
        self.probe_axis = dataset.probe
        if probe_range is not None:
            self.probe_axis = self.probe_axis[(self.probe_axis >= probe_range[0]) & (self.probe_axis <= probe_range[1])]

        # Get raw ROI from dataset.Z
        p_mask = (dataset.pump >= self.pump_axis[0]) & (dataset.pump <= self.pump_axis[-1])
        pr_mask = (dataset.probe >= self.probe_axis[0]) & (dataset.probe <= self.probe_axis[-1])
        raw_Z_roi = dataset.Z[p_mask, :, :][:, pr_mask, :]

        # Determine upsampling factor
        s = pm.get_settings()
        factor = int(getattr(s, "sd_interpolation_factor", 4))

        # Upsample axes
        pump_fine = np.linspace(self.pump_axis[0], self.pump_axis[-1], len(self.pump_axis) * factor)
        probe_fine = np.linspace(self.probe_axis[0], self.probe_axis[-1], len(self.probe_axis) * factor)

        # Interpolate the entire raw_Z_roi 3D cube onto the fine grid
        from scipy.interpolate import RegularGridInterpolator
        PP_fine, PR_fine = np.meshgrid(pump_fine, probe_fine, indexing='ij')
        pts = np.column_stack([PP_fine.ravel(), PR_fine.ravel()])

        # Sort axes ascending for RegularGridInterpolator
        p_indices = np.argsort(self.pump_axis)
        pr_indices = np.argsort(self.probe_axis)
        sorted_pump = self.pump_axis[p_indices]
        sorted_probe = self.probe_axis[pr_indices]

        Z_exp_roi_fine = np.zeros((len(pump_fine), len(probe_fine), raw_Z_roi.shape[2]))
        for idx in range(raw_Z_roi.shape[2]):
            sorted_slice = raw_Z_roi[p_indices, :, idx][:, pr_indices]
            slice_interp = RegularGridInterpolator(
                (sorted_pump, sorted_probe), sorted_slice,
                bounds_error=False, fill_value=0.0
            )
            Z_exp_roi_fine[:, :, idx] = slice_interp(pts).reshape(PP_fine.shape)

        # Re-assign to fine grids and pre-interpolated Z_roi
        self.pump_axis = pump_fine
        self.probe_axis = probe_fine
        self.Z_exp_roi = Z_exp_roi_fine

        self.init_ui()
        self.update_plots()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # Left Panel: Controls
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        # Delay Selector
        delay_grp = QGroupBox("Delay Selector")
        delay_layout = QVBoxLayout(delay_grp)
        self.delay_cb = QComboBox()
        self.delay_cb.addItems([f"{self.dataset.delays[i]:.2f} ps" for i in self.delay_indices])
        self.delay_cb.currentIndexChanged.connect(self.update_plots)
        delay_layout.addWidget(self.delay_cb)
        control_layout.addWidget(delay_grp)

        # Parameter Table Group
        param_grp = QGroupBox("Kubo Bands && Decays Table")
        param_layout = QVBoxLayout(param_grp)

        # Component management buttons
        comp_btn_layout = QGridLayout()
        self.btn_add_band = QPushButton("Add Band")
        self.btn_add_band.clicked.connect(lambda: self.add_band())
        self.btn_del_band = QPushButton("Delete Selected Band")
        self.btn_del_band.clicked.connect(self.delete_band)

        self.btn_add_band_plot = QPushButton("Add Band (Click/Drag Plot)")
        self.btn_add_band_plot.setCheckable(True)
        self.btn_add_band_plot.setChecked(False)
        self.btn_add_band_plot.setStyleSheet(
            "QPushButton:checked { background-color: #2563eb; color: white; font-weight: bold; }"
        )
        self.btn_add_band_plot.toggled.connect(self._on_add_band_toggled)

        self.btn_add_decay = QPushButton("Add Kubo Decay")
        self.btn_add_decay.clicked.connect(self.add_decay)
        self.btn_del_decay = QPushButton("Delete Kubo Decay")
        self.btn_del_decay.clicked.connect(self.delete_decay)

        comp_btn_layout.addWidget(self.btn_add_band, 0, 0)
        comp_btn_layout.addWidget(self.btn_del_band, 0, 1)
        comp_btn_layout.addWidget(self.btn_add_band_plot, 1, 0, 1, 2)
        comp_btn_layout.addWidget(self.btn_add_decay, 2, 0)
        comp_btn_layout.addWidget(self.btn_del_decay, 2, 1)
        param_layout.addLayout(comp_btn_layout)

        # Table Widget
        self.param_table = QTableWidget()
        self.param_table.setColumnCount(5)
        self.param_table.setHorizontalHeaderLabels(["Parameter", "p₀", "LB", "UB", "Fix?"])
        self.param_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.param_table.setAlternatingRowColors(True)
        param_layout.addWidget(self.param_table)

        # Copy fitted parameters to initial value button
        self.btn_pfit_to_p0 = QPushButton("pFit -> p0")
        self.btn_pfit_to_p0.clicked.connect(lambda: self.pfit_to_p0(show_dialog=True))
        self.btn_pfit_to_p0.setStyleSheet("font-weight: bold; background-color: #27ae60; color: white;")
        param_layout.addWidget(self.btn_pfit_to_p0)

        control_layout.addWidget(param_grp)

        # Add the first band and decay by default
        self.add_band()
        self.add_decay()

        # FT Parameters Box
        ft_grp = QGroupBox("Fourier Transform && Simulation Settings")
        ft_grid = QGridLayout(ft_grp)

        ft_grid.addWidget(QLabel("t_max (ps):"), 0, 0)
        self.ft_tmax_spin = QDoubleSpinBox()
        self.ft_tmax_spin.setRange(0.5, 500.0)
        self.ft_tmax_spin.setValue(5.0)
        self.ft_tmax_spin.setSingleStep(0.5)
        self.ft_tmax_spin.setToolTip("Maximum simulation time in ps (t1 and t3 axes). Longer times increase spectral resolution.")
        ft_grid.addWidget(self.ft_tmax_spin, 0, 1)

        ft_grid.addWidget(QLabel("Steps (nt):"), 1, 0)
        self.ft_nt_spin = QSpinBox()
        self.ft_nt_spin.setRange(8, 4096)
        self.ft_nt_spin.setValue(128)
        self.ft_nt_spin.setSingleStep(8)
        self.ft_nt_spin.setToolTip("Number of grid points calculated in the time domain (t1 and t3).")
        ft_grid.addWidget(self.ft_nt_spin, 1, 1)

        self.ft_auto_check = QCheckBox("Auto nt (Nyquist Limit)")
        self.ft_auto_check.setChecked(False)
        self.ft_auto_check.setToolTip("Automatically determine the minimum number of time steps required based on the Nyquist limit.")
        ft_grid.addWidget(self.ft_auto_check, 2, 0, 1, 2)

        ft_grid.addWidget(QLabel("Undersampling U:"), 3, 0)
        self.ft_u_spin = QDoubleSpinBox()
        self.ft_u_spin.setRange(0.1, 20.0)
        self.ft_u_spin.setValue(1.0)
        self.ft_u_spin.setSingleStep(0.1)
        self.ft_u_spin.setToolTip("Allows calculation of fewer time steps by raising the Nyquist step size; resolution is recovered via increased zero padding.")
        ft_grid.addWidget(self.ft_u_spin, 3, 1)

        ft_grid.addWidget(QLabel("Rotating Frame w_rot:"), 4, 0)
        self.ft_wrot_spin = QDoubleSpinBox()
        self.ft_wrot_spin.setRange(100.0, 5000.0)
        self.ft_wrot_spin.setValue(1900.0)
        self.ft_wrot_spin.setSingleStep(10.0)
        self.ft_wrot_spin.setToolTip("Reference frequency (cm⁻¹) for the rotating frame representation.")
        ft_grid.addWidget(self.ft_wrot_spin, 4, 1)

        self.ft_wrot_auto_check = QCheckBox("Auto w_rot (min probe - 150 cm⁻¹)")
        self.ft_wrot_auto_check.setChecked(True)
        self.ft_wrot_auto_check.setToolTip("Sets the rotating frame frequency automatically to probe minimum minus 150 cm⁻¹.")
        self.ft_wrot_spin.setEnabled(False)
        self.ft_wrot_auto_check.stateChanged.connect(lambda state: self.ft_wrot_spin.setEnabled(state == 0))
        ft_grid.addWidget(self.ft_wrot_auto_check, 5, 0, 1, 2)

        control_layout.addWidget(ft_grp)

        # FTIR Integration Layout
        ftir_grp = QGroupBox("1st Order FTIR Response")
        ftir_layout = QVBoxLayout(ftir_grp)
        self.btn_load_ftir = QPushButton("Load FTIR Spectrum")
        self.btn_load_ftir.clicked.connect(self.load_ftir_spectrum)
        self.ftir_status = QLabel("No FTIR loaded (using Kubo lineshape).")
        self.ftir_status.setWordWrap(True)
        self.ftir_status.setStyleSheet("color: gray;")
        ftir_layout.addWidget(self.btn_load_ftir)
        ftir_layout.addWidget(self.ftir_status)
        control_layout.addWidget(ftir_grp)

        # Explicit Update Button
        self.btn_update = QPushButton("Update Plot")
        self.btn_update.clicked.connect(self.update_plots)
        self.btn_update.setStyleSheet("font-weight: bold; background-color: #2a82e6; color: white;")
        control_layout.addWidget(self.btn_update)

        # Fitting Buttons
        self.chk_auto_update_fit = QCheckBox("Auto-update plots during fit")
        self.chk_auto_update_fit.setChecked(True)
        control_layout.addWidget(self.chk_auto_update_fit)

        self.chk_global_fit = QCheckBox("Fit Kubo parameters globally")
        self.chk_global_fit.setChecked(False)
        control_layout.addWidget(self.chk_global_fit)

        self.chk_auto_update_p0 = QCheckBox("Auto-update p0 with best fit")
        self.chk_auto_update_p0.setChecked(True)
        control_layout.addWidget(self.chk_auto_update_p0)

        btn_layout = QHBoxLayout()
        self.btn_fit_active = QPushButton("Fit Active Delay")
        self.btn_fit_active.clicked.connect(self.run_active_fit)
        self.btn_fit_all = QPushButton("Fit All Delays")
        self.btn_fit_all.clicked.connect(self.run_all_fit)
        btn_layout.addWidget(self.btn_fit_active)
        btn_layout.addWidget(self.btn_fit_all)
        control_layout.addLayout(btn_layout)

        self.btn_save_fit = QPushButton("Save Fit Results")
        self.btn_save_fit.clicked.connect(self.save_fit_results)
        self.btn_save_fit.setStyleSheet("font-weight: bold; background-color: #8e44ad; color: white;")
        control_layout.addWidget(self.btn_save_fit)

        main_layout.addWidget(control_widget, stretch=1)

        # Right Panel: 2x2 Canvas
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)

        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
        from matplotlib.figure import Figure

        self.fig = Figure(figsize=(10, 8))
        ax0 = self.fig.add_subplot(2, 2, 1)
        ax1 = self.fig.add_subplot(2, 2, 2, sharex=ax0, sharey=ax0)
        ax2 = self.fig.add_subplot(2, 2, 3, sharex=ax0, sharey=ax0)
        ax3 = self.fig.add_subplot(2, 2, 4)
        self.axes = [ax0, ax1, ax2, ax3]

        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self._drag_start = None
        self.canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self.canvas.mpl_connect("button_release_event", self._on_canvas_release)

        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        main_layout.addWidget(plot_widget, stretch=2)

    def _on_add_band_toggled(self, checked):
        if checked:
            self.btn_add_band_plot.setText("Click GSB & Drag to ESA...")
        else:
            self.btn_add_band_plot.setText("Add Band (Click/Drag Plot)")

    def _on_canvas_press(self, event):
        if not hasattr(self, "btn_add_band_plot") or not self.btn_add_band_plot.isChecked():
            return
        if getattr(self.toolbar, "mode", "") != "":
            return
        if event.inaxes != self.axes[0]:
            return
        if event.xdata is None or event.ydata is None:
            return

        self._drag_start = (float(event.xdata), float(event.ydata))

    def _on_canvas_release(self, event):
        if not hasattr(self, "btn_add_band_plot") or not self.btn_add_band_plot.isChecked():
            return
        if getattr(self.toolbar, "mode", "") != "":
            return
        if event.inaxes != self.axes[0] or self._drag_start is None:
            self._drag_start = None
            return
        if event.xdata is None or event.ydata is None:
            self._drag_start = None
            return

        start_x, start_y = self._drag_start
        end_x, end_y = float(event.xdata), float(event.ydata)
        self._drag_start = None

        s = pm.get_settings()
        pump_axis_val = getattr(s, "pump_axis", "Horizontal")
        if hasattr(pump_axis_val, "value"):
            pump_axis_val = pump_axis_val.value
        is_vertical = (pump_axis_val == "Vertical")

        if is_vertical:
            w1_gsb = start_y
            w3_gsb = start_x
            w3_esa_click = end_x
        else:
            w1_gsb = start_x
            w3_gsb = start_y
            w3_esa_click = end_y

        anharm_drag = abs(w3_gsb - w3_esa_click)
        if anharm_drag >= 1.0:
            anharm = float(anharm_drag)
            w3_esa = float(w3_esa_click)
        else:
            anharm = 20.0
            w3_esa = float(w3_gsb - anharm)

        # Autocalculate best initial amplitudes from experimental 2D ROI data
        i_t2 = self.delay_cb.currentIndex()
        if 0 <= i_t2 < self.Z_exp_roi.shape[2]:
            map_2d = self.Z_exp_roi[:, :, i_t2]
        else:
            map_2d = self.Z_exp_roi[:, :, 0]

        p_idx = int(np.argmin(np.abs(self.pump_axis - w1_gsb)))
        r_gsb_idx = int(np.argmin(np.abs(self.probe_axis - w3_gsb)))
        r_esa_idx = int(np.argmin(np.abs(self.probe_axis - w3_esa)))

        amp_gsb_raw = float(map_2d[p_idx, r_gsb_idx])
        amp_esa_raw = float(map_2d[p_idx, r_esa_idx])

        amp_gsb = amp_gsb_raw if amp_gsb_raw != 0 else 1.0
        amp_esa = amp_esa_raw if amp_esa_raw != 0 else 0.8

        self.add_band(w01=w1_gsb, dw=anharm, a_gsb=amp_gsb, a_esa=amp_esa)
        self.btn_add_band_plot.setChecked(False)
        self.update_plots()

    def add_band(self, w01=None, dw=None, a_gsb=None, a_esa=None):
        self.num_bands += 1
        k = self.num_bands

        p_mean = float(np.mean(self.pump_axis))
        p_min = float(np.min(self.pump_axis))
        p_max = float(np.max(self.pump_axis))
        w01_lb = p_min - (p_max - p_min) * 0.5
        w01_ub = p_max + (p_max - p_min) * 0.5

        val_w01 = float(w01) if w01 is not None else p_mean
        val_dw = float(dw) if dw is not None else 20.0
        val_g = float(a_gsb) if a_gsb is not None else 1.0
        val_e = float(a_esa) if a_esa is not None else 0.8

        defaults = [
            (f"w01_{k} (cm⁻¹)", val_w01, w01_lb, w01_ub),
            (f"Δ_{k} (cm⁻¹)", val_dw, -100.0, 100.0),
            (f"A_GSB_{k}", val_g, -100.0, 100.0),
            (f"A_ESA_{k}", val_e, -100.0, 100.0)
        ]

        # Insert bands at the row boundary between bands and decays
        insert_idx = 4 * (k - 1)
        for name, val, lb, ub in defaults:
            self.param_table.insertRow(insert_idx)

            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            name_item.setBackground(Qt.GlobalColor.lightGray)
            self.param_table.setItem(insert_idx, 0, name_item)

            val_item = QTableWidgetItem(f"{val:.2f}")
            self.param_table.setItem(insert_idx, 1, val_item)

            lb_item = QTableWidgetItem(f"{lb:.2f}")
            self.param_table.setItem(insert_idx, 2, lb_item)

            ub_item = QTableWidgetItem(f"{ub:.2f}")
            self.param_table.setItem(insert_idx, 3, ub_item)

            chk_widget = QWidget()
            chk_layout = QHBoxLayout(chk_widget)
            chk_layout.setContentsMargins(0, 0, 0, 0)
            chk_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chk = QCheckBox()
            chk_layout.addWidget(chk)
            self.param_table.setCellWidget(insert_idx, 4, chk_widget)

            insert_idx += 1

    def delete_band(self):
        if self.num_bands <= 1:
            QMessageBox.information(self, "Minimum Bands", "Must keep at least 1 Kubo band.")
            return

        selected_rows = self.param_table.selectionModel().selectedRows()
        if not selected_rows:
            row = self.param_table.currentRow()
            if row >= 0:
                selected_rows = [self.param_table.model().index(row, 0)]

        if selected_rows:
            sel_row = selected_rows[0].row()
            if sel_row < 4 * self.num_bands:
                band_k = sel_row // 4
            else:
                band_k = self.num_bands - 1
        else:
            band_k = self.num_bands - 1

        start_row = 4 * band_k
        for _ in range(4):
            self.param_table.removeRow(start_row)

        self.num_bands -= 1
        self.relabel_bands_and_decays()
        self.update_plots()

    def relabel_bands_and_decays(self):
        for k in range(self.num_bands):
            idx = 4 * k
            if self.param_table.item(idx, 0):
                self.param_table.item(idx + 0, 0).setText(f"w01_{k+1} (cm⁻¹)")
                self.param_table.item(idx + 1, 0).setText(f"Δ_{k+1} (cm⁻¹)")
                self.param_table.item(idx + 2, 0).setText(f"A_GSB_{k+1}")
                self.param_table.item(idx + 3, 0).setText(f"A_ESA_{k+1}")

    def add_decay(self):
        self.num_decays += 1
        k = self.num_decays

        defaults = [
            (f"d1_{k} (cm⁻¹)", 20.0, 0.0, 200.0),
            (f"tau_{k} (ps)", 5.0, 0.01, 100.0),
            (f"dT2_{k} (cm⁻¹)", 1.0, 0.0, 100.0)
        ]

        # Decays are appended to the very end of the table
        for name, val, lb, ub in defaults:
            row = self.param_table.rowCount()
            self.param_table.insertRow(row)

            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.param_table.setItem(row, 0, name_item)

            val_item = QTableWidgetItem(f"{val:.2f}")
            self.param_table.setItem(row, 1, val_item)

            lb_item = QTableWidgetItem(f"{lb:.2f}")
            self.param_table.setItem(row, 2, lb_item)

            ub_item = QTableWidgetItem(f"{ub:.2f}")
            self.param_table.setItem(row, 3, ub_item)

            chk_widget = QWidget()
            chk_layout = QHBoxLayout(chk_widget)
            chk_layout.setContentsMargins(0, 0, 0, 0)
            chk_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chk = QCheckBox()
            chk_layout.addWidget(chk)
            self.param_table.setCellWidget(row, 4, chk_widget)

    def delete_decay(self):
        if self.num_decays > 1:
            # Delete last 3 rows of the table
            for _ in range(3):
                self.param_table.removeRow(self.param_table.rowCount() - 1)
            self.num_decays -= 1

    def pfit_to_p0(self, show_dialog=False):
        if self.last_fit_params is None:
            if show_dialog:
                QMessageBox.warning(self, "No Fitted Parameters", "Please run a fit first before copying fitted values.")
            return

        idx = 0
        # Bands
        for k in range(self.num_bands):
            row_offset = 4 * k
            # w01
            self.param_table.item(row_offset + 0, 1).setText(f"{self.last_fit_params[idx]:.4f}")
            idx += 1
            # dw
            self.param_table.item(row_offset + 1, 1).setText(f"{self.last_fit_params[idx]:.4f}")
            idx += 1
            # a_gsb
            self.param_table.item(row_offset + 2, 1).setText(f"{self.last_fit_params[idx]:.4f}")
            idx += 1
            # a_esa
            self.param_table.item(row_offset + 3, 1).setText(f"{self.last_fit_params[idx]:.4f}")
            idx += 1

        # Decays
        for k in range(self.num_decays):
            row_offset = 4 * self.num_bands + 3 * k
            # dw_1
            self.param_table.item(row_offset + 0, 1).setText(f"{self.last_fit_params[idx]:.4f}")
            idx += 1
            # tau_c
            self.param_table.item(row_offset + 1, 1).setText(f"{self.last_fit_params[idx]:.4f}")
            idx += 1
            # dw_T2
            self.param_table.item(row_offset + 2, 1).setText(f"{self.last_fit_params[idx]:.4f}")
            idx += 1

        if show_dialog:
            QMessageBox.information(self, "Success", "Fitted parameters copied successfully to Initial Value (p0) column.")

    def get_current_params(self):
        p0 = []
        bounds = []

        # Read Bands: 4 rows each
        for k in range(self.num_bands):
            row_offset = 4 * k
            try:
                w01 = float(self.param_table.item(row_offset + 0, 1).text())
                w01_lb = float(self.param_table.item(row_offset + 0, 2).text())
                w01_ub = float(self.param_table.item(row_offset + 0, 3).text())
                w01_fixed = self.param_table.cellWidget(row_offset + 0, 4).findChild(QCheckBox).isChecked()

                dw = float(self.param_table.item(row_offset + 1, 1).text())
                dw_lb = float(self.param_table.item(row_offset + 1, 2).text())
                dw_ub = float(self.param_table.item(row_offset + 1, 3).text())
                dw_fixed = self.param_table.cellWidget(row_offset + 1, 4).findChild(QCheckBox).isChecked()

                a_gsb = float(self.param_table.item(row_offset + 2, 1).text())
                a_gsb_lb = float(self.param_table.item(row_offset + 2, 2).text())
                a_gsb_ub = float(self.param_table.item(row_offset + 2, 3).text())
                a_gsb_fixed = self.param_table.cellWidget(row_offset + 2, 4).findChild(QCheckBox).isChecked()

                a_esa = float(self.param_table.item(row_offset + 3, 1).text())
                a_esa_lb = float(self.param_table.item(row_offset + 3, 2).text())
                a_esa_ub = float(self.param_table.item(row_offset + 3, 3).text())
                a_esa_fixed = self.param_table.cellWidget(row_offset + 3, 4).findChild(QCheckBox).isChecked()
            except Exception:
                p_mean = float(np.mean(self.pump_axis))
                p_min = float(np.min(self.pump_axis))
                p_max = float(np.max(self.pump_axis))
                w01 = p_mean
                w01_lb = p_min - (p_max - p_min) * 0.5
                w01_ub = p_max + (p_max - p_min) * 0.5
                w01_fixed = False
                dw, dw_lb, dw_ub, dw_fixed = 20.0, -100.0, 100.0, False
                a_gsb, a_gsb_lb, a_gsb_ub, a_gsb_fixed = 1.0, 0.0, 100.0, False
                a_esa, a_esa_lb, a_esa_ub, a_esa_fixed = 0.8, 0.0, 100.0, False

            p0.extend([w01, dw, a_gsb, a_esa])
            bounds.extend([
                (w01, w01) if w01_fixed else (w01_lb, w01_ub),
                (dw, dw) if dw_fixed else (dw_lb, dw_ub),
                (a_gsb, a_gsb) if a_gsb_fixed else (a_gsb_lb, a_gsb_ub),
                (a_esa, a_esa) if a_esa_fixed else (a_esa_lb, a_esa_ub)
            ])

        # Read Decays: 3 rows each (starting after index 4 * num_bands)
        for k in range(self.num_decays):
            row_offset = 4 * self.num_bands + 3 * k
            try:
                dw_1 = float(self.param_table.item(row_offset + 0, 1).text())
                dw_1_lb = float(self.param_table.item(row_offset + 0, 2).text())
                dw_1_ub = float(self.param_table.item(row_offset + 0, 3).text())
                dw_1_fixed = self.param_table.cellWidget(row_offset + 0, 4).findChild(QCheckBox).isChecked()

                tau_c = float(self.param_table.item(row_offset + 1, 1).text())
                tau_c_lb = float(self.param_table.item(row_offset + 1, 2).text())
                tau_c_ub = float(self.param_table.item(row_offset + 1, 3).text())
                tau_c_fixed = self.param_table.cellWidget(row_offset + 1, 4).findChild(QCheckBox).isChecked()

                dw_T2 = float(self.param_table.item(row_offset + 2, 1).text())
                dw_T2_lb = float(self.param_table.item(row_offset + 2, 2).text())
                dw_T2_ub = float(self.param_table.item(row_offset + 2, 3).text())
                dw_T2_fixed = self.param_table.cellWidget(row_offset + 2, 4).findChild(QCheckBox).isChecked()
            except Exception:
                dw_1, dw_1_lb, dw_1_ub, dw_1_fixed = 20.0, 0.0, 200.0, False
                tau_c, tau_c_lb, tau_c_ub, tau_c_fixed = 5.0, 0.01, 100.0, False
                dw_T2, dw_T2_lb, dw_T2_ub, dw_T2_fixed = 1.0, 0.0, 100.0, False

            p0.extend([dw_1, tau_c, dw_T2])
            bounds.extend([
                (dw_1, dw_1) if dw_1_fixed else (dw_1_lb, dw_1_ub),
                (tau_c, tau_c) if tau_c_fixed else (tau_c_lb, tau_c_ub),
                (dw_T2, dw_T2) if dw_T2_fixed else (dw_T2_lb, dw_T2_ub)
            ])

        return p0, bounds

    def load_ftir_spectrum(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load FTIR Spectrum File", "", "Text Files (*.txt *.csv *.dat);;All Files (*)"
        )
        if not path:
            return

        try:
            data = np.loadtxt(path)
            w_ftir = data[:, 0]
            abs_ftir = data[:, 1]

            if w_ftir[1] < w_ftir[0]:
                w_ftir = w_ftir[::-1]
                abs_ftir = abs_ftir[::-1]

            w_center = np.mean(self.pump_axis)
            w_grid = np.linspace(w_center - 300, w_center + 300, 4096)
            abs_grid = np.interp(w_grid, w_ftir, abs_ftir, left=0.0, right=0.0)

            fid = np.fft.ifft(np.fft.fftshift(abs_grid))

            from ..twoD.kubo_fit import c_0
            dw = w_grid[1] - w_grid[0]
            dt_ftir = 1.0 / (len(w_grid) * dw * c_0)
            fid_t = np.arange(len(fid)) * dt_ftir
            fid_abs = np.abs(fid)
            fid_abs /= np.max(fid_abs)

            self.ftir_spectrum = (w_ftir, abs_ftir)
            self.fid_data = {
                "fid": (fid_t, fid_abs),
                "spectrum": (w_ftir, abs_ftir)
            }
            self.ftir_status.setText("FTIR Loaded. Using experimental 1st order FID.")
            self.ftir_status.setStyleSheet("color: green; font-weight: bold;")
            self.update_plots()
        except Exception as exc:
            QMessageBox.critical(self, "Load Failed", f"Failed to compute 1st order response function: {exc}")

    def get_ft_settings(self):
        tmax = self.ft_tmax_spin.value()
        nt = None if self.ft_auto_check.isChecked() else self.ft_nt_spin.value()
        u = self.ft_u_spin.value()
        w_rot = (np.min(self.probe_axis) - 150.0) if self.ft_wrot_auto_check.isChecked() else self.ft_wrot_spin.value()
        return tmax, nt, u, w_rot

    def update_plots(self):
        self.btn_update.setEnabled(False)
        self.btn_update.setText("Updating...")
        from PyQt6.QtCore import QCoreApplication
        QCoreApplication.processEvents()

        try:
            p0, _ = self.get_current_params()
            total_params = len(p0)
            params_per_band = (total_params - 3 * self.num_decays) // self.num_bands

            bands = []
            for i in range(self.num_bands):
                offset = params_per_band * i
                if params_per_band == 2:
                    bands.append((p0[offset], p0[offset+1], 1.0, 1.0))
                else:
                    bands.append((p0[offset], p0[offset+1], p0[offset+2], p0[offset+3]))

            decays = []
            for j in range(self.num_decays):
                offset = params_per_band * self.num_bands + 3 * j
                decays.append((p0[offset], p0[offset+1], p0[offset+2]))

            delay_idx = self.delay_cb.currentIndex()
            actual_idx = self.delay_indices[delay_idx]
            t2 = self.dataset.delays[actual_idx]

            Z_exp_roi = self.Z_exp_roi[:, :, actual_idx]
            exp_max = np.max(np.abs(Z_exp_roi))
            Z_exp_roi_norm = Z_exp_roi / exp_max if exp_max > 0 else Z_exp_roi

            tmax, nt, u, w_rot = self.get_ft_settings()

            try:
                Z_sim = simulate_2d_spectrum(
                    bands, decays, self.pump_axis, self.probe_axis, t2,
                    tmax=tmax, nt=nt, w_rot=w_rot,
                    undersampling_factor=u, fid_data=self.fid_data
                )
            except Exception:
                import traceback
                traceback.print_exc()
                Z_sim = np.zeros_like(Z_exp_roi_norm)

            sim_max = np.max(np.abs(Z_sim))
            Z_sim_scaled = Z_sim / sim_max if sim_max > 0 else Z_sim
            Z_residuals = Z_exp_roi_norm - Z_sim_scaled

            # Recreate axes to clean up colorbars and keep them synced
            self.fig.clear()
            ax0 = self.fig.add_subplot(2, 2, 1)
            ax1 = self.fig.add_subplot(2, 2, 2, sharex=ax0, sharey=ax0)
            ax2 = self.fig.add_subplot(2, 2, 3, sharex=ax0, sharey=ax0)
            ax3 = self.fig.add_subplot(2, 2, 4)
            self.axes = [ax0, ax1, ax2, ax3]

            self.plot_contour(self.axes[0], Z_exp_roi_norm, -1.0, 1.0, "Experimental (ROI)")
            self.plot_contour(self.axes[1], Z_sim_scaled, -1.0, 1.0, "Simulation")
            self.plot_contour(self.axes[2], Z_residuals, -0.1, 0.1, "Residuals (Zscale=0.1)")

            # Overlay GSB/ESA peak markers for defined Kubo bands on ax0 (Experimental ROI)
            s = pm.get_settings()
            pump_axis_val = getattr(s, "pump_axis", "Horizontal")
            if hasattr(pump_axis_val, "value"):
                pump_axis_val = pump_axis_val.value
            is_vert = (pump_axis_val == "Vertical")

            for b in bands:
                w01, dw = b[0], b[1]
                w3_gsb = w01
                w3_esa = w01 - dw
                if is_vert:
                    mx_g, my_g = w3_gsb, w01
                    mx_e, my_e = w3_esa, w01
                else:
                    mx_g, my_g = w01, w3_gsb
                    mx_e, my_e = w01, w3_esa
                self.axes[0].plot(mx_g, my_g, "bo", markersize=7, markeredgecolor="white", markeredgewidth=1.5)
                self.axes[0].plot(mx_e, my_e, "rx", markersize=8, markeredgewidth=2.0)
                self.axes[0].plot([mx_g, mx_e], [my_g, my_e], "k:", alpha=0.6)

            # Bottom-Right: Kubo Fit Progress Plot (\Delta_1 vs \tau_c)
            if self.fit_history:
                history_arr = np.array(self.fit_history)
                self.axes[3].plot(history_arr[:, 1], history_arr[:, 0], "o-", color="darkorange", linewidth=1.5, markersize=4)
                # Mark starting and final points (empty circle and X, same size)
                self.axes[3].plot(history_arr[0, 1], history_arr[0, 0], "o", color="red", markerfacecolor="none", markeredgewidth=1.5, label="Start")
                self.axes[3].plot(history_arr[-1, 1], history_arr[-1, 0], "x", color="green", markeredgewidth=1.5, label="Final")
                self.axes[3].legend(loc="upper right", fontsize=8)
            elif hasattr(self.dataset, "_sd_results") and "Kubo_Fit" in self.dataset._sd_results:
                arr = self.dataset._sd_results["Kubo_Fit"]
                if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] > 6:
                    self.axes[3].plot(arr[:, 6], arr[:, 5], "o-", color="darkorange", linewidth=1.5, markersize=4)

            self.axes[3].set_xlabel(r"Correlation time $\tau_{c}$ (ps)")
            self.axes[3].set_ylabel(r"Fluctuation $\Delta_{1}$ ($\mathrm{cm}^{-1}$)")
            self.axes[3].set_title("Fitting Progress (Kubo coordinates)")

            self.fig.tight_layout()
            self.canvas.draw_idle()
        finally:
            self.btn_update.setEnabled(True)
            self.btn_update.setText("Update Plot")

    def plot_contour(self, ax, Z, vmin, vmax, title):
        from pymorgan import helpers as hlp

        from ..settings import get_settings
        s = get_settings()
        cmap_ID = s.cmap
        white_levels = s.white_levels

        # Check if pump axis is vertical
        pump_axis_val = s.pump_axis
        if hasattr(pump_axis_val, "value"):
            pump_axis_val = pump_axis_val.value
        is_vertical = (pump_axis_val == "Vertical")

        if is_vertical:
            X = self.probe_axis
            Y = self.pump_axis
            Z_contour = Z
        else:
            X = self.pump_axis
            Y = self.probe_axis
            Z_contour = Z.T

        NctrF = 30
        nw = int(white_levels) if white_levels else 2
        cm_obj, _ = hlp.CalcCMAP(cmap_ID, NctrF, Nwhite=nw)
        if white_levels:
            cm_obj = hlp.zero_center_cmap(cm_obj, NctrF, int(white_levels))

        ctrLvl_F = np.linspace(vmin, vmax, NctrF)

        cf = ax.contourf(X, Y, Z_contour, vmin=vmin, vmax=vmax, levels=ctrLvl_F, cmap=cm_obj, extend="neither")
        ax.set_title(title)
        ax.set_aspect('equal')

        # Add colorbar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        self.fig.colorbar(cf, cax=cax)

    def run_active_fit(self):
        p0_full, bounds_full = self.get_current_params()
        delay_idx = self.delay_cb.currentIndex()
        actual_idx = self.delay_indices[delay_idx]
        t2 = self.dataset.delays[actual_idx]

        Z_exp_roi = self.Z_exp_roi[:, :, actual_idx:actual_idx+1]

        tmax, nt, u, w_rot = self.get_ft_settings()

        parallel_mode = "disabled"

        # Clear and define real-time callback
        self.fit_history = []

        def iteration_callback(params):
            if not self.chk_auto_update_fit.isChecked():
                return
            if len(params) == 6:
                self.fit_history.append((params[3], params[4]))
            else:
                total_params = len(params)
                params_per_band = (total_params - 3 * self.num_decays) // self.num_bands
                offset = params_per_band * self.num_bands
                if len(params) >= (offset + 2):
                    self.fit_history.append((params[offset], params[offset+1]))

            self.axes[3].clear()
            if self.fit_history:
                history_arr = np.array(self.fit_history)
                self.axes[3].plot(history_arr[:, 1], history_arr[:, 0], "o-", color="darkorange", linewidth=1.5, markersize=4)
                self.axes[3].plot(history_arr[0, 1], history_arr[0, 0], "o", color="red", markerfacecolor="none", markeredgewidth=1.5, label="Start")
                self.axes[3].plot(history_arr[-1, 1], history_arr[-1, 0], "x", color="green", markeredgewidth=1.5, label="Final")
                self.axes[3].legend(loc="upper right", fontsize=8)
            self.axes[3].set_xlabel(r"Correlation time $\tau_{c}$ (ps)")
            self.axes[3].set_ylabel(r"Fluctuation $\Delta_{1}$ ($\mathrm{cm}^{-1}$)")
            self.axes[3].set_title("Fitting Progress (Kubo coordinates)")
            self.canvas.draw_idle()
            from PyQt6.QtCore import QCoreApplication
            QCoreApplication.processEvents()

        try:
            actual_fid = self.fid_data.get("fid") if isinstance(self.fid_data, dict) else self.fid_data
            fit_results = run_kubo_fit(
                Z_exp_roi, self.pump_axis, self.probe_axis, [t2],
                p0_full, bounds_full,
                tmax=tmax, nt=nt, w_rot=w_rot,
                undersampling_factor=u, fid_data=actual_fid,
                num_bands=self.num_bands, num_decays=self.num_decays,
                parallel_mode=parallel_mode
            )
            res = fit_results[0]

            self.fit_history = res.get("history", [])
            self.last_fit_params = res["params"]

            # Store results on the dataset
            if not hasattr(self.dataset, "_sd_results"):
                self.dataset._sd_results = {}

            # Build/update Kubo_Fit numpy array
            row = [t2] + list(res["params"])
            if "Kubo_Fit" in self.dataset._sd_results:
                arr = self.dataset._sd_results["Kubo_Fit"]
                # Replace row if already exists for this delay
                idx_matches = np.where(np.isclose(arr[:, 0], t2))[0]
                if len(idx_matches) > 0:
                    arr[idx_matches[0]] = row
                else:
                    self.dataset._sd_results["Kubo_Fit"] = np.vstack([arr, row])
            else:
                self.dataset._sd_results["Kubo_Fit"] = np.array([row])

            if not hasattr(self.dataset, "_sd_kubo_details"):
                self.dataset._sd_kubo_details = {}
            self.dataset._sd_kubo_details[t2] = {
                "params": res["params"],
                "errors": res["errors"],
                "num_bands": self.num_bands,
                "num_decays": self.num_decays,
                "param_names": [self.param_table.item(r, 0).text() for r in range(self.param_table.rowCount())],
                "tmax": tmax,
                "nt": nt,
                "u": u,
                "w_rot": w_rot,
                "fid_data": self.fid_data
            }

            # Construct or update the in-memory simulated dataset
            n_pump = len(self.dataset.pump)
            n_probe = len(self.dataset.probe)

            # Read or initialize Z_sim_cube
            if getattr(self.dataset, "_kubo_fit_dataset", None) is not None:
                Z_sim_cube = np.array(self.dataset._kubo_fit_dataset.Z)
            else:
                Z_sim_cube = np.zeros((n_pump, n_probe, len(self.dataset.delays)))

            # Reconstruct bands & decays
            if len(res["params"]) == 6:
                bands = [(res["params"][0], res["params"][1], 1.0, 1.0)]
                decays = [(res["params"][3], res["params"][4], res["params"][5])]
            else:
                total_params = len(res["params"])
                params_per_band = (total_params - 3 * self.num_decays) // self.num_bands
                bands = []
                for i in range(self.num_bands):
                    offset = params_per_band * i
                    if params_per_band == 2:
                        bands.append((res["params"][offset], res["params"][offset+1], 1.0, 1.0))
                    else:
                        bands.append((res["params"][offset], res["params"][offset+1], res["params"][offset+2], res["params"][offset+3]))
                decays = []
                for j in range(self.num_decays):
                    offset = params_per_band * self.num_bands + 3 * j
                    decays.append((res["params"][offset], res["params"][offset+1], res["params"][offset+2]))

            # Scale using fine ROI grids
            Z_exp_roi = self.Z_exp_roi[:, :, actual_idx]
            exp_max = np.max(np.abs(Z_exp_roi))
            Z_exp_roi_norm = Z_exp_roi / exp_max if exp_max > 0 else Z_exp_roi

            Z_sim_roi = simulate_2d_spectrum(
                bands, decays, self.pump_axis, self.probe_axis, t2,
                tmax=tmax, nt=nt, w_rot=w_rot,
                undersampling_factor=u, fid_data=self.fid_data
            )
            sim_roi_max = np.max(np.abs(Z_sim_roi))
            if sim_roi_max > 0:
                Z_sim_roi = Z_sim_roi / sim_roi_max
            scale = np.sum(Z_exp_roi_norm * Z_sim_roi) / (np.sum(Z_sim_roi**2) + 1e-12)

            Z_sim_full = simulate_2d_spectrum(
                bands, decays, self.dataset.pump, self.dataset.probe, t2,
                tmax=tmax, nt=nt, w_rot=w_rot,
                undersampling_factor=u, fid_data=self.fid_data
            )
            sim_full_max = np.max(np.abs(Z_sim_full))
            if sim_full_max > 0:
                Z_sim_full = Z_sim_full / sim_full_max
            Z_sim_cube[:, :, actual_idx] = scale * Z_sim_full * exp_max

            from ..twoD.dataset import Dataset2D
            self.dataset._kubo_fit_dataset = Dataset2D(
                Z_sim_cube, self.dataset.pump, self.dataset.probe, self.dataset.delays,
                self.dataset.units, self.dataset.freq_units,
                source="In-Memory Kubo Simulation", data_type="P2DAT", datatype="processed"
            )

            if self.chk_auto_update_p0.isChecked():
                self.pfit_to_p0(show_dialog=False)
            self.update_plots()
        except Exception as exc:
            QMessageBox.critical(self, "Fitting Failed", str(exc))

    def run_all_fit(self):
        p0_full, bounds_full = self.get_current_params()

        fit_delays = self.dataset.delays[self.delay_indices]
        Z_exp_roi = self.Z_exp_roi[:, :, self.delay_indices]
        if len(fit_delays) == 0:
            QMessageBox.warning(self, "Fit Error", "No delays selected in the specified t2 range.")
            return

        tmax, nt, u, w_rot = self.get_ft_settings()

        s = pm.get_settings()
        parallel_mode = s.parallel_fitting.value if hasattr(s.parallel_fitting, "value") else s.parallel_fitting

        import concurrent.futures

        from PyQt6.QtCore import QCoreApplication, Qt
        from PyQt6.QtWidgets import QProgressDialog

        from ..twoD.kubo_fit import fit_single_delay_worker

        progress = QProgressDialog("Fitting Kubo model across delays...", "Cancel", 0, len(fit_delays), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        QCoreApplication.processEvents()

        # Define callback to update fitting progress plot live
        self.fit_history = []
        def iteration_callback(params):
            if not self.chk_auto_update_fit.isChecked():
                return
            if len(params) == 6:
                self.fit_history.append((params[3], params[4]))
            else:
                total_params = len(params)
                params_per_band = (total_params - 3 * self.num_decays) // self.num_bands
                offset = params_per_band * self.num_bands
                if len(params) >= (offset + 2):
                    self.fit_history.append((params[offset], params[offset+1]))

            self.axes[3].clear()
            if self.fit_history:
                history_arr = np.array(self.fit_history)
                self.axes[3].plot(history_arr[:, 1], history_arr[:, 0], "o-", color="darkorange", linewidth=1.5, markersize=4)
                self.axes[3].plot(history_arr[0, 1], history_arr[0, 0], "o", color="red", markerfacecolor="none", markeredgewidth=1.5, label="Start")
                self.axes[3].plot(history_arr[-1, 1], history_arr[-1, 0], "x", color="green", markeredgewidth=1.5, label="Final")
                self.axes[3].legend(loc="upper right", fontsize=8)
            self.axes[3].set_xlabel(r"Correlation time $\tau_{c}$ (ps)")
            self.axes[3].set_ylabel(r"Fluctuation $\Delta_{1}$ ($\mathrm{cm}^{-1}$)")
            self.axes[3].set_title("Fitting Progress (Kubo coordinates)")
            self.canvas.draw_idle()
            QCoreApplication.processEvents()

        is_global = self.chk_global_fit.isChecked()
        actual_fid = self.fid_data.get("fid") if isinstance(self.fid_data, dict) else self.fid_data
        tasks = []
        for idx, t2 in enumerate(fit_delays):
            ic = iteration_callback if (parallel_mode == "disabled" and not is_global) else None
            tasks.append((Z_exp_roi[:, :, idx], self.pump_axis, self.probe_axis, t2, p0_full, bounds_full,
                          tmax, nt, w_rot, u, actual_fid, self.num_bands, self.num_decays, ic))

        fit_results = []

        if is_global:
            # We will implement global fitting here.
            # State description:
            # - We have num_bands bands, each with 4 parameters: [w01, dw, a_gsb, a_esa]
            # - We have num_decays decays, each with 3 parameters: [d1, tau, dT2]
            # Global fit parameters layout:
            # 1. Global linked parameters:
            #    - For each band: w01, dw (total 2 * num_bands)
            #    - For each decay: d1, tau, dT2 (total 3 * num_decays)
            # 2. Per-delay free parameters:
            #    - For each delay: a_gsb_1, a_esa_1, ..., a_gsb_N, a_esa_N (total 2 * num_bands * len(fit_delays))
            #
            # Let's construct initial values p0_global and bounds_global.
            p0_global = []
            bounds_global = []

            total_params = len(p0_full)
            params_per_band = (total_params - 3 * self.num_decays) // self.num_bands

            # 1. Linked band parameters (w01 and dw only)
            for i in range(self.num_bands):
                offset = params_per_band * i
                p0_global.extend([p0_full[offset], p0_full[offset + 1]])
                bounds_global.extend([bounds_full[offset], bounds_full[offset + 1]])

            # 2. Linked decay parameters
            for j in range(self.num_decays):
                offset = params_per_band * self.num_bands + 3 * j
                p0_global.extend([p0_full[offset], p0_full[offset + 1], p0_full[offset + 2]])
                bounds_global.extend([bounds_full[offset], bounds_full[offset + 1], bounds_full[offset + 2]])

            # 3. Per-delay GSB & ESA amplitudes
            for _ in range(len(fit_delays)):
                for i in range(self.num_bands):
                    offset = params_per_band * i
                    if params_per_band > 2:
                        p0_global.extend([p0_full[offset + 2], p0_full[offset + 3]])
                        bounds_global.extend([bounds_full[offset + 2], bounds_full[offset + 3]])
                    else:
                        p0_global.extend([1.0, 1.0])
                        bounds_global.extend([(0.0, 100.0), (0.0, 100.0)])

            # Helper logic to unpack the global parameters into bands and decays list for simulate_2d_spectrum for a given delay index.
            def unpack_delay_params(params_opt, delay_idx_in):
                # Unpacks global fit parameters for simulate_2d_spectrum at delay index delay_idx_in
                idx_ptr = 2 * self.num_bands + 3 * self.num_decays
                # skip previous delays
                idx_ptr += 2 * self.num_bands * delay_idx_in

                bands_opt = []
                for i in range(self.num_bands):
                    # Linked: w01, dw
                    w01 = params_opt[2 * i]
                    dw = params_opt[2 * i + 1]
                    # Free per delay: a_gsb, a_esa
                    a_gsb = params_opt[idx_ptr]
                    a_esa = params_opt[idx_ptr + 1]
                    idx_ptr += 2
                    bands_opt.append((w01, dw, a_gsb, a_esa))

                decays_opt = []
                linked_decays_start = 2 * self.num_bands
                for j in range(self.num_decays):
                    d1 = params_opt[linked_decays_start + 3 * j]
                    tau = params_opt[linked_decays_start + 3 * j + 1]
                    dT2 = params_opt[linked_decays_start + 3 * j + 2]
                    decays_opt.append((d1, tau, dT2))

                return bands_opt, decays_opt

            # Define joint cost function
            actual_fid = self.fid_data.get("fid") if isinstance(self.fid_data, dict) else self.fid_data

            # Pre-crop FTIR spectrum to pump_axis range if available
            ftir_w = None
            ftir_y = None
            if isinstance(self.fid_data, dict):
                ftir_spec = self.fid_data.get("spectrum")
                if ftir_spec is not None:
                    ftir_w, ftir_y = ftir_spec

            ftir_target_y = None
            if ftir_w is not None and ftir_y is not None:
                ftir_target_y = np.interp(self.pump_axis, ftir_w, ftir_y, left=0.0, right=0.0)
                ftir_max = np.max(np.abs(ftir_target_y))
                if ftir_max > 0:
                    ftir_target_y = ftir_target_y / ftir_max

            def global_cost_fun(params_opt):
                cost = 0.0
                # Sum 2D residuals over all active delays
                for idx_in, t2_in in enumerate(fit_delays):
                    bands_opt, decays_opt = unpack_delay_params(params_opt, idx_in)

                    Z_exp_slice = Z_exp_roi[:, :, idx_in]
                    exp_max = np.max(np.abs(Z_exp_slice))
                    Z_exp_norm = Z_exp_slice / exp_max if exp_max > 0 else Z_exp_slice

                    Z_sim = simulate_2d_spectrum(
                        bands_opt, decays_opt, self.pump_axis, self.probe_axis, t2_in,
                        tmax=tmax, nt=nt, w_rot=w_rot,
                        undersampling_factor=u, fid_data=actual_fid
                    )

                    sim_max = np.max(np.abs(Z_sim))
                    if sim_max > 0:
                        Z_sim = Z_sim / sim_max
                    scale = np.maximum(0.0, np.sum(Z_exp_norm * Z_sim) / (np.sum(Z_sim**2) + 1e-12))

                    residuals_2d = Z_exp_norm - scale * Z_sim
                    cost += np.sum(residuals_2d**2)

                # Add simultaneous FTIR fitting contribution (linked parameters, evaluate once)
                if ftir_target_y is not None:
                    # Retrieve the linked parameters (evaluate at delay index 0 or use delay-independent part)
                    bands_opt, decays_opt = unpack_delay_params(params_opt, 0)
                    t_ftir = np.linspace(0, tmax, nt or 128)
                    dt_ftir = t_ftir[1] - t_ftir[0]

                    from ..twoD.kubo_fit import c_0, g_function
                    g_t = np.zeros_like(t_ftir)
                    if actual_fid is not None:
                        fid_t, fid_abs = actual_fid
                        g_t = -np.log(np.interp(t_ftir, fid_t, fid_abs) + 1e-12)
                    else:
                        for d1, tau, dT2 in decays_opt:
                            T2 = 1.0 / (np.pi * c_0 * dT2) if dT2 > 0 else 0.0
                            g_t += g_function(t_ftir, 0.0, d1, tau, T2)

                    ftir_sim = np.zeros_like(self.pump_axis)
                    for w01, _dw, a_gsb, _a_esa in bands_opt:
                        w_shift = self.pump_axis - w01
                        integrand = np.exp(-g_t)[None, :] * np.exp(-1j * w_shift[:, None] * 2 * np.pi * c_0 * t_ftir[None, :])
                        band_profile = np.real(np.sum(integrand, axis=1) * dt_ftir)
                        ftir_sim += a_gsb * band_profile

                    ftir_sim_max = np.max(np.abs(ftir_sim))
                    if ftir_sim_max > 0:
                        ftir_sim = ftir_sim / ftir_sim_max

                    scale_ftir = np.sum(ftir_target_y * ftir_sim) / (np.sum(ftir_sim**2) + 1e-12)
                    residuals_ftir = ftir_target_y - scale_ftir * ftir_sim

                    s_sett = pm.get_settings()
                    weight = getattr(s_sett, "kubo_ftir_weight", 1.0)
                    cost += weight * np.sum(residuals_ftir**2)

                return cost

            progress.setLabelText("Running Global linked Kubo fit...")
            progress.setMaximum(50)
            progress.setValue(0)
            QCoreApplication.processEvents()

            # Minimise global cost function using reasonable tolerances to avoid getting stuck in noise
            from scipy.optimize import minimize

            # Allow PyQt event loop processing and visual updates to happen periodically
            # instead of at every single function call/step to avoid overloading Qt and hanging the system.
            step_counter = 0
            def callback(xk):
                nonlocal step_counter
                step_counter += 1
                progress.setValue(min(step_counter, 50))
                QCoreApplication.processEvents()

                # Update global parameter trace visual progress on axes[3]
                linked_decays_start = 2 * self.num_bands
                if len(xk) >= (linked_decays_start + 2):
                    self.fit_history.append((xk[linked_decays_start], xk[linked_decays_start + 1]))

                self.axes[3].clear()
                if self.fit_history:
                    history_arr = np.array(self.fit_history)
                    self.axes[3].plot(history_arr[:, 1], history_arr[:, 0], "o-", color="darkorange", linewidth=1.5, markersize=4)
                    self.axes[3].plot(history_arr[0, 1], history_arr[0, 0], "o", color="red", markerfacecolor="none", markeredgewidth=1.5, label="Start")
                    self.axes[3].plot(history_arr[-1, 1], history_arr[-1, 0], "x", color="green", markeredgewidth=1.5, label="Final")
                    self.axes[3].legend(loc="upper right", fontsize=8)
                self.axes[3].set_xlabel(r"Correlation time $\tau_{c}$ (ps)")
                self.axes[3].set_ylabel(r"Fluctuation $\Delta_{1}$ ($\mathrm{cm}^{-1}$)")
                self.axes[3].set_title("Fitting Progress (Kubo coordinates)")
                self.canvas.draw_idle()
                QCoreApplication.processEvents()

                if progress.wasCanceled():
                    raise KeyboardInterrupt("Fitting cancelled by user.")

            res_opt = minimize(
                global_cost_fun, p0_global, bounds=bounds_global, method='L-BFGS-B',
                callback=callback,
                options={'ftol': 1e-4, 'gtol': 1e-4, 'maxiter': 50}
            )

            if "CONVERGENCE" in str(res_opt.message).upper():
                interpretation = "The optimisation successfully converged to a local minimum."
            else:
                interpretation = str(res_opt.message)
            logger.info(
                "\n%s\nGlobal Kubo Fit Optimisation Finished.\n"
                "Status message: %s\nInterpretation: %s\n%s\n",
                "=" * 80,
                res_opt.message,
                interpretation,
                "=" * 80,
            )

            # Estimate errors using inverse Hessian (if available)
            errors_global = np.zeros(len(res_opt.x))
            if hasattr(res_opt, "hess_inv") and res_opt.hess_inv is not None:
                try:
                    if hasattr(res_opt.hess_inv, "todense"):
                        h_inv = res_opt.hess_inv.todense()
                    else:
                        h_inv = np.array(res_opt.hess_inv)
                    dof = max(1, Z_exp_roi.size - len(res_opt.x))
                    mse = res_opt.fun / dof
                    errors_global = np.sqrt(np.maximum(0.0, np.diag(h_inv) * mse))
                except Exception:
                    logger.debug(
                        "Could not estimate parameter errors from the inverse Hessian; "
                        "reporting zeros.", exc_info=True
                    )

            # Reconstruct single delay result records for downstream code compatibility
            for idx_in, t2_in in enumerate(fit_delays):
                bands_opt, decays_opt = unpack_delay_params(res_opt.x, idx_in)

                # Format parameters in legacy/standard array shape: [w01, dw, a_gsb, a_esa, d1, tau, dT2]
                row_params = []
                row_errors = []

                # 1. Band params
                idx_ptr = 2 * self.num_bands + 3 * self.num_decays + 2 * self.num_bands * idx_in
                for i in range(self.num_bands):
                    row_params.extend([res_opt.x[2 * i], res_opt.x[2 * i + 1], res_opt.x[idx_ptr], res_opt.x[idx_ptr + 1]])
                    row_errors.extend([errors_global[2 * i], errors_global[2 * i + 1], errors_global[idx_ptr], errors_global[idx_ptr + 1]])
                    idx_ptr += 2

                # 2. Decay params
                linked_decays_start = 2 * self.num_bands
                for j in range(self.num_decays):
                    row_params.extend([res_opt.x[linked_decays_start + 3 * j], res_opt.x[linked_decays_start + 3 * j + 1], res_opt.x[linked_decays_start + 3 * j + 2]])
                    row_errors.extend([errors_global[linked_decays_start + 3 * j], errors_global[linked_decays_start + 3 * j + 1], errors_global[linked_decays_start + 3 * j + 2]])

                # Sim slice
                Z_sim_slice = simulate_2d_spectrum(
                    bands_opt, decays_opt, self.pump_axis, self.probe_axis, t2_in,
                    tmax=tmax, nt=nt, w_rot=w_rot,
                    undersampling_factor=u, fid_data=actual_fid
                )
                # Since simulated amplitudes are optimised relative to Z_exp_norm,
                # we just need to scale it by exp_max to restore it back to the original experimental scale.
                Z_sim_restored = Z_sim_slice * np.max(np.abs(Z_exp_roi[:, :, idx_in]))

                fit_results.append({
                    "delay": t2_in,
                    "params": np.array(row_params),
                    "errors": np.array(row_errors),
                    "success": res_opt.success,
                    "cost": res_opt.fun,
                    "Z_sim": Z_sim_restored,
                    "Z_exp": Z_exp_roi[:, :, idx_in],
                    "global_fit": True
                })
        else:
            try:
                if parallel_mode in ("threadpool", "processpool") and len(fit_delays) > 1:
                    pool_class = concurrent.futures.ThreadPoolExecutor if parallel_mode == "threadpool" else concurrent.futures.ProcessPoolExecutor
                    with pool_class() as executor:
                        futures = {executor.submit(fit_single_delay_worker, t): idx for idx, t in enumerate(tasks)}
                        results_map = {}

                        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                            if progress.wasCanceled():
                                for fut in futures:
                                    fut.cancel()
                                raise KeyboardInterrupt("Fitting cancelled by user.")

                            orig_idx = futures[future]
                            results_map[orig_idx] = future.result()
                            progress.setValue(idx + 1)
                            QCoreApplication.processEvents()

                        fit_results = [results_map[i] for i in range(len(fit_delays))]
                else:
                    for idx, t in enumerate(tasks):
                        if progress.wasCanceled():
                            raise KeyboardInterrupt("Fitting cancelled by user.")

                        self.fit_history = []  # reset path per delay
                        res = fit_single_delay_worker(t)
                        fit_results.append(res)
                        progress.setValue(idx + 1)
                        QCoreApplication.processEvents()
            except KeyboardInterrupt as exc:
                progress.close()
                QMessageBox.warning(self, "Fitting Cancelled", str(exc))
                return
            except Exception as exc:
                progress.close()
                QMessageBox.critical(self, "Fitting Failed", str(exc))
                return

        progress.setValue(len(fit_delays))

        if fit_results:
            self.fit_history = fit_results[-1].get("history", [])
            self.last_fit_params = fit_results[-1]["params"]

        # Store results on the dataset
        if not hasattr(self.dataset, "_sd_results"):
            self.dataset._sd_results = {}

        rows = []
        for res in fit_results:
            r = [res["delay"]]
            r.extend(list(res["params"]))
            rows.append(r)

        self.dataset._sd_results["Kubo_Fit"] = np.array(rows)

        if not hasattr(self.dataset, "_sd_kubo_details"):
            self.dataset._sd_kubo_details = {}

        for res in fit_results:
            self.dataset._sd_kubo_details[res["delay"]] = {
                "params": res["params"],
                "errors": res["errors"],
                "num_bands": self.num_bands,
                "num_decays": self.num_decays,
                "param_names": [self.param_table.item(r, 0).text() for r in range(self.param_table.rowCount())],
                "tmax": tmax,
                "nt": nt,
                "u": u,
                "w_rot": w_rot,
                "fid_data": self.fid_data
            }

        # Construct in-memory simulated dataset
        n_pump = len(self.dataset.pump)
        n_probe = len(self.dataset.probe)
        Z_sim_cube = np.zeros((n_pump, n_probe, len(self.dataset.delays)))

        for idx, t2 in enumerate(self.dataset.delays):
            if t2 in self.dataset._sd_kubo_details:
                details = self.dataset._sd_kubo_details[t2]
                params = details["params"]
                num_bands = details["num_bands"]
                num_decays = details["num_decays"]
                tmax = details["tmax"]
                nt = details["nt"]
                u = details["u"]
                w_rot = details["w_rot"]
                fid_data = details["fid_data"]

                bands = []
                for i in range(num_bands):
                    if len(params) >= 4 * num_bands:
                        bands.append((params[4*i], params[4*i+1], params[4*i+2], params[4*i+3]))
                    else:
                        bands.append((params[2*i], params[2*i+1], 1.0, 1.0))
                decays = []
                for j in range(num_decays):
                    if len(params) >= 4 * num_bands:
                        offset = 4 * num_bands + 3 * j
                    else:
                        offset = 2 * num_bands + 3 * j
                    decays.append((params[offset], params[offset+1], params[offset+2]))

                # Scale using fine ROI grids matching the mapped delay index
                mapped_idx = self.delay_indices.index(idx) if idx in self.delay_indices else 0
                Z_exp_roi = self.Z_exp_roi[:, :, mapped_idx]
                exp_max = np.max(np.abs(Z_exp_roi))

                Z_sim_full = simulate_2d_spectrum(
                    bands, decays, self.dataset.pump, self.dataset.probe, t2,
                    tmax=tmax, nt=nt, w_rot=w_rot,
                    undersampling_factor=u, fid_data=fid_data
                )
                Z_sim_cube[:, :, idx] = Z_sim_full * exp_max

        from ..twoD.dataset import Dataset2D
        self.dataset._kubo_fit_dataset = Dataset2D(
            Z_sim_cube, self.dataset.pump, self.dataset.probe, self.dataset.delays,
            self.dataset.units, self.dataset.freq_units,
            source="In-Memory Kubo Simulation", data_type="P2DAT", datatype="processed"
        )

        if self.chk_auto_update_p0.isChecked():
            self.pfit_to_p0(show_dialog=False)

        QMessageBox.information(
            self, "Fit Complete",
            f"Fit across all {len(fit_delays)} delays completed successfully!\n\n"
            f"Method Reference:\n"
            f"Robben, K. C.; Cheatum, C. M. Least-Squares Fitting of Multidimensional Spectra to Kubo Line-Shape Models. J. Phys. Chem. B 2021, 125, 46, 12876-12891.\n"
            f"DOI: 10.1021/acs.jpcb.1c08764"
        )
        self.update_plots()

    def save_fit_results(self):
        if not hasattr(self.dataset, "_sd_kubo_details") or not self.dataset._sd_kubo_details:
            QMessageBox.warning(self, "No Fit Results", "No fit results are available to save. Please run a fit first.")
            return

        import datetime
        import os
        from pathlib import Path

        from PyQt6.QtCore import QCoreApplication, Qt
        from PyQt6.QtWidgets import QProgressDialog

        default_dir = ""
        default_base = "kubo_fit"
        if hasattr(self.dataset, "source") and self.dataset.source:
            source_path = Path(self.dataset.source)
            default_dir = str(source_path.parent)
            default_base = source_path.stem
        elif self.parent() and hasattr(self.parent(), "_rootdir_text"):
            default_dir = self.parent()._rootdir_text() or ""

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"{default_base}_fit_{timestamp}.p2dat"
        default_path = os.path.join(default_dir, default_name)

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Fit Results", default_path, "P2DAT (*.p2dat);;All files (*)"
        )
        if not path:
            return

        progress = QProgressDialog("Simulating full fit spectrum for saving...", "Cancel", 0, len(self.dataset.delays), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        QCoreApplication.processEvents()

        n_pump = len(self.dataset.pump)
        n_probe = len(self.dataset.probe)
        Z_sim_cube = np.zeros((n_pump, n_probe, len(self.dataset.delays)))

        try:
            for idx, t2 in enumerate(self.dataset.delays):
                if progress.wasCanceled():
                    raise KeyboardInterrupt("Save simulation cancelled by user.")

                if t2 in self.dataset._sd_kubo_details:
                    details = self.dataset._sd_kubo_details[t2]
                    params = details["params"]
                    num_bands = details["num_bands"]
                    num_decays = details["num_decays"]
                    tmax = details["tmax"]
                    nt = details["nt"]
                    u = details["u"]
                    w_rot = details["w_rot"]
                    fid_data = details["fid_data"]

                    bands = []
                    for i in range(num_bands):
                        if len(params) >= 4 * num_bands:
                            bands.append((params[4*i], params[4*i+1], params[4*i+2], params[4*i+3]))
                        else:
                            bands.append((params[2*i], params[2*i+1], 1.0, 1.0))
                    decays = []
                    for j in range(num_decays):
                        if len(params) >= 4 * num_bands:
                            offset = 4 * num_bands + 3 * j
                        else:
                            offset = 2 * num_bands + 3 * j
                        decays.append((params[offset], params[offset+1], params[offset+2]))

                    # Scale using fine ROI grids
                    Z_exp_roi = self.Z_exp_roi[:, :, idx]
                    exp_max = np.max(np.abs(Z_exp_roi))
                    Z_exp_roi_norm = Z_exp_roi / exp_max if exp_max > 0 else Z_exp_roi

                    # ROI Simulation for scaling factor
                    Z_sim_roi = simulate_2d_spectrum(
                        bands, decays, self.pump_axis, self.probe_axis, t2,
                        tmax=tmax, nt=nt, w_rot=w_rot,
                        undersampling_factor=u, fid_data=fid_data
                    )
                    sim_roi_max = np.max(np.abs(Z_sim_roi))
                    if sim_roi_max > 0:
                        Z_sim_roi = Z_sim_roi / sim_roi_max

                    scale = np.sum(Z_exp_roi_norm * Z_sim_roi) / (np.sum(Z_sim_roi**2) + 1e-12)

                    # Full Parent Axes Simulation
                    Z_sim_full = simulate_2d_spectrum(
                        bands, decays, self.dataset.pump, self.dataset.probe, t2,
                        tmax=tmax, nt=nt, w_rot=w_rot,
                        undersampling_factor=u, fid_data=fid_data
                    )
                    sim_full_max = np.max(np.abs(Z_sim_full))
                    if sim_full_max > 0:
                        Z_sim_full = Z_sim_full / sim_full_max

                    Z_sim_cube[:, :, idx] = scale * Z_sim_full * exp_max
                else:
                    # Write zeros for un-fitted/excluded delays
                    Z_sim_cube[:, :, idx] = np.zeros((n_pump, n_probe))

                progress.setValue(idx + 1)
                QCoreApplication.processEvents()
        except Exception as exc:
            progress.close()
            QMessageBox.critical(self, "Simulation Failed", f"Could not simulate full axes: {exc}")
            return

        progress.setValue(len(self.dataset.delays))

        try:
            # Write P2DAT Simulated File
            PUMP, PROBE = np.meshgrid(self.dataset.pump, self.dataset.probe, indexing="ij")
            pump_flat = PUMP.ravel(order="F")
            probe_flat = PROBE.ravel(order="F")

            n_rows = len(pump_flat) + 1
            n_cols = 2 + len(self.dataset.delays)

            out = np.zeros((n_rows, n_cols))
            out[0, 0] = 0.0
            out[0, 1] = 0.0
            out[0, 2:] = self.dataset.delays
            out[1:, 0] = pump_flat
            out[1:, 1] = probe_flat

            for i in range(len(self.dataset.delays)):
                out[1:, 2 + i] = Z_sim_cube[:, :, i].ravel(order="F")

            np.savetxt(path, out, delimiter=",", fmt="%.18g")

            # Write companion parameter txt file
            txt_path = os.path.splitext(path)[0] + ".txt"
            is_global = self.chk_global_fit.isChecked()

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("Kubo Model Fitting Results & Parameter Bounds\n")
                f.write("============================================\n")
                f.write(f"Source Dataset: {self.dataset.source}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Fit Type: {'Global Linked (Amplitudes Free)' if is_global else 'Single Delay Optimisation'}\n\n")

                first_delay = list(self.dataset._sd_kubo_details.keys())[0]
                param_names = self.dataset._sd_kubo_details[first_delay]["param_names"]

                if is_global:
                    # Collect linked parameters vs amplitude parameter names
                    linked_names = []
                    amp_names = []
                    for name in param_names:
                        if "GSB" in name or "ESA" in name:
                            amp_names.append(name)
                        else:
                            linked_names.append(name)

                    f.write("I. GLOBAL LINKED PARAMETERS\n")
                    f.write("--------------------------------------------\n")
                    f.write(f"{'Parameter':<30} | {'Value':<12} | {'Std Error':<12}\n")
                    f.write("-" * 60 + "\n")

                    first_details = self.dataset._sd_kubo_details[first_delay]
                    for name, val, err in zip(first_details["param_names"], first_details["params"], first_details["errors"], strict=True):
                        if name in linked_names:
                            f.write(f"{name:<30} | {val:<12.5f} | {err:<12.5f}\n")
                    f.write("\n")

                    f.write("II. PER-DELAY FREE AMPLITUDES\n")
                    f.write("--------------------------------------------\n")
                    header_line = f"{'t2 (ps)':<10}"
                    for name in amp_names:
                        header_line += f" | {name:<12} (Err)"
                    f.write(header_line + "\n")
                    f.write("-" * len(header_line) + "\n")

                    for t2, details in sorted(self.dataset._sd_kubo_details.items()):
                        line_str = f"{t2:<10.2f}"
                        for name, val, err in zip(details["param_names"], details["params"], details["errors"], strict=True):
                            if name in amp_names:
                                line_str += f" | {val:<12.5f} ({err:.3f})"
                        f.write(line_str + "\n")
                    f.write("\n")

                    # LaTeX representation for Global Fitting
                    f.write("LaTeX Table Code (for Copy-Paste in SI/paper):\n")
                    f.write("============================================\n")
                    f.write("\\begin{table}[htbp]\n")
                    f.write("  \\centring\n")
                    f.write("  \\caption{Globally fitted linked Kubo model parameters}\n")
                    f.write("  \\begin{tabular}{ccc}\n")
                    f.write("    \\toprule\n")
                    f.write("    Parameter & Value & Error \\\\\n")
                    f.write("    \\midrule\n")
                    for name, val, err in zip(first_details["param_names"], first_details["params"], first_details["errors"], strict=True):
                        if name in linked_names:
                            latex_n = name.replace("_", "\\_").replace("ω", "$\\omega$").replace("τ", "$\\tau$").replace("Δ", "$\\Delta$").replace("₁", "$_1$").replace("₀", "$_0$").replace("⁻¹", "$^{-1}$").replace("¹", "$^1$")
                            f.write(f"    {latex_n} & {val:.4f} & {err:.4f} \\\\\n")
                    f.write("    \\bottomrule\n")
                    f.write("  \\end{tabular}\n")
                    f.write("\\end{table}\n\n")

                    f.write("\\begin{table}[htbp]\n")
                    f.write("  \\centring\n")
                    f.write("  \\caption{Per-delay free GSB and ESA amplitudes}\n")
                    cols_layout = "c" * (1 + 2 * len(amp_names))
                    f.write(f"  \\begin{{tabular}}{{{cols_layout}}}\n")
                    f.write("    \\toprule\n")

                    latex_amp_names = [n.replace("_", "\\_").replace("ω", "$\\omega$").replace("τ", "$\\tau$").replace("Δ", "$\\Delta$").replace("₁", "$_1$").replace("₀", "$_0$").replace("⁻¹", "$^{-1}$").replace("¹", "$^1$") for n in amp_names]
                    header_row1 = "    $t_2$ (ps)"
                    for name in latex_amp_names:
                        header_row1 += f" & \\multicolumn{{2}}{{c}}{{{name}}}"
                    header_row1 += " \\\\\n"
                    f.write(header_row1)

                    header_row2 = "              "
                    for _ in latex_amp_names:
                        header_row2 += " & Value & Error"
                    header_row2 += " \\\\\n"
                    f.write(header_row2)
                    f.write("    \\midrule\n")

                    for t2, details in sorted(self.dataset._sd_kubo_details.items()):
                        row_str = f"    {t2:.2f}"
                        for name, val, err in zip(details["param_names"], details["params"], details["errors"], strict=True):
                            if name in amp_names:
                                row_str += f" & {val:.4f} & {err:.4f}"
                        row_str += " \\\\\n"
                        f.write(row_str)
                    f.write("    \\bottomrule\n")
                    f.write("  \\end{tabular}\n")
                    f.write("\\end{table}\n")
                else:
                    # Single-delay formatted table
                    for t2, details in sorted(self.dataset._sd_kubo_details.items()):
                        f.write(f"Population Delay (t2): {t2} ps\n")
                        f.write("--------------------------------------------\n")
                        f.write(f"{'Parameter':<30} | {'Value':<12} | {'Std Error':<12}\n")
                        f.write("-" * 60 + "\n")
                        params = details["params"]
                        errors = details["errors"]
                        names = details["param_names"]
                        for n, val, err in zip(names, params, errors, strict=True):
                            f.write(f"{n:<30} | {val:<12.5f} | {err:<12.5f}\n")
                        f.write("\n")

                    f.write("\nLaTeX Table Code (for Copy-Paste in SI/paper):\n")
                    f.write("============================================\n")
                    f.write("\\begin{table}[htbp]\n")
                    f.write("  \\centring\n")
                    f.write("  \\caption{Fitted Kubo model parameters and standard errors}\n")
                    latex_names = [n.replace("_", "\\_").replace("ω", "$\\omega$").replace("τ", "$\\tau$").replace("Δ", "$\\Delta$").replace("₁", "$_1$").replace("₀", "$_0$").replace("⁻¹", "$^{-1}$").replace("¹", "$^1$") for n in param_names]

                    cols_layout = "c" * (1 + 2 * len(param_names))
                    f.write(f"  \\begin{{tabular}}{{{cols_layout}}}\n")
                    f.write("    \\toprule\n")

                    header_row1 = "    $t_2$ (ps)"
                    for name in latex_names:
                        header_row1 += f" & \\multicolumn{{2}}{{c}}{{{name}}}"
                    header_row1 += " \\\\\n"
                    f.write(header_row1)

                    header_row2 = "              "
                    for _ in latex_names:
                        header_row2 += " & Value & Error"
                    header_row2 += " \\\\\n"
                    f.write(header_row2)
                    f.write("    \\midrule\n")

                    for t2, details in sorted(self.dataset._sd_kubo_details.items()):
                        row_str = f"    {t2:.2f}"
                        for val, err in zip(details["params"], details["errors"], strict=True):
                            row_str += f" & {val:.4f} & {err:.4f}"
                        row_str += " \\\\\n"
                        f.write(row_str)
                    f.write("    \\bottomrule\n")
                    f.write("  \\end{tabular}\n")
                    f.write("\\end{table}\n")

            QMessageBox.information(
                self, "Save Complete",
                f"Fit results successfully saved!\n\n"
                f"Simulated P2DAT: {path}\n"
                f"Parameters & Errors TXT: {txt_path}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", f"Failed to write output files: {exc}")

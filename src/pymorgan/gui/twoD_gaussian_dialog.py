"""Dialog for interactive click-to-pick 2D Gaussian spectral fitting with tabs and peak linking."""

from __future__ import annotations

import copy
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
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
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


class CrossPeakGeneratorDialog(QDialog):
    """Modal dialog to configure and generate coupled or population-transfer cross-peaks."""

    def __init__(self, parent, modes: list[dict]):
        super().__init__(parent)
        self.modes = modes
        self.setWindowTitle("Generate 2D Cross-Peaks")
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)

        lbl_desc = QLabel("Select two diagonal modes to generate linked cross-peak pairs:")
        lbl_desc.setWordWrap(True)
        layout.addWidget(lbl_desc)

        # Mode A and Mode B selectors
        form_layout = QVBoxLayout()

        row_a = QHBoxLayout()
        row_a.addWidget(QLabel("Diagonal Mode A:"))
        self.cb_mode_a = QComboBox()
        row_a.addWidget(self.cb_mode_a)
        form_layout.addLayout(row_a)

        row_b = QHBoxLayout()
        row_b.addWidget(QLabel("Diagonal Mode B:"))
        self.cb_mode_b = QComboBox()
        row_b.addWidget(self.cb_mode_b)
        form_layout.addLayout(row_b)

        for i, m in enumerate(modes):
            label = f"#{i + 1} (ω1={m.get('w1', 0):.1f}, ω3={m.get('w3', 0):.1f}, Δ={m.get('anharm', 15):.1f})"
            self.cb_mode_a.addItem(label, i)
            self.cb_mode_b.addItem(label, i)

        if len(modes) >= 2:
            self.cb_mode_b.setCurrentIndex(1)

        layout.addLayout(form_layout)

        # Mechanism Selection
        grp_mech = QGroupBox("Cross-Peak Type & Anharmonicity Linking")
        mech_layout = QVBoxLayout(grp_mech)

        self.rb_coupling = QRadioButton("Vibrational Coupling (Independent Cross-Anharm Δ_AB)")
        self.rb_coupling.setChecked(True)
        self.rb_coupling.setToolTip(
            "Cross-peak excitation and detection are linked to Mode A & B frequencies, but cross-anharmonicity Δ remains a free parameter."
        )
        mech_layout.addWidget(self.rb_coupling)

        self.rb_exchange = QRadioButton("Population Transfer / Exchange (Δ linked to destination mode)")
        self.rb_exchange.setToolTip(
            "Cross-peak A→B has Δ linked to Mode B, and B→A has Δ linked to Mode A."
        )
        mech_layout.addWidget(self.rb_exchange)
        layout.addWidget(grp_mech)

        # Amplitude ratio & bidirectional option
        opt_layout = QHBoxLayout()
        opt_layout.addWidget(QLabel("Initial Cross-Peak Amp Ratio:"))
        self.spn_ratio = QDoubleSpinBox()
        self.spn_ratio.setRange(0.01, 2.0)
        self.spn_ratio.setSingleStep(0.05)
        self.spn_ratio.setValue(0.3)
        self.spn_ratio.setToolTip("Initial amplitude of cross-peaks relative to the diagonal parent modes.")
        opt_layout.addWidget(self.spn_ratio)
        layout.addLayout(opt_layout)

        self.chk_both = QCheckBox("Generate both cross-peaks (A→B and B→A)")
        self.chk_both.setChecked(True)
        layout.addWidget(self.chk_both)

        # Action buttons
        btn_layout = QHBoxLayout()
        self.btn_gen = QPushButton("Generate Cross-Peaks")
        self.btn_gen.setStyleSheet("font-weight: bold; background-color: #2563eb; color: white;")
        self.btn_gen.clicked.connect(self.accept)
        btn_layout.addWidget(self.btn_gen)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)


class TwoDGaussianFitDialog(QDialog):
    """Interactive dialog for 2D Gaussian/Lorentzian spectral fitting with click-to-pick and peak linking."""

    def __init__(self, parent, dataset: Dataset2D):
        super().__init__(parent)
        self.dataset = dataset
        self.modes: list[dict] = []
        self.fitted_modes: list[dict] | None = None
        self.fit_map: np.ndarray | None = None
        self.residual_map: np.ndarray | None = None
        self._is_populating_table = False

        self.setWindowTitle("2D Spectral Peak Fitting")

        screen = QApplication.primaryScreen()
        screen_w = screen.availableGeometry().width() if screen else 1900
        dlg_w = min(1900, screen_w)
        self.resize(dlg_w, 750)

        self._init_ui()
        self._update_plot()

    def _init_ui(self):
        layout = QHBoxLayout(self)

        # Left panel: Tab Widget + Bottom Action Buttons
        controls_layout = QVBoxLayout()

        self.tab_widget = QTabWidget()
        self.tab_widget.setMinimumWidth(560)

        # ==========================================
        # Tab 1: Peaks & Modes
        # ==========================================
        self.tab_peaks = QWidget()
        peaks_layout = QVBoxLayout(self.tab_peaks)

        lbl_info = QLabel(
            "<b>Add Peak:</b> Click button below, then click/drag peak on Data plot.<br/>"
            "• Blue circle (o) = GSB | Red cross (x) = ESA"
        )
        lbl_info.setWordWrap(True)
        peaks_layout.addWidget(lbl_info)

        # Add Peak & Cross-Peak Generator Buttons
        btn_peak_bar = QHBoxLayout()
        self.btn_add_peak = QPushButton("Add Peak (Click Plot)")
        self.btn_add_peak.setCheckable(True)
        self.btn_add_peak.setChecked(False)
        self.btn_add_peak.setStyleSheet(
            "QPushButton:checked { background-color: #2563eb; color: white; font-weight: bold; }"
        )
        self.btn_add_peak.toggled.connect(self._on_add_peak_toggled)
        btn_peak_bar.addWidget(self.btn_add_peak)

        self.btn_gen_cross_peaks = QPushButton("Generate Cross-Peaks...")
        self.btn_gen_cross_peaks.setToolTip("Auto-generate linked cross-peak pairs between two diagonal modes")
        self.btn_gen_cross_peaks.clicked.connect(self._open_cross_peak_generator)
        btn_peak_bar.addWidget(self.btn_gen_cross_peaks)
        peaks_layout.addLayout(btn_peak_bar)

        # Mode Table
        # Columns: [#, w1, Link w1, w3, Link w3, Δ, Link Δ, GSB Amp, ESA Amp, σ1/Γ1, σ3/Γ3]
        self.table_modes = QTableWidget(0, 11)
        self._update_table_headers()
        self.table_modes.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        col_widths = [32, 54, 62, 54, 62, 48, 62, 58, 58, 48, 48]
        for col, width in enumerate(col_widths):
            self.table_modes.setColumnWidth(col, width)
        self.table_modes.cellChanged.connect(self._on_table_cell_changed)
        peaks_layout.addWidget(self.table_modes)

        # Mode Action Buttons
        btn_layout_modes = QHBoxLayout()
        self.btn_delete_mode = QPushButton("Delete Selected")
        self.btn_delete_mode.clicked.connect(self._delete_selected_mode)
        btn_layout_modes.addWidget(self.btn_delete_mode)

        self.btn_clear_modes = QPushButton("Clear All")
        self.btn_clear_modes.clicked.connect(self._clear_modes)
        btn_layout_modes.addWidget(self.btn_clear_modes)
        peaks_layout.addLayout(btn_layout_modes)

        self.btn_fit_to_p0 = QPushButton("Use Fit as Initial (pFit → p₀)")
        self.btn_fit_to_p0.setStyleSheet("font-weight: bold; background-color: #2563eb; color: white;")
        self.btn_fit_to_p0.clicked.connect(self._use_fit_as_initial)
        peaks_layout.addWidget(self.btn_fit_to_p0)

        self.tab_widget.addTab(self.tab_peaks, "Peaks & Modes")

        # ==========================================
        # Tab 2: Fitting Options & Bounds
        # ==========================================
        self.tab_options = QWidget()
        options_layout = QVBoxLayout(self.tab_options)

        grp_general = QGroupBox("General Fit Configuration")
        grp_gen_layout = QVBoxLayout(grp_general)

        # Peak Shape (Gaussian / Lorentzian)
        shape_layout = QHBoxLayout()
        shape_layout.addWidget(QLabel("Peak Shape:"))
        self.cb_peak_shape = QComboBox()
        self.cb_peak_shape.addItems(["Gaussian", "Lorentzian"])
        self.cb_peak_shape.currentIndexChanged.connect(self._on_shape_changed)
        shape_layout.addWidget(self.cb_peak_shape)
        grp_gen_layout.addLayout(shape_layout)

        # Correlated
        self.chk_correlated = QCheckBox("Correlated (tilted) 2D Gaussians")
        self.chk_correlated.setChecked(True)
        self.chk_correlated.toggled.connect(self._on_options_changed)
        grp_gen_layout.addWidget(self.chk_correlated)

        # Global fit
        self.chk_global = QCheckBox("Global t₂ fit across all delays")
        self.chk_global.setChecked(False)
        grp_gen_layout.addWidget(self.chk_global)

        # Normalise t2
        self.chk_norm_t2 = QCheckBox("Normalise each t₂ delay by max amplitude")
        self.chk_norm_t2.setToolTip("Rescales each t₂ map by its peak amplitude so weak cross-peaks and long delays contribute equally to the fit")
        self.chk_norm_t2.setChecked(False)
        grp_gen_layout.addWidget(self.chk_norm_t2)

        # Default Anharmonicity guess
        anharm_layout = QHBoxLayout()
        anharm_layout.addWidget(QLabel("Default Anharm (cm⁻¹):"))
        self.spn_default_anharm = QDoubleSpinBox()
        self.spn_default_anharm.setRange(0.5, 200.0)
        self.spn_default_anharm.setValue(15.0)
        anharm_layout.addWidget(self.spn_default_anharm)
        grp_gen_layout.addLayout(anharm_layout)

        # Preview t2 delay combo box
        prev_t2_layout = QHBoxLayout()
        prev_t2_layout.addWidget(QLabel("Preview t₂ delay:"))
        self.cb_t2_delay = QComboBox()
        for idx, t2_val in enumerate(self.dataset.delays):
            self.cb_t2_delay.addItem(f"{idx + 1}: {t2_val:.2f} ps")
        self.cb_t2_delay.currentIndexChanged.connect(self._on_t2_delay_changed)
        prev_t2_layout.addWidget(self.cb_t2_delay)
        grp_gen_layout.addLayout(prev_t2_layout)

        options_layout.addWidget(grp_general)

        # Parameter Bounds Group Box
        bounds_group = QGroupBox("Fit Parameter Bounds")
        bounds_layout = QVBoxLayout(bounds_group)

        pos_bnd_layout = QHBoxLayout()
        pos_bnd_layout.addWidget(QLabel("Pos Tol ± (cm⁻¹):"))
        self.spn_pos_tol = QDoubleSpinBox()
        self.spn_pos_tol.setRange(0.1, 200.0)
        self.spn_pos_tol.setValue(10.0)
        self.spn_pos_tol.setToolTip("Maximum allowed deviation for w1 and w3 peak centres from initial guesses during fitting.")
        pos_bnd_layout.addWidget(self.spn_pos_tol)
        bounds_layout.addLayout(pos_bnd_layout)

        anharm_bnd_layout = QHBoxLayout()
        anharm_bnd_layout.addWidget(QLabel("Anharm Δ:"))
        self.spn_anharm_min = QDoubleSpinBox()
        self.spn_anharm_min.setRange(0.1, 300.0)
        self.spn_anharm_min.setValue(5.0)
        self.spn_anharm_min.setPrefix("Min: ")
        anharm_bnd_layout.addWidget(self.spn_anharm_min)

        self.spn_anharm_max = QDoubleSpinBox()
        self.spn_anharm_max.setRange(0.5, 500.0)
        self.spn_anharm_max.setValue(50.0)
        self.spn_anharm_max.setPrefix("Max: ")
        anharm_bnd_layout.addWidget(self.spn_anharm_max)
        bounds_layout.addLayout(anharm_bnd_layout)

        anharm_tol_layout = QHBoxLayout()
        anharm_tol_layout.addWidget(QLabel("Anharm Tol ±:"))
        self.spn_anharm_tol = QDoubleSpinBox()
        self.spn_anharm_tol.setRange(0.0, 100.0)
        self.spn_anharm_tol.setValue(0.0)
        self.spn_anharm_tol.setSpecialValueText("None")
        self.spn_anharm_tol.setToolTip("Optional tight tolerance around initial anharmonicity guess (set to 'None'/0 to use Min/Max only).")
        anharm_tol_layout.addWidget(self.spn_anharm_tol)
        bounds_layout.addLayout(anharm_tol_layout)

        sigma_bnd_layout = QHBoxLayout()
        self.lbl_sigma = QLabel("Width σ:")
        sigma_bnd_layout.addWidget(self.lbl_sigma)
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

        self.chk_constrain_signs = QCheckBox("Constrain signs (GSB ≤ 0, ESA ≥ 0)")
        self.chk_constrain_signs.setChecked(True)
        self.chk_constrain_signs.setToolTip("Enforce negative amplitude for GSB (bleach) and positive for ESA to prevent unphysical cancellation.")
        bounds_layout.addWidget(self.chk_constrain_signs)

        options_layout.addWidget(bounds_group)
        options_layout.addStretch()

        self.tab_widget.addTab(self.tab_options, "Fitting Options & Bounds")

        controls_layout.addWidget(self.tab_widget)

        # Fixed bottom controls
        btn_action_box = QHBoxLayout()
        self.btn_run_fit = QPushButton("Run 2D Fit")
        self.btn_run_fit.setStyleSheet("font-weight: bold; background-color: #059669; color: white; padding: 7px; font-size: 13px;")
        self.btn_run_fit.clicked.connect(self._run_fit)
        btn_action_box.addWidget(self.btn_run_fit)

        self.btn_save_p2dat = QPushButton("Save Simulated P2DAT...")
        self.btn_save_p2dat.setStyleSheet("font-weight: bold; background-color: #2563eb; color: white; padding: 7px; font-size: 13px;")
        self.btn_save_p2dat.setToolTip("Export the simulated 2D spectra across all delays as a standard P2DAT file.")
        self.btn_save_p2dat.clicked.connect(self._save_simulated_p2dat)
        btn_action_box.addWidget(self.btn_save_p2dat)

        controls_layout.addLayout(btn_action_box)

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)
        controls_layout.addWidget(self.btn_close)

        # Right panel: 3 Matplotlib axes (Data, Fit, Residual)
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

        layout.addLayout(controls_layout, stretch=2)
        layout.addLayout(plot_layout, stretch=5)

    def _open_cross_peak_generator(self):
        if len(self.modes) < 2:
            QMessageBox.warning(
                self,
                "Need at least 2 Modes",
                "Please add at least 2 diagonal modes on the plot before generating cross-peaks.",
            )
            return

        dlg = CrossPeakGeneratorDialog(self, self.modes)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        idx_a = dlg.cb_mode_a.currentIndex()
        idx_b = dlg.cb_mode_b.currentIndex()
        if idx_a == idx_b:
            QMessageBox.warning(self, "Invalid Selection", "Mode A and Mode B must be different modes.")
            return

        mode_a = self.modes[idx_a]
        mode_b = self.modes[idx_b]
        is_exchange = dlg.rb_exchange.isChecked()
        ratio = dlg.spn_ratio.value()
        gen_both = dlg.chk_both.isChecked()

        anharm_ab = mode_b.get("anharm", 15.0) if is_exchange else float((mode_a.get("anharm", 15.0) + mode_b.get("anharm", 15.0)) / 2.0)
        link_anharm_ab = idx_b if is_exchange else None

        mode_ab = {
            "w1": float(mode_a.get("w1", 0.0)),
            "w3": float(mode_b.get("w3", 0.0)),
            "anharm": float(anharm_ab),
            "amp_gsb": float(mode_a.get("amp_gsb", -1.0)) * ratio,
            "amp_esa": float(mode_a.get("amp_esa", 0.8)) * ratio,
            "sigma_w1": float(mode_a.get("sigma_w1", 10.0)),
            "sigma_w3": float(mode_b.get("sigma_w3", 10.0)),
            "link_w1": idx_a,
            "link_w3": idx_b,
            "link_anharm": link_anharm_ab,
        }
        self.modes.append(mode_ab)

        if gen_both:
            anharm_ba = mode_a.get("anharm", 15.0) if is_exchange else float((mode_a.get("anharm", 15.0) + mode_b.get("anharm", 15.0)) / 2.0)
            link_anharm_ba = idx_a if is_exchange else None

            mode_ba = {
                "w1": float(mode_b.get("w1", 0.0)),
                "w3": float(mode_a.get("w3", 0.0)),
                "anharm": float(anharm_ba),
                "amp_gsb": float(mode_b.get("amp_gsb", -1.0)) * ratio,
                "amp_esa": float(mode_b.get("amp_esa", 0.8)) * ratio,
                "sigma_w1": float(mode_b.get("sigma_w1", 10.0)),
                "sigma_w3": float(mode_a.get("sigma_w3", 10.0)),
                "link_w1": idx_b,
                "link_w3": idx_a,
                "link_anharm": link_anharm_ba,
            }
            self.modes.append(mode_ba)

        self.fitted_modes = None
        self.fit_map = None
        self.residual_map = None
        self._populate_table()
        self._update_plot()

    def _on_add_peak_toggled(self, checked):
        if checked:
            self.btn_add_peak.setText("Click GSB & Drag to ESA...")
        else:
            self.btn_add_peak.setText("Add Peak (Click Plot)")

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
            "link_w1": None,
            "link_w3": None,
            "link_anharm": None,
        }
        self.modes.append(new_mode)
        self.fitted_modes = None
        self.fit_map = None
        self.residual_map = None
        self.btn_add_peak.setChecked(False)
        self._populate_table()
        self._update_plot()

    def _sync_linked_modes(self):
        """Synchronize slave parameter values with their master mode values."""
        n_modes = len(self.modes)
        for i, m in enumerate(self.modes):
            lw1 = m.get("link_w1")
            if lw1 is not None and isinstance(lw1, int) and 0 <= lw1 < n_modes and lw1 != i:
                m["w1"] = float(self.modes[lw1].get("w1", m["w1"]))

            lw3 = m.get("link_w3")
            if lw3 is not None and isinstance(lw3, int) and 0 <= lw3 < n_modes and lw3 != i:
                m["w3"] = float(self.modes[lw3].get("w3", m["w3"]))

            lanharm = m.get("link_anharm")
            if lanharm is not None and isinstance(lanharm, int) and 0 <= lanharm < n_modes and lanharm != i:
                m["anharm"] = float(self.modes[lanharm].get("anharm", m["anharm"]))

    def _on_table_cell_changed(self, row: int, col: int):
        if self._is_populating_table:
            return
        if row < 0 or row >= len(self.modes):
            return

        item = self.table_modes.item(row, col)
        if item is None:
            return

        try:
            val = float(item.text())
        except ValueError:
            self._populate_table()
            return

        key_map = {
            1: "w1",
            3: "w3",
            5: "anharm",
            7: "amp_gsb",
            8: "amp_esa",
            9: "sigma_w1",
            10: "sigma_w3",
        }
        key = key_map.get(col)
        if key:
            self.modes[row][key] = val
            self._sync_linked_modes()
            self.fitted_modes = None
            self.fit_map = None
            self.residual_map = None
            self._populate_table()
            self._update_plot()

    def _on_link_combo_changed(self, row: int, param_key: str, combo_idx: int):
        if self._is_populating_table:
            return
        if row < 0 or row >= len(self.modes):
            return

        link_key = f"link_{param_key}"
        if combo_idx == 0:
            self.modes[row][link_key] = None
        else:
            master_idx = combo_idx - 1
            if master_idx != row and 0 <= master_idx < len(self.modes):
                self.modes[row][link_key] = master_idx
            else:
                self.modes[row][link_key] = None

        self._sync_linked_modes()
        self.fitted_modes = None
        self.fit_map = None
        self.residual_map = None
        self._populate_table()
        self._update_plot()

    def _use_fit_as_initial(self):
        if self.fitted_modes is None:
            QMessageBox.information(
                self,
                "No Fitted Modes",
                "Run a 2D fit first before copying fitted parameters as initial guesses.",
            )
            return
        self.modes = copy.deepcopy(self.fitted_modes)
        self.fitted_modes = None
        self._populate_table()
        self._update_plot()
        QMessageBox.information(self, "Success", "Fitted mode parameters copied to initial modes.")

    def _populate_table(self):
        self._is_populating_table = True
        try:
            self._sync_linked_modes()
            display_modes = self.fitted_modes if self.fitted_modes is not None else self.modes
            n_modes = len(display_modes)
            self.table_modes.setRowCount(n_modes)

            for row, m in enumerate(display_modes):
                w1_val = m.get("w1", 0.0)
                w3_val = m.get("w3", 0.0)
                anharm_val = m.get("anharm", 15.0)
                amp_g_val = m.get("amp_gsb", -1.0)
                amp_e_val = m.get("amp_esa", 0.8)
                sig1_val = m.get("sigma_w1", 10.0)
                sig3_val = m.get("sigma_w3", 10.0)

                # Col 0: Mode # (read-only)
                item_idx = QTableWidgetItem(f"#{row + 1}")
                item_idx.setFlags(item_idx.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item_idx.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table_modes.setItem(row, 0, item_idx)

                # Col 1: w1
                item_w1 = QTableWidgetItem(f"{w1_val:.1f}")
                if m.get("link_w1") is not None:
                    item_w1.setFlags(item_w1.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    item_w1.setToolTip(f"Linked to Mode #{int(m['link_w1']) + 1}")
                self.table_modes.setItem(row, 1, item_w1)

                # Col 2: Link w1 combo
                cb_link_w1 = QComboBox()
                cb_link_w1.addItem("None")
                for k in range(n_modes):
                    cb_link_w1.addItem(f"#{k + 1}")
                cur_link_w1 = m.get("link_w1")
                if cur_link_w1 is not None and 0 <= int(cur_link_w1) < n_modes and int(cur_link_w1) != row:
                    cb_link_w1.setCurrentIndex(int(cur_link_w1) + 1)
                else:
                    cb_link_w1.setCurrentIndex(0)
                cb_link_w1.currentIndexChanged.connect(
                    lambda idx, r=row: self._on_link_combo_changed(r, "w1", idx)
                )
                self.table_modes.setCellWidget(row, 2, cb_link_w1)

                # Col 3: w3
                item_w3 = QTableWidgetItem(f"{w3_val:.1f}")
                if m.get("link_w3") is not None:
                    item_w3.setFlags(item_w3.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    item_w3.setToolTip(f"Linked to Mode #{int(m['link_w3']) + 1}")
                self.table_modes.setItem(row, 3, item_w3)

                # Col 4: Link w3 combo
                cb_link_w3 = QComboBox()
                cb_link_w3.addItem("None")
                for k in range(n_modes):
                    cb_link_w3.addItem(f"#{k + 1}")
                cur_link_w3 = m.get("link_w3")
                if cur_link_w3 is not None and 0 <= int(cur_link_w3) < n_modes and int(cur_link_w3) != row:
                    cb_link_w3.setCurrentIndex(int(cur_link_w3) + 1)
                else:
                    cb_link_w3.setCurrentIndex(0)
                cb_link_w3.currentIndexChanged.connect(
                    lambda idx, r=row: self._on_link_combo_changed(r, "w3", idx)
                )
                self.table_modes.setCellWidget(row, 4, cb_link_w3)

                # Col 5: Δ
                item_anharm = QTableWidgetItem(f"{anharm_val:.1f}")
                if m.get("link_anharm") is not None:
                    item_anharm.setFlags(item_anharm.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    item_anharm.setToolTip(f"Linked to Mode #{int(m['link_anharm']) + 1}")
                self.table_modes.setItem(row, 5, item_anharm)

                # Col 6: Link Δ combo
                cb_link_anharm = QComboBox()
                cb_link_anharm.addItem("None")
                for k in range(n_modes):
                    cb_link_anharm.addItem(f"#{k + 1}")
                cur_link_anharm = m.get("link_anharm")
                if cur_link_anharm is not None and 0 <= int(cur_link_anharm) < n_modes and int(cur_link_anharm) != row:
                    cb_link_anharm.setCurrentIndex(int(cur_link_anharm) + 1)
                else:
                    cb_link_anharm.setCurrentIndex(0)
                cb_link_anharm.currentIndexChanged.connect(
                    lambda idx, r=row: self._on_link_combo_changed(r, "anharm", idx)
                )
                self.table_modes.setCellWidget(row, 6, cb_link_anharm)

                # Col 7: GSB Amp
                self.table_modes.setItem(row, 7, QTableWidgetItem(f"{amp_g_val:.2f}"))

                # Col 8: ESA Amp
                self.table_modes.setItem(row, 8, QTableWidgetItem(f"{amp_e_val:.2f}"))

                # Col 9: σ1 / Γ1
                self.table_modes.setItem(row, 9, QTableWidgetItem(f"{sig1_val:.1f}"))

                # Col 10: σ3 / Γ3
                self.table_modes.setItem(row, 10, QTableWidgetItem(f"{sig3_val:.1f}"))

        finally:
            self._is_populating_table = False

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

        # Clear invalid link indices
        n_modes = len(self.modes)
        for m in self.modes:
            for lk in ("link_w1", "link_w3", "link_anharm"):
                if m.get(lk) is not None and (int(m[lk]) >= n_modes):
                    m[lk] = None

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

    def _on_shape_changed(self):
        self._update_table_headers()
        is_lorentz = "lorentz" in self.cb_peak_shape.currentText().lower() if hasattr(self, "cb_peak_shape") else False
        if hasattr(self, "chk_correlated"):
            if is_lorentz:
                self.chk_correlated.setChecked(False)
                self.chk_correlated.setEnabled(False)
                self.chk_correlated.setToolTip("Lorentzian peaks are strictly uncorrelated")
            else:
                self.chk_correlated.setEnabled(True)
                self.chk_correlated.setToolTip("Correlated (tilted) 2D Gaussians")
        self.fitted_modes = None
        self.fit_map = None
        self.residual_map = None
        self._update_plot()

    def _update_table_headers(self):
        is_lorentz = "lorentz" in self.cb_peak_shape.currentText().lower() if hasattr(self, "cb_peak_shape") else False
        w_label1 = "Γ1" if is_lorentz else "σ1"
        w_label3 = "Γ3" if is_lorentz else "σ3"
        self.table_modes.setHorizontalHeaderLabels([
            "#", "w1", "Link w1", "w3", "Link w3", "Δ", "Link Δ", "GSB Amp", "ESA Amp", w_label1, w_label3
        ])
        if hasattr(self, "lbl_sigma"):
            self.lbl_sigma.setText("Width Γ:" if is_lorentz else "Width σ:")

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
        peak_shape = self.cb_peak_shape.currentText().lower()

        bounds_config = {
            "pos_tol": self.spn_pos_tol.value(),
            "anharm_min": self.spn_anharm_min.value(),
            "anharm_max": self.spn_anharm_max.value(),
            "anharm_tol": self.spn_anharm_tol.value() if self.spn_anharm_tol.value() > 0 else None,
            "sigma_min": self.spn_sigma_min.value(),
            "sigma_max": self.spn_sigma_max.value(),
            "constrain_signs": self.chk_constrain_signs.isChecked(),
            "peak_shape": peak_shape,
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
                    peak_shape=peak_shape,
                    show_progress=True,
                )
                self.fitted_modes = all_delay_modes[i_t2]
                self._all_delay_modes = all_delay_modes
                self._fit_cube = fit_cube
                self.fit_map = fit_cube[:, :, i_t2]
                self.residual_map = res_cube[:, :, i_t2]
                QMessageBox.information(
                    self,
                    "Global Fit Completed",
                    f"Successfully fitted {len(self.modes)} mode(s) globally across {self.dataset.n_maps} delays.",
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
                    peak_shape=peak_shape,
                    show_progress=True,
                )
                self.fitted_modes = f_modes
                self._all_delay_modes = None
                self._fit_cube = None
                self.fit_map = f_map
                self.residual_map = r_map
                QMessageBox.information(
                    self,
                    "Fit Completed",
                    f"Successfully fitted {len(self.modes)} mode(s) for delay {self.dataset.delays[i_t2]:.2f} ps.",
                )

            self._populate_table()
            self._update_plot()
        except Exception as exc:
            QMessageBox.critical(self, "Fit Failed", f"2D {self.cb_peak_shape.currentText()} fitting error:\n{exc}")

    def _save_simulated_p2dat(self):
        if not self.modes:
            QMessageBox.warning(
                self, "No Modes", "Please add at least one peak mode before exporting a simulated spectrum."
            )
            return

        from pymorgan.twoD.load import write_P2DAT

        shape_name = self.cb_peak_shape.currentText()
        active_shape = shape_name.lower()
        correlated = self.chk_correlated.isChecked()
        is_corr = correlated and ("lorentz" not in active_shape)

        n_delays = len(self.dataset.delays)
        Z_sim = np.zeros((len(self.dataset.pump), len(self.dataset.probe), n_delays))

        if getattr(self, "_all_delay_modes", None) is not None and len(self._all_delay_modes) == n_delays:
            for i_t2, m_t2 in enumerate(self._all_delay_modes):
                Z_sim[:, :, i_t2] = evaluate_2d_gaussian_map(
                    self.dataset.pump,
                    self.dataset.probe,
                    m_t2,
                    correlated=is_corr,
                    peak_shape=active_shape,
                )
        else:
            modes_to_use = self.fitted_modes if self.fitted_modes is not None else self.modes
            sim_single = evaluate_2d_gaussian_map(
                self.dataset.pump,
                self.dataset.probe,
                modes_to_use,
                correlated=is_corr,
                peak_shape=active_shape,
            )
            for i_t2 in range(n_delays):
                Z_sim[:, :, i_t2] = sim_single

        # Prompt for save path
        default_base = os.path.splitext(os.path.basename(self.dataset.source or "dataset"))[0]
        default_name = f"{default_base}_simulated_{active_shape}.p2dat"
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Simulated P2DAT",
            default_name,
            "P2DAT (*.p2dat);;All files (*)",
        )
        if not save_path:
            return

        try:
            write_P2DAT(save_path, self.dataset.pump, self.dataset.probe, self.dataset.delays, Z_sim)

            # Write companion parameter txt file
            txt_path = os.path.splitext(save_path)[0] + "_parameters.txt"
            modes_to_use = self.fitted_modes if self.fitted_modes is not None else self.modes
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"PyMORGAN 2D Spectral Peak Simulation ({shape_name})\n")
                f.write("=" * 80 + "\n")
                f.write(f"Source Dataset: {self.dataset.source}\n")
                f.write(f"Peak Shape: {shape_name}\n")
                f.write(f"Correlated (Tilted): {is_corr}\n")
                f.write(f"Number of Modes: {len(modes_to_use)}\n")
                f.write(f"Number of Delays: {n_delays}\n\n")
                f.write(
                    f"{'Mode':<6} | {'w1 (cm⁻¹)':<12} | {'w3 (cm⁻¹)':<12} | {'Δ (cm⁻¹)':<10} | {'GSB Amp':<10} | {'ESA Amp':<10} | {'Width 1':<10} | {'Width 3':<10} | {'Links'}\n"
                )
                f.write("-" * 105 + "\n")
                for idx_m, m in enumerate(modes_to_use):
                    w1 = m.get("w1", 0.0)
                    w3 = m.get("w3", 0.0)
                    anh = m.get("anharm", 15.0)
                    g = m.get("amp_gsb", -1.0)
                    e = m.get("amp_esa", 0.8)
                    s1 = m.get("sigma_w1", 10.0)
                    s3 = m.get("sigma_w3", 10.0)
                    links = []
                    if m.get("link_w1") is not None:
                        links.append(f"w1→#{int(m['link_w1']) + 1}")
                    if m.get("link_w3") is not None:
                        links.append(f"w3→#{int(m['link_w3']) + 1}")
                    if m.get("link_anharm") is not None:
                        links.append(f"Δ→#{int(m['link_anharm']) + 1}")
                    link_str = ", ".join(links) if links else "None"
                    f.write(
                        f"#{idx_m + 1:<5} | {w1:<12.2f} | {w3:<12.2f} | {anh:<10.2f} | {g:<10.4f} | {e:<10.4f} | {s1:<10.2f} | {s3:<10.2f} | {link_str}\n"
                    )

            QMessageBox.information(
                self,
                "Saved Successfully",
                f"Simulated P2DAT file saved successfully to:\n{save_path}\n\nParameters written to:\n{txt_path}",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", f"Failed to save P2DAT file:\n{exc}")

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
        for idx_m, m in enumerate(active_modes):
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
            self.ax_data.text(
                mx_gsb, my_gsb + 2.0, f"#{idx_m + 1}", color="blue", fontsize=9, fontweight="bold",
                ha="center", va="bottom"
            )

        # 2. Plot Fit (no Y labels/ticks)
        shape_name = self.cb_peak_shape.currentText() if hasattr(self, "cb_peak_shape") else "Gaussian"
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
            self.ax_fit.set_title(f"2D {shape_name} Fit")
        else:
            self.ax_fit.set_title(f"2D {shape_name} Fit (Not run)")
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


TwoDPeakFitDialog = TwoDGaussianFitDialog
PeakFitDialog = TwoDGaussianFitDialog

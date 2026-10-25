"""Interactive 2D-IR Direct Kubo Line-Shape Fitting Dialog.

Implements:
- Multi-band diagonal & off-diagonal (cross-peak) Kubo line-shape modeling
- Parameter linking (ω₁, ω₃, Δ) and Cross-Peak Generator modal
- Shared or per-band Frequency Fluctuation Correlation Functions (FFCF) with
  homogeneous dephasing Γ_hom (dT2), static inhomogeneity Δ₀, and multi-exponential
  Kubo decay terms (Δ_m, τc,m)
- Trust-Region Reflective (TRF) + Variable Projection (VarPro) and multi-core
  parallel execution (ProcessPool / ThreadPool / Parallel Jacobian)
- Analytical 1st-order FTIR Joint Fitting & Empirical FID Envelope modes
- Fit Diagnostics, 95% confidence intervals, and Robben & Cheatum SIGN metrics
- Selectable 4th Diagnostic Panel (Trajectory, 1D Spectral Cuts, FFCF C(t) & Kinetics, FTIR Fit)
"""

from __future__ import annotations

import copy
import datetime
import os
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt6.QtCore import QCoreApplication, Qt
from PyQt6.QtGui import QColor, QGuiApplication
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import pymorgan as pm

from ..log import get_logger
from ..twoD.dataset import Dataset2D
from ..twoD.kubo_fit import (
    c_0,
    evaluate_ffcf_ct,
    run_kubo_fit,
    run_kubo_global_fit,
    simulate_1d_ftir_spectrum,
    simulate_2d_basis_maps,
    simulate_2d_spectrum,
    simulate_kubo_cube_parallel,
)

logger = get_logger(__name__)


class KuboCrossPeakDialog(QDialog):
    """Modal dialog to generate symmetric cross-peak pairs between existing diagonal bands."""

    def __init__(self, parent: QWidget | None, band_labels: list[str]):
        super().__init__(parent)
        self.setWindowTitle("Generate Cross-Peaks")
        self.resize(420, 290)

        layout = QVBoxLayout(self)
        info = QLabel(
            "Generate a symmetric pair of off-diagonal cross-peaks (A → B and B → A) "
            "coupled to two parent diagonal bands."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        form = QFormLayout()

        self.cb_band_a = QComboBox()
        self.cb_band_a.addItems(band_labels)
        if len(band_labels) > 0:
            self.cb_band_a.setCurrentIndex(0)
        form.addRow("Parent Band A:", self.cb_band_a)

        self.cb_band_b = QComboBox()
        self.cb_band_b.addItems(band_labels)
        if len(band_labels) > 1:
            self.cb_band_b.setCurrentIndex(1)
        form.addRow("Parent Band B:", self.cb_band_b)

        self.spin_amp_ratio = QDoubleSpinBox()
        self.spin_amp_ratio.setRange(0.01, 2.0)
        self.spin_amp_ratio.setSingleStep(0.05)
        self.spin_amp_ratio.setDecimals(2)
        self.spin_amp_ratio.setValue(0.35)
        self.spin_amp_ratio.setToolTip(
            "Initial cross-peak amplitude as a fraction of the parent diagonal bands' geometric mean amplitude."
        )
        form.addRow("Initial Amplitude Ratio:", self.spin_amp_ratio)

        self.spin_corr_eta = QDoubleSpinBox()
        self.spin_corr_eta.setRange(-1.0, 1.0)
        self.spin_corr_eta.setSingleStep(0.1)
        self.spin_corr_eta.setDecimals(2)
        self.spin_corr_eta.setValue(0.50)
        self.spin_corr_eta.setToolTip(
            "Cross-correlation coefficient η_AB ∈ [-1, 1] between fluctuating bath frequencies of modes A and B."
        )
        form.addRow("Bath Correlation η_AB:", self.spin_corr_eta)

        self.cb_anharm_mode = QComboBox()
        self.cb_anharm_mode.addItems([
            "Population Transfer (Link Δ to probed mode)",
            "Excitonic Coupling (Free off-diagonal Δ_AB)",
        ])
        form.addRow("Anharmonicity Linking:", self.cb_anharm_mode)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_config(self) -> dict[str, Any]:
        return {
            "idx_a": self.cb_band_a.currentIndex(),
            "idx_b": self.cb_band_b.currentIndex(),
            "amp_ratio": float(self.spin_amp_ratio.value()),
            "corr_eta": float(self.spin_corr_eta.value()),
            "link_anharm_to_probed": self.cb_anharm_mode.currentIndex() == 0,
        }


class KuboDialog(QDialog):
    """Interactive GUI dialog for Direct Kubo Line-Shape Fitting of 2D-IR spectra."""

    def __init__(self, parent, dataset, pump_range=None, probe_range=None, t2_range=None):
        super().__init__(parent)
        self.setWindowTitle("Direct Kubo Line-Shape Fitting — PyMORGAN")
        self.resize(1380, 880)

        self.dataset = dataset
        self.pump_range = pump_range
        self.probe_range = probe_range
        self.t2_range = t2_range

        self.fid_data: dict[str, Any] | None = None
        self.ftir_spectrum: tuple[np.ndarray, np.ndarray] | None = None
        self.fit_history: list[tuple[float, float]] = []
        self.last_fit_params: np.ndarray | None = None
        self.last_fit_result: dict[str, Any] | None = None
        self._updating_tables = False
        self._drag_start: tuple[float, float] | None = None

        # Setup t2 range filtering indices
        if t2_range is not None:
            t2_min, t2_max = min(t2_range), max(t2_range)
            self.delay_indices = [
                i for i, d in enumerate(dataset.delays) if t2_min <= d <= t2_max
            ]
        else:
            self.delay_indices = list(range(len(dataset.delays)))

        if not self.delay_indices:
            self.delay_indices = list(range(len(dataset.delays)))

        # Extract native experimental ROI coordinates & cube
        pump_full = np.asarray(dataset.pump, dtype=float)
        probe_full = np.asarray(dataset.probe, dtype=float)

        if pump_range is not None:
            p_lo, p_hi = min(pump_range), max(pump_range)
            p_mask = (pump_full >= p_lo) & (pump_full <= p_hi)
            if not np.any(p_mask):
                p_mask = np.ones(len(pump_full), dtype=bool)
        else:
            p_mask = np.ones(len(pump_full), dtype=bool)

        if probe_range is not None:
            pr_lo, pr_hi = min(probe_range), max(probe_range)
            pr_mask = (probe_full >= pr_lo) & (probe_full <= pr_hi)
            if not np.any(pr_mask):
                pr_mask = np.ones(len(probe_full), dtype=bool)
        else:
            pr_mask = np.ones(len(probe_full), dtype=bool)

        self.raw_pump_axis = pump_full[p_mask]
        self.raw_probe_axis = probe_full[pr_mask]
        self.raw_Z_roi = np.asarray(dataset.Z[p_mask, :, :][:, pr_mask, :], dtype=float)

        # Default to native 1x experimental ROI grid (eliminates 16x slowdown and 4x error underestimation)
        self.grid_factor = 1
        self.pump_axis = self.raw_pump_axis.copy()
        self.probe_axis = self.raw_probe_axis.copy()
        self.Z_exp_roi = self.raw_Z_roi.copy()

        # Structured model state
        # Each band dict: {w1, w3, anharm, amp_gsb, amp_esa, link_w1, link_w3, link_anharm, ffcf_id, corr_eta, fixed}
        self._bands_state: list[dict[str, Any]] = []
        # Each FFCF group dict: {
        #   name: str,
        #   dw_T2: (val, lb, ub, fixed),
        #   dw_0:  (val, lb, ub, fixed),
        #   terms: [ (dw_val, dw_lb, dw_ub, dw_fix, tau_val, tau_lb, tau_ub, tau_fix), ... ]
        # }
        self._ffcf_groups: list[dict[str, Any]] = [
            {
                "name": "FFCF #1 (Shared)",
                "dw_T2": [1.5, 0.0, 60.0, False],
                "dw_0": [0.0, 0.0, 100.0, True],
                "terms": [],
            }
        ]
        self._active_ffcf_idx = 0
        self.num_bands = 0
        self.num_decays = 0

        self.init_ui()
        self.update_plots()

    # ------------------------------------------------------------------ #
    #                     Grid Resolution Management                     #
    # ------------------------------------------------------------------ #

    def _apply_grid_resolution(self, factor: int = 1):
        """Set active ROI grid to `1×` (native) or `2×`/`4×` interpolated."""
        factor = max(1, int(factor))
        self.grid_factor = factor
        if factor == 1 or len(self.raw_pump_axis) < 2 or len(self.raw_probe_axis) < 2:
            self.pump_axis = self.raw_pump_axis.copy()
            self.probe_axis = self.raw_probe_axis.copy()
            self.Z_exp_roi = self.raw_Z_roi.copy()
            return

        from scipy.interpolate import RegularGridInterpolator

        pump_fine = np.linspace(
            self.raw_pump_axis[0],
            self.raw_pump_axis[-1],
            len(self.raw_pump_axis) * factor,
        )
        probe_fine = np.linspace(
            self.raw_probe_axis[0],
            self.raw_probe_axis[-1],
            len(self.raw_probe_axis) * factor,
        )
        PP_fine, PR_fine = np.meshgrid(pump_fine, probe_fine, indexing="ij")
        pts = np.column_stack([PP_fine.ravel(), PR_fine.ravel()])

        p_indices = np.argsort(self.raw_pump_axis)
        pr_indices = np.argsort(self.raw_probe_axis)
        sorted_pump = self.raw_pump_axis[p_indices]
        sorted_probe = self.raw_probe_axis[pr_indices]

        Z_fine = np.zeros((len(pump_fine), len(probe_fine), self.raw_Z_roi.shape[2]), dtype=float)
        for idx in range(self.raw_Z_roi.shape[2]):
            sorted_slice = self.raw_Z_roi[p_indices, :, idx][:, pr_indices]
            interp = RegularGridInterpolator(
                (sorted_pump, sorted_probe),
                sorted_slice,
                bounds_error=False,
                fill_value=0.0,
            )
            Z_fine[:, :, idx] = interp(pts).reshape(PP_fine.shape)

        self.pump_axis = pump_fine
        self.probe_axis = probe_fine
        self.Z_exp_roi = Z_fine

    def _on_grid_res_changed(self, idx: int):
        factor_map = {0: 1, 1: 2, 2: 4}
        factor = factor_map.get(idx, 1)
        self._apply_grid_resolution(factor)
        self.update_plots()

    # ------------------------------------------------------------------ #
    #                          UI Construction                           #
    # ------------------------------------------------------------------ #

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(10)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        main_layout.addWidget(splitter)

        # ===================== LEFT PANEL: CONTROLS ===================== #
        control_widget = QWidget()
        control_widget.setMinimumWidth(480)
        control_layout = QVBoxLayout(control_widget)
        control_layout.setContentsMargins(2, 2, 2, 2)
        control_layout.setSpacing(6)

        # Top Bar: Population Delay Selector + Prev/Next + Update Preview
        delay_grp = QGroupBox("Population Delay (t₂) & Live Preview")
        delay_bar = QHBoxLayout(delay_grp)
        delay_bar.setContentsMargins(8, 6, 8, 6)

        self.btn_prev_delay = QPushButton("◀")
        self.btn_prev_delay.setFixedWidth(30)
        self.btn_prev_delay.setToolTip("Previous population delay t₂")
        self.btn_prev_delay.clicked.connect(self._step_delay_prev)

        self.delay_cb = QComboBox()
        self.delay_cb.addItems([f"t₂ = {self.dataset.delays[i]:.2f} ps" for i in self.delay_indices])
        self.delay_cb.currentIndexChanged.connect(self._on_delay_changed)

        self.btn_next_delay = QPushButton("▶")
        self.btn_next_delay.setFixedWidth(30)
        self.btn_next_delay.setToolTip("Next population delay t₂")
        self.btn_next_delay.clicked.connect(self._step_delay_next)

        self.btn_update = QPushButton("↻ Update Preview")
        self.btn_update.setToolTip("Recompute and update the 2D Kubo simulation preview")
        self.btn_update.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #2563eb; color: white; padding: 4px 10px; border-radius: 4px; }"
        )
        self.btn_update.clicked.connect(self.update_plots)

        delay_bar.addWidget(self.btn_prev_delay)
        delay_bar.addWidget(self.delay_cb, stretch=1)
        delay_bar.addWidget(self.btn_next_delay)
        delay_bar.addWidget(self.btn_update)
        control_layout.addWidget(delay_grp)

        # 3-Tab Container
        self.tabs = QTabWidget()
        control_layout.addWidget(self.tabs, stretch=1)

        self._build_tab1_model()
        self._build_tab2_solver_ft_ftir()
        self._build_tab3_diagnostics()

        # Legacy hidden/synced param_table for 100% API compatibility
        self.param_table = QTableWidget()
        self.param_table.setColumnCount(5)
        self.param_table.setHorizontalHeaderLabels(["Parameter", "p₀", "LB", "UB", "Fix?"])
        self.param_table.setVisible(False)

        # Add default initial band and Kubo decay
        self.add_band()
        self.add_decay()

        # Bottom Action Footer
        footer_grp = QGroupBox("Fitting Execution && Export")
        footer_layout = QVBoxLayout(footer_grp)
        footer_layout.setContentsMargins(8, 6, 8, 6)
        footer_layout.setSpacing(5)

        chk_row = QHBoxLayout()
        self.chk_global_fit = QCheckBox("Fit globally across t₂")
        self.chk_global_fit.setChecked(False)
        self.chk_global_fit.setToolTip(
            "Global fit: share band positions, anharmonicities, and FFCF parameters across all t₂ delays while projecting per-delay amplitudes via VarPro."
        )

        self.chk_auto_update_fit = QCheckBox("Live plot trace")
        self.chk_auto_update_fit.setChecked(True)

        self.chk_auto_update_p0 = QCheckBox("Auto pFit → p₀")
        self.chk_auto_update_p0.setChecked(True)

        chk_row.addWidget(self.chk_global_fit)
        chk_row.addWidget(self.chk_auto_update_fit)
        chk_row.addWidget(self.chk_auto_update_p0)
        footer_layout.addLayout(chk_row)

        btn_row = QHBoxLayout()
        self.btn_fit_active = QPushButton("▶ Fit Active Delay")
        self.btn_fit_active.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #1d4ed8; color: white; padding: 6px 10px; border-radius: 4px; }"
        )
        self.btn_fit_active.clicked.connect(self.run_active_fit)

        self.btn_fit_all = QPushButton("⚡ Fit All Delays (Multi-Core)")
        self.btn_fit_all.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #059669; color: white; padding: 6px 10px; border-radius: 4px; }"
        )
        self.btn_fit_all.clicked.connect(self.run_all_fit)

        self.btn_save_fit = QPushButton("💾 Save Fit Results...")
        self.btn_save_fit.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #7c3aed; color: white; padding: 6px 10px; border-radius: 4px; }"
        )
        self.btn_save_fit.clicked.connect(self.save_fit_results)

        btn_row.addWidget(self.btn_fit_active)
        btn_row.addWidget(self.btn_fit_all)
        btn_row.addWidget(self.btn_save_fit)
        footer_layout.addLayout(btn_row)

        control_layout.addWidget(footer_grp)
        splitter.addWidget(control_widget)

        # ===================== RIGHT PANEL: 2x2 CANVAS ===================== #
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(2, 2, 2, 2)
        plot_layout.setSpacing(4)

        self.fig = Figure(figsize=(10.5, 8.2))
        ax0 = self.fig.add_subplot(2, 2, 1)
        ax1 = self.fig.add_subplot(2, 2, 2, sharex=ax0, sharey=ax0)
        ax2 = self.fig.add_subplot(2, 2, 3, sharex=ax0, sharey=ax0)
        ax3 = self.fig.add_subplot(2, 2, 4)
        self.axes = [ax0, ax1, ax2, ax3]

        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Canvas Top Controls Bar (Toolbar + Residual Scale + 4th Panel Selector)
        top_canvas_bar = QHBoxLayout()
        top_canvas_bar.addWidget(self.toolbar, stretch=1)

        top_canvas_bar.addWidget(QLabel("Residual Scale:"))
        self.res_scale_cb = QComboBox()
        self.res_scale_cb.addItems(["10× (±0.10)", "5× (±0.20)", "2× (±0.50)", "1× (±1.00)"])
        self.res_scale_cb.setCurrentIndex(0)
        self.res_scale_cb.currentIndexChanged.connect(self.update_plots)
        top_canvas_bar.addWidget(self.res_scale_cb)

        top_canvas_bar.addWidget(QLabel("4th Panel View:"))
        self.diag_view_cb = QComboBox()
        self.diag_view_cb.addItems([
            "1. Optimization Trajectory (Δ₁ vs τc)",
            "2. 1D Spectral Cuts (Diagonal & Probe Slice)",
            "3. FFCF C(t) Decay & t₂ Kinetics",
            "4. 1st-Order FTIR Fit vs. Experiment",
        ])
        self.diag_view_cb.setCurrentIndex(0)
        self.diag_view_cb.currentIndexChanged.connect(self._update_fourth_panel_only)
        top_canvas_bar.addWidget(self.diag_view_cb)

        plot_layout.addLayout(top_canvas_bar)
        plot_layout.addWidget(self.canvas, stretch=1)

        self.canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self.canvas.mpl_connect("button_release_event", self._on_canvas_release)

        splitter.addWidget(plot_widget)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)

    def _build_tab1_model(self):
        """Construct Tab 1: Vibrational Bands & FFCF Model."""
        tab1 = QWidget()
        layout = QVBoxLayout(tab1)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # --- Group 1: Vibrational Bands & Cross-Peaks --- #
        bands_grp = QGroupBox("1. Vibrational Bands && Coupled Cross-Peaks")
        bands_layout = QVBoxLayout(bands_grp)
        bands_layout.setSpacing(5)

        btn_bar = QHBoxLayout()
        self.btn_add_band_plot = QPushButton("🎯 Add Band (Click/Drag Plot)")
        self.btn_add_band_plot.setCheckable(True)
        self.btn_add_band_plot.setChecked(False)
        self.btn_add_band_plot.setStyleSheet(
            "QPushButton:checked { background-color: #2563eb; color: white; font-weight: bold; }"
        )
        self.btn_add_band_plot.toggled.connect(self._on_add_band_toggled)

        self.btn_add_band = QPushButton("+ Add Band")
        self.btn_add_band.clicked.connect(lambda: self.add_band())

        self.btn_cross_peaks = QPushButton("⇄ Generate Cross-Peaks...")
        self.btn_cross_peaks.setToolTip("Automatically generate coupled off-diagonal cross-peaks between two bands")
        self.btn_cross_peaks.clicked.connect(self._open_cross_peak_generator)

        self.btn_del_band = QPushButton("Delete Selected")
        self.btn_del_band.clicked.connect(self.delete_band)

        self.btn_clear_bands = QPushButton("Clear")
        self.btn_clear_bands.clicked.connect(self._clear_extra_bands)

        btn_bar.addWidget(self.btn_add_band_plot)
        btn_bar.addWidget(self.btn_add_band)
        btn_bar.addWidget(self.btn_cross_peaks)
        btn_bar.addWidget(self.btn_del_band)
        btn_bar.addWidget(self.btn_clear_bands)
        bands_layout.addLayout(btn_bar)

        self.bands_table = QTableWidget()
        self.bands_table.setColumnCount(11)
        self.bands_table.setHorizontalHeaderLabels([
            "#",
            "ω₁ (cm⁻¹)",
            "Link ω₁",
            "ω₃ (cm⁻¹)",
            "Link ω₃",
            "Δ (cm⁻¹)",
            "Link Δ",
            "A_GSB",
            "A_ESA",
            "FFCF",
            "Fix?",
        ])
        self.bands_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.bands_table.horizontalHeader().setStretchLastSection(True)
        self.bands_table.setColumnWidth(0, 36)
        self.bands_table.setColumnWidth(1, 74)
        self.bands_table.setColumnWidth(2, 74)
        self.bands_table.setColumnWidth(3, 74)
        self.bands_table.setColumnWidth(4, 92)
        self.bands_table.setColumnWidth(5, 64)
        self.bands_table.setColumnWidth(6, 74)
        self.bands_table.setColumnWidth(7, 62)
        self.bands_table.setColumnWidth(8, 62)
        self.bands_table.setColumnWidth(9, 76)
        self.bands_table.setColumnWidth(10, 42)
        self.bands_table.setAlternatingRowColors(True)
        self.bands_table.itemChanged.connect(self._on_bands_table_item_changed)
        bands_layout.addWidget(self.bands_table)

        layout.addWidget(bands_grp, stretch=5)

        # --- Group 2: FFCF Line-Broadening g(t) --- #
        ffcf_grp = QGroupBox("2. FFCF Line-Broadening Model g(t)  [Γ_hom, Static Δ₀, Kubo Decays]")
        ffcf_layout = QVBoxLayout(ffcf_grp)
        ffcf_layout.setSpacing(5)

        ffcf_top = QHBoxLayout()
        ffcf_top.addWidget(QLabel("Target FFCF:"))
        self.ffcf_group_cb = QComboBox()
        self.ffcf_group_cb.addItem("FFCF #1 (Shared)")
        self.ffcf_group_cb.currentIndexChanged.connect(self._on_ffcf_group_selected)
        ffcf_top.addWidget(self.ffcf_group_cb, stretch=1)

        self.btn_add_ffcf_group = QPushButton("+ New Band FFCF")
        self.btn_add_ffcf_group.setToolTip("Create an independent FFCF bath group that can be assigned to specific bands")
        self.btn_add_ffcf_group.clicked.connect(self._add_new_ffcf_group)
        ffcf_top.addWidget(self.btn_add_ffcf_group)

        self.btn_add_decay = QPushButton("+ Add Kubo Decay")
        self.btn_add_decay.clicked.connect(self.add_decay)
        ffcf_top.addWidget(self.btn_add_decay)

        self.btn_del_decay = QPushButton("− Delete Decay")
        self.btn_del_decay.clicked.connect(self.delete_decay)
        ffcf_top.addWidget(self.btn_del_decay)
        ffcf_layout.addLayout(ffcf_top)

        self.ffcf_table = QTableWidget()
        self.ffcf_table.setColumnCount(5)
        self.ffcf_table.setHorizontalHeaderLabels(["Parameter", "p₀ (Initial)", "Lower Bound (LB)", "Upper Bound (UB)", "Fix?"])
        self.ffcf_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.ffcf_table.setAlternatingRowColors(True)
        self.ffcf_table.itemChanged.connect(self._on_ffcf_table_item_changed)
        ffcf_layout.addWidget(self.ffcf_table)

        self.btn_pfit_to_p0 = QPushButton("↻ Copy Best Fit to Initial Guess (pFit → p₀)")
        self.btn_pfit_to_p0.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #16a34a; color: white; padding: 4px 8px; border-radius: 4px; }"
        )
        self.btn_pfit_to_p0.clicked.connect(lambda: self.pfit_to_p0(show_dialog=True))
        ffcf_layout.addWidget(self.btn_pfit_to_p0)

        layout.addWidget(ffcf_grp, stretch=5)
        self.tabs.addTab(tab1, "1. Bands && FFCF Model")

    def _build_tab2_solver_ft_ftir(self):
        """Construct Tab 2: FT Grid, Solver & Multi-Core Execution, and 1st-Order FTIR."""
        tab2 = QWidget()
        layout = QVBoxLayout(tab2)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # --- Group 1: FT & Grid Resolution --- #
        ft_grp = QGroupBox("Fourier Transform && ROI Grid Settings")
        ft_grid = QGridLayout(ft_grp)

        ft_grid.addWidget(QLabel("t_max (ps):"), 0, 0)
        self.ft_tmax_spin = QDoubleSpinBox()
        self.ft_tmax_spin.setRange(0.5, 500.0)
        self.ft_tmax_spin.setValue(5.0)
        self.ft_tmax_spin.setSingleStep(0.5)
        self.ft_tmax_spin.setToolTip("Maximum coherence time t₁ and t₃ in ps.")
        ft_grid.addWidget(self.ft_tmax_spin, 0, 1)

        ft_grid.addWidget(QLabel("Time Steps (nt):"), 0, 2)
        self.ft_nt_spin = QSpinBox()
        self.ft_nt_spin.setRange(8, 4096)
        self.ft_nt_spin.setValue(64)
        self.ft_nt_spin.setSingleStep(8)
        self.ft_nt_spin.setToolTip("Number of time-domain grid steps along t₁ and t₃.")
        ft_grid.addWidget(self.ft_nt_spin, 0, 3)

        self.ft_auto_check = QCheckBox("Auto nt (Nyquist Limit)")
        self.ft_auto_check.setChecked(False)
        ft_grid.addWidget(self.ft_auto_check, 1, 0, 1, 2)

        ft_grid.addWidget(QLabel("Undersampling U:"), 1, 2)
        self.ft_u_spin = QDoubleSpinBox()
        self.ft_u_spin.setRange(0.1, 20.0)
        self.ft_u_spin.setValue(1.0)
        self.ft_u_spin.setSingleStep(0.1)
        ft_grid.addWidget(self.ft_u_spin, 1, 3)

        ft_grid.addWidget(QLabel("Rotating Frame ω_rot:"), 2, 0)
        self.ft_wrot_spin = QDoubleSpinBox()
        self.ft_wrot_spin.setRange(100.0, 5000.0)
        self.ft_wrot_spin.setValue(1900.0)
        self.ft_wrot_spin.setSingleStep(10.0)
        self.ft_wrot_spin.setEnabled(False)
        ft_grid.addWidget(self.ft_wrot_spin, 2, 1)

        self.ft_wrot_auto_check = QCheckBox("Auto ω_rot (min probe − 150 cm⁻¹)")
        self.ft_wrot_auto_check.setChecked(True)
        self.ft_wrot_auto_check.stateChanged.connect(
            lambda state: self.ft_wrot_spin.setEnabled(state == 0)
        )
        ft_grid.addWidget(self.ft_wrot_auto_check, 2, 2, 1, 2)

        ft_grid.addWidget(QLabel("ROI Fit Grid Resolution:"), 3, 0)
        self.grid_res_cb = QComboBox()
        self.grid_res_cb.addItems([
            "1× (Native Experimental Grid — Recommended)",
            "2× Interpolated Grid",
            "4× Fine Grid (Legacy)",
        ])
        self.grid_res_cb.setCurrentIndex(0)
        self.grid_res_cb.setToolTip(
            "Fitting on the 1× native experimental grid avoids artificial pixel correlation and provides 16× faster evaluation."
        )
        self.grid_res_cb.currentIndexChanged.connect(self._on_grid_res_changed)
        ft_grid.addWidget(self.grid_res_cb, 3, 1, 1, 3)

        layout.addWidget(ft_grp)

        # --- Group 2: Solver, Bounds & Multi-Core Execution --- #
        solver_grp = QGroupBox("Nonlinear Least-Squares Solver && Multi-Core Parallelism")
        solver_grid = QGridLayout(solver_grp)

        solver_grid.addWidget(QLabel("Solver Algorithm:"), 0, 0)
        self.solver_cb = QComboBox()
        self.solver_cb.addItems([
            "Trust-Region Reflective + VarPro (Recommended)",
            "Trust-Region Reflective (Full Parameter)",
            "L-BFGS-B (Scalar Cost Legacy)",
        ])
        self.solver_cb.setCurrentIndex(0)
        self.solver_cb.currentIndexChanged.connect(self._on_solver_changed)
        solver_grid.addWidget(self.solver_cb, 0, 1, 1, 3)

        solver_grid.addWidget(QLabel("Parallel Backend:"), 1, 0)
        self.parallel_cb = QComboBox()
        self.parallel_cb.addItems([
            "ProcessPool (Multi-Process CPU)",
            "ThreadPool (Shared-Memory Multi-Core)",
            "Disabled (Single Core)",
        ])
        s_cfg = pm.get_settings()
        par_default = getattr(s_cfg, "parallel_fitting", "threadpool")
        if hasattr(par_default, "value"):
            par_default = par_default.value
        par_str = str(par_default).lower()
        if "process" in par_str:
            self.parallel_cb.setCurrentIndex(0)
        elif "disable" in par_str or "none" in par_str:
            self.parallel_cb.setCurrentIndex(2)
        else:
            self.parallel_cb.setCurrentIndex(1)
        solver_grid.addWidget(self.parallel_cb, 1, 1)

        solver_grid.addWidget(QLabel("CPU Workers:"), 1, 2)
        self.workers_spin = QSpinBox()
        max_cpu = max(1, os.cpu_count() or 4)
        self.workers_spin.setRange(1, max_cpu * 2)
        self.workers_spin.setValue(max(1, max_cpu - 1) if max_cpu > 1 else 1)
        self.workers_spin.setToolTip(
            "Number of parallel worker processes/threads used across t₂ delays and parallel Jacobian evaluations."
        )
        solver_grid.addWidget(self.workers_spin, 1, 3)

        solver_grid.addWidget(QLabel("Band Pos Tol ± (cm⁻¹):"), 2, 0)
        self.pos_tol_spin = QDoubleSpinBox()
        self.pos_tol_spin.setRange(0.5, 200.0)
        self.pos_tol_spin.setValue(8.0)
        self.pos_tol_spin.setSingleStep(1.0)
        self.pos_tol_spin.setToolTip("Allowed ± excursion around initial ω₁ (and ω₃) during optimization.")
        solver_grid.addWidget(self.pos_tol_spin, 2, 1)

        solver_grid.addWidget(QLabel("Anharm Δ Bounds (cm⁻¹):"), 2, 2)
        anh_box = QHBoxLayout()
        self.anharm_min_spin = QDoubleSpinBox()
        self.anharm_min_spin.setRange(-200.0, 200.0)
        self.anharm_min_spin.setValue(2.0)
        self.anharm_max_spin = QDoubleSpinBox()
        self.anharm_max_spin.setRange(0.5, 300.0)
        self.anharm_max_spin.setValue(60.0)
        anh_box.addWidget(self.anharm_min_spin)
        anh_box.addWidget(QLabel("to"))
        anh_box.addWidget(self.anharm_max_spin)
        solver_grid.addLayout(anh_box, 2, 3)

        self.chk_varpro = QCheckBox(
            "Variable Projection (VarPro): Solve linear pathway amplitudes (A_GSB, A_ESA) analytically per step"
        )
        self.chk_varpro.setChecked(True)
        solver_grid.addWidget(self.chk_varpro, 3, 0, 1, 4)

        self.chk_constrain_signs = QCheckBox(
            "Constrain pathway amplitude signs (A_GSB ≥ 0 → negative bleach, A_ESA ≥ 0 → positive ESA)"
        )
        self.chk_constrain_signs.setChecked(True)
        solver_grid.addWidget(self.chk_constrain_signs, 4, 0, 1, 4)

        layout.addWidget(solver_grp)

        # --- Group 3: 1st-Order FTIR Integration --- #
        ftir_grp = QGroupBox("1st-Order FTIR Spectrum Constraint")
        ftir_layout = QGridLayout(ftir_grp)

        self.btn_load_ftir = QPushButton("📂 Load FTIR Spectrum...")
        self.btn_load_ftir.clicked.connect(self.load_ftir_spectrum)
        ftir_layout.addWidget(self.btn_load_ftir, 0, 0, 1, 2)

        self.btn_clear_ftir = QPushButton("✕ Clear FTIR")
        self.btn_clear_ftir.clicked.connect(self.clear_ftir_spectrum)
        ftir_layout.addWidget(self.btn_clear_ftir, 0, 2, 1, 2)

        ftir_layout.addWidget(QLabel("FTIR Mode:"), 1, 0)
        self.ftir_mode_cb = QComboBox()
        self.ftir_mode_cb.addItems([
            "Joint Fit (Constrain Analytical Kubo FFCF — Recommended)",
            "Empirical FID Envelope (Legacy — Locks t₁/t₃ Decay)",
        ])
        self.ftir_mode_cb.setCurrentIndex(0)
        self.ftir_mode_cb.currentIndexChanged.connect(self._on_ftir_mode_changed)
        ftir_layout.addWidget(self.ftir_mode_cb, 1, 1, 1, 2)

        ftir_layout.addWidget(QLabel("Weight:"), 1, 3)
        self.ftir_weight_spin = QDoubleSpinBox()
        self.ftir_weight_spin.setRange(0.0, 100.0)
        self.ftir_weight_spin.setValue(1.0)
        self.ftir_weight_spin.setSingleStep(0.25)
        ftir_layout.addWidget(self.ftir_weight_spin, 1, 4)

        self.ftir_status = QLabel("No FTIR loaded (using pure analytical Kubo line-shape).")
        self.ftir_status.setWordWrap(True)
        self.ftir_status.setStyleSheet("color: #64748b;")
        ftir_layout.addWidget(self.ftir_status, 2, 0, 1, 5)

        layout.addWidget(ftir_grp)
        layout.addStretch(1)
        self.tabs.addTab(tab2, "2. FT, Solver && FTIR")

    def _build_tab3_diagnostics(self):
        """Construct Tab 3: Fit Statistics, 95% CI & Robben SIGN Diagnostics."""
        tab3 = QWidget()
        layout = QVBoxLayout(tab3)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # Summary KPI row
        kpi_grp = QGroupBox("Fit Quality && Convergence Summary")
        kpi_grid = QGridLayout(kpi_grp)

        self.lbl_r2 = QLabel("R²: —")
        self.lbl_r2.setStyleSheet("font-weight: bold; font-size: 13px; color: #16a34a;")
        self.lbl_rmse = QLabel("RMSE: —")
        self.lbl_rmse.setStyleSheet("font-weight: bold; font-size: 13px; color: #1e293b;")
        self.lbl_chi2 = QLabel("Reduced χ²_ν: —")
        self.lbl_chi2.setStyleSheet("font-weight: bold; font-size: 13px; color: #2563eb;")
        self.lbl_perf = QLabel("Evals / Time: —")
        self.lbl_perf.setStyleSheet("font-weight: bold; font-size: 12px; color: #475569;")

        kpi_grid.addWidget(self.lbl_r2, 0, 0)
        kpi_grid.addWidget(self.lbl_rmse, 0, 1)
        kpi_grid.addWidget(self.lbl_chi2, 1, 0)
        kpi_grid.addWidget(self.lbl_perf, 1, 1)
        layout.addWidget(kpi_grp)

        # Diagnostics Table
        diag_grp = QGroupBox("Parameter Uncertainties (±σ, 95% CI) && Robben-Cheatum SIGN Metric")
        diag_layout = QVBoxLayout(diag_grp)

        self.diag_table = QTableWidget()
        self.diag_table.setColumnCount(6)
        self.diag_table.setHorizontalHeaderLabels([
            "Parameter",
            "Fitted Value",
            "± Std Err (σ)",
            "95% CI",
            "Robben SIGN",
            "Identifiability",
        ])
        self.diag_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.diag_table.setAlternatingRowColors(True)
        diag_layout.addWidget(self.diag_table)

        note_lbl = QLabel(
            "Robben & Cheatum SIGN metric (J. Phys. Chem. B 2021, 125, 12876): "
            "SIGN > −1 indicates well-determined parameters; SIGN < −2 warns of strong parameter collinearity (e.g. Δ₁·τc product in fast modulation)."
        )
        note_lbl.setWordWrap(True)
        note_lbl.setStyleSheet("color: #64748b; font-size: 11px;")
        diag_layout.addWidget(note_lbl)

        exp_row = QHBoxLayout()
        self.btn_copy_latex = QPushButton("📋 Copy LaTeX Table")
        self.btn_copy_latex.clicked.connect(self._copy_diagnostics_latex)
        self.btn_copy_csv = QPushButton("📋 Copy CSV")
        self.btn_copy_csv.clicked.connect(self._copy_diagnostics_csv)
        exp_row.addWidget(self.btn_copy_latex)
        exp_row.addWidget(self.btn_copy_csv)
        diag_layout.addLayout(exp_row)

        layout.addWidget(diag_grp, stretch=1)
        self.tabs.addTab(tab3, "3. Diagnostics && SIGN")

    # ------------------------------------------------------------------ #
    #                     Delay & Solver UI Callbacks                    #
    # ------------------------------------------------------------------ #

    def _step_delay_prev(self):
        idx = self.delay_cb.currentIndex()
        if idx > 0:
            self.delay_cb.setCurrentIndex(idx - 1)

    def _step_delay_next(self):
        idx = self.delay_cb.currentIndex()
        if idx < self.delay_cb.count() - 1:
            self.delay_cb.setCurrentIndex(idx + 1)

    def _on_delay_changed(self, _idx: int):
        # If this delay already has fitted details, refresh diagnostics table
        delay_idx = self.delay_cb.currentIndex()
        if 0 <= delay_idx < len(self.delay_indices):
            actual_idx = self.delay_indices[delay_idx]
            t2 = float(self.dataset.delays[actual_idx])
            details_map = getattr(self.dataset, "_sd_kubo_details", {})
            for k_t2, det in details_map.items():
                if np.isclose(float(k_t2), t2, atol=1e-6):
                    self._populate_diagnostics_from_result(det)
                    break
        self.update_plots()

    def _on_solver_changed(self, idx: int):
        if idx == 0:
            self.chk_varpro.setChecked(True)
        elif idx == 1:
            self.chk_varpro.setChecked(False)

    def _get_solver_and_parallel_config(self) -> tuple[str, bool, str, int]:
        s_idx = self.solver_cb.currentIndex()
        use_varpro = self.chk_varpro.isChecked()
        if s_idx == 2:
            solver = "lbfgsb"
        elif use_varpro:
            solver = "trf_varpro"
        else:
            solver = "trf"

        p_idx = self.parallel_cb.currentIndex()
        if p_idx == 0:
            parallel_mode = "processpool"
        elif p_idx == 1:
            parallel_mode = "threadpool"
        else:
            parallel_mode = "disabled"

        workers = int(self.workers_spin.value())
        return solver, use_varpro, parallel_mode, workers

    # ------------------------------------------------------------------ #
    #                  Canvas Interactive Peak Picking                   #
    # ------------------------------------------------------------------ #

    def _on_add_band_toggled(self, checked: bool):
        if checked:
            self.btn_add_band_plot.setText("🎯 Click GSB & Drag to ESA...")
        else:
            self.btn_add_band_plot.setText("🎯 Add Band (Click/Drag Plot)")

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
        is_vertical = pump_axis_val == "Vertical"

        if is_vertical:
            w1_gsb = start_y
            w3_gsb = start_x
            w3_esa_click = end_x
        else:
            w1_gsb = start_x
            w3_gsb = start_y
            w3_esa_click = end_y

        anharm_drag = abs(w3_gsb - w3_esa_click)
        if anharm_drag >= 1.5:
            anharm = float(anharm_drag)
            w3_esa = float(w3_esa_click)
        else:
            anharm = 16.0
            w3_esa = float(w3_gsb - anharm)

        # Estimate normalized initial positive amplitudes from experimental ROI slice
        delay_idx = self.delay_cb.currentIndex()
        actual_idx = self.delay_indices[delay_idx] if 0 <= delay_idx < len(self.delay_indices) else 0
        map_2d = self.Z_exp_roi[:, :, actual_idx]
        exp_max = float(np.max(np.abs(map_2d)))
        if exp_max <= 0:
            exp_max = 1.0

        p_idx = int(np.argmin(np.abs(self.pump_axis - w1_gsb)))
        r_gsb_idx = int(np.argmin(np.abs(self.probe_axis - w3_gsb)))
        r_esa_idx = int(np.argmin(np.abs(self.probe_axis - w3_esa)))

        amp_gsb_raw = float(map_2d[p_idx, r_gsb_idx])
        amp_esa_raw = float(map_2d[p_idx, r_esa_idx])

        # GSB has a negative prefactor (-2 * A_GSB) in simulate_2d_spectrum, so A_GSB > 0 gives negative bleach!
        amp_gsb = max(0.10, round(-amp_gsb_raw / exp_max, 3)) if amp_gsb_raw < 0 else max(0.25, round(abs(amp_gsb_raw) / exp_max, 3))
        amp_esa = max(0.10, round(amp_esa_raw / exp_max, 3)) if amp_esa_raw > 0 else max(0.20, round(0.8 * amp_gsb, 3))

        # Check if user clicked an off-diagonal peak (|w1 - w3| > 6 cm-1)
        is_off_diag = abs(w1_gsb - w3_gsb) > 6.0
        link_w3 = "free" if is_off_diag else "diagonal"

        self.add_band(
            w01=w1_gsb,
            dw=anharm,
            a_gsb=amp_gsb,
            a_esa=amp_esa,
            w3=w3_gsb,
            link_w3=link_w3,
        )
        self.btn_add_band_plot.setChecked(False)
        self.update_plots()

    # ------------------------------------------------------------------ #
    #                  Bands & Cross-Peaks Management                    #
    # ------------------------------------------------------------------ #

    def add_band(
        self,
        w01: float | None = None,
        dw: float | None = None,
        a_gsb: float | None = None,
        a_esa: float | None = None,
        *,
        w3: float | None = None,
        link_w1: int | None = None,
        link_w3: str | int = "diagonal",
        link_anharm: int | None = None,
        ffcf_id: int = 0,
        corr_eta: float = 1.0,
        fixed: bool = False,
    ):
        """Add a vibrational band (diagonal or cross-peak) to the model."""
        self._sync_state_from_tables()

        p_mean = float(np.mean(self.pump_axis))
        val_w1 = float(w01) if w01 is not None else p_mean
        val_w3 = float(w3) if w3 is not None else val_w1
        val_dw = float(dw) if dw is not None else 16.0
        val_g = float(a_gsb) if a_gsb is not None else 1.0
        val_e = float(a_esa) if a_esa is not None else 0.8

        self._bands_state.append({
            "w1": val_w1,
            "w3": val_w3,
            "anharm": val_dw,
            "amp_gsb": val_g,
            "amp_esa": val_e,
            "link_w1": link_w1,
            "link_w3": link_w3,
            "link_anharm": link_anharm,
            "ffcf_id": int(ffcf_id),
            "corr_eta": float(corr_eta),
            "fixed": bool(fixed),
        })
        self.num_bands = len(self._bands_state)
        self._rebuild_bands_table()
        self._sync_legacy_param_table()

    def delete_band(self):
        """Delete the currently selected band (or last band if none selected)."""
        self._sync_state_from_tables()
        if len(self._bands_state) <= 1:
            QMessageBox.information(self, "Minimum Bands", "Must keep at least 1 Kubo band.")
            return

        sel_row = self.bands_table.currentRow()
        if sel_row < 0 or sel_row >= len(self._bands_state):
            # Also check legacy param_table selection
            leg_row = self.param_table.currentRow()
            if 0 <= leg_row < 4 * len(self._bands_state):
                sel_row = leg_row // 4
            else:
                sel_row = len(self._bands_state) - 1

        self._bands_state.pop(sel_row)
        # Clean up any links pointing to or past sel_row
        for idx, bd in enumerate(self._bands_state):
            for key in ("link_w1", "link_w3", "link_anharm"):
                val = bd.get(key)
                if isinstance(val, int):
                    if val == sel_row or val == idx:
                        bd[key] = "diagonal" if key == "link_w3" else None
                    elif val > sel_row:
                        bd[key] = val - 1

        self.num_bands = len(self._bands_state)
        self._rebuild_bands_table()
        self._sync_legacy_param_table()
        self.update_plots()

    def _clear_extra_bands(self):
        """Reset bands table to just the first band."""
        self._sync_state_from_tables()
        if len(self._bands_state) > 1:
            self._bands_state = [self._bands_state[0]]
            self._bands_state[0]["link_w1"] = None
            self._bands_state[0]["link_w3"] = "diagonal"
            self._bands_state[0]["link_anharm"] = None
            self.num_bands = 1
            self._rebuild_bands_table()
            self._sync_legacy_param_table()
            self.update_plots()

    def relabel_bands_and_decays(self):
        """Rebuild tables to keep labels consistent."""
        self._rebuild_bands_table()
        self._rebuild_ffcf_table()
        self._sync_legacy_param_table()

    def _open_cross_peak_generator(self):
        self._sync_state_from_tables()
        if len(self._bands_state) < 2:
            QMessageBox.information(
                self,
                "Need At Least 2 Bands",
                "Please add at least 2 diagonal bands before generating coupled cross-peaks.",
            )
            return

        labels = [f"Band {i + 1} (ω₁ = {b['w1']:.1f} cm⁻¹)" for i, b in enumerate(self._bands_state)]
        dlg = KuboCrossPeakDialog(self, labels)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        cfg = dlg.get_config()
        ia, ib = cfg["idx_a"], cfg["idx_b"]
        if ia == ib:
            QMessageBox.warning(self, "Invalid Selection", "Please choose two distinct parent bands.")
            return

        ba = self._bands_state[ia]
        bb = self._bands_state[ib]
        ratio = cfg["amp_ratio"]
        eta = cfg["corr_eta"]
        link_anh = cfg["link_anharm_to_probed"]

        g_xp = round(ratio * np.sqrt(max(0.01, abs(ba["amp_gsb"]) * abs(bb["amp_gsb"]))), 3)
        e_xp = round(ratio * np.sqrt(max(0.01, abs(ba["amp_esa"]) * abs(bb["amp_esa"]))), 3)
        anh_avg = 0.5 * (ba["anharm"] + bb["anharm"])

        # Cross-peak A -> B (pump at A, probe at B)
        self.add_band(
            w01=ba["w1"],
            w3=bb["w1"],
            dw=bb["anharm"] if link_anh else anh_avg,
            a_gsb=g_xp,
            a_esa=e_xp,
            link_w1=ia,
            link_w3=ib,
            link_anharm=ib if link_anh else None,
            ffcf_id=ba.get("ffcf_id", 0),
            corr_eta=eta,
        )
        # Cross-peak B -> A (pump at B, probe at A)
        self.add_band(
            w01=bb["w1"],
            w3=ba["w1"],
            dw=ba["anharm"] if link_anh else anh_avg,
            a_gsb=g_xp,
            a_esa=e_xp,
            link_w1=ib,
            link_w3=ia,
            link_anharm=ia if link_anh else None,
            ffcf_id=bb.get("ffcf_id", 0),
            corr_eta=eta,
        )
        self.update_plots()

    def _rebuild_bands_table(self):
        """Render `self._bands_state` into `self.bands_table`."""
        self._updating_tables = True
        try:
            n_b = len(self._bands_state)
            self.bands_table.setRowCount(n_b)

            for i, bd in enumerate(self._bands_state):
                # Resolve linked values for display
                lw1 = bd.get("link_w1")
                lw3 = bd.get("link_w3", "diagonal")
                lanh = bd.get("link_anharm")

                if isinstance(lw1, int) and 0 <= lw1 < n_b and lw1 != i:
                    bd["w1"] = self._bands_state[lw1]["w1"]
                if lw3 == "diagonal":
                    bd["w3"] = bd["w1"]
                elif isinstance(lw3, int) and 0 <= lw3 < n_b and lw3 != i:
                    bd["w3"] = self._bands_state[lw3]["w1"]
                if isinstance(lanh, int) and 0 <= lanh < n_b and lanh != i:
                    bd["anharm"] = self._bands_state[lanh]["anharm"]

                is_xp = (lw3 != "diagonal") and abs(bd["w1"] - bd["w3"]) > 1e-3
                lbl_txt = f"XP{i + 1}" if is_xp else f"B{i + 1}"
                lbl_item = QTableWidgetItem(lbl_txt)
                lbl_item.setFlags(lbl_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                lbl_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl_item.setBackground(QColor("#fef3c7" if is_xp else "#dbeafe"))
                self.bands_table.setItem(i, 0, lbl_item)

                # Col 1: w1
                w1_item = QTableWidgetItem(f"{bd['w1']:.2f}")
                if isinstance(lw1, int) and lw1 != i:
                    w1_item.setFlags(w1_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    w1_item.setForeground(QColor("#64748b"))
                self.bands_table.setItem(i, 1, w1_item)

                # Col 2: Link w1 combo
                cb_w1 = QComboBox()
                cb_w1.addItem("Free", None)
                for j in range(n_b):
                    if j != i:
                        cb_w1.addItem(f"= B{j + 1}", j)
                if isinstance(lw1, int) and lw1 != i:
                    idx_f = cb_w1.findData(lw1)
                    if idx_f >= 0:
                        cb_w1.setCurrentIndex(idx_f)
                cb_w1.currentIndexChanged.connect(self._on_band_link_combo_changed)
                self.bands_table.setCellWidget(i, 2, cb_w1)

                # Col 3: w3
                w3_item = QTableWidgetItem(f"{bd['w3']:.2f}")
                if lw3 == "diagonal" or (isinstance(lw3, int) and lw3 != i):
                    w3_item.setFlags(w3_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    w3_item.setForeground(QColor("#64748b"))
                self.bands_table.setItem(i, 3, w3_item)

                # Col 4: Link w3 combo
                cb_w3 = QComboBox()
                cb_w3.addItem("= ω₁ (Diag)", "diagonal")
                cb_w3.addItem("Free (Off-Diag)", "free")
                for j in range(n_b):
                    if j != i:
                        cb_w3.addItem(f"= B{j + 1} (ω₁)", j)
                idx_w3 = cb_w3.findData(lw3)
                if idx_w3 >= 0:
                    cb_w3.setCurrentIndex(idx_w3)
                cb_w3.currentIndexChanged.connect(self._on_band_link_combo_changed)
                self.bands_table.setCellWidget(i, 4, cb_w3)

                # Col 5: Anharmonicity Δ
                dw_item = QTableWidgetItem(f"{bd['anharm']:.2f}")
                if isinstance(lanh, int) and lanh != i:
                    dw_item.setFlags(dw_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    dw_item.setForeground(QColor("#64748b"))
                self.bands_table.setItem(i, 5, dw_item)

                # Col 6: Link Δ combo
                cb_anh = QComboBox()
                cb_anh.addItem("Free", None)
                for j in range(n_b):
                    if j != i:
                        cb_anh.addItem(f"= B{j + 1}", j)
                if isinstance(lanh, int) and lanh != i:
                    idx_a = cb_anh.findData(lanh)
                    if idx_a >= 0:
                        cb_anh.setCurrentIndex(idx_a)
                cb_anh.currentIndexChanged.connect(self._on_band_link_combo_changed)
                self.bands_table.setCellWidget(i, 6, cb_anh)

                # Col 7 & 8: A_GSB, A_ESA
                g_item = QTableWidgetItem(f"{bd['amp_gsb']:.3f}")
                e_item = QTableWidgetItem(f"{bd['amp_esa']:.3f}")
                self.bands_table.setItem(i, 7, g_item)
                self.bands_table.setItem(i, 8, e_item)

                # Col 9: FFCF group selector
                cb_ffcf = QComboBox()
                for g_idx in range(len(self._ffcf_groups)):
                    cb_ffcf.addItem(f"FFCF #{g_idx + 1}", g_idx)
                f_id = min(int(bd.get("ffcf_id", 0)), len(self._ffcf_groups) - 1)
                cb_ffcf.setCurrentIndex(max(0, f_id))
                cb_ffcf.currentIndexChanged.connect(self._on_band_link_combo_changed)
                self.bands_table.setCellWidget(i, 9, cb_ffcf)

                # Col 10: Fix checkbox
                chk_w = QWidget()
                chk_l = QHBoxLayout(chk_w)
                chk_l.setContentsMargins(0, 0, 0, 0)
                chk_l.setAlignment(Qt.AlignmentFlag.AlignCenter)
                chk = QCheckBox()
                chk.setChecked(bool(bd.get("fixed", False)))
                chk.stateChanged.connect(self._on_band_link_combo_changed)
                chk_l.addWidget(chk)
                self.bands_table.setCellWidget(i, 10, chk_w)
        finally:
            self._updating_tables = False

    def _on_band_link_combo_changed(self, *_args):
        if self._updating_tables:
            return
        self._sync_state_from_tables()
        self._rebuild_bands_table()
        self._sync_legacy_param_table()

    def _on_bands_table_item_changed(self, _item):
        if self._updating_tables:
            return
        self._sync_state_from_tables()
        self._rebuild_bands_table()
        self._sync_legacy_param_table()

    # ------------------------------------------------------------------ #
    #                  FFCF Decay Groups Management                      #
    # ------------------------------------------------------------------ #

    def _add_new_ffcf_group(self):
        self._sync_state_from_tables()
        new_idx = len(self._ffcf_groups) + 1
        # Clone structure of group 0 so number of terms stays consistent or independent
        base_grp = copy.deepcopy(self._ffcf_groups[0])
        base_grp["name"] = f"FFCF #{new_idx} (Band-Specific)"
        self._ffcf_groups.append(base_grp)
        self.ffcf_group_cb.addItem(base_grp["name"])
        self.ffcf_group_cb.setCurrentIndex(len(self._ffcf_groups) - 1)
        self._rebuild_bands_table()
        self._rebuild_ffcf_table()
        self._sync_legacy_param_table()

    def _on_ffcf_group_selected(self, idx: int):
        if self._updating_tables or idx < 0 or idx >= len(self._ffcf_groups):
            return
        self._sync_state_from_tables()
        self._active_ffcf_idx = idx
        self._rebuild_ffcf_table()

    def add_decay(self):
        """Add a Kubo decay pair (Δ_m, τc,m) to the active FFCF group (and keep num_decays updated)."""
        self._sync_state_from_tables()
        grp = self._ffcf_groups[self._active_ffcf_idx]
        m = len(grp["terms"]) + 1
        default_dw = 12.0 if m == 1 else 6.0
        default_tau = 1.5 if m == 1 else 10.0
        # [dw_val, dw_lb, dw_ub, dw_fix, tau_val, tau_lb, tau_ub, tau_fix]
        grp["terms"].append([default_dw, 0.05, 150.0, False, default_tau, 0.02, 200.0, False])
        self.num_decays = len(self._ffcf_groups[0]["terms"])
        self._rebuild_ffcf_table()
        self._sync_legacy_param_table()

    def delete_decay(self):
        """Delete the last Kubo decay term from the active FFCF group."""
        self._sync_state_from_tables()
        grp = self._ffcf_groups[self._active_ffcf_idx]
        if len(grp["terms"]) <= 1:
            QMessageBox.information(
                self, "Minimum Decays", "Each FFCF group must keep at least 1 Kubo decay term."
            )
            return
        grp["terms"].pop()
        self.num_decays = len(self._ffcf_groups[0]["terms"])
        self._rebuild_ffcf_table()
        self._sync_legacy_param_table()

    def _rebuild_ffcf_table(self):
        """Render the active FFCF group (`self._active_ffcf_idx`) into `self.ffcf_table`."""
        self._updating_tables = True
        try:
            grp = self._ffcf_groups[self._active_ffcf_idx]
            n_terms = len(grp["terms"])
            self.ffcf_table.setRowCount(2 + 2 * n_terms)

            rows_spec = [
                ("Γ_hom / dT2 (cm⁻¹)", grp["dw_T2"], "#e0f2fe"),
                ("Δ₀ Static (cm⁻¹)", grp["dw_0"], "#f1f5f9"),
            ]
            for m, term in enumerate(grp["terms"]):
                rows_spec.append((f"Δ_{m + 1} Amplitude (cm⁻¹)", term[0:4], "#ffffff"))
                rows_spec.append((f"τc,{m + 1} Correlation (ps)", term[4:8], "#ffffff"))

            for r_idx, (label, vals, bg_hex) in enumerate(rows_spec):
                p_val, lb_val, ub_val, is_fix = vals
                name_item = QTableWidgetItem(label)
                name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                name_item.setBackground(QColor(bg_hex))
                self.ffcf_table.setItem(r_idx, 0, name_item)

                self.ffcf_table.setItem(r_idx, 1, QTableWidgetItem(f"{float(p_val):.3f}"))
                self.ffcf_table.setItem(r_idx, 2, QTableWidgetItem(f"{float(lb_val):.3f}"))
                self.ffcf_table.setItem(r_idx, 3, QTableWidgetItem(f"{float(ub_val):.3f}"))

                chk_w = QWidget()
                chk_l = QHBoxLayout(chk_w)
                chk_l.setContentsMargins(0, 0, 0, 0)
                chk_l.setAlignment(Qt.AlignmentFlag.AlignCenter)
                chk = QCheckBox()
                chk.setChecked(bool(is_fix))
                chk.stateChanged.connect(self._on_ffcf_table_item_changed)
                chk_l.addWidget(chk)
                self.ffcf_table.setCellWidget(r_idx, 4, chk_w)
        finally:
            self._updating_tables = False

    def _on_ffcf_table_item_changed(self, *_args):
        if self._updating_tables:
            return
        self._sync_state_from_tables()
        self._sync_legacy_param_table()

    # ------------------------------------------------------------------ #
    #            Table State Synchronization & Parameter Packing         #
    # ------------------------------------------------------------------ #

    def _sync_state_from_tables(self):
        """Read current user edits from `self.bands_table` and `self.ffcf_table` into state dicts."""
        if self._updating_tables:
            return

        # 1. Read bands_table
        if hasattr(self, "bands_table") and self.bands_table.rowCount() == len(self._bands_state):
            for i, bd in enumerate(self._bands_state):
                try:
                    if self.bands_table.item(i, 1):
                        bd["w1"] = float(self.bands_table.item(i, 1).text())
                    if self.bands_table.item(i, 3):
                        bd["w3"] = float(self.bands_table.item(i, 3).text())
                    if self.bands_table.item(i, 5):
                        bd["anharm"] = float(self.bands_table.item(i, 5).text())
                    if self.bands_table.item(i, 7):
                        bd["amp_gsb"] = float(self.bands_table.item(i, 7).text())
                    if self.bands_table.item(i, 8):
                        bd["amp_esa"] = float(self.bands_table.item(i, 8).text())
                except ValueError:
                    pass

                cb_w1 = self.bands_table.cellWidget(i, 2)
                if isinstance(cb_w1, QComboBox):
                    bd["link_w1"] = cb_w1.currentData()

                cb_w3 = self.bands_table.cellWidget(i, 4)
                if isinstance(cb_w3, QComboBox):
                    bd["link_w3"] = cb_w3.currentData()

                cb_anh = self.bands_table.cellWidget(i, 6)
                if isinstance(cb_anh, QComboBox):
                    bd["link_anharm"] = cb_anh.currentData()

                cb_ffcf = self.bands_table.cellWidget(i, 9)
                if isinstance(cb_ffcf, QComboBox) and cb_ffcf.currentData() is not None:
                    bd["ffcf_id"] = int(cb_ffcf.currentData())

                chk_w = self.bands_table.cellWidget(i, 10)
                if chk_w is not None:
                    chk = chk_w.findChild(QCheckBox)
                    if chk is not None:
                        bd["fixed"] = chk.isChecked()

        # Resolve inter-band links immediately
        n_b = len(self._bands_state)
        for i, bd in enumerate(self._bands_state):
            lw1 = bd.get("link_w1")
            if isinstance(lw1, int) and 0 <= lw1 < n_b and lw1 != i:
                bd["w1"] = self._bands_state[lw1]["w1"]
            lw3 = bd.get("link_w3", "diagonal")
            if lw3 == "diagonal":
                bd["w3"] = bd["w1"]
            elif isinstance(lw3, int) and 0 <= lw3 < n_b and lw3 != i:
                bd["w3"] = self._bands_state[lw3]["w1"]
            lanh = bd.get("link_anharm")
            if isinstance(lanh, int) and 0 <= lanh < n_b and lanh != i:
                bd["anharm"] = self._bands_state[lanh]["anharm"]

        # 2. Read ffcf_table for active group
        grp = self._ffcf_groups[self._active_ffcf_idx]
        expected_rows = 2 + 2 * len(grp["terms"])
        if hasattr(self, "ffcf_table") and self.ffcf_table.rowCount() == expected_rows:
            def _read_row(r: int, fallback: list):
                try:
                    v = float(self.ffcf_table.item(r, 1).text())
                    lb = float(self.ffcf_table.item(r, 2).text())
                    ub = float(self.ffcf_table.item(r, 3).text())
                    cw = self.ffcf_table.cellWidget(r, 4)
                    fx = cw.findChild(QCheckBox).isChecked() if cw else False
                    return [v, lb, ub, fx]
                except Exception:
                    return list(fallback)

            grp["dw_T2"] = _read_row(0, grp["dw_T2"])
            grp["dw_0"] = _read_row(1, grp["dw_0"])
            for m in range(len(grp["terms"])):
                dw_part = _read_row(2 + 2 * m, grp["terms"][m][0:4])
                tau_part = _read_row(2 + 2 * m + 1, grp["terms"][m][4:8])
                grp["terms"][m] = dw_part + tau_part

    def _sync_legacy_param_table(self):
        """Synchronize the full flat parameter list into `self.param_table` for compatibility."""
        p0, bounds, names = self._build_flat_params_bounds_names()
        self.param_table.setRowCount(len(p0))
        for r, (nm, val, (lb, ub)) in enumerate(zip(names, p0, bounds)):
            self.param_table.setItem(r, 0, QTableWidgetItem(nm))
            self.param_table.setItem(r, 1, QTableWidgetItem(f"{val:.4f}"))
            self.param_table.setItem(r, 2, QTableWidgetItem(f"{lb:.4f}"))
            self.param_table.setItem(r, 3, QTableWidgetItem(f"{ub:.4f}"))
            chk_w = QWidget()
            chk_l = QHBoxLayout(chk_w)
            chk_l.setContentsMargins(0, 0, 0, 0)
            chk = QCheckBox()
            chk.setChecked(abs(ub - lb) <= 1e-11)
            chk_l.addWidget(chk)
            self.param_table.setCellWidget(r, 4, chk_w)

    def _get_ffcf_structure(self) -> list[int]:
        return [len(grp["terms"]) for grp in self._ffcf_groups]

    def _get_band_links_config(self) -> list[dict[str, Any]]:
        self._sync_state_from_tables()
        links = []
        for bd in self._bands_state:
            links.append({
                "w3": float(bd["w3"]),
                "link_w1": bd.get("link_w1"),
                "link_w3": bd.get("link_w3", "diagonal"),
                "link_anharm": bd.get("link_anharm"),
                "ffcf_id": int(bd.get("ffcf_id", 0)),
                "ffcf_id_w3": int(bd.get("ffcf_id", 0)),
                "corr_eta": float(bd.get("corr_eta", 1.0)),
            })
        return links

    def _build_flat_params_bounds_names(
        self,
    ) -> tuple[list[float], list[tuple[float, float]], list[str]]:
        """Build `(p0, bounds, param_names)` matching `unpack_kubo_params(..., has_dw0=True, ffcf_structure=...)`."""
        self._sync_state_from_tables()
        p0: list[float] = []
        bounds: list[tuple[float, float]] = []
        names: list[str] = []

        pos_tol = float(self.pos_tol_spin.value()) if hasattr(self, "pos_tol_spin") else 8.0
        anh_min = float(self.anharm_min_spin.value()) if hasattr(self, "anharm_min_spin") else 2.0
        anh_max = float(self.anharm_max_spin.value()) if hasattr(self, "anharm_max_spin") else 60.0
        constrain_signs = (
            self.chk_constrain_signs.isChecked()
            if hasattr(self, "chk_constrain_signs")
            else True
        )
        amp_lb = 0.0 if constrain_signs else -50.0
        amp_ub = 50.0

        n_b = len(self._bands_state)
        for i, bd in enumerate(self._bands_state):
            k = i + 1
            w1 = float(bd["w1"])
            dw = float(bd["anharm"])
            g = float(bd["amp_gsb"])
            e = float(bd["amp_esa"])
            is_fixed = bool(bd.get("fixed", False))
            lw1 = bd.get("link_w1")
            lanh = bd.get("link_anharm")

            w1_is_linked = isinstance(lw1, int) and 0 <= lw1 < n_b and lw1 != i
            dw_is_linked = isinstance(lanh, int) and 0 <= lanh < n_b and lanh != i

            w1_bounds = (w1, w1) if (is_fixed or w1_is_linked) else (w1 - pos_tol, w1 + pos_tol)
            dw_clamped = float(np.clip(dw, anh_min, anh_max))
            dw_bounds = (
                (dw, dw)
                if (is_fixed or dw_is_linked)
                else (min(dw - 0.1, anh_min), max(dw + 0.1, anh_max))
            )
            g_bounds = (max(amp_lb, g) if not is_fixed else g, amp_ub if not is_fixed else g)
            if not is_fixed:
                g_bounds = (amp_lb, amp_ub)
                e_bounds = (amp_lb, amp_ub)
            else:
                g_bounds = (g, g)
                e_bounds = (e, e)

            p0.extend([w1, dw_clamped if not (is_fixed or dw_is_linked) else dw, g, e])
            bounds.extend([w1_bounds, dw_bounds, g_bounds, e_bounds])
            names.extend([
                f"ω₁_{k} (cm⁻¹)",
                f"Δ_{k} (cm⁻¹)",
                f"A_GSB_{k}",
                f"A_ESA_{k}",
            ])

        # FFCF groups: [dw_T2, dw_0, (dw_1, tau_1), ...]
        n_groups = len(self._ffcf_groups)
        for g_idx, grp in enumerate(self._ffcf_groups):
            prefix = f"G{g_idx + 1}_" if n_groups > 1 else ""
            for label_nm, spec in [
                (f"{prefix}Γ_hom (cm⁻¹)", grp["dw_T2"]),
                (f"{prefix}Δ₀ (cm⁻¹)", grp["dw_0"]),
            ]:
                val, lb, ub, fix = float(spec[0]), float(spec[1]), float(spec[2]), bool(spec[3])
                p0.append(val)
                bounds.append((val, val) if fix else (min(lb, val), max(ub, val)))
                names.append(label_nm)

            for m, term in enumerate(grp["terms"]):
                dw_v, dw_lb, dw_ub, dw_fx = float(term[0]), float(term[1]), float(term[2]), bool(term[3])
                tau_v, tau_lb, tau_ub, tau_fx = float(term[4]), float(term[5]), float(term[6]), bool(term[7])
                p0.extend([dw_v, tau_v])
                bounds.append((dw_v, dw_v) if dw_fx else (min(dw_lb, dw_v), max(dw_ub, dw_v)))
                bounds.append((tau_v, tau_v) if tau_fx else (min(tau_lb, tau_v), max(tau_ub, tau_v)))
                names.extend([
                    f"{prefix}Δ_{m + 1} (cm⁻¹)",
                    f"{prefix}τc,{m + 1} (ps)",
                ])

        return p0, bounds, names

    def get_current_params(self) -> tuple[list[float], list[tuple[float, float]]]:
        """Return `(p0, bounds)` flat lists for the current model state."""
        p0, bounds, _ = self._build_flat_params_bounds_names()
        return p0, bounds

    def get_structured_bands_and_decays(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return structured `(bands_list, ffcf_groups_list)` for `simulate_2d_spectrum`."""
        self._sync_state_from_tables()
        bands_out = []
        for bd in self._bands_state:
            bands_out.append({
                "w1": float(bd["w1"]),
                "w3": float(bd["w3"]),
                "anharm": float(bd["anharm"]),
                "amp_gsb": float(bd["amp_gsb"]),
                "amp_esa": float(bd["amp_esa"]),
                "ffcf_id": int(bd.get("ffcf_id", 0)),
                "ffcf_id_w3": int(bd.get("ffcf_id", 0)),
                "corr_eta": float(bd.get("corr_eta", 1.0)),
                "link_w1": bd.get("link_w1"),
                "link_w3": bd.get("link_w3", "diagonal"),
                "link_anharm": bd.get("link_anharm"),
            })

        decays_out = []
        for grp in self._ffcf_groups:
            terms = [(float(t[0]), float(t[4])) for t in grp["terms"]]
            decays_out.append({
                "dw_T2": float(grp["dw_T2"][0]),
                "dw_0": float(grp["dw_0"][0]),
                "terms": terms,
            })
        return bands_out, decays_out

    def pfit_to_p0(self, show_dialog: bool = False):
        """Copy the last fitted parameters `self.last_fit_params` into the initial guess tables."""
        if self.last_fit_params is None:
            if show_dialog:
                QMessageBox.warning(
                    self, "No Fitted Parameters", "Please run a fit first before copying fitted values."
                )
            return

        vec = np.asarray(self.last_fit_params, dtype=float)
        n_b = len(self._bands_state)
        if len(vec) < 4 * n_b:
            return

        for i in range(n_b):
            self._bands_state[i]["w1"] = float(vec[4 * i + 0])
            self._bands_state[i]["anharm"] = float(vec[4 * i + 1])
            self._bands_state[i]["amp_gsb"] = float(vec[4 * i + 2])
            self._bands_state[i]["amp_esa"] = float(vec[4 * i + 3])

        # Resolve links after updating w1/anharm
        for i, bd in enumerate(self._bands_state):
            lw1 = bd.get("link_w1")
            if isinstance(lw1, int) and 0 <= lw1 < n_b and lw1 != i:
                bd["w1"] = self._bands_state[lw1]["w1"]
            lw3 = bd.get("link_w3", "diagonal")
            if lw3 == "diagonal":
                bd["w3"] = bd["w1"]
            elif isinstance(lw3, int) and 0 <= lw3 < n_b and lw3 != i:
                bd["w3"] = self._bands_state[lw3]["w1"]
            lanh = bd.get("link_anharm")
            if isinstance(lanh, int) and 0 <= lanh < n_b and lanh != i:
                bd["anharm"] = self._bands_state[lanh]["anharm"]

        ptr = 4 * n_b
        for grp in self._ffcf_groups:
            if ptr + 2 <= len(vec):
                grp["dw_T2"][0] = float(vec[ptr])
                grp["dw_0"][0] = float(vec[ptr + 1])
                ptr += 2
            for m in range(len(grp["terms"])):
                if ptr + 2 <= len(vec):
                    grp["terms"][m][0] = float(vec[ptr])
                    grp["terms"][m][4] = float(vec[ptr + 1])
                    ptr += 2

        self._rebuild_bands_table()
        self._rebuild_ffcf_table()
        self._sync_legacy_param_table()

        if show_dialog:
            QMessageBox.information(
                self, "Success", "Fitted parameters copied to Initial Guess (p₀)."
            )

    # ------------------------------------------------------------------ #
    #                  1st-Order FTIR Spectrum Handling                  #
    # ------------------------------------------------------------------ #

    def _on_ftir_mode_changed(self, _idx: int):
        if self.fid_data is not None:
            mode_str = "joint_fit" if self.ftir_mode_cb.currentIndex() == 0 else "empirical_fid"
            self.fid_data["mode"] = mode_str
            self.update_plots()

    def clear_ftir_spectrum(self):
        self.fid_data = None
        self.ftir_spectrum = None
        self.ftir_status.setText("No FTIR loaded (using pure analytical Kubo line-shape).")
        self.ftir_status.setStyleSheet("color: #64748b;")
        self.update_plots()

    def load_ftir_spectrum(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load FTIR Spectrum File", "", "Text Files (*.txt *.csv *.dat);;All Files (*)"
        )
        if not path:
            return

        try:
            try:
                data = np.loadtxt(path)
            except ValueError:
                data = np.loadtxt(path, delimiter=",")

            w_ftir = np.asarray(data[:, 0], dtype=float)
            abs_ftir = np.asarray(data[:, 1], dtype=float)

            if w_ftir[1] < w_ftir[0]:
                w_ftir = w_ftir[::-1]
                abs_ftir = abs_ftir[::-1]

            w_center = float(np.mean(self.pump_axis))
            w_grid = np.linspace(w_center - 300.0, w_center + 300.0, 4096)
            abs_grid = np.interp(w_grid, w_ftir, abs_ftir, left=0.0, right=0.0)

            fid = np.fft.ifft(np.fft.fftshift(abs_grid))
            dw = w_grid[1] - w_grid[0]
            dt_ftir = 1.0 / (len(w_grid) * dw * c_0)
            fid_t = np.arange(len(fid)) * dt_ftir
            fid_abs = np.abs(fid)
            max_fid = float(np.max(fid_abs))
            if max_fid > 0:
                fid_abs = fid_abs / max_fid

            mode_str = "joint_fit" if self.ftir_mode_cb.currentIndex() == 0 else "empirical_fid"
            self.ftir_spectrum = (w_ftir, abs_ftir)
            self.fid_data = {
                "mode": mode_str,
                "fid": (fid_t, fid_abs),
                "spectrum": (w_ftir, abs_ftir),
            }
            self.ftir_status.setText(
                f"Loaded FTIR ({Path(path).name}) — Mode: {'Joint Analytical Fit' if mode_str == 'joint_fit' else 'Empirical FID'}."
            )
            self.ftir_status.setStyleSheet("color: #16a34a; font-weight: bold;")
            self.update_plots()
        except Exception as exc:
            QMessageBox.critical(
                self, "Load Failed", f"Failed to load 1st-order FTIR spectrum: {exc}"
            )

    def get_ft_settings(self):
        tmax = float(self.ft_tmax_spin.value())
        nt = None if self.ft_auto_check.isChecked() else int(self.ft_nt_spin.value())
        u = float(self.ft_u_spin.value())
        w_rot = (
            float(np.min(self.probe_axis) - 150.0)
            if self.ft_wrot_auto_check.isChecked()
            else float(self.ft_wrot_spin.value())
        )
        return tmax, nt, u, w_rot

    # ------------------------------------------------------------------ #
    #                  2x2 Canvas & Diagnostic Plotting                  #
    # ------------------------------------------------------------------ #

    def _get_residual_scale_limit(self) -> float:
        idx = self.res_scale_cb.currentIndex() if hasattr(self, "res_scale_cb") else 0
        return {0: 0.10, 1: 0.20, 2: 0.50, 3: 1.00}.get(idx, 0.10)

    def update_plots(self):
        """Render the 2x2 canvas: Experimental ROI, Simulated Kubo Fit, Residuals, and 4th Diagnostic Panel."""
        self.btn_update.setEnabled(False)
        self.btn_update.setText("Updating...")
        QCoreApplication.processEvents()

        try:
            bands_struct, decays_struct = self.get_structured_bands_and_decays()

            delay_idx = max(0, self.delay_cb.currentIndex())
            actual_idx = self.delay_indices[min(delay_idx, len(self.delay_indices) - 1)]
            t2 = float(self.dataset.delays[actual_idx])

            Z_exp_slice = self.Z_exp_roi[:, :, actual_idx]
            exp_max = float(np.max(np.abs(Z_exp_slice)))
            Z_exp_norm = Z_exp_slice / exp_max if exp_max > 0 else Z_exp_slice

            tmax, nt, u, w_rot = self.get_ft_settings()

            try:
                Z_sim_scaled = simulate_2d_spectrum(
                    bands_struct,
                    decays_struct,
                    self.pump_axis,
                    self.probe_axis,
                    t2,
                    tmax=tmax,
                    nt=nt,
                    w_rot=w_rot,
                    undersampling_factor=u,
                    fid_data=self.fid_data,
                    normalize_unit_basis=True,
                )
            except Exception:
                logger.debug("Simulation preview failed", exc_info=True)
                Z_sim_scaled = np.zeros_like(Z_exp_norm)

            Z_residuals = Z_exp_norm - Z_sim_scaled

            # Cache latest maps for fast 4th-panel switching
            self._cached_plot_state = {
                "t2": t2,
                "Z_exp_norm": Z_exp_norm,
                "Z_sim_scaled": Z_sim_scaled,
                "Z_residuals": Z_residuals,
                "bands": bands_struct,
                "decays": decays_struct,
                "tmax": tmax,
                "nt": nt,
            }

            # Recreate axes to clean up colorbars and keep sharex/sharey synced
            self.fig.clear()
            ax0 = self.fig.add_subplot(2, 2, 1)
            ax1 = self.fig.add_subplot(2, 2, 2, sharex=ax0, sharey=ax0)
            ax2 = self.fig.add_subplot(2, 2, 3, sharex=ax0, sharey=ax0)
            ax3 = self.fig.add_subplot(2, 2, 4)
            self.axes = [ax0, ax1, ax2, ax3]

            res_lim = self._get_residual_scale_limit()
            self.plot_contour(ax0, Z_exp_norm, -1.0, 1.0, f"Experimental ROI ($t_2$ = {t2:.2f} ps)")
            self.plot_contour(ax1, Z_sim_scaled, -1.0, 1.0, "Kubo Model Simulation")
            self.plot_contour(
                ax2, Z_residuals, -res_lim, res_lim, f"Residuals (Scale ±{res_lim:.2f})"
            )

            # Overlay numbered GSB ('bo') and ESA ('rx') markers on ax0
            s = pm.get_settings()
            pump_axis_val = getattr(s, "pump_axis", "Horizontal")
            if hasattr(pump_axis_val, "value"):
                pump_axis_val = pump_axis_val.value
            is_vert = pump_axis_val == "Vertical"

            for idx_b, b in enumerate(bands_struct):
                w1 = b["w1"]
                w3_gsb = b["w3"]
                w3_esa = w3_gsb - b["anharm"]
                if is_vert:
                    mx_g, my_g = w3_gsb, w1
                    mx_e, my_e = w3_esa, w1
                else:
                    mx_g, my_g = w1, w3_gsb
                    mx_e, my_e = w1, w3_esa
                ax0.plot(
                    mx_g,
                    my_g,
                    "bo",
                    markersize=7,
                    markeredgecolor="white",
                    markeredgewidth=1.5,
                )
                ax0.plot(mx_e, my_e, "rx", markersize=8, markeredgewidth=2.0)
                ax0.plot([mx_g, mx_e], [my_g, my_e], "k:", alpha=0.65)
                ax0.annotate(
                    f"{idx_b + 1}",
                    (mx_g, my_g),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.15", fc="#1d4ed8", ec="none", alpha=0.8),
                )

            self._render_fourth_panel(ax3)
            self.fig.tight_layout()
            self.canvas.draw_idle()
        finally:
            self.btn_update.setEnabled(True)
            self.btn_update.setText("↻ Update Preview")

    def _update_fourth_panel_only(self, *_args):
        if not hasattr(self, "axes") or len(self.axes) < 4:
            self.update_plots()
            return
        ax3 = self.axes[3]
        ax3.clear()
        self._render_fourth_panel(ax3)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _render_fourth_panel(self, ax3):
        """Render the user-selected diagnostic view into the bottom-right panel (`ax3`)."""
        mode_idx = self.diag_view_cb.currentIndex() if hasattr(self, "diag_view_cb") else 0
        state = getattr(self, "_cached_plot_state", None)

        if mode_idx == 0:
            # 1. Optimization Trajectory (Delta_1 vs tau_c)
            if self.fit_history:
                history_arr = np.asarray(self.fit_history, dtype=float)
                ax3.plot(
                    history_arr[:, 1],
                    history_arr[:, 0],
                    "o-",
                    color="#d97706",
                    linewidth=1.6,
                    markersize=4,
                    label="Step Path",
                )
                ax3.plot(
                    history_arr[0, 1],
                    history_arr[0, 0],
                    "o",
                    color="#dc2626",
                    markerfacecolor="none",
                    markeredgewidth=1.8,
                    markersize=7,
                    label="Start ($p_0$)",
                )
                ax3.plot(
                    history_arr[-1, 1],
                    history_arr[-1, 0],
                    "X",
                    color="#16a34a",
                    markeredgewidth=1.5,
                    markersize=8,
                    label="Converged",
                )
                ax3.legend(loc="best", fontsize=8)
            elif hasattr(self.dataset, "_sd_results") and "Kubo_Fit" in self.dataset._sd_results:
                arr = self.dataset._sd_results["Kubo_Fit"]
                if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] > 6:
                    ax3.plot(
                        arr[:, 6],
                        arr[:, 5],
                        "o-",
                        color="#d97706",
                        linewidth=1.5,
                        markersize=4,
                    )
            ax3.set_xlabel(r"Correlation Time $\tau_{c,1}$ (ps)")
            ax3.set_ylabel(r"Fluctuation Amplitude $\Delta_{1}$ ($\mathrm{cm}^{-1}$)")
            ax3.set_title("Optimization Trajectory (Kubo Coordinates)")
            ax3.grid(True, linestyle="--", alpha=0.35)

        elif mode_idx == 1 and state is not None:
            # 2. 1D Spectral Cuts (Diagonal & Pump Slice through Band 1)
            Z_exp = state["Z_exp_norm"]
            Z_sim = state["Z_sim_scaled"]
            bands = state["bands"]
            w1_target = bands[0]["w1"] if bands else float(np.mean(self.pump_axis))
            p_idx = int(np.argmin(np.abs(self.pump_axis - w1_target)))

            ax3.plot(
                self.probe_axis,
                Z_exp[p_idx, :],
                "o",
                color="#1e293b",
                markersize=3.5,
                alpha=0.75,
                label=rf"Exp Slice ($\omega_1$={self.pump_axis[p_idx]:.1f})",
            )
            ax3.plot(
                self.probe_axis,
                Z_sim[p_idx, :],
                "-",
                color="#2563eb",
                linewidth=1.8,
                label="Kubo Fit Slice",
            )
            ax3.plot(
                self.probe_axis,
                (Z_exp[p_idx, :] - Z_sim[p_idx, :]),
                "--",
                color="#dc2626",
                linewidth=1.1,
                alpha=0.8,
                label="Residual",
            )
            ax3.axhline(0.0, color="gray", linewidth=0.7, alpha=0.5)
            ax3.set_xlabel(r"Probe Frequency $\omega_3$ ($\mathrm{cm}^{-1}$)")
            ax3.set_ylabel(r"Normalized $\Delta A$")
            ax3.set_title(rf"1D Probe Slice at $\omega_1$ = {self.pump_axis[p_idx]:.1f} $\mathrm{{cm}}^{{-1}}$")
            ax3.legend(loc="best", fontsize=8)
            ax3.grid(True, linestyle="--", alpha=0.35)

        elif mode_idx == 2 and state is not None:
            # 3. FFCF C(t) Decay & t2 Kinetics
            decays = state["decays"]
            tmax = state["tmax"]
            t_grid = np.linspace(0.0, max(5.0, tmax * 2.0), 200)
            c_t = evaluate_ffcf_ct(t_grid, decays, ffcf_id=0)
            ax3.plot(
                t_grid,
                c_t,
                "-",
                color="#7c3aed",
                linewidth=2.0,
                label=r"Analytical $C(t) = \Delta_0^2 + \sum \Delta_m^2 e^{-t/\tau_{c,m}}$",
            )
            # Mark current active delay t2
            t2_now = state["t2"]
            c_now = float(evaluate_ffcf_ct(np.array([t2_now]), decays, ffcf_id=0)[0])
            ax3.plot(
                [t2_now],
                [c_now],
                "o",
                color="#dc2626",
                markersize=7,
                label=rf"Active $t_2$ = {t2_now:.2f} ps",
            )
            ax3.set_xlabel(r"Waiting Time $t_2$ (ps)")
            ax3.set_ylabel(r"$C(t_2)$ ($\mathrm{cm}^{-2}$)")
            ax3.set_title(r"FFCF Time-Correlation Decay $C(t_2)$")
            ax3.legend(loc="best", fontsize=8)
            ax3.grid(True, linestyle="--", alpha=0.35)

        elif mode_idx == 3 and state is not None:
            # 4. 1st-Order FTIR Fit vs Experiment
            bands = state["bands"]
            decays = state["decays"]
            tmax = state["tmax"]
            nt = state["nt"] or 256
            ftir_sim = simulate_1d_ftir_spectrum(
                bands, decays, self.pump_axis, tmax=tmax, nt=nt, normalize=True
            )
            if self.ftir_spectrum is not None:
                w_f, y_f = self.ftir_spectrum
                y_interp = np.interp(self.pump_axis, w_f, y_f, left=0.0, right=0.0)
                ym = float(np.max(np.abs(y_interp)))
                if ym > 0:
                    y_interp = y_interp / ym
                scale_f = float(
                    np.maximum(0.0, np.sum(y_interp * ftir_sim) / (np.sum(ftir_sim**2) + 1e-12))
                )
                ax3.plot(
                    self.pump_axis,
                    y_interp,
                    "o",
                    color="#1e293b",
                    markersize=3.5,
                    alpha=0.75,
                    label="Experimental FTIR",
                )
                ax3.plot(
                    self.pump_axis,
                    scale_f * ftir_sim,
                    "-",
                    color="#16a34a",
                    linewidth=1.8,
                    label="Analytical Kubo 1D FTIR",
                )
            else:
                ax3.plot(
                    self.pump_axis,
                    ftir_sim,
                    "-",
                    color="#16a34a",
                    linewidth=1.8,
                    label="Simulated Kubo 1D Absorption",
                )
            ax3.set_xlabel(r"Frequency $\omega$ ($\mathrm{cm}^{-1}$)")
            ax3.set_ylabel("Normalized Absorbance")
            ax3.set_title("1st-Order Linear Absorption (FTIR)")
            ax3.legend(loc="best", fontsize=8)
            ax3.grid(True, linestyle="--", alpha=0.35)

    def plot_contour(self, ax, Z, vmin, vmax, title):
        """Render a 2D map on `ax` using `pm.twoD.plot_map` with fallback to `contourf`."""
        try:
            ds_tmp = Dataset2D(
                Z=np.asarray(Z, dtype=float)[:, :, None],
                pump=self.pump_axis,
                probe=self.probe_axis,
                delays=np.array([0.0]),
                units=getattr(self.dataset, "units", {}),
                freq_units=getattr(self.dataset, "freq_units", "cm-1"),
            )
            z_pct = float(abs(vmax) * 100.0)
            pm.twoD.plot_map(
                ds_tmp,
                t2=0.0,
                ax=ax,
                z_scale_pct=z_pct,
                add_colorbar=True,
                add_diagonal=True,
                title=title,
            )
            ax.set_aspect("equal")
            return
        except Exception:
            pass

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from pymorgan import helpers as hlp
        from ..settings import get_settings

        s = get_settings()
        cmap_ID = s.cmap
        white_levels = s.white_levels
        pump_axis_val = s.pump_axis
        if hasattr(pump_axis_val, "value"):
            pump_axis_val = pump_axis_val.value
        is_vertical = pump_axis_val == "Vertical"

        if is_vertical:
            X, Y, Z_contour = self.probe_axis, self.pump_axis, Z
        else:
            X, Y, Z_contour = self.pump_axis, self.probe_axis, Z.T

        NctrF = 30
        nw = int(white_levels) if white_levels else 2
        cm_obj, _ = hlp.CalcCMAP(cmap_ID, NctrF, Nwhite=nw)
        if white_levels:
            cm_obj = hlp.zero_center_cmap(cm_obj, NctrF, int(white_levels))

        ctrLvl_F = np.linspace(vmin, vmax, NctrF)
        cf = ax.contourf(
            X, Y, Z_contour, vmin=vmin, vmax=vmax, levels=ctrLvl_F, cmap=cm_obj, extend="neither"
        )
        ax.plot(self.pump_axis, self.pump_axis, "k--", linewidth=0.8, alpha=0.5)
        ax.set_title(title)
        ax.set_aspect("equal")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        self.fig.colorbar(cf, cax=cax)

    # ------------------------------------------------------------------ #
    #                  Diagnostics & SIGN Table Population               #
    # ------------------------------------------------------------------ #

    def _populate_diagnostics_from_result(self, res: dict[str, Any]):
        """Update Tab 3 summary labels and `self.diag_table` from a fit result dictionary."""
        self.last_fit_result = res
        r2 = res.get("r_squared", float("nan"))
        rmse = res.get("rmse", float("nan"))
        chi2 = res.get("reduced_chi2", float("nan"))
        nfev = res.get("nfev", 0)
        elapsed = res.get("elapsed_s", 0.0)

        if np.isfinite(r2):
            self.lbl_r2.setText(f"R²: {r2:.4f}")
        if np.isfinite(rmse):
            self.lbl_rmse.setText(f"RMSE: {rmse:.4e}")
        if np.isfinite(chi2):
            self.lbl_chi2.setText(f"Reduced χ²_ν: {chi2:.3e}")
        self.lbl_perf.setText(f"Evals / Time: {nfev} evals ({elapsed:.2f} s)")

        _, bounds, default_names = self._build_flat_params_bounds_names()
        names = res.get("param_names", default_names)
        params = np.asarray(res.get("params", []), dtype=float)
        errors = np.asarray(res.get("errors", np.zeros_like(params)), dtype=float)
        ci95 = np.asarray(res.get("ci95", 1.96 * errors), dtype=float)
        signs = np.asarray(res.get("sign", np.zeros_like(params)), dtype=float)

        n_rows = min(len(names), len(params))
        self.diag_table.setRowCount(n_rows)

        for r in range(n_rows):
            nm = names[r]
            val = float(params[r])
            err = float(errors[r]) if r < len(errors) else 0.0
            ci = float(ci95[r]) if r < len(ci95) else 0.0
            sg = float(signs[r]) if r < len(signs) else 0.0
            is_fixed = (r < len(bounds)) and abs(bounds[r][1] - bounds[r][0]) <= 1e-11
            is_amp = ("A_GSB" in nm) or ("A_ESA" in nm)

            if is_fixed:
                status = "Fixed / Linked"
                bg = "#f1f5f9"
                fg = "#475569"
            elif is_amp and self.chk_varpro.isChecked():
                status = "VarPro Exact"
                bg = "#dbeafe"
                fg = "#1d4ed8"
            elif sg >= -1.2 and (err == 0.0 or abs(err) < max(abs(val) * 0.5, 5.0)):
                status = "Optimal"
                bg = "#dcfce7"
                fg = "#15803d"
            elif sg >= -2.2:
                status = "Coupled (Δ·τc)"
                bg = "#fef3c7"
                fg = "#b45309"
            else:
                status = "Collinear / Weak"
                bg = "#fee2e2"
                fg = "#b91c1c"

            self.diag_table.setItem(r, 0, QTableWidgetItem(nm))
            self.diag_table.setItem(r, 1, QTableWidgetItem(f"{val:.4f}"))
            self.diag_table.setItem(r, 2, QTableWidgetItem("—" if is_fixed else f"±{err:.4f}"))
            self.diag_table.setItem(
                r, 3, QTableWidgetItem("—" if is_fixed else f"[{val - ci:.3f}, {val + ci:.3f}]")
            )
            self.diag_table.setItem(r, 4, QTableWidgetItem("—" if is_fixed else f"{sg:.2f}"))

            st_item = QTableWidgetItem(status)
            st_item.setBackground(QColor(bg))
            st_item.setForeground(QColor(fg))
            st_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.diag_table.setItem(r, 5, st_item)

    def _copy_diagnostics_latex(self):
        if self.diag_table.rowCount() == 0:
            return
        lines = [
            "\\begin{table}[htbp]",
            "  \\centering",
            "  \\caption{Direct Kubo Line-Shape Fit Parameters and Uncertainties}",
            "  \\begin{tabular}{lcccc}",
            "    \\toprule",
            "    Parameter & Fitted Value & $\\pm \\sigma$ & 95\\% CI & Robben SIGN \\\\",
            "    \\midrule",
        ]
        for r in range(self.diag_table.rowCount()):
            vals = [
                self.diag_table.item(r, c).text() if self.diag_table.item(r, c) else ""
                for c in range(5)
            ]
            lines.append("    " + " & ".join(vals) + " \\\\")
        lines.extend([
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
        ])
        QGuiApplication.clipboard().setText("\n".join(lines))
        QMessageBox.information(self, "Copied", "LaTeX table copied to clipboard.")

    def _copy_diagnostics_csv(self):
        if self.diag_table.rowCount() == 0:
            return
        lines = ["Parameter,Fitted_Value,Std_Err,CI_95,Robben_SIGN,Identifiability"]
        for r in range(self.diag_table.rowCount()):
            vals = [
                f'"{self.diag_table.item(r, c).text()}"' if self.diag_table.item(r, c) else ""
                for c in range(6)
            ]
            lines.append(",".join(vals))
        QGuiApplication.clipboard().setText("\n".join(lines))
        QMessageBox.information(self, "Copied", "CSV diagnostics table copied to clipboard.")

    # ------------------------------------------------------------------ #
    #                  Active-Delay & Multi-Delay Fitting                #
    # ------------------------------------------------------------------ #

    def _make_live_iteration_callback(self):
        """Return a throttled iteration callback that updates `ax3` trajectory live."""
        self.fit_history = []
        ffcf_struct = self._get_ffcf_structure()
        step_counter = [0]

        def iteration_callback(params):
            if not self.chk_auto_update_fit.isChecked():
                return
            step_counter[0] += 1
            from ..twoD.kubo_fit import _extract_first_kubo_pair

            pair = _extract_first_kubo_pair(
                params,
                self.num_bands,
                self.num_decays,
                has_dw0=True,
                ffcf_structure=ffcf_struct,
            )
            if not self.fit_history or (
                abs(self.fit_history[-1][0] - pair[0]) > 1e-5
                or abs(self.fit_history[-1][1] - pair[1]) > 1e-5
            ):
                self.fit_history.append(pair)

            if step_counter[0] % 2 != 0:
                return

            if hasattr(self, "diag_view_cb") and self.diag_view_cb.currentIndex() == 0:
                ax3 = self.axes[3]
                ax3.clear()
                if self.fit_history:
                    history_arr = np.asarray(self.fit_history, dtype=float)
                    ax3.plot(
                        history_arr[:, 1],
                        history_arr[:, 0],
                        "o-",
                        color="#d97706",
                        linewidth=1.5,
                        markersize=4,
                    )
                    ax3.plot(
                        history_arr[0, 1],
                        history_arr[0, 0],
                        "o",
                        color="#dc2626",
                        markerfacecolor="none",
                        markeredgewidth=1.5,
                        label="Start",
                    )
                    ax3.plot(
                        history_arr[-1, 1],
                        history_arr[-1, 0],
                        "x",
                        color="#16a34a",
                        markeredgewidth=1.5,
                        label="Current",
                    )
                    ax3.legend(loc="upper right", fontsize=8)
                ax3.set_xlabel(r"Correlation time $\tau_{c}$ (ps)")
                ax3.set_ylabel(r"Fluctuation $\Delta_{1}$ ($\mathrm{cm}^{-1}$)")
                ax3.set_title("Fitting Progress (Kubo coordinates)")
                self.canvas.draw_idle()
                QCoreApplication.processEvents()

        return iteration_callback

    def _notify_parent_fit_updated(self):
        parent = self.parent()
        if parent is not None:
            if hasattr(parent, "_update_spectral_diffusion_controls_state"):
                try:
                    parent._update_spectral_diffusion_controls_state()
                except Exception:
                    pass

    def run_active_fit(self):
        """Run Kubo optimization on the currently selected population delay t₂."""
        p0_full, bounds_full, param_names = self._build_flat_params_bounds_names()
        delay_idx = max(0, self.delay_cb.currentIndex())
        actual_idx = self.delay_indices[min(delay_idx, len(self.delay_indices) - 1)]
        t2 = float(self.dataset.delays[actual_idx])

        Z_exp_slice = self.Z_exp_roi[:, :, actual_idx : actual_idx + 1]
        tmax, nt, u, w_rot = self.get_ft_settings()
        solver, use_varpro, parallel_mode, workers = self._get_solver_and_parallel_config()
        band_links = self._get_band_links_config()
        ffcf_struct = self._get_ffcf_structure()
        ftir_weight = float(self.ftir_weight_spin.value())

        ic = self._make_live_iteration_callback()

        self.btn_fit_active.setEnabled(False)
        self.btn_fit_active.setText("Fitting...")
        QCoreApplication.processEvents()

        try:
            fit_results = run_kubo_fit(
                Z_exp_slice,
                self.pump_axis,
                self.probe_axis,
                [t2],
                p0_full,
                bounds_full,
                tmax=tmax,
                nt=nt,
                w_rot=w_rot,
                undersampling_factor=u,
                fid_data=self.fid_data,
                num_bands=self.num_bands,
                num_decays=self.num_decays,
                parallel_mode=parallel_mode,
                iteration_callback=ic,
                has_dw0=True,
                ffcf_structure=ffcf_struct,
                solver=solver,
                use_varpro=use_varpro,
                band_links=band_links,
                ftir_weight=ftir_weight,
                max_workers=workers,
            )
            res = fit_results[0]
            res["param_names"] = param_names

            self.fit_history = res.get("history", self.fit_history)
            self.last_fit_params = np.asarray(res["params"], dtype=float)

            # Store in dataset._sd_results["Kubo_Fit"]
            if not hasattr(self.dataset, "_sd_results"):
                self.dataset._sd_results = {}

            row = [t2] + list(res["params"])
            if (
                "Kubo_Fit" in self.dataset._sd_results
                and self.dataset._sd_results["Kubo_Fit"].ndim == 2
                and self.dataset._sd_results["Kubo_Fit"].shape[1] == len(row)
            ):
                arr = self.dataset._sd_results["Kubo_Fit"]
                idx_matches = np.where(np.isclose(arr[:, 0], t2))[0]
                if len(idx_matches) > 0:
                    arr[idx_matches[0]] = row
                else:
                    self.dataset._sd_results["Kubo_Fit"] = np.vstack([arr, row])
            else:
                self.dataset._sd_results["Kubo_Fit"] = np.array([row], dtype=float)

            if not hasattr(self.dataset, "_sd_kubo_details"):
                self.dataset._sd_kubo_details = {}

            self.dataset._sd_kubo_details[t2] = {
                "params": res["params"],
                "errors": res["errors"],
                "ci95": res.get("ci95"),
                "sign": res.get("sign"),
                "r_squared": res.get("r_squared"),
                "rmse": res.get("rmse"),
                "reduced_chi2": res.get("reduced_chi2"),
                "nfev": res.get("nfev"),
                "elapsed_s": res.get("elapsed_s"),
                "num_bands": self.num_bands,
                "num_decays": self.num_decays,
                "has_dw0": True,
                "ffcf_structure": ffcf_struct,
                "bands_fitted": res.get("bands_fitted"),
                "ffcf_fitted": res.get("ffcf_fitted"),
                "unit_scale": res.get("unit_scale", 1.0),
                "exp_max": res.get("exp_max", float(np.max(np.abs(Z_exp_slice)))),
                "param_names": param_names,
                "tmax": tmax,
                "nt": nt,
                "u": u,
                "w_rot": w_rot,
                "fid_data": self.fid_data,
            }

            # Update full simulated cube for this delay using exact ROI unit basis scale
            n_pump = len(self.dataset.pump)
            n_probe = len(self.dataset.probe)
            if getattr(self.dataset, "_kubo_fit_dataset", None) is not None:
                Z_sim_cube = np.array(self.dataset._kubo_fit_dataset.Z, dtype=float)
            else:
                Z_sim_cube = np.zeros((n_pump, n_probe, len(self.dataset.delays)), dtype=float)

            bands_fit = res["bands_fitted"]
            decays_fit = res["ffcf_fitted"]
            unit_scale = float(res.get("unit_scale", 1.0))
            exp_max = float(res.get("exp_max", np.max(np.abs(Z_exp_slice))))

            Z_sim_full_raw = simulate_2d_spectrum(
                bands_fit,
                decays_fit,
                self.dataset.pump,
                self.dataset.probe,
                t2,
                tmax=tmax,
                nt=nt,
                w_rot=w_rot,
                undersampling_factor=u,
                fid_data=self.fid_data,
                normalize_unit_basis=False,
            )
            Z_sim_cube[:, :, actual_idx] = (Z_sim_full_raw / max(unit_scale, 1e-12)) * exp_max

            self.dataset._kubo_fit_dataset = Dataset2D(
                Z_sim_cube,
                self.dataset.pump,
                self.dataset.probe,
                self.dataset.delays,
                self.dataset.units,
                self.dataset.freq_units,
                source="In-Memory Kubo Simulation",
                data_type="P2DAT",
                datatype="processed",
            )

            self._populate_diagnostics_from_result(self.dataset._sd_kubo_details[t2])
            if self.chk_auto_update_p0.isChecked():
                self.pfit_to_p0(show_dialog=False)
            self._notify_parent_fit_updated()
            self.update_plots()
        except Exception as exc:
            logger.error("Active delay Kubo fit failed", exc_info=True)
            QMessageBox.critical(self, "Fitting Failed", str(exc))
        finally:
            self.btn_fit_active.setEnabled(True)
            self.btn_fit_active.setText("▶ Fit Active Delay")

    def run_all_fit(self):
        """Run multi-core Kubo fitting across all selected delays (independent or global)."""
        p0_full, bounds_full, param_names = self._build_flat_params_bounds_names()
        fit_delays = self.dataset.delays[self.delay_indices]
        Z_exp_roi_sub = self.Z_exp_roi[:, :, self.delay_indices]
        if len(fit_delays) == 0:
            QMessageBox.warning(self, "Fit Error", "No delays selected in the specified t₂ range.")
            return

        tmax, nt, u, w_rot = self.get_ft_settings()
        solver, use_varpro, parallel_mode, workers = self._get_solver_and_parallel_config()
        band_links = self._get_band_links_config()
        ffcf_struct = self._get_ffcf_structure()
        ftir_weight = float(self.ftir_weight_spin.value())
        is_global = self.chk_global_fit.isChecked()

        max_prog = 50 if is_global else len(fit_delays)
        progress = QProgressDialog(
            "Running Global Multi-Core Kubo Fit..."
            if is_global
            else f"Fitting Kubo model across {len(fit_delays)} delays ({parallel_mode}, {workers} workers)...",
            "Cancel",
            0,
            max_prog,
            self,
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        QCoreApplication.processEvents()

        ic = self._make_live_iteration_callback()

        def on_progress(done: int, total: int, _res_item=None):
            progress.setMaximum(max(1, total))
            progress.setValue(min(done, total))
            QCoreApplication.processEvents()

        try:
            if is_global:
                fit_results = run_kubo_global_fit(
                    Z_exp_roi_sub,
                    self.pump_axis,
                    self.probe_axis,
                    fit_delays,
                    p0_full,
                    bounds_full,
                    tmax=tmax,
                    nt=nt,
                    w_rot=w_rot,
                    undersampling_factor=u,
                    fid_data=self.fid_data,
                    num_bands=self.num_bands,
                    num_decays=self.num_decays,
                    has_dw0=True,
                    ffcf_structure=ffcf_struct,
                    solver=solver,
                    use_varpro=use_varpro,
                    band_links=band_links,
                    ftir_weight=ftir_weight,
                    parallel_mode=parallel_mode,
                    max_workers=workers,
                    iteration_callback=ic,
                    progress_callback=on_progress,
                    should_cancel=progress.wasCanceled,
                )
            else:
                fit_results = run_kubo_fit(
                    Z_exp_roi_sub,
                    self.pump_axis,
                    self.probe_axis,
                    fit_delays,
                    p0_full,
                    bounds_full,
                    tmax=tmax,
                    nt=nt,
                    w_rot=w_rot,
                    undersampling_factor=u,
                    fid_data=self.fid_data,
                    num_bands=self.num_bands,
                    num_decays=self.num_decays,
                    parallel_mode=parallel_mode,
                    iteration_callback=ic,
                    has_dw0=True,
                    ffcf_structure=ffcf_struct,
                    solver=solver,
                    use_varpro=use_varpro,
                    band_links=band_links,
                    ftir_weight=ftir_weight,
                    max_workers=workers,
                    progress_callback=on_progress,
                    should_cancel=progress.wasCanceled,
                )
        except (InterruptedError, KeyboardInterrupt) as exc:
            progress.close()
            QMessageBox.warning(self, "Fitting Cancelled", str(exc))
            return
        except Exception as exc:
            progress.close()
            logger.error("Multi-delay Kubo fit failed", exc_info=True)
            QMessageBox.critical(self, "Fitting Failed", str(exc))
            return

        if fit_results:
            self.fit_history = fit_results[-1].get("history", self.fit_history)
            self.last_fit_params = np.asarray(fit_results[-1]["params"], dtype=float)

        if not hasattr(self.dataset, "_sd_results"):
            self.dataset._sd_results = {}
        if not hasattr(self.dataset, "_sd_kubo_details"):
            self.dataset._sd_kubo_details = {}

        rows = []
        for res in fit_results:
            t2_val = float(res["delay"])
            rows.append([t2_val] + list(res["params"]))
            self.dataset._sd_kubo_details[t2_val] = {
                "params": res["params"],
                "errors": res["errors"],
                "ci95": res.get("ci95"),
                "sign": res.get("sign"),
                "r_squared": res.get("r_squared"),
                "rmse": res.get("rmse"),
                "reduced_chi2": res.get("reduced_chi2"),
                "nfev": res.get("nfev"),
                "elapsed_s": res.get("elapsed_s"),
                "num_bands": self.num_bands,
                "num_decays": self.num_decays,
                "has_dw0": True,
                "ffcf_structure": ffcf_struct,
                "bands_fitted": res.get("bands_fitted"),
                "ffcf_fitted": res.get("ffcf_fitted"),
                "unit_scale": res.get("unit_scale", 1.0),
                "exp_max": res.get("exp_max", 1.0),
                "param_names": param_names,
                "tmax": tmax,
                "nt": nt,
                "u": u,
                "w_rot": w_rot,
                "fid_data": self.fid_data,
            }

        self.dataset._sd_results["Kubo_Fit"] = np.array(rows, dtype=float)

        # Reconstruct full 3D simulated cube in parallel across all delays
        progress.setLabelText("Reconstructing full 3D simulated cube in parallel...")
        progress.setMaximum(len(self.dataset.delays))
        progress.setValue(0)
        QCoreApplication.processEvents()

        try:
            Z_sim_cube = simulate_kubo_cube_parallel(
                self.dataset.delays,
                self.dataset._sd_kubo_details,
                self.dataset.pump,
                self.dataset.probe,
                self.pump_axis,
                self.probe_axis,
                self.Z_exp_roi,
                self.delay_indices,
                parallel_mode=parallel_mode,
                max_workers=workers,
                progress_callback=lambda d, tot: (
                    progress.setValue(d),
                    QCoreApplication.processEvents(),
                ),
                should_cancel=progress.wasCanceled,
            )
            self.dataset._kubo_fit_dataset = Dataset2D(
                Z_sim_cube,
                self.dataset.pump,
                self.dataset.probe,
                self.dataset.delays,
                self.dataset.units,
                self.dataset.freq_units,
                source="In-Memory Kubo Simulation",
                data_type="P2DAT",
                datatype="processed",
            )
        except (InterruptedError, KeyboardInterrupt):
            progress.close()
            return
        finally:
            progress.close()

        # Populate diagnostics for the currently selected delay
        active_delay_idx = max(0, self.delay_cb.currentIndex())
        active_t2 = float(
            self.dataset.delays[self.delay_indices[min(active_delay_idx, len(self.delay_indices) - 1)]]
        )
        for k_t2, det in self.dataset._sd_kubo_details.items():
            if np.isclose(float(k_t2), active_t2, atol=1e-6):
                self.last_fit_params = np.asarray(det["params"], dtype=float)
                self._populate_diagnostics_from_result(det)
                break

        if self.chk_auto_update_p0.isChecked():
            self.pfit_to_p0(show_dialog=False)

        self._notify_parent_fit_updated()
        self.update_plots()

        QMessageBox.information(
            self,
            "Fit Complete",
            f"Fit across {len(fit_delays)} delay(s) completed successfully!\n\n"
            f"Solver: {solver.upper()} | Parallel: {parallel_mode} ({workers} workers)\n\n"
            f"Method Reference:\n"
            f"Robben, K. C.; Cheatum, C. M. Least-Squares Fitting of Multidimensional Spectra "
            f"to Kubo Line-Shape Models. J. Phys. Chem. B 2021, 125, 46, 12876-12891.\n"
            f"DOI: 10.1021/acs.jpcb.1c08764",
        )

    # ------------------------------------------------------------------ #
    #                  Saving Fit Results & Reports                      #
    # ------------------------------------------------------------------ #

    def save_fit_results(self):
        """Export simulated P2DAT cube and companion TXT + LaTeX parameter report."""
        if not hasattr(self.dataset, "_sd_kubo_details") or not self.dataset._sd_kubo_details:
            QMessageBox.warning(
                self,
                "No Fit Results",
                "No fit results are available to save. Please run a fit first.",
            )
            return

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

        _, _, parallel_mode, workers = self._get_solver_and_parallel_config()

        if getattr(self.dataset, "_kubo_fit_dataset", None) is not None:
            Z_sim_cube = np.asarray(self.dataset._kubo_fit_dataset.Z, dtype=float)
        else:
            progress = QProgressDialog(
                "Simulating full fit spectrum for saving...",
                "Cancel",
                0,
                len(self.dataset.delays),
                self,
            )
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            QCoreApplication.processEvents()
            try:
                Z_sim_cube = simulate_kubo_cube_parallel(
                    self.dataset.delays,
                    self.dataset._sd_kubo_details,
                    self.dataset.pump,
                    self.dataset.probe,
                    self.pump_axis,
                    self.probe_axis,
                    self.Z_exp_roi,
                    self.delay_indices,
                    parallel_mode=parallel_mode,
                    max_workers=workers,
                    progress_callback=lambda d, _tot: (
                        progress.setValue(d),
                        QCoreApplication.processEvents(),
                    ),
                    should_cancel=progress.wasCanceled,
                )
            except Exception as exc:
                progress.close()
                QMessageBox.critical(
                    self, "Simulation Failed", f"Could not simulate full axes: {exc}"
                )
                return
            finally:
                progress.close()

        try:
            PUMP, PROBE = np.meshgrid(self.dataset.pump, self.dataset.probe, indexing="ij")
            pump_flat = PUMP.ravel(order="F")
            probe_flat = PROBE.ravel(order="F")

            n_rows = len(pump_flat) + 1
            n_cols = 2 + len(self.dataset.delays)

            out = np.zeros((n_rows, n_cols), dtype=float)
            out[0, 0] = 0.0
            out[0, 1] = 0.0
            out[0, 2:] = self.dataset.delays
            out[1:, 0] = pump_flat
            out[1:, 1] = probe_flat

            for i in range(len(self.dataset.delays)):
                out[1:, 2 + i] = Z_sim_cube[:, :, i].ravel(order="F")

            np.savetxt(path, out, delimiter=",", fmt="%.18g")

            txt_path = os.path.splitext(path)[0] + ".txt"
            is_global = self.chk_global_fit.isChecked()

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("Kubo Model Fitting Results & Parameter Diagnostics\n")
                f.write("==================================================\n")
                f.write(f"Source Dataset: {getattr(self.dataset, 'source', 'Unknown')}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(
                    f"Fit Type: {'Global Linked (VarPro Amplitudes)' if is_global else 'Single/Multi-Delay Optimisation'}\n\n"
                )

                first_delay = sorted(self.dataset._sd_kubo_details.keys())[0]
                first_details = self.dataset._sd_kubo_details[first_delay]
                param_names = first_details["param_names"]

                for t2, details in sorted(self.dataset._sd_kubo_details.items()):
                    f.write(
                        f"Population Delay (t2): {t2:.2f} ps  |  R² = {details.get('r_squared', float('nan')):.4f}  |  RMSE = {details.get('rmse', float('nan')):.4e}\n"
                    )
                    f.write("-" * 78 + "\n")
                    f.write(
                        f"{'Parameter':<28} | {'Value':<12} | {'Std Error':<12} | {'Robben SIGN':<12}\n"
                    )
                    f.write("-" * 78 + "\n")
                    params = details["params"]
                    errors = details["errors"]
                    signs = details.get("sign", np.zeros_like(params))
                    names = details["param_names"]
                    for n, val, err, sg in zip(names, params, errors, signs):
                        f.write(f"{n:<28} | {val:<12.5f} | {err:<12.5f} | {sg:<12.2f}\n")
                    f.write("\n")

                f.write("\nLaTeX Table Code (for Copy-Paste in SI/paper):\n")
                f.write("==================================================\n")
                f.write("\\begin{table}[htbp]\n")
                f.write("  \\centering\n")
                f.write("  \\caption{Fitted Kubo model parameters and standard errors}\n")
                latex_names = [
                    n.replace("_", "\\_")
                    .replace("ω", "$\\omega$")
                    .replace("τ", "$\\tau$")
                    .replace("Δ", "$\\Delta$")
                    .replace("Γ", "$\\Gamma$")
                    .replace("₁", "$_1$")
                    .replace("₀", "$_0$")
                    .replace("⁻¹", "$^{-1}$")
                    for n in param_names
                ]
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
                    for val, err in zip(details["params"], details["errors"]):
                        row_str += f" & {val:.4f} & {err:.4f}"
                    row_str += " \\\\\n"
                    f.write(row_str)
                f.write("    \\bottomrule\n")
                f.write("  \\end{tabular}\n")
                f.write("\\end{table}\n")

            QMessageBox.information(
                self,
                "Save Complete",
                f"Fit results successfully saved!\n\n"
                f"Simulated P2DAT: {path}\n"
                f"Parameters & Diagnostics TXT: {txt_path}",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", f"Failed to write output files: {exc}")

    def closeEvent(self, event):
        if hasattr(self, "canvas") and self.canvas is not None:
            idle_timer = getattr(self.canvas, "_idle_draw_cnt", None)
            if idle_timer is not None and hasattr(idle_timer, "stop"):
                try:
                    idle_timer.stop()
                except Exception:
                    pass
        super().closeEvent(event)


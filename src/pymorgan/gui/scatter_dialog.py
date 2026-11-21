"""Interactive dialog for multi-band scatter and dynamic background removal (1-D).

Provides the Option A 4-quadrant/4-tab dashboard:
- Controls sidebar: pre-zero window, single-scan processing + noise update,
  multi-band table (auto-detect, interactive SpanSelector pick, add/delete,
  per-band rank or coupling), and algorithm parameters.
- Tab 1 (2D Maps): Static Pre-Zero vs Advanced Clean vs Removed Scatter Diff.
- Tab 2 (Spectral Slices): Before vs After transient spectra via ``plot_spectra``.
- Tab 3 (Kinetics): Before vs After wing vs core kinetics via ``plot_kinetics``.
- Tab 4 (Scatter Diagnostics): Pre-zero background via ``plot_background`` with
  drag-to-pick ``SpanSelector``, basis shapes, and delay trajectories a(t)/dl(t).
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import SpanSelector
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import pymorgan as pm
from pymorgan import helpers as hlp
from pymorgan.gui.canvas import MplCanvas, attach_contour_scroll_zoom
from pymorgan.gui.theme import is_dark_palette, style_figure
from pymorgan.log import get_logger
from pymorgan.oneD.dataset import Dataset1D
from pymorgan.oneD.plot import (
    _CBAR_PAD_IN,
    _CBAR_WIDTH_IN,
    _apply_time_xscale,
    _resolve_x_axis,
)
from pymorgan.oneD.scatter import (
    ScatterBand,
    ScatterRemovalResult,
    _band_widths,
    _resolve_pre_zero_window,
    compute_scatter_removal,
)

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

_MODE_ITEMS = [
    ("Auto", None, None),
    ("Rank 1", 1, None),
    ("Rank 2", 2, None),
    ("Rank 3", 3, None),
    ("Rank 4", 4, None),
    ("Linked → #1", None, 0),
]


class ScatterRemovalDialog(QDialog):
    """Interactive multi-band scatter and dynamic background removal dialog."""

    scatter_applied = pyqtSignal(object)
    scatter_reset = pyqtSignal()

    def __init__(
        self,
        parent: QWidget | None,
        dataset: Dataset1D,
        *,
        tmin: float | None = None,
        tmax: float | None = None,
        detector: int = 0,
    ):
        super().__init__(parent)
        self.dataset = dataset
        self.result: ScatterRemovalResult | None = None
        self._block_signals = False
        self._dirty_tabs = [True, True, True, True]
        self._span_selector: SpanSelector | None = None
        self._updating_table = False

        src = getattr(dataset, "source", None)
        title_name = src if src else "Dataset"
        self.setWindowTitle(f"Advanced Scatter & Background Removal \u2014 {title_name}")

        screen = QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            w = min(1280, max(960, int(avail.width() * 0.88)))
            h = min(860, max(660, int(avail.height() * 0.88)))
            self.resize(w, h)
        else:
            self.resize(1180, 800)
        self.setMinimumSize(940, 640)

        self._init_det = int(np.clip(detector, 0, max(0, getattr(dataset, "n_detectors", 1) - 1)))

        # Resolve initial parameters (prefer previously applied parameters on dataset)
        prev = getattr(dataset, "_scatter_removal_params", None) or {}
        delays = np.asarray(dataset.delays, dtype=float)
        fin_d = delays[np.isfinite(delays)]
        self._d_lo = float(np.min(fin_d)) if fin_d.size else -10.0
        self._d_hi = float(np.max(fin_d)) if fin_d.size else 10.0

        t_lo_def, t_hi_def, _ = _resolve_pre_zero_window(
            delays,
            prev.get("tmin", tmin),
            prev.get("tmax", tmax),
        )
        self._init_tmin = t_lo_def
        self._init_tmax = t_hi_def
        self._init_prev = prev

        # Debounce timer for live re-computation
        self._recompute_timer = QTimer(self)
        self._recompute_timer.setSingleShot(True)
        self._recompute_timer.setInterval(90)
        self._recompute_timer.timeout.connect(self._recompute_and_render)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        main_layout.addWidget(self.splitter)

        # 1. Left Controls Sidebar (Scrollable)
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls_scroll.setMinimumWidth(350)
        controls_scroll.setMaximumWidth(420)

        controls_widget = QWidget()
        self.ctrl_layout = QVBoxLayout(controls_widget)
        self.ctrl_layout.setContentsMargins(6, 6, 6, 6)
        self.ctrl_layout.setSpacing(8)
        self._build_controls()
        controls_scroll.setWidget(controls_widget)
        self.splitter.addWidget(controls_scroll)

        # 2. Right Visualization Panel
        right_panel = QWidget(self)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(2, 2, 2, 2)
        right_layout.setSpacing(6)
        self._build_right_panel(right_layout)
        self.splitter.addWidget(right_panel)

        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([375, 840])

        # Populate bands (either previous or auto-detect) and compute initial result
        self._load_initial_bands()
        self._recompute_and_render()

    # ------------------------------------------------------------------ #
    #                         Unit Helpers                               #
    # ------------------------------------------------------------------ #
    def _native_unit(self) -> str:
        if hlp.is_uncalibrated_probe(self.dataset):
            return "px"
        return hlp.native_x_unit(self.dataset.units)

    def _display_unit(self) -> str:
        nat = self._native_unit()
        if nat == "px":
            return "px"
        xu = pm.get_settings().x_axis_unit.value
        return hlp.resolve_x_unit(xu, nat)

    def _to_display(self, values):
        nat = self._native_unit()
        disp = self._display_unit()
        arr = np.asarray(values, dtype=float)
        if nat == "px" or nat == disp:
            return arr
        return hlp.convert_spectral(arr, nat, disp)

    def _to_native(self, values):
        nat = self._native_unit()
        disp = self._display_unit()
        arr = np.asarray(values, dtype=float)
        if nat == "px" or nat == disp:
            return arr
        return hlp.convert_spectral(arr, disp, nat)

    def _active_detector(self) -> int:
        if hasattr(self, "det_combo") and self.det_combo.count() > 0:
            data = self.det_combo.currentData()
            if data is not None:
                return int(data)
        return self._init_det

    def _panel_settings(self):
        return replace(pm.get_settings(), legend_location="best", secondary_axis=False)

    # ------------------------------------------------------------------ #
    #                           UI Builders                              #
    # ------------------------------------------------------------------ #
    def _build_controls(self):
        hdr = QLabel("<b>Advanced Scatter Removal</b>")
        hdr.setStyleSheet("font-size: 12.5pt; margin-bottom: 1px;")
        self.ctrl_layout.addWidget(hdr)

        desc = QLabel(
            "Separates localized pump scatter from broadband background and solves "
            "joint scatter + smooth TA fits per delay (and per scan when available)."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #64748b; font-size: 8.5pt;")
        self.ctrl_layout.addWidget(desc)

        # --- Group 1: Pre-Zero & Scope ---
        grp_scope = QGroupBox("1. Pre-Zero & Scope")
        v_scope = QVBoxLayout(grp_scope)
        v_scope.setSpacing(5)

        n_det = max(1, int(getattr(self.dataset, "n_detectors", 1)))
        self.det_row = QWidget()
        h_det = QHBoxLayout(self.det_row)
        h_det.setContentsMargins(0, 0, 0, 0)
        h_det.addWidget(QLabel("Detector:"))
        self.det_combo = QComboBox()
        for d in range(n_det):
            self.det_combo.addItem(f"Detector {d + 1}", d)
        self.det_combo.setCurrentIndex(self._init_det)
        h_det.addWidget(self.det_combo)
        v_scope.addWidget(self.det_row)
        self.det_row.setVisible(n_det > 1)

        t_unit = str(self.dataset.units.get("unitsT_ltx", "ps")).replace("$", "").strip() or "ps"
        h_pre = QHBoxLayout()
        h_pre.addWidget(QLabel(f"Pre-zero ({t_unit}):"))
        self.spin_tmin = QDoubleSpinBox()
        self.spin_tmin.setRange(self._d_lo - 10.0, self._d_hi)
        self.spin_tmin.setDecimals(3)
        self.spin_tmin.setValue(self._init_tmin)
        self.spin_tmax = QDoubleSpinBox()
        self.spin_tmax.setRange(self._d_lo, self._d_hi + 10.0)
        self.spin_tmax.setDecimals(3)
        self.spin_tmax.setValue(self._init_tmax)
        h_pre.addWidget(self.spin_tmin)
        h_pre.addWidget(QLabel("to"))
        h_pre.addWidget(self.spin_tmax)
        v_scope.addLayout(h_pre)

        has_ss = bool(getattr(self.dataset, "has_single_scans", False))
        n_ss = (
            int(self.dataset.Zss_R.shape[3])
            if has_ss and getattr(self.dataset.Zss_R, "ndim", 0) == 4
            else 0
        )
        ss_txt = (
            f"Process single scans first ({n_ss} scans)"
            if has_ss
            else "Process single scans first (unavailable)"
        )
        self.chk_per_scan = QCheckBox(ss_txt)
        self.chk_per_scan.setEnabled(has_ss)
        self.chk_per_scan.setChecked(
            bool(self._init_prev.get("per_scan", True)) and has_ss
        )
        v_scope.addWidget(self.chk_per_scan)

        self.chk_update_noise = QCheckBox("Update noise (Zstdv) from cleaned scans")
        self.chk_update_noise.setChecked(
            bool(self._init_prev.get("update_noise", True)) and has_ss
        )
        self.chk_update_noise.setEnabled(self.chk_per_scan.isChecked() and n_ss > 1)
        v_scope.addWidget(self.chk_update_noise)

        self.ctrl_layout.addWidget(grp_scope)

        # --- Group 2: Scatter Bands ---
        disp_u = self._display_unit()
        nat_u = self._native_unit()
        grp_bands = QGroupBox(f"2. Scatter Band(s) ({disp_u})")
        v_bands = QVBoxLayout(grp_bands)
        v_bands.setSpacing(5)

        self.band_table = QTableWidget(0, 5)
        self.band_table.setHorizontalHeaderLabels(["", "Band", "Min", "Max", "Mode"])
        hdr_tbl = self.band_table.horizontalHeader()
        hdr_tbl.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr_tbl.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hdr_tbl.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hdr_tbl.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        hdr_tbl.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.band_table.verticalHeader().setVisible(False)
        self.band_table.setMinimumHeight(115)
        self.band_table.setMaximumHeight(155)
        v_bands.addWidget(self.band_table)

        h_band_btns = QHBoxLayout()
        self.btn_auto_detect = QPushButton("Auto-Detect")
        self.btn_pick_band = QPushButton("Pick on Plot")
        self.btn_pick_band.setCheckable(True)
        self.btn_pick_band.setToolTip(
            "Switch to Scatter Diagnostics and drag across the pre-zero spectrum to add a band"
        )
        self.btn_add_band = QPushButton("+")
        self.btn_add_band.setFixedWidth(28)
        self.btn_add_band.setToolTip("Add a scatter band")
        self.btn_del_band = QPushButton("\u2212")
        self.btn_del_band.setFixedWidth(28)
        self.btn_del_band.setToolTip("Remove selected scatter band")
        h_band_btns.addWidget(self.btn_auto_detect)
        h_band_btns.addWidget(self.btn_pick_band)
        h_band_btns.addWidget(self.btn_add_band)
        h_band_btns.addWidget(self.btn_del_band)
        v_bands.addLayout(h_band_btns)

        h_wt = QHBoxLayout()
        h_wt.addWidget(QLabel(f"Wing ({nat_u}):"))
        self.spin_wing = QDoubleSpinBox()
        self.spin_wing.setRange(0.0, 5000.0)
        self.spin_wing.setDecimals(2)
        self.spin_wing.setSpecialValueText("Auto")
        self.spin_wing.setValue(0.0)
        self.spin_wing.setToolTip("0 = automatic wing width (~0.5x band width)")
        h_wt.addWidget(self.spin_wing)

        h_wt.addWidget(QLabel(f"Taper ({nat_u}):"))
        self.spin_taper = QDoubleSpinBox()
        self.spin_taper.setRange(0.0, 5000.0)
        self.spin_taper.setDecimals(2)
        self.spin_taper.setSpecialValueText("Auto")
        self.spin_taper.setValue(0.0)
        self.spin_taper.setToolTip("0 = automatic cosine edge taper (~0.25x band width)")
        h_wt.addWidget(self.spin_taper)
        v_bands.addLayout(h_wt)

        self.chk_allow_pos = QCheckBox("Also auto-detect positive artifacts")
        self.chk_allow_pos.setChecked(bool(self._init_prev.get("allow_positive", False)))
        v_bands.addWidget(self.chk_allow_pos)

        self.ctrl_layout.addWidget(grp_bands)

        # --- Group 3: Algorithm ---
        grp_alg = QGroupBox("3. Algorithm")
        grid_alg = QGridLayout(grp_alg)
        grid_alg.setSpacing(5)

        grid_alg.addWidget(QLabel("Method:"), 0, 0)
        self.combo_method = QComboBox()
        self.combo_method.addItem("Adaptive Local SVD", "local_svd")
        self.combo_method.addItem("Pre-Zero Template + Deriv.", "taylor")
        self.combo_method.addItem("Warped Pre-Zero Template", "warped")
        prev_method = self._init_prev.get("method", "local_svd")
        idx_m = self.combo_method.findData(prev_method)
        if idx_m >= 0:
            self.combo_method.setCurrentIndex(idx_m)
        grid_alg.addWidget(self.combo_method, 0, 1, 1, 3)

        grid_alg.addWidget(QLabel("Rank / Order:"), 1, 0)
        self.spin_rank = QSpinBox()
        self.spin_rank.setRange(1, 5)
        self.spin_rank.setValue(int(self._init_prev.get("rank", 3)))
        grid_alg.addWidget(self.spin_rank, 1, 1)

        grid_alg.addWidget(QLabel("TA poly:"), 1, 2)
        self.spin_poly = QSpinBox()
        self.spin_poly.setRange(0, 3)
        self.spin_poly.setValue(int(self._init_prev.get("poly_deg", 2)))
        grid_alg.addWidget(self.spin_poly, 1, 3)

        self.lbl_t0_excl = QLabel(f"Exclude t\u2080 ({t_unit}):")
        grid_alg.addWidget(self.lbl_t0_excl, 2, 0)
        t0_pair = self._init_prev.get("t0_exclude", (-0.5, 1.5))
        self.spin_t0_min = QDoubleSpinBox()
        self.spin_t0_min.setRange(-1000.0, 1000.0)
        self.spin_t0_min.setDecimals(2)
        self.spin_t0_min.setValue(float(t0_pair[0]))
        self.spin_t0_max = QDoubleSpinBox()
        self.spin_t0_max.setRange(-1000.0, 1000.0)
        self.spin_t0_max.setDecimals(2)
        self.spin_t0_max.setValue(float(t0_pair[1]))
        grid_alg.addWidget(self.spin_t0_min, 2, 1)
        grid_alg.addWidget(QLabel("to"), 2, 2)
        grid_alg.addWidget(self.spin_t0_max, 2, 3)

        self.chk_broadband = QCheckBox("Scale broadband bkg. by slow envelope a(t)")
        self.chk_broadband.setChecked(bool(self._init_prev.get("scale_broadband", True)))
        grid_alg.addWidget(self.chk_broadband, 3, 0, 1, 4)

        self.chk_transmittance = QCheckBox("Exact transmittance space (10^-\u0394A)")
        self.chk_transmittance.setChecked(bool(self._init_prev.get("transmittance_space", False)))
        grid_alg.addWidget(self.chk_transmittance, 4, 0, 1, 4)

        self.chk_mask_sat = QCheckBox("Mask saturated >")
        self.chk_mask_sat.setChecked(bool(self._init_prev.get("mask_saturated", False)))
        self.spin_sat_thresh = QDoubleSpinBox()
        self.spin_sat_thresh.setRange(1.0, 10000.0)
        self.spin_sat_thresh.setDecimals(1)
        self.spin_sat_thresh.setValue(float(self._init_prev.get("saturation_threshold", 500.0)))
        self.spin_sat_thresh.setEnabled(self.chk_mask_sat.isChecked())
        grid_alg.addWidget(self.chk_mask_sat, 5, 0, 1, 2)
        grid_alg.addWidget(self.spin_sat_thresh, 5, 2, 1, 2)

        self.ctrl_layout.addWidget(grp_alg)

        # --- Group 4: Actions ---
        grp_act = QGroupBox("4. Actions")
        v_act = QVBoxLayout(grp_act)
        v_act.setSpacing(5)

        self.btn_apply = QPushButton("Apply to Workspace")
        self.btn_apply.setStyleSheet(
            "font-weight: bold; background-color: #2563eb; color: white; padding: 6px; border-radius: 4px;"
        )
        v_act.addWidget(self.btn_apply)

        h_sub_act = QHBoxLayout()
        self.btn_reset = QPushButton("Reset to Static Bkg.")
        self.btn_close = QPushButton("Close")
        h_sub_act.addWidget(self.btn_reset)
        h_sub_act.addWidget(self.btn_close)
        v_act.addLayout(h_sub_act)

        self.ctrl_layout.addWidget(grp_act)
        self.ctrl_layout.addStretch()

        # Connect sidebar signals
        self.det_combo.currentIndexChanged.connect(self._on_detector_changed)
        self.spin_tmin.valueChanged.connect(self._schedule_recompute)
        self.spin_tmax.valueChanged.connect(self._schedule_recompute)
        self.chk_per_scan.toggled.connect(self._on_per_scan_toggled)
        self.chk_update_noise.toggled.connect(self._schedule_recompute)

        self.btn_auto_detect.clicked.connect(self._on_auto_detect_clicked)
        self.btn_pick_band.toggled.connect(self._on_pick_toggled)
        self.btn_add_band.clicked.connect(self._on_add_band_clicked)
        self.btn_del_band.clicked.connect(self._on_delete_band_clicked)
        self.band_table.itemChanged.connect(self._on_table_item_changed)
        self.band_table.currentCellChanged.connect(self._on_table_row_selected)

        self.spin_wing.valueChanged.connect(self._schedule_recompute)
        self.spin_taper.valueChanged.connect(self._schedule_recompute)
        self.chk_allow_pos.toggled.connect(self._on_auto_detect_clicked)

        self.combo_method.currentIndexChanged.connect(self._on_method_changed)
        self.spin_rank.valueChanged.connect(self._schedule_recompute)
        self.spin_poly.valueChanged.connect(self._schedule_recompute)
        self.spin_t0_min.valueChanged.connect(self._schedule_recompute)
        self.spin_t0_max.valueChanged.connect(self._schedule_recompute)
        self.chk_broadband.toggled.connect(self._schedule_recompute)
        self.chk_transmittance.toggled.connect(self._schedule_recompute)
        self.chk_mask_sat.toggled.connect(self._on_mask_sat_toggled)
        self.spin_sat_thresh.valueChanged.connect(self._schedule_recompute)

        self.btn_apply.clicked.connect(self._on_apply_clicked)
        self.btn_reset.clicked.connect(self._on_reset_clicked)
        self.btn_close.clicked.connect(self.reject)

        self._on_method_changed()

    def _build_right_panel(self, right_layout: QVBoxLayout):
        # Top shared view bar
        h_bar = QHBoxLayout()
        h_bar.setContentsMargins(2, 0, 2, 0)
        h_bar.addWidget(QLabel("Zoom:"))
        self.combo_zoom = QComboBox()
        self.combo_zoom.setMinimumWidth(190)
        h_bar.addWidget(self.combo_zoom)

        h_bar.addSpacing(12)
        self.chk_asinh = QCheckBox("Asinh")
        self.chk_asinh.setChecked(True)
        h_bar.addWidget(self.chk_asinh)

        h_bar.addWidget(QLabel("Zscale %:"))
        self.spin_zscale = QDoubleSpinBox()
        self.spin_zscale.setRange(0.1, 100.0)
        self.spin_zscale.setDecimals(1)
        self.spin_zscale.setSingleStep(5.0)
        self.spin_zscale.setValue(50.0)
        h_bar.addWidget(self.spin_zscale)
        h_bar.addStretch()
        right_layout.addLayout(h_bar)

        # 4 Tabs (Option A)
        self.tabs = QTabWidget(self)

        # Tab 1: 2D Maps
        self.canvas_maps = MplCanvas(self, show_toolbar=True)
        self.tabs.addTab(self.canvas_maps, "2D Maps")

        # Tab 2: Spectral Slices
        tab_spec = QWidget()
        v_spec = QVBoxLayout(tab_spec)
        v_spec.setContentsMargins(4, 4, 4, 4)
        h_spec_ctrl = QHBoxLayout()
        t_unit = str(self.dataset.units.get("unitsT_ltx", "ps")).replace("$", "").strip() or "ps"
        h_spec_ctrl.addWidget(QLabel(f"Slice delays ({t_unit}):"))
        self.edit_spec_delays = QLineEdit(self._default_slice_delays_str())
        btn_spec_go = QPushButton("Update")
        h_spec_ctrl.addWidget(self.edit_spec_delays)
        h_spec_ctrl.addWidget(btn_spec_go)
        v_spec.addLayout(h_spec_ctrl)
        self.canvas_spec = MplCanvas(tab_spec, show_toolbar=True)
        v_spec.addWidget(self.canvas_spec)
        self.tabs.addTab(tab_spec, "Spectral Slices")

        # Tab 3: Kinetics
        tab_kin = QWidget()
        v_kin = QVBoxLayout(tab_kin)
        v_kin.setContentsMargins(4, 4, 4, 4)
        h_kin_ctrl = QHBoxLayout()
        h_kin_ctrl.addWidget(QLabel(f"Probe positions ({self._display_unit()}):"))
        self.edit_kin_pos = QLineEdit("")
        btn_kin_band = QPushButton("From Focused Band")
        h_kin_ctrl.addWidget(self.edit_kin_pos)
        h_kin_ctrl.addWidget(btn_kin_band)
        v_kin.addLayout(h_kin_ctrl)
        self.canvas_kin = MplCanvas(tab_kin, show_toolbar=True)
        v_kin.addWidget(self.canvas_kin)
        self.tabs.addTab(tab_kin, "Kinetics")

        # Tab 4: Scatter Diagnostics
        self.canvas_diag = MplCanvas(self, show_toolbar=True)
        self.tabs.addTab(self.canvas_diag, "Scatter Diagnostics")

        right_layout.addWidget(self.tabs, stretch=1)

        # Live quality readout strip
        self.lbl_metrics = QLabel("Ready.")
        self.lbl_metrics.setWordWrap(True)
        self.lbl_metrics.setStyleSheet(
            "padding: 5px 8px; border: 1px solid #94a3b8; border-radius: 4px; font-size: 8.5pt;"
        )
        right_layout.addWidget(self.lbl_metrics)

        # Connections
        self.combo_zoom.currentIndexChanged.connect(self._on_view_changed)
        self.chk_asinh.toggled.connect(self._on_view_changed)
        self.spin_zscale.valueChanged.connect(self._on_view_changed)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.edit_spec_delays.editingFinished.connect(lambda: self._render_tab(1))
        btn_spec_go.clicked.connect(lambda: self._render_tab(1))
        self.edit_kin_pos.editingFinished.connect(lambda: self._render_tab(2))
        btn_kin_band.clicked.connect(self._sync_kinetics_from_focused_band)

    # ------------------------------------------------------------------ #
    #                         Band Table State                           #
    # ------------------------------------------------------------------ #
    def _default_slice_delays_str(self) -> str:
        d = np.sort(np.asarray(self.dataset.delays, dtype=float))
        d = d[np.isfinite(d)]
        if d.size == 0:
            return "1, 10"
        picks = [float(d[0])]
        for cand in (10.0, 100.0, 500.0, 1500.0):
            if d[0] < cand <= d[-1] * 1.05:
                near = float(d[np.argmin(np.abs(d - cand))])
                if not any(abs(near - x) < 1e-6 for x in picks):
                    picks.append(near)
        if len(picks) < 3 and d[-1] > 0:
            for frac in (0.3, 0.7, 1.0):
                near = float(d[min(d.size - 1, int(frac * (d.size - 1)))])
                if not any(abs(near - x) < 1e-6 for x in picks):
                    picks.append(near)
        return ", ".join(f"{v:g}" for v in picks[:5])

    def _load_initial_bands(self):
        prev_bands = self._init_prev.get("bands") if self._init_prev else None
        if prev_bands is not None:
            bands = [replace(b) for b in prev_bands]
            wing0 = bands[0].wing_width if bands and bands[0].wing_width else 0.0
            taper0 = bands[0].taper_width if bands and bands[0].taper_width else 0.0
            self._block_signals = True
            self.spin_wing.setValue(float(wing0 or 0.0))
            self.spin_taper.setValue(float(taper0 or 0.0))
            self._block_signals = False
        else:
            bands = self.dataset.detect_scatter_bands(
                detector=self._active_detector(),
                tmin=self.spin_tmin.value(),
                tmax=self.spin_tmax.value(),
                allow_positive=self.chk_allow_pos.isChecked(),
            )
        self._set_table_bands(bands)
        self._sync_kinetics_from_focused_band(render=False)

    def _set_table_bands(self, bands: list[ScatterBand]):
        self._updating_table = True
        self.band_table.blockSignals(True)
        try:
            self.band_table.setRowCount(0)
            for row, b in enumerate(bands):
                self.band_table.insertRow(row)

                chk_item = QTableWidgetItem()
                chk_item.setFlags(
                    Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                )
                chk_item.setCheckState(
                    Qt.CheckState.Checked if b.enabled else Qt.CheckState.Unchecked
                )
                self.band_table.setItem(row, 0, chk_item)

                lbl_text = b.label or f"Band {row + 1}"
                lbl_item = QTableWidgetItem(lbl_text)
                self.band_table.setItem(row, 1, lbl_item)

                d_vals = self._to_display([b.p_min, b.p_max])
                d_lo, d_hi = float(min(d_vals)), float(max(d_vals))
                self.band_table.setItem(row, 2, QTableWidgetItem(f"{d_lo:.2f}"))
                self.band_table.setItem(row, 3, QTableWidgetItem(f"{d_hi:.2f}"))

                mode_cbx = QComboBox()
                for txt, r_val, c_val in _MODE_ITEMS:
                    if c_val is not None and row == 0:
                        continue
                    mode_cbx.addItem(txt, (r_val, c_val))
                target_data = (b.rank, b.couple_to)
                sel_idx = 0
                for idx in range(mode_cbx.count()):
                    if mode_cbx.itemData(idx) == target_data:
                        sel_idx = idx
                        break
                mode_cbx.setCurrentIndex(sel_idx)
                mode_cbx.currentIndexChanged.connect(self._schedule_recompute)
                self.band_table.setCellWidget(row, 4, mode_cbx)
        finally:
            self.band_table.blockSignals(False)
            self._updating_table = False

        if self.band_table.rowCount() > 0:
            self.band_table.selectRow(0)
        self._refresh_zoom_combo(select_first_band=True)

    def _read_table_bands(self) -> list[ScatterBand]:
        bands: list[ScatterBand] = []
        wing_val = float(self.spin_wing.value())
        taper_val = float(self.spin_taper.value())
        wing_opt = wing_val if wing_val > 0.0 else None
        taper_opt = taper_val if taper_val > 0.0 else None

        for row in range(self.band_table.rowCount()):
            chk_item = self.band_table.item(row, 0)
            lbl_item = self.band_table.item(row, 1)
            min_item = self.band_table.item(row, 2)
            max_item = self.band_table.item(row, 3)
            mode_cbx = self.band_table.cellWidget(row, 4)

            enabled = (
                chk_item is not None and chk_item.checkState() == Qt.CheckState.Checked
            )
            label = lbl_item.text().strip() if lbl_item is not None else f"Band {row + 1}"
            try:
                v1 = float(min_item.text()) if min_item is not None else 0.0
                v2 = float(max_item.text()) if max_item is not None else 0.0
            except ValueError:
                continue
            nat = self._to_native([v1, v2])
            p_lo, p_hi = float(np.min(nat)), float(np.max(nat))

            rank_val, couple_val = (None, None)
            if isinstance(mode_cbx, QComboBox):
                data = mode_cbx.currentData()
                if data is not None:
                    rank_val, couple_val = data

            bands.append(
                ScatterBand(
                    p_min=p_lo,
                    p_max=p_hi,
                    wing_width=wing_opt,
                    taper_width=taper_opt,
                    enabled=enabled,
                    label=label,
                    rank=rank_val,
                    couple_to=couple_val,
                )
            )
        return bands

    def _refresh_zoom_combo(self, *, select_first_band: bool = False):
        prev_data = self.combo_zoom.currentData()
        bands = self._read_table_bands()
        self.combo_zoom.blockSignals(True)
        try:
            self.combo_zoom.clear()
            self.combo_zoom.addItem("Full Range", None)
            for idx, b in enumerate(bands):
                d_vals = self._to_display([b.p_min, b.p_max])
                lo, hi = float(min(d_vals)), float(max(d_vals))
                lbl = b.label or f"Band {idx + 1}"
                self.combo_zoom.addItem(f"#{idx + 1}: {lbl} [{lo:.1f}..{hi:.1f}]", idx)

            if select_first_band and bands:
                # Focus Band 0 on initial load so scatter detail is immediately legible
                target_idx = self.combo_zoom.findData(0)
                if target_idx >= 0:
                    self.combo_zoom.setCurrentIndex(target_idx)
            elif prev_data is not None:
                target_idx = self.combo_zoom.findData(prev_data)
                if target_idx >= 0:
                    self.combo_zoom.setCurrentIndex(target_idx)
        finally:
            self.combo_zoom.blockSignals(False)

    # ------------------------------------------------------------------ #
    #                         Slots & Events                             #
    # ------------------------------------------------------------------ #
    def _schedule_recompute(self, *_args):
        if self._block_signals or self._updating_table:
            return
        self._recompute_timer.start()

    def _on_detector_changed(self):
        self._on_auto_detect_clicked()

    def _on_per_scan_toggled(self, checked: bool):
        n_ss = (
            int(self.dataset.Zss_R.shape[3])
            if getattr(self.dataset, "has_single_scans", False)
            and getattr(self.dataset.Zss_R, "ndim", 0) == 4
            else 0
        )
        self.chk_update_noise.setEnabled(checked and n_ss > 1)
        self._schedule_recompute()

    def _on_method_changed(self):
        is_svd = self.combo_method.currentData() == "local_svd"
        self.spin_t0_min.setEnabled(is_svd)
        self.spin_t0_max.setEnabled(is_svd)
        self.lbl_t0_excl.setEnabled(is_svd)
        self._schedule_recompute()

    def _on_mask_sat_toggled(self, checked: bool):
        self.spin_sat_thresh.setEnabled(checked)
        self._schedule_recompute()

    def _on_auto_detect_clicked(self):
        bands = self.dataset.detect_scatter_bands(
            detector=self._active_detector(),
            tmin=self.spin_tmin.value(),
            tmax=self.spin_tmax.value(),
            allow_positive=self.chk_allow_pos.isChecked(),
        )
        self._set_table_bands(bands)
        self._sync_kinetics_from_focused_band(render=False)
        self._recompute_and_render()

    def _on_pick_toggled(self, checked: bool):
        if checked:
            self.tabs.setCurrentIndex(3)
        if self._span_selector is not None:
            self._span_selector.set_active(checked)

    def _on_span_picked(self, vmin: float, vmax: float):
        lo, hi = min(float(vmin), float(vmax)), max(float(vmin), float(vmax))
        if hi <= lo:
            return
        nat = self._to_native([lo, hi])
        p_lo, p_hi = float(np.min(nat)), float(np.max(nat))
        bands = self._read_table_bands()
        center = 0.5 * (p_lo + p_hi)
        bands.append(
            ScatterBand(
                p_min=p_lo,
                p_max=p_hi,
                enabled=True,
                label=f"Custom ({center:.1f})",
            )
        )
        self._set_table_bands(bands)
        self.btn_pick_band.setChecked(False)
        self._recompute_and_render()

    def _on_add_band_clicked(self):
        probe = np.sort(self.dataset._detector_probe(self._active_detector()))
        if probe.size >= 2:
            mid = float(np.median(probe))
            half = max(float(np.ptp(probe)) * 0.03, float(np.median(np.abs(np.diff(probe)))) * 4.0)
            p_lo, p_hi = max(float(probe[0]), mid - half), min(float(probe[-1]), mid + half)
        else:
            p_lo, p_hi = 350.0, 360.0
        bands = self._read_table_bands()
        bands.append(ScatterBand(p_min=p_lo, p_max=p_hi, enabled=True, label="Custom"))
        self._set_table_bands(bands)
        self._recompute_and_render()

    def _on_delete_band_clicked(self):
        row = self.band_table.currentRow()
        if row < 0:
            row = self.band_table.rowCount() - 1
        if row < 0:
            return
        bands = self._read_table_bands()
        if 0 <= row < len(bands):
            bands.pop(row)
        self._set_table_bands(bands)
        self._recompute_and_render()

    def _on_table_item_changed(self, item: QTableWidgetItem):
        if self._updating_table or self._block_signals:
            return
        self._refresh_zoom_combo()
        if item.column() in (0, 2, 3):
            self._schedule_recompute()

    def _on_table_row_selected(self, row: int, *_args):
        if self._updating_table or row < 0:
            return
        self._sync_kinetics_from_focused_band(render=False)
        if self.tabs.currentIndex() == 3:
            self._dirty_tabs[3] = True
            self._render_tab(3)

    def _on_view_changed(self):
        if self._block_signals:
            return
        self._sync_kinetics_from_focused_band(render=False)
        self._dirty_tabs = [True, True, True, True]
        self._render_tab(self.tabs.currentIndex())

    def _on_tab_changed(self, idx: int):
        if 0 <= idx < len(self._dirty_tabs) and self._dirty_tabs[idx]:
            self._render_tab(idx)

    # ------------------------------------------------------------------ #
    #                     Computation & Rendering                        #
    # ------------------------------------------------------------------ #
    def _recompute_and_render(self):
        self._recompute_timer.stop()
        bands = self._read_table_bands()
        try:
            self.result = compute_scatter_removal(
                self.dataset,
                bands=bands,
                tmin=float(self.spin_tmin.value()),
                tmax=float(self.spin_tmax.value()),
                method=str(self.combo_method.currentData() or "local_svd"),
                rank=int(self.spin_rank.value()),
                poly_deg=int(self.spin_poly.value()),
                t0_exclude=(float(self.spin_t0_min.value()), float(self.spin_t0_max.value())),
                scale_broadband=self.chk_broadband.isChecked(),
                per_scan=self.chk_per_scan.isChecked(),
                update_noise=self.chk_update_noise.isChecked(),
                transmittance_space=self.chk_transmittance.isChecked(),
                mask_saturated=self.chk_mask_sat.isChecked(),
                saturation_threshold=float(self.spin_sat_thresh.value()),
                allow_positive=self.chk_allow_pos.isChecked(),
                detector=self._active_detector(),
            )
            self._update_metrics_label()
            self._dirty_tabs = [True, True, True, True]
            self._render_tab(self.tabs.currentIndex())
        except Exception as exc:
            logger.exception("Scatter removal computation failed: %s", exc)
            self.lbl_metrics.setText(f"<span style='color:#dc2626;'>Error: {exc}</span>")

    def _update_metrics_label(self):
        if self.result is None:
            return
        m = self.result.metrics
        u_z = str(self.dataset.units.get("unitsZ_ltx", "mOD")).replace("$", "")
        pre_b = m.get("pre_zero_rms_before", float("nan"))
        pre_a = m.get("pre_zero_rms_after", float("nan"))
        pos_b = m.get("pos_roughness_before", float("nan"))
        pos_a = m.get("pos_roughness_after", float("nan"))
        wing_ref = m.get("wing_roughness_ref", float("nan"))
        ss_b = m.get("interscan_std_before", float("nan"))
        ss_a = m.get("interscan_std_after", float("nan"))

        parts = []
        if np.isfinite(pre_b) and np.isfinite(pre_a):
            ratio = pre_b / max(pre_a, 1e-12)
            parts.append(f"<b>Pre-zero RMS:</b> {pre_b:.2f} \u2192 {pre_a:.2f} {u_z} ({ratio:.1f}\u00d7)")
        if np.isfinite(pos_b) and np.isfinite(pos_a):
            ref_txt = f" (wing ref: {wing_ref:.2f})" if np.isfinite(wing_ref) else ""
            parts.append(f"<b>Spectral roughness:</b> {pos_b:.2f} \u2192 {pos_a:.2f} {u_z}{ref_txt}")
        if np.isfinite(ss_b) and np.isfinite(ss_a):
            ss_ratio = ss_b / max(ss_a, 1e-12)
            parts.append(f"<b>Inter-scan STD:</b> {ss_b:.2f} \u2192 {ss_a:.2f} {u_z} ({ss_ratio:.1f}\u00d7)")
        if m.get("n_saturated_masked", 0) > 0:
            parts.append(f"<b>Masked sat. cells:</b> {m['n_saturated_masked']}")
        self.lbl_metrics.setText(" &nbsp;|&nbsp; ".join(parts) if parts else "Done.")

    def _make_preview_ds(
        self,
        Z_arr: np.ndarray,
        bkg_arr: np.ndarray | None = None,
        std_arr: np.ndarray | None = None,
    ) -> Dataset1D:
        """Build a lightweight non-copying Dataset1D wrapper for ``oneD.plot`` reuse."""
        ds = Dataset1D(
            Z_arr,
            self.dataset.delays,
            self.dataset.probe,
            self.dataset.units,
            nscans=self.dataset.nscans,
            Zstdv=std_arr,
            source=self.dataset.source,
            data_type=self.dataset.data_type,
        )
        ds.Zavg_C = Z_arr
        if bkg_arr is not None:
            b = np.asarray(bkg_arr, dtype=float)
            if b.ndim == 1:
                b = b[:, np.newaxis]
            ds.bkg_avg = b
        return ds

    def _focused_band_index(self) -> int | None:
        z_data = self.combo_zoom.currentData()
        if z_data is not None:
            return int(z_data)
        row = self.band_table.currentRow()
        if row >= 0:
            return row
        return 0 if self.band_table.rowCount() > 0 else None

    def _zoom_xlim_display(self) -> tuple[float, float] | None:
        idx = self.combo_zoom.currentData()
        if idx is None or self.result is None:
            return None
        bands = self.result.bands
        if not (0 <= int(idx) < len(bands)):
            return None
        b = bands[int(idx)]
        det = self._active_detector()
        p = np.sort(self.dataset._detector_probe(det))
        med_step = float(np.median(np.abs(np.diff(p)))) if p.size > 1 else 1.0
        wing_w, _ = _band_widths(b, med_step)
        pad = max(2.0 * wing_w, 0.5 * (b.p_max - b.p_min))
        lo_nat = max(float(p[0]), b.p_min - pad)
        hi_nat = min(float(p[-1]), b.p_max + pad)
        disp = self._to_display([lo_nat, hi_nat])
        return float(np.min(disp)), float(np.max(disp))

    def _sync_kinetics_from_focused_band(self, *, render: bool = True):
        bands = self._read_table_bands()
        det = self._active_detector()
        probe = np.sort(self.dataset._detector_probe(det))
        if probe.size == 0:
            return
        idx = self._focused_band_index()
        if idx is not None and 0 <= idx < len(bands):
            b = bands[idx]
            med_step = float(np.median(np.abs(np.diff(probe)))) if probe.size > 1 else 1.0
            wing_w, _ = _band_widths(b, med_step)
            pos_nat = [
                max(float(probe[0]), b.p_min - 0.6 * wing_w),
                0.5 * (b.p_min + b.p_max),
                min(float(probe[-1]), b.p_max + 0.6 * wing_w),
            ]
        else:
            pos_nat = [float(probe[probe.size // 4]), float(probe[probe.size // 2]), float(probe[(3 * probe.size) // 4])]
        pos_disp = self._to_display(pos_nat)
        self.edit_kin_pos.setText(", ".join(f"{v:.1f}" for v in pos_disp))
        if render and self.tabs.currentIndex() == 2:
            self._render_tab(2)

    def _render_tab(self, tab_idx: int):
        if self.result is None:
            return
        pm.apply_style()
        if tab_idx == 0:
            self._render_maps_tab()
        elif tab_idx == 1:
            self._render_spectra_tab()
        elif tab_idx == 2:
            self._render_kinetics_tab()
        elif tab_idx == 3:
            self._render_diagnostics_tab()
        if 0 <= tab_idx < len(self._dirty_tabs):
            self._dirty_tabs[tab_idx] = False

    def _render_maps_tab(self):
        res = self.result
        fig = self.canvas_maps.figure
        fig.clear()
        fig.set_layout_engine(None)

        det = self._active_detector()
        s = self._panel_settings()
        asinh = self.chk_asinh.isChecked()
        zscale = float(self.spin_zscale.value())

        ds_before = self._make_preview_ds(res.Z_static_C, res.bkg_avg)
        ds_after = self._make_preview_ds(res.Zavg_C, res.bkg_avg, res.Zstdv)
        ds_diff = self._make_preview_ds(res.Z_diff, res.bkg_avg)

        # Shared symmetric Z limits for Static vs Clean (computed on clean signal inside zoom window)
        xlim_disp = self._zoom_xlim_display()
        p_nat = self.dataset._detector_probe(det)
        p_disp = self._to_display(p_nat)
        if xlim_disp is not None:
            view_cols = (p_disp >= xlim_disp[0]) & (p_disp <= xlim_disp[1])
        else:
            view_cols = np.ones(p_nat.size, dtype=bool)
        if not np.any(view_cols):
            view_cols = np.ones(p_nat.size, dtype=bool)

        clean_slice = ds_after._detector_slice(det)[:, view_cols]
        fin_clean = clean_slice[np.isfinite(clean_slice)]
        if fin_clean.size > 0:
            ref_amp = float(np.nanpercentile(np.abs(fin_clean), 99.5))
        else:
            ref_amp = 1.0
        if not np.isfinite(ref_amp) or ref_amp <= 1e-12:
            ref_amp = 1.0
        vmax_sig = max(1e-6, ref_amp * (zscale / 100.0 if zscale < 100.0 else 1.0))

        diff_slice = ds_diff._detector_slice(det)[:, view_cols]
        fin_diff = diff_slice[np.isfinite(diff_slice)]
        vmax_diff = float(np.nanmax(np.abs(fin_diff))) if fin_diff.size else 1.0
        if not np.isfinite(vmax_diff) or vmax_diff <= 1e-12:
            vmax_diff = 1.0

        gs = GridSpec(
            1, 3, figure=fig, wspace=0.28, left=0.07, right=0.91, top=0.88, bottom=0.12
        )
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1], sharex=ax0, sharey=ax0)
        ax2 = fig.add_subplot(gs[2], sharex=ax0, sharey=ax0)

        # Match panel widths on ax0 with invisible divider spacer
        div0 = make_axes_locatable(ax0)
        cax0_dummy = div0.append_axes("right", size=_CBAR_WIDTH_IN, pad=_CBAR_PAD_IN)
        cax0_dummy.axis("off")

        ds_before.plot_contour(
            ax=ax0,
            detector=det,
            settings=s,
            quick=True,
            Zmin=-vmax_sig,
            Zmax=vmax_sig,
            Asinh=asinh,
            show_colorbar=False,
            show_ylabel=True,
        )
        ax0.set_title("Static Pre-Zero", fontsize=10, fontweight="bold")

        ds_after.plot_contour(
            ax=ax1,
            detector=det,
            settings=s,
            quick=True,
            Zmin=-vmax_sig,
            Zmax=vmax_sig,
            Asinh=asinh,
            show_colorbar=True,
            show_colorbar_label=True,
            show_ylabel=False,
        )
        ax1.set_title("Advanced Clean", fontsize=10, fontweight="bold")
        ax1.tick_params(labelleft=False)

        ds_diff.plot_contour(
            ax=ax2,
            detector=det,
            settings=s,
            quick=True,
            Zmin=-vmax_diff,
            Zmax=vmax_diff,
            Asinh=asinh,
            show_colorbar=True,
            show_colorbar_label=True,
            show_ylabel=False,
        )
        ax2.set_title("Removed Scatter (Diff)", fontsize=10, fontweight="bold")
        ax2.tick_params(labelleft=False)

        if xlim_disp is not None:
            ax0.set_xlim(xlim_disp[0], xlim_disp[1])

        for b in res.bands:
            if b.enabled:
                b_disp = self._to_display([b.p_min, b.p_max])
                for ax in (ax0, ax1, ax2):
                    ax.axvline(float(b_disp[0]), color="0.35", ls="--", lw=0.8, alpha=0.6)
                    ax.axvline(float(b_disp[1]), color="0.35", ls="--", lw=0.8, alpha=0.6)

        for ax in (ax0, ax1, ax2):
            attach_contour_scroll_zoom(ax)

        style_figure(fig, is_dark_palette())
        self.canvas_maps.draw()

    def _render_spectra_tab(self):
        res = self.result
        fig = self.canvas_spec.figure
        fig.clear()
        fig.set_layout_engine(None)

        det = self._active_detector()
        s = self._panel_settings()
        delays_req = hlp.parse_value_list(self.edit_spec_delays.text(), self.dataset.delays)
        if not delays_req:
            delays_req = hlp.parse_value_list(self._default_slice_delays_str(), self.dataset.delays) or [
                float(self.dataset.delays[0])
            ]

        ds_before = self._make_preview_ds(res.Z_static_C, res.bkg_avg)
        ds_after = self._make_preview_ds(res.Zavg_C, res.bkg_avg, res.Zstdv)

        gs = GridSpec(2, 1, figure=fig, hspace=0.28, left=0.10, right=0.96, top=0.92, bottom=0.11)
        ax_top = fig.add_subplot(gs[0])
        ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

        ds_before.plot_spectra(delays_req, ax=ax_top, detector=det, settings=s, show_xlabel=False)
        ax_top.set_title("Static Pre-Zero Subtraction", fontsize=10, fontweight="bold")
        ax_top.tick_params(labelbottom=False)

        ds_after.plot_spectra(delays_req, ax=ax_bot, detector=det, settings=s, show_xlabel=True)
        ax_bot.set_title("Advanced Scatter Removal", fontsize=10, fontweight="bold")

        for b in res.bands:
            if b.enabled:
                bd = self._to_display([b.p_min, b.p_max])
                lo, hi = float(min(bd)), float(max(bd))
                ax_top.axvspan(lo, hi, color="#94a3b8", alpha=0.15)
                ax_bot.axvspan(lo, hi, color="#94a3b8", alpha=0.15)

        xlim_disp = self._zoom_xlim_display()
        if xlim_disp is not None:
            ax_top.set_xlim(xlim_disp[0], xlim_disp[1])

        style_figure(fig, is_dark_palette())
        self.canvas_spec.draw()

    def _render_kinetics_tab(self):
        res = self.result
        fig = self.canvas_kin.figure
        fig.clear()
        fig.set_layout_engine(None)

        det = self._active_detector()
        s = self._panel_settings()
        pos_disp = hlp.parse_value_list(self.edit_kin_pos.text())
        if not pos_disp:
            self._sync_kinetics_from_focused_band(render=False)
            pos_disp = hlp.parse_value_list(self.edit_kin_pos.text()) or [
                float(self._to_display(self.dataset._detector_probe(det))[0])
            ]
        pos_nat = [float(x) for x in self._to_native(pos_disp)]

        ds_before = self._make_preview_ds(res.Z_static_C, res.bkg_avg)
        ds_after = self._make_preview_ds(res.Zavg_C, res.bkg_avg, res.Zstdv)

        gs = GridSpec(2, 1, figure=fig, hspace=0.28, left=0.10, right=0.96, top=0.92, bottom=0.11)
        ax_top = fig.add_subplot(gs[0])
        ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

        ds_before.plot_kinetics(pos_nat, ax=ax_top, detector=det, settings=s, show_xlabel=False)
        ax_top.set_title("Static Pre-Zero Kinetics", fontsize=10, fontweight="bold")
        ax_top.tick_params(labelbottom=False)

        ds_after.plot_kinetics(pos_nat, ax=ax_bot, detector=det, settings=s, show_xlabel=True)
        ax_bot.set_title("Advanced Clean Kinetics", fontsize=10, fontweight="bold")

        style_figure(fig, is_dark_palette())
        self.canvas_kin.draw()

    def _render_diagnostics_tab(self):
        res = self.result
        fig = self.canvas_diag.figure
        fig.clear()
        fig.set_layout_engine(None)

        det = self._active_detector()
        s = self._panel_settings()
        Units = self.dataset._resolve(s, None, None)[2]
        label_style = s.label_style.value

        gs = GridSpec(
            2,
            2,
            figure=fig,
            height_ratios=[1.0, 1.0],
            hspace=0.42,
            wspace=0.28,
            left=0.09,
            right=0.91,
            top=0.93,
            bottom=0.11,
        )
        ax_pre = fig.add_subplot(gs[0, 0])
        ax_basis = fig.add_subplot(gs[0, 1])
        ax_traj = fig.add_subplot(gs[1, :])

        # 1. Pre-zero background via plot_background + band shaded spans + SpanSelector
        ds_bkg = self._make_preview_ds(res.Zavg_C, res.bkg_avg)
        ds_bkg.plot_background(
            ax=ax_pre,
            detector=det,
            settings=s,
            title="Pre-Zero Spectrum (Drag to Pick Band)",
        )
        focus_idx = self._focused_band_index()
        for idx, b in enumerate(res.bands):
            bd = self._to_display([b.p_min, b.p_max])
            lo, hi = float(min(bd)), float(max(bd))
            if b.enabled:
                color = "#2563eb" if idx == focus_idx else "#0ea5e9"
                ax_pre.axvspan(lo, hi, color=color, alpha=0.22 if idx == focus_idx else 0.12)
            else:
                ax_pre.axvspan(lo, hi, color="#94a3b8", alpha=0.10, ls="--", ec="#64748b")

        self._span_selector = SpanSelector(
            ax_pre,
            self._on_span_picked,
            "horizontal",
            useblit=False,
            interactive=False,
            props={"alpha": 0.25, "facecolor": "#2563eb"},
        )
        self._span_selector.set_active(self.btn_pick_band.isChecked())

        # Resolve display X unit metadata for basis plot
        p_probe = self.dataset._detector_probe(det)
        _, XUnits, _, conv = _resolve_x_axis(p_probe, Units, s, secondary_axis=False)

        # 2. Extracted basis shapes for focused band
        diag = None
        if res.band_diagnostics:
            for d_item in res.band_diagnostics:
                if d_item["band_index"] == focus_idx:
                    diag = d_item
                    break
            if diag is None:
                diag = res.band_diagnostics[0]

        if diag is not None:
            w_disp = conv(diag["w_fit"])
            tmpl = diag["template"]
            tmpl_max = float(np.max(np.abs(tmpl))) if tmpl.size else 1.0
            tmpl_norm = tmpl / tmpl_max if tmpl_max > 1e-12 else tmpl
            ax_basis.plot(w_disp, tmpl_norm, "k--", lw=1.4, label="Template $P_0$")

            basis = diag["basis"]
            for m in range(min(basis.shape[1], 4)):
                bm = basis[:, m]
                bm_max = float(np.max(np.abs(bm))) if bm.size else 1.0
                bm_norm = bm / bm_max if bm_max > 1e-12 else bm
                ax_basis.plot(w_disp, bm_norm, lw=1.4, label=f"Mode {m + 1}")
            ax_basis.axhline(0, color="0.75", lw=0.75)
            hlp.setXYlabels(ax_basis, label_style, XUnits, {"lbl": "Norm. Weight", "ltx": ""})
            b_lbl = diag["band"].label or f"Band {diag['band_index'] + 1}"
            ax_basis.set_title(f"Scatter Basis ({b_lbl})", fontsize=10, fontweight="bold")
            ax_basis.legend(loc="best", fontsize=8, frameon=False)
        else:
            ax_basis.text(0.5, 0.5, "No enabled scatter bands", ha="center", va="center", transform=ax_basis.transAxes)
            ax_basis.set_title("Scatter Basis", fontsize=10, fontweight="bold")

        # 3. Scatter amplitude & centroid shift trajectory vs delay
        delays = np.asarray(self.dataset.delays, dtype=float)
        if diag is not None:
            amp_raw = diag["amplitude"]
            amp_sm = diag["amplitude_smooth"]
            shifts = diag["centroid_shift"]
            ax_traj.plot(delays, amp_raw, "-", color="#2563eb", alpha=0.35, lw=1.0, label="Raw $a(t)$")
            ax_traj.plot(delays, amp_sm, "-", color="#2563eb", lw=1.8, label="Smooth $a_{smooth}(t)$")
            ax_traj.axhline(1.0, color="0.75", ls=":", lw=0.8)

            _apply_time_xscale(
                ax_traj,
                s.time_axis_scale.value,
                delays,
                time_axis_label=s.time_axis_label.value,
                settings=s,
                axis="x",
            )
            TUnits = {"lbl": "Delay", "ltx": Units["unitsT_ltx"]}
            hlp.setXYlabels(ax_traj, label_style, TUnits, {"lbl": "Rel. Amplitude", "ltx": ""})
            ax_traj.set_title("Scatter Amplitude & Centroid Shift vs Delay", fontsize=10, fontweight="bold")

            if not diag.get("coupled", False) and np.any(np.abs(shifts) > 1e-6):
                ax_sh = ax_traj.twinx()
                ax_sh.plot(delays, shifts, "-", color="#dc2626", alpha=0.85, lw=1.2, label=r"$\delta\lambda(t)$")
                ax_sh.set_ylabel(f"Shift ({self._native_unit()})", color="#dc2626")
                ax_sh.tick_params(axis="y", labelcolor="#dc2626")

            ax_traj.legend(loc="best", fontsize=8, frameon=False)
        else:
            ax_traj.text(0.5, 0.5, "No trajectory available", ha="center", va="center", transform=ax_traj.transAxes)

        style_figure(fig, is_dark_palette())
        self.canvas_diag.draw()

    # ------------------------------------------------------------------ #
    #                         Apply / Reset                              #
    # ------------------------------------------------------------------ #
    def _on_apply_clicked(self):
        if self.result is None:
            return
        self.scatter_applied.emit(self.result)
        self.accept()

    def _on_reset_clicked(self):
        reply = QMessageBox.question(
            self,
            "Reset Scatter Removal",
            "Reset the active dataset to standard static pre-zero background subtraction?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.scatter_reset.emit()
            self.reject()

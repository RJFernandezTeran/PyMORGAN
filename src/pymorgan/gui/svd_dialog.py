"""PyQt6 Dialog for Singular Value Decomposition (SVD) and Rank-k Recomposition.

Implements the unified 4-quadrant interactive dashboard (Option A):
- Scree plot (with log / linear toggle)
- Extracted temporal U(t) and spectral V(lambda) components (with symlog / linear time scale toggle)
- Side-by-side Raw, Recomposed, and Residuals 2D maps
- Interactive time & probe window limits, centering options, rank-k slider, and data export.
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pymorgan.gui.canvas import MplCanvas
from pymorgan.log import get_logger
from pymorgan.oneD.svd import SVDResult

if TYPE_CHECKING:
    from pymorgan.oneD.dataset import Dataset1D

logger = get_logger(__name__)


class SVDDialog(QDialog):
    """Unified 4-Quadrant SVD Dashboard Dialog."""

    # Signal emitted when user requests to apply the recomposed dataset back to the main window
    dataset_applied = pyqtSignal(object)

    def __init__(self, parent: QWidget | None, dataset: Dataset1D):
        super().__init__(parent)
        self.dataset = dataset
        self.setWindowTitle(
            f"SVD Analysis & Truncation \u2014 {getattr(dataset, 'source', None) or 'Dataset'}"
        )
        self.resize(1200, 850)
        self.setMinimumSize(950, 700)

        # State
        self.svd_res: SVDResult | None = None
        self._block_signals = False

        # Units
        u = getattr(dataset, "units", {}) or {}
        self.time_label = u.get("unitsT_lbl", "Delay")
        self.time_unit = u.get("unitsT_ltx", "ps")
        self.probe_label = u.get("unitsL_lbl", "Probe")
        self.probe_unit = u.get("unitsL_ltx", "")
        self.z_label = u.get("unitsZ_lbl", r"$\Delta$A")
        self.z_unit = u.get("unitsZ_ltx", "mOD")

        # Top-level layout with Splitter
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        main_layout.addWidget(splitter)

        # 1. Left Controls Area (Scrollable)
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls_scroll.setMinimumWidth(300)
        controls_scroll.setMaximumWidth(360)

        controls_widget = QWidget()
        self.ctrl_layout = QVBoxLayout(controls_widget)
        self.ctrl_layout.setContentsMargins(4, 4, 4, 4)
        self.ctrl_layout.setSpacing(10)

        self._build_controls()
        controls_scroll.setWidget(controls_widget)
        splitter.addWidget(controls_scroll)

        # 2. Right Visualization Canvas
        self.canvas = MplCanvas(self, show_toolbar=True)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Canvas click connection for interactive bar picking on scree plot
        self.canvas.canvas.mpl_connect("button_press_event", self._on_canvas_clicked)

        # Initial Computation & Render
        self._recompute_svd()

    def _build_controls(self):
        """Construct the sidebar controls."""
        # --- Header ---
        hdr = QLabel("<b>SVD Analysis & Truncation</b>")
        hdr.setStyleSheet("font-size: 13pt; margin-bottom: 2px;")
        self.ctrl_layout.addWidget(hdr)

        desc = QLabel("Inspect singular components, scree variance, and recompose rank-<i>k</i> approximations.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #64748b; font-size: 9pt; margin-bottom: 4px;")
        self.ctrl_layout.addWidget(desc)

        # --- Group 1: Data Channel & Array ---
        grp_data = QGroupBox("Data Channel & Signal Array")
        vbox_data = QVBoxLayout(grp_data)

        # Detector selection
        h_det = QHBoxLayout()
        h_det.addWidget(QLabel("Detector:"))
        self.det_combo = QComboBox()
        n_det = getattr(self.dataset, "n_detectors", 1)
        for d in range(n_det):
            self.det_combo.addItem(f"Detector {d + 1}", d)
        self.det_combo.currentIndexChanged.connect(self._recompute_svd)
        h_det.addWidget(self.det_combo)
        vbox_data.addLayout(h_det)
        if n_det <= 1:
            self.det_combo.setEnabled(False)

        # Array selection
        h_arr = QHBoxLayout()
        h_arr.addWidget(QLabel("Array:"))
        self.arr_combo = QComboBox()
        self.arr_combo.addItem("Background-Corrected (Zavg_C)", "corrected")
        self.arr_combo.addItem("Raw Signal (Zavg_R)", "raw")
        if self.dataset.Zavg_C is None:
            self.arr_combo.setCurrentIndex(1)
        self.arr_combo.currentIndexChanged.connect(self._recompute_svd)
        h_arr.addWidget(self.arr_combo)
        vbox_data.addLayout(h_arr)

        self.ctrl_layout.addWidget(grp_data)

        # --- Group 2: Window Limits & Centering ---
        grp_lim = QGroupBox("Window Limits & Centering")
        vbox_lim = QVBoxLayout(grp_lim)

        t_min_data = float(np.min(self.dataset.delays))
        t_max_data = float(np.max(self.dataset.delays))
        p_min_data = float(np.min(self.dataset.probe))
        p_max_data = float(np.max(self.dataset.probe))

        # Time range
        vbox_lim.addWidget(QLabel(f"<b>Time Window ({self.time_unit}):</b>"))
        h_t = QHBoxLayout()
        self.tmin_spin = QDoubleSpinBox()
        self.tmin_spin.setRange(-1e7, 1e7)
        self.tmin_spin.setDecimals(3)
        self.tmin_spin.setValue(t_min_data)
        self.tmax_spin = QDoubleSpinBox()
        self.tmax_spin.setRange(-1e7, 1e7)
        self.tmax_spin.setDecimals(3)
        self.tmax_spin.setValue(t_max_data)
        h_t.addWidget(QLabel("Min:"))
        h_t.addWidget(self.tmin_spin)
        h_t.addWidget(QLabel("Max:"))
        h_t.addWidget(self.tmax_spin)
        vbox_lim.addLayout(h_t)

        btn_t_reset = QPushButton("Reset Time Window")
        btn_t_reset.clicked.connect(lambda: self._reset_time_limits(t_min_data, t_max_data))
        vbox_lim.addWidget(btn_t_reset)

        # Probe range
        vbox_lim.addWidget(QLabel(f"<b>Probe Window ({self.probe_unit}):</b>"))
        h_p = QHBoxLayout()
        self.pmin_spin = QDoubleSpinBox()
        self.pmin_spin.setRange(-1e7, 1e7)
        self.pmin_spin.setDecimals(2)
        self.pmin_spin.setValue(min(p_min_data, p_max_data))
        self.pmax_spin = QDoubleSpinBox()
        self.pmax_spin.setRange(-1e7, 1e7)
        self.pmax_spin.setDecimals(2)
        self.pmax_spin.setValue(max(p_min_data, p_max_data))
        h_p.addWidget(QLabel("Min:"))
        h_p.addWidget(self.pmin_spin)
        h_p.addWidget(QLabel("Max:"))
        h_p.addWidget(self.pmax_spin)
        vbox_lim.addLayout(h_p)

        btn_p_reset = QPushButton("Reset Probe Window")
        btn_p_reset.clicked.connect(lambda: self._reset_probe_limits(p_min_data, p_max_data))
        vbox_lim.addWidget(btn_p_reset)

        # Centering mode
        h_cent = QHBoxLayout()
        h_cent.addWidget(QLabel("Centering:"))
        self.cent_combo = QComboBox()
        self.cent_combo.addItem("None (Uncentered)", "none")
        self.cent_combo.addItem("Average Spectrum", "spectrum")
        self.cent_combo.addItem("Average Kinetic", "kinetic")
        self.cent_combo.addItem("Both (Double centered)", "both")
        h_cent.addWidget(self.cent_combo)
        vbox_lim.addLayout(h_cent)

        # Apply limits button
        self.btn_apply_limits = QPushButton("Apply Window / Centering")
        self.btn_apply_limits.setStyleSheet("font-weight: bold; padding: 5px;")
        self.btn_apply_limits.clicked.connect(self._recompute_svd)
        vbox_lim.addWidget(self.btn_apply_limits)

        self.ctrl_layout.addWidget(grp_lim)

        # --- Group 3: Rank-k Truncation ---
        grp_k = QGroupBox("Rank-k Truncation")
        vbox_k = QVBoxLayout(grp_k)

        h_k = QHBoxLayout()
        h_k.addWidget(QLabel("Components (k):"))
        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 20)
        self.k_spin.setValue(2)
        h_k.addWidget(self.k_spin)
        vbox_k.addLayout(h_k)

        self.k_slider = QSlider(Qt.Orientation.Horizontal)
        self.k_slider.setRange(1, 20)
        self.k_slider.setValue(2)
        vbox_k.addWidget(self.k_slider)

        self.var_label = QLabel("Explained Variance: --")
        self.var_label.setStyleSheet("font-weight: bold; color: #1e293b; padding: 2px 0;")
        vbox_k.addWidget(self.var_label)

        # Quick preset buttons for 90%, 95%, 99%
        h_presets = QHBoxLayout()
        for pct_target in [90.0, 95.0, 99.0, 99.9]:
            btn_pct = QPushButton(f"{pct_target:g}%")
            btn_pct.setStyleSheet("font-size: 8pt; padding: 2px;")
            btn_pct.clicked.connect(lambda _, target=pct_target: self._set_k_for_target_variance(target))
            h_presets.addWidget(btn_pct)
        vbox_k.addLayout(h_presets)

        self.k_spin.valueChanged.connect(self._on_k_spin_changed)
        self.k_slider.valueChanged.connect(self._on_k_slider_changed)

        self.ctrl_layout.addWidget(grp_k)

        # --- Group 4: Display Scales (User Requirement) ---
        grp_scales = QGroupBox("Plot Scale Settings")
        vbox_scales = QVBoxLayout(grp_scales)

        # 4.1 Time Scale Toggle
        vbox_scales.addWidget(QLabel("<b>Time Axis Scale:</b>"))
        self.time_scale_group = QButtonGroup(self)
        self.rad_time_linear = QRadioButton("Linear")
        self.rad_time_symlog = QRadioButton("Symlog (t0 linthresh)")
        self.rad_time_linear.setChecked(True)
        self.time_scale_group.addButton(self.rad_time_linear)
        self.time_scale_group.addButton(self.rad_time_symlog)

        h_time_scales = QHBoxLayout()
        h_time_scales.addWidget(self.rad_time_linear)
        h_time_scales.addWidget(self.rad_time_symlog)
        vbox_scales.addLayout(h_time_scales)

        # 4.2 Scree Scale Toggle
        vbox_scales.addWidget(QLabel("<b>Scree Plot Y-Scale:</b>"))
        self.scree_scale_group = QButtonGroup(self)
        self.rad_scree_log = QRadioButton("Log10")
        self.rad_scree_linear = QRadioButton("Linear")
        self.rad_scree_log.setChecked(True)
        self.scree_scale_group.addButton(self.rad_scree_log)
        self.scree_scale_group.addButton(self.rad_scree_linear)

        h_scree_scales = QHBoxLayout()
        h_scree_scales.addWidget(self.rad_scree_log)
        h_scree_scales.addWidget(self.rad_scree_linear)
        vbox_scales.addLayout(h_scree_scales)

        # Connect scale toggles
        self.rad_time_linear.toggled.connect(self._update_plots)
        self.rad_scree_log.toggled.connect(self._update_plots)

        self.ctrl_layout.addWidget(grp_scales)

        # --- Group 5: Actions ---
        grp_actions = QGroupBox("Actions & Export")
        vbox_act = QVBoxLayout(grp_actions)

        self.btn_apply_dataset = QPushButton("Apply Recomposed to Workspace")
        self.btn_apply_dataset.setStyleSheet(
            "font-weight: bold; background-color: #2563eb; color: white; padding: 6px; border-radius: 4px;"
        )
        self.btn_apply_dataset.clicked.connect(self._apply_to_workspace)
        vbox_act.addWidget(self.btn_apply_dataset)

        self.btn_export_recon = QPushButton("Export Recomposed Data...")
        self.btn_export_recon.clicked.connect(self._export_recomposed)
        vbox_act.addWidget(self.btn_export_recon)

        self.btn_export_comp = QPushButton("Export Components (CSV)...")
        self.btn_export_comp.clicked.connect(self._export_components)
        vbox_act.addWidget(self.btn_export_comp)

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)
        vbox_act.addWidget(self.btn_close)

        self.ctrl_layout.addWidget(grp_actions)
        self.ctrl_layout.addStretch()

    def _reset_time_limits(self, tmin: float, tmax: float):
        self.tmin_spin.setValue(tmin)
        self.tmax_spin.setValue(tmax)
        self._recompute_svd()

    def _reset_probe_limits(self, pmin: float, pmax: float):
        self.pmin_spin.setValue(min(pmin, pmax))
        self.pmax_spin.setValue(max(pmin, pmax))
        self._recompute_svd()

    def _on_k_spin_changed(self, val: int):
        if self._block_signals:
            return
        self._block_signals = True
        self.k_slider.setValue(val)
        self._block_signals = False
        self._update_plots()

    def _on_k_slider_changed(self, val: int):
        if self._block_signals:
            return
        self._block_signals = True
        self.k_spin.setValue(val)
        self._block_signals = False
        self._update_plots()

    def _set_k_for_target_variance(self, target_pct: float):
        """Find the smallest k that satisfies target cumulative variance."""
        if self.svd_res is None:
            return
        target_frac = target_pct / 100.0
        cum = self.svd_res.cumulative_variance
        idx = int(np.searchsorted(cum, target_frac))
        k_val = min(idx + 1, self.svd_res.n_components)
        self.k_spin.setValue(k_val)

    def _recompute_svd(self):
        """Recompute SVD decomposition from current limits and array selection."""
        try:
            det = self.det_combo.currentData()
            arr_mode = self.arr_combo.currentData()
            cent_mode = self.cent_combo.currentData()
            tmin = float(self.tmin_spin.value())
            tmax = float(self.tmax_spin.value())
            pmin = float(self.pmin_spin.value())
            pmax = float(self.pmax_spin.value())
            k_cur = self.k_spin.value()

            self.svd_res = self.dataset.svd(
                k=k_cur,
                detector=det,
                array=arr_mode,
                tmin=tmin,
                tmax=tmax,
                probe_min=pmin,
                probe_max=pmax,
                centering=cent_mode,
            )

            max_comp = min(self.svd_res.n_components, 20)
            self._block_signals = True
            self.k_spin.setMaximum(max_comp)
            self.k_slider.setMaximum(max_comp)
            k_clamped = min(self.k_spin.value(), max_comp)
            self.k_spin.setValue(k_clamped)
            self.k_slider.setValue(k_clamped)
            self._block_signals = False

            self._update_plots()

        except Exception as exc:
            logger.exception("SVD calculation failed: %s", exc)
            QMessageBox.critical(self, "SVD Error", f"SVD calculation failed:\n{exc}")

    def _update_plots(self):
        """Redraw all 4 quadrants in the Matplotlib canvas."""
        if self.svd_res is None:
            return

        k = self.k_spin.value()
        res = self.svd_res
        n_comp = res.n_components
        cum_pct = res.variance_at(k) * 100.0
        self.var_label.setText(f"Explained Variance: {cum_pct:.2f}% (Rank {k})")

        is_time_symlog = self.rad_time_symlog.isChecked()
        is_scree_log = self.rad_scree_log.isChecked()

        fig = self.canvas.figure
        fig.clear()

        # GridSpec: 2 rows x 3 columns
        # Row 0: Scree plot (Col 0), Components (Cols 1-2)
        # Row 1: Raw (Col 0), Recomposed (Col 1), Residuals (Col 2)
        gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.15], hspace=0.35, wspace=0.32)

        # ----------------------------------------------------------------- #
        # Quadrant 1: Scree Plot (Row 0, Col 0)                            #
        # ----------------------------------------------------------------- #
        ax_scree = fig.add_subplot(gs[0, 0])
        n_display = min(n_comp, 15)
        comp_idx = np.arange(1, n_display + 1)
        s_vals = res.S[:n_display]

        # Singular value bars & line
        min_s = float(np.min(s_vals))
        bar_bottom = min_s * 0.4 if is_scree_log else 0.0

        colors = ["#2563eb" if i <= k else "#94a3b8" for i in comp_idx]
        ax_scree.bar(
            comp_idx,
            s_vals,
            color=colors,
            alpha=0.75,
            edgecolor="#1e293b",
            linewidth=0.8,
            bottom=bar_bottom,
            width=0.65,
        )
        ax_scree.plot(comp_idx, s_vals, "o-", color="#1e3a8a", markersize=4, linewidth=1.2)

        # Cutoff line at k
        ax_scree.axvline(k, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Rank k={k}")

        ax_scree.set_xlabel("Singular Component Index", fontsize=9)
        ax_scree.set_ylabel(rf"Singular Value $\sigma$ ({'Log10' if is_scree_log else 'Linear'})", fontsize=9)
        ax_scree.set_xticks(comp_idx)
        ax_scree.set_title(r"Scree Plot ($\sigma_i$)", fontsize=10, fontweight="bold")
        ax_scree.tick_params(labelsize=8)

        if is_scree_log:
            ax_scree.set_yscale("log")
            ax_scree.set_ylim(bottom=bar_bottom, top=float(np.max(s_vals)) * 2.0)
        else:
            ax_scree.set_yscale("linear")
            ax_scree.set_ylim(bottom=0.0, top=float(np.max(s_vals)) * 1.08)

        # Twinx cumulative variance curve
        ax_scree_cum = ax_scree.twinx()
        cum_vals = res.cumulative_variance[:n_display] * 100.0
        ax_scree_cum.plot(comp_idx, cum_vals, "s--", color="#059669", markersize=4, linewidth=1.2)
        ax_scree_cum.set_ylabel("Cumulative Variance (%)", color="#059669", fontsize=9)
        ax_scree_cum.tick_params(axis="y", labelcolor="#059669", labelsize=8)
        ax_scree_cum.set_ylim(0, 105)
        ax_scree_cum.grid(False)

        # ----------------------------------------------------------------- #
        # Quadrant 2: SVD Components (Row 0, Cols 1-2)                      #
        # ----------------------------------------------------------------- #
        gs_comp = gs[0, 1:].subgridspec(2, 1, hspace=0.35)
        ax_u = fig.add_subplot(gs_comp[0, 0])
        ax_v = fig.add_subplot(gs_comp[1, 0])

        cmap_tab = plt.get_cmap("tab10")
        k_plot = min(k, 6)

        # Temporal components U_i(t) * S_i
        for i in range(k_plot):
            c = cmap_tab(i % 10)
            u_trace = res.component_time(i, scaled=True)
            ax_u.plot(res.delays, u_trace, color=c, label=f"$U_{{{i+1}}} \\cdot S_{{{i+1}}}$", linewidth=1.3)
        ax_u.axhline(0, color="#94a3b8", linestyle=":", linewidth=0.8)
        ax_u.set_ylabel("$U_k(t) \\cdot S_k$", fontsize=8)
        ax_u.set_title(f"First {k_plot} Temporal Components $U_k(t)$", fontsize=9, fontweight="bold")
        ax_u.legend(loc="upper right", fontsize=7, ncol=min(k_plot, 3), framealpha=0.6)
        ax_u.tick_params(labelsize=8)

        if is_time_symlog:
            linthresh = 1.0 if np.any(res.delays > 1.0) else 0.1
            ax_u.set_xscale("symlog", linthresh=linthresh)
        else:
            ax_u.set_xscale("linear")

        # Spectral components V_i(lambda)
        for i in range(k_plot):
            c = cmap_tab(i % 10)
            v_trace = res.component_probe(i)
            ax_v.plot(res.probe, v_trace, color=c, label=f"$V_{{{i+1}}}$", linewidth=1.3)
        ax_v.axhline(0, color="#94a3b8", linestyle=":", linewidth=0.8)
        ax_v.set_xlabel(f"{self.probe_label} ({self.probe_unit})", fontsize=8)
        ax_v.set_ylabel("$V_k(\\lambda)$", fontsize=8)
        ax_v.set_title(f"First {k_plot} Spectral Components $V_k(\\lambda)$", fontsize=9, fontweight="bold")
        ax_v.legend(loc="upper right", fontsize=7, ncol=min(k_plot, 3), framealpha=0.6)
        ax_v.tick_params(labelsize=8)

        # ----------------------------------------------------------------- #
        # Quadrants 3 & 4: 2D Maps (Row 1, Cols 0, 1, 2)                     #
        # ----------------------------------------------------------------- #
        ax_raw = fig.add_subplot(gs[1, 0])
        ax_recon = fig.add_subplot(gs[1, 1], sharex=ax_raw, sharey=ax_raw)
        ax_resid = fig.add_subplot(gs[1, 2], sharex=ax_raw, sharey=ax_raw)

        raw_mat = res.raw_matrix
        recon_mat = res.recompose(k)
        resid_mat = res.residuals(k)

        # Calculate symmetric color limits based on raw data
        vmax = float(np.nanpercentile(np.abs(raw_mat), 99.0))
        if vmax <= 0:
            vmax = 1.0
        vmin = -vmax

        # Map colormap (RdBu_r standard symmetric)
        cmap = "RdBu_r"

        # Pcolormesh
        p_grid, t_grid = np.meshgrid(res.probe, res.delays)
        ax_raw.pcolormesh(p_grid, t_grid, raw_mat, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        ax_raw.set_title("Raw Data $A(t, \\lambda)$", fontsize=10, fontweight="bold")
        ax_raw.set_ylabel(f"{self.time_label} ({self.time_unit})", fontsize=9)
        ax_raw.set_xlabel(f"{self.probe_label} ({self.probe_unit})", fontsize=9)
        ax_raw.tick_params(labelsize=8)

        ax_recon.pcolormesh(p_grid, t_grid, recon_mat, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        ax_recon.set_title(f"Recomposed $\\hat{{A}}_k$ (Rank {k})", fontsize=10, fontweight="bold")
        ax_recon.set_xlabel(f"{self.probe_label} ({self.probe_unit})", fontsize=9)
        ax_recon.tick_params(labelsize=8)

        c_res = ax_resid.pcolormesh(
            p_grid, t_grid, resid_mat, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto"
        )
        ax_resid.set_title("Residuals ($A - \\hat{A}_k$)", fontsize=10, fontweight="bold")
        ax_resid.set_xlabel(f"{self.probe_label} ({self.probe_unit})", fontsize=9)
        ax_resid.tick_params(labelsize=8)

        # Shared time scale
        if is_time_symlog:
            linthresh = 1.0 if np.any(res.delays > 1.0) else 0.1
            ax_raw.set_yscale("symlog", linthresh=linthresh)
        else:
            ax_raw.set_yscale("linear")

        # Colorbar
        cbar = fig.colorbar(c_res, ax=[ax_raw, ax_recon, ax_resid], orientation="horizontal", fraction=0.04, pad=0.18)
        cbar.set_label(f"{self.z_label} ({self.z_unit})", fontsize=8)
        cbar.ax.tick_params(labelsize=8)

        self.canvas.draw()

    def _on_canvas_clicked(self, event):
        """Allow clicking on the scree plot to select rank k."""
        if event.inaxes is None or self.svd_res is None:
            return
        # If click was inside the scree plot
        if event.xdata is not None and 0.5 <= event.xdata <= min(self.svd_res.n_components, 20) + 0.5:
            # Check if clicked in row 0, col 0
            if event.ydata is not None and event.ydata > 0:
                clicked_k = int(np.round(event.xdata))
                if 1 <= clicked_k <= min(self.svd_res.n_components, 20):
                    self.k_spin.setValue(clicked_k)

    def _apply_to_workspace(self):
        """Emit signal or update the active dataset in the main window."""
        if self.svd_res is None:
            return
        k = self.k_spin.value()
        recomp_ds = self.svd_res.to_dataset(k=k)
        reply = QMessageBox.question(
            self,
            "Apply Recomposed Data",
            f"Replace the active dataset in the workspace with the rank-{k} recomposed data?\n\n"
            f"({self.svd_res.variance_at(k) * 100.0:.2f}% of total variance retained)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.dataset_applied.emit(recomp_ds)
            QMessageBox.information(
                self,
                "Applied",
                f"Active workspace updated with SVD rank-{k} recomposed signal.",
            )

    def _export_recomposed(self):
        """Export the rank-k recomposed dataset to a file (PDAT, HDF5, or CSV)."""
        if self.svd_res is None:
            return
        k = self.k_spin.value()
        filepath, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Recomposed Dataset",
            f"svd_recomposed_rank{k}.pdat",
            "PDAT format (*.pdat);;PyMORGAN HDF5 (*.h5);;CSV Matrix (*.csv)",
        )
        if not filepath:
            return

        try:
            recomp_ds = self.svd_res.to_dataset(k=k)
            if filepath.endswith(".pdat") or "pdat" in selected_filter.lower():
                recomp_ds.export_pdat(filepath)
            elif filepath.endswith(".h5") or filepath.endswith(".hdf5") or "h5" in selected_filter.lower():
                recomp_ds.save_hdf5(filepath)
            else:
                # CSV 2D matrix
                recon = self.svd_res.recompose(k)
                with open(filepath, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Delay / Probe"] + [str(p) for p in self.svd_res.probe])
                    for t_idx, d in enumerate(self.svd_res.delays):
                        writer.writerow([str(d)] + [f"{v:.6e}" for v in recon[t_idx, :]])

            QMessageBox.information(self, "Export Complete", f"Recomposed data exported to:\n{filepath}")
        except Exception as exc:
            logger.exception("Export failed: %s", exc)
            QMessageBox.critical(self, "Export Failed", f"Could not export data:\n{exc}")

    def _export_components(self):
        """Export SVD components and scree table to CSV."""
        if self.svd_res is None:
            return
        k = self.k_spin.value()
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export SVD Components",
            f"svd_components_rank{k}.csv",
            "CSV files (*.csv)",
        )
        if not filepath:
            return

        try:
            self.svd_res.export_csv(filepath, k=k)
            QMessageBox.information(self, "Export Complete", f"SVD components exported to:\n{filepath}")
        except Exception as exc:
            logger.exception("Export components failed: %s", exc)
            QMessageBox.critical(self, "Export Failed", f"Could not export components:\n{exc}")

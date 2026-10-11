"""PyQt6 Dialog for Timescale Stitching (femtosecond + nanosecond/microsecond delay merging).

Allows user to harmonize units (ps -> ns), align spectrometer wavelengths (m * λ + b),
fit wavelength-dependent quadratic scaling S(λ), and blend timescales seamlessly
using a weighted linear ramp.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.gridspec import GridSpec
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pymorgan.gui.canvas import MplCanvas
from pymorgan.log import get_logger
from pymorgan.oneD.dataset import Dataset1D
from pymorgan.oneD.stitch import StitchResult, stitch_time

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class TimescaleStitchDialog(QDialog):
    """Dialog for Timescale Stitching of ultrafast and nanosecond/microsecond datasets."""

    stitch_applied = pyqtSignal(object)

    def __init__(self, parent: QWidget | None, dataset_fs: Dataset1D, dataset_ns: Dataset1D | None = None):
        super().__init__(parent)
        self.ds_fs = dataset_fs
        self.ds_ns = dataset_ns
        self.result: StitchResult | None = None
        self._block_signals = False

        self.setWindowTitle(
            f"Timescale Stitching (fs + ns) \u2014 {getattr(dataset_fs, 'source', None) or 'FS Dataset'}"
        )
        screen = QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            w = min(1200, max(750, int(avail.width() * 0.88)))
            h = min(850, max(500, int(avail.height() * 0.88)))
            self.resize(w, h)
        else:
            self.resize(1100, 750)
        self.setMinimumSize(960, 620)

        # Main Layout with Splitter
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        main_layout.addWidget(self.splitter)

        # 1. Left Controls Area (Scrollable)
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        controls_scroll.setMinimumWidth(380)

        controls_widget = QWidget()
        self.ctrl_layout = QVBoxLayout(controls_widget)
        self.ctrl_layout.setContentsMargins(8, 8, 8, 8)
        self.ctrl_layout.setSpacing(6)

        self._build_controls()
        controls_scroll.setWidget(controls_widget)
        self.splitter.addWidget(controls_scroll)

        # 2. Right Canvas Area
        self.canvas = MplCanvas(self, show_toolbar=True)
        self.splitter.addWidget(self.canvas)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([420, 780])

        # Initial calculation if both datasets are provided
        if self.ds_fs is not None and self.ds_ns is not None:
            self._init_overlap_parameters()
            self._update_stitching()

    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, "_splitter_initialized", False):
            self._splitter_initialized = True
            left_w = max(420, int(self.width() * 0.36))
            self.splitter.setSizes([left_w, max(500, self.width() - left_w)])
            self._init_overlap_parameters()
            self._update_stitching()

    def _build_controls(self):
        """Construct the sidebar controls."""
        # Header
        hdr = QLabel("<b>Timescale Stitching</b>")
        hdr.setStyleSheet("font-size: 13pt; margin-bottom: 2px;")
        self.ctrl_layout.addWidget(hdr)

        subtitle = QLabel("<b>Femtosecond + Nanosecond Merging</b>")
        subtitle.setStyleSheet("color: #2563eb; font-size: 9.5pt; font-weight: bold;")
        self.ctrl_layout.addWidget(subtitle)

        desc = QLabel("Harmonizes time units, aligns wavelengths, and blends dynamic timescales.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #64748b; font-size: 8.5pt; margin-bottom: 4px;")
        self.ctrl_layout.addWidget(desc)

        # --- Group 1: Input Datasets & Units ---
        grp_inputs = QGroupBox("1. Input Datasets & Units")
        vbox_in = QVBoxLayout(grp_inputs)

        vbox_in.addWidget(QLabel("<b>Dataset 1 (Femtosecond / ps, Active):</b>"))
        src_fs = str(getattr(self.ds_fs, "source", "FS Dataset"))
        name_fs = Path(src_fs).name if src_fs else "FS Dataset"
        self.lbl_fs = QLabel(f"• {name_fs} (delays in ps -> ns)")
        self.lbl_fs.setToolTip(src_fs)
        self.lbl_fs.setWordWrap(True)
        self.lbl_fs.setStyleSheet("color: #0284c7; font-size: 8.5pt;")
        vbox_in.addWidget(self.lbl_fs)

        vbox_in.addWidget(QLabel("<b>Dataset 2 (Nanosecond / ns):</b>"))
        h_ns = QHBoxLayout()
        self.txt_ns = QLineEdit()
        self.txt_ns.setReadOnly(True)
        self.txt_ns.setPlaceholderText("Select Nanosecond dataset file...")
        if self.ds_ns is not None:
            src_ns = str(getattr(self.ds_ns, "source", "NS Dataset"))
            self.txt_ns.setText(Path(src_ns).name)
            self.txt_ns.setToolTip(src_ns)
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse_dataset_ns)
        h_ns.addWidget(self.txt_ns)
        h_ns.addWidget(btn_browse)
        vbox_in.addLayout(h_ns)

        self.ctrl_layout.addWidget(grp_inputs)

        # --- Group 2: Spectral Alignment & Exclusion ---
        grp_align = QGroupBox("2. Spectral Alignment & Exclusion")
        vbox_al = QVBoxLayout(grp_align)

        self.lbl_wl_range = QLabel("Common WL Range: --")
        self.lbl_wl_range.setStyleSheet("color: #475569; font-size: 8.5pt;")
        vbox_al.addWidget(self.lbl_wl_range)

        self.chk_align_wl = QCheckBox("Linear Wavelength Shift (m * λ + b)")
        self.chk_align_wl.setChecked(True)
        vbox_al.addWidget(self.chk_align_wl)

        self.lbl_shift_params = QLabel("Fitted: m = 1.0000, b = 0.00 nm")
        self.lbl_shift_params.setStyleSheet("color: #15803d; font-size: 8.5pt; margin-left: 16px;")
        vbox_al.addWidget(self.lbl_shift_params)

        self.chk_scatter = QCheckBox("Exclude laser scatter band:")
        vbox_al.addWidget(self.chk_scatter)

        h_sc = QHBoxLayout()
        self.spin_sc_min = QDoubleSpinBox()
        self.spin_sc_min.setRange(200.0, 3000.0)
        self.spin_sc_min.setValue(500.0)
        self.spin_sc_max = QDoubleSpinBox()
        self.spin_sc_max.setRange(200.0, 3000.0)
        self.spin_sc_max.setValue(550.0)
        self.spin_sc_min.setEnabled(False)
        self.spin_sc_max.setEnabled(False)
        self.chk_scatter.toggled.connect(self.spin_sc_min.setEnabled)
        self.chk_scatter.toggled.connect(self.spin_sc_max.setEnabled)
        h_sc.addWidget(QLabel("Min:"))
        h_sc.addWidget(self.spin_sc_min)
        h_sc.addWidget(QLabel("Max:"))
        h_sc.addWidget(self.spin_sc_max)
        vbox_al.addLayout(h_sc)

        self.ctrl_layout.addWidget(grp_align)

        # --- Group 3: Overlap Scaling & Blend Window ---
        grp_scale = QGroupBox("3. Overlap Scaling & Blend Window")
        vbox_sc = QVBoxLayout(grp_scale)

        vbox_sc.addWidget(QLabel("Delay Overlap Window (ns):"))
        h_ov = QHBoxLayout()
        self.spin_t_ov_min = QDoubleSpinBox()
        self.spin_t_ov_min.setRange(0.001, 10000.0)
        self.spin_t_ov_min.setDecimals(3)
        self.spin_t_ov_min.setValue(0.5)
        self.spin_t_ov_max = QDoubleSpinBox()
        self.spin_t_ov_max.setRange(0.001, 10000.0)
        self.spin_t_ov_max.setDecimals(3)
        self.spin_t_ov_max.setValue(2.0)
        h_ov.addWidget(QLabel("Min:"))
        h_ov.addWidget(self.spin_t_ov_min)
        h_ov.addWidget(QLabel("Max:"))
        h_ov.addWidget(self.spin_t_ov_max)
        vbox_sc.addLayout(h_ov)

        vbox_sc.addWidget(QLabel("Scaling Model:"))
        self.model_group = QButtonGroup(self)
        self.rad_quad = QRadioButton("Quadratic S(λ) model")
        self.rad_scalar = QRadioButton("Single scalar C")
        self.rad_quad.setChecked(True)
        self.model_group.addButton(self.rad_quad)
        self.model_group.addButton(self.rad_scalar)
        vbox_sc.addWidget(self.rad_quad)
        vbox_sc.addWidget(self.rad_scalar)

        vbox_sc.addWidget(QLabel("Blend Transition Window (ns):"))
        h_bl = QHBoxLayout()
        self.spin_blend_start = QDoubleSpinBox()
        self.spin_blend_start.setRange(0.001, 10000.0)
        self.spin_blend_start.setDecimals(3)
        self.spin_blend_start.setValue(1.0)
        self.spin_blend_end = QDoubleSpinBox()
        self.spin_blend_end.setRange(0.001, 10000.0)
        self.spin_blend_end.setDecimals(3)
        self.spin_blend_end.setValue(1.7)
        h_bl.addWidget(QLabel("Start:"))
        h_bl.addWidget(self.spin_blend_start)
        h_bl.addWidget(QLabel("End:"))
        h_bl.addWidget(self.spin_blend_end)
        vbox_md = vbox_sc  # alias
        vbox_sc.addLayout(h_bl)

        self.ctrl_layout.addWidget(grp_scale)

        # --- Group 4: Preview Controls ---
        grp_prev = QGroupBox("Preview Controls")
        vbox_pr = QVBoxLayout(grp_prev)

        h_wl = QHBoxLayout()
        h_wl.addWidget(QLabel("Preview Probe (nm):"))
        self.spin_preview_wl = QDoubleSpinBox()
        self.spin_preview_wl.setRange(100.0, 5000.0)
        self.spin_preview_wl.setDecimals(1)
        probe = self.ds_fs.probe
        self.spin_preview_wl.setValue(float(probe[len(probe) // 2]))
        h_wl.addWidget(self.spin_preview_wl)
        vbox_pr.addLayout(h_wl)

        self.ctrl_layout.addWidget(grp_prev)

        # --- Group 5: Action Buttons ---
        grp_act = QGroupBox("Actions & Export")
        vbox_act = QVBoxLayout(grp_act)

        self.btn_apply = QPushButton("Apply Stitched Dataset to Workspace")
        self.btn_apply.setStyleSheet(
            "font-weight: bold; background-color: #2563eb; color: white; padding: 7px; border-radius: 4px;"
        )
        self.btn_apply.clicked.connect(lambda: self._apply_to_workspace(prompt_confirmation=True))
        vbox_act.addWidget(self.btn_apply)

        self.btn_export = QPushButton("Export Stitched Dataset...")
        self.btn_export.clicked.connect(self._export_stitched)
        vbox_act.addWidget(self.btn_export)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        vbox_act.addWidget(btn_close)

        self.ctrl_layout.addWidget(grp_act)
        self.ctrl_layout.addStretch()

        # Connect signals
        self.chk_align_wl.toggled.connect(self._update_stitching)
        self.chk_scatter.toggled.connect(self._update_stitching)
        self.spin_sc_min.valueChanged.connect(self._update_stitching)
        self.spin_sc_max.valueChanged.connect(self._update_stitching)

        self.spin_t_ov_min.valueChanged.connect(self._update_stitching)
        self.spin_t_ov_max.valueChanged.connect(self._update_stitching)
        self.rad_quad.toggled.connect(self._update_stitching)
        self.rad_scalar.toggled.connect(self._update_stitching)
        self.spin_blend_start.valueChanged.connect(self._update_stitching)
        self.spin_blend_end.valueChanged.connect(self._update_stitching)

        self.spin_preview_wl.valueChanged.connect(self._update_plots)

    def _browse_dataset_ns(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Long-Delay Dataset (Nanosecond / Microsecond)",
            "",
            "Transient Absorption (*.pdat *.h5 *.mat *.csv);;All Files (*)",
        )
        if filepath:
            try:
                self.ds_ns = Dataset1D.from_file(filepath)
                self.txt_ns.setText(Path(filepath).name)
                self.txt_ns.setToolTip(str(filepath))
                self._init_overlap_parameters()
                self._update_stitching()
            except Exception as exc:
                logger.exception("Failed to load NS dataset: %s", exc)
                QMessageBox.critical(self, "Load Error", f"Failed to load NS dataset:\n{exc}")

    def _init_overlap_parameters(self):
        """Set default parameters based on spectral coverage."""
        if self.ds_fs is None or self.ds_ns is None:
            return

        wl1 = np.asarray(self.ds_fs._detector_probe(0), dtype=float)
        wl2 = np.asarray(self.ds_ns._detector_probe(0), dtype=float)

        ov_min = max(float(np.min(wl1)), float(np.min(wl2)))
        ov_max = min(float(np.max(wl1)), float(np.max(wl2)))

        if ov_min >= ov_max:
            self.lbl_wl_range.setText("No spectral overlap detected!")
            self.lbl_wl_range.setStyleSheet("color: #dc2626; font-weight: bold;")
            return

        self.lbl_wl_range.setText(f"Common WL Range: {ov_min:.1f} – {ov_max:.1f} nm")
        self._block_signals = True
        self.spin_preview_wl.setValue((ov_min + ov_max) / 2.0)
        self._block_signals = False

    def _update_stitching(self):
        """Run stitch_time with current parameters and update plots."""
        if self._block_signals or self.ds_fs is None or self.ds_ns is None:
            return

        try:
            t_ov = (self.spin_t_ov_min.value(), self.spin_t_ov_max.value())
            t_bl = (self.spin_blend_start.value(), self.spin_blend_end.value())
            scale_mdl = "quadratic" if self.rad_quad.isChecked() else "scalar"
            do_align = self.chk_align_wl.isChecked()
            cut_sc = self.chk_scatter.isChecked()
            sc_rgn = (self.spin_sc_min.value(), self.spin_sc_max.value())

            self.result = stitch_time(
                self.ds_fs,
                self.ds_ns,
                time_overlap=t_ov,
                blend_window=t_bl,
                scale_model=scale_mdl,
                align_wl=do_align,
                cut_scatter=cut_sc,
                scatter_region=sc_rgn,
            )

            # Update linear shift readout
            if self.result.wl_shift_params is not None:
                m, b = self.result.wl_shift_params
                self.lbl_shift_params.setText(f"Fitted: m = {m:.4f}, b = {b:.2f} nm")
            else:
                self.lbl_shift_params.setText("Shift: Disabled (1.0 * λ + 0.0)")

            self._update_plots()

        except Exception as exc:
            logger.warning("Timescale stitching update failed: %s", exc)

    def _update_plots(self):
        """Render diagnostic kinetic trace, scaling factor curve, and 2D stitched contour map."""
        if self.result is None:
            return

        fig = self.canvas.figure
        fig.clear()
        fig.set_layout_engine(None)

        gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.25], width_ratios=[1.6, 1.0], hspace=0.36, wspace=0.28)

        target_wl = float(self.spin_preview_wl.value())
        stitched_ds = self.result.dataset
        master_wl = stitched_ds._detector_probe(0)
        idx_p = int(np.argmin(np.abs(master_wl - target_wl)))
        actual_wl = master_wl[idx_p]

        # ----------------------------------------------------------------- #
        # Top-Left: Kinetic Cross-Section                                   #
        # ----------------------------------------------------------------- #
        ax_kin = fig.add_subplot(gs[0, 0])

        fs_t = self.ds_fs.delays
        if "ps" in str(self.ds_fs.units.get("unitsT_ltx", "ps")).lower() or float(np.max(fs_t)) > 100.0:
            fs_t_ns = fs_t / 1000.0
        else:
            fs_t_ns = np.copy(fs_t)

        ns_t = self.ds_ns.delays

        # Extract kinetic slices
        idx_p_fs = int(np.argmin(np.abs(self.ds_fs._detector_probe(0) - actual_wl)))
        idx_p_ns = int(np.argmin(np.abs(self.ds_ns._detector_probe(0) - actual_wl)))

        fs_raw_trace = self.ds_fs._detector_slice(0)[:, idx_p_fs]
        ns_raw_trace = self.ds_ns._detector_slice(0)[:, idx_p_ns]

        s_fac = self.result.scaling_factor
        scale_val = s_fac[idx_p] if isinstance(s_fac, np.ndarray) else float(s_fac)
        fs_scaled_trace = fs_raw_trace * scale_val

        stitched_trace = stitched_ds._detector_slice(0)[:, idx_p]
        joined_t = stitched_ds.delays

        ax_kin.plot(fs_t_ns, fs_raw_trace, ".", color="#94a3b8", markersize=3.5, label="FS Raw (Unscaled)")
        ax_kin.plot(ns_t, ns_raw_trace, "o", color="#dc2626", markersize=3.5, fillstyle="none", label="NS Raw")
        ax_kin.plot(fs_t_ns, fs_scaled_trace, "--", color="#0284c7", linewidth=1.3, label=f"FS Scaled (x {scale_val:.2f})")
        ax_kin.plot(joined_t, stitched_trace, "-", color="#16a34a", linewidth=2.0, label="Merged Kinetics")

        if self.result.blend_window is not None:
            t_b_start, t_b_end = self.result.blend_window
            ax_kin.axvspan(t_b_start, t_b_end, color="#bbf7d0", alpha=0.35, label=f"Blend ({t_b_start:.1f}–{t_b_end:.1f} ns)")
            ax_kin.axvline(t_b_start, color="#16a34a", linestyle="--", linewidth=1.0)
            ax_kin.axvline(t_b_end, color="#16a34a", linestyle="--", linewidth=1.0)

        ax_kin.set_xscale("symlog", linthresh=0.5)
        ax_kin.set_xlim(-0.005, 50.0)
        u_z = stitched_ds.units.get("unitsZ_lbl", r"$\Delta$A")
        ax_kin.set_xlabel("Delay Time (ns) [Symlog]", fontsize=9.5, fontweight="bold")
        ax_kin.set_ylabel(f"{u_z} ({stitched_ds.units.get('unitsZ_ltx', 'mOD')})", fontsize=9.5, fontweight="bold")
        ax_kin.set_title(f"Diagnostic Kinetic Trace at {actual_wl:.1f} nm", fontsize=10.5, fontweight="bold")
        ax_kin.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
        ax_kin.grid(True, linestyle="--", alpha=0.4)

        # ----------------------------------------------------------------- #
        # Top-Right: Scaling Factor S(lambda)                               #
        # ----------------------------------------------------------------- #
        ax_s = fig.add_subplot(gs[0, 1])
        if isinstance(self.result.scaling_factor, np.ndarray):
            individual = self.result.scaling_params.get("individual_ratios")
            if individual is not None:
                ax_s.plot(master_wl, individual, ".", color="#94a3b8", markersize=3.5, label="Individual WL Fit")
            ax_s.plot(master_wl, self.result.scaling_factor, color="#2563eb", linewidth=2.0, label="Quadratic Fit S(λ)")
        else:
            ax_s.axhline(float(self.result.scaling_factor), color="#2563eb", linewidth=2.0, label=f"Scalar C = {float(self.result.scaling_factor):.3f}")

        ax_s.axhline(1.0, color="#64748b", linestyle=":", linewidth=0.8)
        ax_s.set_xlabel("Wavelength (nm)", fontsize=9.5, fontweight="bold")
        ax_s.set_ylabel("Scaling Factor", fontsize=9.5, fontweight="bold")
        ax_s.set_title("Wavelength Scaling Factor", fontsize=10.5, fontweight="bold")
        ax_s.legend(loc="best", fontsize=8)
        ax_s.grid(True, linestyle="--", alpha=0.4)

        # ----------------------------------------------------------------- #
        # Bottom: Full Stitched 2D Map                                      #
        # ----------------------------------------------------------------- #
        ax_map = fig.add_subplot(gs[1, :])
        mat = stitched_ds._detector_slice(0)
        vmax = float(np.nanpercentile(np.abs(mat), 98.5))
        if vmax <= 0:
            vmax = 1.0
        vmin = -vmax

        p_grid, t_mesh = np.meshgrid(master_wl, joined_t)
        c = ax_map.pcolormesh(p_grid, t_mesh, mat, cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="auto")

        if self.result.blend_window is not None:
            t_b_start, t_b_end = self.result.blend_window
            ax_map.axhline(t_b_start, color="#15803d", linestyle="--", linewidth=1.2)
            ax_map.axhline(t_b_end, color="#15803d", linestyle="--", linewidth=1.2)
            ax_map.text(master_wl[20], t_b_start * 0.25, "Femtosecond Region (< 1 ns)", color="#0369a1", fontsize=9, fontweight="bold")
            ax_map.text(master_wl[20], t_b_end * 6.0, "Nanosecond Region (> 1.7 ns)", color="#b91c1c", fontsize=9, fontweight="bold")

        ax_map.set_yscale("symlog", linthresh=0.5)
        ax_map.set_xlim(float(np.min(master_wl)), float(np.max(master_wl)))
        ax_map.set_ylim(-0.005, float(np.max(joined_t)))
        u_l = stitched_ds.units.get("unitsL_lbl", "Probe")
        ax_map.set_xlabel(f"{u_l} (nm)", fontsize=9.5, fontweight="bold")
        ax_map.set_ylabel("Delay Time (ns) [Symlog]", fontsize=9.5, fontweight="bold")
        ax_map.set_title(
            f"Full Stitched 2D Map: Femtosecond to Microsecond Delays (-2 ps to {np.max(joined_t)/1000.0:.1f} µs)",
            fontsize=10.5,
            fontweight="bold",
        )

        cbar = fig.colorbar(c, ax=ax_map, orientation="vertical", fraction=0.025, pad=0.02)
        cbar.set_label(f"{u_z} ({stitched_ds.units.get('unitsZ_ltx', 'mOD')})", fontsize=8.5)
        cbar.ax.tick_params(labelsize=8)

        fig.subplots_adjust(left=0.10, right=0.91, top=0.93, bottom=0.10, hspace=0.36, wspace=0.28)
        self.canvas.draw()

    def _apply_to_workspace(self, *, prompt_confirmation: bool = True):
        """Emit stitched dataset signal and update active workspace."""
        if self.result is None:
            return
        if prompt_confirmation:
            reply = QMessageBox.question(
                self,
                "Apply Stitched Dataset",
                "Replace the active dataset in the workspace with this timescale-stitched dataset?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self.stitch_applied.emit(self.result.dataset)
        self.accept()

    def _export_stitched(self):
        """Export the stitched dataset to file."""
        if self.result is None:
            return
        filepath, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Timescale-Stitched Dataset",
            "timescale_stitched.pdat",
            "PDAT format (*.pdat);;PyMORGAN HDF5 (*.h5);;CSV Matrix (*.csv)",
        )
        if not filepath:
            return

        try:
            self.result.export(filepath)
            QMessageBox.information(self, "Export Complete", f"Stitched dataset exported to:\n{filepath}")
        except Exception as exc:
            logger.exception("Export failed: %s", exc)
            QMessageBox.critical(self, "Export Failed", f"Could not export data:\n{exc}")

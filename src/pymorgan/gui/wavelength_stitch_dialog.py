"""PyQt6 Dialog for Wavelength Stitching (UV/Vis + NIR spectral window merging).

Allows user to align delays, fit overlap scaling factors, and stitch two spectral
windows using either a weighted linear cross-fade ramp (Blend) or direct
concatenation (Hard Cutover).
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
from pymorgan.oneD.stitch import StitchResult, stitch_wavelength

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class WavelengthStitchDialog(QDialog):
    """Dialog for Wavelength Stitching of multiple spectral windows."""

    stitch_applied = pyqtSignal(object)

    def __init__(self, parent: QWidget | None, dataset1: Dataset1D, dataset2: Dataset1D | None = None):
        super().__init__(parent)
        self.ds1 = dataset1
        self.ds2 = dataset2
        self.result: StitchResult | None = None
        self._block_signals = False

        self.setWindowTitle(
            f"Wavelength Stitching (UV/Vis + NIR) \u2014 {getattr(dataset1, 'source', None) or 'Dataset 1'}"
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
        if self.ds1 is not None and self.ds2 is not None:
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
        hdr = QLabel("<b>Wavelength Stitching</b>")
        hdr.setStyleSheet("font-size: 13pt; margin-bottom: 2px;")
        self.ctrl_layout.addWidget(hdr)

        subtitle = QLabel("<b>UV/Vis + NIR Spectral Window Merging</b>")
        subtitle.setStyleSheet("color: #2563eb; font-size: 9.5pt; font-weight: bold;")
        self.ctrl_layout.addWidget(subtitle)

        desc = QLabel("Aligns delay sampling, optimizes overlap scaling, and merges spectra.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #64748b; font-size: 8.5pt; margin-bottom: 4px;")
        self.ctrl_layout.addWidget(desc)

        # --- Group 1: Input Datasets ---
        grp_inputs = QGroupBox("1. Input Datasets")
        vbox_in = QVBoxLayout(grp_inputs)

        vbox_in.addWidget(QLabel("<b>Dataset 1 (UV/Vis / Short &lambda;, Active):</b>"))
        src1 = str(getattr(self.ds1, "source", "Dataset 1"))
        name1 = Path(src1).name if src1 else "Dataset 1"
        self.lbl_ds1 = QLabel(f"• {name1}")
        self.lbl_ds1.setToolTip(src1)
        self.lbl_ds1.setWordWrap(True)
        self.lbl_ds1.setStyleSheet("color: #0284c7; font-size: 8.5pt;")
        vbox_in.addWidget(self.lbl_ds1)

        vbox_in.addWidget(QLabel("<b>Dataset 2 (NIR / Long &lambda;):</b>"))
        h_ds2 = QHBoxLayout()
        self.txt_ds2 = QLineEdit()
        self.txt_ds2.setReadOnly(True)
        self.txt_ds2.setPlaceholderText("Select Dataset 2 file...")
        if self.ds2 is not None:
            src2 = str(getattr(self.ds2, "source", "Dataset 2"))
            self.txt_ds2.setText(Path(src2).name)
            self.txt_ds2.setToolTip(src2)
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse_dataset2)
        h_ds2.addWidget(self.txt_ds2)
        h_ds2.addWidget(btn_browse)
        vbox_in.addLayout(h_ds2)

        self.ctrl_layout.addWidget(grp_inputs)

        # --- Group 2: Overlap & Scaling Optimization ---
        grp_scale = QGroupBox("2. Overlap & Scaling Optimization")
        vbox_sc = QVBoxLayout(grp_scale)

        self.lbl_overlap = QLabel("Detected Spectral Overlap: --")
        self.lbl_overlap.setStyleSheet("color: #475569; font-size: 8.5pt;")
        vbox_sc.addWidget(self.lbl_overlap)

        vbox_sc.addWidget(QLabel("Scaling Fit Window (nm):"))
        h_fit = QHBoxLayout()
        self.spin_fit_min = QDoubleSpinBox()
        self.spin_fit_min.setRange(100.0, 5000.0)
        self.spin_fit_min.setDecimals(1)
        self.spin_fit_max = QDoubleSpinBox()
        self.spin_fit_max.setRange(100.0, 5000.0)
        self.spin_fit_max.setDecimals(1)
        h_fit.addWidget(QLabel("Min:"))
        h_fit.addWidget(self.spin_fit_min)
        h_fit.addWidget(QLabel("Max:"))
        h_fit.addWidget(self.spin_fit_max)
        vbox_sc.addLayout(h_fit)

        self.lbl_scaling_factor = QLabel("Optimal Scaling Factor C: --")
        self.lbl_scaling_factor.setStyleSheet("font-weight: bold; color: #15803d; font-size: 9pt; margin-top: 2px;")
        vbox_sc.addWidget(self.lbl_scaling_factor)

        h_ov = QHBoxLayout()
        self.chk_override = QCheckBox("Manual override:")
        self.spin_override = QDoubleSpinBox()
        self.spin_override.setRange(0.001, 1000.0)
        self.spin_override.setDecimals(4)
        self.spin_override.setValue(1.0)
        self.spin_override.setEnabled(False)
        self.chk_override.toggled.connect(self.spin_override.setEnabled)
        h_ov.addWidget(self.chk_override)
        h_ov.addWidget(self.spin_override)
        vbox_sc.addLayout(h_ov)

        self.ctrl_layout.addWidget(grp_scale)

        # --- Group 3: Stitching / Transition Method ---
        grp_mode = QGroupBox("3. Stitching & Blending Method")
        vbox_md = QVBoxLayout(grp_mode)

        self.mode_group = QButtonGroup(self)
        self.rad_blend = QRadioButton("Mode A: Linear Cross-Fade (Blend)")
        self.rad_cutover = QRadioButton("Mode B: Hard Cutover (Concatenate)")
        self.rad_blend.setChecked(True)
        self.mode_group.addButton(self.rad_blend)
        self.mode_group.addButton(self.rad_cutover)

        vbox_md.addWidget(self.rad_blend)
        # Blend sub-controls
        h_bl = QHBoxLayout()
        self.spin_blend_start = QDoubleSpinBox()
        self.spin_blend_start.setRange(100.0, 5000.0)
        self.spin_blend_start.setDecimals(1)
        self.spin_blend_end = QDoubleSpinBox()
        self.spin_blend_end.setRange(100.0, 5000.0)
        self.spin_blend_end.setDecimals(1)
        h_bl.addWidget(QLabel("Start:"))
        h_bl.addWidget(self.spin_blend_start)
        h_bl.addWidget(QLabel("End:"))
        h_bl.addWidget(self.spin_blend_end)
        vbox_md.addLayout(h_bl)

        vbox_md.addWidget(self.rad_cutover)
        # Cutover sub-controls
        h_cut = QHBoxLayout()
        h_cut.addWidget(QLabel("Cutover point (nm):"))
        self.spin_cutover = QDoubleSpinBox()
        self.spin_cutover.setRange(100.0, 5000.0)
        self.spin_cutover.setDecimals(1)
        self.spin_cutover.setEnabled(False)
        h_cut.addWidget(self.spin_cutover)
        vbox_md.addLayout(h_cut)

        self.ctrl_layout.addWidget(grp_mode)

        # --- Group 4: Preview Controls ---
        grp_prev = QGroupBox("Preview Controls")
        vbox_pr = QVBoxLayout(grp_prev)

        h_t = QHBoxLayout()
        h_t.addWidget(QLabel("Preview Delay:"))
        self.spin_preview_delay = QDoubleSpinBox()
        self.spin_preview_delay.setRange(-1e6, 1e6)
        self.spin_preview_delay.setDecimals(2)
        self.spin_preview_delay.setValue(1.0)
        u_t = self.ds1.units.get("unitsT_ltx", "ps")
        self.spin_preview_delay.setSuffix(f" {u_t}")
        h_t.addWidget(self.spin_preview_delay)
        vbox_pr.addLayout(h_t)

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
        self.spin_fit_min.valueChanged.connect(self._update_stitching)
        self.spin_fit_max.valueChanged.connect(self._update_stitching)
        self.chk_override.toggled.connect(self._update_stitching)
        self.spin_override.valueChanged.connect(self._update_stitching)

        self.rad_blend.toggled.connect(self._on_mode_changed)
        self.rad_cutover.toggled.connect(self._on_mode_changed)
        self.spin_blend_start.valueChanged.connect(self._update_stitching)
        self.spin_blend_end.valueChanged.connect(self._update_stitching)
        self.spin_cutover.valueChanged.connect(self._update_stitching)
        self.spin_preview_delay.valueChanged.connect(self._update_plots)

    def _on_mode_changed(self):
        is_blend = self.rad_blend.isChecked()
        self.spin_blend_start.setEnabled(is_blend)
        self.spin_blend_end.setEnabled(is_blend)
        self.spin_cutover.setEnabled(not is_blend)
        self._update_stitching()

    def _browse_dataset2(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Second Spectral Dataset (NIR / Long Wavelength)",
            "",
            "Transient Absorption (*.pdat *.h5 *.mat *.csv);;All Files (*)",
        )
        if filepath:
            try:
                self.ds2 = Dataset1D.from_file(filepath)
                self.txt_ds2.setText(Path(filepath).name)
                self.txt_ds2.setToolTip(str(filepath))
                self._init_overlap_parameters()
                self._update_stitching()
            except Exception as exc:
                logger.exception("Failed to load Dataset 2: %s", exc)
                QMessageBox.critical(self, "Load Error", f"Failed to load Dataset 2:\n{exc}")

    def _init_overlap_parameters(self):
        """Detect overlap between ds1 and ds2 and set default spinbox ranges."""
        if self.ds1 is None or self.ds2 is None:
            return

        wl1 = np.asarray(self.ds1._detector_probe(0), dtype=float)
        wl2 = np.asarray(self.ds2._detector_probe(0), dtype=float)

        ov_min = max(float(np.min(wl1)), float(np.min(wl2)))
        ov_max = min(float(np.max(wl1)), float(np.max(wl2)))

        if ov_min >= ov_max:
            self.lbl_overlap.setText("No spectral overlap detected!")
            self.lbl_overlap.setStyleSheet("color: #dc2626; font-weight: bold;")
            return

        self.lbl_overlap.setText(f"Detected Overlap: {ov_min:.1f} – {ov_max:.1f} nm")
        self.lbl_overlap.setStyleSheet("color: #475569; font-size: 8.5pt;")

        span = ov_max - ov_min
        fit_min = ov_min + 0.2 * span
        fit_max = ov_max - 0.2 * span
        blend_start = ov_min + 0.1 * span
        blend_end = ov_max - 0.1 * span
        cutover = (ov_min + ov_max) / 2.0

        self._block_signals = True
        self.spin_fit_min.setValue(fit_min)
        self.spin_fit_max.setValue(fit_max)
        self.spin_blend_start.setValue(blend_start)
        self.spin_blend_end.setValue(blend_end)
        self.spin_cutover.setValue(cutover)
        self._block_signals = False

    def _update_stitching(self):
        """Run stitch_wavelength with current parameters and update plots."""
        if self._block_signals or self.ds1 is None or self.ds2 is None:
            return

        try:
            fit_rgn = (self.spin_fit_min.value(), self.spin_fit_max.value())
            mode = "blend" if self.rad_blend.isChecked() else "cutover"
            blend_win = (self.spin_blend_start.value(), self.spin_blend_end.value())
            cut_val = self.spin_cutover.value()
            override = self.spin_override.value() if self.chk_override.isChecked() else None

            self.result = stitch_wavelength(
                self.ds1,
                self.ds2,
                stitch_region=fit_rgn,
                mode=mode,
                blend_window=blend_win,
                cutover_wl=cut_val,
                scale_override=override,
            )

            # Update scaling readout
            s_factor = self.result.scaling_factor
            if isinstance(s_factor, (float, int, np.floating)):
                self.lbl_scaling_factor.setText(f"Optimal Scaling Factor C:  {float(s_factor):.4f}")
                if not self.chk_override.isChecked():
                    self.spin_override.setValue(float(s_factor))

            self._update_plots()

        except Exception as exc:
            logger.warning("Wavelength stitching update failed: %s", exc)

    def _update_plots(self):
        """Render diagnostic spectral slice and 2D stitched contour map."""
        if self.result is None:
            return

        fig = self.canvas.figure
        fig.clear()
        fig.set_layout_engine(None)

        gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.25], hspace=0.35)

        # ----------------------------------------------------------------- #
        # Top Panel: Diagnostic Spectral Cross-Section                      #
        # ----------------------------------------------------------------- #
        ax_spec = fig.add_subplot(gs[0])
        t_inspect = float(self.spin_preview_delay.value())

        stitched_ds = self.result.dataset
        t_grid = stitched_ds.delays
        idx_t = int(np.argmin(np.abs(t_grid - t_inspect)))
        actual_t = t_grid[idx_t]

        # Extract spectral cross-sections
        wl1 = self.ds1._detector_probe(0)
        wl2 = self.ds2._detector_probe(0)

        # Slice ds1 and ds2 at t_inspect
        idx_t1 = int(np.argmin(np.abs(self.ds1.delays - actual_t)))
        idx_t2 = int(np.argmin(np.abs(self.ds2.delays - actual_t)))

        spec1_raw = self.ds1._detector_slice(0)[idx_t1, :]
        spec2_raw = self.ds2._detector_slice(0)[idx_t2, :]
        scale_c = float(self.result.scaling_factor) if np.isscalar(self.result.scaling_factor) else 1.0
        spec1_scaled = spec1_raw * scale_c

        spec_stitched = stitched_ds._detector_slice(0)[idx_t, :]
        wl_stitched = stitched_ds._detector_probe(0)

        ax_spec.plot(wl1, spec1_raw, color="#94a3b8", linestyle=":", linewidth=1.2, label="Dataset 1 Raw (Unscaled)")
        ax_spec.plot(wl1, spec1_scaled, color="#0284c7", linewidth=1.4, label=f"Dataset 1 Scaled (x {scale_c:.3f})")
        ax_spec.plot(wl2, spec2_raw, color="#dc2626", linewidth=1.4, label="Dataset 2 Raw")
        ax_spec.plot(wl_stitched, spec_stitched, color="#16a34a", linewidth=2.0, label="Stitched Result (Merged)")

        # Blend or Cutover markers
        if self.result.blend_window is not None:
            b_start, b_end = self.result.blend_window
            ax_spec.axvspan(b_start, b_end, color="#bbf7d0", alpha=0.35, label=f"Blend Region ({b_start:.0f}–{b_end:.0f} nm)")
            ax_spec.axvline(b_start, color="#16a34a", linestyle="--", linewidth=1.0)
            ax_spec.axvline(b_end, color="#16a34a", linestyle="--", linewidth=1.0)
        elif self.result.cutover_val is not None:
            c_val = self.result.cutover_val
            ax_spec.axvline(c_val, color="#16a34a", linestyle="--", linewidth=1.4, label=f"Cutover ({c_val:.1f} nm)")

        u_l = stitched_ds.units.get("unitsL_lbl", "Probe")
        u_z = stitched_ds.units.get("unitsZ_lbl", r"$\Delta$A")
        ax_spec.set_xlabel(f"{u_l} (nm)", fontsize=9.5, fontweight="bold")
        ax_spec.set_ylabel(f"{u_z} ({stitched_ds.units.get('unitsZ_ltx', 'mOD')})", fontsize=9.5, fontweight="bold")
        ax_spec.set_title(
            f"Diagnostic Spectral Alignment & Stitch Transition (Delay = {actual_t:.2f} {stitched_ds.units.get('unitsT_ltx', 'ps')})",
            fontsize=10.5,
            fontweight="bold",
        )
        ax_spec.legend(loc="upper right", fontsize=8.5, framealpha=0.9, ncol=2)
        ax_spec.grid(True, linestyle="--", alpha=0.4)

        # ----------------------------------------------------------------- #
        # Bottom Panel: Full Stitched 2D Map                                #
        # ----------------------------------------------------------------- #
        ax_map = fig.add_subplot(gs[1])
        mat = stitched_ds._detector_slice(0)
        vmax = float(np.nanpercentile(np.abs(mat), 98.5))
        if vmax <= 0:
            vmax = 1.0
        vmin = -vmax

        p_grid, t_mesh = np.meshgrid(wl_stitched, t_grid)
        c = ax_map.pcolormesh(p_grid, t_mesh, mat, cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="auto")

        if self.result.blend_window is not None:
            b_start, b_end = self.result.blend_window
            ax_map.axvline(b_start, color="#15803d", linestyle="--", linewidth=1.2)
            ax_map.axvline(b_end, color="#15803d", linestyle="--", linewidth=1.2)
        elif self.result.cutover_val is not None:
            c_val = self.result.cutover_val
            ax_map.axvline(c_val, color="#15803d", linestyle="--", linewidth=1.4)

        u_t = stitched_ds.units.get("unitsT_lbl", "Delay")
        ax_map.set_yscale("symlog", linthresh=1.0)
        ax_map.set_xlabel(f"{u_l} (nm)", fontsize=9.5, fontweight="bold")
        ax_map.set_ylabel(f"{u_t} ({stitched_ds.units.get('unitsT_ltx', 'ps')}) [Symlog]", fontsize=9.5, fontweight="bold")
        ax_map.set_title(
            f"Full Stitched 2D Transient Absorption Map ({wl_stitched[0]:.1f} – {wl_stitched[-1]:.1f} nm)",
            fontsize=10.5,
            fontweight="bold",
        )

        cbar = fig.colorbar(c, ax=ax_map, orientation="vertical", fraction=0.03, pad=0.02)
        cbar.set_label(f"{u_z} ({stitched_ds.units.get('unitsZ_ltx', 'mOD')})", fontsize=8.5)
        cbar.ax.tick_params(labelsize=8)

        fig.subplots_adjust(left=0.12, right=0.91, top=0.93, bottom=0.10, hspace=0.35)
        self.canvas.draw()

    def _apply_to_workspace(self, *, prompt_confirmation: bool = True):
        """Emit stitched dataset signal and update active workspace."""
        if self.result is None:
            return
        if prompt_confirmation:
            reply = QMessageBox.question(
                self,
                "Apply Stitched Dataset",
                "Replace the active dataset in the workspace with this wavelength-stitched dataset?",
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
            "Export Wavelength-Stitched Dataset",
            "wavelength_stitched.pdat",
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

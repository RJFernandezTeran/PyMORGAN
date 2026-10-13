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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
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
from pymorgan.oneD.plot import _CBAR_PAD_IN, _CBAR_WIDTH_IN
from pymorgan.oneD.stitch import StitchResult, stitch_wavelength

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class WavelengthStitchDialog(QDialog):
    """Dialog for Spectral Stitching of multiple probe windows."""

    stitch_applied = pyqtSignal(object)

    def __init__(self, parent: QWidget | None, dataset1: Dataset1D, dataset2: Dataset1D | None = None):
        super().__init__(parent)
        self.ds1 = dataset1
        self.ds2 = dataset2
        self.result: StitchResult | None = None
        self._block_signals = False

        self.setWindowTitle("Stitch Spectral Windows - PyMORGAN")
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

        # Configure splitter collapsibility after adding widgets
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([420, 780])

        # Initial calculation if both datasets are provided
        if self.ds1 is not None and self.ds2 is not None:
            self._init_overlap_parameters()
            self._update_stitching()

    def _probe_unit_display(self) -> str:
        """Return clean ASCII probe unit string ('cm-1', 'nm', 'eV', 'THz')."""
        if self.ds1 is None:
            return "nm"
        raw = str(getattr(self.ds1, "units", {}).get("unitsL_ltx", "nm")).strip()
        if "cm" in raw.lower():
            return "cm-1"
        elif "nm" in raw.lower():
            return "nm"
        elif "ev" in raw.lower():
            return "eV"
        elif "thz" in raw.lower():
            return "THz"
        return raw.replace("$", "").replace("\\mathrm", "").replace("{", "").replace("}", "").strip()

    def _probe_name(self) -> str:
        """Return semantic probe axis name based on units."""
        u = self._probe_unit_display()
        if "cm" in u:
            return "Wavenumber"
        elif "nm" in u:
            return "Wavelength"
        return "Spectral"

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
        probe_u = self._probe_unit_display()
        _probe_n = self._probe_name()

        # Header
        hdr = QLabel("<b>Stitch Spectral Windows</b>")
        hdr.setStyleSheet("font-size: 13pt; margin-bottom: 2px;")
        self.ctrl_layout.addWidget(hdr)

        subtitle = QLabel("<b>Merge Adjacent Probe Windows</b>")
        subtitle.setStyleSheet("color: #2563eb; font-size: 9.5pt; font-weight: bold;")
        self.ctrl_layout.addWidget(subtitle)

        desc = QLabel("Aligns delay sampling, optimizes overlap scaling, and blends adjacent probe windows.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #64748b; font-size: 8.5pt; margin-bottom: 4px;")
        self.ctrl_layout.addWidget(desc)

        # --- Group 1: Input Datasets && Units ---
        grp_inputs = QGroupBox("1. Input Datasets && Units")
        vbox_in = QVBoxLayout(grp_inputs)

        vbox_in.addWidget(QLabel("<b>Dataset 1 (Active):</b>"))
        src1 = str(getattr(self.ds1, "source", "Dataset 1"))
        name1 = Path(src1).name if src1 else "Dataset 1"
        self.lbl_ds1 = QLabel(f"* {name1} ({probe_u})")
        self.lbl_ds1.setToolTip(src1)
        self.lbl_ds1.setWordWrap(True)
        self.lbl_ds1.setStyleSheet("color: #0284c7; font-size: 8.5pt;")
        vbox_in.addWidget(self.lbl_ds1)

        vbox_in.addWidget(QLabel("<b>Dataset 2:</b>"))
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

        # --- Group 2: Overlap && Scaling Optimization ---
        grp_scale = QGroupBox("2. Overlap && Scaling Optimization")
        vbox_sc = QVBoxLayout(grp_scale)

        self.lbl_overlap = QLabel("Detected Spectral Overlap: --")
        self.lbl_overlap.setStyleSheet("color: #475569; font-size: 8.5pt;")
        vbox_sc.addWidget(self.lbl_overlap)

        self.lbl_fit_window = QLabel(f"Scaling Fit Window ({probe_u}):")
        vbox_sc.addWidget(self.lbl_fit_window)
        h_fit = QHBoxLayout()
        self.spin_fit_min = QDoubleSpinBox()
        self.spin_fit_min.setRange(-1e6, 1e6)
        self.spin_fit_min.setDecimals(2)
        self.spin_fit_max = QDoubleSpinBox()
        self.spin_fit_max.setRange(-1e6, 1e6)
        self.spin_fit_max.setDecimals(2)
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
        self.spin_override.setRange(0.0001, 10000.0)
        self.spin_override.setDecimals(4)
        self.spin_override.setValue(1.0)
        self.spin_override.setEnabled(False)
        self.chk_override.toggled.connect(self.spin_override.setEnabled)
        h_ov.addWidget(self.chk_override)
        h_ov.addWidget(self.spin_override)
        vbox_sc.addLayout(h_ov)

        self.ctrl_layout.addWidget(grp_scale)

        # --- Group 3: Stitching && Blending Method ---
        grp_mode = QGroupBox("3. Stitching && Blending Method")
        vbox_md = QVBoxLayout(grp_mode)

        self.mode_group = QButtonGroup(self)
        self.rad_blend = QRadioButton("Mode A: Blended Transition Window")
        self.rad_cutover = QRadioButton("Mode B: Hard Cutover (Single Point)")
        self.rad_blend.setChecked(True)
        self.mode_group.addButton(self.rad_blend)
        self.mode_group.addButton(self.rad_cutover)

        vbox_md.addWidget(self.rad_blend)

        # Blend function dropdown
        h_fn = QHBoxLayout()
        h_fn.addWidget(QLabel("Blend Function:"))
        self.combo_blend = QComboBox()
        self.combo_blend.addItem("Cubic Polynomial (Smoothstep)", "cubic")
        self.combo_blend.addItem("Quintic Polynomial (Smootherstep)", "quintic")
        self.combo_blend.addItem("Cosine S-Curve", "cosine")
        self.combo_blend.addItem("Linear Cross-Fade", "linear")
        self.combo_blend.addItem("Hard Step", "step")
        h_fn.addWidget(self.combo_blend)
        vbox_md.addLayout(h_fn)

        # Blend sub-controls
        h_bl = QHBoxLayout()
        self.spin_blend_start = QDoubleSpinBox()
        self.spin_blend_start.setRange(-1e6, 1e6)
        self.spin_blend_start.setDecimals(2)
        self.spin_blend_end = QDoubleSpinBox()
        self.spin_blend_end.setRange(-1e6, 1e6)
        self.spin_blend_end.setDecimals(2)
        h_bl.addWidget(QLabel("Start:"))
        h_bl.addWidget(self.spin_blend_start)
        h_bl.addWidget(QLabel("End:"))
        h_bl.addWidget(self.spin_blend_end)
        vbox_md.addLayout(h_bl)

        vbox_md.addWidget(self.rad_cutover)
        # Cutover sub-controls
        h_cut = QHBoxLayout()
        self.lbl_cutover = QLabel(f"Cutover point ({probe_u}):")
        h_cut.addWidget(self.lbl_cutover)
        self.spin_cutover = QDoubleSpinBox()
        self.spin_cutover.setRange(-1e6, 1e6)
        self.spin_cutover.setDecimals(2)
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

        h_asinh = QHBoxLayout()
        self.chk_asinh = QCheckBox("arcsinh scaling")
        self.chk_asinh.setChecked(False)
        self.spin_asinh_pct = QDoubleSpinBox()
        self.spin_asinh_pct.setRange(0.1, 50.0)
        self.spin_asinh_pct.setDecimals(1)
        self.spin_asinh_pct.setSingleStep(0.5)
        self.spin_asinh_pct.setValue(5.0)
        self.spin_asinh_pct.setSuffix(" %")
        self.spin_asinh_pct.setEnabled(False)
        self.chk_asinh.toggled.connect(self.spin_asinh_pct.setEnabled)
        h_asinh.addWidget(self.chk_asinh)
        h_asinh.addWidget(self.spin_asinh_pct)
        vbox_pr.addLayout(h_asinh)

        self.ctrl_layout.addWidget(grp_prev)

        # --- Group 5: Action Buttons ---
        grp_act = QGroupBox("Actions && Export")
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
        self.combo_blend.currentIndexChanged.connect(self._update_stitching)
        self.spin_blend_start.valueChanged.connect(self._update_stitching)
        self.spin_blend_end.valueChanged.connect(self._update_stitching)
        self.spin_cutover.valueChanged.connect(self._update_stitching)
        self.spin_preview_delay.valueChanged.connect(self._update_plots)
        self.chk_asinh.toggled.connect(self._update_plots)
        self.spin_asinh_pct.valueChanged.connect(self._update_plots)

    def _on_mode_changed(self):
        is_blend = self.rad_blend.isChecked()
        self.combo_blend.setEnabled(is_blend)
        self.spin_blend_start.setEnabled(is_blend)
        self.spin_blend_end.setEnabled(is_blend)
        self.spin_cutover.setEnabled(not is_blend)
        self._update_stitching()

    def _browse_dataset2(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Second Spectral Dataset (UV/Vis, NIR, or Mid-IR)",
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

        u_probe = self._probe_unit_display()

        if ov_min >= ov_max:
            self.lbl_overlap.setText("No spectral overlap detected!")
            self.lbl_overlap.setStyleSheet("color: #dc2626; font-weight: bold;")
            return

        self.lbl_overlap.setText(f"Detected Overlap: {ov_min:.1f} - {ov_max:.1f} {u_probe}")
        self.lbl_overlap.setStyleSheet("color: #475569; font-size: 8.5pt;")

        span = ov_max - ov_min
        fit_min = ov_min + 0.15 * span
        fit_max = ov_max - 0.15 * span
        blend_start = ov_min + 0.08 * span
        blend_end = ov_max - 0.08 * span
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
            blend_fn = str(self.combo_blend.currentData() or "cubic")
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
                blend_method=blend_fn,
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

        gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.25], hspace=0.40, top=0.88, bottom=0.09, left=0.10, right=0.96)

        # ----------------------------------------------------------------- #
        # Top Panel: Diagnostic Spectral Cross-Section                      #
        # ----------------------------------------------------------------- #
        ax_spec = fig.add_subplot(gs[0])

        # Align right edge with the bottom contour map (which uses divider colorbar)
        div_spec = make_axes_locatable(ax_spec)
        cax_spec_dummy = div_spec.append_axes("right", size=_CBAR_WIDTH_IN, pad=_CBAR_PAD_IN)
        cax_spec_dummy.axis("off")

        t_inspect = float(self.spin_preview_delay.value())
        stitched_ds = self.result.dataset
        t_grid = stitched_ds.delays
        idx_t = int(np.argmin(np.abs(t_grid - t_inspect)))
        actual_t = t_grid[idx_t]

        # Extract spectral cross-sections
        wl1 = self.ds1._detector_probe(0)
        wl2 = self.ds2._detector_probe(0)

        # Slice ds1 and ds2 at actual_t
        idx_t1 = int(np.argmin(np.abs(self.ds1.delays - actual_t)))
        idx_t2 = int(np.argmin(np.abs(self.ds2.delays - actual_t)))

        spec1_raw = self.ds1._detector_slice(0)[idx_t1, :]
        spec2_raw = self.ds2._detector_slice(0)[idx_t2, :]
        scale_c = float(self.result.scaling_factor) if np.isscalar(self.result.scaling_factor) else 1.0
        spec1_scaled = spec1_raw * scale_c

        spec_stitched = stitched_ds._detector_slice(0)[idx_t, :]
        wl_stitched = stitched_ds._detector_probe(0)

        ax_spec.axhline(0, color="0.75", linewidth=0.75)
        ax_spec.plot(wl1, spec1_raw, color="#94a3b8", linestyle=":", linewidth=1.2, label="Dataset 1 Raw (Unscaled)")
        ax_spec.plot(wl1, spec1_scaled, color="#2563eb", linestyle="-", linewidth=1.4, label=f"Dataset 1 Scaled (x {scale_c:.3f})")
        ax_spec.plot(wl2, spec2_raw, color="#dc2626", linestyle="-", linewidth=1.4, label="Dataset 2 Raw")
        ax_spec.plot(wl_stitched, spec_stitched, color="#0f172a", linestyle="-", linewidth=2.0, label="Stitched Result (Merged)")

        u = self._probe_unit_display()

        # Blend or Cutover markers
        if self.result.blend_window is not None:
            b_start, b_end = self.result.blend_window
            ax_spec.axvspan(b_start, b_end, color="#cbd5e1", alpha=0.35, label=f"Blend Region ({b_start:.1f}-{b_end:.1f} {u})")
            ax_spec.axvline(b_start, color="#64748b", linestyle="--", linewidth=0.9)
            ax_spec.axvline(b_end, color="#64748b", linestyle="--", linewidth=0.9)
        elif self.result.cutover_val is not None:
            c_val = self.result.cutover_val
            ax_spec.axvline(c_val, color="#64748b", linestyle="--", linewidth=1.2, label=f"Cutover ({c_val:.1f} {u})")

        u_l = stitched_ds.units.get("unitsL_lbl", self._probe_name())
        u_z_unit = str(stitched_ds.units.get("unitsZ_ltx", "mOD")).replace("$", "")
        u_t_unit = str(stitched_ds.units.get("unitsT_ltx", "ps")).replace("$", "")

        ax_spec.set_xlabel(f"{u_l} ({u})", fontsize=9.5, fontweight="bold")
        ax_spec.set_ylabel(r"$\Delta A$ (%s)" % u_z_unit, fontsize=9.5, fontweight="bold")
        ax_spec.set_xlim(float(np.min(wl_stitched)), float(np.max(wl_stitched)))
        ax_spec.set_title(
            f"Diagnostic Spectral Alignment & Stitch Transition (Delay = {actual_t:.2f} {u_t_unit})",
            fontsize=10.5,
            fontweight="bold",
            pad=34,
        )
        leg = ax_spec.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=3,
            fontsize=8.0,
            frameon=False,
            columnspacing=1.0,
            handletextpad=0.4,
        )
        if leg is not None:
            leg.set_draggable(True)

        # ----------------------------------------------------------------- #
        # Bottom Panel: Full Stitched 2D Map (PyMORGAN plot_contour)       #
        # ----------------------------------------------------------------- #
        ax_map = fig.add_subplot(gs[1])
        stitched_ds.plot_contour(
            ax=ax_map,
            Asinh=self.chk_asinh.isChecked(),
            asinh_pct=self.spin_asinh_pct.value(),
        )

        if self.result.blend_window is not None:
            b_start, b_end = self.result.blend_window
            ax_map.axvline(b_start, color="0.25", linestyle="--", linewidth=1.1)
            ax_map.axvline(b_end, color="0.25", linestyle="--", linewidth=1.1)
        elif self.result.cutover_val is not None:
            c_val = self.result.cutover_val
            ax_map.axvline(c_val, color="0.25", linestyle="--", linewidth=1.2)

        # Subtle delay inspection marker line
        ax_map.axhline(actual_t, color="#0284c7", linestyle=":", linewidth=1.0, alpha=0.85)

        ax_map.set_title(
            "Full Stitched 2D Transient Absorption Map",
            fontsize=10.5,
            fontweight="bold",
        )

        self.canvas.draw()

    def _apply_to_workspace(self, *, prompt_confirmation: bool = True):
        """Emit stitched dataset signal and update active workspace."""
        if self.result is None:
            return
        if prompt_confirmation:
            reply = QMessageBox.question(
                self,
                "Apply Stitched Dataset",
                "Replace the active dataset in the workspace with this spectral-stitched dataset?",
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
            "Export Stitched Dataset",
            "spectral_stitched.pdat",
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

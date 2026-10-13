"""PyQt6 Dialog for computing 1D Polarization Anisotropy.

Supports single multi-detector datasets (channel 1 = par, channel 2 = perp)
or pairing two separate datasets with G-factor correction, denominator thresholding,
and live kinetic/spectral preview.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from pymorgan.gui.canvas import MplCanvas
from pymorgan.log import get_logger
from pymorgan.oneD.dataset import Dataset1D

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class AnisotropyDialog(QDialog):
    """Dialog for computing and previewing 1D transient polarization anisotropy."""

    anisotropy_applied = pyqtSignal(object)

    def __init__(self, parent: QWidget | None, dataset: Dataset1D):
        super().__init__(parent)
        self.dataset = dataset
        self.perp_dataset: Dataset1D | None = None
        self.aniso_dataset: Dataset1D | None = None
        self.iso_dataset: Dataset1D | None = None
        self.components: dict[str, Dataset1D] = {}

        self.setWindowTitle("Polarization Anisotropy Calculation")
        self.resize(920, 700)

        layout = QVBoxLayout(self)

        # Header description
        hdr = QLabel(
            "<b>Calculate Polarization Anisotropy:</b><br>"
            "<i>r(t, &lambda;) = [I<sub>||</sub> - G &middot; I<sub>&perp;</sub>] / "
            "[I<sub>||</sub> + 2 G &middot; I<sub>&perp;</sub>]</i>"
        )
        hdr.setStyleSheet("font-size: 10pt; margin-bottom: 2px;")
        layout.addWidget(hdr)

        # PyMESS dataset badge if detected
        is_pymess = (dataset.data_type == "PyMESS_PP") or ("pymess_metadata" in (dataset.units or {}))
        if is_pymess:
            pm_badge = QLabel("<b>PyMESS Dataset:</b> Multi-state slow-modulation polarization channels detected.")
            pm_badge.setStyleSheet("color: #3b82f6; font-size: 8.5pt; margin-bottom: 4px;")
            layout.addWidget(pm_badge)

        # Settings
        settings_layout = QHBoxLayout()

        # Group 1: Source & Polarization Setup
        grp_src = QGroupBox("Signal Inputs")
        vbox_src = QVBoxLayout(grp_src)

        has_multi = getattr(dataset, "n_detectors", 1) >= 2
        self.rad_channels = QRadioButton("Multi-channel states (current dataset)")
        self.rad_file = QRadioButton("Separate perpendicular dataset from file")

        if has_multi:
            self.rad_channels.setChecked(True)
        else:
            self.rad_channels.setEnabled(False)
            self.rad_file.setChecked(True)

        vbox_src.addWidget(self.rad_channels)

        # Detector / Channel combos with state names
        h_chans = QHBoxLayout()
        h_chans.addWidget(QLabel("Parallel (I<sub>||</sub>):"))
        self.combo_par = QComboBox()
        h_chans.addWidget(self.combo_par)
        h_chans.addWidget(QLabel("Perpendicular (I<sub>&perp;</sub>):"))
        self.combo_perp = QComboBox()
        h_chans.addWidget(self.combo_perp)

        det_names = dataset.units.get("detectors") if dataset.units else None
        n_det = getattr(dataset, "n_detectors", 1)
        par_idx, perp_idx = 0, min(1, max(0, n_det - 1))

        for i in range(n_det):
            lbl = f"Detector {i+1}"
            if det_names and i < len(det_names) and det_names[i]:
                lbl = f"Channel {i+1}: {det_names[i]}"
                nm_lower = str(det_names[i]).lower()
                if "par" in nm_lower:
                    par_idx = i
                elif "perp" in nm_lower:
                    perp_idx = i
            self.combo_par.addItem(lbl, i)
            self.combo_perp.addItem(lbl, i)

        if n_det >= 2:
            self.combo_par.setCurrentIndex(par_idx)
            self.combo_perp.setCurrentIndex(perp_idx)
        vbox_src.addLayout(h_chans)

        vbox_src.addWidget(self.rad_file)
        h_file = QHBoxLayout()
        self.txt_file = QLineEdit()
        self.txt_file.setPlaceholderText("Select perpendicular dataset file...")
        self.txt_file.setReadOnly(True)
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse_perp_file)
        h_file.addWidget(self.txt_file)
        h_file.addWidget(btn_browse)
        vbox_src.addLayout(h_file)

        settings_layout.addWidget(grp_src)

        # Group 2: Parameters
        grp_params = QGroupBox("Anisotropy Parameters")
        vbox_params = QVBoxLayout(grp_params)

        h_g = QHBoxLayout()
        h_g.addWidget(QLabel("G-factor:"))
        self.spin_g = QDoubleSpinBox()
        self.spin_g.setRange(0.01, 100.0)
        self.spin_g.setDecimals(4)

        initial_g = 1.0
        if dataset.units:
            if "g_factor" in dataset.units:
                try:
                    initial_g = float(dataset.units["g_factor"])
                except Exception:
                    pass
            elif "pymess_metadata" in dataset.units:
                pm_meta = dataset.units.get("pymess_metadata") or {}
                for gk in ("g_factor", "G_factor", "gfactor"):
                    if gk in pm_meta:
                        try:
                            initial_g = float(pm_meta[gk])
                            break
                        except Exception:
                            pass
        self.spin_g.setValue(initial_g)
        self.spin_g.setSingleStep(0.05)
        h_g.addWidget(self.spin_g)
        vbox_params.addLayout(h_g)

        h_thresh = QHBoxLayout()
        h_thresh.addWidget(QLabel("Min Denom Threshold:"))
        self.spin_thresh = QDoubleSpinBox()
        self.spin_thresh.setRange(0.0, 1.0)
        self.spin_thresh.setDecimals(5)
        self.spin_thresh.setValue(1e-4)
        h_thresh.addWidget(self.spin_thresh)
        vbox_params.addLayout(h_thresh)

        self.chk_corrected = QCheckBox("Use background-corrected signal (Zavg_C)")
        self.chk_corrected.setChecked(dataset.Zavg_C is not None)
        vbox_params.addWidget(self.chk_corrected)

        # Probe selection for kinetic preview
        h_pr = QHBoxLayout()
        h_pr.addWidget(QLabel("Preview Probe:"))
        self.spin_preview_probe = QDoubleSpinBox()
        probe = dataset.probe
        p_min, p_max = float(np.min(probe)), float(np.max(probe))
        self.spin_preview_probe.setRange(min(p_min, p_max), max(p_min, p_max))
        self.spin_preview_probe.setDecimals(2)
        u_l = dataset.units.get("unitsL_lbl", "nm") if dataset.units else "nm"
        disp_u = "cm⁻¹" if u_l in ("cm-1", "cm^-1", "cm⁻¹") else u_l
        self.spin_preview_probe.setSuffix(f" {disp_u}")
        self.spin_preview_probe.setValue(float(probe[len(probe) // 2]))
        h_pr.addWidget(self.spin_preview_probe)
        vbox_params.addLayout(h_pr)

        settings_layout.addWidget(grp_params)
        layout.addLayout(settings_layout)

        # Canvas
        self.canvas = MplCanvas(self)
        layout.addWidget(self.canvas)

        # Actions & Output Component Selector
        h_act = QHBoxLayout()
        h_act.addWidget(QLabel("Component to apply:"))
        self.combo_target_component = QComboBox()
        self.combo_target_component.addItem("Anisotropy: r(t)", "anisotropy")
        self.combo_target_component.addItem("Magic Angle: I_MA (Isotropic)", "magic_angle")
        self.combo_target_component.addItem("Polarization Difference: Delta I", "difference")
        self.combo_target_component.addItem("Parallel Channel: I_par", "parallel")
        self.combo_target_component.addItem("Perpendicular Channel: I_perp", "perpendicular")
        h_act.addWidget(self.combo_target_component)

        self.btn_apply = QPushButton("Apply to Workspace")
        self.btn_apply.setStyleSheet(
            "font-weight: bold; background-color: #2563eb; color: white; padding: 6px;"
        )
        self.btn_apply.clicked.connect(self._apply_to_workspace)
        h_act.addWidget(self.btn_apply)

        self.btn_export = QPushButton("Export Dataset...")
        self.btn_export.clicked.connect(self._export_anisotropy)
        h_act.addWidget(self.btn_export)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        h_act.addWidget(btn_close)
        layout.addLayout(h_act)

        # Connect signals
        self.rad_channels.toggled.connect(self._on_source_mode_changed)
        self.rad_file.toggled.connect(self._on_source_mode_changed)
        self.combo_par.currentIndexChanged.connect(self._update_calculation)
        self.combo_perp.currentIndexChanged.connect(self._update_calculation)
        self.spin_g.valueChanged.connect(self._update_calculation)
        self.spin_thresh.valueChanged.connect(self._update_calculation)
        self.chk_corrected.toggled.connect(self._update_calculation)
        self.spin_preview_probe.valueChanged.connect(self._update_preview_plots)

        self._on_source_mode_changed()
        self._update_calculation()

    def _on_source_mode_changed(self):
        use_chan = self.rad_channels.isChecked()
        self.combo_par.setEnabled(use_chan)
        self.combo_perp.setEnabled(use_chan)
        self.txt_file.setEnabled(not use_chan)
        self._update_calculation()

    def _browse_perp_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Perpendicular Polarization Dataset",
            "",
            "Transient Absorption (*.pdat *.h5 *.mat *.csv);;All Files (*)",
        )
        if filepath:
            self.txt_file.setText(filepath)
            try:
                self.perp_dataset = Dataset1D.from_file(filepath)
                self.rad_file.setChecked(True)
                self._update_calculation()
            except Exception as exc:
                QMessageBox.critical(self, "Load Error", f"Failed to load perpendicular file:\n{exc}")

    def _update_calculation(self):
        try:
            g = float(self.spin_g.value())
            thresh = float(self.spin_thresh.value())
            use_corr = self.chk_corrected.isChecked()

            if self.rad_channels.isChecked():
                det_par = self.combo_par.currentData() or 0
                det_perp = self.combo_perp.currentData() or 1
                comps = self.dataset.calculate_anisotropy(
                    g_factor=g,
                    threshold=thresh,
                    return_components=True,
                    detector_par=det_par,
                    detector_perp=det_perp,
                    use_corrected=use_corr,
                )
            else:
                if self.perp_dataset is None:
                    return
                comps = self.dataset.calculate_anisotropy(
                    perpendicular=self.perp_dataset,
                    g_factor=g,
                    threshold=thresh,
                    return_components=True,
                    use_corrected=use_corr,
                )

            self.components = comps
            self.aniso_dataset = comps.get("anisotropy")
            self.iso_dataset = comps.get("magic_angle")
            self._update_preview_plots()

        except Exception as exc:
            logger.warning("Anisotropy update failed: %s", exc)

    def _update_preview_plots(self):
        if self.aniso_dataset is None:
            return

        fig = self.canvas.figure
        fig.clear()

        # Two stacked subplots: Top = Intensities (Par, Perp, Magic Angle), Bottom = Anisotropy r(t)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)

        probe_val = float(self.spin_preview_probe.value())
        idx_p = int(np.argmin(np.abs(self.dataset.probe - probe_val)))
        actual_p = self.dataset.probe[idx_p]

        # Extract kinetic slices
        if self.rad_channels.isChecked():
            det_par = self.combo_par.currentData() or 0
            det_perp = self.combo_perp.currentData() or 1
            z_par = self.dataset._detector_slice(det_par)[:, idx_p]
            z_perp = self.dataset._detector_slice(det_perp)[:, idx_p]
        else:
            z_par = self.dataset.Z[:, idx_p]
            z_perp = self.perp_dataset.Z[:, idx_p] if self.perp_dataset else np.zeros_like(z_par)

        z_iso = self.iso_dataset.Z[:, idx_p] if self.iso_dataset else None
        r_trace = self.aniso_dataset.Z[:, idx_p]

        delays = self.aniso_dataset.delays
        u_t = self.dataset.units.get("unitsT_ltx", "ps")
        u_z = self.dataset.units.get("unitsZ_ltx", "mOD")

        # Top plot: Intensities
        ax1.plot(delays, z_par, color="#2563eb", label=r"$I_{\parallel}$", linewidth=1.4)
        ax1.plot(delays, z_perp, color="#dc2626", label=r"$I_{\perp}$", linewidth=1.4)
        if z_iso is not None:
            ax1.plot(delays, z_iso, color="#059669", linestyle="--", label=r"$I_{MA}$ (Isotropic)", linewidth=1.2)
        ax1.axhline(0, color="#94a3b8", linestyle=":", linewidth=0.8)
        ax1.set_ylabel(f"Signal ({u_z})", fontsize=9)
        ax1.set_title(f"Polarization Traces at {actual_p:.1f} {self.dataset.units.get('unitsL_lbl', 'nm')}", fontsize=10, fontweight="bold")
        ax1.legend(loc="best", fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Bottom plot: Anisotropy r(t)
        ax2.plot(delays, r_trace, color="#7c3aed", label=r"$r(t)$", linewidth=1.5)
        ax2.axhline(0.4, color="#94a3b8", linestyle="--", linewidth=0.8, label="Theoretical max (0.4)")
        ax2.axhline(-0.2, color="#94a3b8", linestyle="--", linewidth=0.8, label="Theoretical min (-0.2)")
        ax2.axhline(0.0, color="#94a3b8", linestyle=":", linewidth=0.8)
        ax2.set_xlabel(f"Delay ({u_t})", fontsize=9)
        ax2.set_ylabel("Anisotropy r(t)", fontsize=9)
        ax2.set_ylim(-1.0, 1.0)
        ax2.legend(loc="best", fontsize=8)
        ax2.grid(True, alpha=0.3)

        self.canvas.draw()

    def _apply_to_workspace(self):
        target_key = self.combo_target_component.currentData() or "anisotropy"
        ds_to_apply = self.components.get(target_key, self.aniso_dataset)
        if ds_to_apply is None:
            return
        comp_name = self.combo_target_component.currentText()
        reply = QMessageBox.question(
            self,
            "Apply to Workspace",
            f"Replace the active dataset in the workspace with {comp_name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.anisotropy_applied.emit(ds_to_apply)
            self.accept()

    def _export_anisotropy(self):
        target_key = self.combo_target_component.currentData() or "anisotropy"
        ds_to_export = self.components.get(target_key, self.aniso_dataset)
        if ds_to_export is None:
            return
        comp_name = self.combo_target_component.currentText()
        filepath, selected_filter = QFileDialog.getSaveFileName(
            self,
            f"Export {comp_name}",
            f"{target_key}.pdat",
            "PDAT format (*.pdat);;PyMORGAN HDF5 (*.h5);;All Files (*)",
        )
        if not filepath:
            return
        try:
            if filepath.endswith(".h5") or filepath.endswith(".hdf5") or "h5" in selected_filter.lower():
                ds_to_export.save_hdf5(filepath)
            else:
                ds_to_export.export_pdat(filepath)
            QMessageBox.information(self, "Export Complete", f"{comp_name} exported to:\n{filepath}")
        except Exception as exc:
            logger.exception("Export failed: %s", exc)
            QMessageBox.critical(self, "Export Failed", f"Could not export dataset:\n{exc}")

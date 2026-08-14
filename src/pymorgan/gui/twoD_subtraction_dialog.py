"""Dialog for interactive 2D spectrum subtraction."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_API", "pyqt6")

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import pymorgan as pm
from pymorgan.twoD.dataset import Dataset2D


class TwoDSubtractionDialog(QDialog):
    """Interactive dialog for subtracting a reference 2-D dataset."""

    def __init__(self, parent, sample_dataset: Dataset2D):
        super().__init__(parent)
        self.sample_dataset = sample_dataset
        self.ref_dataset: Dataset2D | None = getattr(sample_dataset, "_subtraction_ref", None)

        self.setWindowTitle("Subtract 2D Spectra")
        self.resize(1100, 600)

        self._init_ui()
        if self.ref_dataset is not None:
            self._update_ref_info()
            self._update_plot()

    def _init_ui(self):
        layout = QHBoxLayout(self)

        # Left panel: controls
        controls_layout = QVBoxLayout()
        controls_group = QGroupBox("Reference Dataset & Controls")
        group_layout = QVBoxLayout(controls_group)

        # File loading
        load_layout = QHBoxLayout()
        self.lbl_ref_path = QLabel(
            self.ref_dataset.source if self.ref_dataset and self.ref_dataset.source else "No reference dataset loaded"
        )
        self.lbl_ref_path.setWordWrap(True)
        self.btn_load_ref = QPushButton("Load Ref Dataset...")
        self.btn_load_ref.clicked.connect(self._browse_and_load_ref)
        load_layout.addWidget(self.btn_load_ref)
        group_layout.addLayout(load_layout)
        group_layout.addWidget(self.lbl_ref_path)

        # Data type combo for loading ref
        dt_layout = QHBoxLayout()
        dt_layout.addWidget(QLabel("Data Type:"))
        self.cb_datatype = QComboBox()
        self.cb_datatype.addItems(["P2DAT", "MESS_2DIR", "RAL_Proc"])
        dt_layout.addWidget(self.cb_datatype)
        group_layout.addLayout(dt_layout)

        # Scale factor
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Scale factor (k):"))
        self.spn_scale = QDoubleSpinBox()
        self.spn_scale.setRange(-50.0, 50.0)
        self.spn_scale.setSingleStep(0.05)
        initial_scale = getattr(self.sample_dataset, "_subtraction_scale", 1.0)
        self.spn_scale.setValue(initial_scale)
        self.spn_scale.valueChanged.connect(self._update_plot)
        scale_layout.addWidget(self.spn_scale)
        group_layout.addLayout(scale_layout)

        # Delay selection
        delay_layout = QHBoxLayout()
        delay_layout.addWidget(QLabel("Delay mode:"))
        self.cb_delay_mode = QComboBox()
        self.cb_delay_mode.addItems(["Match corresponding t2", "Use fixed reference t2 delay"])
        self.cb_delay_mode.currentIndexChanged.connect(self._on_delay_mode_changed)
        delay_layout.addWidget(self.cb_delay_mode)
        group_layout.addLayout(delay_layout)

        fixed_t2_layout = QHBoxLayout()
        self.lbl_fixed_t2 = QLabel("Ref t2 Index:")
        self.spn_fixed_t2 = QSpinBox()
        self.spn_fixed_t2.setRange(1, 1000)
        self.spn_fixed_t2.setValue(1)
        self.spn_fixed_t2.valueChanged.connect(self._update_plot)
        fixed_t2_layout.addWidget(self.lbl_fixed_t2)
        fixed_t2_layout.addWidget(self.spn_fixed_t2)
        group_layout.addLayout(fixed_t2_layout)
        self.lbl_fixed_t2.setEnabled(False)
        self.spn_fixed_t2.setEnabled(False)

        # Selected preview delay index for display
        prev_t2_layout = QHBoxLayout()
        prev_t2_layout.addWidget(QLabel("Preview t2 delay:"))
        self.spn_preview_t2 = QSpinBox()
        self.spn_preview_t2.setRange(1, len(self.sample_dataset.delays))
        self.spn_preview_t2.setValue(1)
        self.spn_preview_t2.valueChanged.connect(self._update_plot)
        prev_t2_layout.addWidget(self.spn_preview_t2)
        group_layout.addLayout(prev_t2_layout)

        group_layout.addStretch()

        # Action Buttons
        self.btn_apply = QPushButton("Apply Subtraction")
        self.btn_apply.setStyleSheet("font-weight: bold; background-color: #2563eb; color: white;")
        self.btn_apply.clicked.connect(self._apply_subtraction)

        self.btn_reset = QPushButton("Reset / Remove Subtraction")
        self.btn_reset.clicked.connect(self._reset_subtraction)

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)

        controls_layout.addWidget(controls_group)
        controls_layout.addWidget(self.btn_apply)
        controls_layout.addWidget(self.btn_reset)
        controls_layout.addWidget(self.btn_close)

        # Right panel: 3-panel Matplotlib canvas
        self.fig, (self.ax_orig, self.ax_ref, self.ax_sub) = plt.subplots(1, 3, figsize=(10, 4.5))
        self.canvas = FigureCanvas(self.fig)

        layout.addLayout(controls_layout, stretch=1)
        layout.addWidget(self.canvas, stretch=3)

    def _browse_and_load_ref(self):
        dt = self.cb_datatype.currentText()
        path = QFileDialog.getExistingDirectory(self, "Select Reference 2D Dataset Folder")
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Reference 2D Dataset File", "", "All files (*)"
            )
        if not path:
            return

        try:
            ref_ds = pm.load_2D(path, data_type=dt)
            self.ref_dataset = ref_ds
            self.lbl_ref_path.setText(f"{Path(path).name} ({ref_ds.n_maps} delays)")
            self._update_ref_info()
            self._update_plot()
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", f"Could not load reference dataset:\n{exc}")

    def _update_ref_info(self):
        if self.ref_dataset is not None:
            self.spn_fixed_t2.setMaximum(self.ref_dataset.n_maps)

    def _on_delay_mode_changed(self, index: int):
        is_fixed = (index == 1)
        self.lbl_fixed_t2.setEnabled(is_fixed)
        self.spn_fixed_t2.setEnabled(is_fixed)
        self._update_plot()

    def _update_plot(self):
        if self.sample_dataset is None:
            return

        i_t2 = self.spn_preview_t2.value() - 1
        i_t2 = int(np.clip(i_t2, 0, len(self.sample_dataset.delays) - 1))
        t2_val = self.sample_dataset.delays[i_t2]

        sample_map = self.sample_dataset.Z_R[:, :, i_t2]

        self.ax_orig.clear()
        self.ax_ref.clear()
        self.ax_sub.clear()

        # Plot original
        vmax_orig = np.max(np.abs(sample_map)) or 1.0
        self.ax_orig.contourf(
            self.sample_dataset.probe,
            self.sample_dataset.pump,
            sample_map,
            levels=20,
            cmap="bwr",
            vmin=-vmax_orig,
            vmax=vmax_orig,
        )
        self.ax_orig.plot(self.sample_dataset.probe, self.sample_dataset.probe, "k--", alpha=0.5)
        self.ax_orig.set_title(f"Original (t2={t2_val:.1f} ps)")
        self.ax_orig.set_xlabel("Probe (cm$^{-1}$)")
        self.ax_orig.set_ylabel("Pump (cm$^{-1}$)")

        if self.ref_dataset is None:
            self.ax_ref.set_title("No Reference")
            self.ax_sub.set_title("Subtracted")
            self.canvas.draw()
            return

        scale = self.spn_scale.value()
        is_fixed = (self.cb_delay_mode.currentIndex() == 1)
        ref_t2_idx = (self.spn_fixed_t2.value() - 1) if is_fixed else None

        # Determine reference map for preview
        if ref_t2_idx is not None:
            r_map = self.ref_dataset.Z[:, :, ref_t2_idx]
            ref_delay_str = f"index {ref_t2_idx + 1}"
        else:
            r_idx = self.ref_dataset.map_index(t2_val)
            r_map = self.ref_dataset.Z[:, :, r_idx]
            ref_delay_str = f"t2={self.ref_dataset.delays[r_idx]:.1f} ps"

        # Interpolate if needed
        pump_match = np.array_equal(self.sample_dataset.pump, self.ref_dataset.pump)
        probe_match = np.array_equal(self.sample_dataset.probe, self.ref_dataset.probe)

        if pump_match and probe_match:
            ref_map_interp = r_map
        else:
            from scipy.interpolate import RegularGridInterpolator
            interp = RegularGridInterpolator(
                (self.ref_dataset.pump, self.ref_dataset.probe),
                r_map,
                bounds_error=False,
                fill_value=0.0,
            )
            P_grid, R_grid = np.meshgrid(self.sample_dataset.pump, self.sample_dataset.probe, indexing="ij")
            ref_map_interp = interp((P_grid, R_grid))

        scaled_ref = scale * ref_map_interp
        subtracted_map = sample_map - scaled_ref

        # Plot scaled reference
        vmax_ref = np.max(np.abs(scaled_ref)) or 1.0
        self.ax_ref.contourf(
            self.sample_dataset.probe,
            self.sample_dataset.pump,
            scaled_ref,
            levels=20,
            cmap="bwr",
            vmin=-vmax_ref,
            vmax=vmax_ref,
        )
        self.ax_ref.plot(self.sample_dataset.probe, self.sample_dataset.probe, "k--", alpha=0.5)
        self.ax_ref.set_title(f"Scaled Ref ({scale:.2f}x, {ref_delay_str})")
        self.ax_ref.set_xlabel("Probe (cm$^{-1}$)")

        # Plot subtracted
        vmax_sub = np.max(np.abs(subtracted_map)) or 1.0
        self.ax_sub.contourf(
            self.sample_dataset.probe,
            self.sample_dataset.pump,
            subtracted_map,
            levels=20,
            cmap="bwr",
            vmin=-vmax_sub,
            vmax=vmax_sub,
        )
        self.ax_sub.plot(self.sample_dataset.probe, self.sample_dataset.probe, "k--", alpha=0.5)
        self.ax_sub.set_title("Subtracted Result")
        self.ax_sub.set_xlabel("Probe (cm$^{-1}$)")

        self.fig.tight_layout()
        self.canvas.draw()

    def _apply_subtraction(self):
        if self.ref_dataset is None:
            QMessageBox.warning(self, "No Reference", "Please load a reference 2D dataset first.")
            return

        scale = self.spn_scale.value()
        is_fixed = (self.cb_delay_mode.currentIndex() == 1)
        ref_t2_idx = (self.spn_fixed_t2.value() - 1) if is_fixed else None

        self.sample_dataset.subtract_spectrum(self.ref_dataset, scale=scale, ref_t2_idx=ref_t2_idx)
        QMessageBox.information(self, "Subtraction Applied", f"Reference spectrum subtracted (scale={scale:.2f}).")
        self.accept()

    def _reset_subtraction(self):
        self.sample_dataset.reset_subtraction()
        QMessageBox.information(self, "Subtraction Reset", "Dataset restored to un-subtracted state.")
        self.accept()

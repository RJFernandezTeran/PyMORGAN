"""Dialogs and auxiliary windows owned by the main window.

``SpectralDiffusionDialog`` (2D analysis routine + t2 range), ``OtherPlotWindow``
(the detached time-domain / phasing plot window) and
``SolventSubtractionDialog`` (1D solvent subtraction). Split out of
``main_window.py`` so the per-tab mixins can use them directly.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyqt6")


import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import pymorgan as pm
from pymorgan.log import get_logger

logger = get_logger(__name__)


class SpectralDiffusionDialog(QDialog):
    def __init__(self, parent, delays):
        super().__init__(parent)
        self.setWindowTitle("Spectral Diffusion Analysis")
        from PyQt6.QtWidgets import QDialogButtonBox, QLabel, QLineEdit, QVBoxLayout

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Select analysis routine:"))
        self.method_cb = QComboBox()
        self.method_cb.addItems(
            [
                "CLS (Centre-Line Slope)",
                "IvCLS (Inverse Centre-Line Slope)",
                "CLS+IvCLS (both combined)",
                "NLS (Nodal Line Slope)",
            ]
        )
        layout.addWidget(self.method_cb)

        t2_max_val = np.max(delays)
        layout.addWidget(
            QLabel(f"Enter t2 range to fit/analyze as min:max (default: 0:{t2_max_val:.2f}):")
        )
        self.t2_range_input = QLineEdit(f"0:{t2_max_val:.2f}")
        layout.addWidget(self.t2_range_input)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_values(self):
        return self.method_cb.currentText(), self.t2_range_input.text()


class OtherPlotWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(None)  # Parent=None to avoid C++ ownership cycle
        self.setWindowTitle("PyMORGAN - Time-Domain / Phasing Plots")
        self.resize(750, 600)

        # Create a central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(5, 5, 5, 5)

        # Add the plot canvas
        from pymorgan.gui.canvas import MplCanvas

        self.canvas = MplCanvas(self)
        layout.addWidget(self.canvas)

        import weakref

        self.main_window = weakref.ref(parent) if parent is not None else None

    def update_plot(self):
        main_win = self.main_window() if self.main_window is not None else None
        if main_win is None or main_win.twoD_dataset is None:
            self.canvas.reset()
            self.canvas.draw()
            return

        ax = self.canvas.reset()
        main_win._plot_twoD_other_axes(ax, is_qt=True)
        ax.tick_params(axis="both", which="major", labelsize=12)
        self.canvas.draw()

    def closeEvent(self, event):
        main_win = self.main_window() if self.main_window is not None else None
        if main_win is not None:
            main_win._twoD_pop_window = None
        import matplotlib.pyplot as plt

        if hasattr(self, "canvas") and self.canvas is not None and hasattr(self.canvas, "figure"):
            plt.close(self.canvas.figure)
        super().closeEvent(event)


class SolventSubtractionDialog(QDialog):
    """Small custom dialog to gather time-offset and scaling parameters for solvent subtraction."""

    def __init__(self, parent, filename: str, sample_ds, solvent_ds, default_dt, default_scale):
        super().__init__(parent)
        self.setWindowTitle("Solvent Subtraction Parameters")

        self.sample_ds = sample_ds
        self.solvent_ds = solvent_ds

        # Track the actual values (which can be arrays or floats)
        self.dt_val = default_dt
        self.scale_val = default_scale

        layout = QVBoxLayout(self)

        lbl = QLabel(f"Solvent dataset: {filename}", self)
        lbl.setStyleSheet("font-weight: bold;")
        layout.addWidget(lbl)

        # Grid layout for inputs
        grid = QGridLayout()
        layout.addLayout(grid)

        # Time offset
        grid.addWidget(QLabel("Time offset (ps):", self), 0, 0)
        self.dt_box = QDoubleSpinBox(self)
        self.dt_box.setRange(-10.0, 10.0)
        self.dt_box.setSingleStep(0.01)
        self.dt_box.setDecimals(3)

        init_dt = np.nanmean(default_dt) if np.ndim(default_dt) > 0 else default_dt
        self.dt_box.setValue(init_dt)
        grid.addWidget(self.dt_box, 0, 1)

        # Amplitude scale
        grid.addWidget(QLabel("Scaling factor:", self), 1, 0)
        self.scale_box = QDoubleSpinBox(self)
        self.scale_box.setRange(0.0, 100.0)
        self.scale_box.setSingleStep(0.01)
        self.scale_box.setDecimals(3)

        init_scale = np.nanmean(default_scale) if np.ndim(default_scale) > 0 else default_scale
        self.scale_box.setValue(init_scale)
        grid.addWidget(self.scale_box, 1, 1)

        # Checkboxes for Autofit Options
        self.per_pixel_dt_cb = QCheckBox("Per-pixel t₀ offset", self)
        self.per_pixel_scale_cb = QCheckBox("Per-pixel amplitude scale", self)

        settings = pm.get_settings()
        self.per_pixel_dt_cb.setChecked(settings.solvent_per_pixel_dt)
        self.per_pixel_scale_cb.setChecked(settings.solvent_per_pixel_scale)

        grid.addWidget(self.per_pixel_dt_cb, 2, 0)
        grid.addWidget(self.per_pixel_scale_cb, 2, 1)

        # Auto-fit button
        self.auto_btn = QPushButton("Auto Fit (erfc rise)", self)
        self.auto_btn.setToolTip(
            "Automatically fit timing offsets and scaling factor to match an erf-step rise"
        )
        self.auto_btn.clicked.connect(self._run_auto_fit)
        grid.addWidget(self.auto_btn, 3, 0, 1, 2)

        # Info label for auto-fit status
        self.info_lbl = QLabel("", self)
        self.info_lbl.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.info_lbl)

        # Standard dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Connect value changes to reset back to scalar if user manual-edits
        self.dt_box.valueChanged.connect(self._on_dt_changed)
        self.scale_box.valueChanged.connect(self._on_scale_changed)

    def _run_auto_fit(self):
        try:
            self.setCursor(Qt.CursorShape.WaitCursor)
        except Exception:
            pass
        try:
            scale_opt, dt_opt = self.sample_ds.fit_solvent_auto(
                self.solvent_ds,
                per_pixel_dt=self.per_pixel_dt_cb.isChecked(),
                per_pixel_scale=self.per_pixel_scale_cb.isChecked(),
            )

            # Temporarily block signals so setting values doesn't trigger changed slots
            self.dt_box.blockSignals(True)
            self.scale_box.blockSignals(True)

            self.dt_val = dt_opt
            self.scale_val = scale_opt

            mean_scale = float(np.nanmean(scale_opt))
            mean_dt = float(np.nanmean(dt_opt)) if np.ndim(dt_opt) > 0 else float(dt_opt)
            self.dt_box.setValue(mean_dt)
            self.scale_box.setValue(mean_scale)

            if np.ndim(dt_opt) > 0:
                self.info_lbl.setText(
                    f"Auto-fit complete:\nMean time offset = {mean_dt:.3f} ps\nMean scale = {mean_scale:.3f}"
                )
            else:
                self.info_lbl.setText(
                    f"Auto-fit complete:\nGlobal time offset = {dt_opt:.3f} ps\nMean scale = {mean_scale:.3f}"
                )

            # Automatically pop open diagnostic plot
            self._plot_diagnostics(scale_opt, dt_opt)

        except Exception as exc:
            QMessageBox.critical(self, "Auto Fit Failed", f"Optimisation failed:\n{exc}")
        finally:
            self.dt_box.blockSignals(False)
            self.scale_box.blockSignals(False)
            try:
                self.unsetCursor()
            except Exception:
                pass

    def _plot_diagnostics(self, scale_opt, dt_opt):
        import matplotlib.pyplot as plt

        # Create new figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        # Wavelength axis
        probe = self.sample_ds.probe

        # Plot 1: Amplitude scale
        if np.ndim(scale_opt) > 0:
            ax1.plot(probe, scale_opt, "r-", linewidth=2, label="Fitted Scale $A(\\lambda)$")
            ax1.set_ylabel("Amplitude Scale")
        else:
            ax1.axhline(
                scale_opt,
                color="r",
                linestyle="--",
                linewidth=2,
                label=f"Global Scale $A = {scale_opt:.3f}$",
            )
            ax1.set_ylabel("Amplitude Scale")
        ax1.grid(True, linestyle=":")
        ax1.legend()
        ax1.set_title("Solvent Subtraction Fit Diagnostics")

        # Plot 2: Time offset
        if np.ndim(dt_opt) > 0:
            ax2.plot(probe, dt_opt, "b-", linewidth=2, label="Fitted offset $dt(\\lambda)$")
            ax2.set_ylabel("Time offset $dt$ (ps)")
        else:
            ax2.axhline(
                dt_opt,
                color="b",
                linestyle="--",
                linewidth=2,
                label=f"Global offset $dt = {dt_opt:.3f}$ ps",
            )
            ax2.set_ylabel("Time offset (ps)")
        ax2.set_xlabel("Probe Wavelength")
        ax2.grid(True, linestyle=":")
        ax2.legend()

        fig.tight_layout()
        plt.show(block=False)

    def _on_dt_changed(self):
        self.dt_val = self.dt_box.value()
        self.info_lbl.setText("Custom manual time offset set (scalar).")

    def _on_scale_changed(self):
        self.scale_val = self.scale_box.value()
        self.info_lbl.setText("Custom manual scaling factor set.")

    def values(self) -> tuple[float | np.ndarray, float | np.ndarray]:
        return self.dt_val, self.scale_val


# --------------------------------------------------------------------------- #
#                      Value-list prompt (cuts, delays)                       #
# --------------------------------------------------------------------------- #
def ask_values(parent, title: str, label: str, default, all_values=None) -> list[float] | None:
    """Prompt for a list of cuts, pre-filled with ``default``.

    Shared by every window that asks "which delays?" or "which probe
    positions?", so the accepted syntax is the same everywhere: PyMORGAN's own
    cut plots and PyRATE-TA's pop-out traces both call this rather than each
    growing their own parser.

    Returns ``None`` if the dialog was cancelled, and an empty list if nothing
    usable was typed (the caller reports that; it is not an error here).
    """
    from PyQt6.QtWidgets import QInputDialog

    from pymorgan.helpers import parse_value_list

    default_text = ", ".join(f"{v:g}" for v in (default or []))
    text, ok = QInputDialog.getText(parent, title, label, text=default_text)
    if not ok:
        return None

    values = parse_value_list(text, all_values)
    if values is None:
        QMessageBox.warning(parent, title, "Could not retrieve all available values.")
        return None
    if not values:
        QMessageBox.information(parent, title, "No valid numbers entered.")
    return values

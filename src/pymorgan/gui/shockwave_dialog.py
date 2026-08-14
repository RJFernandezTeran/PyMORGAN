"""Dialog for configuring and previewing shockwave subtraction on a 1D dataset."""

from __future__ import annotations

import re

import numpy as np
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from pymorgan.gui.canvas import MplCanvas
from pymorgan.oneD.process import subtract_shockwave


class ShockwaveSubtractionDialog(QDialog):
    """Dialog for shockwave subtraction with live preview plot and interactive picking."""

    def __init__(self, parent, dataset):
        super().__init__(parent)
        self.setWindowTitle("Subtract Shock Wave")
        self.resize(650, 520)

        self.dataset = dataset
        self.selected_pixels: np.ndarray | None = None
        self.shockwave_trace: np.ndarray | None = None
        self._picker = None

        layout = QVBoxLayout(self)

        # Header description
        hdr = QLabel(
            "Select pixel(s) or a probe range to average into a 1-D shockwave kinetic trace,\n"
            "which will be subtracted from all pixels in the dataset."
        )
        hdr.setStyleSheet("font-weight: bold; margin-bottom: 4px;")
        layout.addWidget(hdr)

        # Selection mode radio buttons
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Selection mode:"))
        self.rb_pixels = QRadioButton("Pixel Index / Range")
        self.rb_probe = QRadioButton("Probe Value / Range")
        self.rb_pixels.setChecked(True)
        mode_layout.addWidget(self.rb_pixels)
        mode_layout.addWidget(self.rb_probe)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        # Input box and Pick button
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("e.g. 1-5 or 1, 2, 3")
        input_layout.addWidget(self.input_field)

        self.pick_btn = QPushButton("Pick on Contour Plot...")
        self.pick_btn.setToolTip("Pick pixels or range interactively on the main contour plot")
        input_layout.addWidget(self.pick_btn)

        layout.addLayout(input_layout)

        # Status / Range label
        probe = dataset.probe
        p_min, p_max = float(np.min(probe)), float(np.max(probe))
        u = dataset.units.get("unitsL_lbl", "") if dataset.units else ""
        self.status_lbl = QLabel(
            f"Dataset pixels: 1 .. {probe.size} | Probe range: {p_min:.1f} .. {p_max:.1f} {u}"
        )
        self.status_lbl.setStyleSheet("color: gray;")
        layout.addWidget(self.status_lbl)

        # Preview Plot Canvas
        self.canvas = MplCanvas(self)
        layout.addWidget(self.canvas)

        # Standard buttons
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        # Event connections
        self.rb_pixels.toggled.connect(self._on_mode_changed)
        self.input_field.textChanged.connect(self._update_preview)
        self.pick_btn.clicked.connect(self._start_interactive_pick)

        # Set default input (1-based: 1-3)
        default_str = f"1-{min(3, probe.size)}" if probe.size > 1 else "1"
        self.input_field.setText(default_str)
        self._update_preview()

    def _on_mode_changed(self):
        if self.rb_pixels.isChecked():
            self.input_field.setPlaceholderText("e.g. 1-5 or 1, 2, 3")
        else:
            p_min = float(np.min(self.dataset.probe))
            self.input_field.setPlaceholderText(f"e.g. {p_min:.1f}:{p_min + 20:.1f} or {p_min:.1f}")
        self._update_preview()

    def _parse_input(self) -> np.ndarray | None:
        text = self.input_field.text().strip()
        if not text:
            return None

        probe = self.dataset.probe
        Npixels = probe.size

        try:
            if self.rb_pixels.isChecked():
                # Parse 1-based pixel numbers or ranges (e.g. "1-5", "1:5", "1,2,3")
                if "-" in text or ":" in text or ".." in text:
                    parts = re.split(r"[-:\.]+", text)
                    if len(parts) == 2:
                        start, end = int(parts[0]), int(parts[1])
                        user_pix = np.arange(min(start, end), max(start, end) + 1)
                    else:
                        return None
                else:
                    parts = re.split(r"[,\s]+", text)
                    user_pix = np.array([int(p) for p in parts if p], dtype=int)
                user_pix = np.unique(user_pix)
                if np.any(user_pix < 1) or np.any(user_pix > Npixels):
                    return None
                # Convert 1-based user input to 0-based array index
                return user_pix - 1
            else:
                # Parse probe ranges or values (e.g. "1500-1520", "1500:1520", "1500, 1510")
                if "-" in text or ":" in text or ".." in text:
                    # Care with negative probe values if any: split on ':' or '..' first
                    delim = ":" if ":" in text else (".." if ".." in text else "-")
                    parts = text.split(delim)
                    if len(parts) == 2:
                        v1, v2 = float(parts[0]), float(parts[1])
                        p_min, p_max = min(v1, v2), max(v1, v2)
                        mask = (probe >= p_min) & (probe <= p_max)
                        idx = np.where(mask)[0]
                        return idx if idx.size > 0 else None
                    return None
                else:
                    parts = re.split(r"[,\s]+", text)
                    vals = [float(p) for p in parts if p]
                    nearest = [int(np.argmin(np.abs(probe - v))) for v in vals]
                    idx = np.unique(np.array(nearest, dtype=int))
                    return idx if idx.size > 0 else None
        except Exception:
            return None

    def _update_preview(self):
        pix_idx = self._parse_input()
        ax = self.canvas.ax
        ax.clear()

        ok_btn = self.buttons.button(QDialogButtonBox.StandardButton.Ok)

        if pix_idx is None or pix_idx.size == 0:
            ax.text(
                0.5,
                0.5,
                "Invalid pixel selection / range",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="red",
                fontsize=9,
            )
            self.canvas.draw()
            if ok_btn:
                ok_btn.setEnabled(False)
            self.selected_pixels = None
            self.shockwave_trace = None
            return

        try:
            Z_corr, trace = subtract_shockwave(self.dataset.Z, pix_idx, one_based=False)
            delays = self.dataset.delays

            from pymorgan import helpers as hlp
            from pymorgan.oneD.plot import _apply_time_xscale

            s, labelStyle, Units = self.dataset._resolve(None, None, None)

            # Plot kinetic trace(s) without legend for single detector channel
            if trace.ndim == 1:
                ax.plot(delays, trace, "-", color="C0", linewidth=1.2)
            elif trace.ndim == 2:  # [Ndelays x Ndet]
                for d in range(trace.shape[1]):
                    ax.plot(delays, trace[:, d], "-", label=f"Det {d}", linewidth=1.2)
                if trace.shape[1] > 1:
                    ax.legend(loc="best", fontsize=7, frameon=False)
            elif trace.ndim == 3:  # [Ndelays x Ndet x Nscans]
                trace_avg = np.nanmean(trace, axis=-1)
                for d in range(trace_avg.shape[1]):
                    ax.plot(delays, trace_avg[:, d], "-", label=f"Det {d}", linewidth=1.2)
                if trace_avg.shape[1] > 1:
                    ax.legend(loc="best", fontsize=7, frameon=False)

            # Zero reference line and delay-axis scale recycled from 1D plot module
            ax.axhline(y=0, color="0.75", linewidth=0.75)
            _apply_time_xscale(ax, s.time_axis_scale.value, delays)

            # Recycled axis labels matching pipeline conventions
            XUnits = {"lbl": "Delay", "ltx": Units.get("unitsT_ltx", "ps")}
            YUnits = {
                "lbl": Units.get("unitsZ_lbl", r"\Delta A"),
                "ltx": Units.get("unitsZ_ltx", "mOD"),
            }
            hlp.setXYlabels(ax, labelStyle, XUnits, YUnits)

            # Compact font sizes for dialog preview
            ax.tick_params(labelsize=8)
            ax.xaxis.label.set_fontsize(8)
            ax.yaxis.label.set_fontsize(8)

            # Short concise title using 1-based pixel numbering
            if pix_idx.size == 1:
                pix_str = f"Pix {pix_idx[0] + 1}"
            else:
                pix_str = f"Pix {pix_idx[0] + 1}..{pix_idx[-1] + 1}"
            ax.set_title(f"Shockwave ({pix_str})", fontsize=9)

            self.canvas.draw()
            if ok_btn:
                ok_btn.setEnabled(True)
            self.selected_pixels = pix_idx
            self.shockwave_trace = trace
        except Exception as exc:
            ax.text(
                0.5,
                0.5,
                f"Error: {exc}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="red",
                fontsize=9,
            )
            self.canvas.draw()
            if ok_btn:
                ok_btn.setEnabled(False)

    def _start_interactive_pick(self):
        """Trigger interactive picking on the parent window's contour plot."""
        parent = self.parent()
        if parent is None:
            return

        from pymorgan.gui.picker import ContourPicker

        # Check for main window contour axis
        ax_contour = getattr(parent, "ax_contour", None)
        canvas = getattr(parent, "canvas", None)
        if canvas is None:
            canvas = getattr(parent, "PPaxes", None)

        if ax_contour is None or canvas is None:
            return

        # Minimise dialog while picking so parent contour plot is clear
        self.hide()

        def _on_done(points):
            self.show()
            self.raise_()
            self.activateWindow()
            if not points:
                return

            probe = self.dataset.probe
            if len(points) == 1:
                val = points[0]
                if self.rb_pixels.isChecked():
                    idx = int(np.argmin(np.abs(probe - val)))
                    self.input_field.setText(str(idx + 1))
                else:
                    self.input_field.setText(f"{val:.2f}")
            else:
                val1, val2 = min(points), max(points)
                if self.rb_pixels.isChecked():
                    idx1 = int(np.argmin(np.abs(probe - val1)))
                    idx2 = int(np.argmin(np.abs(probe - val2)))
                    self.input_field.setText(f"{min(idx1, idx2) + 1}-{max(idx1, idx2) + 1}")
                else:
                    self.input_field.setText(f"{val1:.2f}:{val2:.2f}")

        self._picker = ContourPicker(canvas, ax_contour, axis="x", on_done=_on_done).start()

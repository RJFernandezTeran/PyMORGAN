"""Dialog for creating lightweight GIF, WebP, and APNG animations of 2D contour maps across t2."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

if TYPE_CHECKING:
    from pymorgan.twoD.dataset import Dataset2D


class MakeMovieDialog(QDialog):
    """Dialog to configure and execute 2D contour map animation generation."""

    def __init__(self, dataset: Dataset2D, parent=None, plot_controls=None):
        super().__init__(parent)
        self.setWindowTitle("Export 2D Animation (GIF / WebP / APNG)")
        self.resize(450, 360)

        self.dataset = dataset
        self.plot_controls = plot_controls

        delays = np.asarray(dataset.delays, dtype=float)
        min_delay = float(np.nanmin(delays))
        max_delay = float(np.nanmax(delays))

        layout = QVBoxLayout(self)
        form = QFormLayout()

        # t2 range controls
        t2_layout = QHBoxLayout()
        self.t2_start_sbx = QDoubleSpinBox(self)
        self.t2_start_sbx.setRange(min_delay, max_delay)
        self.t2_start_sbx.setValue(min_delay)
        self.t2_start_sbx.setSuffix(" ps")

        self.t2_end_sbx = QDoubleSpinBox(self)
        self.t2_end_sbx.setRange(min_delay, max_delay)
        self.t2_end_sbx.setValue(max_delay)
        self.t2_end_sbx.setSuffix(" ps")

        t2_layout.addWidget(QLabel("Min:"))
        t2_layout.addWidget(self.t2_start_sbx)
        t2_layout.addWidget(QLabel("Max:"))
        t2_layout.addWidget(self.t2_end_sbx)
        form.addRow("t₂ Delay Range:", t2_layout)

        # Frame delay / speed
        delay_layout = QHBoxLayout()
        self.frame_delay_sbx = QSpinBox(self)
        self.frame_delay_sbx.setRange(20, 5000)
        self.frame_delay_sbx.setValue(100)
        self.frame_delay_sbx.setSuffix(" ms")
        self.frame_delay_sbx.setSingleStep(10)
        self.frame_delay_sbx.valueChanged.connect(self._update_fps_label)

        self.fps_label = QLabel("(10.0 FPS)", self)
        delay_layout.addWidget(self.frame_delay_sbx)
        delay_layout.addWidget(self.fps_label)
        form.addRow("Frame Delay:", delay_layout)

        # Repetitions / Loop count
        self.loop_count_sbx = QSpinBox(self)
        self.loop_count_sbx.setRange(0, 100)
        self.loop_count_sbx.setValue(0)
        self.loop_count_sbx.setToolTip("0 = Infinite loop, 1+ = exact number of repeats.")
        form.addRow("Loop Count (0 = Infinite):", self.loop_count_sbx)

        # Colourmap scaling strategy
        self.scale_cbx = QComboBox(self)
        self.scale_cbx.addItems([
            "Constant z-limits across all frames (global vmin/vmax)",
            "Rescale z-limits dynamically per frame (local vmin/vmax)",
        ])
        form.addRow("Colorbar Scaling:", self.scale_cbx)

        # Output format (lightweight, zero external binary dependencies)
        self.format_cbx = QComboBox(self)
        self.format_cbx.addItems([
            "Animated GIF (*.gif)",
            "WebP Animation (*.webp)",
            "Animated PNG (*.png)",
        ])
        self.format_cbx.currentIndexChanged.connect(self._on_format_changed)
        form.addRow("Format:", self.format_cbx)

        # File output path selection
        path_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit(self)
        default_dir = os.getcwd()
        if dataset.source:
            base_name = os.path.splitext(os.path.basename(dataset.source))[0]
            default_path = os.path.join(os.path.dirname(dataset.source), f"{base_name}_anim.gif")
        else:
            default_path = os.path.join(default_dir, "2D_movie.gif")
        self.file_path_edit.setText(default_path)

        self.browse_btn = QPushButton("Browse...", self)
        self.browse_btn.setStyleSheet(
            "QPushButton { font-weight: bold; color: #7e5109; background-color: #fdebd0; border: 1px solid #f9e79f; border-radius: 4px; padding: 4px 8px; }"
            "QPushButton:hover { background-color: #f9e79f; border-color: #f4d03f; }"
            "QPushButton:pressed { background-color: #f4d03f; }"
        )
        self.browse_btn.clicked.connect(self._browse_output_file)

        path_layout.addWidget(self.file_path_edit)
        path_layout.addWidget(self.browse_btn)
        form.addRow("Save Location:", path_layout)

        layout.addLayout(form)

        # Dialog Buttons
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        ok_btn = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        if ok_btn:
            ok_btn.setText("Export Animation")
            ok_btn.setStyleSheet(
                "QPushButton { font-weight: bold; color: #196f3d; background-color: #d5f5e3; border: 1px solid #abebc6; border-radius: 4px; padding: 4px 12px; }"
                "QPushButton:hover { background-color: #abebc6; border-color: #58d68d; }"
                "QPushButton:pressed { background-color: #58d68d; }"
            )
        cancel_btn = self.button_box.button(QDialogButtonBox.StandardButton.Cancel)
        if cancel_btn:
            cancel_btn.setStyleSheet(
                "QPushButton { font-weight: bold; color: #2c3e50; background-color: #e5e8e8; border: 1px solid #d5dbdb; border-radius: 4px; padding: 4px 12px; }"
                "QPushButton:hover { background-color: #d5dbdb; border-color: #bdc3c7; }"
                "QPushButton:pressed { background-color: #bdc3c7; }"
            )
        self.button_box.accepted.connect(self._on_generate_movie)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def _update_fps_label(self, val_ms: int):
        if val_ms > 0:
            fps = 1000.0 / val_ms
            self.fps_label.setText(f"({fps:.1f} FPS)")

    def _on_format_changed(self, index: int):
        curr_text = self.file_path_edit.text()
        base, _ = os.path.splitext(curr_text)
        exts = [".gif", ".webp", ".png"]
        if 0 <= index < len(exts):
            self.file_path_edit.setText(base + exts[index])

    def _browse_output_file(self):
        fmt = self.format_cbx.currentText()
        if "WebP" in fmt:
            filter_str = "WebP Files (*.webp);;GIF Files (*.gif);;PNG Files (*.png);;All Files (*)"
        elif "PNG" in fmt:
            filter_str = "PNG Files (*.png);;GIF Files (*.gif);;WebP Files (*.webp);;All Files (*)"
        else:
            filter_str = "GIF Files (*.gif);;WebP Files (*.webp);;PNG Files (*.png);;All Files (*)"

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Animation", self.file_path_edit.text(), filter_str
        )
        if path:
            self.file_path_edit.setText(path)

    def _on_generate_movie(self):
        output_path = self.file_path_edit.text().strip()
        if not output_path:
            QMessageBox.warning(self, "Invalid Path", "Please select a valid output file path.")
            return

        t2_min = self.t2_start_sbx.value()
        t2_max = self.t2_end_sbx.value()
        if t2_min > t2_max:
            t2_min, t2_max = t2_max, t2_min

        delays = np.asarray(self.dataset.delays, dtype=float)
        indices = np.where((delays >= t2_min) & (delays <= t2_max))[0]

        if len(indices) == 0:
            QMessageBox.warning(
                self, "No Delays Selected", "No t₂ delay points found in the specified range."
            )
            return

        delay_ms = self.frame_delay_sbx.value()
        loop_count = self.loop_count_sbx.value()
        constant_scale = self.scale_cbx.currentIndex() == 0

        success = create_2d_movie_animation(
            dataset=self.dataset,
            indices=indices,
            output_path=output_path,
            delay_ms=delay_ms,
            loop_count=loop_count,
            constant_scale=constant_scale,
            plot_controls=self.plot_controls,
            parent_widget=self,
        )

        if success:
            QMessageBox.information(
                self, "Animation Exported", f"Successfully saved animation to:\n{output_path}"
            )
            self.accept()


def create_2d_movie_animation(
    dataset: Dataset2D,
    indices: np.ndarray,
    output_path: str,
    delay_ms: int = 100,
    loop_count: int = 0,
    constant_scale: bool = True,
    plot_controls=None,
    parent_widget=None,
) -> bool:
    """Render 2D contour maps on a single offscreen Agg canvas and write an animated image."""
    delays = np.asarray(dataset.delays, dtype=float)

    # Determine global vmin/vmax if constant scaling is requested
    global_vmin, global_vmax = None, None
    if constant_scale:
        Z_data = (
            dataset.Z_C if (dataset.is_corrected and dataset.Z_C is not None) else dataset.Z_R
        )
        sub_cube = Z_data[:, :, indices]
        max_abs = np.nanmax(np.abs(sub_cube))
        if not np.isfinite(max_abs) or max_abs == 0:
            max_abs = 1.0
        global_vmax = float(max_abs)
        global_vmin = -float(max_abs)

    # Extract plot controls settings if provided
    kwargs = {}
    if plot_controls is not None:
        if hasattr(plot_controls, "showlines_chk"):
            kwargs["ShowLines"] = plot_controls.showlines_chk.isChecked()
        if hasattr(plot_controls, "nskip_spn"):
            kwargs["Nskip"] = plot_controls.nskip_spn.value()
        if hasattr(plot_controls, "text_white_bg_chk"):
            kwargs["text_white_bg"] = plot_controls.text_white_bg_chk.isChecked()
        if hasattr(plot_controls, "cmap_combo"):
            kwargs["cmap_ID"] = plot_controls.cmap_combo.currentText()
        if hasattr(plot_controls, "white_lvl_spn"):
            kwargs["white_levels"] = plot_controls.white_lvl_spn.value()
        if hasattr(plot_controls, "cut_chk") and plot_controls.cut_chk.isChecked():
            kwargs["cut_plot"] = True
            kwargs["pump_lim"] = (plot_controls.pump_min.value(), plot_controls.pump_max.value())
            kwargs["probe_lim"] = (
                plot_controls.probe_min.value(),
                plot_controls.probe_max.value(),
            )

    n_frames = len(indices)
    progress = QProgressDialog("Rendering frames...", "Cancel", 0, n_frames, parent_widget)
    progress.setWindowTitle("Exporting 2D Animation")
    progress.setWindowModality(Qt.WindowModality.WindowModal)
    progress.setMinimumDuration(0)
    progress.setValue(0)
    progress.show()
    QApplication.processEvents()

    frames: list[Image.Image] = []
    main_win = parent_widget
    if (
        main_win is not None
        and not hasattr(main_win, "render_twoD_contour_frame")
        and hasattr(main_win, "parent")
    ):
        main_win = main_win.parent()

    # Reuse one offscreen Figure + FigureCanvasAgg across all frames (fast, no GUI window overhead)
    fig = Figure(figsize=(5.5, 5.5), dpi=100, facecolor="white")
    canvas = FigureCanvasAgg(fig)

    try:
        for idx_count, delay_idx in enumerate(indices):
            QApplication.processEvents()
            if progress.wasCanceled():
                return False

            progress.setValue(idx_count)
            t2_val = float(delays[delay_idx])
            progress.setLabelText(
                f"Rendering frame {idx_count + 1}/{n_frames} (t₂ = {t2_val:.2f} ps)"
            )
            QApplication.processEvents()

            fig.clear()
            ax = fig.add_subplot(111)

            if main_win is not None and hasattr(main_win, "render_twoD_contour_frame"):
                main_win.render_twoD_contour_frame(
                    t2_val, ax=ax, vmin=global_vmin, vmax=global_vmax
                )
            else:
                dataset.plot_map(
                    t2=t2_val,
                    ax=ax,
                    vmin=global_vmin,
                    vmax=global_vmax,
                    **kwargs,
                )

            fig.tight_layout()
            canvas.draw()
            buf = np.asarray(canvas.buffer_rgba())
            frames.append(Image.fromarray(buf).convert("RGB"))

        progress.setValue(n_frames)
        QApplication.processEvents()

        if not frames:
            return False

        ext = os.path.splitext(output_path)[1].lower()
        if ext not in (".gif", ".webp", ".png"):
            output_path = os.path.splitext(output_path)[0] + ".gif"

        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=delay_ms,
            loop=loop_count,
            optimize=True,
        )
        return True

    except Exception as e:
        if parent_widget:
            QMessageBox.critical(
                parent_widget, "Animation Export Error", f"Error exporting animation:\n{e}"
            )
        return False
    finally:
        progress.close()

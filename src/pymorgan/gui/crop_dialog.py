"""Interactive Crop Dataset dialog for 1D and 2D PyMORGAN datasets.

Provides live numeric axis bounds, a draggable Matplotlib ``RectangleSelector``
overlay with shaded out-of-bounds margins, kept-point counts, undoable in-place
application to the active workspace, and non-destructive export of a cropped copy.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyqt6")

import contextlib
from pathlib import Path
from typing import Any

import matplotlib.patches as patches
import numpy as np
from matplotlib.widgets import RectangleSelector
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

import pymorgan as pm
from pymorgan import helpers as hlp

from ..log import get_logger
from .theme import is_dark_palette, style_figure
from .widgetplot import WidgetPlot

logger = get_logger(__name__)


def _axis_minmax(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return (0.0, 1.0)
    return (float(np.min(finite)), float(np.max(finite)))


def _make_bound_spin(lo: float, hi: float, suffix: str, *, decimals: int = 3) -> QDoubleSpinBox:
    spn = QDoubleSpinBox()
    spn.setKeyboardTracking(False)
    spn.setDecimals(decimals)
    span = abs(hi - lo)
    margin = max(span * 0.25, 1.0)
    spn.setRange(lo - margin, hi + margin)
    spn.setSingleStep(max(span / 100.0, 0.01))
    if suffix:
        spn.setSuffix(f" {suffix}")
    return spn


class CropDialog(QDialog):
    """Interactive crop dialog supporting both :class:`Dataset1D` and :class:`Dataset2D`."""

    dataset_cropped = pyqtSignal(object)

    def __init__(
        self,
        parent=None,
        dataset: Any = None,
        *,
        initial_bounds: dict[str, float] | None = None,
        view_bounds: dict[str, float] | None = None,
    ):
        super().__init__(parent)
        if dataset is None:
            raise ValueError("CropDialog requires a loaded 1D or 2D dataset.")
        self.dataset = dataset
        self._is_2d = hasattr(dataset, "pump") and hasattr(dataset, "Z_R")
        self._view_bounds = dict(view_bounds) if view_bounds else None
        self._rect_selector: RectangleSelector | None = None
        self._overlay_patches: list[patches.Rectangle] = []
        self._syncing = False
        self._is_vertical_2d = False

        src_name = Path(str(getattr(dataset, "source", "") or "Dataset")).name
        kind_lbl = "2D" if self._is_2d else "1D"
        self.setWindowTitle(f"Crop {kind_lbl} Dataset — {src_name}")
        self.setModal(True)

        screen = QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen else None
        w = min(1040, avail.width() - 40) if avail else 980
        h = min(660, avail.height() - 60) if avail else 620
        self.resize(max(780, w), max(520, h))

        self._init_bounds_metadata()
        self._init_ui()
        if initial_bounds:
            self._apply_bounds_dict(initial_bounds)
        self._render_preview()
        self._update_summary_and_overlays()

    # ------------------------------------------------------------------ #
    #                        Dataset Axis Metadata                       #
    # ------------------------------------------------------------------ #
    def _init_bounds_metadata(self) -> None:
        units = getattr(self.dataset, "units", None) or {}
        t_unit = str(units.get("unitsT_ltx", "ps") or "ps").replace("$", "")
        self._t_unit = t_unit

        if self._is_2d:
            self._t_arr = np.asarray(self.dataset.delays, dtype=float)
            self._pu_arr = np.asarray(self.dataset.pump, dtype=float)
            pr = np.asarray(self.dataset.probe, dtype=float)
            self._pr_arr = pr[:, 0] if pr.ndim > 1 else pr
            self._t_min_full, self._t_max_full = _axis_minmax(self._t_arr)
            self._pu_min_full, self._pu_max_full = _axis_minmax(self._pu_arr)
            self._pr_min_full, self._pr_max_full = _axis_minmax(self._pr_arr)
            self._spec_unit = str(
                getattr(self.dataset, "freq_units", None) or units.get("unitsL", "cm-1")
            ).replace("cm^{-1}", "cm\u207b\u00b9")
            pump_axis_val = pm.get_settings().pump_axis
            if hasattr(pump_axis_val, "value"):
                pump_axis_val = pump_axis_val.value
            self._is_vertical_2d = pump_axis_val == "Vertical"
        else:
            self._t_arr = np.asarray(self.dataset.delays, dtype=float)
            pc = getattr(self.parent(), "plot_controls", None)
            det = getattr(pc, "detector", 0) if pc is not None else 0
            self._detector = int(det)
            pr = (
                self.dataset._detector_probe(self._detector)
                if hasattr(self.dataset, "_detector_probe")
                else self.dataset.probe
            )
            pr_arr = np.asarray(pr, dtype=float)
            self._pr_arr = pr_arr[:, 0] if pr_arr.ndim > 1 else pr_arr
            self._t_min_full, self._t_max_full = _axis_minmax(self._t_arr)
            self._pr_min_full, self._pr_max_full = _axis_minmax(self._pr_arr)
            if hlp.is_uncalibrated_probe(self.dataset):
                self._spec_unit = "px"
            else:
                self._spec_unit = hlp.plain_spectral_unit(
                    hlp.native_x_unit(units), scaled=False
                )

    # ------------------------------------------------------------------ #
    #                            UI Construction                         #
    # ------------------------------------------------------------------ #
    def _init_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setSpacing(12)

        # Left column: Contour preview + optional 2D t2 slice slider
        left_col = QVBoxLayout()
        self.plot_widget = WidgetPlot(self, show_toolbar=False)
        left_col.addWidget(self.plot_widget, stretch=1)

        if self._is_2d:
            t2_row = QHBoxLayout()
            t2_row.addWidget(QLabel("Preview t\u2082 slice:", self))
            self.t2_slider = QSlider(Qt.Orientation.Horizontal, self)
            n_t2 = max(1, len(self._t_arr))
            self.t2_slider.setRange(0, n_t2 - 1)
            self.t2_slider.setValue(0)
            self.t2_slider.setEnabled(n_t2 > 1)
            self.t2_label = QLabel(f"{self._t_arr[0]:.3g} {self._t_unit}" if self._t_arr.size else "n/a", self)
            self.t2_label.setMinimumWidth(90)
            self.t2_slider.valueChanged.connect(self._on_t2_slider_changed)
            t2_row.addWidget(self.t2_slider, stretch=1)
            t2_row.addWidget(self.t2_label)
            left_col.addLayout(t2_row)
        else:
            self.t2_slider = None
            self.t2_label = None

        hint = QLabel(
            "Drag a rectangle on the contour map or edit bounds on the right. "
            "Shaded regions outside the box will be discarded.",
            self,
        )
        hint.setWordWrap(True)
        left_col.addWidget(hint)
        root.addLayout(left_col, stretch=3)

        # Right column: Bounds spinboxes, quick presets, live count, actions
        right_col = QVBoxLayout()
        right_col.setSpacing(10)

        grp_bounds = QGroupBox("Crop Bounds", self)
        form = QFormLayout(grp_bounds)
        form.setSpacing(8)

        if self._is_2d:
            self.spn_pump_min = _make_bound_spin(self._pu_min_full, self._pu_max_full, self._spec_unit)
            self.spn_pump_max = _make_bound_spin(self._pu_min_full, self._pu_max_full, self._spec_unit)
            self.spn_probe_min = _make_bound_spin(self._pr_min_full, self._pr_max_full, self._spec_unit)
            self.spn_probe_max = _make_bound_spin(self._pr_min_full, self._pr_max_full, self._spec_unit)
            self.spn_t2_min = _make_bound_spin(self._t_min_full, self._t_max_full, self._t_unit)
            self.spn_t2_max = _make_bound_spin(self._t_min_full, self._t_max_full, self._t_unit)

            self.spn_pump_min.setValue(self._pu_min_full)
            self.spn_pump_max.setValue(self._pu_max_full)
            self.spn_probe_min.setValue(self._pr_min_full)
            self.spn_probe_max.setValue(self._pr_max_full)
            self.spn_t2_min.setValue(self._t_min_full)
            self.spn_t2_max.setValue(self._t_max_full)

            form.addRow("Pump min:", self.spn_pump_min)
            form.addRow("Pump max:", self.spn_pump_max)
            form.addRow("Probe min:", self.spn_probe_min)
            form.addRow("Probe max:", self.spn_probe_max)
            form.addRow("t\u2082 min:", self.spn_t2_min)
            form.addRow("t\u2082 max:", self.spn_t2_max)

            self._bound_spins = (
                self.spn_pump_min,
                self.spn_pump_max,
                self.spn_probe_min,
                self.spn_probe_max,
                self.spn_t2_min,
                self.spn_t2_max,
            )
        else:
            self.spn_probe_min = _make_bound_spin(self._pr_min_full, self._pr_max_full, self._spec_unit)
            self.spn_probe_max = _make_bound_spin(self._pr_min_full, self._pr_max_full, self._spec_unit)
            self.spn_tmin = _make_bound_spin(self._t_min_full, self._t_max_full, self._t_unit)
            self.spn_tmax = _make_bound_spin(self._t_min_full, self._t_max_full, self._t_unit)

            self.spn_probe_min.setValue(self._pr_min_full)
            self.spn_probe_max.setValue(self._pr_max_full)
            self.spn_tmin.setValue(self._t_min_full)
            self.spn_tmax.setValue(self._t_max_full)

            form.addRow("Probe min:", self.spn_probe_min)
            form.addRow("Probe max:", self.spn_probe_max)
            form.addRow("Delay min:", self.spn_tmin)
            form.addRow("Delay max:", self.spn_tmax)

            self._bound_spins = (
                self.spn_probe_min,
                self.spn_probe_max,
                self.spn_tmin,
                self.spn_tmax,
            )

        for spn in self._bound_spins:
            spn.valueChanged.connect(self._on_spin_edited)

        preset_row = QHBoxLayout()
        self.btn_match_view = QPushButton("Match Current View", self)
        self.btn_match_view.setEnabled(bool(self._view_bounds))
        self.btn_match_view.setToolTip("Copy current X/Y limits from the main contour plot")
        self.btn_match_view.clicked.connect(self._on_match_view)

        self.btn_reset_full = QPushButton("Reset Full Range", self)
        self.btn_reset_full.clicked.connect(self._on_reset_full)

        preset_row.addWidget(self.btn_match_view)
        preset_row.addWidget(self.btn_reset_full)
        form.addRow(preset_row)
        right_col.addWidget(grp_bounds)

        grp_summary = QGroupBox("Kept Points Preview", self)
        sum_layout = QVBoxLayout(grp_summary)
        self.lbl_summary = QLabel("", self)
        self.lbl_summary.setWordWrap(True)
        sum_layout.addWidget(self.lbl_summary)
        right_col.addWidget(grp_summary)

        right_col.addStretch(1)

        self.btn_apply = QPushButton("Apply to Workspace", self)
        self.btn_apply.setDefault(True)
        self.btn_apply.setToolTip("Crop the active workspace dataset (undoable via Ctrl+Z)")
        self.btn_apply.clicked.connect(self._on_apply)

        self.btn_export = QPushButton("Export Cropped Copy...", self)
        self.btn_export.setToolTip(
            "Save a cropped copy to disk without modifying the active workspace dataset"
        )
        self.btn_export.clicked.connect(self._on_export_copy)

        self.btn_cancel = QPushButton("Cancel", self)
        self.btn_cancel.clicked.connect(self.reject)

        right_col.addWidget(self.btn_apply)
        right_col.addWidget(self.btn_export)
        right_col.addWidget(self.btn_cancel)

        root.addLayout(right_col, stretch=2)

    # ------------------------------------------------------------------ #
    #                        Bounds Reading & Writing                    #
    # ------------------------------------------------------------------ #
    def get_crop_kwargs(self) -> dict[str, float]:
        """Return ordered keyword arguments suitable for ``dataset.crop(**kw)``."""
        if self._is_2d:
            pu_lo, pu_hi = sorted((self.spn_pump_min.value(), self.spn_pump_max.value()))
            pr_lo, pr_hi = sorted((self.spn_probe_min.value(), self.spn_probe_max.value()))
            t_lo, t_hi = sorted((self.spn_t2_min.value(), self.spn_t2_max.value()))
            return {
                "t2_min": t_lo,
                "t2_max": t_hi,
                "pump_min": pu_lo,
                "pump_max": pu_hi,
                "probe_min": pr_lo,
                "probe_max": pr_hi,
            }
        pr_lo, pr_hi = sorted((self.spn_probe_min.value(), self.spn_probe_max.value()))
        t_lo, t_hi = sorted((self.spn_tmin.value(), self.spn_tmax.value()))
        return {
            "tmin": t_lo,
            "tmax": t_hi,
            "probe_min": pr_lo,
            "probe_max": pr_hi,
        }

    def _apply_bounds_dict(self, bounds: dict[str, float]) -> None:
        self._syncing = True
        try:
            if self._is_2d:
                mapping = {
                    "pump_min": self.spn_pump_min,
                    "pump_max": self.spn_pump_max,
                    "probe_min": self.spn_probe_min,
                    "probe_max": self.spn_probe_max,
                    "t2_min": self.spn_t2_min,
                    "t2_max": self.spn_t2_max,
                }
            else:
                mapping = {
                    "probe_min": self.spn_probe_min,
                    "probe_max": self.spn_probe_max,
                    "tmin": self.spn_tmin,
                    "tmax": self.spn_tmax,
                }
            for key, widget in mapping.items():
                if key in bounds and bounds[key] is not None and np.isfinite(bounds[key]):
                    widget.setValue(float(bounds[key]))
        finally:
            self._syncing = False

    def _on_match_view(self) -> None:
        if self._view_bounds:
            self._apply_bounds_dict(self._view_bounds)
            self._update_summary_and_overlays()

    def _on_reset_full(self) -> None:
        if self._is_2d:
            self._apply_bounds_dict(
                {
                    "pump_min": self._pu_min_full,
                    "pump_max": self._pu_max_full,
                    "probe_min": self._pr_min_full,
                    "probe_max": self._pr_max_full,
                    "t2_min": self._t_min_full,
                    "t2_max": self._t_max_full,
                }
            )
        else:
            self._apply_bounds_dict(
                {
                    "probe_min": self._pr_min_full,
                    "probe_max": self._pr_max_full,
                    "tmin": self._t_min_full,
                    "tmax": self._t_max_full,
                }
            )
        self._update_summary_and_overlays()

    # ------------------------------------------------------------------ #
    #                          Preview & Overlays                        #
    # ------------------------------------------------------------------ #
    def _on_t2_slider_changed(self, idx: int) -> None:
        if self._t_arr.size and self.t2_label is not None:
            i = int(np.clip(idx, 0, len(self._t_arr) - 1))
            self.t2_label.setText(f"{self._t_arr[i]:.3g} {self._t_unit}")
        self._render_preview()
        self._update_summary_and_overlays()

    def _render_preview(self) -> None:
        fig = self.plot_widget.figure
        fig.clear()
        ax = fig.add_subplot(111)
        self.plot_widget.ax = ax
        self._overlay_patches.clear()

        try:
            if self._is_2d:
                from pymorgan.twoD.plot import plot_map

                pc = getattr(self.parent(), "twoD_plot_controls", None)
                t2_idx = self.t2_slider.value() if self.t2_slider is not None else 0
                t2_idx = int(np.clip(t2_idx, 0, max(0, len(self._t_arr) - 1)))
                t2_val = float(self._t_arr[t2_idx]) if self._t_arr.size else 0.0
                inv_kw = (
                    {
                        "invert_pump": pc.pump_descending(),
                        "invert_probe": pc.probe_descending(),
                    }
                    if pc is not None
                    else {}
                )
                native_u = getattr(self.dataset, "freq_units", None) or (
                    (getattr(self.dataset, "units", None) or {}).get("unitsL", "cm-1")
                )
                plot_map(
                    self.dataset,
                    t2=t2_val,
                    ax=ax,
                    show_colorbar=True,
                    t2_label=False,
                    aspect="auto",
                    freq_unit=native_u,
                    secondary_axis=False,
                    **inv_kw,
                )
            else:
                from pymorgan.oneD.plot import plot_contour

                units = getattr(self.dataset, "units", None) or {}
                x_unit = None if hlp.is_uncalibrated_probe(self.dataset) else hlp.native_x_unit(units)
                pc = getattr(self.parent(), "plot_controls", None)
                yscale = None
                if pc is not None:
                    yscale = pc.time_axis_kwargs().get("Yscale")
                plot_contour(
                    self.dataset,
                    ax=ax,
                    detector=self._detector,
                    x_axis_unit=x_unit,
                    secondary_axis=False,
                    show_colorbar=True,
                    aspect="auto",
                    Yscale=yscale,
                )
        except Exception as exc:
            logger.debug("CropDialog preview plot fallback: %s", exc, exc_info=True)
            ax.text(0.5, 0.5, f"Preview unavailable:\n{exc}", ha="center", va="center")

        style_figure(fig, is_dark_palette())
        with contextlib.suppress(Exception):
            fig.set_layout_engine(None)
            fig.tight_layout()

        selector_kw = dict(useblit=True, button=[1], minspanx=0, minspany=0, interactive=True)
        try:
            self._rect_selector = RectangleSelector(
                ax,
                self._on_rect_selected,
                props=dict(facecolor="none", edgecolor="#ea580c", linewidth=1.8, linestyle="--"),
                **selector_kw,
            )
        except TypeError:
            self._rect_selector = RectangleSelector(
                ax,
                self._on_rect_selected,
                rectprops=dict(
                    facecolor="none", edgecolor="#ea580c", linewidth=1.8, linestyle="--"
                ),
                **selector_kw,
            )
        self.plot_widget.canvas.draw_idle()

    def _on_rect_selected(self, eclick, erelease) -> None:
        if self._syncing:
            return
        if (
            eclick.xdata is None
            or erelease.xdata is None
            or eclick.ydata is None
            or erelease.ydata is None
        ):
            return
        x0, x1 = sorted((float(eclick.xdata), float(erelease.xdata)))
        y0, y1 = sorted((float(eclick.ydata), float(erelease.ydata)))
        if abs(x1 - x0) <= 1e-12 and abs(y1 - y0) <= 1e-12:
            return

        if self._is_2d:
            if self._is_vertical_2d:
                bounds = {"probe_min": x0, "probe_max": x1, "pump_min": y0, "pump_max": y1}
            else:
                bounds = {"pump_min": x0, "pump_max": x1, "probe_min": y0, "probe_max": y1}
        else:
            bounds = {"probe_min": x0, "probe_max": x1, "tmin": y0, "tmax": y1}

        self._apply_bounds_dict(bounds)
        self._update_summary_and_overlays(sync_selector=False)

    def _on_spin_edited(self) -> None:
        if self._syncing:
            return
        self._update_summary_and_overlays(sync_selector=True)

    def _kept_counts(self) -> tuple[bool, str]:
        kw = self.get_crop_kwargs()
        if self._is_2d:
            n_pu = int(np.count_nonzero((self._pu_arr >= kw["pump_min"]) & (self._pu_arr <= kw["pump_max"])))
            n_pr = int(np.count_nonzero((self._pr_arr >= kw["probe_min"]) & (self._pr_arr <= kw["probe_max"])))
            n_t2 = int(np.count_nonzero((self._t_arr >= kw["t2_min"]) & (self._t_arr <= kw["t2_max"])))
            ok = n_pu > 0 and n_pr > 0 and n_t2 > 0
            msg = (
                f"Keeping {n_pu} / {self._pu_arr.size} pump \u00d7 "
                f"{n_pr} / {self._pr_arr.size} probe \u00d7 "
                f"{n_t2} / {self._t_arr.size} t\u2082 delays"
            )
        else:
            n_t = int(np.count_nonzero((self._t_arr >= kw["tmin"]) & (self._t_arr <= kw["tmax"])))
            n_pr = int(np.count_nonzero((self._pr_arr >= kw["probe_min"]) & (self._pr_arr <= kw["probe_max"])))
            ok = n_t > 0 and n_pr > 0
            msg = (
                f"Keeping {n_t} / {self._t_arr.size} delays \u00d7 "
                f"{n_pr} / {self._pr_arr.size} {self._spec_unit}"
            )
        return ok, msg

    def _update_summary_and_overlays(self, *, sync_selector: bool = True) -> None:
        ok, msg = self._kept_counts()
        if not ok:
            self.lbl_summary.setStyleSheet("color: #c0392b; font-weight: bold;")
            self.lbl_summary.setText(f"{msg}\n(At least 1 point per axis is required.)")
        else:
            self.lbl_summary.setStyleSheet("")
            self.lbl_summary.setText(msg)
        self.btn_apply.setEnabled(ok)
        self.btn_export.setEnabled(ok)

        ax = getattr(self.plot_widget, "ax", None)
        if ax is None:
            return
        for p in self._overlay_patches:
            with contextlib.suppress(Exception):
                p.remove()
        self._overlay_patches.clear()

        kw = self.get_crop_kwargs()
        if self._is_2d:
            if self._is_vertical_2d:
                bx0, bx1 = kw["probe_min"], kw["probe_max"]
                by0, by1 = kw["pump_min"], kw["pump_max"]
            else:
                bx0, bx1 = kw["pump_min"], kw["pump_max"]
                by0, by1 = kw["probe_min"], kw["probe_max"]
        else:
            bx0, bx1 = kw["probe_min"], kw["probe_max"]
            by0, by1 = kw["tmin"], kw["tmax"]

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax_x0, ax_x1 = min(xlim), max(xlim)
        ax_y0, ax_y1 = min(ylim), max(ylim)

        cx0 = float(np.clip(bx0, ax_x0, ax_x1))
        cx1 = float(np.clip(bx1, ax_x0, ax_x1))
        cy0 = float(np.clip(by0, ax_y0, ax_y1))
        cy1 = float(np.clip(by1, ax_y0, ax_y1))

        shade_kw = dict(facecolor="0.1", edgecolor="none", alpha=0.38, zorder=5)
        margins = [
            (ax_x0, ax_y0, max(0.0, cx0 - ax_x0), max(0.0, ax_y1 - ax_y0)),
            (cx1, ax_y0, max(0.0, ax_x1 - cx1), max(0.0, ax_y1 - ax_y0)),
            (cx0, ax_y0, max(0.0, cx1 - cx0), max(0.0, cy0 - ax_y0)),
            (cx0, cy1, max(0.0, cx1 - cx0), max(0.0, ax_y1 - cy1)),
        ]
        for rx, ry, rw, rh in margins:
            if rw > 0.0 and rh > 0.0:
                rect = patches.Rectangle((rx, ry), rw, rh, **shade_kw)
                ax.add_patch(rect)
                self._overlay_patches.append(rect)

        border = patches.Rectangle(
            (cx0, cy0),
            max(0.0, cx1 - cx0),
            max(0.0, cy1 - cy0),
            facecolor="none",
            edgecolor="#ea580c",
            linewidth=1.8,
            linestyle="--",
            zorder=6,
        )
        ax.add_patch(border)
        self._overlay_patches.append(border)

        if sync_selector and self._rect_selector is not None:
            with contextlib.suppress(Exception):
                self._rect_selector.extents = (bx0, bx1, by0, by1)

        self.plot_widget.canvas.draw_idle()

    # ------------------------------------------------------------------ #
    #                         Apply & Export Actions                     #
    # ------------------------------------------------------------------ #
    def _on_apply(self) -> None:
        kw = self.get_crop_kwargs()
        try:
            self.dataset.crop(**kw)
        except Exception as exc:
            logger.exception("Crop failed: %s", exc)
            QMessageBox.critical(self, "Crop Failed", f"Could not crop dataset:\n{exc}")
            return
        self.dataset_cropped.emit(self.dataset)
        self.accept()

    def _on_export_copy(self) -> None:
        kw = self.get_crop_kwargs()
        try:
            cropped = self.dataset.copy().crop(**kw)
        except Exception as exc:
            logger.exception("Crop copy failed: %s", exc)
            QMessageBox.critical(self, "Crop Failed", f"Could not crop dataset copy:\n{exc}")
            return

        src_stem = Path(str(getattr(self.dataset, "source", "") or "dataset")).stem or "dataset"
        if self._is_2d:
            filters = "PyMORGAN 2D HDF5 (*.h5);;P2DAT format (*.p2dat)"
            default_name = f"{src_stem}_cropped.h5"
        else:
            filters = (
                "PDAT format (*.pdat);;"
                "PyMORGAN HDF5 (*.h5);;"
                "Glotaran ASCII (*.ascii);;"
                "Glotaran NetCDF4 (*.nc)"
            )
            default_name = f"{src_stem}_cropped.pdat"

        filepath, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Cropped Dataset Copy",
            default_name,
            filters,
        )
        if not filepath:
            return

        try:
            low_path = filepath.lower()
            low_filt = (selected_filter or "").lower()
            written_msg = filepath
            if self._is_2d:
                if low_path.endswith((".h5", ".hdf5")) or "hdf5" in low_filt or "h5" in low_filt:
                    out = cropped.to_hdf5(filepath)
                else:
                    out = cropped.to_p2dat(filepath)
                written_msg = str(out)
            else:
                if low_path.endswith(".pdat") or "pdat" in low_filt:
                    paths = cropped.to_pdat(filepath)
                    if paths:
                        written_msg = "\n".join(str(p) for p in paths)
                elif low_path.endswith((".h5", ".hdf5")) or "hdf5" in low_filt or "h5" in low_filt:
                    written_msg = str(cropped.save_hdf5(filepath))
                elif low_path.endswith(".ascii") or "ascii" in low_filt:
                    written_msg = str(cropped.to_glotaran_ascii(filepath))
                elif low_path.endswith(".nc") or "netcdf" in low_filt:
                    written_msg = str(cropped.to_glotaran_nc(filepath))
                else:
                    paths = cropped.to_pdat(filepath)
                    if paths:
                        written_msg = "\n".join(str(p) for p in paths)

            QMessageBox.information(
                self,
                "Export Complete",
                f"Cropped dataset copy exported to:\n{written_msg}",
            )
        except Exception as exc:
            logger.exception("Export cropped copy failed: %s", exc)
            QMessageBox.critical(
                self, "Export Failed", f"Could not export cropped dataset:\n{exc}"
            )

"""2D tab: dataset loading, contour preview, cuts, slices and spectral diffusion.

Part of the :class:`~pymorgan.gui.main_window.MainWindow` implementation, split out
as a mixin so each tab lives in its own module. The methods are unchanged moves:
they run as ``MainWindow`` methods and bind to the widgets declared in
``main_window.ui``, so ``self`` is always the main window.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyqt6")

from pathlib import Path

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import (
    QStandardItem,
)
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QMessageBox,
    QPushButton,
    QSpinBox,
)

import pymorgan as pm
from pymorgan import helpers as hlp

from ...log import get_logger
from ..busy import busy_guard
from ..mw_common import (
    _safe_set_limits,
    get_combo_datatype,
)

logger = get_logger(__name__)


class TwoDTabMixin:
    """2D tab: dataset loading, contour preview, cuts, slices and spectral diffusion."""

    def _on_twoD_datatype_changed(self, text: str):
        self.twoD_refresh_dataset_list()
        dt = self._current_twoD_datatype()
        is_preprocessed = dt in ("P2DAT", "RAL_Proc")
        self._set_twoD_phasing_tab_visible(not is_preprocessed)

    def _set_twoD_phasing_tab_visible(self, visible: bool):
        subtabs = getattr(self, "twoD_subtabs", None)
        if subtabs is None or self._twoD_phasingft_tab_widget is None:
            return
        index = -1
        for i in range(subtabs.count()):
            if subtabs.widget(i) == self._twoD_phasingft_tab_widget:
                index = i
                break
        if visible:
            if index == -1:
                subtabs.insertTab(
                    1, self._twoD_phasingft_tab_widget, self._twoD_phasingft_tab_title
                )
        else:
            if index != -1:
                subtabs.removeTab(index)

    def twoD_reload_data(self):
        if self._twoD_current_path:
            dt = getattr(self.twoD_dataset, "data_type", None)
            self.twoD_load_path(self._twoD_current_path, data_type=dt, force=True)

    @busy_guard("Loading 2D dataset...")
    def twoD_load_path(self, path: str, data_type: str | None = None, force: bool = False):
        """Load a 2-D dataset from ``path``."""
        combo = getattr(self, "twoD_datatype_cbx", None)
        dt = data_type or get_combo_datatype(combo, "P2DAT")

        if (
            not force
            and self._twoD_current_path == path
            and self.twoD_dataset is not None
            and getattr(self.twoD_dataset, "data_type", None) == dt
        ):
            return

        prev_delay = None
        prev_idx = None
        if self.twoD_dataset is not None:
            lst_delays = getattr(self, "twoD_t2delay_lst", None)
            if lst_delays is not None:
                curr_idx = lst_delays.currentIndex().row()
                if curr_idx >= 0 and curr_idx < len(self.twoD_dataset.delays):
                    prev_delay = self.twoD_dataset.delays[curr_idx]
                    prev_idx = curr_idx

        is_slow = dt not in ("P2DAT", "RAL_Proc")
        progress = None
        if is_slow:
            try:
                from ...twoD.progress import ProgressTracker

                progress = ProgressTracker(
                    1, title="Loading & Processing 2D Dataset", label="Initializing..."
                )
            except Exception:
                progress = None

        try:
            self.twoD_dataset = pm.load_2D(path, data_type=dt, progress_tracker=progress)
            self._twoD_current_path = path

            if dt == "P2DAT":
                sub_chk = getattr(self, "twoD_sub_scat_chk", None)
                if sub_chk is not None:
                    sub_chk.blockSignals(True)
                    sub_chk.setChecked(False)
                    sub_chk.blockSignals(False)

            if self.twoD_dataset is not None and self.twoD_dataset.datatype in (
                "shaper",
                "interferometer",
            ):
                # Get apodization method
                apod_cbx = getattr(self, "twoD_apodisation_cbx", None)
                apodise_method = apod_cbx.currentText() if apod_cbx is not None else "0"

                # Get zero-padding parameters
                pad_chk = getattr(self, "twoD_zeropad_chk", None)
                zeropad_enable = pad_chk.isChecked() if pad_chk is not None else True
                pad_factor = getattr(self, "twoD_zeropad_factor", None)
                zeropad_factor = pad_factor.value() if pad_factor is not None else 1
                pad_next2k = getattr(self, "twoD_zeropad_next2k_chk", None)
                zeropad_next2k = pad_next2k.isChecked() if pad_next2k is not None else False

                # Get phase parameters
                phase_cbx = getattr(self, "twoD_phase_fit_cbx", None)
                phase_method = phase_cbx.currentText() if phase_cbx is not None else "No fit"
                phase_range = getattr(self, "twoD_phase_fit_range", None)
                phase_points = phase_range.value() if phase_range is not None else 10

                # Get pump correction
                pump_corr = getattr(self, "twoD_pump_corr_chk", None)
                pumpcorrection = pump_corr.isChecked() if pump_corr is not None else False

                # Get background scattering subtraction
                sub_chk = getattr(self, "twoD_sub_scat_chk", None)
                sub_delay = getattr(self, "twoD_sub_scat_delay", None)
                bkg_sub = sub_chk.isChecked() if sub_chk is not None else False
                bkgIdx = (sub_delay.value() - 1) if (sub_delay is not None and bkg_sub) else 0
                if bkgIdx < 0:
                    bkgIdx = 0

                self._twoD_raw_probe = self.twoD_dataset.probe.copy() if hasattr(self.twoD_dataset, "probe") and self.twoD_dataset.probe is not None else None
                self._twoD_raw_cal_level = getattr(self.twoD_dataset, "cal_level", None)

                # Get probe autocalibration
                auto_cal = getattr(self, "twoD_auto_cal_chk", None)
                autocalibrate_probe = auto_cal.isChecked() if auto_cal is not None else False
                cal_probe_vec = self._get_cal_tab_probe_curve(len(self.twoD_dataset.probe)) if autocalibrate_probe else None

                self.twoD_dataset.process(
                    apodise_method=apodise_method,
                    zeropad_enable=zeropad_enable,
                    zeropad_factor=zeropad_factor,
                    zeropad_next2k=zeropad_next2k,
                    phase_method=phase_method,
                    phase_points=phase_points,
                    pumpcorrection=pumpcorrection,
                    bkg_sub=False,
                    bkgIdx=bkgIdx,
                    autocalibrate_probe=autocalibrate_probe,
                    cal_probe_vector=cal_probe_vec,
                    progress_tracker=progress,
                )
                if autocalibrate_probe and cal_probe_vec is not None:
                    self.twoD_dataset.probe = cal_probe_vec.copy()
                    self.twoD_dataset.cal_level = "autocalibrated"
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            self.twoD_dataset = None
            self._update_spectral_diffusion_controls_state()
            self._set_twoD_dataset_widgets_visible(False)
            self._update_twoD_sample_info()
            # Restore phasing tab visibility to combo default on failure
            self._set_twoD_phasing_tab_visible(dt not in ("P2DAT", "RAL_Proc"))
            # Clear other plot
            if hasattr(self, "_twoD_update_other_plot"):
                self._twoD_update_other_plot()
            return
        finally:
            if progress is not None:
                progress.close()

        name = Path(path).name
        dt_type = (
            f"{self.twoD_dataset.datatype} - IN PROGRESS"
            if self.twoD_dataset.in_progress
            else self.twoD_dataset.datatype
        )
        cal_str = self.twoD_dataset.calibration_status()
        msg = f"Loaded 2D ({dt_type}): {name} — Probe calibration: {cal_str}"
        self.statusBar().showMessage(msg)

        self._update_twoD_delay_list(prev_delay, prev_idx)
        self._set_twoD_phasing_tab_visible(dt not in ("P2DAT", "RAL_Proc"))
        self._update_spectral_diffusion_controls_state()

        if self.twoD_plot_controls is not None:
            lst_delays = getattr(self, "twoD_t2delay_lst", None)
            active_idx = lst_delays.currentIndex().row() if lst_delays is not None else 0
            if active_idx < 0 or active_idx >= len(self.twoD_dataset.delays):
                active_idx = max(0, len(self.twoD_dataset.delays) - 1)
            self.twoD_plot_controls.set_dataset(self.twoD_dataset, active_idx)

        # Update scatter delay subtraction spin box maximum to match delays count
        _scat_delay = getattr(self, "twoD_sub_scat_delay", None)
        if _scat_delay is not None:
            _scat_delay.blockSignals(True)
            _scat_delay.setRange(1, len(self.twoD_dataset.delays))
            _scat_delay.setValue(1)  # Default to 1st delay as specified by user
            _scat_delay.blockSignals(False)

        # Configure TD/PH/CAL buttons and spinner range
        if hasattr(self, "twoD_other_btn_TD"):
            has_shaper = dt not in ("P2DAT", "RAL_Proc")
            self.twoD_other_btn_TD.setEnabled(has_shaper)
            self.twoD_other_btn_PH.setEnabled(has_shaper)

            n_pixels = self.twoD_dataset.Z_R.shape[1]
            self.twoD_other_pixel_spin.blockSignals(True)
            self.twoD_other_pixel_spin.setRange(0, n_pixels)
            self.twoD_other_pixel_spin.setValue(0)
            self.twoD_other_pixel_spin.blockSignals(False)

            if not has_shaper:
                self.twoD_other_btn_CAL.setChecked(True)

        self._twoD_preview_contour()
        self._update_twoD_sample_info()

        # Explicitly update the other plot after preview contour (which handles bkg correct)
        if hasattr(self, "_twoD_update_other_plot"):
            self._twoD_update_other_plot()

        self._set_twoD_dataset_widgets_visible(True)

    def _update_twoD_delay_list(self, prev_delay: float | None = None, prev_idx: int | None = None):
        """Populate the delays list and select the one closest to ``prev_delay`` (or the last delay if index exceeds new dataset delays)."""
        self._twoD_delay_model.clear()
        if self.twoD_dataset is None:
            return

        delays = self.twoD_dataset.delays
        n_delays = len(delays)
        if n_delays == 0:
            return

        for delay in delays:
            if abs(delay) < 1 and delay != 0:
                delay_str = f"{delay * 1000:.3g} fs"
            elif abs(delay) >= 1 and abs(delay) < 1e3:
                delay_str = f"{delay:.3g} ps"
            elif abs(delay) >= 1e3 and abs(delay) < 1e6:
                delay_str = f"{delay / 1000:.3g} ns"
            else:
                delay_str = f"{delay:.3g} ps"
            item = QStandardItem(delay_str)
            item.setEditable(False)
            self._twoD_delay_model.appendRow(item)

        lst_delays = getattr(self, "twoD_t2delay_lst", None)
        if lst_delays is not None:
            target_idx = 0
            if prev_idx is not None and prev_idx >= n_delays:
                target_idx = n_delays - 1
            elif prev_delay is not None:
                target_idx = int(np.argmin(np.abs(delays - prev_delay)))
                target_idx = max(0, min(target_idx, n_delays - 1))

            index = self._twoD_delay_model.index(target_idx, 0)
            lst_delays.blockSignals(True)
            lst_delays.setCurrentIndex(index)
            lst_delays.blockSignals(False)

    def _on_twoD_delay_activated(self, index, _previous=None):
        if index is None or not index.isValid() or self.twoD_dataset is None:
            return
        row = index.row()
        if self.twoD_plot_controls is not None:
            self.twoD_plot_controls.update_delay_index(row, force_reset_pct=False)
        self._twoD_preview_contour()
        self._twoD_update_phase_coeffs_display()

    def _on_twoD_sync_limits(self):
        if self.twoD_dataset is None or self.twoD_plot_controls is None:
            return
        ax = getattr(self.twoDaxes, "ax", None)
        if ax is None:
            return
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        pc = self.twoD_plot_controls
        for w, val in (
            (pc.pump_min, xmin),
            (pc.pump_max, xmax),
            (pc.probe_min, ymin),
            (pc.probe_max, ymax),
        ):
            if w is not None:
                w.blockSignals(True)
                w.setValue(val)
                w.blockSignals(False)

        pc.limitsChanged.emit()

    def _update_twoD_sample_info(self):
        """Refresh the twoD sample info text box."""
        widget = getattr(self, "twoD_SampleInfo_text", None)
        if widget is None:
            return
        widget.setHtml("" if self.twoD_dataset is None else self.twoD_dataset.sample_info())

    def _on_twoD_include_ftir_toggled(self, checked: bool):
        if checked and self.twoD_dataset is not None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Overlay Spectrum (FTIR / Pump-Probe)",
                self._rootdir_text() or "",
                "Spectra (*.csv *.dat *.txt *.dpt *.pdat);;All files (*)",
            )
            if path:
                try:
                    if path.lower().endswith(".pdat"):
                        ds = pm.load_1D(path)
                        y = ds.Z.mean(axis=0)
                        self._twoD_overlay = (ds.probe, y)
                    else:
                        spec = pm.load_spectrum(path)
                        self._twoD_overlay = (spec.x, spec.y)
                except Exception as exc:
                    QMessageBox.warning(self, "Load overlay failed", str(exc))
                    self._twoD_overlay = None
                    _include_ftir = getattr(self, "twoD_include_ftir_chk", None)
                    if _include_ftir is not None:
                        _include_ftir.blockSignals(True)
                        _include_ftir.setChecked(False)
                        _include_ftir.blockSignals(False)
            else:
                _include_ftir = getattr(self, "twoD_include_ftir_chk", None)
                if _include_ftir is not None:
                    _include_ftir.blockSignals(True)
                    _include_ftir.setChecked(False)
                    _include_ftir.blockSignals(False)
        else:
            self._twoD_overlay = None
        self._twoD_rerender_preview()

    def render_twoD_contour_frame(self, t2_val: float, ax=None, vmin=None, vmax=None):
        """Render a single 2D contour map frame matching the 'Plot Contour' button output."""
        if self.twoD_dataset is None or ax is None:
            return

        pm.apply_style()

        kwargs = (
            self.twoD_plot_controls.contour_kwargs() if self.twoD_plot_controls is not None else {}
        )

        s = pm.get_settings()
        pump_axis_val = s.pump_axis
        if hasattr(pump_axis_val, "value"):
            pump_axis_val = pump_axis_val.value
        is_vertical = pump_axis_val == "Vertical"

        top_spectrum = None
        include_ftir = getattr(self, "twoD_include_ftir_chk", None)
        if include_ftir is not None and include_ftir.isChecked() and self._twoD_overlay is not None:
            top_spectrum = self._twoD_overlay

        if vmin is None or vmax is None:
            zmin, zmax = (
                self.twoD_plot_controls.zlimits()
                if self.twoD_plot_controls is not None
                else (None, None)
            )
            if vmin is None:
                vmin = zmin
            if vmax is None:
                vmax = zmax

        pm.twoD.plot_map(
            self.twoD_dataset,
            t2_val,
            ax=ax,
            cmap_ID=kwargs.get("cmap_ID"),
            ShowLines=kwargs.get("ShowLines"),
            Nskip=kwargs.get("Nskip"),
            white_levels=kwargs.get("white_levels"),
            top_spectrum=top_spectrum,
            top_label="Overlay" if top_spectrum is not None else None,
            vmin=vmin,
            vmax=vmax,
            text_white_bg=kwargs.get("text_white_bg"),
            cut_plot=kwargs.get("cut_plot", False),
            pump_lim=kwargs.get("pump_lim"),
            probe_lim=kwargs.get("probe_lim"),
            Nlevels=kwargs.get("Nlevels"),
            filled=kwargs.get("filled"),
        )

        if self.twoD_plot_controls is not None:
            xlim = (
                self.twoD_plot_controls.probe_lim()
                if is_vertical
                else self.twoD_plot_controls.pump_lim()
            )
            ylim = (
                self.twoD_plot_controls.pump_lim()
                if is_vertical
                else self.twoD_plot_controls.probe_lim()
            )
            _safe_set_limits(ax, xlim, ylim)
        sq_chk = getattr(self, "twoD_PC_square", None)
        if sq_chk is not None and sq_chk.isChecked():
            ax.set_aspect("equal")
        else:
            ax.set_aspect("auto")

    def _plot_twoD_cut(self, kind: str):
        """Open the active 2D contour or 3D surface plot in a new Matplotlib window."""
        if self.twoD_dataset is None:
            return

        import matplotlib.pyplot as plt

        lst_delays = getattr(self, "twoD_t2delay_lst", None)
        active_idx = lst_delays.currentIndex().row() if lst_delays is not None else 0
        if active_idx < 0 or active_idx >= len(self.twoD_dataset.delays):
            active_idx = max(0, len(self.twoD_dataset.delays) - 1)
        active_delay = self.twoD_dataset.delays[active_idx]

        try:
            if kind == "surface":
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection="3d")
                # Inherit the pump/probe window and the colour scale currently
                # shown in the embedded contour preview.
                pc = self.twoD_plot_controls
                surf_kwargs = {}
                if pc is not None:
                    zmin, zmax = pc.zlimits()
                    surf_kwargs = {
                        "pump_lim": pc.pump_lim(),
                        "probe_lim": pc.probe_lim(),
                        "zlim": (zmin, zmax),
                        "vmin": zmin,
                        "vmax": zmax,
                    }
                self.twoD_dataset.plot_surface(t2=active_delay, ax=ax, **surf_kwargs)
                fig.tight_layout()
                plt.show(block=False)
            else:
                fig, ax = plt.subplots(figsize=(6, 6))
                self.render_twoD_contour_frame(active_delay, ax=ax)
                fig.tight_layout()
                plt.show(block=False)
        except Exception as exc:
            QMessageBox.warning(self, "Plot failed", str(exc))

    def _twoD_interactive(self) -> bool:
        chk = getattr(self, "twoD_interactive_chk", None)
        return chk.isChecked() if chk is not None else False

    def _on_twoD_kinetics_clicked(self):
        """Plot kinetics at coordinates or integrated area for the 2D dataset."""
        if self.twoD_dataset is None:
            return

        choice, ok = QInputDialog.getItem(
            self,
            "Plot Kinetics",
            "Choose kinetics selection mode:",
            ["Single Coordinate", "Area Integration"],
            0,
            False,
        )
        if not ok:
            return

        s = pm.get_settings()
        pump_axis_val = s.pump_axis
        if hasattr(pump_axis_val, "value"):
            pump_axis_val = pump_axis_val.value
        is_vertical = pump_axis_val == "Vertical"

        ax = getattr(self.twoDaxes, "ax", None)
        if ax is None:
            return

        if choice == "Single Coordinate":
            if self._twoD_interactive():
                from ..picker import ContourPicker

                self.statusBar().showMessage(
                    "Interactive kinetics: left-click to pick coordinate(s), right-click to remove last, Enter to plot, Esc to cancel."
                )

                def done(points):
                    self._picker = None
                    if not points:
                        self.statusBar().showMessage("Interactive selection cancelled.")
                        return
                    self.statusBar().clearMessage()

                    resolved_coords = []
                    for x, y in points:
                        pump_wn = y if is_vertical else x
                        probe_wn = x if is_vertical else y
                        resolved_coords.append((pump_wn, probe_wn))
                    self._plot_single_coordinates(resolved_coords)

                self._picker = ContourPicker(self.twoDaxes.canvas, ax, "xy", done).start()
            else:
                text, ok = QInputDialog.getText(
                    self,
                    "Kinetics coordinates",
                    "Enter coordinates as pump,probe (semicolon-separated for multiple, e.g. 1650,1650; 1660,1650):",
                )
                if not ok or not text.strip():
                    return
                try:
                    coords = []
                    for term in text.split(";"):
                        p_str, pr_str = term.split(",")
                        coords.append((float(p_str), float(pr_str)))
                    self._plot_single_coordinates(coords)
                except Exception as exc:
                    QMessageBox.warning(self, "Invalid Input", f"Error parsing input: {exc}")

        else:  # Area Integration
            if self._twoD_interactive():
                from matplotlib.widgets import RectangleSelector

                self.statusBar().showMessage(
                    "Draw a rectangle on the contour plot to select the area. Press ENTER to confirm."
                )
                self._selected_areas = []
                self._rect_cid = None

                def onselect(eclick, erelease):
                    pass

                def on_key(event):
                    if event.key == "enter":
                        if self._rect_selector is None:
                            return
                        is_visible = getattr(self._rect_selector, "visible", True)
                        if not is_visible:
                            return
                        extents = self._rect_selector.extents
                        xmin, xmax, ymin, ymax = extents
                        if abs(xmax - xmin) < 1e-5 or abs(ymax - ymin) < 1e-5:
                            return

                        if is_vertical:
                            pump_range = (ymin, ymax)
                            probe_range = (xmin, xmax)
                        else:
                            pump_range = (xmin, xmax)
                            probe_range = (ymin, ymax)

                        self._selected_areas.append((pump_range, probe_range))

                        from matplotlib.patches import Rectangle

                        rect_patch = Rectangle(
                            (xmin, ymin),
                            xmax - xmin,
                            ymax - ymin,
                            linewidth=1,
                            edgecolor="orange",
                            facecolor="none",
                            linestyle="--",
                        )
                        ax.add_patch(rect_patch)
                        self.twoDaxes.canvas.draw_idle()

                        reply = QMessageBox.question(
                            self,
                            "Area Selection",
                            "Area selected. Do you want to select another area?",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.No,
                        )
                        if reply == QMessageBox.StandardButton.Yes:
                            self.statusBar().showMessage(
                                "Draw another rectangle on the contour plot. Press ENTER to confirm."
                            )
                        else:
                            if self._rect_cid is not None:
                                self.twoDaxes.canvas.mpl_disconnect(self._rect_cid)
                                self._rect_cid = None
                            self._rect_selector.set_active(False)
                            self._rect_selector = None
                            self.statusBar().clearMessage()
                            self._plot_areas_kinetics(self._selected_areas)
                            self._twoD_rerender_preview()

                selector_kwargs = dict(
                    useblit=True, button=[1], minspanx=5, minspany=5, interactive=True
                )
                try:
                    self._rect_selector = RectangleSelector(
                        ax,
                        onselect,
                        props=dict(
                            facecolor="orange", edgecolor="darkorange", alpha=0.3, fill=True
                        ),
                        **selector_kwargs,
                    )
                except TypeError:
                    self._rect_selector = RectangleSelector(
                        ax,
                        onselect,
                        rectprops=dict(
                            facecolor="orange", edgecolor="darkorange", alpha=0.3, fill=True
                        ),
                        **selector_kwargs,
                    )
                self._rect_cid = self.twoDaxes.canvas.mpl_connect("key_press_event", on_key)
                self.twoDaxes.canvas.setFocus()
            else:
                text, ok = QInputDialog.getText(
                    self,
                    "Area Integration ranges",
                    "Enter ranges as pump_min:pump_max,probe_min:probe_max (semicolon-separated for multiple, e.g. 1640:1660,1640:1660):",
                )
                if not ok or not text.strip():
                    return
                try:
                    areas = []
                    for term in text.split(";"):
                        p_str, pr_str = term.split(",")
                        p_min, p_max = map(float, p_str.split(":"))
                        pr_min, pr_max = map(float, pr_str.split(":"))
                        areas.append(((p_min, p_max), (pr_min, pr_max)))
                    self._plot_areas_kinetics(areas)
                except Exception as exc:
                    QMessageBox.warning(self, "Invalid Input", f"Error parsing input: {exc}")

    @busy_guard("Plotting kinetics...")
    def _plot_single_coordinates(self, coords):
        import matplotlib.pyplot as plt

        from ... import helpers as hlp
        from ...oneD.plot import plot_kinetics

        pm.apply_style()
        s = pm.get_settings()

        fig, ax = plt.subplots(figsize=s.kinetics_figsize)

        Nplots = len(coords)
        cm = hlp.get_trace_cmap(s.traces_cmap, Nplots)

        for i, (pump_wn, probe_wn) in enumerate(coords):
            slice_ds = self.twoD_dataset.get_slice_at_pump(pump_wn)
            actual_pump = self.twoD_dataset.pump[
                int(np.argmin(np.abs(self.twoD_dataset.pump - pump_wn)))
            ]
            actual_probe = self.twoD_dataset.probe[
                int(np.argmin(np.abs(self.twoD_dataset.probe - probe_wn)))
            ]

            lbl = f"[{actual_pump:.1f}, {actual_probe:.1f}] cm$^{{-1}}$"

            n_before = len(ax.get_lines())
            plot_kinetics(slice_ds, [probe_wn], ax=ax, fig=fig, normY=self._twoD_norm())

            # Post-process colour and label
            line = ax.get_lines()[n_before]
            line.set_color(cm[i])
            line.set_label(lbl)

        ax.legend()
        fig.tight_layout()
        plt.show(block=False)

    @busy_guard("Plotting area kinetics...")
    def _plot_areas_kinetics(self, areas):
        import matplotlib.pyplot as plt
        from scipy.integrate import trapezoid

        from ... import helpers as hlp
        from ...oneD.plot import plot_kinetics
        from ...twoD.analyse import SliceDataset1D

        pm.apply_style()
        s = pm.get_settings()

        fig, ax = plt.subplots(figsize=s.kinetics_figsize)

        Nplots = len(areas)
        cm = hlp.get_trace_cmap(s.traces_cmap, Nplots)

        for i, (pump_range, probe_range) in enumerate(areas):
            p_min, p_max = min(pump_range), max(pump_range)
            pr_min, pr_max = min(probe_range), max(probe_range)

            pump_mask = (self.twoD_dataset.pump >= p_min) & (self.twoD_dataset.pump <= p_max)
            probe_mask = (self.twoD_dataset.probe >= pr_min) & (self.twoD_dataset.probe <= pr_max)

            if not np.any(pump_mask) or not np.any(probe_mask):
                continue

            sub_Z = self.twoD_dataset.Z[pump_mask, :, :]
            sub_Z = sub_Z[:, probe_mask, :]

            p_sub = self.twoD_dataset.pump[pump_mask]
            pr_sub = self.twoD_dataset.probe[probe_mask]

            if len(p_sub) > 1:
                Z_int = trapezoid(sub_Z, p_sub, axis=0)
            else:
                Z_int = sub_Z[0, :, :]

            if len(pr_sub) > 1:
                decay = trapezoid(Z_int, pr_sub, axis=0)
            else:
                decay = Z_int[0, :]

            dummy_probe = [(pr_min + pr_max) / 2.0]
            slice_ds = SliceDataset1D(
                delays=self.twoD_dataset.delays,
                probe=dummy_probe,
                Z_matrix=decay[:, np.newaxis],
                units=self.twoD_dataset.units,
            )

            n_before = len(ax.get_lines())
            plot_kinetics(slice_ds, dummy_probe, ax=ax, fig=fig, normY=self._twoD_norm())

            # Post-process colour and label
            line = ax.get_lines()[n_before]
            line.set_color(cm[i])
            line.set_label(f"Int. [{p_min:.1f}:{p_max:.1f}, {pr_min:.1f}:{pr_max:.1f}]")

        ax.legend()
        fig.tight_layout()
        plt.show(block=False)

    def _on_twoD_slices_clicked(self):
        """Plot slices of the 2D dataset at the active delay."""
        if self.twoD_dataset is None:
            return

        lst_delays = getattr(self, "twoD_t2delay_lst", None)
        active_idx = lst_delays.currentIndex().row() if lst_delays is not None else 0
        if active_idx < 0 or active_idx >= len(self.twoD_dataset.delays):
            active_idx = max(0, len(self.twoD_dataset.delays) - 1)
        active_delay = self.twoD_dataset.delays[active_idx]

        options = [
            "Diagonal",
            "Off-diagonal (parallel offset)",
            "Anti-diagonal",
            "Combined Diagonal + Anti-diagonal (Normalised)",
            "Along fixed pump WN",
            "Along fixed probe WN",
            "Integrate along pump axis",
            "Integrate along probe axis",
            "Along several pump WN",
            "Along several probe WN",
        ]
        choice, ok = QInputDialog.getItem(
            self, "Plot Slices", "Choose slice type:", options, 0, False
        )
        if not ok:
            return

        s = pm.get_settings()
        pump_axis_val = s.pump_axis
        if hasattr(pump_axis_val, "value"):
            pump_axis_val = pump_axis_val.value
        is_vertical = pump_axis_val == "Vertical"

        ax = getattr(self.twoDaxes, "ax", None)
        if ax is None:
            return

        contour_xlim = ax.get_xlim()
        contour_ylim = ax.get_ylim()
        contour_pump_lim = contour_ylim if is_vertical else contour_xlim
        contour_probe_lim = contour_xlim if is_vertical else contour_ylim

        import matplotlib.pyplot as plt

        from ...oneD.plot import plot_spectra
        from ...twoD.analyse import SliceDataset1D
        from ..picker import ContourPicker

        if choice == "Diagonal":
            try:
                freq, signal = self.twoD_dataset.diagonal(active_delay)
                slice_ds = SliceDataset1D(
                    delays=[active_delay],
                    probe=freq,
                    Z_matrix=signal[np.newaxis, :],
                    units=self.twoD_dataset.units,
                )
                fig, new_ax = plt.subplots(figsize=(7.5, 4.375))
                n_before = len(new_ax.get_lines())
                plot_spectra(slice_ds, [active_delay], ax=new_ax, fig=fig, normY=self._twoD_norm())

                # Make the line black and label it "Diagonal"
                line = new_ax.get_lines()[n_before]
                line.set_color("k")
                line.set_label("Diagonal")

                pumpAll, probeAll, pump_symbol, probe_symbol = self._get_twoD_axis_labels()
                new_ax.set_xlabel(f"{pump_symbol} = {probe_symbol}")
                new_ax.set_title(f"Diagonal at $t_2$ = {active_delay:.2f} ps")
                new_ax.set_xlim(contour_pump_lim)
                new_ax.legend()
                fig.tight_layout()
                plt.show(block=False)
            except Exception as exc:
                QMessageBox.warning(self, "Plot failed", str(exc))

        elif choice == "Off-diagonal (parallel offset)":
            offset, ok_off = QInputDialog.getDouble(
                self, "Off-diagonal Offset", "Enter offset (probe = pump + offset in cm⁻¹):", 15.0, -500.0, 500.0, 1
            )
            if ok_off:
                try:
                    freq, signal = self.twoD_dataset.diagonal(active_delay, offset=offset, method="cubic")
                    slice_ds = SliceDataset1D(
                        delays=[active_delay],
                        probe=freq,
                        Z_matrix=signal[np.newaxis, :],
                        units=self.twoD_dataset.units,
                    )
                    fig, new_ax = plt.subplots(figsize=(7.5, 4.375))
                    n_before = len(new_ax.get_lines())
                    plot_spectra(slice_ds, [active_delay], ax=new_ax, fig=fig, normY=self._twoD_norm())

                    line = new_ax.get_lines()[n_before]
                    line.set_color("purple")
                    line.set_label(f"Off-diagonal (offset={offset:+.1f} " + r"cm$^{-1}$)")

                    pumpAll, probeAll, pump_symbol, probe_symbol = self._get_twoD_axis_labels()
                    new_ax.set_xlabel(f"{pump_symbol} ({probe_symbol} = {pump_symbol} {offset:+.1f})")
                    new_ax.set_title(f"Off-diagonal Cut (offset={offset:+.1f} " + r"cm$^{-1}$) at $t_2$ = " + f"{active_delay:.2f} ps")
                    new_ax.legend()
                    fig.tight_layout()
                    plt.show(block=False)
                except Exception as exc:
                    QMessageBox.warning(self, "Plot failed", str(exc))

        elif choice == "Anti-diagonal":
            w1_mid = float(np.mean(self.twoD_dataset.pump))
            w3_mid = float(np.mean(self.twoD_dataset.probe))
            w1_val, ok1 = QInputDialog.getDouble(
                self, "Anti-diagonal Centre", "Centre pump wavenumber w1 (cm⁻¹):", w1_mid, float(np.min(self.twoD_dataset.pump)), float(np.max(self.twoD_dataset.pump)), 1
            )
            if ok1:
                w3_val, ok2 = QInputDialog.getDouble(
                    self, "Anti-diagonal Centre", "Centre probe wavenumber w3 (cm⁻¹):", w3_mid, float(np.min(self.twoD_dataset.probe)), float(np.max(self.twoD_dataset.probe)), 1
                )
                if ok2:
                    try:
                        rel_disp, signal = self.twoD_dataset.antidiagonal(centre=(w1_val, w3_val), t2=active_delay, method="cubic")
                        slice_ds = SliceDataset1D(
                            delays=[active_delay],
                            probe=rel_disp,
                            Z_matrix=signal[np.newaxis, :],
                            units=self.twoD_dataset.units,
                        )
                        fig, new_ax = plt.subplots(figsize=(7.5, 4.375))
                        n_before = len(new_ax.get_lines())
                        plot_spectra(slice_ds, [active_delay], ax=new_ax, fig=fig, normY=self._twoD_norm())

                        line = new_ax.get_lines()[n_before]
                        line.set_color("r")
                        line.set_label("Anti-diagonal")

                        new_ax.set_xlabel(r"Relative Displacement $\Delta\omega$ (cm$^{-1}$)")
                        new_ax.set_title(f"Anti-diagonal Cut at ({w1_val:.1f}, {w3_val:.1f}) | $t_2$ = {active_delay:.2f} ps")
                        new_ax.legend()
                        fig.tight_layout()
                        plt.show(block=False)
                    except Exception as exc:
                        QMessageBox.warning(self, "Plot failed", str(exc))

        elif choice == "Combined Diagonal + Anti-diagonal (Normalised)":
            w1_mid = float(np.mean(self.twoD_dataset.pump))
            w3_mid = float(np.mean(self.twoD_dataset.probe))
            w1_val, ok1 = QInputDialog.getDouble(
                self, "Cut Centre", "Centre pump wavenumber w1 (cm⁻¹):", w1_mid, float(np.min(self.twoD_dataset.pump)), float(np.max(self.twoD_dataset.pump)), 1
            )
            if ok1:
                w3_val, ok2 = QInputDialog.getDouble(
                    self, "Cut Centre", "Centre probe wavenumber w3 (cm⁻¹):", w3_mid, float(np.min(self.twoD_dataset.probe)), float(np.max(self.twoD_dataset.probe)), 1
                )
                if ok2:
                    try:
                        rel_disp, norm_diag, norm_antidiag = self.twoD_dataset.compare_diag_antidiag(centre=(w1_val, w3_val), t2=active_delay, method="cubic")

                        fig, new_ax = plt.subplots(figsize=(7.5, 4.5))
                        new_ax.plot(rel_disp, norm_diag, "k-", linewidth=2.0, label="Diagonal (Normalised)")
                        new_ax.plot(rel_disp, norm_antidiag, "r--", linewidth=2.0, label="Anti-diagonal (Normalised)")

                        new_ax.set_xlabel(r"Relative Wavenumber Displacement $\Delta\omega$ (cm$^{-1}$)")
                        new_ax.set_ylabel("Normalised Intensity (max = 1.0)")
                        new_ax.set_title(f"Combined Diagonal & Anti-diagonal Profiles at ({w1_val:.1f}, {w3_val:.1f}) | $t_2$ = {active_delay:.2f} ps")
                        new_ax.grid(True, linestyle=":", alpha=0.6)
                        new_ax.legend(loc="best")

                        # Draw cut line overlays on 2D contour plot
                        ax.plot([w3_val + rel_disp[0], w3_val + rel_disp[-1]], [w1_val + rel_disp[0], w1_val + rel_disp[-1]], "k-", linewidth=1.5, alpha=0.8)
                        ax.plot([w3_val - rel_disp[0], w3_val - rel_disp[-1]], [w1_val + rel_disp[0], w1_val + rel_disp[-1]], "r--", linewidth=1.5, alpha=0.8)
                        self.twoDaxes.canvas.draw()

                        fig.tight_layout()
                        plt.show(block=False)
                    except Exception as exc:
                        QMessageBox.warning(self, "Plot failed", str(exc))

        elif choice == "Along fixed pump WN":
            if self._twoD_interactive():
                self.statusBar().showMessage(
                    "Interactive slice: click a pump wavenumber on the plot."
                )

                def done(points):
                    self._picker = None
                    if not points:
                        return
                    pump_wn = points[0]
                    t2_delays = self._ask_values(
                        "Transient spectra slices",
                        "Delays to plot (comma-separated or MATLAB range e.g. 1:0.1:5, 'all'):",
                        [active_delay],
                        all_values=self.twoD_dataset.delays,
                    )
                    if t2_delays:
                        self._plot_pump_slice(pump_wn, t2_delays, contour_probe_lim)

                picker_axis = "y" if is_vertical else "x"
                self._picker = ContourPicker(self.twoDaxes.canvas, ax, picker_axis, done).start()
            else:
                val, ok = QInputDialog.getDouble(
                    self,
                    "Pump wavenumber",
                    "Enter pump wavenumber (cm⁻¹):",
                    float(np.mean(self.twoD_dataset.pump)),
                    float(self.twoD_dataset.pump[0]),
                    float(self.twoD_dataset.pump[-1]),
                    1,
                )
                if ok:
                    t2_delays = self._ask_values(
                        "Transient spectra slices",
                        "Delays to plot (comma-separated or MATLAB range e.g. 1:0.1:5, 'all'):",
                        [active_delay],
                        all_values=self.twoD_dataset.delays,
                    )
                    if t2_delays:
                        self._plot_pump_slice(val, t2_delays, contour_probe_lim)

        elif choice == "Along fixed probe WN":
            if self._twoD_interactive():
                self.statusBar().showMessage(
                    "Interactive slice: click a probe wavenumber on the plot."
                )

                def done(points):
                    self._picker = None
                    if not points:
                        return
                    probe_wn = points[0]
                    t2_delays = self._ask_values(
                        "Transient spectra slices",
                        "Delays to plot (comma-separated or MATLAB range e.g. 1:0.1:5, 'all'):",
                        [active_delay],
                        all_values=self.twoD_dataset.delays,
                    )
                    if t2_delays:
                        self._plot_probe_slice(probe_wn, t2_delays, contour_pump_lim)

                picker_axis = "x" if is_vertical else "y"
                self._picker = ContourPicker(self.twoDaxes.canvas, ax, picker_axis, done).start()
            else:
                val, ok = QInputDialog.getDouble(
                    self,
                    "Probe wavenumber",
                    "Enter probe wavenumber (cm⁻¹):",
                    float(np.mean(self.twoD_dataset.probe)),
                    float(self.twoD_dataset.probe[0]),
                    float(self.twoD_dataset.probe[-1]),
                    1,
                )
                if ok:
                    t2_delays = self._ask_values(
                        "Transient spectra slices",
                        "Delays to plot (comma-separated or MATLAB range e.g. 1:0.1:5, 'all'):",
                        [active_delay],
                        all_values=self.twoD_dataset.delays,
                    )
                    if t2_delays:
                        self._plot_probe_slice(val, t2_delays, contour_pump_lim)

        elif choice == "Integrate along pump axis":
            if self._twoD_interactive():
                from matplotlib.widgets import RectangleSelector

                self.statusBar().showMessage(
                    "Draw a rectangle to specify the pump integration range. Press ENTER to confirm."
                )
                self._rect_cid = None

                def onselect(eclick, erelease):
                    pass

                def on_key(event):
                    if event.key == "enter":
                        if self._rect_selector is None:
                            return
                        is_visible = getattr(self._rect_selector, "visible", True)
                        if not is_visible:
                            return
                        extents = self._rect_selector.extents
                        xmin, xmax, ymin, ymax = extents
                        if abs(xmax - xmin) < 1e-5 or abs(ymax - ymin) < 1e-5:
                            return

                        pump_range = (ymin, ymax) if is_vertical else (xmin, xmax)

                        if self._rect_cid is not None:
                            self.twoDaxes.canvas.mpl_disconnect(self._rect_cid)
                            self._rect_cid = None
                        self._rect_selector.set_active(False)
                        self._rect_selector = None
                        self.statusBar().clearMessage()

                        t2_delays = self._ask_values(
                            "Transient spectra slices",
                            "Delays to plot (comma-separated or MATLAB range e.g. 1:0.1:5, 'all'):",
                            [active_delay],
                            all_values=self.twoD_dataset.delays,
                        )
                        if t2_delays:
                            self._plot_integrate_pump_slice(
                                pump_range, t2_delays, contour_probe_lim
                            )
                        self._twoD_rerender_preview()

                selector_kwargs = dict(
                    useblit=True, button=[1], minspanx=5, minspany=5, interactive=True
                )
                try:
                    self._rect_selector = RectangleSelector(
                        ax,
                        onselect,
                        props=dict(
                            facecolor="orange", edgecolor="darkorange", alpha=0.3, fill=True
                        ),
                        **selector_kwargs,
                    )
                except TypeError:
                    self._rect_selector = RectangleSelector(
                        ax,
                        onselect,
                        rectprops=dict(
                            facecolor="orange", edgecolor="darkorange", alpha=0.3, fill=True
                        ),
                        **selector_kwargs,
                    )
                self._rect_cid = self.twoDaxes.canvas.mpl_connect("key_press_event", on_key)
                self.twoDaxes.canvas.setFocus()
            else:
                text, ok = QInputDialog.getText(
                    self, "Pump Integration range", "Enter pump range as min:max (e.g. 1640:1660):"
                )
                if ok and ":" in text:
                    try:
                        p_min, p_max = map(float, text.split(":"))
                        t2_delays = self._ask_values(
                            "Transient spectra slices",
                            "Delays to plot (comma-separated or MATLAB range e.g. 1:0.1:5, 'all'):",
                            [active_delay],
                            all_values=self.twoD_dataset.delays,
                        )
                        if t2_delays:
                            self._plot_integrate_pump_slice(
                                (p_min, p_max), t2_delays, contour_probe_lim
                            )
                    except ValueError:
                        logger.debug(
                            "Ignoring unparseable pump integration range %r.", text, exc_info=True
                        )

        elif choice == "Integrate along probe axis":
            if self._twoD_interactive():
                from matplotlib.widgets import RectangleSelector

                self.statusBar().showMessage(
                    "Draw a rectangle to specify the probe integration range. Press ENTER to confirm."
                )
                self._rect_cid = None

                def onselect(eclick, erelease):
                    pass

                def on_key(event):
                    if event.key == "enter":
                        if self._rect_selector is None:
                            return
                        is_visible = getattr(self._rect_selector, "visible", True)
                        if not is_visible:
                            return
                        extents = self._rect_selector.extents
                        xmin, xmax, ymin, ymax = extents
                        if abs(xmax - xmin) < 1e-5 or abs(ymax - ymin) < 1e-5:
                            return

                        probe_range = (xmin, xmax) if is_vertical else (ymin, ymax)

                        if self._rect_cid is not None:
                            self.twoDaxes.canvas.mpl_disconnect(self._rect_cid)
                            self._rect_cid = None
                        self._rect_selector.set_active(False)
                        self._rect_selector = None
                        self.statusBar().clearMessage()

                        t2_delays = self._ask_values(
                            "Transient spectra slices",
                            "Delays to plot (comma-separated or MATLAB range e.g. 1:0.1:5, 'all'):",
                            [active_delay],
                            all_values=self.twoD_dataset.delays,
                        )
                        if t2_delays:
                            self._plot_integrate_probe_slice(
                                probe_range, t2_delays, contour_pump_lim
                            )
                        self._twoD_rerender_preview()

                selector_kwargs = dict(
                    useblit=True, button=[1], minspanx=5, minspany=5, interactive=True
                )
                try:
                    self._rect_selector = RectangleSelector(
                        ax,
                        onselect,
                        props=dict(
                            facecolor="orange", edgecolor="darkorange", alpha=0.3, fill=True
                        ),
                        **selector_kwargs,
                    )
                except TypeError:
                    self._rect_selector = RectangleSelector(
                        ax,
                        onselect,
                        rectprops=dict(
                            facecolor="orange", edgecolor="darkorange", alpha=0.3, fill=True
                        ),
                        **selector_kwargs,
                    )
                self._rect_cid = self.twoDaxes.canvas.mpl_connect("key_press_event", on_key)
                self.twoDaxes.canvas.setFocus()
            else:
                text, ok = QInputDialog.getText(
                    self,
                    "Probe Integration range",
                    "Enter probe range as min:max (e.g. 1640:1660):",
                )
                if ok and ":" in text:
                    try:
                        pr_min, pr_max = map(float, text.split(":"))
                        t2_delays = self._ask_values(
                            "Transient spectra slices",
                            "Delays to plot (comma-separated or MATLAB range e.g. 1:0.1:5, 'all'):",
                            [active_delay],
                            all_values=self.twoD_dataset.delays,
                        )
                        if t2_delays:
                            self._plot_integrate_probe_slice(
                                (pr_min, pr_max), t2_delays, contour_pump_lim
                            )
                    except ValueError:
                        logger.debug(
                            "Ignoring unparseable probe integration range %r.", text, exc_info=True
                        )

        elif choice == "Along several pump WN":
            if self._twoD_interactive():
                self.statusBar().showMessage(
                    "Interactive slice: left-click to pick pump wavenumbers, right-click to undo, Enter to plot."
                )

                def done(points):
                    self._picker = None
                    if not points:
                        return
                    self._plot_several_pump_slices(points, active_delay, contour_probe_lim)

                picker_axis = "y" if is_vertical else "x"
                self._picker = ContourPicker(self.twoDaxes.canvas, ax, picker_axis, done).start()
            else:
                vals = self._ask_values(
                    "Pump wavenumbers",
                    "Enter pump wavenumbers (comma-separated):",
                    [float(np.mean(self.twoD_dataset.pump))],
                )
                if vals:
                    self._plot_several_pump_slices(vals, active_delay, contour_probe_lim)

        elif choice == "Along several probe WN":
            if self._twoD_interactive():
                self.statusBar().showMessage(
                    "Interactive slice: left-click to pick probe wavenumbers, right-click to undo, Enter to plot."
                )

                def done(points):
                    self._picker = None
                    if not points:
                        return
                    self._plot_several_probe_slices(points, active_delay, contour_pump_lim)

                picker_axis = "x" if is_vertical else "y"
                self._picker = ContourPicker(self.twoDaxes.canvas, ax, picker_axis, done).start()
            else:
                vals = self._ask_values(
                    "Probe wavenumbers",
                    "Enter probe wavenumbers (comma-separated):",
                    [float(np.mean(self.twoD_dataset.probe))],
                )
                if vals:
                    self._plot_several_probe_slices(vals, active_delay, contour_pump_lim)

    def _get_twoD_axis_labels(self):

        s = pm.get_settings()
        flabel = s.freq_label.value if hasattr(s.freq_label, "value") else s.freq_label
        lstyle = s.label_style.value if hasattr(s.label_style, "value") else s.label_style

        flabel_delim = lstyle if lstyle in ("()", "[]", "/") else "()"
        units_dict = (
            self.twoD_dataset.units if self.twoD_dataset is not None else {"unitsL": "cm-1"}
        )
        unitsL = units_dict.get("unitsL", "cm-1")

        pumpAll, probeAll = hlp.fmt2Dlabel(flabel_delim, flabel, unitsL)

        match flabel:
            case "Omega_n":
                pump_symbol, probe_symbol = r"$\omega_{1}$", r"$\omega_{3}$"
            case "Omega_PP":
                pump_symbol, probe_symbol = r"$\omega_{\text{pump}}$", r"$\omega_{\text{probe}}$"
            case "Pump-Probe":
                pump_symbol, probe_symbol = "Pump", "Probe"
            case "Omega_n/2pic":
                pump_symbol, probe_symbol = r"$\omega_{1}/2{\pi}c_{0}$", r"$\omega_{3}/2{\pi}c_{0}$"
            case _:
                pump_symbol, probe_symbol = r"$\omega_{1}$", r"$\omega_{3}$"

        return pumpAll, probeAll, pump_symbol, probe_symbol

    @busy_guard("Plotting pump slice...")
    def _plot_pump_slice(self, pump_wn, t2_delays, xlim):
        import matplotlib.pyplot as plt

        from ...oneD.plot import plot_spectra

        pm.apply_style()
        slice_ds = self.twoD_dataset.get_slice_at_pump(pump_wn)
        fig, ax = plt.subplots(figsize=(7.5, 4.375))
        plot_spectra(slice_ds, t2_delays, ax=ax, fig=fig, normY=self._twoD_norm())

        pumpAll, probeAll, pump_symbol, probe_symbol = self._get_twoD_axis_labels()
        ax.set_xlabel(probeAll)
        ax.set_title(f"Slice at {pump_symbol} = {pump_wn:.1f} cm$^{{-1}}$")
        ax.set_xlim(xlim)
        fig.tight_layout()
        plt.show(block=False)

    @busy_guard("Plotting probe slice...")
    def _plot_probe_slice(self, probe_wn, t2_delays, xlim):
        import matplotlib.pyplot as plt

        from ...oneD.plot import plot_spectra

        pm.apply_style()
        slice_ds = self.twoD_dataset.get_slice_at_probe(probe_wn)
        fig, ax = plt.subplots(figsize=(7.5, 4.375))
        plot_spectra(slice_ds, t2_delays, ax=ax, fig=fig, normY=self._twoD_norm())

        pumpAll, probeAll, pump_symbol, probe_symbol = self._get_twoD_axis_labels()
        ax.set_xlabel(pumpAll)
        ax.set_title(f"Slice at {probe_symbol} = {probe_wn:.1f} cm$^{{-1}}$")
        ax.set_xlim(xlim)
        fig.tight_layout()
        plt.show(block=False)

    @busy_guard("Integrating along the pump axis...")
    def _plot_integrate_pump_slice(self, pump_range, t2_delays, xlim):
        import matplotlib.pyplot as plt

        from ...oneD.plot import plot_spectra

        pm.apply_style()
        slice_ds = self.twoD_dataset.get_slice_integrate_pump(pump_range[0], pump_range[1])
        fig, ax = plt.subplots(figsize=(7.5, 4.375))
        plot_spectra(slice_ds, t2_delays, ax=ax, fig=fig, normY=self._twoD_norm())

        pumpAll, probeAll, pump_symbol, probe_symbol = self._get_twoD_axis_labels()
        ax.set_xlabel(probeAll)
        ax.set_title(
            f"Integrated {pump_symbol} [{pump_range[0]:.1f}:{pump_range[1]:.1f}] cm$^{{-1}}$"
        )
        ax.set_xlim(xlim)
        fig.tight_layout()
        plt.show(block=False)

    @busy_guard("Integrating along the probe axis...")
    def _plot_integrate_probe_slice(self, probe_range, t2_delays, xlim):
        import matplotlib.pyplot as plt

        from ...oneD.plot import plot_spectra

        pm.apply_style()
        slice_ds = self.twoD_dataset.get_slice_integrate_probe(probe_range[0], probe_range[1])
        fig, ax = plt.subplots(figsize=(7.5, 4.375))
        plot_spectra(slice_ds, t2_delays, ax=ax, fig=fig, normY=self._twoD_norm())

        pumpAll, probeAll, pump_symbol, probe_symbol = self._get_twoD_axis_labels()
        ax.set_xlabel(pumpAll)
        ax.set_title(
            f"Integrated {probe_symbol} [{probe_range[0]:.1f}:{probe_range[1]:.1f}] cm$^{{-1}}$"
        )
        ax.set_xlim(xlim)
        fig.tight_layout()
        plt.show(block=False)

    @busy_guard("Plotting pump slices...")
    def _plot_several_pump_slices(self, pump_wns, t2, xlim):
        import matplotlib.pyplot as plt

        from ... import helpers as hlp
        from ...oneD.plot import plot_spectra

        pm.apply_style()
        s = pm.get_settings()
        fig, ax = plt.subplots(figsize=s.spectra_figsize)
        Nplots = len(pump_wns)
        cm = hlp.get_trace_cmap(s.traces_cmap, Nplots)
        pumpAll, probeAll, pump_symbol, probe_symbol = self._get_twoD_axis_labels()

        lines = []
        labels = []
        for i, p_wn in enumerate(pump_wns):
            slice_ds = self.twoD_dataset.get_slice_at_pump(p_wn)
            n_before = len(ax.get_lines())
            plot_spectra(slice_ds, [t2], ax=ax, fig=fig, normY=self._twoD_norm())
            line = ax.get_lines()[n_before]
            line.set_color(cm[i])
            lbl = f"{p_wn:.1f}"
            line.set_label(lbl)
            lines.append(line)
            labels.append(lbl)

        ax.set_xlabel(probeAll)
        ax.set_title(f"{pump_symbol} slices at $t_2$ = {t2:.2f} ps")
        ax.set_xlim(xlim)

        leg = ax.legend(
            handles=lines,
            labels=labels,
            title=f"{pump_symbol} (cm$^{{-1}}$)",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            handlelength=0.75,
            labelspacing=s.legend_label_spacing if hasattr(s, "legend_label_spacing") else 0.5,
        )
        leg.set_draggable(True)
        fig.tight_layout()
        plt.show(block=False)

    @busy_guard("Plotting probe slices...")
    def _plot_several_probe_slices(self, probe_wns, t2, xlim):
        import matplotlib.pyplot as plt

        from ... import helpers as hlp
        from ...oneD.plot import plot_spectra

        pm.apply_style()
        s = pm.get_settings()
        fig, ax = plt.subplots(figsize=s.spectra_figsize)
        Nplots = len(probe_wns)
        cm = hlp.get_trace_cmap(s.traces_cmap, Nplots)
        pumpAll, probeAll, pump_symbol, probe_symbol = self._get_twoD_axis_labels()

        lines = []
        labels = []
        for i, pr_wn in enumerate(probe_wns):
            slice_ds = self.twoD_dataset.get_slice_at_probe(pr_wn)
            n_before = len(ax.get_lines())
            plot_spectra(slice_ds, [t2], ax=ax, fig=fig, normY=self._twoD_norm())
            line = ax.get_lines()[n_before]
            line.set_color(cm[i])
            lbl = f"{pr_wn:.1f}"
            line.set_label(lbl)
            lines.append(line)
            labels.append(lbl)

        ax.set_xlabel(pumpAll)
        ax.set_title(f"{probe_symbol} slices at $t_2$ = {t2:.2f} ps")
        ax.set_xlim(xlim)

        leg = ax.legend(
            handles=lines,
            labels=labels,
            title=f"{probe_symbol} (cm$^{{-1}}$)",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            handlelength=0.75,
            labelspacing=s.legend_label_spacing if hasattr(s, "legend_label_spacing") else 0.5,
        )
        leg.set_draggable(True)
        fig.tight_layout()
        plt.show(block=False)

    def _update_spectral_diffusion_controls_state(self):
        """Enable or disable spectral diffusion overlays and kinetics widgets based on whether analysis results are available."""
        # Enable individual checkboxes if their specific result lists are present in sd_results
        sd_results = (
            getattr(self.twoD_dataset, "_sd_results", {}) if self.twoD_dataset is not None else {}
        )

        cls_avail = "CLS" in sd_results
        ivcls_avail = "IvCLS" in sd_results
        nls_avail = "NLS" in sd_results
        has_any = len(sd_results) > 0

        for name, avail in [
            ("twoD_show_cls_chk", cls_avail),
            ("twoD_show_ivcls_chk", ivcls_avail),
            ("twoD_show_nls_chk", nls_avail),
        ]:
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(avail)
                if not avail:
                    widget.blockSignals(True)
                    widget.setChecked(False)
                    widget.blockSignals(False)
                else:
                    # Automatically check them if they become available and aren't already checked
                    if not widget.isChecked():
                        widget.blockSignals(True)
                        widget.setChecked(True)
                        widget.blockSignals(False)

        widget_btn = getattr(self, "twoD_plot_sd_kinetics_btn", None)
        if widget_btn is not None:
            widget_btn.setEnabled(has_any)

        # Update Kubo Group box state
        has_dataset = self.twoD_dataset is not None
        has_kubo = False
        if has_dataset:
            sd_results = getattr(self.twoD_dataset, "_sd_results", {})
            has_kubo = ("Kubo_Fit" in sd_results) or (
                getattr(self.twoD_dataset, "_kubo_fit_dataset", None) is not None
            )

        if hasattr(self, "twoD_kubo_fit_grp") and self.twoD_kubo_fit_grp is not None:
            self.twoD_kubo_fit_grp.setEnabled(has_dataset)
            if hasattr(self, "twoD_kubo_mode_cb"):
                self.twoD_kubo_mode_cb.setEnabled(has_kubo)

        btn_sub = getattr(self, "twoD_subtract_spectra_btn", None)
        if btn_sub is not None:
            btn_sub.setEnabled(has_dataset)

        btn_gauss = getattr(self, "twoD_gaussian_fit_btn", None)
        if btn_gauss is not None:
            btn_gauss.setEnabled(has_dataset)

        btn_int = getattr(self, "twoD_integral_dynamics_btn", None)
        if btn_int is not None:
            btn_int.setEnabled(has_dataset)

    def _init_spectral_diffusion_controls(self):
        """Connect the static spectral diffusion widgets loaded from the UI file."""
        if hasattr(self, "twoD_show_cls_chk") and self.twoD_show_cls_chk is not None:
            self.twoD_show_cls_chk.toggled.connect(self._twoD_rerender_preview)
        if hasattr(self, "twoD_show_ivcls_chk") and self.twoD_show_ivcls_chk is not None:
            self.twoD_show_ivcls_chk.toggled.connect(self._twoD_rerender_preview)
        if hasattr(self, "twoD_show_nls_chk") and self.twoD_show_nls_chk is not None:
            self.twoD_show_nls_chk.toggled.connect(self._twoD_rerender_preview)
        if (
            hasattr(self, "twoD_plot_sd_kinetics_btn")
            and self.twoD_plot_sd_kinetics_btn is not None
        ):
            self.twoD_plot_sd_kinetics_btn.clicked.connect(self._on_twoD_plot_sd_kinetics_clicked)

        if hasattr(self, "twoD_subtract_spectra_btn") and self.twoD_subtract_spectra_btn is not None:
            self.twoD_subtract_spectra_btn.clicked.connect(self._open_2d_subtraction_dialog)

        if hasattr(self, "twoD_gaussian_fit_btn") and self.twoD_gaussian_fit_btn is not None:
            self.twoD_gaussian_fit_btn.clicked.connect(self._open_2d_gaussian_fit_dialog)

        if hasattr(self, "twoD_integral_dynamics_btn") and self.twoD_integral_dynamics_btn is not None:
            self.twoD_integral_dynamics_btn.clicked.connect(self._open_2d_integral_dynamics_dialog)

        # Create and add the Kubo group box programmatically to the layout
        if hasattr(self, "twoD_resultstr2d_layout") and self.twoD_resultstr2d_layout is not None:
            from PyQt6.QtWidgets import (
                QComboBox,
                QGroupBox,
                QHBoxLayout,
                QLabel,
                QPushButton,
                QVBoxLayout,
            )

            self.twoD_kubo_fit_grp = QGroupBox("Show Kubo model fit results")
            self.twoD_kubo_fit_grp.setEnabled(False)
            kubo_layout = QVBoxLayout(self.twoD_kubo_fit_grp)

            sel_layout = QHBoxLayout()
            sel_layout.addWidget(QLabel("Display Mode:"))
            self.twoD_kubo_mode_cb = QComboBox()
            self.twoD_kubo_mode_cb.addItems(["EXP", "FIT", "BOTH", "RES"])
            self.twoD_kubo_mode_cb.currentTextChanged.connect(self._on_kubo_mode_changed)
            sel_layout.addWidget(self.twoD_kubo_mode_cb)
            kubo_layout.addLayout(sel_layout)

            self.btn_load_kubo_fit = QPushButton("Load Kubo fit...")
            self.btn_load_kubo_fit.clicked.connect(self._load_kubo_fit_dataset)
            self.btn_load_kubo_fit.setStyleSheet("font-weight: bold;")
            kubo_layout.addWidget(self.btn_load_kubo_fit)

            self.twoD_resultstr2d_layout.insertWidget(2, self.twoD_kubo_fit_grp)

        self._update_spectral_diffusion_controls_state()

    def _open_2d_subtraction_dialog(self):
        """Open the interactive 2D spectrum subtraction dialog."""
        if self.twoD_dataset is None:
            QMessageBox.warning(self, "No Dataset", "Please load a 2D dataset first.")
            return

        from ..twoD_subtraction_dialog import TwoDSubtractionDialog

        dlg = TwoDSubtractionDialog(self, self.twoD_dataset)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._twoD_rerender_preview()

    def _open_2d_gaussian_fit_dialog(self):
        """Open the interactive 2D Gaussian fitting dialog."""
        if self.twoD_dataset is None:
            QMessageBox.warning(self, "No Dataset", "Please load a 2D dataset first.")
            return

        from ..twoD_gaussian_dialog import TwoDGaussianFitDialog

        dlg = TwoDGaussianFitDialog(self, self.twoD_dataset)
        dlg.exec()

    def _open_2d_integral_dynamics_dialog(self):
        """Open the interactive 2D ROI integral dynamics dialog."""
        if self.twoD_dataset is None:
            QMessageBox.warning(self, "No Dataset", "Please load a 2D dataset first.")
            return

        from ..twoD_integral_dialog import TwoDIntegralDynamicsDialog

        dlg = TwoDIntegralDynamicsDialog(self, self.twoD_dataset)
        dlg.exec()

    def _on_kubo_mode_changed(self, mode):
        """Update the Z scale slider/spinner to 10% when switching to residuals (RES) mode."""
        if mode == "RES" and self.twoD_plot_controls is not None:
            if (
                hasattr(self.twoD_plot_controls, "z_slider")
                and self.twoD_plot_controls.z_slider is not None
            ):
                self.twoD_plot_controls.z_slider.setValue(10)
            if (
                hasattr(self.twoD_plot_controls, "z_pct")
                and self.twoD_plot_controls.z_pct is not None
            ):
                self.twoD_plot_controls.z_pct.setValue(10)
        self._twoD_rerender_preview()

    @busy_guard("Loading the Kubo fit dataset...")
    def _load_kubo_fit_dataset(self):
        if self.twoD_dataset is None:
            QMessageBox.information(
                self, "Load Kubo Fit", "Please load an experimental 2D dataset first."
            )
            return

        default_dir = ""
        if self._twoD_current_path:
            default_dir = str(Path(self._twoD_current_path).parent)

        path, _ = QFileDialog.getOpenFileName(
            self, "Load Kubo simulated fit data", default_dir, "P2DAT (*.p2dat);;All files (*)"
        )
        if not path:
            return

        try:
            import pymorgan as pm

            sim_dataset = pm.load_2D(path, data_type="P2DAT")

            if (
                len(sim_dataset.pump) != len(self.twoD_dataset.pump)
                or len(sim_dataset.probe) != len(self.twoD_dataset.probe)
                or len(sim_dataset.delays) != len(self.twoD_dataset.delays)
            ):
                QMessageBox.warning(
                    self,
                    "Dimension Mismatch",
                    f"The loaded simulated dataset dimensions do not match the parent dataset.\n\n"
                    f"Parent: Pump={len(self.twoD_dataset.pump)}, Probe={len(self.twoD_dataset.probe)}, Delays={len(self.twoD_dataset.delays)}\n"
                    f"Simulated: Pump={len(sim_dataset.pump)}, Probe={len(sim_dataset.probe)}, Delays={len(self.twoD_dataset.delays)}",
                )
                return

            self.twoD_dataset._kubo_fit_dataset = sim_dataset
            self._update_spectral_diffusion_controls_state()
            self._twoD_rerender_preview()
            QMessageBox.information(
                self, "Load Complete", f"Kubo model fit dataset loaded successfully from:\n{path}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Load Failed", f"Failed to load Kubo fit dataset: {exc}")

    @busy_guard("Plotting spectral diffusion kinetics...")
    def _on_twoD_plot_sd_kinetics_clicked(self):
        """Plot the spectral diffusion kinetics (CLS, IvCLS, NLS) with exponential fits and save option."""
        if self.twoD_dataset is None:
            return

        # Get what analyses to plot (based on what's selected, or prompt if nothing is selected)
        active_plots = []
        if self.twoD_show_cls_chk is not None and self.twoD_show_cls_chk.isChecked():
            active_plots.append("CLS")
        if self.twoD_show_ivcls_chk is not None and self.twoD_show_ivcls_chk.isChecked():
            active_plots.append("IvCLS")
        if self.twoD_show_nls_chk is not None and self.twoD_show_nls_chk.isChecked():
            active_plots.append("NLS")

        if not active_plots:
            # If none checked, ask what to plot
            options = ["CLS", "IvCLS", "NLS", "All available"]
            choice, ok = QInputDialog.getItem(
                self,
                "Select Kinetics to Plot",
                "Choose which spectral diffusion kinetics to plot:",
                options,
                0,
                False,
            )
            if not ok:
                return
            if choice == "All available":
                active_plots = ["CLS", "IvCLS", "NLS"]
            else:
                active_plots = [choice]

        # Ask user for the fit model: Single-exponential, Bi-exponential, Both, None
        fit_choices = ["Single-exponential", "Bi-exponential", "Both", "None"]
        fit_choice, ok = QInputDialog.getItem(
            self,
            "Select Fitting Model",
            "Choose the exponential decay fit model:",
            fit_choices,
            0,
            False,
        )
        if not ok:
            return

        t2_max_val = np.max(self.twoD_dataset.delays)
        t2_range_text, ok = QInputDialog.getText(
            self,
            "Select t2 Fit Range",
            f"Enter t2 range to fit as min:max (default: 0:{t2_max_val:.2f}):",
            text=f"0:{t2_max_val:.2f}",
        )
        if not ok:
            return

        t2_min = 0.0
        t2_max = t2_max_val
        if t2_range_text.strip():
            try:
                parts = t2_range_text.split(":")
                t2_min = float(parts[0])
                t2_max = float(parts[1])
            except Exception:
                QMessageBox.warning(
                    self, "Invalid Input", "Invalid t2 range format. Using default range."
                )

        import matplotlib.pyplot as plt
        from scipy.optimize import curve_fit

        # Exponential fit models
        def single_exp(t, A, tau, y0):
            return A * np.exp(-t / tau) + y0

        def bi_exp(t, A1, tau1, A2, tau2, y0):
            return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + y0

        sd_results = getattr(self.twoD_dataset, "_sd_results", {})

        for metric in active_plots:
            # Use pre-computed results from the analysis button — do NOT re-run the analysis here
            if metric not in sd_results:
                QMessageBox.warning(
                    self,
                    "Analysis not run",
                    f"{metric} has not been computed yet. Please run the Spectral Diffusion analysis first.",
                )
                continue

            data_arr = sd_results[metric]

            if metric == "CLS":
                title = "Centre-Line Slope (CLS) Dynamics"
                ylabel = "CLS"
                colour = "blue"
            elif metric == "IvCLS":
                title = "Inverse Centre-Line Slope (IvCLS) Dynamics"
                ylabel = "IvCLS"
                colour = "red"
            else:
                title = "Nodal Line Slope (NLS) Dynamics"
                ylabel = "NLS"
                colour = "green"

            t_all = data_arr[:, 0]
            y_all = data_arr[:, 1]

            # Filter NaNs and apply selected t2 range for fitting
            mask = np.isfinite(y_all) & (t_all >= t2_min) & (t_all <= t2_max)
            t_fit = t_all[mask]
            y_fit = y_all[mask]

            if len(t_fit) < 3:
                QMessageBox.warning(
                    self,
                    "Fitting error",
                    f"Not enough data points to fit/plot kinetics for {metric}.",
                )
                continue

            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            # Use ax.scatter with high visibility options to ensure dots are rendered clearly
            ax.scatter(
                t_fit,
                y_fit,
                color=colour,
                s=40,
                edgecolor="black",
                alpha=0.9,
                label=f"Measured {metric}",
                zorder=4,
            )

            # Disable gridlines and add an axhline at 0
            ax.grid(False)
            ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, zorder=1)

            # Spectral-diffusion decays always use a linear t2 axis: the
            # population times are a short, linearly sampled series and the
            # exponential fits below are read off a linear axis. The global
            # Settings.time_axis_scale (symlog by default, meant for the much
            # wider 1-D delay range) is deliberately not applied here.
            ax.set_xscale("linear")
            ax.set_xlim(0.0, float(np.max(t_all)))

            # Show ylim=[0,1] by default if the diagonal is contained in the spectrum
            # The diagonal is contained if there is overlap between pump and probe axes ranges
            pump_min, pump_max = np.min(self.twoD_dataset.pump), np.max(self.twoD_dataset.pump)
            probe_min, probe_max = np.min(self.twoD_dataset.probe), np.max(self.twoD_dataset.probe)
            overlap = max(pump_min, probe_min) <= min(pump_max, probe_max)
            if overlap:
                ax.set_ylim(0.0, 1.0)

            t_fine = np.linspace(t_fit[0], t_fit[-1], 200)
            param_text = []

            if fit_choice in ("Single-exponential", "Both"):
                # Initial guesses
                y0_guess = y_fit[-1]
                A_guess = y_fit[0] - y0_guess
                tau_guess = (t_fit[-1] - t_fit[0]) / 3.0
                try:
                    popt, _ = curve_fit(single_exp, t_fit, y_fit, p0=[A_guess, tau_guess, y0_guess])
                    ax.plot(
                        t_fine,
                        single_exp(t_fine, *popt),
                        "--",
                        color="black",
                        linewidth=2,
                        label="Single-exp Fit",
                        zorder=3,
                    )
                    param_text.append(
                        f"Single-exponential:\n  A = {popt[0]:.3f}\n  \u03c4 = {popt[1]:.2f} ps\n  y\u2080 = {popt[2]:.3f}"
                    )
                except Exception as exc:
                    param_text.append(f"Single-exponential Fit Failed:\n  {exc}")

            if fit_choice in ("Bi-exponential", "Both"):
                y0_guess = y_fit[-1]
                A_tot = y_fit[0] - y0_guess
                p0_bi = [
                    0.5 * A_tot,
                    (t_fit[-1] - t_fit[0]) / 10.0,
                    0.5 * A_tot,
                    (t_fit[-1] - t_fit[0]) / 2.0,
                    y0_guess,
                ]
                try:
                    popt, _ = curve_fit(bi_exp, t_fit, y_fit, p0=p0_bi)
                    ax.plot(
                        t_fine,
                        bi_exp(t_fine, *popt),
                        "-",
                        color="magenta",
                        linewidth=2,
                        label="Bi-exp Fit",
                        zorder=3,
                    )
                    param_text.append(
                        f"Bi-exponential:\n  A\u2081 = {popt[0]:.3f}, \u03c4\u2081 = {popt[1]:.2f} ps\n  A\u2082 = {popt[2]:.3f}, \u03c4\u2082 = {popt[3]:.2f} ps\n  y\u2080 = {popt[4]:.3f}"
                    )
                except Exception as exc:
                    param_text.append(f"Bi-exponential Fit Failed:\n  {exc}")

            ax.set_title(title)
            ax.set_xlabel("t\u2082 (ps)")
            ax.set_ylabel(ylabel)
            ax.legend(loc="best")

            if param_text:
                props = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.7")
                ax.text(
                    0.95,
                    0.95,
                    "\n\n".join(param_text),
                    transform=ax.transAxes,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=props,
                    fontsize=9,
                )

            fig.tight_layout()
            plt.show(block=False)

            # Prompt user to save CLS/IvCLS/NLS data
            reply = QMessageBox.question(
                self,
                "Save Data",
                f"Would you like to save the {metric} kinetics data to a CSV file?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                file_path, _ = QFileDialog.getSaveFileName(
                    self, f"Save {metric} Data", f"{metric}_kinetics.csv", "CSV Files (*.csv)"
                )
                if file_path:
                    try:
                        with open(file_path, "w") as f:
                            f.write(f"t2 (ps),{metric}_slope\n")
                            for t_v, y_v in zip(t_fit, y_fit, strict=True):
                                f.write(f"{t_v:.3f},{y_v:.6f}\n")
                        QMessageBox.information(self, "Success", f"Data saved to {file_path}")
                    except Exception as exc:
                        QMessageBox.warning(self, "Error", f"Failed to save data: {exc}")

    def _on_twoD_spectral_diffusion_clicked(self):
        """Invoke CLS, IvCLS or NLS analysis using the unified SpectralDiffusionDialog."""
        if self.twoD_dataset is None:
            return

        from ..dialogs import SpectralDiffusionDialog

        dialog = SpectralDiffusionDialog(self, self.twoD_dataset.delays)
        if not dialog.exec():
            return

        choice, t2_range_text = dialog.get_values()

        t2_max_val = np.max(self.twoD_dataset.delays)
        t2_min = 0.0
        t2_max = t2_max_val
        if t2_range_text.strip():
            try:
                parts = t2_range_text.split(":")
                t2_min = float(parts[0])
                t2_max = float(parts[1])
            except Exception:
                QMessageBox.warning(
                    self, "Invalid Input", "Invalid t2 range format. Using default range."
                )
                t2_min = 0.0
                t2_max = t2_max_val

        self._current_t2_range = (t2_min, t2_max)
        self._active_choice = choice
        self._active_analysis_type = "spectral_diffusion"

        self._launch_roi_selector()

    def _open_2d_gaussian_fit_dialog(self):
        """Open the interactive 2D Gaussian fitting dialog after prompting for t2 range and ROI region."""
        if self.twoD_dataset is None:
            QMessageBox.warning(self, "No Dataset", "Please load a 2D dataset first.")
            return

        t2_max_val = float(np.max(self.twoD_dataset.delays))
        t2_range_text, ok_t2 = QInputDialog.getText(
            self,
            "Select t2 Range to Fit",
            f"Enter t2 range to fit as min:max (default: 0:{t2_max_val:.2f}):",
            text=f"0:{t2_max_val:.2f}",
        )
        if not ok_t2:
            return

        t2_min = 0.0
        t2_max = t2_max_val
        if t2_range_text.strip():
            try:
                parts = t2_range_text.split(":")
                t2_min = float(parts[0])
                t2_max = float(parts[1])
            except Exception:
                QMessageBox.warning(
                    self, "Invalid Input", "Invalid t2 range format. Using default range."
                )
                t2_min = 0.0
                t2_max = t2_max_val

        self._current_t2_range = (t2_min, t2_max)
        self._active_analysis_type = "gaussian_fit"

        self._launch_roi_selector()

    def _open_2d_integral_dynamics_dialog(self):
        """Open the interactive 2D ROI integral dynamics dialog."""
        if self.twoD_dataset is None:
            QMessageBox.warning(self, "No Dataset", "Please load a 2D dataset first.")
            return

        from ..twoD_integral_dialog import TwoDIntegralDynamicsDialog

        dlg = TwoDIntegralDynamicsDialog(self, self.twoD_dataset)
        dlg.exec()

    def _on_twoD_timedomain_fit_clicked(self):
        """Invoke the Direct Kubo Model Fitting analysis."""
        if self.twoD_dataset is None:
            return

        t2_max_val = np.max(self.twoD_dataset.delays)
        t2_range_text, ok_t2 = QInputDialog.getText(
            self,
            "Select t2 Range to Fit",
            f"Enter t2 range to fit as min:max (default: 0:{t2_max_val:.2f}):",
            text=f"0:{t2_max_val:.2f}",
        )
        if not ok_t2:
            return

        t2_min = 0.0
        t2_max = t2_max_val
        if t2_range_text.strip():
            try:
                parts = t2_range_text.split(":")
                t2_min = float(parts[0])
                t2_max = float(parts[1])
            except Exception:
                QMessageBox.warning(
                    self, "Invalid Input", "Invalid t2 range format. Using default range."
                )
                t2_min = 0.0
                t2_max = t2_max_val

        self._current_t2_range = (t2_min, t2_max)
        self._active_analysis_type = "timedomain_fit"

        self._launch_roi_selector()

    @busy_guard("Preparing the movie export...")
    def _on_twoD_make_gif_clicked(self):
        """Open MakeMovieDialog to configure and generate a 2D contour movie/GIF animation."""
        if self.twoD_dataset is None:
            QMessageBox.warning(self, "No Dataset", "Please load a 2D dataset first.")
            return

        from ..movie_dialog import MakeMovieDialog

        plot_controls = getattr(self, "twoD_plot_controls", None)
        dlg = MakeMovieDialog(self.twoD_dataset, parent=self, plot_controls=plot_controls)
        dlg.exec()

    def _cancel_roi_selector(self):
        """Tear down a rectangle selector left armed by a previous request."""
        cid = getattr(self, "_rect_cid", None)
        if cid is not None:
            self.twoDaxes.canvas.mpl_disconnect(cid)
        self._rect_cid = None
        selector = getattr(self, "_rect_selector", None)
        if selector is not None:
            try:
                selector.set_active(False)
                selector.set_visible(False)
            except Exception:
                logger.debug("Could not deactivate the previous ROI selector.", exc_info=True)
            self.twoDaxes.canvas.draw_idle()
        self._rect_selector = None

    def _launch_roi_selector(self):
        s = pm.get_settings()
        pump_axis_val = s.pump_axis
        if hasattr(pump_axis_val, "value"):
            pump_axis_val = pump_axis_val.value
        is_vertical = pump_axis_val == "Vertical"

        ax = getattr(self.twoDaxes, "ax", None)
        if ax is None:
            return

        if self._twoD_interactive():
            from matplotlib.widgets import RectangleSelector

            self._cancel_roi_selector()  # a previous request may still be armed
            self.statusBar().showMessage(
                "Draw a rectangle on the contour plot to define the analysis region. Press ENTER to confirm."
            )
            self._rect_cid = None

            def onselect(eclick, erelease):
                pass

            def on_key(event):
                if event.key == "enter":
                    has_drawn_rect = False
                    if self._rect_selector is not None and getattr(self._rect_selector, "visible", True):
                        try:
                            extents = self._rect_selector.extents
                            xmin, xmax, ymin, ymax = extents
                            if abs(xmax - xmin) > 1e-5 and abs(ymax - ymin) > 1e-5:
                                has_drawn_rect = True
                        except Exception:
                            pass

                    if has_drawn_rect:
                        if is_vertical:
                            pump_range = (ymin, ymax)
                            probe_range = (xmin, xmax)
                        else:
                            pump_range = (xmin, xmax)
                            probe_range = (ymin, ymax)
                    else:
                        # Default to current view limits of the 2D plot axis
                        xlim = ax.get_xlim()
                        ylim = ax.get_ylim()
                        x_lo, x_hi = min(xlim), max(xlim)
                        y_lo, y_hi = min(ylim), max(ylim)

                        if is_vertical:
                            pump_range = (y_lo, y_hi)
                            probe_range = (x_lo, x_hi)
                        else:
                            pump_range = (x_lo, x_hi)
                            probe_range = (y_lo, y_hi)

                    if self._rect_cid is not None:
                        self.twoDaxes.canvas.mpl_disconnect(self._rect_cid)
                        self._rect_cid = None
                    if self._rect_selector is not None:
                        self._rect_selector.set_active(False)
                        self._rect_selector = None
                    self.statusBar().clearMessage()

                    self._run_analysis_with_roi(pump_range, probe_range)

            selector_kwargs = dict(
                useblit=True, button=[1], minspanx=5, minspany=5, interactive=True
            )
            try:
                self._rect_selector = RectangleSelector(
                    ax,
                    onselect,
                    props=dict(facecolor="orange", edgecolor="darkorange", alpha=0.3, fill=True),
                    **selector_kwargs,
                )
            except TypeError:
                self._rect_selector = RectangleSelector(
                    ax,
                    onselect,
                    rectprops=dict(
                        facecolor="orange", edgecolor="darkorange", alpha=0.3, fill=True
                    ),
                    **selector_kwargs,
                )
            self._rect_cid = self.twoDaxes.canvas.mpl_connect("key_press_event", on_key)
            # Focus has to be given once the caller has returned: a slot running
            # under ``busy_guard`` has the tab widget disabled, and a disabled
            # parent silently rejects setFocus(), which would leave ENTER dead.
            QTimer.singleShot(0, self.twoDaxes.canvas.setFocus)
        else:
            text, ok = QInputDialog.getText(
                self,
                "Analysis region",
                "Enter analysis region as pump_min:pump_max,probe_min:probe_max (leave empty for entire range):",
            )
            if not ok:
                return
            pump_range = None
            probe_range = None
            if text.strip():
                try:
                    p_part, pr_part = text.split(",")
                    p_min, p_max = map(float, p_part.split(":"))
                    pr_min, pr_max = map(float, pr_part.split(":"))
                    pump_range = (p_min, p_max)
                    probe_range = (pr_min, pr_max)
                except Exception as exc:
                    QMessageBox.warning(self, "Invalid Input", f"Error parsing ranges: {exc}")
                    return
            self._run_analysis_with_roi(pump_range, probe_range)

    @busy_guard("Running the analysis...")
    def _run_analysis_with_roi(self, pump_range, probe_range):
        if getattr(self, "_active_analysis_type", None) == "spectral_diffusion":
            self._run_spectral_diffusion_analysis_execution(
                self._active_choice, pump_range, probe_range
            )
        elif getattr(self, "_active_analysis_type", None) == "timedomain_fit":
            from ..kubo_dialog import KuboDialog

            dialog = KuboDialog(
                self,
                self.twoD_dataset,
                pump_range=pump_range,
                probe_range=probe_range,
                t2_range=self._current_t2_range,
            )
            dialog.exec()
        elif getattr(self, "_active_analysis_type", None) == "gaussian_fit":
            from ..twoD_gaussian_dialog import TwoDGaussianFitDialog
            from pymorgan.twoD.dataset import Dataset2D

            ds = self.twoD_dataset
            t2_range = getattr(self, "_current_t2_range", None)

            p_min, p_max = (min(pump_range), max(pump_range)) if pump_range else (float(np.min(ds.pump)), float(np.max(ds.pump)))
            pr_min, pr_max = (min(probe_range), max(probe_range)) if probe_range else (float(np.min(ds.probe)), float(np.max(ds.probe)))

            p_mask = (ds.pump >= p_min) & (ds.pump <= p_max)
            pr_mask = (ds.probe >= pr_min) & (ds.probe <= pr_max)

            if not np.any(p_mask):
                p_mask = np.ones(len(ds.pump), dtype=bool)
            if not np.any(pr_mask):
                pr_mask = np.ones(len(ds.probe), dtype=bool)

            if t2_range is not None:
                t2_min, t2_max = (min(t2_range), max(t2_range))
                t2_mask = (ds.delays >= t2_min) & (ds.delays <= t2_max)
                if not np.any(t2_mask):
                    t2_mask = np.ones(len(ds.delays), dtype=bool)
            else:
                t2_mask = np.ones(len(ds.delays), dtype=bool)

            cropped_pump = ds.pump[p_mask]
            cropped_probe = ds.probe[pr_mask]
            cropped_delays = ds.delays[t2_mask]
            cropped_Z = ds.Z[np.ix_(p_mask, pr_mask, t2_mask)]

            cropped_ds = Dataset2D(
                delays=cropped_delays,
                pump=cropped_pump,
                probe=cropped_probe,
                Z=cropped_Z,
                units=ds.units,
                freq_units=ds.freq_units,
                datatype=getattr(ds, "datatype", "2D"),
            )

            dialog = TwoDGaussianFitDialog(self, cropped_ds)
            dialog.exec()

    @busy_guard("Running spectral diffusion analysis...")
    def _run_spectral_diffusion_analysis_execution(self, choice, pump_range, probe_range):
        """Execute the spectral diffusion routines with specific ranges and cache robust results."""
        t2_range = getattr(self, "_current_t2_range", None)
        try:
            if not hasattr(self.twoD_dataset, "_sd_results"):
                self.twoD_dataset._sd_results = {}

            if choice == "CLS (Centre-Line Slope)":
                result = self.twoD_dataset.center_line_slope(
                    pump_range=pump_range, probe_range=probe_range, t2_range=t2_range
                )
                self.twoD_dataset._sd_results["CLS"] = result
            elif choice == "IvCLS (Inverse Centre-Line Slope)":
                result = self.twoD_dataset.center_line_slope(
                    inverse=True, pump_range=pump_range, probe_range=probe_range, t2_range=t2_range
                )
                self.twoD_dataset._sd_results["IvCLS"] = result
            elif choice == "CLS+IvCLS (both combined)":
                result = self.twoD_dataset.center_line_slope(
                    both=True, pump_range=pump_range, probe_range=probe_range, t2_range=t2_range
                )
                self.twoD_dataset._sd_results["CLS"] = result[:, [0, 1]]
                self.twoD_dataset._sd_results["IvCLS"] = result[:, [0, 2]]
            elif choice == "NLS (Nodal Line Slope)":
                result = self.twoD_dataset.nodal_line_slope(
                    pump_range=pump_range, probe_range=probe_range, t2_range=t2_range
                )
                self.twoD_dataset._sd_results["NLS"] = result

            self._update_spectral_diffusion_controls_state()
            self._twoD_rerender_preview()
            self.statusBar().showMessage(f"{choice} analysis complete.", 4000)

        except NotImplementedError as exc:
            QMessageBox.information(self, "Not Implemented", str(exc))
        except Exception as exc:
            QMessageBox.warning(self, "Analysis failed", str(exc))

    def _twoD_rerender_preview(self):
        """Re-draw the embedded 2D contour."""
        if self.twoD_dataset is not None:
            self._twoD_preview_contour()

    def _twoD_apply_view_limits(self):
        """Apply the panel X/Y limits to the embedded 2D axis without re-plotting."""
        if (
            self.twoD_plot_controls is not None
            and self.twoD_plot_controls.cut_plot_chk is not None
            and self.twoD_plot_controls.cut_plot_chk.isChecked()
        ):
            self._twoD_rerender_preview()
            return

        ax = getattr(self.twoDaxes, "ax", None)
        if ax is None or self.twoD_dataset is None or self.twoD_plot_controls is None:
            return
        s = pm.get_settings()
        pump_axis_val = s.pump_axis
        if hasattr(pump_axis_val, "value"):
            pump_axis_val = pump_axis_val.value
        is_vertical = pump_axis_val == "Vertical"
        xlim = (
            self.twoD_plot_controls.probe_lim()
            if is_vertical
            else self.twoD_plot_controls.pump_lim()
        )
        ylim = (
            self.twoD_plot_controls.pump_lim()
            if is_vertical
            else self.twoD_plot_controls.probe_lim()
        )
        # The panel holds limits in the dataset's own units; the axes may be
        # drawn in another (Settings.twoD_freq_unit).
        to_display = pm.twoD.plot.display_converter(self.twoD_dataset, settings=s)
        xlim = tuple(to_display(xlim))
        ylim = tuple(to_display(ylim))
        _safe_set_limits(ax, xlim, ylim)
        self.twoDaxes.canvas.draw_idle()

    def _twoD_preview_contour(self):
        """Draw the 2D contour map on the twoDaxes canvas."""
        if self.twoD_dataset is None or self.twoD_plot_controls is None:
            return
        pm.apply_style()

        # Background subtraction (scatter subtraction)
        sub_chk = getattr(self, "twoD_sub_scat_chk", None)
        sub_delay = getattr(self, "twoD_sub_scat_delay", None)
        do_correct = sub_chk.isChecked() if sub_chk is not None else False
        ref_idx = (sub_delay.value() - 1) if (sub_delay is not None and do_correct) else None
        self.twoD_dataset.background_correct(reference=ref_idx, do_correct=do_correct)

        w, ax = self._twoD_fresh_axis()

        kwargs = self.twoD_plot_controls.contour_kwargs()

        s = pm.get_settings()
        pump_axis_val = s.pump_axis
        if hasattr(pump_axis_val, "value"):
            pump_axis_val = pump_axis_val.value
        is_vertical = pump_axis_val == "Vertical"

        top_spectrum = None
        include_ftir = getattr(self, "twoD_include_ftir_chk", None)
        if include_ftir is not None and include_ftir.isChecked() and self._twoD_overlay is not None:
            top_spectrum = self._twoD_overlay

        lst_delay = getattr(self, "twoD_t2delay_lst", None)
        curr_idx = lst_delay.currentIndex().row() if lst_delay is not None else 0
        if curr_idx < 0 or curr_idx >= len(self.twoD_dataset.delays):
            curr_idx = max(0, len(self.twoD_dataset.delays) - 1)
        active_delay = self.twoD_dataset.delays[curr_idx]

        self.twoD_plot_controls.update_delay_index(curr_idx, force_reset_pct=False)

        # Only draw overlays from the existing cache — analysis must be triggered explicitly
        # via the "Spectral Diffusion" analysis button, not from checkbox toggling.
        cls_points = None
        if (
            getattr(self, "twoD_show_cls_chk", None) is not None
            and self.twoD_show_cls_chk.isChecked()
        ):
            cache = getattr(self.twoD_dataset, "_cls_cache", {})
            entry = cache.get(active_delay)
            if entry is not None:
                cls_points = (entry["points"][:, 0], entry["points"][:, 1], entry["fit"])

        ivcls_points = None
        if (
            getattr(self, "twoD_show_ivcls_chk", None) is not None
            and self.twoD_show_ivcls_chk.isChecked()
        ):
            cache = getattr(self.twoD_dataset, "_ivcls_cache", {})
            entry = cache.get(active_delay)
            if entry is not None:
                ivcls_points = (entry["points"][:, 0], entry["points"][:, 1], entry["fit"])

        nls_points = None
        if (
            getattr(self, "twoD_show_nls_chk", None) is not None
            and self.twoD_show_nls_chk.isChecked()
        ):
            cache = getattr(self.twoD_dataset, "_nls_cache", {})
            entry = cache.get(active_delay)
            if entry is not None:
                nls_points = (entry["points"][:, 0], entry["points"][:, 1], entry["fit"])

        zmin, zmax = (
            self.twoD_plot_controls.zlimits()
            if self.twoD_plot_controls is not None
            else (None, None)
        )

        kubo_mode = "EXP"
        if hasattr(self, "twoD_kubo_mode_cb") and self.twoD_kubo_mode_cb is not None:
            kubo_mode = self.twoD_kubo_mode_cb.currentText()

        plot_dataset = self.twoD_dataset

        # Adjust plot dataset or limits based on display mode
        if getattr(self.twoD_dataset, "_kubo_fit_dataset", None) is not None:
            if kubo_mode == "FIT":
                plot_dataset = self.twoD_dataset._kubo_fit_dataset
            elif kubo_mode == "RES":
                from ...twoD.dataset import Dataset2D

                res_Z = self.twoD_dataset.Z - self.twoD_dataset._kubo_fit_dataset.Z
                plot_dataset = Dataset2D(
                    res_Z,
                    self.twoD_dataset.pump,
                    self.twoD_dataset.probe,
                    self.twoD_dataset.delays,
                    self.twoD_dataset.units,
                    self.twoD_dataset.freq_units,
                    source="Residuals",
                    data_type="P2DAT",
                    datatype="processed",
                )
                if zmin is None or zmax is None:
                    # Find maximum of exp slice first
                    idx = self.twoD_dataset.map_index(active_delay)
                    ref_max = np.nanmax(np.abs(self.twoD_dataset.Z[:, :, idx]))
                    # Use a default 10% bounds if not set, else use what's returned by plot controls
                    zmin = -0.1 * ref_max
                    zmax = 0.1 * ref_max

        try:
            pm.twoD.plot_map(
                plot_dataset,
                active_delay,
                ax=ax,
                cmap_ID=kwargs.get("cmap_ID"),
                ShowLines=kwargs.get("ShowLines"),
                Nskip=kwargs.get("Nskip"),
                white_levels=kwargs.get("white_levels"),
                top_spectrum=top_spectrum,
                top_label="Overlay" if top_spectrum is not None else None,
                vmin=zmin,
                vmax=zmax,
                text_white_bg=kwargs.get("text_white_bg"),
                cut_plot=kwargs.get("cut_plot", False),
                pump_lim=kwargs.get("pump_lim"),
                probe_lim=kwargs.get("probe_lim"),
                Nlevels=kwargs.get("Nlevels"),
                filled=kwargs.get("filled"),
                cls_points=cls_points,
                ivcls_points=ivcls_points,
                nls_points=nls_points,
            )

            # Draw FIT as black contour lines on top if BOTH
            if (
                kubo_mode == "BOTH"
                and getattr(self.twoD_dataset, "_kubo_fit_dataset", None) is not None
            ):
                sim_ds = self.twoD_dataset._kubo_fit_dataset
                idx = sim_ds.map_index(active_delay)
                Zmap_sim = sim_ds.Z[:, :, idx]

                # Check for cut_plot limits
                pump_full = np.asarray(sim_ds.pump, dtype=float)
                probe_full = np.asarray(sim_ds.probe, dtype=float)

                if (
                    kwargs.get("cut_plot", False)
                    and kwargs.get("pump_lim") is not None
                    and kwargs.get("probe_lim") is not None
                ):
                    pump_lim_val = kwargs.get("pump_lim")
                    probe_lim_val = kwargs.get("probe_lim")
                    p_min_v, p_max_v = min(pump_lim_val), max(pump_lim_val)
                    pr_min_v, pr_max_v = min(probe_lim_val), max(probe_lim_val)

                    pump_indices = np.nonzero((pump_full >= p_min_v) & (pump_full <= p_max_v))[0]
                    probe_indices = np.nonzero((probe_full >= pr_min_v) & (probe_full <= pr_max_v))[
                        0
                    ]

                    if len(pump_indices) > 0 and len(probe_indices) > 0:
                        pump_full = pump_full[pump_indices]
                        probe_full = probe_full[probe_indices]
                        Zmap_sim = Zmap_sim[np.ix_(pump_indices, probe_indices)]

                if is_vertical:
                    X_sim = probe_full
                    Y_sim = pump_full
                    Z_contour = Zmap_sim
                else:
                    X_sim = pump_full
                    Y_sim = probe_full
                    Z_contour = Zmap_sim.T

                sim_max = np.nanmax(np.abs(Zmap_sim))
                if sim_max > 0:
                    levels = np.linspace(-sim_max, sim_max, kwargs.get("Nlevels") or 30)
                    ax.contour(X_sim, Y_sim, Z_contour, levels=levels, colors="k", linewidths=0.5)

            # Draw indicator line if in "Other plots" subtab and TD mode is active
            subtabs = getattr(self, "twoD_subtabs", None)
            other_tab = getattr(self, "twoD_otherplots_tab", None)
            in_other_tab = (
                subtabs is not None and other_tab is not None and subtabs.currentWidget() == other_tab
            )

            if in_other_tab and hasattr(self, "twoD_other_btn_TD") and self.twoD_other_btn_TD.isChecked():
                pixel_idx = self.twoD_other_pixel_spin.value()
                if pixel_idx >= 1 and pixel_idx - 1 < len(self.twoD_dataset.probe):
                    probe_val = self.twoD_dataset.probe[pixel_idx - 1]
                    if is_vertical:
                        ax.axvline(
                            probe_val, color="black", linestyle="--", linewidth=1.5, alpha=0.8
                        )
                    else:
                        ax.axhline(
                            probe_val, color="black", linestyle="--", linewidth=1.5, alpha=0.8
                        )
        except Exception as exc:
            import traceback

            traceback.print_exc()
            self.statusBar().showMessage(f"Plot failed: {exc}", 5000)

        xlim = (
            self.twoD_plot_controls.probe_lim()
            if is_vertical
            else self.twoD_plot_controls.pump_lim()
        )
        ylim = (
            self.twoD_plot_controls.pump_lim()
            if is_vertical
            else self.twoD_plot_controls.probe_lim()
        )
        _safe_set_limits(ax, xlim, ylim)

        sq_chk = getattr(self, "twoD_PC_square", None)
        if sq_chk is not None and sq_chk.isChecked():
            ax.set_aspect("equal")
        else:
            ax.set_aspect("auto")

        self._style_preview(w)
        w.canvas.draw_idle()

        # Update the other plot if the 'Other Plots' tab is active
        subtabs = getattr(self, "twoD_subtabs", None)
        if subtabs is not None and subtabs.widget(subtabs.currentIndex()) == getattr(
            self, "twoD_otherplots_tab", None
        ):
            if hasattr(self, "_twoD_update_other_plot"):
                self._twoD_update_other_plot()

    def _twoD_fresh_axis(self):
        w = self.twoDaxes
        w.figure.clear()
        w.ax = w.figure.add_subplot(111)
        if getattr(self, "_2d_scroll_cid", None) is None and hasattr(w, "canvas"):
            self._2d_scroll_cid = w.canvas.mpl_connect("scroll_event", self._on_twoD_contour_scroll)
        return w, w.ax

    def _on_twoD_contour_scroll(self, event):
        """Scroll mouse wheel over the 2D contour plot to adjust Z scale %."""
        if event.inaxes is None or not hasattr(self, "twoD_plot_controls") or self.twoD_plot_controls is None:
            return
        step = getattr(event, "step", 0)
        if step == 0:
            step = 1.0 if getattr(event, "button", None) == "up" else (-1.0 if getattr(event, "button", None) == "down" else 0)
        if step == 0:
            return
        pc = self.twoD_plot_controls
        if pc.z_pct is None:
            return
        curr_pct = float(pc.z_pct.value())
        eps = float(np.finfo(float).eps)
        factor = 1.1 ** step
        new_pct = max(eps, min(100.0, curr_pct * factor))
        pc.z_pct.setValue(new_pct)

    def _init_twoD_other_plot(self):
        """Programmatically initialize the controls and plot widget in the 'Other Plots' tab."""
        other_layout = getattr(self, "twoD_otherplots_layout", None)
        if other_layout is None:
            return

        # Hide the default label from .ui
        lbl = getattr(self, "twoD_otherplots_lbl", None)
        if lbl is not None:
            lbl.hide()

        # Adjust the layout margins and spacing for a tight fit
        other_layout.setContentsMargins(4, 4, 4, 4)
        other_layout.setSpacing(4)

        # Create control layout at the top
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(4)

        from PyQt6.QtWidgets import QButtonGroup

        self.twoD_other_btn_group = QButtonGroup(self)

        # TD Button
        self.twoD_other_btn_TD = QPushButton("TD", self)
        self.twoD_other_btn_TD.setCheckable(True)
        self.twoD_other_btn_TD.setFixedWidth(45)

        # PH Button
        self.twoD_other_btn_PH = QPushButton("PH", self)
        self.twoD_other_btn_PH.setCheckable(True)
        self.twoD_other_btn_PH.setFixedWidth(45)

        # CAL Button
        self.twoD_other_btn_CAL = QPushButton("CAL", self)
        self.twoD_other_btn_CAL.setCheckable(True)
        self.twoD_other_btn_CAL.setFixedWidth(45)

        self.twoD_other_btn_group.addButton(self.twoD_other_btn_TD)
        self.twoD_other_btn_group.addButton(self.twoD_other_btn_PH)
        self.twoD_other_btn_group.addButton(self.twoD_other_btn_CAL)

        # CAL checked by default
        self.twoD_other_btn_CAL.setChecked(True)

        controls_layout.addWidget(self.twoD_other_btn_TD)
        controls_layout.addWidget(self.twoD_other_btn_PH)
        controls_layout.addWidget(self.twoD_other_btn_CAL)

        # Pixel spinner (initially disabled/hidden because CAL is default)
        self.twoD_other_pixel_spin = QSpinBox(self)
        self.twoD_other_pixel_spin.setPrefix("Px: ")
        self.twoD_other_pixel_spin.setRange(0, 0)
        self.twoD_other_pixel_spin.setFixedWidth(70)
        self.twoD_other_pixel_spin.setEnabled(False)
        self.twoD_other_pixel_spin.setVisible(False)
        self.twoD_other_pixel_spin.setToolTip(
            "0 = interferometer; pixels 1-N correspond to probe pixels (1-based)."
        )

        controls_layout.addWidget(self.twoD_other_pixel_spin)

        # Pop Button to open in an isolated window
        self.twoD_other_btn_pop = QPushButton("Pop", self)
        self.twoD_other_btn_pop.setFixedWidth(45)

        controls_layout.addWidget(self.twoD_other_btn_pop)
        controls_layout.addStretch()

        other_layout.addLayout(controls_layout)

        # Matplotlib canvas widget
        from ..widgetplot import WidgetPlot

        self.twoD_other_plot = WidgetPlot(self, show_toolbar=False)
        self.twoD_other_plot.setMaximumHeight(250)
        other_layout.addWidget(self.twoD_other_plot)

        # Wire control events
        self.twoD_other_btn_TD.toggled.connect(self._on_twoD_other_mode_changed)
        self.twoD_other_btn_PH.toggled.connect(self._on_twoD_other_mode_changed)
        self.twoD_other_btn_CAL.toggled.connect(self._on_twoD_other_mode_changed)
        self.twoD_other_pixel_spin.valueChanged.connect(self._twoD_update_other_plot)
        self.twoD_other_pixel_spin.valueChanged.connect(self._twoD_preview_contour)
        self.twoD_other_btn_pop.clicked.connect(self._pop_twoD_other_plot)

        if hasattr(self, "_apply_button_aesthetics"):
            self._apply_button_aesthetics()

    def _on_twoD_other_mode_changed(self):
        """Update pixel spinner visibility/enablement when modes change."""
        is_td = self.twoD_other_btn_TD.isChecked()
        self.twoD_other_pixel_spin.setEnabled(is_td)
        self.twoD_other_pixel_spin.setVisible(is_td)
        self._twoD_update_other_plot()
        self._twoD_preview_contour()

    def _on_twoD_subtabs_changed(self, index):
        """Trigger update when the user switches to the 'Other Plots' tab."""
        subtabs = getattr(self, "twoD_subtabs", None)
        if subtabs is not None and subtabs.widget(index) == getattr(
            self, "twoD_otherplots_tab", None
        ):
            self._twoD_update_other_plot()

    def _get_cal_tab_probe_curve(self, n_pixels: int | None = None) -> np.ndarray | None:
        """Fetch calibration probe curve from CAL tab fit result or CalibratedProbe.csv."""
        # 1. Try result from CAL tab fit
        res = getattr(self, "_cal_fit_result_det1", None) or getattr(self, "_cal_shaper_fit_res", None)
        if res is not None and getattr(res, "wavenumber_cm1", None) is not None:
            vec = np.asarray(res.wavenumber_cm1, dtype=float)
            if n_pixels is None or len(vec) == n_pixels:
                return vec

        # Try running CAL tab fit if exp data is loaded
        try:
            if getattr(self, "_cal_exp_data", None) is not None:
                self._on_cal_do_fit()
                res = getattr(self, "_cal_fit_result_det1", None)
                if res is not None and getattr(res, "wavenumber_cm1", None) is not None:
                    vec = np.asarray(res.wavenumber_cm1, dtype=float)
                    if n_pixels is None or len(vec) == n_pixels:
                        return vec
        except Exception:
            pass

        # 2. Try loading CalibratedProbe.csv from dataset folder or parent folder
        current_path = getattr(self, "_twoD_current_path", None)
        if current_path:
            folder = Path(current_path)
            if folder.is_file():
                folder = folder.parent
            cands = [
                folder / "CalibratedProbe.csv",
                folder.parent / "CalibratedProbe.csv",
                folder.parent.parent / "CalibratedProbe.csv",
            ]
            for cand in cands:
                if cand.exists():
                    try:
                        vec = np.loadtxt(cand, delimiter=",").ravel().astype(float)
                        if n_pixels is None or len(vec) == n_pixels:
                            return vec
                    except Exception:
                        pass
        return None

    def _on_twoD_auto_cal_toggled(self, checked: bool | None = None):
        """Handle toggling of 'automatically calibrate probe axis' checkbox."""
        if self.twoD_dataset is None or getattr(self.twoD_dataset, "probe", None) is None:
            return

        if checked is None:
            auto_cal = getattr(self, "twoD_auto_cal_chk", None)
            checked = auto_cal.isChecked() if auto_cal is not None else False

        if checked:
            if not hasattr(self, "_twoD_raw_probe") or self._twoD_raw_probe is None:
                self._twoD_raw_probe = self.twoD_dataset.probe.copy()
                self._twoD_raw_cal_level = getattr(self.twoD_dataset, "cal_level", None)
            vec = self._get_cal_tab_probe_curve(len(self.twoD_dataset.probe))
            if vec is not None:
                self.twoD_dataset.probe = vec.copy()
                self.twoD_dataset.cal_level = "autocalibrated"
                self.statusBar().showMessage("Probe axis calibrated using CAL tab calibration curve.")
            else:
                self.twoD_dataset.cal_level = "autocalibrated"
                self.statusBar().showMessage("Probe axis autocalibrated.")
        else:
            if hasattr(self, "_twoD_raw_probe") and self._twoD_raw_probe is not None:
                self.twoD_dataset.probe = self._twoD_raw_probe.copy()
                self.twoD_dataset.cal_level = getattr(self, "_twoD_raw_cal_level", None)

        self._twoD_preview_contour()
        if hasattr(self, "_refresh_current_probe_line"):
            self._refresh_current_probe_line()

    def _twoD_save_probe_cal(self):
        """Save the currently calibrated/processed probe axis to CalibratedProbe.csv."""
        if self.twoD_dataset is None or self.twoD_dataset.source is None:
            return

        from pathlib import Path

        folder_path = Path(self.twoD_dataset.source)
        if folder_path.is_file():
            folder_path = folder_path.parent

        dest_file = folder_path / "CalibratedProbe.csv"
        if dest_file.exists():
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Calibration file exists")
            msg_box.setText(
                "The calibration file 'CalibratedProbe.csv' already exists.\n\nChoose 'Overwrite' to replace it, or 'Keep Existing' to rename the current file as 'CalibratedProbe_old.csv' and save the new one."
            )

            msg_box.addButton("Overwrite", QMessageBox.ButtonRole.DestructiveRole)
            keep_btn = msg_box.addButton(
                "Keep Existing (Rename)", QMessageBox.ButtonRole.ActionRole
            )
            cancel_btn = msg_box.addButton(QMessageBox.StandardButton.Cancel)

            msg_box.exec()

            clicked = msg_box.clickedButton()
            if clicked == cancel_btn:
                return
            elif clicked == keep_btn:
                old_file = folder_path / "CalibratedProbe_old.csv"
                try:
                    if old_file.exists():
                        old_file.unlink()
                    dest_file.rename(old_file)
                except Exception as exc:
                    QMessageBox.critical(
                        self,
                        "Rename failed",
                        f"Could not rename current calibration file to 'CalibratedProbe_old.csv':\n{exc}",
                    )
                    return

        try:
            np.savetxt(dest_file, self.twoD_dataset.probe, fmt="%.6f")
            self.statusBar().showMessage(f"Saved calibrated probe to {dest_file.name}")
            QMessageBox.information(
                self,
                "Calibration Saved",
                f"Successfully saved calibrated probe axis to:\n{dest_file}",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", f"Could not save calibration file:\n{exc}")

    def _plot_twoD_other_axes(self, ax, is_qt=False):
        """Plot TD, PH, or CAL onto the given axes."""
        if self.twoD_dataset is None:
            return

        from ...twoD.process import robust_polyfit

        if self.twoD_other_btn_CAL.isChecked():
            # CAL Plot:
            ref_idx = 0
            _scat_delay = getattr(self, "twoD_sub_scat_delay", None)
            if _scat_delay is not None:
                ref_idx = _scat_delay.value() - 1

            ref_idx = max(0, min(ref_idx, len(self.twoD_dataset.delays) - 1))
            Z_ref = self.twoD_dataset.Z_R[:, :, ref_idx]
            pump = self.twoD_dataset.pump

            has_shaper = self.twoD_dataset.data_type not in ("P2DAT", "RAL_Proc")
            start_idx = 1 if has_shaper else 0

            n_pixels = Z_ref.shape[1]
            pixels = np.arange(start_idx, n_pixels)

            # Find index of maximum absolute value along pump axis (axis 0) for each probe pixel
            max_indices = np.argmax(np.abs(Z_ref[:, start_idx:]), axis=0)
            scattering_maxima = pump[max_indices]

            # Plot raw points
            ax.plot(pixels + 1, scattering_maxima, "xr", label="Data")

            # Robust Parabolic fit
            y_fit = None
            if len(pixels) > 2:
                try:
                    coeffs = robust_polyfit(pixels + 1, scattering_maxima, 2)
                    p = np.poly1d(coeffs)
                    pixels_fit = np.linspace(pixels.min() + 1, pixels.max() + 1, 200)
                    y_fit = p(pixels_fit)
                    ax.plot(
                        pixels_fit,
                        y_fit,
                        "k-",
                        linewidth=1.0 if is_qt else 1.5,
                        label="Fit",
                    )
                except Exception:
                    logger.debug("Could not draw the calibration fit overlay.", exc_info=True)
                    y_fit = None

            # Plot current pixel calibration (dashed, olive green)
            curr_probe = getattr(self.twoD_dataset, "probe", None)
            if curr_probe is not None and len(curr_probe) > 0:
                curr_pixels = np.arange(1, len(curr_probe) + 1)
                ax.plot(
                    curr_pixels,
                    curr_probe,
                    color="olive",
                    linestyle="--",
                    linewidth=1.0 if is_qt else 1.5,
                    label="Current",
                )

            ax.set_xlabel("Pixel number", fontsize=10 if is_qt else 12, fontweight="bold")
            ax.set_ylabel(
                f"Fitted freq. ({self.twoD_dataset.freq_units})",
                fontsize=10 if is_qt else 12,
                fontweight="bold",
                rotation=90,
            )
            if not is_qt:
                ax.set_title("Probe Calibration", fontsize=12, fontweight="bold")
            ax.set_xlim(pixels.min() + 0.5, pixels.max() + 1.5)
            y_vals = [scattering_maxima]
            if y_fit is not None and len(y_fit) > 0:
                y_vals.append(y_fit)
            if curr_probe is not None and len(curr_probe) > 0:
                y_vals.append(curr_probe)
            y_concat = np.concatenate(y_vals)
            if len(y_concat) > 0:
                ax.set_ylim(y_concat.min() - 5, y_concat.max() + 5)
            ax.legend(loc="best", frameon=False, fontsize=8 if is_qt else 10)

        elif self.twoD_other_btn_TD.isChecked():
            # TD Plot (Interferogram vs t1 delay):
            if getattr(self.twoD_dataset, "raw_signal", None) is None:
                ax.text(
                    0.5,
                    0.5,
                    "No raw time-domain data available\n(Processed dataset)",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="gray",
                )
                return

            ref_idx = 0
            lst_delays = getattr(self, "twoD_t2delay_lst", None)
            if lst_delays is not None:
                ref_idx = lst_delays.currentIndex().row()
            ref_idx = max(0, min(ref_idx, len(self.twoD_dataset.delays) - 1))

            t1 = self.twoD_dataset.raw_t1delays
            interferogram = self.twoD_dataset.raw_interferogram[:, ref_idx]

            pixel_idx = 0
            if hasattr(self, "twoD_other_pixel_spin"):
                pixel_idx = self.twoD_other_pixel_spin.value()

            # Map 1-based pixel selection to 0-based Python index
            if pixel_idx >= 1:
                python_pixel_idx = max(
                    0, min(pixel_idx - 1, self.twoD_dataset.raw_signal.shape[1] - 1)
                )
            else:
                python_pixel_idx = 0

            # Fetch apodization function if available
            apod_func = None
            if (
                hasattr(self.twoD_dataset, "proc_apod_func")
                and self.twoD_dataset.proc_apod_func is not None
            ):
                apod_func = self.twoD_dataset.proc_apod_func[:, ref_idx]

            # Add zero line
            ax.axhline(0, lw=0.75, c="0.75")

            if pixel_idx == 0:
                # Plot Interferogram only on left axis
                ax.plot(t1, interferogram, "k-", linewidth=1.0 if is_qt else 1.5, label="Intf.")
                ax.set_xlabel(r"$t_1$ delay (fs)", fontsize=10 if is_qt else 12, fontweight="bold")
                ax.set_ylabel(
                    "Interferogram (a.u.)",
                    fontsize=10 if is_qt else 12,
                    fontweight="bold",
                    color="k",
                    rotation=90,
                )
                ax.tick_params(axis="y", labelcolor="k", labelsize=8 if is_qt else 12)

                # Text caption showing min/max and pulse contrast
                min_val = np.nanmin(interferogram)
                max_val = np.nanmax(interferogram)
                if min_val >= 0:
                    contrast = (
                        (max_val - min_val) / (max_val + min_val)
                        if (max_val + min_val) > 0
                        else 0.0
                    )
                else:
                    contrast = (
                        (max_val - min_val) / (max_val + abs(min_val))
                        if (max_val + abs(min_val)) > 0
                        else 0.0
                    )
                text_str = f"Min/Max: {min_val:.2f}/{max_val:.2f} a.u.\nContrast: {contrast:.2%}"
            else:
                # Plot Signal only on left axis
                signal = self.twoD_dataset.raw_signal[:, python_pixel_idx, ref_idx]
                ax.plot(t1, signal, "r-", linewidth=1.0 if is_qt else 1.5, label="Signal")
                ax.set_xlabel(r"$t_1$ delay (fs)", fontsize=10 if is_qt else 12, fontweight="bold")
                ax.set_ylabel(
                    "Signal (mOD)",
                    fontsize=10 if is_qt else 12,
                    fontweight="bold",
                    color="r",
                    rotation=90,
                )
                ax.tick_params(axis="y", labelcolor="r", labelsize=8 if is_qt else 12)

                # Text caption showing min/max in mOD
                min_val = np.nanmin(signal)
                max_val = np.nanmax(signal)
                text_str = f"Min/Max: {min_val:.2f}/{max_val:.2f} mOD"

            # Add text comment in top-left corner
            ax.text(
                0.05,
                0.95,
                text_str,
                transform=ax.transAxes,
                fontsize=8 if is_qt else 9,
                va="top",
                ha="left",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.2"),
            )

            # Secondary axis on right: Apodisation function all the time
            if apod_func is not None:
                ax2 = ax.twinx()
                ax2.plot(
                    t1[: len(apod_func)],
                    apod_func,
                    color="gray",
                    linestyle="-",
                    linewidth=1.0 if is_qt else 1.5,
                    label="Apod.",
                )
                ax2.set_ylabel(
                    "Apodisation Window",
                    fontsize=10 if is_qt else 12,
                    fontweight="bold",
                    color="gray",
                    rotation=90,
                )
                ax2.tick_params(axis="y", labelcolor="gray", labelsize=8 if is_qt else 12)
                ax2.set_ylim(-0.05, 1.05)

                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(
                    lines1 + lines2,
                    labels1 + labels2,
                    loc="best",
                    frameon=False,
                    fontsize=8 if is_qt else 10,
                )
            else:
                ax.legend(loc="best", frameon=False, fontsize=8 if is_qt else 10)

            if not is_qt:
                ax.set_title("Time Domain Data", fontsize=12, fontweight="bold")

        elif self.twoD_other_btn_PH.isChecked():
            # PH Plot (Fourier Transform & Phasing):
            if getattr(self.twoD_dataset, "raw_signal", None) is None:
                ax.text(
                    0.5,
                    0.5,
                    "No raw time-domain data available\n(Processed dataset)",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="gray",
                )
                return

            ref_idx = 0
            lst_delays = getattr(self, "twoD_t2delay_lst", None)
            if lst_delays is not None:
                ref_idx = lst_delays.currentIndex().row()
            ref_idx = max(0, min(ref_idx, len(self.twoD_dataset.delays) - 1))

            if (
                not hasattr(self.twoD_dataset, "proc_pump_full")
                or self.twoD_dataset.proc_pump_full is None
            ):
                ax.text(
                    0.5,
                    0.5,
                    "No processed FFT data available",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="gray",
                )
                return

            pump_full = self.twoD_dataset.proc_pump_full
            n_ft = len(pump_full)
            n_pos = n_ft // 2

            pump_pos = pump_full[:n_pos]
            abs_fft = self.twoD_dataset.proc_absFFT_ZPint[:n_pos, ref_idx]

            # Add zero line
            ax.axhline(0, lw=0.75, c="0.75")

            # Left axis: Pump spectrum amplitude
            ax.plot(pump_pos, abs_fft, "k-", linewidth=1.0 if is_qt else 1.5, label="Pump")
            ax.set_xlabel(
                f"Pump Frequency ({self.twoD_dataset.freq_units})",
                fontsize=10 if is_qt else 12,
                fontweight="bold",
            )
            ax.set_ylabel(
                "Amplitude (a.u.)",
                fontsize=10 if is_qt else 12,
                fontweight="bold",
                color="k",
                rotation=90,
            )
            ax.tick_params(axis="y", labelcolor="k", labelsize=8 if is_qt else 12)

            # Default x-limits: fit the spectrum of the pump where amplitude >= 5% of max
            max_val = np.nanmax(abs_fft)
            fit_mask = (
                (abs_fft >= 0.05 * max_val) if max_val > 0 else np.ones_like(abs_fft, dtype=bool)
            )
            if np.any(fit_mask):
                pump_min = pump_pos[fit_mask].min()
                pump_max = pump_pos[fit_mask].max()
                pad = 0.05 * (pump_max - pump_min) if pump_max > pump_min else 10.0
                ax.set_xlim(pump_min - pad, pump_max + pad)

            # Fit and plot Gaussian to the pump spectrum
            if np.any(fit_mask):
                try:
                    from scipy.optimize import curve_fit

                    def gaus(x, a, x0, sigma):
                        return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))

                    x_fit = pump_pos[fit_mask]
                    y_fit = abs_fft[fit_mask]
                    p0 = [
                        np.nanmax(y_fit),
                        x_fit[np.nanargmax(y_fit)],
                        (x_fit[-1] - x_fit[0]) / 4.0,
                    ]
                    popt, _ = curve_fit(gaus, x_fit, y_fit, p0=p0, maxfev=2000)

                    x_fine = np.linspace(pump_pos.min(), pump_pos.max(), 500)
                    y_fine = gaus(x_fine, *popt)
                    ax.plot(
                        x_fine, y_fine, "g--", linewidth=1.0 if is_qt else 1.5, label="Gauss Fit"
                    )

                    fwhm = 2.35482 * abs(popt[2])
                    gauss_center = popt[1]
                    ax.text(
                        gauss_center,
                        popt[0] / 2.0,
                        f"FWHM:\n{fwhm:.1f} {self.twoD_dataset.freq_units}",
                        fontsize=8 if is_qt else 9,
                        color="green",
                        ha="center",
                        va="center",
                        bbox=dict(
                            facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round,pad=0.2"
                        ),
                    )
                    # Zoom x-axis to ±2×FWHM
                    ax.set_xlim(gauss_center - 2.0 * fwhm, gauss_center + 2.0 * fwhm)
                except Exception:
                    logger.debug(
                        "Could not annotate/zoom the Gaussian fit overlay.", exc_info=True
                    )

            if self.twoD_dataset.datatype == "interferometer":
                raw_phase = self.twoD_dataset.proc_ZP_phase[:n_pos, ref_idx]
                fitted_phase = self.twoD_dataset.proc_fittedPhase[:n_pos, ref_idx]

                # Re-centre phase: subtract fitted phase value at the spectral peak
                binspecmax = 0
                phase_points = 10
                if hasattr(self.twoD_dataset, "proc_binspecmax"):
                    binspecmax = self.twoD_dataset.proc_binspecmax[ref_idx]
                if hasattr(self.twoD_dataset, "proc_phase_points"):
                    phase_points = self.twoD_dataset.proc_phase_points

                center_freq = pump_pos[binspecmax]

                # Offset: value of fitted phase at the spectral centre
                phase_offset = fitted_phase[binspecmax] if binspecmax < len(fitted_phase) else 0.0
                raw_phase = np.angle(np.exp(1j * (raw_phase - phase_offset)))
                fitted_phase = np.angle(np.exp(1j * (fitted_phase - phase_offset)))

                in_range_indices = np.where(
                    (pump_pos >= center_freq - phase_points)
                    & (pump_pos <= center_freq + phase_points)
                )[0]
                if len(in_range_indices) > 0:
                    fit_start = in_range_indices[0]
                    fit_end = in_range_indices[-1]
                else:
                    fit_start = max(0, binspecmax - 10)
                    fit_end = min(n_pos - 1, binspecmax + 10)

                fit_mask_phase = np.zeros_like(pump_pos, dtype=bool)
                fit_mask_phase[fit_start : fit_end + 1] = True

                # Right axis: Phase
                ax2 = ax.twinx()

                # Plot non-fit points in faded red, fit points in bold red
                ax2.plot(
                    pump_pos[~fit_mask_phase],
                    raw_phase[~fit_mask_phase],
                    ".",
                    color="#ff9999",
                    markersize=3,
                    alpha=0.4,
                    label="Phase",
                )
                ax2.plot(
                    pump_pos[fit_mask_phase],
                    raw_phase[fit_mask_phase],
                    "r.",
                    markersize=5,
                    label="Phase Fit",
                )
                ax2.plot(pump_pos, fitted_phase, "b-", linewidth=1.0 if is_qt else 1.5, label="Fit")
                ax2.axhline(0, lw=0.75, c="0.75")
                ax2.set_ylabel(
                    "Phase (rad)",
                    fontsize=10 if is_qt else 12,
                    fontweight="bold",
                    color="b",
                    rotation=90,
                )
                ax2.tick_params(axis="y", labelcolor="b", labelsize=8 if is_qt else 12)
                ax2.set_ylim(-np.pi, np.pi)

                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()

                # Deduplicate combined legend
                combined_lines = lines1 + lines2
                combined_labels = labels1 + labels2
                seen = set()
                uniq_lines, uniq_labels = [], []
                for l, lbl in zip(combined_lines, combined_labels, strict=True):
                    if lbl not in seen:
                        seen.add(lbl)
                        uniq_lines.append(l)
                        uniq_labels.append(lbl)
                ax.legend(
                    uniq_lines, uniq_labels, loc="best", frameon=False, fontsize=8 if is_qt else 10
                )
            else:
                ax.legend(loc="best", frameon=False, fontsize=8 if is_qt else 10)

            if not is_qt:
                ax.set_title("FT and Phasing", fontsize=12, fontweight="bold")

    def _pop_twoD_other_plot(self):
        """Recreate the active 'Other Plots' figure in a tied Qt window."""
        if self.twoD_dataset is None:
            return

        import os

        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            return

        if self._twoD_pop_window is not None:
            if self._twoD_pop_window.isVisible():
                self._twoD_pop_window.raise_()
                self._twoD_pop_window.activateWindow()
        else:
            from ..dialogs import OtherPlotWindow

            self._twoD_pop_window = OtherPlotWindow(self)
            self._twoD_pop_window.show()

        self._twoD_pop_window.update_plot()

    def _twoD_update_other_plot(self):
        """Redraw the active plot option (TD, PH, or CAL) on twoD_other_plot and tied window."""
        if self.twoD_dataset is None:
            if hasattr(self, "twoD_other_plot"):
                self.twoD_other_plot.reset()
                self.twoD_other_plot.draw()
            if self._twoD_pop_window is not None:
                self._twoD_pop_window.update_plot()
            return

        ax = self.twoD_other_plot.reset()
        self._plot_twoD_other_axes(ax, is_qt=True)
        ax.tick_params(axis="both", which="major", labelsize=8)
        self.twoD_other_plot.draw()

        self._twoD_update_phase_coeffs_display()

        if self._twoD_pop_window is not None:
            self._twoD_pop_window.update_plot()

    def _twoD_update_phase_coeffs_display(self):
        """Update the read-only Phase coeffs. field with the fitted polynomial for the active delay."""
        field = getattr(self, "twoD_phase_coeffs", None)
        if field is None or self.twoD_dataset is None:
            return
        lst_delays = getattr(self, "twoD_t2delay_lst", None)
        ref_idx = lst_delays.currentIndex().row() if lst_delays is not None else 0
        ref_idx = max(0, min(ref_idx, len(self.twoD_dataset.delays) - 1))
        coeffs = None
        if hasattr(self.twoD_dataset, "proc_phase_coeffs"):
            coeffs = self.twoD_dataset.proc_phase_coeffs[ref_idx]
        if coeffs is None:
            field.setText("—")
        else:
            parts = [f"{c:.3g}" for c in coeffs]
            field.setText("  ".join(parts))

    def _twoD_norm(self) -> bool:
        chk = getattr(self, "twoD_normalise_chk", None)
        return bool(chk.isChecked()) if chk is not None else False

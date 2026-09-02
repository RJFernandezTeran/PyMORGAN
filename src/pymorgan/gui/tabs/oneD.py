"""Pump-Probe (1D) tab: dataset loading, state controls, previews and plots.

Part of the :class:`~pymorgan.gui.main_window.MainWindow` implementation, split out
as a mixin so each tab lives in its own module. The methods are unchanged moves:
they run as ``MainWindow`` methods and bind to the widgets declared in
``main_window.ui``, so ``self`` is always the main window.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyqt6")

import contextlib
import re
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import (
    QActionGroup,
)
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QSpinBox,
    QWidget,
)

import pymorgan as pm
from pymorgan.oneD.chirp import (
    default_chirp_filename,
    fit_chirp_automatic,
    fit_chirp_step,
    fit_chirp_wavelet,
    save_chirp_fit,
)
from pymorgan.oneD.load import (
    MESS_ANISOTROPY_MODES,
    describe_dataset,
    mess_calibration_status,
    mess_recalc_average,
    parse_scan_selection,
)
from ..busy import busy_guard
from ..mw_common import (
    _DEFAULT_SPEC_DELAYS,
    _safe_set_limits,
    DATA_TYPE_DISPLAY_NAMES,
    get_combo_datatype,
)


class OneDTabMixin:
    """Pump-Probe (1D) tab: dataset loading, state controls, previews and plots."""

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open data file", "", "Data (*.pdat *.csv *.dat);;All files (*)"
        )
        if path:
            self.load_path(path)

    def reload_data(self):
        if self._current_path:
            dt = getattr(self.dataset, "data_type", None)
            self.load_path(self._current_path, data_type=dt, force=True)
        else:
            self.open_file()

    @busy_guard("Loading dataset...")
    def load_path(
        self,
        path: str,
        data_type: str | None = None,
        *,
        reset_selection: bool = True,
        force: bool = False,
    ):
        """Load a 1-D dataset (also used by tests).

        For directory-based formats with selectable states (e.g. MESS_TRIR) the
        active spectrum / slow-modulation / anisotropy selection is forwarded to
        the loader. ``reset_selection`` restores the defaults (first state, no
        anisotropy) for a freshly chosen dataset; state-control changes pass
        ``False`` to reload the same dataset with the new selection.

        Re-selecting the dataset that is already loaded is a no-op unless
        ``force=True`` (as used by "Reload Dataset"); this avoids re-reading
        from disk when the file list is repopulated, e.g. on tab changes.
        """
        combo = getattr(self, "PP_datatype_cbx", None)
        dt = data_type or get_combo_datatype(combo, "PDAT")
        if (
            not force
            and reset_selection
            and self._current_path == path
            and self.dataset is not None
            and getattr(self.dataset, "data_type", None) == dt
        ):
            return
        meta = describe_dataset(dt, path)
        if reset_selection:
            self._reset_state_selection()
        kwargs = {}
        if meta is not None:
            kwargs = {
                "spectrum": self._mess_spectrum,
                "slowmod": self._mess_slowmod,
                "anisotropy": self._mess_anisotropy,
            }
        try:
            self.dataset = pm.load_1D(path, data_type=dt, **kwargs)
            self._current_path = path
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            self.dataset = None
            self._configure_state_controls(None)
            self._set_dataset_widgets_visible(False)
            self._update_sample_info()
            return
        self._configure_state_controls(meta)
        name = Path(path).name
        dt_display = DATA_TYPE_DISPLAY_NAMES.get(dt, dt)
        cal_str = self.dataset.calibration_status()
        msg = f"Loaded {name} ({dt_display}) — Probe calibration: {cal_str}"
        self.statusBar().showMessage(msg)
        if getattr(self, "plot_controls", None) is not None:
            self.plot_controls.set_dataset(self.dataset)
        self._setup_background_controls()
        self._preview_contour()
        self._update_sample_info()
        self._set_dataset_widgets_visible(True)

        act_deriv = getattr(self, "actionTimeDerivative", None)
        if act_deriv is not None:
            tab_idx = self.MainTabs.currentIndex() if hasattr(self, "MainTabs") and self.MainTabs is not None else 0
            act_deriv.setEnabled(tab_idx == 0 and self.dataset is not None)

        if self.dataset is not None:
            has_single = bool(np.isfinite(self.dataset.nscans) and self.dataset.nscans > 0)
            box_single = getattr(self, "oneD_singleScan_box", None)
            if box_single is not None:
                box_single.setVisible(has_single)

            btn_noise = getattr(self, "PP_plotNoise_btn", None)
            if btn_noise is not None:
                btn_noise.setEnabled(bool(self.dataset.has_noise))

            btn_counts = getattr(self, "PP_plotCounts_btn", None)
            if btn_counts is not None:
                has_counts = bool(getattr(self.dataset, "counts", None) is not None)
                btn_counts.setVisible(has_counts)
                btn_counts.setEnabled(has_counts)
        else:
            btn_noise = getattr(self, "PP_plotNoise_btn", None)
            if btn_noise is not None:
                btn_noise.setEnabled(False)
            btn_counts = getattr(self, "PP_plotCounts_btn", None)
            if btn_counts is not None:
                btn_counts.setVisible(False)
                btn_counts.setEnabled(False)

    def _update_sample_info(self):
        """Refresh the sample-info box from the active dataset (clear if none)."""
        widget = getattr(self, "PP_SampleInfo_text", None)
        if widget is None:
            return
        widget.setHtml("" if self.dataset is None else self.dataset.sample_info())

    def _apply_view_limits(self):
        """Apply the panel X/Y limits to the embedded axis without re-plotting."""
        ax = getattr(self.PPaxes, "ax", None)
        if ax is None or self.dataset is None:
            return
        _safe_set_limits(ax, self.plot_controls.xlim(), self.plot_controls.ylim())
        self.PPaxes.canvas.draw_idle()

    def _load_spectrum(self):
        """Load a steady-state spectrum and plot it in a new figure."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load steady-state spectrum",
            self._rootdir_text(),
            "Spectra (*.csv *.dat *.txt *.dpt);;All files (*)",
        )
        if not path:
            return
        import matplotlib.pyplot as plt

        pm.apply_style()
        try:
            pm.load_spectrum(path).plot()
            plt.show(block=False)
        except Exception as exc:
            QMessageBox.warning(self, "Load failed", str(exc))

    def _load_steady_state(self, kind: str):
        """Load a steady-state (0-D) absorption/emission spectrum as an overlay.

        The spectrum is stored and drawn as a shaded area on every transient-
        spectra figure opened afterwards (forwarded as ``Abs`` / ``Em`` to
        :meth:`Dataset1D.plot_spectra`). The overlay colours, line and fill
        transparencies are taken from the ``ss_*`` settings.
        """
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load steady-state %s" % kind,
            self._rootdir_text(),
            "Spectra (*.csv *.dat *.txt *.dpt);;All files (*)",
        )
        if not path:
            return
        try:
            overlay = pm.load_spectrum(path, kind).as_overlay_dict()
        except Exception as exc:
            QMessageBox.warning(self, "Load failed", str(exc))
            return
        if kind == "absorption":
            self._ss_abs = overlay
        else:
            self._ss_em = overlay
        self.statusBar().showMessage(
            "Loaded steady-state %s (%s) — overlaid on new transient-spectra figures"
            % (kind, Path(path).name)
        )

    def _clear_steady_state(self):
        """Forget the loaded steady-state absorption / emission overlays."""
        self._ss_abs = None
        self._ss_em = None
        self.statusBar().showMessage("Cleared steady-state overlays")

    def _init_state_controls(self):
        """Create the spectrum/slow-mod spin boxes and the anisotropy menu.

        Built programmatically (not in the .ui) and inserted into the data-type
        group as a thin row of ``Spectrum`` / ``Slow mod`` spin boxes above the
        dataset list. Both are hidden until a loaded dataset offers more than one
        state; the anisotropy menu is enabled only for multi-polarisation data.
        """
        self._mess_spectrum = 0
        self._mess_slowmod = 0
        self._mess_anisotropy = "NONE"
        self._mess_meta = None
        self._suppress_state = False
        self._sp_spin = self._sm_spin = self._state_bar = None
        self._sp_label = self._sm_label = None

        group = getattr(self, "PP_datagrp", None)
        lst = getattr(self, "PP_datafolderlist_lst", None)
        layout = group.layout() if group is not None else None
        if layout is not None and lst is not None:
            bar = QWidget(group)
            hb = QHBoxLayout(bar)
            hb.setContentsMargins(0, 0, 0, 0)
            hb.setSpacing(6)
            self._sp_label = QLabel("Spectrum:", bar)
            self._sp_spin = QSpinBox(bar)
            self._sp_spin.setMinimum(0)
            self._sm_label = QLabel("Slow mod:", bar)
            self._sm_spin = QSpinBox(bar)
            self._sm_spin.setMinimum(0)
            for w in (self._sp_label, self._sp_spin, self._sm_label, self._sm_spin):
                hb.addWidget(w)
            hb.addStretch(1)
            self._state_bar = bar
            # Reflow the grid: spin row at row 1, dataset list pushed to row 2.
            layout.removeWidget(lst)
            layout.addWidget(bar, 1, 0, 1, 2)
            layout.addWidget(lst, 2, 0, 1, 2)
            self._sp_spin.valueChanged.connect(self._on_spectrum_changed)
            self._sm_spin.valueChanged.connect(self._on_slowmod_changed)
            bar.setVisible(False)

        self._build_anisotropy_menu()

    def _build_anisotropy_menu(self):
        """Add a menu-bar 'Anisotropy' menu listing the recalculation modes."""
        self._aniso_menu = None
        self._aniso_actions = {}
        menubar = self.menuBar()
        if menubar is None:
            return
        from PyQt6.QtWidgets import QMenu

        menu_help = getattr(self, "menuHelp", None)
        if menu_help is not None:
            menu = QMenu("Anisotropy", self)
            menubar.insertMenu(menu_help.menuAction(), menu)
        else:
            menu = menubar.addMenu("Anisotropy")
        group = QActionGroup(self)
        group.setExclusive(True)
        for mode in MESS_ANISOTROPY_MODES:
            act = menu.addAction(mode)
            act.setCheckable(True)
            act.setChecked(mode == "NONE")
            group.addAction(act)
            act.triggered.connect(lambda _checked=False, m=mode: self._on_anisotropy_changed(m))
            self._aniso_actions[mode] = act
        self._aniso_group = group
        self._aniso_menu = menu
        menu.setEnabled(False)

    def _reset_state_selection(self):
        """Restore the default selection (first state, no anisotropy)."""
        self._mess_spectrum = 0
        self._mess_slowmod = 0
        self._mess_anisotropy = "NONE"
        actions = getattr(self, "_aniso_actions", None)
        act = actions.get("NONE") if actions else None
        if act is not None:
            act.setChecked(True)

    def _configure_state_controls(self, meta):
        """Range/show the spin boxes and enable the anisotropy menu for ``meta``.

        ``meta`` is the loader's describe-dict (``n_spectra`` / ``n_slowmod``) or
        ``None`` for formats without selectable states (controls hidden).
        """
        self._mess_meta = meta
        n_spec = int(meta.get("n_spectra", 1)) if meta else 1
        n_slow = int(meta.get("n_slowmod", 1)) if meta else 1
        self._suppress_state = True
        try:
            if self._sp_spin is not None:
                self._sp_spin.setMaximum(max(n_spec - 1, 0))
                self._sp_spin.setValue(min(self._mess_spectrum, max(n_spec - 1, 0)))
                vis = n_spec > 1
                self._sp_label.setVisible(vis)
                self._sp_spin.setVisible(vis)
                self._sp_spin.setEnabled(vis)
            if self._sm_spin is not None:
                self._sm_spin.setMaximum(max(n_slow - 1, 0))
                self._sm_spin.setValue(min(self._mess_slowmod, max(n_slow - 1, 0)))
                vis = n_slow > 1
                self._sm_label.setVisible(vis)
                self._sm_spin.setVisible(vis)
                self._sm_spin.setEnabled(vis and self._mess_anisotropy == "NONE")
            if self._state_bar is not None:
                self._state_bar.setVisible(n_spec > 1 or n_slow > 1)
            if getattr(self, "_aniso_menu", None) is not None:
                self._aniso_menu.setEnabled(n_slow > 1)
                if n_slow <= 1 and self._mess_anisotropy != "NONE":
                    self._reset_state_selection()
        finally:
            self._suppress_state = False

    def _on_spectrum_changed(self, value):
        if self._suppress_state:
            return
        self._mess_spectrum = int(value)
        self._reload_selection()

    def _on_slowmod_changed(self, value):
        if self._suppress_state:
            return
        self._mess_slowmod = int(value)
        self._reload_selection()

    def _on_anisotropy_changed(self, mode):
        self._mess_anisotropy = mode
        if self._sm_spin is not None:
            n_slow = int((self._mess_meta or {}).get("n_slowmod", 1))
            self._sm_spin.setEnabled(n_slow > 1 and mode == "NONE")
        if not self._suppress_state:
            self._reload_selection()

    def _reload_selection(self):
        """Reload the current dataset with the active state selection."""
        if self._current_path and self.dataset is not None:
            self.load_path(self._current_path, reset_selection=False)

    def _fresh_axis(self):
        """Clear the promoted plot widget and return ``(widget, axis)``."""
        w = self.PPaxes
        w.figure.clear()
        w.ax = w.figure.add_subplot(111)
        if getattr(self, "_1d_scroll_cid", None) is None and hasattr(w, "canvas"):
            self._1d_scroll_cid = w.canvas.mpl_connect("scroll_event", self._on_contour_scroll)
        return w, w.ax

    def _on_contour_scroll(self, event):
        """Scroll mouse wheel over the 1D contour plot to adjust Z scale %."""
        if event.inaxes is None or not hasattr(self, "plot_controls") or self.plot_controls is None:
            return
        step = getattr(event, "step", 0)
        if step == 0:
            step = 1.0 if getattr(event, "button", None) == "up" else (-1.0 if getattr(event, "button", None) == "down" else 0)
        if step == 0:
            return
        pc = self.plot_controls
        if pc.z_pct is None:
            return
        curr_pct = float(pc.z_pct.value())
        eps = float(np.finfo(float).eps)
        factor = 1.1 ** step
        new_pct = max(eps, min(100.0, curr_pct * factor))
        pc.z_pct.setValue(new_pct)

    def _style_preview(self, w):
        """Recolour the embedded figure to match a dark/light GUI theme.

        Only the embedded preview is themed; figures opened by the plot buttons
        keep the white ``.mplstyle`` background.
        """
        from ..theme import is_dark_palette, style_figure

        style_figure(w.figure, is_dark_palette())

    def _setup_background_controls(self):
        """Configure the background-subtraction widgets for the loaded dataset.

        The ``tmin``/``tmax`` spin boxes accept any value within the dataset's
        delay range and default to the two most negative delays (subtracting the
        average spectrum over them). The "Subtract bkg." checkbox defaults to
        off for already-processed PDAT data and on for any other (raw) format.
        """
        if self.dataset is None:
            return
        delays = np.asarray(self.dataset.delays, dtype=float)
        delays = delays[np.isfinite(delays)]
        tmin_box = getattr(self, "PP_bkg_tmin", None)
        tmax_box = getattr(self, "PP_bkg_tmax", None)
        if delays.size and tmin_box is not None and tmax_box is not None:
            lo, hi = float(delays.min()), float(delays.max())
            ordered = np.sort(delays)
            d0 = float(ordered[0])
            d1 = float(ordered[1]) if ordered.size > 1 else d0
            for box, value in ((tmin_box, d0), (tmax_box, d1)):
                box.blockSignals(True)
                box.setRange(lo, hi)
                box.setValue(value)
                box.blockSignals(False)
        sub = getattr(self, "PP_SubtactBkg_chk", None)
        if sub is not None:
            sub.setChecked(self.dataset.data_type != "PDAT")

    def _on_bkg_limit_entered(self):
        """Snap an entered background-window limit to the nearest delay, then render.

        Called when editing of ``PP_bkg_tmin``/``PP_bkg_tmax`` finishes (Enter or
        focus-out). The entered value may be arbitrary; it is rounded to the
        closest delay in the dataset before the background is recomputed and the
        embedded preview re-rendered.
        """
        if self.dataset is None:
            return
        delays = np.asarray(self.dataset.delays, dtype=float)
        delays = delays[np.isfinite(delays)]
        if not delays.size:
            return
        for name in ("PP_bkg_tmin", "PP_bkg_tmax"):
            box = getattr(self, name, None)
            if box is None:
                continue
            nearest = float(delays[np.argmin(np.abs(delays - box.value()))])
            if nearest != box.value():
                box.blockSignals(True)
                box.setValue(nearest)
                box.blockSignals(False)
        self._rerender_preview()

    def _apply_background(self, force: bool = False):
        if self.dataset is None:
            return
        sub = getattr(self, "PP_SubtactBkg_chk", None)
        do_correct = force or (sub.isChecked() if sub is not None else False)
        tmin_box = getattr(self, "PP_bkg_tmin", None)
        tmax_box = getattr(self, "PP_bkg_tmax", None)
        tmin = tmin_box.value() if tmin_box is not None else -20.0
        tmax = tmax_box.value() if tmax_box is not None else -5.0
        self.dataset.background_correct(tmin, tmax, do_correct=do_correct)

        # If a solvent background subtraction has been performed, re-apply it on top
        solvent_ds = getattr(self.dataset, "_solvent_dataset", None)
        if solvent_ds is not None:
            dt = getattr(self.dataset, "_solvent_dt", 0.0)
            scale = getattr(self.dataset, "_solvent_scale", 1.0)
            self.dataset.subtract_solvent(solvent_ds, dt=dt, scale=scale)

        # If a shockwave subtraction has been performed, re-apply it on top
        sw_pixels = getattr(self.dataset, "_shockwave_pixels", None)
        if sw_pixels is not None and sw_pixels.size > 0:
            self.dataset.subtract_shockwave(pixels=sw_pixels, one_based=False)

    def _norm(self) -> bool:
        chk = getattr(self, "PP_Normalise_chk", None)
        return bool(chk.isChecked()) if chk is not None else False

    def _smooth_value(self) -> int:
        """``doSmooth`` for cut plots, taken from the plot-controls Smooth box."""
        pc = getattr(self, "plot_controls", None)
        return pc.smooth() if pc is not None else 0

    def _preview_contour(self):
        """Draw the contour on the embedded canvas (used on load)."""
        if self.dataset is None:
            return
        pm.apply_style()
        self._apply_background()
        w, ax = self._fresh_axis()
        pc = getattr(self, "plot_controls", None)
        kwargs = pc.contour_kwargs() if pc is not None else {}
        try:
            self.dataset.plot_contour(ax=ax, **kwargs)
        except Exception as exc:
            QMessageBox.warning(self, "Plot failed", str(exc))
        ax.set_aspect("auto")
        # plot_contour only lays out figures it created itself (so composite
        # GridSpec figures survive), so the embedded canvas does it here.
        with contextlib.suppress(Exception):
            w.figure.set_layout_engine(None)
            w.figure.tight_layout()
        if pc is not None:
            _safe_set_limits(ax, pc.xlim(), pc.ylim())
        self._style_preview(w)
        w.canvas.draw()

    def _plot_background(self):
        """Plot the stored pre-zero background spectrum in a new window.

        Routed through :meth:`Dataset1D.plot_background`, so it follows the same
        rules as the transient-spectra cuts (figure size, axis labels, gap
        masking, smoothing/normalisation) but without a legend. The background is
        computed over the current ``tmin``/``tmax`` window and is always stored on
        the dataset (independent of the Subtract-bkg. state).
        """
        if self.dataset is None:
            self.open_file()
            return
        import matplotlib.pyplot as plt

        pm.apply_style()
        self._apply_background()
        pc = getattr(self, "plot_controls", None)
        det = getattr(pc, "detector", 0) if pc is not None else 0
        try:
            self.dataset.plot_background(doSmooth=self._smooth_value(), normY=self._norm(), detector=det)
            plt.show(block=False)
        except Exception as exc:
            QMessageBox.warning(self, "Plot failed", str(exc))

    def _plot_noise(self):
        """Plot the per-point noise 3D surface map in a new window."""
        if self.dataset is None:
            return
        import matplotlib.pyplot as plt

        noise = self.dataset.noise_array()
        if noise is None:
            QMessageBox.warning(self, "Plot failed", "No noise or single-scan data available.")
            return

        pm.apply_style()
        self._apply_background()
        pc = getattr(self, "plot_controls", None)
        det = getattr(pc, "detector", 0) if pc is not None else 0
        try:
            ax = self.dataset.plot_noise_3d(detector=det)
            if pc is not None:
                _safe_set_limits(ax, pc.xlim(), pc.ylim())
            plt.show(block=False)
        except Exception as exc:
            QMessageBox.warning(self, "Plot failed", str(exc))

    def _plot_counts(self):
        """Plot accumulation counts vs probe delay in a new window."""
        if self.dataset is None:
            self.open_file()
            return
        import matplotlib.pyplot as plt

        counts = getattr(self.dataset, "counts", None)
        if counts is None and hasattr(self.dataset, "units"):
            counts = self.dataset.units.get("counts")

        if counts is None:
            QMessageBox.warning(self, "Plot failed", "No accumulation counts available for this dataset.")
            return

        pm.apply_style()
        pc = getattr(self, "plot_controls", None)
        kwargs = pc.time_axis_kwargs() if pc is not None else {}

        try:
            self.dataset.plot_counts(**kwargs)
            plt.show(block=False)
        except Exception as exc:
            QMessageBox.warning(self, "Plot failed", str(exc))

    def _plot_cut(self, kind: str):
        """Open a NEW figure window for a contour/surface/spectral/kinetic cut.

        The spectral and kinetic cuts prompt for the delays / probe positions.
        """
        if self.dataset is None:
            self.open_file()
            return
        import matplotlib.pyplot as plt

        pm.apply_style()
        self._apply_background()
        pc = getattr(self, "plot_controls", None)
        det = getattr(pc, "detector", 0) if pc is not None else 0
        try:
            if kind == "contour":
                kwargs = pc.contour_kwargs() if pc is not None else {}
                ax = self.dataset.plot_contour(**kwargs)  # ax=None -> new figure
                if pc is not None:
                    _safe_set_limits(ax, pc.xlim(), pc.ylim())
            elif kind == "surface":
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection="3d")
                # Match the embedded contour: same delay-axis scale and label
                # format, and the probe/delay limits set in the panel.
                surf_kwargs = pc.time_axis_kwargs() if pc is not None else {}
                self.dataset.plot_surface(ax=ax, detector=det, **surf_kwargs)
                if pc is not None:
                    _safe_set_limits(ax, pc.xlim(), pc.ylim())
                if hasattr(ax, "mouse_init"):
                    ax.mouse_init()
                fig.tight_layout()
                plt.show(block=False)
            elif kind == "spectra":
                if self._interactive():
                    self._pick_cut("spectra")
                    return
                delays = self._ask_values(
                    "Transient spectra",
                    "Delays to plot (comma-separated or MATLAB range e.g. 1:0.1:5, 'all'):",
                    self._auto_delays(),
                    all_values=self.dataset.delays,
                )
                if not delays:
                    return
                ax = self.dataset.plot_spectra(
                    delays,
                    doSmooth=self._smooth_value(),
                    normY=self._norm(),
                    Abs=self._ss_abs,
                    Em=self._ss_em,
                    detector=det,
                )
                self._inherit_cut_limits("spectra", ax)
            elif kind == "kinetics":
                if self._interactive():
                    self._pick_cut("kinetics")
                    return
                probe_disp = self.dataset._detector_probe(det)
                if pc is not None:
                    probe_disp = pc.probe_from_native(probe_disp)
                positions = self._ask_values(
                    "Kinetic cuts",
                    "Probe wavelengths / wavenumbers (comma-separated or MATLAB range e.g. 1:0.1:5, 'all'):",
                    self._auto_wavelengths(),
                    all_values=probe_disp,
                )
                if not positions:
                    return
                positions = self._probe_to_native(positions)
                ax = self.dataset.plot_kinetics(positions, normY=self._norm(), detector=det)[0]
                self._inherit_cut_limits("kinetics", ax)
            plt.show(block=False)
        except Exception as exc:
            QMessageBox.warning(self, "Plot failed", str(exc))

    def _binscans_value(self) -> int:
        """Scan-bin size from the ``PP_BinScans`` spin box (>= 1)."""
        w = getattr(self, "PP_BinScans", None)
        try:
            return max(int(w.value()), 1) if w is not None else 1
        except (ValueError, TypeError):
            return 1

    def _plot_scan_cut(self, kind: str):
        """Open a NEW figure with per-scan kinetic / spectral cuts.

        Uses the single-scan data (one trace per scan, or per ``Bin`` group of
        scans averaged together) at the requested probe positions / delays, and
        otherwise mirrors :meth:`_plot_cut` (interactive picking, smoothing,
        normalisation and inherited panel limits).
        """
        if self.dataset is None:
            self.open_file()
            return
        if not getattr(self.dataset, "has_single_scans", False):
            QMessageBox.information(
                self,
                "Per-scan plot",
                "This dataset has no per-scan data loaded. Enable 'Load single "
                "scans' in Settings and reload the dataset.",
            )
            return
        import matplotlib.pyplot as plt

        pm.apply_style()
        self._apply_background()
        binsize = self._binscans_value()
        try:
            if kind == "spectra":
                if self._interactive():
                    self._pick_cut("spectra", scans=True)
                    return
                delays = self._ask_values(
                    "Per-scan spectra",
                    "Delays to plot (comma-separated or MATLAB range e.g. 1:0.1:5, 'all'):",
                    self._auto_delays(),
                    all_values=self.dataset.delays,
                )
                if not delays:
                    return
                ax = self.dataset.plot_scan_spectra(
                    delays, binsize=binsize, doSmooth=self._smooth_value(), normY=self._norm()
                )
                self._inherit_cut_limits("spectra", ax)
            elif kind == "kinetics":
                if self._interactive():
                    self._pick_cut("kinetics", scans=True)
                    return
                det = getattr(self.plot_controls, "detector", 0) if hasattr(self, "plot_controls") else 0
                probe_disp = self.dataset._detector_probe(det)
                pc = getattr(self, "plot_controls", None)
                if pc is not None:
                    probe_disp = pc.probe_from_native(probe_disp)
                positions = self._ask_values(
                    "Per-scan kinetics",
                    "Probe wavelengths / wavenumbers (comma-separated or MATLAB range e.g. 1:0.1:5, 'all'):",
                    self._auto_wavelengths(),
                    all_values=probe_disp,
                )
                if not positions:
                    return
                positions = self._probe_to_native(positions)
                ax = self.dataset.plot_scan_kinetics(
                    positions, binsize=binsize, normY=self._norm()
                )[0]
                self._inherit_cut_limits("kinetics", ax)
            plt.show(block=False)
        except Exception as exc:
            QMessageBox.warning(self, "Plot failed", str(exc))

    @busy_guard("Recalculating the scan average...")
    def _recalc_average(self):
        """Recompute the noise-weighted average over a chosen subset of scans.

        Prompts for a 1-based scan selection (e.g. ``1-4 6 9``), reads the
        per-scan signal and noise from the dataset's ``temp/`` folder, and
        replaces the averaged signal/noise by their inverse-variance weighted
        combination. Selecting every scan reproduces the original averaged data.
        Available for MESS directory datasets with the un-combined signal
        (anisotropy = NONE).
        """
        if self.dataset is None:
            self.open_file()
            return
        if getattr(self, "_mess_anisotropy", "NONE") != "NONE":
            QMessageBox.information(
                self,
                "Recalc. average",
                "Recalculation applies to the un-combined signal only "
                "(set the anisotropy mode to NONE).",
            )
            return
        source = getattr(self.dataset, "source", None)
        nsc = getattr(self.dataset, "nscans", float("nan"))
        if not source or not np.isfinite(nsc) or int(nsc) < 1:
            QMessageBox.information(
                self, "Recalc. average", "This dataset has no per-scan data to recombine."
            )
            return
        n = int(nsc)
        text, ok = QInputDialog.getText(
            self,
            "Recalc. average",
            f"Scans to combine (1-{n}), e.g. '1-{n}', '1-3 5 {n}':",
            text=f"1-{n}",
        )
        if not ok or not text.strip():
            return
        try:
            idx = parse_scan_selection(text, n)
        except ValueError as exc:
            QMessageBox.warning(self, "Recalc. average", str(exc))
            return
        if not idx:
            QMessageBox.warning(self, "Recalc. average", "No scans selected.")
            return
        try:
            Zavg, Zstdv = mess_recalc_average(
                source,
                idx,
                spectrum=getattr(self, "_mess_spectrum", 0),
                slowmod=getattr(self, "_mess_slowmod", 0),
            )
        except Exception as exc:
            QMessageBox.warning(self, "Recalc. average", str(exc))
            return
        self.dataset.Zavg_R = Zavg
        self.dataset.Zstdv = Zstdv
        self.dataset.nscans = len(idx)
        self.dataset.Zavg_C = None  # force the background correction to re-apply
        if getattr(self, "plot_controls", None) is not None:
            self.plot_controls.set_dataset(self.dataset)
        self._preview_contour()
        self._update_sample_info()
        btn_noise = getattr(self, "PP_plotNoise_btn", None)
        if btn_noise is not None:
            btn_noise.setEnabled(bool(self.dataset.has_noise))
        self.statusBar().showMessage(
            f"Recalculated noise-weighted average over {len(idx)} scan(s): {text.strip()}"
        )

    @busy_guard("Masking probe region...")
    def _mask_probe_region(self):
        """Prompt to mask one or more ranges in the probe axis using MATLAB-type syntax."""
        if self.dataset is None:
            self.open_file()
            if self.dataset is None:
                return

        if not hasattr(self, "_last_mask_text"):
            self._last_mask_text = ""

        text, ok = QInputDialog.getText(
            self,
            "Mask Probe Region",
            "Enter probe ranges to mask out (e.g. '0,350;550,5000' or '0:350;550:5000'):",
            text=self._last_mask_text,
        )
        if not ok or not text.strip():
            return

        try:
            parts = text.split(";")
            ranges_display = []
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                subparts = [p.strip() for p in re.split(r"[,:]", part) if p.strip()]
                if len(subparts) != 2:
                    raise ValueError(
                        f"Invalid range format: '{part}'. Ranges must be start,end or start:end."
                    )
                try:
                    val1 = float(subparts[0])
                    val2 = float(subparts[1])
                except ValueError as err:
                    raise ValueError(f"Could not parse numbers in range: '{part}'") from err
                ranges_display.append((min(val1, val2), max(val1, val2)))
        except ValueError as exc:
            QMessageBox.warning(self, "Mask Probe Region", str(exc))
            return

        self._last_mask_text = text.strip()

        # Convert display units to native units
        ranges_native = []
        for r_min, r_max in ranges_display:
            converted = self._probe_to_native([r_min, r_max])
            ranges_native.append((min(converted), max(converted)))

        try:
            det = getattr(self.plot_controls, "detector", 0) if hasattr(self, "plot_controls") else 0

            # Apply mask to current dataset for active detector
            self.dataset.mask_probe_regions(ranges_native, detector=det)

            # Apply mask to solvent dataset if one is active/loaded
            solvent_ds = getattr(self.dataset, "_solvent_dataset", None)
            if solvent_ds is not None:
                solvent_ds.mask_probe_regions(ranges_native, detector=det)

            # Update the plot controls dataset cache
            if getattr(self, "plot_controls", None) is not None:
                self.plot_controls.set_dataset(self.dataset)

            # Force re-rendering
            self._preview_contour()
            self._update_sample_info()
            self.statusBar().showMessage(
                f"Applied probe mask to detector {det + 1}: {self._last_mask_text}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Mask Probe Region Failed", str(exc))

    @busy_guard("Shifting t0...")
    def _shift_t0(self):
        """Prompt to shift the time axis t0 by a user-specified delay offset."""
        if self.dataset is None:
            self.open_file()
            if self.dataset is None:
                return

        unit_str = self.dataset.units.get("unitsT_ltx", "ps")
        t0_val, ok = QInputDialog.getDouble(
            self,
            "Shift t0",
            f"Enter new t0 position ({unit_str}) to shift to 0:",
            value=0.0,
            decimals=4,
        )
        if not ok or t0_val == 0.0:
            return

        self.dataset.shift_t0(t0_val)
        if getattr(self, "plot_controls", None) is not None:
            self.plot_controls.set_dataset(self.dataset)
        self._preview_contour()
        self._update_sample_info()
        self.statusBar().showMessage(f"Shifted time axis t0 by {-t0_val:+.4g} {unit_str}")

    @busy_guard("Subtracting solvent...")
    def _subtract_solvent(self):
        """Prompt to select a solvent dataset and subtract it with time-offset/scaling."""
        if self.dataset is None:
            self.open_file()
            if self.dataset is None:
                return

        from pymorgan.oneD.load import describe_dataset, is_directory_format

        dt = self.dataset.data_type or self._current_datatype()
        start = self._rootdir_text() or ""

        if is_directory_format(dt):
            path = QFileDialog.getExistingDirectory(self, "Select Solvent Dataset Folder", start)
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Solvent Dataset File", start, "Data Files (*);;All files (*)"
            )
        if not path:
            return

        kwargs = {}
        meta = describe_dataset(dt, path)
        if meta is not None:
            kwargs = {
                "spectrum": self._mess_spectrum,
                "slowmod": self._mess_slowmod,
                "anisotropy": self._mess_anisotropy,
            }
        try:
            solvent_ds = pm.load_1D(path, data_type=dt, **kwargs)
        except Exception as exc:
            QMessageBox.critical(
                self, "Load Solvent Failed", f"Could not load solvent dataset:\n{exc}"
            )
            return

        if hasattr(self.dataset, "mask_ranges_by_det"):
            for d, r_list in self.dataset.mask_ranges_by_det.items():
                solvent_ds.mask_probe_regions(r_list, detector=d)
        elif hasattr(self.dataset, "mask_ranges") and self.dataset.mask_ranges:
            det = getattr(self.plot_controls, "detector", 0) if hasattr(self, "plot_controls") else 0
            solvent_ds.mask_probe_regions(self.dataset.mask_ranges, detector=det)

        if solvent_ds.n_detectors != self.dataset.n_detectors:
            QMessageBox.warning(
                self,
                "Dimension Mismatch",
                "Solvent dataset must have the same detector channels as the sample dataset.",
            )
            return

        # Check if probe axes overlap
        min_sample, max_sample = np.min(self.dataset.probe), np.max(self.dataset.probe)
        min_solvent, max_solvent = np.min(solvent_ds.probe), np.max(solvent_ds.probe)
        if max_solvent < min_sample or min_solvent > max_sample:
            QMessageBox.warning(
                self,
                "Probe Range Mismatch",
                "The solvent and sample probe ranges do not overlap.",
            )
            return

        default_dt = getattr(self.dataset, "_solvent_dt", 0.0)
        default_scale = getattr(self.dataset, "_solvent_scale", 1.0)

        from ..dialogs import SolventSubtractionDialog

        dlg = SolventSubtractionDialog(
            self, Path(path).name, self.dataset, solvent_ds, default_dt, default_scale
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        dt_val, scale_val = dlg.values()

        try:
            self.dataset.subtract_solvent(solvent_ds, dt=dt_val, scale=scale_val)
            self._preview_contour()
            dt_mean = float(np.nanmean(dt_val)) if np.ndim(dt_val) > 0 else float(dt_val)
            scale_mean = (
                float(np.nanmean(scale_val)) if np.ndim(scale_val) > 0 else float(scale_val)
            )
            self.statusBar().showMessage(
                f"Solvent background subtracted (dt={dt_mean:.3f} ps, scale={scale_mean:.3f})"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Solvent Subtraction Failed", str(exc))

    @busy_guard("Subtracting shock wave...")
    def _subtract_shockwave(self):
        """Prompt to select pixel(s) or probe range and subtract an average shockwave kinetic trace."""
        if self.dataset is None:
            self.open_file()
            if self.dataset is None:
                return

        from pymorgan.gui.shockwave_dialog import ShockwaveSubtractionDialog

        dlg = ShockwaveSubtractionDialog(self, self.dataset)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        if dlg.selected_pixels is None or dlg.selected_pixels.size == 0:
            return

        try:
            self.dataset.subtract_shockwave(pixels=dlg.selected_pixels, one_based=False)
            self._preview_contour()
            p_idx = dlg.selected_pixels
            if p_idx.size == 1:
                p_str = f"pixel {p_idx[0] + 1}"
            else:
                p_str = f"pixels {p_idx[0] + 1}..{p_idx[-1] + 1}"
            self.statusBar().showMessage(f"Shockwave kinetic trace subtracted ({p_str})")
        except Exception as exc:
            QMessageBox.critical(self, "Shockwave Subtraction Failed", str(exc))

    @busy_guard("Calculating time derivative...")
    def _calculate_time_derivative(self):
        """Prompt for options and compute the time derivative d(Delta A)/dt of transient spectra."""
        if self.dataset is None:
            self.open_file()
            if self.dataset is None:
                return

        from pymorgan.gui.time_derivative_dialog import TimeDerivativeDialog

        dlg = TimeDerivativeDialog(self, self.dataset)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        try:
            opts = dlg.get_options()
            self.dataset = self.dataset.time_derivative(**opts)
            self._preview_contour()
            self._update_sample_info()
            self.statusBar().showMessage("Calculated time derivative d(\u0394A)/dt of spectra")
        except Exception as exc:
            QMessageBox.critical(self, "Time Derivative Failed", str(exc))

    @busy_guard("Fitting chirp correction...")
    def _fit_chirp(self):
        """Fit the chirp/dispersion correction (Automatic / Manual / Step Function).

        Shows the options dialog, runs the selected fit (synchronously, with a
        cancellable progress dialog for the per-pixel Automatic/Step modes; the
        Manual mode instead starts interactive point-picking on the embedded
        contour), then hands off to :meth:`_finish_chirp_fit` for the diagnostic
        plots, accept/reject prompt and ``.mat`` save. Loading a saved fit back
        and applying it as a correction is not implemented yet (``PP_loadChirpCorr_btn``
        is intentionally left unwired).
        """
        if self.dataset is None:
            self.open_file()
            return
        from ..chirp_dialog import (
            MODE_AUTOMATIC,
            MODE_MANUAL,
            MODE_STEP,
            MODE_WAVELET,
            ChirpFitOptionsDialog,
        )
        from ..chirp_progress_dialog import ChirpFitProgressDialog

        det = getattr(self.plot_controls, "detector", 0) if hasattr(self, "plot_controls") else 0
        dlg = ChirpFitOptionsDialog(self, n_detectors=self.dataset.n_detectors, active_detector=det)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        mode = dlg.mode()
        kwargs = dlg.fit_kwargs()
        show_preview = dlg.preview_enabled()

        if mode == MODE_MANUAL:
            self._fit_chirp_manual_pick(kwargs)
            return

        if mode == MODE_AUTOMATIC:
            fit_func = fit_chirp_automatic
        elif mode == MODE_STEP:
            fit_func = fit_chirp_step
        elif mode == MODE_WAVELET:
            fit_func = fit_chirp_wavelet
        else:
            return

        progress = ChirpFitProgressDialog(self, show_preview=show_preview)
        progress.show()

        def on_progress(i, n, msg):
            progress.set_progress(i, n, msg)
            QApplication.processEvents()

        def on_preview(wavelength, delays, y, model, t0):
            progress.set_preview(wavelength, delays, y, model, t0)

        def on_opt_progress(iteration, param_str, rms_fs):
            progress.label.setText(
                f"Optimising wavelet parameters (iter {iteration})\n"
                f"{param_str}\n"
                f"RMS residual: {rms_fs:.2f} fs"
            )
            QApplication.processEvents()

        def should_cancel():
            QApplication.processEvents()
            return progress.wasCanceled()

        try:
            fit = fit_func(
                self.dataset,
                progress_callback=on_progress,
                should_cancel=should_cancel,
                preview_callback=on_preview if show_preview else None,
                **(
                    ({"opt_progress_callback": on_opt_progress} if mode == MODE_WAVELET else {})
                    | kwargs
                ),
            )
        except InterruptedError:
            self.statusBar().showMessage("Chirp fit cancelled.")
            return
        except Exception as exc:
            QMessageBox.warning(self, "Fit chirp", str(exc))
            return
        finally:
            progress.close()

        self._finish_chirp_fit(fit)

    def _fit_chirp_manual_pick(self, kwargs: dict):
        """Start interactive (wavelength, delay) point-picking for a Manual chirp fit."""
        from ..picker import ContourPicker

        self._preview_contour()
        ax = getattr(self.PPaxes, "ax", None)
        if ax is None:
            return
        self.statusBar().showMessage(
            "Manual chirp fit: left-click to add (wavelength, delay) points, "
            "right-click to remove the last, Enter to fit, Esc to cancel."
        )

        def done(values):
            self._picker = None
            if not values:
                self.statusBar().showMessage("Manual chirp selection cancelled.")
                return
            self.statusBar().clearMessage()
            try:
                det = getattr(self.plot_controls, "detector", 0) if hasattr(self, "plot_controls") else 0
                fit = self.dataset.fit_chirp_manual(values, detector=det, **kwargs)
            except Exception as exc:
                QMessageBox.warning(self, "Fit chirp", str(exc))
                return
            self._finish_chirp_fit(fit)

        self._picker = ContourPicker(self.PPaxes.canvas, ax, "xy", done).start()

    @busy_guard("Applying chirp correction...")
    def _load_chirp_corr(self):
        """Load a saved chirp correction (.mat, .h5, or .json), apply it, and plot its diagnostics.

        The diagnostic plot is generated *before* the correction is applied so
        that the contour overlay shows the uncorrected dataset and illustrates
        how the loaded chirp curve fits the original data.
        """
        from pathlib import Path

        from PyQt6.QtWidgets import QFileDialog, QMessageBox

        from pymorgan.oneD.chirp import load_chirp_fit

        if self.dataset is None:
            QMessageBox.warning(self, "Load chirp correction", "No dataset is loaded.")
            return

        start_dir = self._rootdir_text() or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load chirp correction",
            start_dir,
            "Chirp correction files (*.mat *.h5 *.hdf5 *.json);;MATLAB files (*.mat);;HDF5 files (*.h5 *.hdf5);;JSON files (*.json);;All files (*)",
        )
        if not path:
            return

        try:
            fit = load_chirp_fit(path)
        except Exception as exc:
            QMessageBox.warning(
                self, "Load chirp correction", f"Error loading chirp correction file: {exc}"
            )
            return

        import matplotlib.pyplot as plt

        # Plot diagnostics on the *uncorrected* dataset so the contour shows the
        # original data with the loaded chirp curve overlaid.
        try:
            self.dataset.plot_chirp_diagnostics(fit)
            plt.show(block=False)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Load chirp correction",
                f"Loaded fit successfully, but plotting diagnostics failed: {exc}",
            )

        try:
            det = getattr(self.plot_controls, "detector", 0) if hasattr(self, "plot_controls") else getattr(fit, "detector", 0)
            self.dataset.apply_chirp_correction(fit, detector=det)
        except Exception as exc:
            QMessageBox.warning(
                self, "Load chirp correction", f"Error applying chirp correction: {exc}"
            )
            return

        self.dataset.chirp_fit = fit
        self._preview_contour()
        self.statusBar().showMessage(f"Chirp correction loaded and applied from: {path}")

        summary_lines = [
            f"Mode: {fit.mode}",
            "Cauchy coefficients: " + ", ".join(f"{c:.4g}" for c in fit.coeffs),
        ]
        if fit.mode == "wavelet" and np.isfinite(getattr(fit, "omega_w", float("nan"))):
            summary_lines.append(
                f"Wavelet parameters: \u03c9 = {fit.omega_w * 1000:.1f} fs, "
                f"\u03b3 = {fit.gamma_w * 1000:.1f} fs"
            )
        if fit.n_skipped:
            summary_lines.append(f"Skipped pixels: {fit.n_skipped}")
        if np.isfinite(fit.mean_irf_fs):
            summary_lines.append(f"Mean IRF: {fit.mean_irf_fs:.1f} fs")
        summary = "\n".join(summary_lines)

        QMessageBox.information(
            self,
            "Load chirp correction",
            f"Successfully loaded and applied chirp correction:\n\n{summary}",
        )

    def _finish_chirp_fit(self, fit):
        """Show the diagnostic plots on the uncorrected data, apply the fit immediately,
        then ask to save the correction to a file (.mat/.h5/.json)."""
        import matplotlib.pyplot as plt
        from PyQt6.QtWidgets import QMessageBox

        # Plot diagnostics *before* applying the correction so the contour shows
        # the uncorrected dataset, illustrating how the fit curve aligns with the
        # original (uncorrected) data.
        try:
            self.dataset.plot_chirp_diagnostics(fit)
            plt.show(block=False)
        except Exception as exc:
            QMessageBox.warning(self, "Fit chirp", f"Fit succeeded but diagnostics failed: {exc}")

        # Apply the correction immediately -- always, regardless of whether the
        # user later chooses to save the file.
        try:
            det = getattr(fit, "detector", 0)
            self.dataset.apply_chirp_correction(fit, detector=det)
            self.dataset.chirp_fit = fit
            self._preview_contour()
        except Exception as exc:
            QMessageBox.warning(self, "Fit chirp", f"Error applying chirp correction: {exc}")
            return

        summary_lines = [
            f"Mode: {fit.mode}",
            "Cauchy coefficients: " + ", ".join(f"{c:.4g}" for c in fit.coeffs),
        ]
        if fit.mode == "wavelet" and np.isfinite(getattr(fit, "omega_w", float("nan"))):
            summary_lines.append(
                f"Wavelet parameters: \u03c9 = {fit.omega_w * 1000:.1f} fs, "
                f"\u03b3 = {fit.gamma_w * 1000:.1f} fs"
            )
        if fit.n_skipped:
            summary_lines.append(f"Skipped pixels: {fit.n_skipped}")
        if np.isfinite(fit.mean_irf_fs):
            summary_lines.append(f"Mean IRF: {fit.mean_irf_fs:.1f} fs")
        summary = "\n".join(summary_lines)

        # Create a non-blocking (window-modal) confirmation box so the user can interact
        # with the opened matplotlib diagnostic plots.
        text = f"{summary}\n\nChirp correction has been applied.\n"
        if fit.mode == "wavelet":
            text += (
                "\nMethod Reference:\n"
                "Kefer, O.; Buckup, T.; Kolesnichenko, P. V. Retroactive correction for "
                "white-light dispersion as an edge-detection problem in ultrafast spectroscopies. "
                "Applied Optics 2024, 63, 15, 4015-4024. DOI: 10.1364/AO.532878\n\n"
            )
        text += "Save the correction to a file?"

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Fit chirp")
        msg_box.setText(text)
        msg_box.setIcon(QMessageBox.Icon.Question)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg_box.setDefaultButton(QMessageBox.StandardButton.No)
        msg_box.setWindowModality(Qt.WindowModality.WindowModal)

        def on_finished(result):
            if result != QMessageBox.StandardButton.Yes:
                self.statusBar().showMessage("Chirp correction applied (not saved).")
                return

            det = getattr(fit, "detector", 0)
            n_det = getattr(self.dataset, "n_detectors", 1)
            start_dir = self._rootdir_text() or str(Path.home())
            default_path = str(Path(start_dir) / default_chirp_filename(detector=det, n_detectors=n_det))
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save chirp correction",
                default_path,
                "MATLAB files (*.mat);;HDF5 files (*.h5 *.hdf5);;JSON files (*.json);;All files (*)",
            )
            if not path:
                self.statusBar().showMessage("Chirp correction applied (not saved).")
                return
            try:
                save_chirp_fit(fit, path, n_detectors=n_det)
                self.statusBar().showMessage(f"Chirp correction applied and saved to: {path}")
            except Exception as exc:
                QMessageBox.warning(self, "Fit chirp", f"Error saving chirp correction file: {exc}")

        msg_box.finished.connect(on_finished)
        msg_box.show()

    def _interactive(self) -> bool:
        """Whether the Interactive tick box is checked (pick cuts on the map)."""
        chk = getattr(self, "PP_Interactive_TickBox", None)
        return bool(chk.isChecked()) if chk is not None else False

    def _pick_cut(self, kind: str, scans: bool = False):
        """Pick kinetic/spectral cut positions by clicking the embedded contour.

        Kinetic cuts read the probe (X) coordinate, spectral cuts the delay (Y).
        Left-click adds a guide-line, right-click removes the last, Enter plots the
        sorted cuts in a new figure, Escape cancels.
        """
        from ..picker import ContourPicker

        if self.dataset is None:
            return
        # Re-draw the preview so picking happens on the current contour.
        self._preview_contour()
        ax = getattr(self.PPaxes, "ax", None)
        if ax is None:
            return
        axis = "x" if kind == "kinetics" else "y"
        what = "probe positions" if kind == "kinetics" else "delays"
        self.statusBar().showMessage(
            f"Interactive {kind}: left-click to add {what}, right-click to remove the "
            "last, Enter to plot, Esc to cancel."
        )

        def done(values):
            self._picker = None
            if not values:
                self.statusBar().showMessage("Interactive selection cancelled.")
                return
            self.statusBar().clearMessage()
            self._plot_picked(kind, values, scans=scans)

        self._picker = ContourPicker(self.PPaxes.canvas, ax, axis, done).start()

    def _plot_picked(self, kind: str, values: list, scans: bool = False):
        """Open the kinetic / spectral cut for interactively picked positions.

        With ``scans=True`` the per-scan plotters are used (one trace per scan or
        per ``Bin`` group), otherwise the averaged-signal cuts are drawn.
        """
        import matplotlib.pyplot as plt

        pm.apply_style()
        self._apply_background()
        pc = getattr(self, "plot_controls", None)
        det = getattr(pc, "detector", 0) if pc is not None else 0
        try:
            if kind == "kinetics":
                values = self._probe_to_native(values)
            if scans:
                binsize = self._binscans_value()
                if kind == "kinetics":
                    ax = self.dataset.plot_scan_kinetics(
                        values, binsize=binsize, normY=self._norm(), detector=det
                    )[0]
                else:
                    ax = self.dataset.plot_scan_spectra(
                        values, binsize=binsize, doSmooth=self._smooth_value(), normY=self._norm(), detector=det
                    )
            elif kind == "kinetics":
                ax = self.dataset.plot_kinetics(values, normY=self._norm(), detector=det)[0]
            else:
                ax = self.dataset.plot_spectra(
                    values,
                    doSmooth=self._smooth_value(),
                    normY=self._norm(),
                    Abs=self._ss_abs,
                    Em=self._ss_em,
                    detector=det,
                )
            self._inherit_cut_limits(kind, ax)
            plt.show(block=False)
        except Exception as exc:
            QMessageBox.warning(self, "Plot failed", str(exc))

    def _inherit_cut_limits(self, kind: str, ax) -> None:
        """Copy the embedded contour's limits onto a new cut figure.

        Active only when ``Settings.inherit_cut_limits`` is set. The cut axes are
        oriented differently from the contour: spectra plot the probe on X,
        kinetics plot the delay on X, and both show the signal on Y. The shared
        axis therefore inherits the contour's matching span (spectra X <- X,
        kinetics X <- Y). The signal (Y) axis inherits the contour Z limits
        unless the plot-controls "Auto-Y in cuts" box is checked, in which case
        it is left to autoscale to the cut data.
        """
        if not getattr(pm.get_settings(), "inherit_cut_limits", False):
            return
        pc = getattr(self, "plot_controls", None)
        if pc is None or ax is None:
            return
        xlim = pc.xlim() if kind == "spectra" else pc.ylim()
        if pc.autoscale_cut_y():
            # Inherit only the shared (X) axis; let the signal (Y) axis autoscale.
            xmin, xmax = xlim
            if ax.get_xscale() == "log" and xmin <= 0:
                ax.set_xlim(right=xmax)
            else:
                ax.set_xlim(xmin, xmax)
        else:
            _safe_set_limits(ax, xlim, pc.zlimits())
        ax.figure.canvas.draw_idle()

    def _auto_delays(self) -> list:
        dmax = float(np.nanmax(self.dataset.delays))
        delays = [d for d in _DEFAULT_SPEC_DELAYS if d <= dmax]
        if not delays:  # dataset shorter than the smallest default
            pool = self.dataset.delays[self.dataset.delays > 0]
            pool = pool if pool.size else self.dataset.delays
            idx = np.unique(np.linspace(0, pool.size - 1, min(6, pool.size)).astype(int))
            delays = [float(v) for v in pool[idx]]
        return delays

    def _probe_to_native(self, values):
        """Convert probe positions from the displayed X unit back to the native unit."""
        pc = getattr(self, "plot_controls", None)
        return pc.probe_to_native(values) if pc is not None else values

    def _auto_wavelengths(self, n: int = 3) -> list:
        det = getattr(self.plot_controls, "detector", 0) if hasattr(self, "plot_controls") else 0
        probe = self.dataset._detector_probe(det)
        idx = np.unique(np.linspace(0, probe.size - 1, min(n, probe.size)).astype(int))
        vals = [float(probe[i]) for i in idx]
        pc = getattr(self, "plot_controls", None)
        if pc is not None:
            vals = pc.probe_from_native(vals)  # show defaults in the displayed unit
        return [round(float(v)) for v in vals]  # rounded suggestions

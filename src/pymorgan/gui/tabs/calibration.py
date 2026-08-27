"""Spectrometer-calibration tab: detector and pulse-shaper calibration.

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
from mpl_axes_aligner import align
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QMessageBox,
    QVBoxLayout,
)

import pymorgan as pm

from ...log import get_logger
from ..busy import busy_guard

logger = get_logger(__name__)


class CalibrationTabMixin:
    """Spectrometer-calibration tab: detector and pulse-shaper calibration."""

    def _ensure_calibration_initialized(self):
        """Lazily initialize the Spectrometer & Shaper Calibration tab canvases & controls on demand."""
        if getattr(self, "_calibration_initialized", False):
            return
        self._calibration_initialized = True
        self._init_calibration_controls()

    def _init_calibration_controls(self):
        """Programmatically initialize the Spectrometer & Shaper Calibration tab canvases & signals."""

        det1_layout = getattr(self, "cal_det1_layout", None)
        det2_layout = getattr(self, "cal_det2_layout", None)
        pump_layout = getattr(self, "cal_pump_layout", None)

        from pymorgan.gui.cal_canvas import CalibrationCanvasWidget

        if det1_layout is not None and not hasattr(self, "cal_canvas_det1"):
            self.cal_canvas_det1 = CalibrationCanvasWidget(self, detector_name="Detector 1")
            self.cal_canvas_det1.hide()  # Hidden until data is loaded
            det1_layout.addWidget(self.cal_canvas_det1)

        if det2_layout is not None and not hasattr(self, "cal_canvas_det2"):
            self.cal_canvas_det2 = CalibrationCanvasWidget(self, detector_name="Detector 2")
            self.cal_canvas_det2.hide()  # Hidden until data is loaded
            det2_layout.addWidget(self.cal_canvas_det2)

        if pump_layout is not None and not hasattr(self, "cal_canvas_pump"):
            self.cal_canvas_pump = CalibrationCanvasWidget(self, detector_name="Pump Spectrum")
            self.cal_canvas_pump.hide()  # Hidden until data is loaded
            pump_layout.addWidget(self.cal_canvas_pump)

        # Connect UI file pop-up buttons for Detector 1
        for btn_name, target_ax, title in (
            ("cal_det1_btn_pop_raw", "ax_raw", "Raw Intensities"),
            ("cal_det1_btn_pop_abs", "ax_abs", "Calculated Absorbance"),
            ("cal_det1_btn_pop_ref", "ax_ref", "Reference Fit Overlay"),
            ("cal_det1_btn_pop_fit", "ax_fit", "Calibration Fit Curve"),
        ):
            btn = getattr(self, btn_name, None)
            if btn is not None and hasattr(self, "cal_canvas_det1"):
                ax = getattr(self.cal_canvas_det1, target_ax)
                btn.clicked.connect(lambda _, a=ax, t=title: self.cal_canvas_det1._popup_subplot(a, t))

        # Connect UI file pop-up buttons for Detector 2
        for btn_name, target_ax, title in (
            ("cal_det2_btn_pop_raw", "ax_raw", "Raw Intensities"),
            ("cal_det2_btn_pop_abs", "ax_abs", "Calculated Absorbance"),
            ("cal_det2_btn_pop_ref", "ax_ref", "Reference Fit Overlay"),
            ("cal_det2_btn_pop_fit", "ax_fit", "Calibration Fit Curve"),
        ):
            btn = getattr(self, btn_name, None)
            if btn is not None and hasattr(self, "cal_canvas_det2"):
                ax = getattr(self.cal_canvas_det2, target_ax)
                btn.clicked.connect(lambda _, a=ax, t=title: self.cal_canvas_det2._popup_subplot(a, t))

        # Connect UI file pop-up buttons for Pump Spectrum
        for btn_name, target_ax, title in (
            ("cal_pump_btn_pop_raw", "ax_raw", "Raw Intensities"),
            ("cal_pump_btn_pop_abs", "ax_abs", "Calculated Absorbance"),
            ("cal_pump_btn_pop_ref", "ax_ref", "Reference Fit Overlay"),
            ("cal_pump_btn_pop_fit", "ax_fit", "Calibration Fit Curve"),
        ):
            btn = getattr(self, btn_name, None)
            if btn is not None and hasattr(self, "cal_canvas_pump"):
                ax = getattr(self.cal_canvas_pump, target_ax)
                btn.clicked.connect(lambda _, a=ax, t=title: self.cal_canvas_pump._popup_subplot(a, t))

        if hasattr(self, "cal_combo_unit_mode") and self.cal_combo_unit_mode is not None:
            self.cal_combo_unit_mode.setCurrentIndex(0)
            self.cal_combo_unit_mode.currentIndexChanged.connect(self._on_cal_unit_mode_changed)

        if hasattr(self, "cal_chk_do_baseline") and self.cal_chk_do_baseline is not None:
            self.cal_chk_do_baseline.toggled.connect(self._on_cal_baseline_toggled)

        if hasattr(self, "cal_type_combo") and self.cal_type_combo is not None:
            default_cal_type = getattr(pm.get_settings(), "default_calibration_datatype", "UniGE TRIR (Intensity)")
            idx = self.cal_type_combo.findText(default_cal_type)
            if idx >= 0:
                self.cal_type_combo.setCurrentIndex(idx)
            self.cal_type_combo.currentIndexChanged.connect(self._on_cal_type_changed)
            self._on_cal_type_changed(self.cal_type_combo.currentIndex())

        if hasattr(self, "cal_spec_btn_load_probe") and self.cal_spec_btn_load_probe is not None:
            self.cal_spec_btn_load_probe.clicked.connect(self._on_cal_load_probe)

        if hasattr(self, "cal_spec_btn_load_solvent") and self.cal_spec_btn_load_solvent is not None:
            self.cal_spec_btn_load_solvent.clicked.connect(self._on_cal_load_solvent)

        if hasattr(self, "cal_spec_btn_load_cal") and self.cal_spec_btn_load_cal is not None:
            self.cal_spec_btn_load_cal.clicked.connect(self._on_cal_load_calref)

        if hasattr(self, "cal_spec_btn_do_fit") and self.cal_spec_btn_do_fit is not None:
            self.cal_spec_btn_do_fit.clicked.connect(self._on_cal_do_fit)

        if hasattr(self, "cal_spec_btn_save") and self.cal_spec_btn_save is not None:
            self.cal_spec_btn_save.clicked.connect(self._on_cal_save)
            self.cal_spec_btn_save.setEnabled(False)

        if hasattr(self, "cal_shaper_btn_save") and self.cal_shaper_btn_save is not None:
            self.cal_shaper_btn_save.setEnabled(False)

        if hasattr(self, "cal_btn_merge") and self.cal_btn_merge is not None:
            self.cal_btn_merge.clicked.connect(self._on_cal_merge)

        if hasattr(self, "cal_btn_split") and self.cal_btn_split is not None:
            self.cal_btn_split.clicked.connect(self._on_cal_split)

        # Wire Pulse Shaper Calibration Buttons
        if hasattr(self, "cal_shaper_btn_load_pump") and self.cal_shaper_btn_load_pump is not None:
            self.cal_shaper_btn_load_pump.clicked.connect(self._on_cal_shaper_load_pump)

        if hasattr(self, "cal_shaper_btn_load_solvent") and self.cal_shaper_btn_load_solvent is not None:
            self.cal_shaper_btn_load_solvent.clicked.connect(self._on_cal_shaper_load_solvent)

        if hasattr(self, "cal_shaper_btn_load_cal") and self.cal_shaper_btn_load_cal is not None:
            self.cal_shaper_btn_load_cal.clicked.connect(self._on_cal_load_calref)

        if hasattr(self, "cal_shaper_btn_load_mask") and self.cal_shaper_btn_load_mask is not None:
            self.cal_shaper_btn_load_mask.clicked.connect(self._on_cal_shaper_load_mask)

        if hasattr(self, "cal_shaper_btn_do_fit") and self.cal_shaper_btn_do_fit is not None:
            self.cal_shaper_btn_do_fit.clicked.connect(self._on_cal_shaper_do_fit)

        if hasattr(self, "cal_shaper_btn_save") and self.cal_shaper_btn_save is not None:
            self.cal_shaper_btn_save.clicked.connect(self._on_cal_shaper_save)
            self.cal_shaper_btn_save.setEnabled(False)

        if hasattr(self, "cal_spec_btn_reset") and self.cal_spec_btn_reset is not None:
            self.cal_spec_btn_reset.clicked.connect(self._on_cal_reset)

        if hasattr(self, "cal_shaper_btn_reset") and self.cal_shaper_btn_reset is not None:
            self.cal_shaper_btn_reset.clicked.connect(self._on_cal_shaper_reset)

    def _on_cal_type_changed(self, index: int):
        """Update Detector 2 tab, fit axis unit mode, and Pulse Shaper group enablement based on selected setup type."""
        combo = getattr(self, "cal_type_combo", None)
        text = combo.itemText(index) if combo is not None and index >= 0 else ""
        cal_type_code = index + 1

        # Default fit axis unit to Wavelength (nm) (index 0) for ALL data types
        unit_combo = getattr(self, "cal_combo_unit_mode", None)
        if unit_combo is not None:
            if unit_combo.currentIndex() != 0:
                unit_combo.setCurrentIndex(0)

        # 2-Detector setups: UoS TRIR, RAL LIFEtime (Absorbance), RAL LIFEtime (Intensity)
        is_two_detector = text in (
            "UoS TRIR",
            "RAL LIFEtime (Absorbance)",
            "RAL LIFEtime (Intensity)",
        )

        # Setups supporting pulse shaper (UniGE TRIR, UoS, RAL)
        has_shaper = text in (
            "UoS TRIR",
            "RAL LIFEtime (Absorbance)",
            "RAL LIFEtime (Intensity)",
            "UniGE TRIR (Intensity)",
            "UniGE TRIR (Absorbance)",
        )

        # Update Detector 2 tab enablement & visibility
        tab_widget = getattr(self, "cal_detector_tabWidget", None)
        if tab_widget is not None:
            if tab_widget.count() > 1:
                tab_widget.setTabVisible(1, is_two_detector)
                tab_widget.setTabEnabled(1, is_two_detector)
                if not is_two_detector and tab_widget.currentIndex() == 1:
                    tab_widget.setCurrentIndex(0)
            if tab_widget.count() > 2:
                tab_widget.setTabVisible(2, has_shaper)
                tab_widget.setTabEnabled(2, has_shaper)
                if not has_shaper and tab_widget.currentIndex() == 2:
                    tab_widget.setCurrentIndex(0)

        # Update Pulse Shaper group enablement & visibility
        shaper_group = getattr(self, "cal_shaper_group", None)
        if shaper_group is not None:
            shaper_group.setEnabled(has_shaper)
            shaper_group.setVisible(has_shaper)

        pm.update_settings(default_calibration_datatype=text)

        # Update Grating Model combo box enablement & default selection
        model_combo = getattr(self, "cal_combo_grating_model", None)
        if model_combo is not None:
            is_prism = text == "UniGE NIR-TA"
            model_combo.setEnabled(not is_prism)
            if is_prism:
                model_combo.setToolTip("NIR-TA uses the SF10 Prism Sellmeier dispersion model")
            else:
                model_combo.setToolTip("Select grating dispersion polynomial model degree")
                default_idx = 0 if is_prism else 1
                model_combo.setCurrentIndex(default_idx)



        # Pre-fill Fit Settings tab widgets according to selected setup type
        from pymorgan.cal import get_default_calibration_params
        p_def = get_default_calibration_params(cal_type_code)

        if hasattr(self, "cal_set_cwl") and self.cal_set_cwl is not None:
            self.cal_set_cwl.setValue(round(float(p_def["cwl"]), 2))
            self.cal_set_cwl.setToolTip("Central Wavelength (nm) or Wavenumber (cm⁻¹) setting of the spectrograph.")
        if hasattr(self, "cal_set_ppnm") and self.cal_set_ppnm is not None:
            self.cal_set_ppnm.setValue(p_def["ppnm_guess"])
            self.cal_set_ppnm.setToolTip("Initial estimate for spectral dispersion pitch (pixels per nm or pixels per cm⁻¹).")
        if hasattr(self, "cal_set_rel_min_nm") and self.cal_set_rel_min_nm is not None:
            self.cal_set_rel_min_nm.setValue(p_def["rel_min_nm"])
            self.cal_set_rel_min_nm.setToolTip("Lower relative wavelength cut bound (nm) relative to central wavelength λ₀.")
        if hasattr(self, "cal_set_rel_max_nm") and self.cal_set_rel_max_nm is not None:
            self.cal_set_rel_max_nm.setValue(p_def["rel_max_nm"])
            self.cal_set_rel_max_nm.setToolTip("Upper relative wavelength cut bound (nm) relative to central wavelength λ₀.")
        if hasattr(self, "cal_set_grating_model") and self.cal_set_grating_model is not None:
            deg = p_def.get("grating_degree", 2)
            self.cal_set_grating_model.setCurrentIndex(deg - 1)
            self.cal_set_grating_model.setToolTip("Polynomial degree for grating dispersion model (1=Linear, 2=Quadratic, 3=Cubic).")
        if hasattr(self, "cal_set_deriv_mode") and self.cal_set_deriv_mode is not None:
            self.cal_set_deriv_mode.setCurrentIndex(1)  # Default to 1st Derivative (dY/dx)
            self.cal_set_deriv_mode.setToolTip("Derivative fitting mode (Standard, 1st Derivative, 2nd Derivative) to eliminate baseline drift.")
        if hasattr(self, "cal_set_savgol_win") and self.cal_set_savgol_win is not None:
            self.cal_set_savgol_win.setToolTip("Window length (odd integer >= 5) for Savitzky-Golay derivative filter.")
        if hasattr(self, "cal_set_cross_corr") and self.cal_set_cross_corr is not None:
            self.cal_set_cross_corr.setChecked(p_def.get("use_cross_corr", False))
            self.cal_set_cross_corr.setToolTip("Pre-align spectrum peaks via 1D cross-correlation to estimate optimal initial guess.")
        if hasattr(self, "cal_set_subtract_ref_baseline") and self.cal_set_subtract_ref_baseline is not None:
            self.cal_set_subtract_ref_baseline.setToolTip("Subtract fitted polynomial baseline from reference spectrum before overlay comparison.")
        if hasattr(self, "cal_set_convolve_ref") and self.cal_set_convolve_ref is not None:
            self.cal_set_convolve_ref.setToolTip("Refine fit by convolving reference spectrum with Gaussian instrument response function.")

    def _get_cal_dialog_filter(self) -> tuple[str, str]:
        """Return (file_filter, default_pattern) sensitive to selected calibration setup type."""
        combo = getattr(self, "cal_type_combo", None)
        text = combo.currentText() if combo is not None else ""

        specs = {
            "UoS TRIR": ("UoS TRIR files (*.2D);;All Files (*.*)", ""),
            "UniGE fsTA": ("UniGE fsTA files (*.dat);;All Files (*.*)", ""),
            "UniGE nsTA": ("UniGE nsTA files (*.dat);;All Files (*.*)", ""),
            "UZH Lab 2": ("UZH Lab 2 files (*.csv);;All Files (*.*)", "*ds0_intensity*.csv"),
            "UniGE NIR-TA": ("UniGE NIR-TA files (*.dat);;All Files (*.*)", ""),
            "RAL LIFEtime (Absorbance)": ("RAL LIFEtime files (*.csv);;All Files (*.*)", ""),
            "RAL LIFEtime (Intensity)": ("RAL LIFEtime files (*.csv);;All Files (*.*)", "*cycle*.csv"),
            "UniGE TRIR (Intensity)": ("UniGE TRIR files (*.csv);;All Files (*.*)", "*ds0_intensity*.csv"),
            "UniGE TRIR (Absorbance)": ("UniGE TRIR files (*.csv);;All Files (*.*)", "*ds0_intensity*.csv"),
            "UniGE TRUVIS-II (Intensity)": ("UniGE TRUVIS-II files (*.csv);;All Files (*.*)", "*ds0_intensity*.csv"),
        }
        return specs.get(text, ("Data Files (*.dat *.csv *.2D *.txt);;All Files (*.*)", ""))

    def _update_cal_raw_plots(self):
        """Update Top-Left raw intensities plot with Probe (black) and Svt/Filt (red) spectra."""
        exp_data = getattr(self, "_cal_exp_data", None)
        svt_data = getattr(self, "_cal_svt_data", None)

        canvases = [
            (getattr(self, "cal_canvas_det1", None), 0, "Detector 1"),
            (getattr(self, "cal_canvas_det2", None), 1, "Detector 2"),
        ]

        for canvas, det_idx, _det_name in canvases:
            if canvas is None:
                continue

            ax = canvas.ax_raw
            ax.clear()
            ax.set_title("")

            has_probe = exp_data is not None and len(exp_data.detector_data) > det_idx
            has_svt = svt_data is not None and len(svt_data.detector_data) > det_idx

            if has_probe or has_svt:
                canvas.show()
            else:
                canvas.hide()

            if has_probe:
                ax.plot(exp_data.detector_data[det_idx], color="black", linewidth=1.2, label="Probe")

            if has_svt:
                ax.plot(svt_data.detector_data[det_idx], color="red", linewidth=1.2, label="Solvent")

            ax.grid(False)
            ax.axhline(0, color="0.75", lw=0.75, zorder=0)
            ax.set_xlabel("Pixel", fontweight="bold", fontsize=8)
            ax.set_ylabel("Counts", fontweight="bold", fontsize=8)
            ax.tick_params(labelbottom=True)
            canvas.make_draggable_legend(ax, title="Raw Intensities")
            canvas.draw()

    def _on_cal_unit_mode_changed(self, index: int):
        mode = "nm" if index == 0 else "cm-1"
        if hasattr(self, "cal_canvas_det1"):
            self.cal_canvas_det1.unit_mode = mode
        if hasattr(self, "cal_canvas_det2"):
            self.cal_canvas_det2.unit_mode = mode

        res = getattr(self, "_cal_fit_result_det1", None)
        if res is not None:
            self._update_cal_fit_plot_det1(res)

    def _on_cal_baseline_toggled(self, checked: bool):
        """Redraw spectrometer and shaper plots when global baseline correction checkbox is toggled."""
        res_det1 = getattr(self, "_cal_fit_result_det1", None)
        if res_det1 is not None:
            self._update_cal_fit_plot_det1(res_det1)
        res_det2 = getattr(self, "_cal_fit_result_det2", None)
        if res_det2 is not None and hasattr(self, "_update_cal_fit_plot_det2"):
            self._update_cal_fit_plot_det2(res_det2)
        if hasattr(self, "_update_shaper_plots"):
            self._update_shaper_plots()

    def _update_cal_fit_plot_det1(self, res):
        """Update ax_fit (Bottom-Right) and ax_ref (Top-Right) without erasing 1st column (ax_raw / ax_abs)."""
        if not hasattr(self, "cal_canvas_det1"):
            return

        unit_idx = self.cal_combo_unit_mode.currentIndex() if hasattr(self, "cal_combo_unit_mode") else 0
        fit_unit_mode = "nm" if unit_idx == 0 else "cm-1"
        cal_type_idx = self.cal_type_combo.currentIndex() + 1 if hasattr(self, "cal_type_combo") else 2
        is_ir = cal_type_idx in (1, 4, 6, 7, 8, 9)

        exp_data = getattr(self, "_cal_exp_data", None)
        n_pix = len(exp_data.detector_data[0]) if (exp_data and len(exp_data.detector_data) > 0) else len(res.wavelength_nm)
        pix = np.arange(1, n_pix + 1, dtype=float)

        # Update calibration fit plot (Bottom-Right ax_fit) - Y axis unit depends on "fit axis unit"
        ax = self.cal_canvas_det1.ax_fit
        ax.clear()
        ax.set_title("")

        if fit_unit_mode == "cm-1":
            fit_y = res.wavenumber_cm1
            fit_unit_label = r"Wavenumber ($\mathrm{cm}^{-1}$)"
        else:
            fit_y = res.wavelength_nm
            fit_unit_label = "Wavelength (nm)"

        # Format short notation legend for calibration fit parameters
        p0_val = res.fit_params[0] if len(res.fit_params) > 0 else 0.0
        ppnm_val = res.fit_params[1] if len(res.fit_params) > 1 else 0.0
        nm_per_pix = (1.0 / ppnm_val) if ppnm_val > 0 else 0.0

        fit_lbl = rf"Fit ($\lambda_{0}$={p0_val:.1f} nm, {nm_per_pix:.2f} nm/px"
        if len(res.fit_params) > 6 and abs(res.fit_params[6]) > 1e-4:
            fit_lbl += rf", $c_{2}$={res.fit_params[6]:.2f}"
        if len(res.fit_params) > 7 and abs(res.fit_params[7]) > 1e-4:
            fit_lbl += rf", $c_{3}$={res.fit_params[7]:.2f}"
        fit_lbl += ")"

        # Linear reference curve connecting first to last point of plot (zorder=1)
        ax.plot([pix[0], pix[-1]], [fit_y[0], fit_y[-1]], color="0.5", lw=1.0, linestyle="--", label="Linear Ref", zorder=1)

        # Plot fitted dispersion curve with alpha=0.5 (zorder=2)
        ax.plot(pix, fit_y, color="tab:green", alpha=0.5, linewidth=1.8, label=fit_lbl, zorder=2)

        # Plot current probe calibration as a dashed line (zorder=3)
        curr_probe = self._get_current_probe_calibration()
        if curr_probe is not None and len(curr_probe) > 0:
            pix_curr = np.arange(1, len(curr_probe) + 1, dtype=float)
            if fit_unit_mode == "cm-1" and np.mean(curr_probe) < 1000:
                curr_plot = 1e7 / np.clip(curr_probe, 1e-3, None)
            elif fit_unit_mode == "nm" and np.mean(curr_probe) > 1000:
                curr_plot = 1e7 / np.clip(curr_probe, 1e-3, None)
            else:
                curr_plot = curr_probe
            ax.plot(
                pix_curr,
                curr_plot,
                color="tab:blue",
                linestyle="--",
                linewidth=1.8,
                label="Current Probe Cal",
                zorder=3,
            )

        # Crosshair lines at central pixel and central wavelength/wavenumber
        cx = (1.0 + len(pix)) / 2.0
        cy = fit_y[len(fit_y) // 2]
        ax.axvline(cx, color="0.75", lw=0.75, ls=":", zorder=0)
        ax.axhline(cy, color="0.75", lw=0.75, ls=":", zorder=0)

        ax.grid(False)
        ax.set_xlabel("Pixel", fontweight="bold", fontsize=8)
        ax.set_ylabel(fit_unit_label, fontweight="bold", fontsize=8)
        self.cal_canvas_det1.make_draggable_legend(ax, title="Calibration Fit")

        # Update Reference Overlay plot (Top-Right ax_ref)
        ax_ref_plot = self.cal_canvas_det1.ax_ref
        ax_ref_plot.clear()
        # Remove any previous twin axis
        for ax_twin in list(ax_ref_plot.figure.axes):
            if ax_twin is not ax_ref_plot and ax_twin.get_label() == "_ref_twin":
                ax_twin.remove()
        ax_ref_plot.set_title("")

        ref_axis_label = r"Wavenumber ($\mathrm{cm}^{-1}$)" if is_ir else "Wavelength (nm)"

        ref_x_full = getattr(res, "ref_x_full", None)
        ref_y_full = getattr(res, "ref_y_full", None)
        ref_x_cut = res.ref_x_cut
        ref_y_cut = res.ref_y_cut

        # Check subtract baseline from reference checkbox state
        subtract_ref_baseline = False
        if hasattr(self, "cal_set_subtract_ref_baseline") and self.cal_set_subtract_ref_baseline is not None:
            subtract_ref_baseline = self.cal_set_subtract_ref_baseline.isChecked()

        do_baseline = self.cal_chk_do_baseline.isChecked() if hasattr(self, "cal_chk_do_baseline") else True

        # Experimental curve: baseline-corrected if do_baseline is checked, otherwise raw model fit
        exp_y_source = (res.exp_corr_y_fit if do_baseline else res.model_y_fit) if res.exp_corr_y_fit is not None else res.model_y_fit

        # Reference curve: baseline-corrected only if checkbox is ticked
        if subtract_ref_baseline and res.ref_y_corr is not None:
            ref_y_source = res.ref_y_corr
        else:
            ref_y_source = ref_y_cut

        sort_cut = np.argsort(ref_x_cut)
        ref_x_cut_plot = ref_x_cut[sort_cut]
        ref_y_cut_plot = ref_y_source[sort_cut]
        model_y_fit_plot = exp_y_source[sort_cut]
        conv_y_plot = res.convolved_ref_y[sort_cut] if (res.convolved_ref_y is not None and not subtract_ref_baseline) else None

        ref_label = "Ref (BC)" if subtract_ref_baseline else "Raw Ref"
        exp_label = "Exp (BC)" if do_baseline else "Exp"

        # --- Dual y-axis: left = reference, right = experimental (BC) ---
        # Create twin axis for experimental data
        ax_exp = ax_ref_plot.twinx()
        ax_exp.set_label("_ref_twin")

        # 1. Full reference spectrum in fainter gray line on LEFT axis
        if ref_x_full is not None and ref_y_full is not None and len(ref_x_full) > 0:
            sort_full = np.argsort(ref_x_full)
            ax_ref_plot.plot(
                ref_x_full[sort_full],
                ref_y_full[sort_full],
                color="0.65",
                linewidth=0.9,
                alpha=0.55,
                linestyle="-",
                label="Full Ref",
                zorder=1,
            )

        # 2. Active reference in solid black on LEFT axis
        ax_ref_plot.plot(ref_x_cut_plot, ref_y_cut_plot, color="black", linewidth=1.4,
                         alpha=1.0, linestyle="-", label=ref_label, zorder=2)
        if conv_y_plot is not None:
            ax_ref_plot.plot(ref_x_cut_plot, conv_y_plot, color="red", linewidth=1.8,
                             alpha=0.6, linestyle="-", label=rf"Conv Ref ($\sigma$={res.sigma_opt:.2f})", zorder=4)

        # 3. Baseline-corrected experimental on RIGHT axis (purple, translucent)
        ax_exp.plot(ref_x_cut_plot, model_y_fit_plot, color="tab:purple", alpha=0.7,
                    linewidth=1.8, label=exp_label, zorder=3)

        # --- Align both y-axes strictly based on ACTIVE plot region data ---
        ref_min, ref_max = float(np.min(ref_y_cut_plot)), float(np.max(ref_y_cut_plot))
        exp_min, exp_max = float(np.min(model_y_fit_plot)), float(np.max(model_y_fit_plot))

        if abs(ref_max - ref_min) < 1e-6:
            ref_min -= 0.05
            ref_max += 0.05
        if abs(exp_max - exp_min) < 1e-6:
            exp_min -= 0.05
            exp_max += 0.05

        ax_ref_plot.set_ylim(ref_min, ref_max)
        ax_exp.set_ylim(exp_min, exp_max)

        align.yaxes(ax_ref_plot, 0.0, ax_exp, 0.0, pos=0.05)

        ax_ref_plot.grid(False)
        ax_ref_plot.axhline(0, color="0.75", lw=0.75, zorder=0)
        ax_ref_plot.set_xlim(np.min(ref_x_cut_plot), np.max(ref_x_cut_plot))
        ax_ref_plot.set_xlabel(ref_axis_label, fontweight="bold", fontsize=8)
        ax_ref_plot.set_ylabel("Absorbance (Ref)", fontweight="bold", fontsize=8, color="black")
        ax_ref_plot.tick_params(axis="y", labelsize=8, labelcolor="black")
        ax_exp.set_ylabel(f"Absorbance ({exp_label})", fontweight="bold", fontsize=8, color="tab:purple")
        ax_exp.tick_params(axis="y", labelsize=8, labelcolor="tab:purple")

        # Merge legends from both axes
        lines1, labels1 = ax_ref_plot.get_legend_handles_labels()
        lines2, labels2 = ax_exp.get_legend_handles_labels()
        leg = ax_ref_plot.legend(lines1 + lines2, labels1 + labels2, fontsize=7,
                                 loc="best", framealpha=0.85)
        if leg is not None:
            leg.set_draggable(True)

        # Update Subplot 2 (ax_abs): show absorbance with baseline dashed line
        # over the ACTIVE fitting range only (mapped from ref_x_cut via wavelength axis)
        ax_abs = self.cal_canvas_det1.ax_abs
        svt_data = getattr(self, "_cal_svt_data", None)
        if exp_data is not None and svt_data is not None and len(exp_data.detector_data) > 0 and len(svt_data.detector_data) > 0:
            i_probe = exp_data.detector_data[0]
            i_svt = svt_data.detector_data[0]
            valid = (i_svt > 0) & (i_probe > 0)
            abs_y = np.zeros_like(i_probe)
            abs_y[valid] = -np.log10(i_svt[valid] / i_probe[valid])

            ax_abs.clear()
            ax_abs.plot(pix, abs_y, color="tab:purple", linewidth=1.8, label="Abs", zorder=1)

            # Baseline is defined on the active ref_x_cut wavelength grid in reference units.
            # model = ((meas - meas_min) / meas_span) * scale + baseline
            # =>  baseline_phys = meas_min - baseline_ref * (meas_span / scale)
            if do_baseline and res.baseline_fit is not None and len(res.wavelength_nm) == n_pix:
                scale = float(res.fit_params[2]) if len(res.fit_params) > 2 else 1.0
                if abs(scale) < 1e-6:
                    scale = 1.0

                meas_min = float(np.min(abs_y))
                meas_span = float(np.max(abs_y) - meas_min)

                if is_ir:
                    ref_wl_cut_nm = 1e7 / np.clip(res.ref_x_cut, 1.0, None)
                else:
                    ref_wl_cut_nm = res.ref_x_cut

                # Active pixel range: pixels whose calibrated wavelength falls in ref_x_cut span
                wl_min_active = np.min(ref_wl_cut_nm)
                wl_max_active = np.max(ref_wl_cut_nm)
                active_pix_mask = (res.wavelength_nm >= wl_min_active) & (res.wavelength_nm <= wl_max_active)

                if np.any(active_pix_mask):
                    # Interpolate baseline (reference units) onto active pixel wavelengths
                    sort_ref = np.argsort(ref_wl_cut_nm)
                    baseline_ref_units = np.interp(
                        res.wavelength_nm[active_pix_mask],
                        ref_wl_cut_nm[sort_ref],
                        res.baseline_fit[sort_ref],
                    )
                    # Convert to physical measurement absorbance units (OD)
                    baseline_meas = meas_min - (baseline_ref_units * (meas_span / scale))

                    ax_abs.plot(
                        pix[active_pix_mask],
                        baseline_meas,
                        color="tab:orange", linestyle="--", linewidth=1.4,
                        label="Baseline (active)", zorder=2,
                    )

            ax_abs.grid(False)
            ax_abs.axhline(0, color="0.75", lw=0.75, zorder=0)
            ax_abs.set_xlabel("Pixel", fontweight="bold", fontsize=8)
            ax_abs.set_ylabel("Absorbance (OD)", fontweight="bold", fontsize=8)
            self.cal_canvas_det1.make_draggable_legend(ax_abs, title="Calculated Absorbance")

        if not getattr(self.cal_canvas_det1, "_tight_layout_done", False):
            import contextlib
            with contextlib.suppress(Exception):
                self.cal_canvas_det1.figure.tight_layout()
            self.cal_canvas_det1._tight_layout_done = True
        self.cal_canvas_det1.draw()

    def _on_cal_load_probe(self):
        """Load probe spectrum measurement file and update LED state."""

        from pymorgan.cal import load_experimental_spectrum

        cal_type_idx = self.cal_type_combo.currentIndex() + 1 if hasattr(self, "cal_type_combo") else 2
        filter_str, def_pattern = self._get_cal_dialog_filter()
        initial_dir = self._rootdir_text() or getattr(self, "rootdir", "")
        initial_path = str(Path(initial_dir) / def_pattern) if (initial_dir and def_pattern) else initial_dir

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Probe Measurement File",
            initial_path,
            filter_str,
        )
        if not file_path:
            return

        led = getattr(self, "cal_spec_led_probe", None)
        if led:
            led.set_state("loading")

        try:
            exp_data = load_experimental_spectrum(file_path, cal_type_idx)
            self._cal_exp_data = exp_data
            if led:
                led.set_state("loaded")
            self.statusBar().showMessage(f"Loaded Probe Spectrum: {Path(file_path).name}")

            # Update Fit Settings tab with parsed metadata
            is_ir = cal_type_idx in (1, 4, 6, 7, 8, 9)
            if hasattr(self, "cal_set_cwl") and exp_data.cwl and exp_data.cwl[0] > 0:
                cwl_val = exp_data.cwl[0]
                self.cal_set_cwl.setValue(round(float(cwl_val), 2))
            if hasattr(self, "cal_set_ppnm") and exp_data.gratings and exp_data.gratings[0] > 0 and cal_type_idx != 10:
                g_val = exp_data.gratings[0]
                ppnm = (g_val / 500.0) if is_ir else (1.1 * g_val / 150.0)
                self.cal_set_ppnm.setValue(ppnm)
                if cal_type_idx in (8, 9) and abs(g_val - 150) < 1e-3:
                    if hasattr(self, "cal_set_rel_min_nm"):
                        self.cal_set_rel_min_nm.setValue(-200.0)
                    if hasattr(self, "cal_set_rel_max_nm"):
                        self.cal_set_rel_max_nm.setValue(200.0)

            # Plot raw intensities (Probe in black, Svt in red)
            self._update_cal_raw_plots()

        except Exception as err:
            if led:
                led.set_state("error")
            self.statusBar().showMessage(f"Error loading probe spectrum: {err}")

    def _on_cal_load_solvent(self):
        """Load solvent reference spectrum file and update LED state."""

        from pymorgan.cal import load_experimental_spectrum

        cal_type_idx = self.cal_type_combo.currentIndex() + 1 if hasattr(self, "cal_type_combo") else 2
        filter_str, def_pattern = self._get_cal_dialog_filter()
        initial_dir = self._rootdir_text() or getattr(self, "rootdir", "")
        initial_path = str(Path(initial_dir) / def_pattern) if (initial_dir and def_pattern) else initial_dir

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Solvent / Reference File",
            initial_path,
            filter_str,
        )
        if not file_path:
            return

        led = getattr(self, "cal_spec_led_solvent", None)
        if led:
            led.set_state("loading")

        try:
            svt_data = load_experimental_spectrum(file_path, cal_type_idx)
            self._cal_svt_data = svt_data
            if led:
                led.set_state("loaded")
            self.statusBar().showMessage(f"Loaded Solvent Spectrum: {Path(file_path).name}")

            # Plot raw intensities (Probe in black, Svt in red)
            self._update_cal_raw_plots()

            # Plot calculated absorbance A = -log10(I_probe / I_svt) if probe loaded
            exp_data = getattr(self, "_cal_exp_data", None)
            if exp_data is not None and len(exp_data.detector_data) > 0 and len(svt_data.detector_data) > 0:
                i_probe = exp_data.detector_data[0]
                i_svt = svt_data.detector_data[0]
                valid = (i_svt > 0) & (i_probe > 0)
                abs_y = np.zeros_like(i_probe)
                # Correct absorbance equation: A = -log10(I / I0) = -log10(i_svt / i_probe)
                abs_y[valid] = -np.log10(i_svt[valid] / i_probe[valid])

                if hasattr(self, "cal_canvas_det1"):
                    ax = self.cal_canvas_det1.ax_abs
                    ax.clear()
                    ax.set_title("")
                    ax.plot(abs_y, color="tab:purple", label="Calculated Abs")
                    ax.grid(False)
                    ax.axhline(0, color="0.75", lw=0.75, zorder=0)
                    ax.set_xlabel("Pixel", fontweight="bold", fontsize=8)
                    ax.set_ylabel("Absorbance (OD)", fontweight="bold", fontsize=8)
                    self.cal_canvas_det1.make_draggable_legend(ax, title="Calculated Absorbance")
                    self.cal_canvas_det1.draw()

        except Exception as err:
            if led:
                led.set_state("error")
            self.statusBar().showMessage(f"Error loading solvent spectrum: {err}")

    def _on_cal_load_calref(self):
        """Load calibration reference spectrum CSV and update LED state."""

        from pymorgan.cal import load_reference_spectrum

        ref_dir = Path(__file__).resolve().parent.parent.parent / "cal" / "ref_spectra"
        initial_dir = str(ref_dir) if ref_dir.exists() else getattr(self, "rootdir", "")

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Calibration Reference Spectrum CSV",
            initial_dir,
            "CSV Files (*.csv *.CSV);;Text Files (*.txt *.dat *.tsv);;All Files (*.*)",
        )
        if not file_path:
            return

        for led_name in ("cal_spec_led_cal", "cal_shaper_led_cal"):
            led = getattr(self, led_name, None)
            if led:
                led.set_state("loading")

        try:
            ref_spec = load_reference_spectrum(file_path)
            self._cal_ref_spec = ref_spec
            for led_name in ("cal_spec_led_cal", "cal_shaper_led_cal"):
                led = getattr(self, led_name, None)
                if led:
                    led.set_state("loaded")
            self.statusBar().showMessage(f"Loaded Reference Spectrum: {ref_spec.name}")

            if hasattr(self, "cal_canvas_det1"):
                ax = self.cal_canvas_det1.ax_ref
                ax.clear()
                ax.set_title("")
                ax.plot(ref_spec.spectral_axis, ref_spec.absorbance, color="black", label=ref_spec.name)
                ax.grid(False)
                ax.axhline(0, color="0.75", lw=0.75, zorder=0)
                ax.set_xlabel(f"Spectral Axis ({ref_spec.axis_unit})", fontweight="bold", fontsize=8)
                ax.set_ylabel("Norm. Absorbance", fontweight="bold", fontsize=8)
                self.cal_canvas_det1.make_draggable_legend(ax, title=f"Reference Spectrum ({ref_spec.axis_unit})")
                self.cal_canvas_det1.draw()

            if hasattr(self, "cal_canvas_pump"):
                self._update_shaper_plots()

        except Exception as err:
            for led_name in ("cal_spec_led_cal", "cal_shaper_led_cal"):
                led = getattr(self, led_name, None)
                if led:
                    led.set_state("error")
            self.statusBar().showMessage(f"Error loading calibration reference: {err}")

    @busy_guard("Fitting the calibration...")
    def _on_cal_do_fit(self):
        """Execute non-linear least squares fit to map pixels to wavelength/wavenumber axis."""
        from pymorgan.cal import fit_wavelength_axis

        exp_data = getattr(self, "_cal_exp_data", None)
        ref_spec = getattr(self, "_cal_ref_spec", None)

        if exp_data is None or len(exp_data.detector_data) == 0:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Fit Calibration", "Please load a probe measurement spectrum first.")
            return

        cal_type_idx = self.cal_type_combo.currentIndex() + 1 if hasattr(self, "cal_type_combo") else 2

        if ref_spec is None:
            ref_dir = Path(__file__).resolve().parent.parent.parent / "cal" / "ref_spectra"
            holmium_ref = ref_dir / "UVVis-Holmium.csv"
            if cal_type_idx in (2, 3, 10, 11) and holmium_ref.exists():
                from pymorgan.cal import load_reference_spectrum
                ref_spec = load_reference_spectrum(holmium_ref)
                self._cal_ref_spec = ref_spec

        if ref_spec is None:
            # Fallback synthetic reference spectrum if none loaded
            ref_x = np.linspace(350, 740, 500)
            ref_y = np.sin(np.linspace(0, np.pi, 500)) ** 2
        else:
            ref_x = ref_spec.spectral_axis
            ref_y = ref_spec.absorbance

        from pymorgan.cal import get_default_calibration_params
        p_def = get_default_calibration_params(cal_type_idx)

        # Read fitting parameters directly from Fit Settings sub-tab widgets
        cwl_val = self.cal_set_cwl.value() if hasattr(self, "cal_set_cwl") else p_def["cwl"]
        ppnm_val = self.cal_set_ppnm.value() if hasattr(self, "cal_set_ppnm") else p_def["ppnm_guess"]
        rel_min_val = self.cal_set_rel_min_nm.value() if hasattr(self, "cal_set_rel_min_nm") else p_def["rel_min_nm"]
        rel_max_val = self.cal_set_rel_max_nm.value() if hasattr(self, "cal_set_rel_max_nm") else p_def["rel_max_nm"]

        deg_idx = (self.cal_set_grating_model.currentIndex() + 1) if hasattr(self, "cal_set_grating_model") else p_def.get("grating_degree", 1)
        use_corr = self.cal_set_cross_corr.isChecked() if hasattr(self, "cal_set_cross_corr") else False
        deriv_idx = self.cal_set_deriv_mode.currentIndex() if hasattr(self, "cal_set_deriv_mode") else 0
        deriv_mode = "none" if deriv_idx == 0 else ("1st" if deriv_idx == 1 else "2nd")
        savgol_win = self.cal_set_savgol_win.value() if hasattr(self, "cal_set_savgol_win") else 9
        do_convolve = self.cal_set_convolve_ref.isChecked() if hasattr(self, "cal_set_convolve_ref") else False
        do_fit_val = self.cal_chk_do_fit.isChecked() if hasattr(self, "cal_chk_do_fit") else True

        try:
            svt_data = getattr(self, "_cal_svt_data", None)
            if svt_data is not None and len(svt_data.detector_data) > 0 and len(exp_data.detector_data) > 0:
                i_probe = exp_data.detector_data[0]
                i_svt = svt_data.detector_data[0]
                valid = (i_svt > 0) & (i_probe > 0)
                meas_y = np.zeros_like(i_probe)
                meas_y[valid] = -np.log10(i_svt[valid] / i_probe[valid])
            else:
                meas_y = exp_data.detector_data[0]
            res = fit_wavelength_axis(
                meas_y=meas_y,
                ref_x=ref_x,
                ref_y=ref_y,
                cal_type_code=cal_type_idx,
                cwl=cwl_val,
                ppnm_guess=ppnm_val,
                rel_min_nm=rel_min_val,
                rel_max_nm=rel_max_val,
                do_fit=do_fit_val,
                grating_degree=deg_idx,
                use_cross_corr=use_corr,
                use_derivative=deriv_mode,
                savgol_window=savgol_win,
                convolve_ref=do_convolve,
            )
            self._cal_fit_result_det1 = res
            self._update_cal_fit_plot_det1(res)

            if hasattr(self, "cal_spec_btn_save") and self.cal_spec_btn_save is not None:
                self.cal_spec_btn_save.setEnabled(True)

            sigma_msg = f" (Refinement σ={res.sigma_opt:.2f} px)" if do_convolve else ""
            self.statusBar().showMessage(
                f"Calibration fit completed successfully{sigma_msg}. RMS Residual: {np.std(res.residuals):.4e}"
            )

        except Exception as err:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Calibration Fit Error", f"Failed to execute calibration fit:\n{err}")

    def _on_cal_save(self):
        """Export calibrated probe vector to CalibratedProbe.csv."""

        from pymorgan.cal import save_calibration_file

        res1 = getattr(self, "_cal_fit_result_det1", None)
        if res1 is None:
            QMessageBox.warning(self, "Save Calibration", "No fitted calibration data available to save. Run 'Do Automatic Calibration' first.")
            return

        initial_dir = self._rootdir_text() or getattr(self, "rootdir", str(Path.cwd()))
        target_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory for CalibratedProbe.csv",
            initial_dir,
        )
        if not target_dir:
            return

        target_path = Path(target_dir)
        cal_file = target_path / "CalibratedProbe.csv"

        # Only ask what to do when there is already a CalibratedProbe.csv file in the target directory
        if cal_file.exists():
            reply = QMessageBox.question(
                self,
                "Overwrite File?",
                f"The file 'CalibratedProbe.csv' already exists in:\n{target_path}\n\nDo you want to overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        try:
            cal_code = (
                self.cal_type_combo.currentIndex() + 1
                if hasattr(self, "cal_type_combo") and self.cal_type_combo is not None
                else None
            )
            is_ir = cal_code in (1, 4, 6, 7, 8, 9) if cal_code is not None else False
            probe_vector = res1.wavenumber_cm1 if is_ir else res1.wavelength_nm
            primary, _ = save_calibration_file(
                probe_vector, target_path, save_timestamped=False, cal_type_code=cal_code
            )
            QMessageBox.information(
                self,
                "Save Calibration",
                f"Calibrated probe vector saved successfully to:\n{primary.name}",
            )
        except Exception as err:
            QMessageBox.critical(self, "Save Error", f"Failed to save calibration file:\n{err}")

    def _on_cal_merge(self):
        """Merge Detector 1 and Detector 2 calibration vectors into CalibratedProbe.csv."""

        from pymorgan.cal import merge_calibration, save_calibration_file

        f1, _ = QFileDialog.getOpenFileName(self, "Select DETECTOR 1 (LHS) Calibration CSV", getattr(self, "rootdir", ""))
        if not f1:
            return
        f2, _ = QFileDialog.getOpenFileName(self, "Select DETECTOR 2 (RHS) Calibration CSV", getattr(self, "rootdir", ""))
        if not f2:
            return

        try:
            cm1 = np.loadtxt(f1, delimiter=",")
            cm2 = np.loadtxt(f2, delimiter=",")
            merged = merge_calibration(cm1, cm2)

            out_dir = QFileDialog.getExistingDirectory(self, "Save Merged Calibration File To...", getattr(self, "rootdir", ""))
            if not out_dir:
                return

            primary, stamp = save_calibration_file(merged, out_dir)
            QMessageBox.information(self, "Merge Calibration", f"Merged calibration files saved to:\n{primary.name}")
        except Exception as err:
            QMessageBox.critical(self, "Merge Error", f"Failed to merge calibration files:\n{err}")

    def _on_cal_split(self):
        """Split combined calibration vector into LHS and RHS detector files."""

        from pymorgan.cal import save_calibration_file, split_calibration

        f_merged, _ = QFileDialog.getOpenFileName(self, "Select Combined Calibration CSV", getattr(self, "rootdir", ""))
        if not f_merged:
            return

        try:
            arr = np.loadtxt(f_merged, delimiter=",")
            lhs, rhs = split_calibration(arr)

            out_dir = QFileDialog.getExistingDirectory(self, "Save Split Calibration Files To...", getattr(self, "rootdir", ""))
            if not out_dir:
                return

            p_lhs, _ = save_calibration_file(lhs, out_dir, filename="CalibratedProbe_LHS.csv", save_timestamped=False)
            p_rhs, _ = save_calibration_file(rhs, out_dir, filename="CalibratedProbe_RHS.csv", save_timestamped=False)

            QMessageBox.information(self, "Split Calibration", f"Split calibration files saved to:\n{p_lhs.name}\n{p_rhs.name}")
        except Exception as err:
            QMessageBox.critical(self, "Split Error", f"Failed to split calibration file:\n{err}")

    def _on_cal_shaper_load_pump(self):
        """Load Pump Spectrum (Air) file for pulse shaper calibration using experimental spectrum loader."""

        from pymorgan.cal import fit_spectrum_gaussian, load_experimental_spectrum

        cal_type_idx = self.cal_type_combo.currentIndex() + 1 if hasattr(self, "cal_type_combo") else 2
        filter_str, def_pattern = self._get_cal_dialog_filter()
        initial_dir = self._rootdir_text() or getattr(self, "rootdir", "")
        initial_path = str(Path(initial_dir) / def_pattern) if (initial_dir and def_pattern) else initial_dir

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Pump Spectrum (Air) File",
            initial_path,
            filter_str,
        )
        if not file_path:
            return

        led = getattr(self, "cal_shaper_led_pump", None)
        if led:
            led.set_state("loading")

        try:
            exp_data = load_experimental_spectrum(file_path, cal_type_idx)
            y_pump = exp_data.detector_data[0] if len(exp_data.detector_data) > 0 else np.array([])
            x_axis = np.arange(1, len(y_pump) + 1, dtype=float)

            self._cal_shaper_pump_air_path = file_path
            self._cal_shaper_pump_air_x = x_axis
            self._cal_shaper_pump_air_y = y_pump

            # Update Fit Settings tab with parsed metadata (EXACTLY LIKE SPECTROMETER PROBE LOADER)
            is_ir = cal_type_idx in (1, 4, 6, 7, 8, 9)
            if hasattr(self, "cal_set_cwl") and getattr(exp_data, "cwl", None) and len(exp_data.cwl) > 0 and exp_data.cwl[0] > 0:
                cwl_val = exp_data.cwl[0]
                self.cal_set_cwl.setValue(round(float(cwl_val), 2))
            if hasattr(self, "cal_set_ppnm") and getattr(exp_data, "gratings", None) and len(exp_data.gratings) > 0 and exp_data.gratings[0] > 0 and cal_type_idx != 10:
                g_val = exp_data.gratings[0]
                ppnm = (g_val / 500.0) if is_ir else (1.1 * g_val / 150.0)
                self.cal_set_ppnm.setValue(ppnm)
                if cal_type_idx in (8, 9) and abs(g_val - 150) < 1e-3:
                    if hasattr(self, "cal_set_rel_min_nm"):
                        self.cal_set_rel_min_nm.setValue(-200.0)
                    if hasattr(self, "cal_set_rel_max_nm"):
                        self.cal_set_rel_max_nm.setValue(200.0)

            res = fit_spectrum_gaussian(y_pump, x_axis) if len(y_pump) > 0 else None
            self._cal_pump_fit_result = res
            if led:
                led.set_state("loaded")

            self._update_shaper_plots()
            msg = f"Loaded Pump Air Spectrum: w0={res.w0:.2f}, FWHM={res.fwhm:.2f}" if res else f"Loaded Pump Air Spectrum: {Path(file_path).name}"
            self.statusBar().showMessage(msg)

        except Exception as err:
            if led:
                led.set_state("error")
            self.statusBar().showMessage(f"Error loading pump air spectrum: {err}")

    def _on_cal_shaper_load_solvent(self):
        """Load Pump Spectrum (Solvent) file for pulse shaper calibration using experimental spectrum loader."""

        from pymorgan.cal import load_experimental_spectrum

        cal_type_idx = self.cal_type_combo.currentIndex() + 1 if hasattr(self, "cal_type_combo") else 2
        filter_str, def_pattern = self._get_cal_dialog_filter()
        initial_dir = self._rootdir_text() or getattr(self, "rootdir", "")
        initial_path = str(Path(initial_dir) / def_pattern) if (initial_dir and def_pattern) else initial_dir

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Pump Spectrum (Solvent) File",
            initial_path,
            filter_str,
        )
        if not file_path:
            return

        led = getattr(self, "cal_shaper_led_solvent", None)
        if led:
            led.set_state("loading")

        try:
            exp_data = load_experimental_spectrum(file_path, cal_type_idx)
            y_solvent = exp_data.detector_data[0] if len(exp_data.detector_data) > 0 else np.array([])
            self._cal_shaper_pump_solvent_y = y_solvent
            if led:
                led.set_state("loaded")

            self._update_shaper_plots()
            self.statusBar().showMessage(f"Loaded Pump Solvent Spectrum: {Path(file_path).name}")

        except Exception as err:
            if led:
                led.set_state("error")
            self.statusBar().showMessage(f"Error loading pump solvent spectrum: {err}")

    def _on_cal_shaper_load_cal(self):
        """Load Single Mask scan file for pulse shaper calibration using experimental spectrum loader."""

        from pymorgan.cal import load_experimental_spectrum

        cal_type_idx = self.cal_type_combo.currentIndex() + 1 if hasattr(self, "cal_type_combo") else 2
        filter_str, def_pattern = self._get_cal_dialog_filter()
        initial_dir = self._rootdir_text() or getattr(self, "rootdir", "")
        initial_path = str(Path(initial_dir) / def_pattern) if (initial_dir and def_pattern) else initial_dir

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Single Mask Scan File",
            initial_path,
            filter_str,
        )
        if not file_path:
            return

        led = getattr(self, "cal_shaper_led_cal", None)
        if led:
            led.set_state("loading")

        try:
            exp_data = load_experimental_spectrum(file_path, cal_type_idx)
            y_single = exp_data.detector_data[0] if len(exp_data.detector_data) > 0 else np.array([])
            self._cal_shaper_single_mask_y = y_single
            if led:
                led.set_state("loaded")

            self._update_shaper_plots()
            self.statusBar().showMessage(f"Loaded Single Mask Scan: {Path(file_path).name}")

        except Exception as err:
            if led:
                led.set_state("error")
            self.statusBar().showMessage(f"Error loading single mask scan: {err}")

    def _on_cal_shaper_load_mask(self):
        """Load MULTIPLE Mask scan file first, then SINGLE Mask scan file for pulse shaper calibration using experimental spectrum loader."""

        from pymorgan.cal import load_experimental_spectrum

        cal_type_idx = self.cal_type_combo.currentIndex() + 1 if hasattr(self, "cal_type_combo") else 9
        filter_str, def_pattern = self._get_cal_dialog_filter()
        initial_dir = self._rootdir_text() or getattr(self, "rootdir", "")
        initial_path = str(Path(initial_dir) / def_pattern) if (initial_dir and def_pattern) else initial_dir

        # 1. Ask FIRST for the MULTIPLE Mask file
        multi_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select MULTIPLE Mask Scan File",
            initial_path,
            filter_str,
        )
        if not multi_file:
            return

        # 2. Ask THEN for the SINGLE Mask file
        single_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select SINGLE Mask Scan File",
            initial_path,
            filter_str,
        )

        led = getattr(self, "cal_shaper_led_mask", None)
        if led:
            led.set_state("loading")

        try:
            exp_multi = load_experimental_spectrum(multi_file, cal_type_idx)
            self._cal_shaper_multi_mask_y = exp_multi.detector_data[0] if len(exp_multi.detector_data) > 0 else np.array([])
            loaded_names = [f"Multiple Mask ({Path(multi_file).name})"]

            if single_file:
                exp_single = load_experimental_spectrum(single_file, cal_type_idx)
                self._cal_shaper_single_mask_y = exp_single.detector_data[0] if len(exp_single.detector_data) > 0 else np.array([])
                loaded_names.append(f"Single Mask ({Path(single_file).name})")

            if led:
                led.set_state("loaded")

            self._update_shaper_plots()
            self.statusBar().showMessage(f"Loaded Mask Scans: {', '.join(loaded_names)}")

        except Exception as err:
            if led:
                led.set_state("error")
            self.statusBar().showMessage(f"Error loading mask scan(s): {err}")

    @busy_guard("Fitting the shaper calibration...")
    def _on_cal_shaper_do_fit(self):
        """Perform pulse shaper calibration fitting: fit pump absorbance to FTIR reference spectrum (same as probe case), fit Gaussian to calibrated pump, and convert mask scans to nm."""

        from pymorgan.cal import fit_spectrum_gaussian, fit_wavelength_axis, load_reference_spectrum

        y_pump_air = getattr(self, "_cal_shaper_pump_air_y", None)
        y_pump_solv = getattr(self, "_cal_shaper_pump_solvent_y", None)
        y_single = getattr(self, "_cal_shaper_single_mask_y", None)
        y_multi = getattr(self, "_cal_shaper_multi_mask_y", None)

        if y_pump_air is None:
            QMessageBox.warning(self, "Shaper Calibration", "Please load Pump Spectrum (Air) first.")
            return

        if y_pump_solv is None:
            QMessageBox.warning(self, "Shaper Calibration", "Please load Pump Svt./Filt. Spectrum first to calculate absorbance for fitting.")
            return

        try:
            n_pix = len(y_pump_air)

            # 1. Calculate pump absorbance A = -log10(I_solvent / I_air)
            valid = (y_pump_air > 0) & (y_pump_solv > 0)
            a_pump = np.zeros_like(y_pump_air)
            a_pump[valid] = np.abs(-np.log10(y_pump_solv[valid] / y_pump_air[valid]))

            # 2. Retrieve loaded reference FTIR spectrum (or load default Dioxane FTIR spectrum)
            ref_spec = getattr(self, "_cal_ref_spec", None)
            if ref_spec is None:
                ref_dir = Path(__file__).resolve().parent.parent.parent / "cal" / "ref_spectra"
                default_ref = ref_dir / "FTIR-DioxaneRAL_2cm-1.csv"
                if not default_ref.exists():
                    default_ref = ref_dir / "FTIR-Dioxane_2cm-1.csv"
                if default_ref.exists():
                    ref_spec = load_reference_spectrum(default_ref)
                    self._cal_ref_spec = ref_spec

            if ref_spec is None:
                QMessageBox.warning(self, "Shaper Calibration", "Please load a Calibration Reference Spectrum (e.g. FTIR Dioxane) first.")
                return

            cal_type_idx = self.cal_type_combo.currentIndex() + 1 if hasattr(self, "cal_type_combo") else 9
            if cal_type_idx not in (1, 4, 6, 7, 8, 9):
                cal_type_idx = 9  # Force IR code for shaper calibration

            cwl_val = self.cal_set_cwl.value() if hasattr(self, "cal_set_cwl") else 2000.0
            ppnm_val = self.cal_set_ppnm.value() if hasattr(self, "cal_set_ppnm") else 1.2
            rel_min_nm = self.cal_set_rel_min_nm.value() if hasattr(self, "cal_set_rel_min_nm") else -200.0
            rel_max_nm = self.cal_set_rel_max_nm.value() if hasattr(self, "cal_set_rel_max_nm") else 200.0

            # 3. Fit wavelength/wavenumber axis to reference FTIR spectrum (EXACTLY like probe case!)
            fit_res = fit_wavelength_axis(
                meas_y=a_pump,
                ref_x=ref_spec.spectral_axis,
                ref_y=ref_spec.absorbance,
                cal_type_code=cal_type_idx,
                cwl=cwl_val,
                ppnm_guess=ppnm_val,
                rel_min_nm=rel_min_nm,
                rel_max_nm=rel_max_nm,
                do_fit=self.cal_chk_do_fit.isChecked() if hasattr(self, "cal_chk_do_fit") else True,
            )
            calibrated_pump_cm = fit_res.wavenumber_cm1
            calibrated_pump_nm = fit_res.wavelength_nm
            self._cal_shaper_fit_res = fit_res

            self._cal_shaper_fit_result = calibrated_pump_cm
            self._cal_shaper_fit_nm = calibrated_pump_nm

            # 4. Fit 1D Gaussian to calibrated pump spectrum
            gauss_res = fit_spectrum_gaussian(y_pump_air, calibrated_pump_cm)
            self._cal_pump_fit_result = gauss_res

            # 5. Convert Single Mask & Multiple Mask scans to nm wavelength format for shaper program (saving raw intensities)
            if y_single is not None and len(y_single) == n_pix:
                self._cal_shaper_single_export = np.column_stack((calibrated_pump_nm, y_single))

            if y_multi is not None and len(y_multi) == n_pix:
                self._cal_shaper_multi_export = np.column_stack((calibrated_pump_nm, y_multi))

            # Enable Save button
            if hasattr(self, "cal_shaper_btn_save") and self.cal_shaper_btn_save is not None:
                self.cal_shaper_btn_save.setEnabled(True)

            self._update_shaper_plots()
            self.statusBar().showMessage(f"Shaper Calibration Fit completed: w0={gauss_res.w0:.2f} cm⁻¹, FWHM={gauss_res.fwhm:.2f} cm⁻¹")
            self._show_shaper_gaussian_popup()

        except Exception as err:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Shaper Fit Error", f"Failed to execute Shaper Calibration Fit:\n{err}")

    def _show_shaper_gaussian_popup(self):
        """Open a standalone popup figure showing the calibrated Pump Spectrum and its 1D Gaussian Fit with FWHM legend."""
        import sys
        y_pump_air = getattr(self, "_cal_shaper_pump_air_y", None)
        fit_cm = getattr(self, "_cal_shaper_fit_result", None)
        if y_pump_air is None or fit_cm is None:
            return

        if getattr(self, "_test_mode", False) or "pytest" in sys.modules:
            return

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
        from matplotlib.figure import Figure

        from pymorgan.cal import fit_spectrum_gaussian

        gauss_res = fit_spectrum_gaussian(y_pump_air, fit_cm)
        self._cal_pump_fit_result = gauss_res

        dlg = QDialog(self)
        dlg.setWindowTitle("Pump Spectrum - Gaussian Fit")
        dlg.resize(750, 550)

        fig = Figure(figsize=(7, 5), dpi=100)
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, dlg)

        layout = QVBoxLayout(dlg)
        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        ax = fig.add_subplot(111)
        ax.plot(fit_cm, y_pump_air, "o", color="tab:orange", markersize=4, label="Pump (Air)")
        ax.plot(
            gauss_res.axis_fit,
            gauss_res.fit_curve,
            "-",
            color="black",
            linewidth=1.8,
            label=rf"Fit ($\omega_{{0}}$={gauss_res.w0:.1f} $\mathrm{{cm}}^{{-1}}$, FWHM={gauss_res.fwhm:.1f} $\mathrm{{cm}}^{{-1}}$)",
        )
        ax.set_xlabel(r"Calibrated Wavenumber ($\mathrm{cm}^{-1}$)", fontweight="bold", fontsize=10)
        ax.set_ylabel("Pump Intensity (Counts)", fontweight="bold", fontsize=10)
        ax.set_title("Calibrated Pump Spectrum & 1D Gaussian Fit", fontweight="bold", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        leg = ax.legend(loc="best", frameon=True, fontsize=9)
        if leg is not None:
            leg.set_draggable(True)

        fig.tight_layout()
        canvas.draw()

        dlg.exec()

    def _on_cal_shaper_save(self):
        """Export CalibratedPump.csv, SingleMask.txt, and MultipleMask.txt calibration files."""

        result_cm = getattr(self, "_cal_shaper_fit_result", None)
        if result_cm is None:
            QMessageBox.warning(self, "Save Calibration", "No calibrated pump spectral vector available to save.")
            return

        initial_dir = self._rootdir_text() or getattr(self, "rootdir", "")
        target_dir = Path(initial_dir) if initial_dir else Path.cwd()

        pump_path = target_dir / "CalibratedPump.csv"
        single_path = target_dir / "SingleMask.txt"
        multi_path = target_dir / "MultipleMask.txt"

        existing_files = [p for p in (pump_path, single_path, multi_path) if p.exists()]
        if existing_files:
            names_str = ", ".join(f"'{p.name}'" for p in existing_files)
            resp = QMessageBox.question(
                self,
                "Overwrite Existing Files?",
                f"The following file(s) already exist in directory:\n{target_dir}\n\n{names_str}\n\nDo you want to overwrite them?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if resp != QMessageBox.StandardButton.Yes:
                return

        try:
            # 1. Save CalibratedPump.csv (1 column, calibrated wavenumbers in cm-1)
            np.savetxt(pump_path, result_cm, fmt="%.6f", delimiter=",")

            # 2. Save SingleMask.txt (2 columns: Wavelength in nm, Absorbance)
            single_exp = getattr(self, "_cal_shaper_single_export", None)
            if single_exp is not None:
                np.savetxt(single_path, single_exp, fmt="%.6f\t%.6f")

            # 3. Save MultipleMask.txt (2 columns: Wavelength in nm, Absorbance)
            multi_exp = getattr(self, "_cal_shaper_multi_export", None)
            if multi_exp is not None:
                np.savetxt(multi_path, multi_exp, fmt="%.6f\t%.6f")

            self.statusBar().showMessage(f"Successfully saved CalibratedPump.csv, SingleMask.txt, and MultipleMask.txt to {target_dir.name}")
            QMessageBox.information(
                self,
                "Calibration Saved",
                f"Successfully saved calibration files to directory:\n{target_dir}\n\n- CalibratedPump.csv\n- SingleMask.txt\n- MultipleMask.txt",
            )

        except Exception as err:
            QMessageBox.critical(self, "Save Error", f"Failed to save shaper calibration files:\n{err}")

    def _on_cal_shaper_reset(self):
        """Reset Shaper calibration data, LED indicators, and plots."""
        self._cal_shaper_pump_air_y = None
        self._cal_shaper_pump_solvent_y = None
        self._cal_shaper_single_mask_y = None
        self._cal_shaper_multi_mask_y = None
        self._cal_shaper_fit_result = None
        self._cal_shaper_fit_nm = None
        self._cal_shaper_fit_res = None
        self._cal_shaper_single_export = None
        self._cal_shaper_multi_export = None

        for led_name in ("cal_shaper_led_pump", "cal_shaper_led_solvent", "cal_shaper_led_cal", "cal_shaper_led_mask"):
            led = getattr(self, led_name, None)
            if led:
                led.set_state("off")

        if hasattr(self, "cal_shaper_btn_save") and self.cal_shaper_btn_save is not None:
            self.cal_shaper_btn_save.setEnabled(False)

        if hasattr(self, "cal_canvas_pump") and self.cal_canvas_pump is not None:
            self.cal_canvas_pump.hide()

        self.statusBar().showMessage("Reset Shaper Calibration data.")

    def _update_shaper_plots(self):
        """Render all subplots on cal_canvas_pump for pulse shaper calibration."""
        if not hasattr(self, "cal_canvas_pump"):
            return

        canvas = self.cal_canvas_pump
        canvas.show()
        ax_raw = canvas.ax_raw
        ax_abs = canvas.ax_abs
        ax_ref = canvas.ax_ref
        ax_fit = canvas.ax_fit

        ax_raw.clear()
        ax_abs.clear()
        ax_ref.clear()
        ax_fit.clear()

        # Clean up any previous twin axes attached to ax_ref.figure.axes
        fig = ax_ref.figure
        for ax_twin in list(fig.axes):
            if ax_twin not in (ax_raw, ax_abs, ax_ref, ax_fit) and ax_twin.get_label() == "_ref_twin":
                ax_twin.remove()

        y_pump_air = getattr(self, "_cal_shaper_pump_air_y", None)
        y_pump_solv = getattr(self, "_cal_shaper_pump_solvent_y", None)
        y_single = getattr(self, "_cal_shaper_single_mask_y", None)
        y_multi = getattr(self, "_cal_shaper_multi_mask_y", None)
        fit_cm = getattr(self, "_cal_shaper_fit_result", None)
        fit_nm = getattr(self, "_cal_shaper_fit_nm", None)
        fit_res = getattr(self, "_cal_shaper_fit_res", None)

        n_pix = len(y_pump_air) if y_pump_air is not None else 64
        pixels = np.arange(1, n_pix + 1, dtype=float)
        unit_idx = self.cal_combo_unit_mode.currentIndex() if hasattr(self, "cal_combo_unit_mode") else 0
        fit_unit_mode = "nm" if unit_idx == 0 else "cm-1"

        # --- SUBPLOT 1: RAW MEASUREMENTS ---
        # Pump (air) = black line; pump (solvent) = red line; single mask = cyan shaded area (alpha=0.4); multiple mask = blue line.
        if y_pump_air is not None:
            ax_raw.plot(pixels, y_pump_air, color="black", linewidth=1.5, label="Pump (Air)")
        if y_pump_solv is not None:
            ax_raw.plot(pixels, y_pump_solv, color="red", linewidth=1.5, label="Pump (Solvent)")
        if y_single is not None:
            ax_raw.fill_between(pixels, 0, y_single, color="cyan", alpha=0.4, label="Single Mask")
        if y_multi is not None:
            ax_raw.plot(pixels, y_multi, color="blue", linewidth=1.5, label="Multiple Mask")

        ax_raw.grid(False)
        ax_raw.axhline(0, color="0.75", lw=0.75, zorder=0)
        ax_raw.set_xlabel("Pixel Index", fontweight="bold", fontsize=8)
        ax_raw.set_ylabel("Intensity Counts", fontweight="bold", fontsize=8)
        canvas.make_draggable_legend(ax_raw, title="Raw Measurements")

        # --- SUBPLOT 2: CALCULATED ABSORBANCE & PHYSICAL BASELINE ---
        # absorbance = purple, baseline = orange (drawn when global do_baseline checkbox is checked).
        do_baseline = self.cal_chk_do_baseline.isChecked() if hasattr(self, "cal_chk_do_baseline") else True
        if y_pump_air is not None and y_pump_solv is not None and len(y_pump_air) == len(y_pump_solv):
            valid = (y_pump_air > 0) & (y_pump_solv > 0)
            a_pump = np.zeros_like(y_pump_air)
            a_pump[valid] = -np.log10(y_pump_solv[valid] / y_pump_air[valid])
            ax_abs.plot(pixels, a_pump, color="tab:purple", linewidth=1.8, label="Abs")

            if do_baseline and fit_res is not None and getattr(fit_res, "baseline_fit", None) is not None:
                baseline_ref_units = fit_res.baseline_fit
                scale = float(fit_res.fit_params[2]) if len(fit_res.fit_params) > 2 else 1.0
                if abs(scale) < 1e-6:
                    scale = 1.0

                meas_min = float(np.min(a_pump))
                meas_span = float(np.max(a_pump) - meas_min)
                ref_wl_cut_nm = 1e7 / np.clip(fit_res.ref_x_cut, 1.0, None)
                wl_min_active = np.min(ref_wl_cut_nm)
                wl_max_active = np.max(ref_wl_cut_nm)
                active_pix_mask = (fit_res.wavelength_nm >= wl_min_active) & (fit_res.wavelength_nm <= wl_max_active)

                if np.any(active_pix_mask):
                    sort_ref = np.argsort(ref_wl_cut_nm)
                    b_interp = np.interp(
                        fit_res.wavelength_nm[active_pix_mask],
                        ref_wl_cut_nm[sort_ref],
                        baseline_ref_units[sort_ref],
                    )
                    baseline_meas = meas_min - (b_interp * (meas_span / scale))
                    ax_abs.plot(
                        pixels[active_pix_mask],
                        baseline_meas,
                        color="tab:orange",
                        linestyle="--",
                        linewidth=1.4,
                        label="Baseline (active)",
                    )

        ax_abs.grid(False)
        ax_abs.axhline(0, color="0.75", lw=0.75, zorder=0)
        ax_abs.set_xlabel("Pixel Index", fontweight="bold", fontsize=8)
        ax_abs.set_ylabel("Absorbance (OD)", fontweight="bold", fontsize=8)
        canvas.make_draggable_legend(ax_abs, title="Calculated Absorbance")

        # --- SUBPLOT 3: IDENTICAL TO SPECTROMETER CALIBRATION ---
        ref_spec = getattr(self, "_cal_ref_spec", None)

        if fit_res is not None and ref_spec is not None:
            ref_x_full = fit_res.ref_x_full
            ref_y_full = fit_res.ref_y_full
            ref_x_cut = fit_res.ref_x_cut
            ref_y_cut = fit_res.ref_y_cut
            exp_y_source = (fit_res.exp_corr_y_fit if do_baseline else fit_res.model_y_fit) if fit_res.exp_corr_y_fit is not None else fit_res.model_y_fit

            sort_cut = np.argsort(ref_x_cut)
            ref_x_cut_plot = ref_x_cut[sort_cut]
            ref_y_cut_plot = ref_y_cut[sort_cut]
            model_y_fit_plot = exp_y_source[sort_cut]
            conv_y_plot = fit_res.convolved_ref_y[sort_cut] if fit_res.convolved_ref_y is not None else None

            ax_exp = ax_ref.twinx()
            ax_exp.set_label("_ref_twin")
            for s in ("top", "bottom", "left"):
                ax_exp.spines[s].set_visible(False)

            # 1. Full reference spectrum in fainter gray line on LEFT axis
            if ref_x_full is not None and ref_y_full is not None and len(ref_x_full) > 0:
                sort_full = np.argsort(ref_x_full)
                ax_ref.plot(
                    ref_x_full[sort_full],
                    ref_y_full[sort_full],
                    color="0.65",
                    linewidth=0.9,
                    alpha=0.55,
                    linestyle="-",
                    label="Full Ref",
                    zorder=1,
                )

            # 2. Active reference in solid black on LEFT axis
            ax_ref.plot(ref_x_cut_plot, ref_y_cut_plot, color="black", linewidth=1.4, alpha=1.0, linestyle="-", label="Raw Ref", zorder=2)
            if conv_y_plot is not None:
                ax_ref.plot(ref_x_cut_plot, conv_y_plot, color="red", linewidth=1.8, alpha=0.6, linestyle="-", label=rf"Conv Ref ($\sigma$={fit_res.sigma_opt:.2f})", zorder=4)

            # 3. Baseline-corrected experimental on RIGHT axis (purple, translucent)
            ax_exp.plot(ref_x_cut_plot, model_y_fit_plot, color="tab:purple", alpha=0.7, linewidth=1.8, label="Exp (BC)", zorder=3)

            ref_min, ref_max = float(np.min(ref_y_cut_plot)), float(np.max(ref_y_cut_plot))
            exp_min, exp_max = float(np.min(model_y_fit_plot)), float(np.max(model_y_fit_plot))

            if abs(ref_max - ref_min) < 1e-6:
                ref_min -= 0.05
                ref_max += 0.05
            if abs(exp_max - exp_min) < 1e-6:
                exp_min -= 0.05
                exp_max += 0.05

            ax_ref.set_ylim(ref_min, ref_max)
            ax_exp.set_ylim(exp_min, exp_max)

            align.yaxes(ax_ref, 0.0, ax_exp, 0.0, pos=0.05)

            ax_ref.grid(False)
            ax_ref.axhline(0, color="0.75", lw=0.75, zorder=0)
            ax_ref.set_xlim(np.min(ref_x_cut_plot), np.max(ref_x_cut_plot))
            ax_ref.set_xlabel(r"Wavenumber ($\mathrm{cm}^{-1}$)", fontweight="bold", fontsize=8)
            ax_ref.set_ylabel("Absorbance (Ref)", fontweight="bold", fontsize=8, color="black")
            ax_ref.tick_params(axis="y", labelsize=8, labelcolor="black")
            ax_exp.set_ylabel("Absorbance (Exp BC)", fontweight="bold", fontsize=8, color="tab:purple")
            ax_exp.tick_params(axis="y", labelsize=8, labelcolor="tab:purple")

            lines1, labels1 = ax_ref.get_legend_handles_labels()
            lines2, labels2 = ax_exp.get_legend_handles_labels()
            leg = ax_ref.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="best", framealpha=0.85)
            if leg is not None:
                leg.set_draggable(True)

        elif ref_spec is not None:
            ax_ref.plot(ref_spec.spectral_axis, ref_spec.absorbance, color="black", linewidth=1.4, label=ref_spec.name)
            ax_ref.grid(False)
            ax_ref.axhline(0, color="0.75", lw=0.75, zorder=0)
            ax_ref.set_xlabel(f"Spectral Axis ({ref_spec.axis_unit})", fontweight="bold", fontsize=8)
            ax_ref.set_ylabel("Norm. Absorbance", fontweight="bold", fontsize=8)
            canvas.make_draggable_legend(ax_ref, title="Reference Spectrum")

        # --- SUBPLOT 4: CALIBRATION FIT (PIXEL VS WAVELENGTH / WAVENUMBER) ---
        if fit_res is not None and (fit_nm is not None or fit_cm is not None):
            pix = pixels
            if fit_unit_mode == "cm-1" and fit_cm is not None:
                fit_y = fit_cm
                fit_unit_label = r"Wavenumber ($\mathrm{cm}^{-1}$)"
            else:
                fit_y = fit_nm if fit_nm is not None else fit_cm
                fit_unit_label = "Wavelength (nm)"

            # Format short notation legend for calibration fit parameters (identical to Detector 1)
            p0_val = fit_res.fit_params[0] if (fit_res.fit_params is not None and len(fit_res.fit_params) > 0) else 0.0
            ppnm_val = fit_res.fit_params[1] if (fit_res.fit_params is not None and len(fit_res.fit_params) > 1) else 0.0
            nm_per_pix = (1.0 / ppnm_val) if ppnm_val > 0 else 0.0

            fit_lbl = rf"Fit ($\lambda_{0}$={p0_val:.1f} nm, {nm_per_pix:.2f} nm/px"
            if fit_res.fit_params is not None and len(fit_res.fit_params) > 6 and abs(fit_res.fit_params[6]) > 1e-4:
                fit_lbl += rf", $c_{2}$={fit_res.fit_params[6]:.2f}"
            if fit_res.fit_params is not None and len(fit_res.fit_params) > 7 and abs(fit_res.fit_params[7]) > 1e-4:
                fit_lbl += rf", $c_{3}$={fit_res.fit_params[7]:.2f}"
            fit_lbl += ")"

            # Linear reference curve connecting first to last point of plot (zorder=1)
            ax_fit.plot([pix[0], pix[-1]], [fit_y[0], fit_y[-1]], color="0.5", lw=1.0, linestyle="--", label="Linear Ref", zorder=1)

            # Plot fitted dispersion curve with alpha=0.5 (zorder=2) in tab:green
            ax_fit.plot(pix, fit_y, color="tab:green", alpha=0.5, linewidth=1.8, label=fit_lbl, zorder=2)

            # Plot current probe calibration as a dashed line (zorder=3)
            curr_probe = self._get_current_probe_calibration()
            if curr_probe is not None and len(curr_probe) > 0:
                pix_curr = np.arange(1, len(curr_probe) + 1, dtype=float)
                if fit_unit_mode == "cm-1" and np.mean(curr_probe) < 1000:
                    curr_plot = 1e7 / np.clip(curr_probe, 1e-3, None)
                elif fit_unit_mode == "nm" and np.mean(curr_probe) > 1000:
                    curr_plot = 1e7 / np.clip(curr_probe, 1e-3, None)
                else:
                    curr_plot = curr_probe
                ax_fit.plot(
                    pix_curr,
                    curr_plot,
                    color="tab:blue",
                    linestyle="--",
                    linewidth=1.8,
                    label="Current Probe Cal",
                    zorder=3,
                )

            # Crosshair lines at central pixel and central wavelength
            cx = (1.0 + len(pix)) / 2.0
            cy = fit_y[len(fit_y) // 2]
            ax_fit.axvline(cx, color="0.75", lw=0.75, ls=":", zorder=0)
            ax_fit.axhline(cy, color="0.75", lw=0.75, ls=":", zorder=0)

            ax_fit.set_ylabel(fit_unit_label, fontweight="bold", fontsize=8)
        else:
            fit_unit_label = r"Wavenumber ($\mathrm{cm}^{-1}$)" if fit_unit_mode == "cm-1" else "Wavelength (nm)"
            ax_fit.set_ylabel(fit_unit_label, fontweight="bold", fontsize=8)

        ax_fit.grid(False)
        ax_fit.set_xlabel("Pixel", fontweight="bold", fontsize=8)
        canvas.make_draggable_legend(ax_fit, title="Calibration Fit")

        if not getattr(canvas, "_tight_layout_done", False):
            import contextlib
            with contextlib.suppress(Exception):
                canvas.figure.tight_layout()
            canvas._tight_layout_done = True
        canvas.draw()

    def _get_current_probe_calibration(self) -> np.ndarray | None:
        """Get the current probe calibration vector from the active 2D or 1D tab dataset."""
        try:
            if getattr(self, "twoD_dataset", None) is not None and getattr(self.twoD_dataset, "probe", None) is not None:
                return self.twoD_dataset.probe
            if getattr(self, "dataset", None) is not None and getattr(self.dataset, "probe", None) is not None:
                return self.dataset.probe
        except Exception:
            pass
        return None

    def _refresh_current_probe_line(self):
        """Re-render the CAL tab fit plots to update the current probe dashed line."""
        res = getattr(self, "_cal_fit_result_det1", None)
        if res is not None:
            self._update_cal_fit_plot_det1(res)
        elif hasattr(self, "_update_shaper_plots"):
            self._update_shaper_plots()

    def _on_cal_reset(self):
        """Reset all calibration data and set all LED status indicators to off (gray)."""
        led_names = [
            "cal_spec_led_probe",
            "cal_spec_led_solvent",
            "cal_spec_led_cal",
            "cal_shaper_led_pump",
            "cal_shaper_led_solvent",
            "cal_shaper_led_cal",
            "cal_shaper_led_mask",
        ]
        for name in led_names:
            led = getattr(self, name, None)
            if led is not None and hasattr(led, "set_state"):
                led.set_state("off")

        self._cal_exp_data = None
        self._cal_svt_data = None
        self._cal_ref_spec = None
        self._cal_fit_result_det1 = None
        self._cal_fit_result_det2 = None

        if hasattr(self, "cal_canvas_det1") and self.cal_canvas_det1 is not None:
            self.cal_canvas_det1.hide()
        if hasattr(self, "cal_canvas_det2") and self.cal_canvas_det2 is not None:
            self.cal_canvas_det2.hide()

        if hasattr(self, "cal_spec_btn_save") and self.cal_spec_btn_save is not None:
            self.cal_spec_btn_save.setEnabled(False)
        if hasattr(self, "cal_shaper_btn_save") and self.cal_shaper_btn_save is not None:
            self.cal_shaper_btn_save.setEnabled(False)

        self.statusBar().showMessage("Calibration data reset. All LEDs reset to off.")

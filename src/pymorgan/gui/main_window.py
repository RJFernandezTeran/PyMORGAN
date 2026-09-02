"""Main window: the original Qt Designer layout wired to the PyMORGAN pipeline.

The shell is loaded from ``main_window.ui`` (the ported original design with its
Pump-Probe / 2D-IR / Calibration tabs and the promoted ``PPaxes`` plot widget).
The Pump-Probe controls are connected to the 1-D pipeline. Wiring is defensive:
each connection is made only if the named widget exists, so the window still
loads if the layout changes.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyqt6")

from pathlib import Path

import numpy as np
from PyQt6 import uic
from PyQt6.QtCore import QEvent, Qt, QTimer, QUrl
from PyQt6.QtGui import (
    QDesktopServices,
    QKeySequence,
    QPixmap,
    QShortcut,
    QStandardItemModel,
)
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

import pymorgan as pm

from ..log import get_logger

# Kept importable from this module for backwards compatibility: these moved to
# ``mw_common``/``dialogs`` when the per-tab mixins were split out.
from .mw_common import (  # noqa: F401
    _DATASET_WIDGETS,
    _DATASET_WIDGETS_2D,
    _ISDIR_ROLE,
    _PATH_ROLE,
    _RENDER_DEBOUNCE_MS,
    DATA_TYPE_DISPLAY_NAMES,
    _safe_set_limits,
    get_combo_datatype,
    populate_datatype_combo,
    set_combo_datatype,
)
from .tabs import CalibrationTabMixin, DatasetBrowserMixin, OneDTabMixin, TwoDTabMixin

logger = get_logger(__name__)

# Greyed-out look for disabled buttons (unimplemented features, and every
# button while a long operation runs -- see gui/busy.py).
_DISABLED_LIGHT_BG, _DISABLED_LIGHT_FG, _DISABLED_LIGHT_BORDER = "#ececec", "#9a9a9a", "#d4d4d4"
_DISABLED_DARK_BG, _DISABLED_DARK_FG, _DISABLED_DARK_BORDER = "#3a3a3a", "#7a7a7a", "#4a4a4a"

_UI_FILE = Path(__file__).with_name("main_window.ui")
_ICON_DIR = Path(__file__).with_name("icons")
_BUTTON_ICONS = {
    "RefreshDir_btn": "refresh_md.png",
    "GoToToday_btn": "today_md.png",
    "UpDir_btn": "uparrow_md.png",
    "BrowseRootdir_btn": "screensearch_md.png",
}

# Menu-action icons (applied programmatically, like _BUTTON_ICONS, to avoid
# embedding fragile file paths in the .ui).
_ACTION_ICONS = {
    "actionOpen": "upload_md.png",
    "actionReload": "refresh_md.png",
    "actionSetRoot": "screensearch_md.png",
    "actionToday": "today_md.png",
    "actionExportFigure": "saveas_md.png",
    "actionExportData": "save_md.png",
    "actionExportPDAT": "savefile_md.png",
    "actionLoadSpectrum": "upload_md.png",
    "actionAbout": "gui.png",
}


class MainWindow(
    DatasetBrowserMixin,
    OneDTabMixin,
    TwoDTabMixin,
    CalibrationTabMixin,
    QMainWindow,
):
    """PyMORGAN main window."""

    def __init__(self):
        super().__init__()
        self._load_ui()
        # Default geometry as declared in main_window.ui, captured before any
        # user-driven resize so that View -> Restore Default Window Size works.
        self._default_size = self.size()
        self.setWindowTitle(f"PyMORGAN v{pm.__version__}")
        self._load_project_settings()
        pm.apply_style()
        self._apply_icons()
        self._apply_tab_theme()

        self.dataset = None
        self.twoD_dataset = None
        self._twoD_overlay = None
        self._twoD_current_path = None
        self._twoD_pop_window = None
        self._current_path = None
        self._picker = None

        self._ss_abs = None
        self._ss_em = None

        self._wire()

        self._build_menus()
        self._init_state_controls()
        self._init_rootdir()
        self._init_plot_controls()
        self._init_spectral_diffusion_controls()
        self._init_twoD_other_plot()
        self._retain_hidden_space()
        self._set_dataset_widgets_visible(False)
        self._set_twoD_dataset_widgets_visible(False)
        if hasattr(self, "twoD_interactive_chk") and self.twoD_interactive_chk is not None:
            self.twoD_interactive_chk.setChecked(True)

        self._apply_button_aesthetics()
        self._apply_button_tooltips_and_states()

    def _load_ui(self):
        """Load the Qt Designer UI, using a compiled module when possible.

        If ``main_window_ui.py`` exists and is up-to-date it is imported
        directly (fastest path).  If ``main_window.ui`` has been modified since
        the compiled file was last written, ``pyuic6`` is run automatically to
        regenerate it, after which the fresh compiled module is imported.
        If compilation fails for any reason the raw XML is parsed via
        ``uic.loadUi`` as a transparent fallback.
        """
        ui_compiled = Path(__file__).with_name("main_window_ui.py")

        # Auto-recompile when the .ui file is newer than the compiled module.
        if _UI_FILE.stat().st_mtime > (ui_compiled.stat().st_mtime if ui_compiled.exists() else 0):
            try:
                import subprocess
                import sys

                result = subprocess.run(
                    [sys.executable, "-m", "PyQt6.uic.pyuic", str(_UI_FILE), "-o", str(ui_compiled)],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    logger.warning(
                        "pyuic6 recompile failed (rc=%d): %s",
                        result.returncode,
                        result.stderr.strip(),
                    )
                    ui_compiled = Path("__nonexistent__")  # force fallback
                else:
                    logger.debug("main_window_ui.py recompiled from main_window.ui")
                    # Invalidate any cached import of the stale compiled module.
                    import importlib
                    import sys as _sys

                    mod_name = f"{__package__}.main_window_ui"
                    if mod_name in _sys.modules:
                        del _sys.modules[mod_name]
            except Exception:
                logger.warning("Could not recompile main_window.ui; falling back to uic.loadUi.", exc_info=True)
                ui_compiled = Path("__nonexistent__")  # force fallback

        if ui_compiled.exists():
            try:
                from .main_window_ui import Ui_MainWindow

                ui = Ui_MainWindow()
                ui.setupUi(self)
                for attr, val in ui.__dict__.items():
                    setattr(self, attr, val)
                return
            except Exception:
                logger.warning("Could not load compiled UI; falling back to uic.loadUi.", exc_info=True)
        uic.loadUi(str(_UI_FILE), self)

    def _apply_button_tooltips_and_states(self):
        """Configure descriptive tooltips for connected buttons and disable unimplemented stubs."""
        # 1. Unimplemented stub buttons -> disable and explain in tooltip
        stubs = {}
        for btn_name, tooltip in stubs.items():
            btn = getattr(self, btn_name, None)
            if btn is not None:
                btn.setToolTip(tooltip)
                btn.setEnabled(False)

        # 2. Connected & working buttons -> ensure informative tooltips
        tooltips = {
            "RefreshDir_btn": "Refresh data directory file listing",
            "GoToToday_btn": "Navigate to today's data directory",
            "UpDir_btn": "Navigate up one parent directory level",
            "BrowseRootdir_btn": "Browse and select root data directory",
            "PP_reloadData_btn": "Reload current 1D dataset from file",
            "PP_fitChirp_btn": "Fit chirp/dispersion correction polynomial curve",
            "PP_loadChirpCorr_btn": "Load chirp correction parameters from file",
            "PP_subtractSolvent_btn": "Subtract scaled solvent background dataset",
            "PP_subtractShockwave_btn": "Subtract shockwave kinetic artefact trace",
            "PP_plotBkg_btn": "Plot fitted background subtraction curve",
            "PP_maskProbe_btn": "Mask specific probe wavelength ranges",
            "PP_shiftT0_btn": "Shift delay time axis so a chosen point becomes the new t0",
            "PP_plotNoise_btn": "Plot noise spectrum across probe pixels",
            "PP_plotCounts_btn": "Plot total accumulation counts vs probe delay",
            "PP_CtrPlot_btn": "Open 2D contour plot window of 1D dataset",
            "PP_kineticsPlot_btn": "Plot kinetic decay traces at selected probe wavelengths",
            "PP_SurfPlot_btn": "Open 3D surface plot window of 1D dataset",
            "PP_trSpecPlot_btn": "Plot transient spectra at selected delay times",
            "PP_ScanKinetics_btn": "Plot kinetics across individual dataset scans",
            "PP_ScanSpectra_btn": "Plot transient spectra across individual dataset scans",
            "PP_RecalcAvg_btn": "Recalculate dataset average excluding unselected scans",
            "PP_LoadSSbutton": "Load steady-state FTIR spectrum from file",
            "PP_ShowSSbutton": "Show or hide steady-state spectrum overlay",
            "PP_ClearSSbutton": "Clear currently loaded steady-state overlay",
            "PC_restore": "Restore default zoom and axis limits for 1D plots",
            "twoD_reloadData_btn": "Reload current 2D dataset from file",
            "twoD_apply_changes_btn": "Apply phase and FT settings and reload 2D dataset",
            "twoD_save_probe_cal_btn": "Save probe wavelength calibration to file",
            "twoD_contour_btn": "Open 2D contour plot window at current t₂ delay",
            "twoD_kinetics_btn": "Plot 2D kinetics at selected pump/probe peak coordinates",
            "twoD_surf_btn": "Open 3D surface plot window of 2D spectrum",
            "twoD_slices_btn": "Plot 1D pump/probe slices at cursor position",
            "twoD_apply_reprocess_btn": "Reprocess phasing and FT data",
            "twoD_plot_sd_kinetics_btn": "Plot spectral diffusion centre line slope (CLS) kinetics",
            "twoD_timedomain_fit_btn": "Fit 2D time-domain signals",
            "twoD_spectral_diffusion_btn": "Analyze 2D spectral diffusion (CLS/ellipticity) dynamics",
            "twoD_make_gif_btn": "Export animated GIF or video movie of 2D contour plots vs t₂ delay",
            "twoD_gaussian_fit_btn": "Fit 2D Gaussian models with GSB/ESA pairs and cross-peaks",
            "twoD_subtract_spectra_btn": "Subtract reference 2D dataset spectrum with scaling",
            "twoD_integral_dynamics_btn": "Integrate 2D spectral volume ROI across population delays",
            "twoD_PC_restore": "Restore default zoom and axis limits for 2D preview",
            "twoD_PC_sync": "Synchronize 2D plot axis limits across subtabs",
        }
        for btn_name, tooltip in tooltips.items():
            btn = getattr(self, btn_name, None)
            if btn is not None:
                btn.setToolTip(tooltip)

    def _apply_button_aesthetics(self):
        """Apply theme-responsive signature stylesheets for every individual button based on light/dark mode."""
        from .theme import is_dark_palette
        dark = is_dark_palette()

        def style(light_bg: str, light_text: str, light_border: str, light_hover: str,
                  dark_bg: str, dark_text: str, dark_border: str, dark_hover: str,
                  font_size: str = "11px") -> str:
            # A disabled button keeps its per-button colours unless the sheet
            # says otherwise, so every style ends with a neutral grey
            # :disabled rule (also killing the hover effect).
            if dark:
                return (
                    f"QPushButton {{ font-weight: bold; font-size: {font_size}; color: {dark_text}; background-color: {dark_bg}; border: 1px solid {dark_border}; border-radius: 4px; padding: 4px 6px; }}"
                    f"QPushButton:hover:enabled {{ background-color: {dark_hover}; border-color: {dark_text}; }}"
                    f"QPushButton:pressed {{ background-color: {dark_border}; }}"
                    f"QPushButton:disabled {{ color: {_DISABLED_DARK_FG}; background-color: {_DISABLED_DARK_BG}; border: 1px solid {_DISABLED_DARK_BORDER}; }}"
                )
            else:
                return (
                    f"QPushButton {{ font-weight: bold; font-size: {font_size}; color: {light_text}; background-color: {light_bg}; border: 1px solid {light_border}; border-radius: 4px; padding: 4px 6px; }}"
                    f"QPushButton:hover:enabled {{ background-color: {light_hover}; border-color: {light_text}; }}"
                    f"QPushButton:pressed {{ background-color: {light_border}; }}"
                    f"QPushButton:disabled {{ color: {_DISABLED_LIGHT_FG}; background-color: {_DISABLED_LIGHT_BG}; border: 1px solid {_DISABLED_LIGHT_BORDER}; }}"
                )

        def toggle_style(
            l_bg: str, l_txt: str, l_border: str,
            d_bg: str, d_txt: str, d_border: str,
            font_size: str = "11px",
        ) -> str:
            if dark:
                return (
                    f"QPushButton {{ font-weight: bold; font-size: {font_size}; color: #94a3b8; background-color: #1e293b; border: 1px solid #334155; border-radius: 4px; padding: 4px 6px; }}"
                    f"QPushButton:hover:enabled {{ background-color: #334155; color: #f8fafc; }}"
                    f"QPushButton:checked {{ color: {d_txt}; background-color: {d_bg}; border: 1px solid {d_border}; }}"
                    f"QPushButton:checked:hover {{ background-color: {d_border}; color: #ffffff; }}"
                    f"QPushButton:disabled {{ color: {_DISABLED_DARK_FG}; background-color: {_DISABLED_DARK_BG}; border: 1px solid {_DISABLED_DARK_BORDER}; }}"
                )
            else:
                return (
                    f"QPushButton {{ font-weight: bold; font-size: {font_size}; color: #475569; background-color: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 4px; padding: 4px 6px; }}"
                    f"QPushButton:hover:enabled {{ background-color: #e2e8f0; color: #0f172a; }}"
                    f"QPushButton:checked {{ color: {l_txt}; background-color: {l_bg}; border: 1px solid {l_border}; }}"
                    f"QPushButton:checked:hover {{ background-color: {l_border}; color: {l_txt}; }}"
                    f"QPushButton:disabled {{ color: {_DISABLED_LIGHT_FG}; background-color: {_DISABLED_LIGHT_BG}; border: 1px solid {_DISABLED_LIGHT_BORDER}; }}"
                )

        # Signature distinct colour for every button function
        button_map = {
            # --- 2D Other Plots Mode Toggle Buttons ---
            "twoD_other_btn_TD": toggle_style("#dbeafe", "#1e40af", "#3b82f6", "#1e3a8a", "#93c5fd", "#3b82f6"),
            "twoD_other_btn_PH": toggle_style("#f3e8ff", "#6b21a8", "#a855f7", "#3b0764", "#e9d5ff", "#a855f7"),
            "twoD_other_btn_CAL": toggle_style("#d1fae5", "#065f46", "#10b981", "#064e3b", "#6ee7b7", "#10b981"),
            # --- Main Plots & Cuts Buttons ---
            # 1. Contour Plot: Dark Navy (Light: Sky Blue tint, Dark: Slate Navy with Sky Cyan text)
            "PP_CtrPlot_btn": style("#d6eaf8", "#1b4f72", "#aed6f1", "#aed6f1", "#1e293b", "#38bdf8", "#0284c7", "#0369a1", "12px"),
            "twoD_contour_btn": style("#d6eaf8", "#1b4f72", "#aed6f1", "#aed6f1", "#1e293b", "#38bdf8", "#0284c7", "#0369a1", "12px"),

            # 2. Plot Kinetics: Forest Green (Light: Mint tint, Dark: Emerald Slate with Mint text)
            "PP_kineticsPlot_btn": style("#d5f5e3", "#196f3d", "#abebc6", "#abebc6", "#064e3b", "#6ee7b7", "#059669", "#047857", "12px"),
            "twoD_kinetics_btn": style("#d5f5e3", "#196f3d", "#abebc6", "#abebc6", "#064e3b", "#6ee7b7", "#059669", "#047857", "12px"),
            "twoD_plot_sd_kinetics_btn": style("#d5f5e3", "#196f3d", "#abebc6", "#abebc6", "#064e3b", "#6ee7b7", "#059669", "#047857"),

            # 3. 3D Surface Plot: Dark Red / Crimson (Light: Rose tint, Dark: Crimson Slate with Coral text)
            "PP_SurfPlot_btn": style("#fadbd8", "#78281f", "#f5b7b1", "#f5b7b1", "#450a0a", "#fca5a5", "#dc2626", "#b91c1c", "12px"),
            "twoD_surf_btn": style("#fadbd8", "#78281f", "#f5b7b1", "#f5b7b1", "#450a0a", "#fca5a5", "#dc2626", "#b91c1c", "12px"),

            # 4. Plot Spectra / Slices: Royal Blue (Light: Soft Royal tint, Dark: Indigo Slate with Blue text)
            "PP_trSpecPlot_btn": style("#ebf5fb", "#21618c", "#aed6f1", "#aed6f1", "#172554", "#93c5fd", "#3b82f6", "#1d4ed8", "12px"),
            "twoD_slices_btn": style("#ebf5fb", "#21618c", "#aed6f1", "#aed6f1", "#172554", "#93c5fd", "#3b82f6", "#1d4ed8", "12px"),

            # --- 1D Preprocess & Utility Buttons ---
            "PP_reloadData_btn": style("#e6f4f1", "#115e59", "#99f6e4", "#ccfbf1", "#134e4a", "#5eead4", "#14b8a6", "#0f766e"),
            "PP_fitChirp_btn": style("#f3e8ff", "#6b21a8", "#e9d5ff", "#ddd6fe", "#3b0764", "#e9d5ff", "#a855f7", "#7e22ce"),
            "PP_loadChirpCorr_btn": style("#ebdef0", "#5b2c6f", "#d7bde2", "#d7bde2", "#2e1065", "#ddd6fe", "#8b5cf6", "#6d28d9"),
            "PP_subtractSolvent_btn": style("#f1f5f9", "#334155", "#cbd5e1", "#e2e8f0", "#334155", "#e2e8f0", "#64748b", "#475569"),
            "PP_subtractShockwave_btn": style("#e2e8f0", "#1e293b", "#94a3b8", "#cbd5e1", "#1e293b", "#cbd5e1", "#475569", "#334155"),
            "PP_plotBkg_btn": style("#e0f2fe", "#0369a1", "#bae6fd", "#bae6fd", "#0c4a6e", "#7dd3fc", "#0284c7", "#0369a1"),
            "PP_maskProbe_btn": style("#ffedd5", "#9a3412", "#fed7aa", "#fed7aa", "#431407", "#fdba74", "#ea580c", "#c2410c"),
            "PP_shiftT0_btn": style("#ffedd5", "#9a3412", "#fed7aa", "#fed7aa", "#431407", "#fdba74", "#ea580c", "#c2410c"),
            "PP_plotNoise_btn": style("#f5f3ff", "#6d28d9", "#ddd6fe", "#ddd6fe", "#3b0764", "#c084fc", "#9333ea", "#7e22ce"),
            "PP_plotCounts_btn": style("#e0f2fe", "#0369a1", "#bae6fd", "#bae6fd", "#0c4a6e", "#38bdf8", "#0284c7", "#0369a1"),

            "PP_ScanKinetics_btn": style("#e6f4f1", "#115e59", "#99f6e4", "#ccfbf1", "#134e4a", "#5eead4", "#14b8a6", "#0f766e"),
            "PP_ScanSpectra_btn": style("#e0f2fe", "#075985", "#bae6fd", "#bae6fd", "#0c4a6e", "#38bdf8", "#0284c7", "#0369a1"),
            "PP_RecalcAvg_btn": style("#d1fae5", "#065f46", "#a7f3d0", "#a7f3d0", "#064e3b", "#34d399", "#10b981", "#059669"),

            "PP_LoadSSbutton": style("#ccfbf1", "#0f766e", "#99f6e4", "#99f6e4", "#134e4a", "#2dd4bf", "#14b8a6", "#0f766e"),
            "PP_ShowSSbutton": style("#dbeafe", "#1e40af", "#bfdbfe", "#bfdbfe", "#1e3a8a", "#60a5fa", "#3b82f6", "#1d4ed8"),
            "PP_ClearSSbutton": style("#ffe4e6", "#9f1239", "#fecdd3", "#fecdd3", "#4c0519", "#fda4af", "#f43f5e", "#e11d48"),

            # --- 2D Preprocess & Phasing ---
            "twoD_reloadData_btn": style("#e6f4f1", "#115e59", "#99f6e4", "#ccfbf1", "#134e4a", "#5eead4", "#14b8a6", "#0f766e"),
            "twoD_apply_changes_btn": style("#d5f5e3", "#196f3d", "#abebc6", "#abebc6", "#064e3b", "#6ee7b7", "#059669", "#047857"),
            "twoD_save_probe_cal_btn": style("#ccfbf1", "#0f766e", "#99f6e4", "#99f6e4", "#134e4a", "#2dd4bf", "#14b8a6", "#0f766e"),
            "twoD_apply_reprocess_btn": style("#d5f5e3", "#196f3d", "#abebc6", "#abebc6", "#064e3b", "#6ee7b7", "#059669", "#047857"),

            # --- 2D Analysis Group ---
            "twoD_integral_dynamics_btn": style("#d5f5e3", "#196f3d", "#abebc6", "#abebc6", "#064e3b", "#6ee7b7", "#059669", "#047857"),
            "twoD_gaussian_fit_btn": style("#d6eaf8", "#1b4f72", "#aed6f1", "#aed6f1", "#1e293b", "#38bdf8", "#0284c7", "#0369a1"),
            "twoD_timedomain_fit_btn": style("#ebdef0", "#5b2c6f", "#d7bde2", "#d7bde2", "#2e1065", "#ddd6fe", "#8b5cf6", "#6d28d9"),
            "twoD_subtract_spectra_btn": style("#fadbd8", "#78281f", "#f5b7b1", "#f5b7b1", "#450a0a", "#fca5a5", "#dc2626", "#b91c1c"),
            "twoD_spectral_diffusion_btn": style("#fdebd0", "#7e5109", "#f9e79f", "#f9e79f", "#451a03", "#fde047", "#ca8a04", "#a16207"),
            "twoD_make_gif_btn": style("#e5e8e8", "#2c3e50", "#d5dbdb", "#d5dbdb", "#334155", "#e2e8f0", "#64748b", "#475569"),

            # --- Calibration Files & Aux ---
            "cal_spec_btn_load_probe": style("#fdebd0", "#7e5109", "#f9e79f", "#f9e79f", "#451a03", "#fde047", "#ca8a04", "#a16207"),
            "cal_spec_btn_load_solvent": style("#fdebd0", "#7e5109", "#f9e79f", "#f9e79f", "#451a03", "#fde047", "#ca8a04", "#a16207"),
            "cal_spec_btn_load_cal": style("#fdebd0", "#7e5109", "#f9e79f", "#f9e79f", "#451a03", "#fde047", "#ca8a04", "#a16207"),
            "cal_shaper_btn_load_pump": style("#fdebd0", "#7e5109", "#f9e79f", "#f9e79f", "#451a03", "#fde047", "#ca8a04", "#a16207"),
            "cal_shaper_btn_load_solvent": style("#fdebd0", "#7e5109", "#f9e79f", "#f9e79f", "#451a03", "#fde047", "#ca8a04", "#a16207"),
            "cal_shaper_btn_load_cal": style("#fdebd0", "#7e5109", "#f9e79f", "#f9e79f", "#451a03", "#fde047", "#ca8a04", "#a16207"),
            "cal_shaper_btn_load_masks": style("#fdebd0", "#7e5109", "#f9e79f", "#f9e79f", "#451a03", "#fde047", "#ca8a04", "#a16207"),
            "cal_btn_merge": style("#fdebd0", "#7e5109", "#f9e79f", "#f9e79f", "#451a03", "#fde047", "#ca8a04", "#a16207"),
            "cal_btn_split": style("#fdebd0", "#7e5109", "#f9e79f", "#f9e79f", "#451a03", "#fde047", "#ca8a04", "#a16207"),
            "cal_btn_add_w0": style("#fdebd0", "#7e5109", "#f9e79f", "#f9e79f", "#451a03", "#fde047", "#ca8a04", "#a16207"),
            "cal_btn_del_w0": style("#fdebd0", "#7e5109", "#f9e79f", "#f9e79f", "#451a03", "#fde047", "#ca8a04", "#a16207"),
            "cal_btn_save_csv": style("#fdebd0", "#7e5109", "#f9e79f", "#f9e79f", "#451a03", "#fde047", "#ca8a04", "#a16207"),
            "cal_spec_btn_do_fit": style("#ebdef0", "#5b2c6f", "#d7bde2", "#d7bde2", "#2e1065", "#ddd6fe", "#8b5cf6", "#6d28d9"),
            "cal_shaper_btn_do_fit": style("#ebdef0", "#5b2c6f", "#d7bde2", "#d7bde2", "#2e1065", "#ddd6fe", "#8b5cf6", "#6d28d9"),
            "cal_spec_btn_save": style("#d5f5e3", "#196f3d", "#abebc6", "#abebc6", "#064e3b", "#6ee7b7", "#059669", "#047857"),
            "cal_shaper_btn_save": style("#d5f5e3", "#196f3d", "#abebc6", "#abebc6", "#064e3b", "#6ee7b7", "#059669", "#047857"),
            "cal_apply_probe_btn": style("#d5f5e3", "#196f3d", "#abebc6", "#abebc6", "#064e3b", "#6ee7b7", "#059669", "#047857"),
            "cal_apply_shaper_btn": style("#d5f5e3", "#196f3d", "#abebc6", "#abebc6", "#064e3b", "#6ee7b7", "#059669", "#047857"),
            "cal_save_settings_btn": style("#d5f5e3", "#196f3d", "#abebc6", "#abebc6", "#064e3b", "#6ee7b7", "#059669", "#047857"),

            # --- Subtractions & Reset ---
            "cal_spec_btn_reset": style("#fadbd8", "#78281f", "#f5b7b1", "#f5b7b1", "#450a0a", "#fca5a5", "#dc2626", "#b91c1c"),
            "cal_shaper_btn_reset": style("#fadbd8", "#78281f", "#f5b7b1", "#f5b7b1", "#450a0a", "#fca5a5", "#dc2626", "#b91c1c"),

            # --- Navigation & Restores ---
            "RefreshDir_btn": style("#f1f5f9", "#334155", "#cbd5e1", "#e2e8f0", "#334155", "#e2e8f0", "#64748b", "#475569"),
            "GoToToday_btn": style("#f1f5f9", "#334155", "#cbd5e1", "#e2e8f0", "#334155", "#e2e8f0", "#64748b", "#475569"),
            "UpDir_btn": style("#f1f5f9", "#334155", "#cbd5e1", "#e2e8f0", "#334155", "#e2e8f0", "#64748b", "#475569"),
            "BrowseRootdir_btn": style("#f1f5f9", "#334155", "#cbd5e1", "#e2e8f0", "#334155", "#e2e8f0", "#64748b", "#475569"),
            "PC_restore": style("#f1f5f9", "#334155", "#cbd5e1", "#e2e8f0", "#334155", "#e2e8f0", "#64748b", "#475569"),
            "twoD_PC_restore": style("#f1f5f9", "#334155", "#cbd5e1", "#e2e8f0", "#334155", "#e2e8f0", "#64748b", "#475569"),
            "twoD_PC_sync": style("#f1f5f9", "#334155", "#cbd5e1", "#e2e8f0", "#334155", "#e2e8f0", "#64748b", "#475569"),
        }

        # Apply mapped styles
        for btn_name, st in button_map.items():
            btn = getattr(self, btn_name, None)
            if btn is not None:
                btn.setStyleSheet(st)

        # Fallback for any unmapped QPushButtons
        fallback_style = style("#f1f5f9", "#334155", "#cbd5e1", "#e2e8f0", "#334155", "#e2e8f0", "#64748b", "#475569")
        for btn in self.findChildren(QPushButton):
            if not btn.styleSheet():
                btn.setStyleSheet(fallback_style)

        # Ensure UpDir_btn is wide enough
        if hasattr(self, "UpDir_btn") and self.UpDir_btn is not None:
            self.UpDir_btn.setMinimumWidth(self.UpDir_btn.sizeHint().width() + 10)

    def closeEvent(self, event):
        try:
            lst = getattr(self, "PP_datafolderlist_lst", None)
            if lst is not None:
                lst.removeEventFilter(self)
        except Exception:
            pass
        try:
            lst_2d = getattr(self, "twoD_datafolderlist_lst", None)
            if lst_2d is not None:
                lst_2d.removeEventFilter(self)
        except Exception:
            pass
        pop_win = getattr(self, "_twoD_pop_window", None)
        if pop_win is not None:
            pop_win.close()
            pop_win.deleteLater()
            self._twoD_pop_window = None
        super().closeEvent(event)

    def handle_reload(self):
        """Reload the dataset for the currently active tab (1D or 2D)."""
        tabs = getattr(self, "MainTabs", None)
        current_tab_idx = tabs.currentIndex() if tabs is not None else 0
        if current_tab_idx == 1:
            self.twoD_reload_data()
        else:
            self.reload_data()

    def eventFilter(self, obj, event):
        try:
            if event is not None and event.type() == QEvent.Type.KeyPress:
                key = event.key()
                if key in (Qt.Key.Key_Up, Qt.Key.Key_Down):
                    focus_w = QApplication.focusWidget()
                    from PyQt6.QtWidgets import QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox, QComboBox
                    if focus_w is not None and isinstance(focus_w, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox, QComboBox)):
                        return False

                    lst_1d = getattr(self, "PP_datafolderlist_lst", None)
                    lst_2d = getattr(self, "twoD_datafolderlist_lst", None)
                    if obj in (lst_1d, lst_2d):
                        tabs = getattr(self, "MainTabs", None)
                        current_tab_idx = tabs.currentIndex() if tabs is not None else 0
                        if current_tab_idx in (0, 1):
                            if self._navigate_dataset_browser(forward=(key == Qt.Key.Key_Down)):
                                return True
        except Exception:
            pass
        return False

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            focus_w = QApplication.focusWidget()
            from PyQt6.QtWidgets import QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox, QComboBox
            if focus_w is not None and isinstance(focus_w, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox, QComboBox)):
                super().keyPressEvent(event)
                return

            tabs = getattr(self, "MainTabs", None)
            current_tab_idx = tabs.currentIndex() if tabs is not None else 0
            if current_tab_idx in (0, 1):
                if self._navigate_dataset_browser(forward=(key == Qt.Key.Key_Down)):
                    event.accept()
                    return

        super().keyPressEvent(event)

    def _navigate_dataset_browser(self, forward: bool) -> bool:
        """Navigate to the next/previous dataset item in the active tab's browser list."""
        tabs = getattr(self, "MainTabs", None)
        current_tab_idx = tabs.currentIndex() if tabs is not None else 0

        if current_tab_idx == 0:
            lst = getattr(self, "PP_datafolderlist_lst", None)
            model = getattr(self, "_folder_model", None)
        elif current_tab_idx == 1:
            lst = getattr(self, "twoD_datafolderlist_lst", None)
            model = getattr(self, "_twoD_folder_model", None)
        else:
            return False

        if lst is None or model is None or model.rowCount() == 0:
            return False

        curr_index = lst.currentIndex()
        curr_row = curr_index.row() if curr_index.isValid() else -1

        if forward:
            start_row = curr_row + 1 if curr_row >= 0 else 0
            step = 1
            stop_row = model.rowCount()
        else:
            start_row = curr_row - 1 if curr_row >= 0 else model.rowCount() - 1
            step = -1
            stop_row = -1

        target_row = None
        # First pass: look for non-directory dataset item
        for r in range(start_row, stop_row, step):
            item = model.item(r)
            if item is not None and item.isEnabled():
                is_dir = item.data(Qt.ItemDataRole.UserRole + 2)
                if not is_dir:
                    target_row = r
                    break

        # Fallback pass: any enabled item if no non-dir dataset found
        if target_row is None:
            for r in range(start_row, stop_row, step):
                item = model.item(r)
                if item is not None and item.isEnabled():
                    target_row = r
                    break

        if target_row is not None and target_row != curr_row:
            new_index = model.index(target_row, 0)
            lst.setCurrentIndex(new_index)
            lst.scrollTo(new_index)
            return True

        return False

    @staticmethod
    def _load_project_settings():
        """Load the project settings.toml so data_dir/profile/... are active."""
        try:
            pm.load_settings()
        except Exception:
            logger.warning(
                "Could not load settings.toml; the built-in defaults are used.", exc_info=True
            )

    def _apply_tab_theme(self):
        """Apply a dark tab style sheet to MainTabs when the palette is dark.

        The .ui ships a light tab gradient with no text colour; in dark mode that
        is unreadable, so override it at runtime. Light mode keeps the .ui style.
        """
        from .theme import dark_tab_stylesheet, is_dark_palette

        tabs = getattr(self, "MainTabs", None)
        if tabs is not None and is_dark_palette():
            tabs.setStyleSheet(dark_tab_stylesheet())

    def _apply_icons(self):
        """Restore the window and toolbar-button icons from shipped files."""
        from PyQt6.QtCore import QSize
        from PyQt6.QtGui import QIcon

        win_icon = _ICON_DIR / "pirate-hat.png"
        if win_icon.exists():
            self.setWindowIcon(QIcon(str(win_icon)))
        for name, fname in _BUTTON_ICONS.items():
            widget = getattr(self, name, None)
            path = _ICON_DIR / fname
            if widget is not None and path.exists() and hasattr(widget, "setIcon"):
                widget.setIcon(QIcon(str(path)))
                widget.setIconSize(QSize(20, 20))
        for name, fname in _ACTION_ICONS.items():
            action = getattr(self, name, None)
            path = _ICON_DIR / fname
            if action is not None and path.exists() and hasattr(action, "setIcon"):
                action.setIcon(QIcon(str(path)))

    # ----------------------------------------------------------------- #
    #                              Wiring                               #
    # ----------------------------------------------------------------- #
    def _wire(self):
        combo = getattr(self, "PP_datatype_cbx", None)
        if combo is not None:
            default_type = getattr(pm.get_settings(), "default_oneD_datatype", "PDAT")
            populate_datatype_combo(combo, pm.available_loaders(), default_type)

        # Dataset browser: root folder (typed or selected) + a list of files
        # matching the selected data type's extension.
        self._folder_model = QStandardItemModel(self)
        lst = getattr(self, "PP_datafolderlist_lst", None)
        if lst is not None:
            lst.setModel(self._folder_model)
            # Load on selection change: single click AND arrow-key navigation.
            lst.selectionModel().currentChanged.connect(self._on_dataset_activated)
            # Double-click enters a folder ([..]/[name]); datasets load on selection.
            lst.doubleClicked.connect(self._on_item_double_clicked)
            lst.installEventFilter(self)
        if combo is not None:
            combo.currentTextChanged.connect(self.refresh_dataset_list)
        self._connect_text("RootDir_field", self._on_rootdir_changed)
        self._connect("RefreshDir_btn", self._force_refresh_dataset_lists)
        self._connect("UpDir_btn", self._go_up_dir)
        self._connect("GoToToday_btn", self._go_to_today)
        self._connect("BrowseRootdir_btn", self.browse_rootdir)

        # Store references for Phasing/FT tab hide/show
        self._twoD_phasingft_tab_widget = getattr(self, "twoD_phasingft_tab", None)
        self._twoD_phasingft_tab_title = ""
        subtabs = getattr(self, "twoD_subtabs", None)
        if subtabs is not None and self._twoD_phasingft_tab_widget is not None:
            for i in range(subtabs.count()):
                if subtabs.widget(i) == self._twoD_phasingft_tab_widget:
                    self._twoD_phasingft_tab_title = subtabs.tabText(i)
                    break

        # 2D dataset browser model & list
        self._twoD_folder_model = QStandardItemModel(self)
        lst_2d = getattr(self, "twoD_datafolderlist_lst", None)
        if lst_2d is not None:
            lst_2d.setModel(self._twoD_folder_model)
            lst_2d.selectionModel().currentChanged.connect(self._on_twoD_dataset_activated)
            lst_2d.doubleClicked.connect(self._on_twoD_item_double_clicked)
            lst_2d.installEventFilter(self)

        # Wire up 2D tab
        combo_2d = getattr(self, "twoD_datatype_cbx", None)
        if combo_2d is not None:
            default_type_2d = getattr(pm.get_settings(), "default_twoD_datatype", "P2DAT")
            populate_datatype_combo(combo_2d, pm.twoD.available_map_loaders(), default_type_2d)
            combo_2d.currentTextChanged.connect(self._on_twoD_datatype_changed)
            self._on_twoD_datatype_changed(combo_2d.currentText())

        # Set t2 delay field title
        delay_lbl = getattr(self, "twoD_t2delay_lbl", None)
        if delay_lbl is not None:
            delay_lbl.setText("t₂ delay")

        # Set gap spacing in selectors layout
        selectors_layout = getattr(self, "twoD_selectors_layout", None)
        if selectors_layout is not None:
            selectors_layout.setSpacing(15)

        # Wire "Apply and reprocess" button in Phasing/FT sub-tab
        self._connect("twoD_apply_reprocess_btn", self.twoD_reload_data)

        self._twoD_delay_model = QStandardItemModel(self)
        lst_delays = getattr(self, "twoD_t2delay_lst", None)
        if lst_delays is not None:
            lst_delays.setModel(self._twoD_delay_model)
            lst_delays.selectionModel().currentChanged.connect(self._on_twoD_delay_activated)

        self._connect("twoD_reloadData_btn", self.twoD_reload_data)

        # Connect scatter delay subtraction widgets
        _scat_chk = getattr(self, "twoD_sub_scat_chk", None)
        if _scat_chk is not None:
            _scat_chk.toggled.connect(self._twoD_rerender_preview)
        _scat_delay = getattr(self, "twoD_sub_scat_delay", None)
        if _scat_delay is not None:
            _scat_delay.valueChanged.connect(self._twoD_rerender_preview)

        # Initialize Phasing/FT tab visibility
        if combo_2d is not None:
            self._on_twoD_datatype_changed(combo_2d.currentText())

        # Connect overlay checkbox
        _include_ftir = getattr(self, "twoD_include_ftir_chk", None)
        if _include_ftir is not None:
            _include_ftir.toggled.connect(self._on_twoD_include_ftir_toggled)

        # Connect probe autocalibration checkbox
        _auto_cal_chk = getattr(self, "twoD_auto_cal_chk", None)
        if _auto_cal_chk is not None:
            _auto_cal_chk.toggled.connect(self._on_twoD_auto_cal_toggled)

        self._connect("twoD_contour_btn", lambda: self._plot_twoD_cut("contour"))
        self._connect("twoD_kinetics_btn", self._on_twoD_kinetics_clicked)
        self._connect("twoD_surf_btn", lambda: self._plot_twoD_cut("surface"))
        self._connect("twoD_slices_btn", self._on_twoD_slices_clicked)
        self._connect("twoD_spectral_diffusion_btn", self._on_twoD_spectral_diffusion_clicked)
        self._connect("twoD_timedomain_fit_btn", self._on_twoD_timedomain_fit_clicked)
        self._connect("twoD_make_gif_btn", self._on_twoD_make_gif_clicked)
        self._connect("twoD_save_probe_cal_btn", self._twoD_save_probe_cal)
        self._connect("twoD_apply_changes_btn", self.twoD_reload_data)
        self._connect("twoD_PC_sync", self._on_twoD_sync_limits)

        # Wire up subtabs currentChanged
        _subtabs = getattr(self, "twoD_subtabs", None)
        if _subtabs is not None:
            _subtabs.currentChanged.connect(self._on_twoD_subtabs_changed)

        self._connect("PP_reloadData_btn", self.reload_data)
        self._connect("PP_CtrPlot_btn", lambda: self._plot_cut("contour"))
        self._connect("PP_SurfPlot_btn", lambda: self._plot_cut("surface"))
        self._connect("PP_trSpecPlot_btn", lambda: self._plot_cut("spectra"))
        self._connect("PP_kineticsPlot_btn", lambda: self._plot_cut("kinetics"))
        self._connect("PP_ScanKinetics_btn", lambda: self._plot_scan_cut("kinetics"))
        self._connect("PP_ScanSpectra_btn", lambda: self._plot_scan_cut("spectra"))
        self._connect("PP_RecalcAvg_btn", self._recalc_average)
        self._connect("PP_plotBkg_btn", self._plot_background)
        self._connect("PP_plotNoise_btn", self._plot_noise)
        self._connect("PP_plotCounts_btn", self._plot_counts)
        self._connect("PP_fitChirp_btn", self._fit_chirp)
        self._connect("PP_loadChirpCorr_btn", self._load_chirp_corr)
        self._connect("PP_subtractSolvent_btn", self._subtract_solvent)
        self._connect("PP_subtractShockwave_btn", self._subtract_shockwave)
        self._connect("PP_maskProbe_btn", self._mask_probe_region)
        self._connect("PP_shiftT0_btn", self._shift_t0)

        # Background-subtraction controls re-render the embedded preview. The
        # tmin/tmax limits recompute only on commit (Enter / focus-out), where the
        # entered value is first snapped to the nearest delay; the checkbox is live.
        for _name in ("PP_bkg_tmin", "PP_bkg_tmax"):
            _box = getattr(self, _name, None)
            if _box is not None and hasattr(_box, "editingFinished"):
                _box.editingFinished.connect(self._on_bkg_limit_entered)
        _sub = getattr(self, "PP_SubtactBkg_chk", None)
        if _sub is not None and hasattr(_sub, "toggled"):
            _sub.toggled.connect(self._rerender_preview)

        _quick = getattr(self, "PP_QuickPlots_chk", None)
        if _quick is not None and hasattr(_quick, "toggled"):
            _quick.blockSignals(True)
            _quick.setChecked(bool(pm.get_settings().quick_plots))
            _quick.blockSignals(False)
            _quick.toggled.connect(self._on_quick_plots_toggled)

        tabs = getattr(self, "MainTabs", None)
        if tabs is not None:
            tabs.currentChanged.connect(self._on_main_tabs_changed)
            self._on_main_tabs_changed(tabs.currentIndex())

    def _connect(self, name: str, slot):
        widget = getattr(self, name, None)
        if widget is not None and hasattr(widget, "clicked"):
            widget.clicked.connect(slot)

    def _connect_text(self, name: str, slot):
        widget = getattr(self, name, None)
        if widget is not None and hasattr(widget, "editingFinished"):
            widget.editingFinished.connect(slot)

    def _on_quick_plots_toggled(self, checked: bool):
        """Toggle quick_plots setting and refresh preview."""
        pm.update_settings(quick_plots=checked)
        self._rerender_preview()

    # ----------------------------------------------------------------- #
    #                              Menus                                #
    # ----------------------------------------------------------------- #
    def _build_menus(self):
        """Connect the menu actions defined in the .ui to their handlers.

        The actions, shortcuts, separators and menu structure now live in
        ``main_window.ui``; here we only wire each action (bound by objectName
        via ``uic.loadUi``) to its slot.
        """
        wiring = {
            "actionOpen": self.open_file,
            "actionReload": self.handle_reload,
            "actionSetRoot": self.browse_rootdir,
            "actionToday": self._go_to_today,
            "actionSettings": self._open_settings_dialog,
            "actionRestoreWindowSize": self._restore_default_window_size,
            "actionQuit": self.close,
            "actionLoadSpectrum": self._load_spectrum,
            "actionSubtractShockwave": self._subtract_shockwave,
            "actionTimeDerivative": self._calculate_time_derivative,
            "actionLoadAbsorption": lambda _=False: self._load_steady_state("absorption"),
            "actionLoadEmission": lambda _=False: self._load_steady_state("emission"),
            "actionClearSteadyState": self._clear_steady_state,
            "actionExportFigure": self._export_figure,
            "actionExportData": self._export_data,
            "actionExportPDAT": self._export_pdat,
            "actionExportP2DAT": self._export_p2dat,
            "actionManual": self._open_manual,
            "actionAbout": self._about,
        }
        for name, slot in wiring.items():
            action = getattr(self, name, None)
            if action is not None:
                action.triggered.connect(slot)

        if hasattr(self, "actionReload") and self.actionReload is not None:
            self.actionReload.setShortcut(QKeySequence(Qt.Key.Key_F5))

    # --- Menu slots ---------------------------------------------------- #
    def _restore_default_window_size(self):
        """Reset the main window to the default size declared in ``main_window.ui``.

        The window is un-maximised/un-minimised first (resizing a maximised
        window has no visible effect), then resized and re-centred on the
        screen it currently occupies.  The target size is clipped to the
        available screen area to remain usable on low-resolution displays.
        """
        from PyQt6.QtCore import QSize

        default = getattr(self, "_default_size", None)
        if default is None or not default.isValid():
            default = QSize(1100, 880)

        if self.isMaximized() or self.isFullScreen() or self.isMinimized():
            self.showNormal()

        screen = self.screen() or QApplication.primaryScreen()
        width, height = default.width(), default.height()
        if screen is not None:
            available = screen.availableGeometry()
            width = min(width, available.width())
            height = min(height, available.height())

        self.resize(width, height)
        if screen is not None:
            frame = self.frameGeometry()
            frame.moveCenter(screen.availableGeometry().center())
            self.move(frame.topLeft())

    def _open_settings_dialog(self):
        """Show the Aesthetics/Settings panel in a modal dialog.

        Changes take effect immediately (live preview).  "Save permanently"
        writes the current settings to ``settings.toml`` so they are restored
        on the next launch; "Close" keeps them for the current session only.
        """
        from .settings_panel import SettingsPanel

        old_load_single = bool(getattr(pm.get_settings(), "load_single_scans", False))

        dlg = QDialog(self)
        dlg.setWindowTitle("Aesthetics & Settings")
        layout = QVBoxLayout(dlg)
        panel = SettingsPanel(dlg)
        panel.changed.connect(self._rerender_preview)
        layout.addWidget(panel)

        # Force layout update to compute default size hint correctly
        layout.activate()
        hint = dlg.sizeHint()

        # Make the dialog 20% broader and 30% taller than default size hint,
        # but ensure it is at least 550x800 to fully fit all Aesthetics settings.
        width = max(int(hint.width() * 1.2), 550)
        height = max(int(hint.height() * 1.3), 800)

        # Cap the height to 90% of the screen geometry to prevent overflow on low-res screens
        screen = self.screen()
        if screen:
            screen_h = screen.availableGeometry().height()
            height = min(height, int(screen_h * 0.9))

        dlg.resize(width, height)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close
        )
        # Re-label the Save button so the intent is unambiguous.
        save_btn = buttons.button(QDialogButtonBox.StandardButton.Save)
        save_btn.setText("Save permanently")

        def _save_settings():
            cfg = pm.settings_path()
            try:
                pm.save_settings(cfg)
                self.statusBar().showMessage(f"Settings saved to {cfg}", 4000)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, "Save failed", str(exc))

        buttons.accepted.connect(_save_settings)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)
        dlg.exec()

        if bool(getattr(pm.get_settings(), "load_single_scans", False)) != old_load_single:
            self.reload_data()

    def _export_figure(self):
        """Save the embedded preview figure to an image/PDF."""
        if self.dataset is None:
            QMessageBox.information(self, "Export figure", "Load a dataset first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save figure", "", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )
        if not path:
            return
        from .theme import style_figure

        fig = self.PPaxes.figure
        try:
            # Always export on a white background, regardless of the GUI theme,
            # then restore the on-screen (possibly dark) styling.
            style_figure(fig, dark=False)
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="white")
            self.statusBar().showMessage(f"Saved figure to {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
        finally:
            self._style_preview(self.PPaxes)
            self.PPaxes.canvas.draw_idle()

    def _export_data(self):
        """Export the processed signal (detector 0) as a delays x probe CSV."""
        if self.dataset is None:
            QMessageBox.information(self, "Export data", "Load a dataset first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export processed data", "", "CSV (*.csv);;All files (*)"
        )
        if not path:
            return
        try:
            self._apply_background()
            z = np.asarray(self.dataset.Z)[:, :, 0]  # [Ndelays x Npixels]
            probe = np.asarray(self.dataset.probe, dtype=float)
            delays = np.asarray(self.dataset.delays, dtype=float)
            table = np.empty((z.shape[0] + 1, z.shape[1] + 1))
            table[0, 0] = np.nan
            table[0, 1:] = probe
            table[1:, 0] = delays
            table[1:, 1:] = z
            np.savetxt(
                path,
                table,
                delimiter=",",
                header="delay\\probe (mOD); first row = probe axis, first col = delays",
            )
            self.statusBar().showMessage(f"Exported data to {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Export failed", str(exc))

    def _export_pdat(self):
        """Export the current 1D dataset as PDAT [+.pdatn (noise) file]."""
        if self.dataset is None:
            QMessageBox.information(self, "Export PDAT", "Load a dataset first.")
            return

        import re

        n_det = getattr(self.dataset, "n_detectors", 1)

        default_dir = self._rootdir_text() or ""
        stem = Path(self._current_path).stem if self._current_path else "dataset"
        if not stem.endswith("_PROCESSED"):
            stem = f"{stem}_PROCESSED"

        if n_det > 1:
            if not re.search(r"_DET\d+", stem, flags=re.IGNORECASE):
                default_name = f"{stem}_DET1.pdat"
            else:
                default_name = f"{stem}.pdat"
        else:
            default_name = f"{stem}.pdat"

        default_path = os.path.join(default_dir, default_name)

        path, _ = QFileDialog.getSaveFileName(
            self, "Export PDAT data", default_path, "PDAT (*.pdat);;All files (*)"
        )
        if not path:
            return
        try:
            self._apply_background()

            delays = np.asarray(self.dataset.delays, dtype=float)
            noise_arr = self.dataset.noise_array()

            lbl = self.dataset.units.get("unitsL_lbl", "")
            if lbl == "Wavenumber":
                probe_unit = "cm^{-1}"
            elif lbl == "Wavelength":
                probe_unit = "nm"
            elif lbl == "Energy":
                probe_unit = "eV"
            else:
                probe_unit = "nm"
            time_unit = self.dataset.units.get("unitsT_ltx", "ps")
            header_line = f"{time_unit}*{probe_unit}"

            exported_paths = []
            base_pdat_path = Path(path)
            if base_pdat_path.suffix.lower() != ".pdat":
                base_pdat_path = base_pdat_path.with_suffix(".pdat")

            for d in range(n_det):
                det_num = d + 1
                if n_det > 1:
                    base_stem = base_pdat_path.stem
                    if re.search(r"_DET\d+", base_stem, flags=re.IGNORECASE):
                        target_stem = re.sub(r"_DET\d+", f"_DET{det_num}", base_stem, flags=re.IGNORECASE)
                    else:
                        target_stem = f"{base_stem}_DET{det_num}"
                    pdat_path = base_pdat_path.with_name(target_stem + ".pdat")
                else:
                    pdat_path = base_pdat_path

                probe_d = np.asarray(self.dataset._detector_probe(d), dtype=float)
                z_d = np.asarray(self.dataset.Z)
                if z_d.ndim == 3:
                    z_d = z_d[:, :, d]

                pdat_table = np.empty((z_d.shape[0] + 1, z_d.shape[1] + 1))
                pdat_table[0, 0] = 0.0
                pdat_table[0, 1:] = probe_d
                pdat_table[1:, 0] = delays
                pdat_table[1:, 1:] = z_d

                with open(pdat_path, "w", newline="") as f:
                    f.write(f"{header_line}\n")
                    np.savetxt(f, pdat_table, delimiter=",", fmt="%.18g")
                exported_paths.append(pdat_path)

                if noise_arr is not None:
                    z_noise = np.asarray(noise_arr)
                    if z_noise.ndim == 3:
                        z_noise = z_noise[:, :, d]
                    pdatn_table = np.empty((z_noise.shape[0] + 1, z_noise.shape[1] + 1))
                    pdatn_table[0, 0] = 0.0
                    pdatn_table[0, 1:] = probe_d
                    pdatn_table[1:, 0] = delays
                    pdatn_table[1:, 1:] = z_noise

                    pdatn_path = pdat_path.with_suffix(".pdatn")
                    with open(pdatn_path, "w", newline="") as f:
                        f.write(f"{header_line}\n")
                        np.savetxt(f, pdatn_table, delimiter=",", fmt="%.18g")

            msg_path = "\n".join(str(p) for p in exported_paths)
            QMessageBox.information(self, "Export complete", f"PDAT file(s) exported at:\n{msg_path}")
            self.statusBar().showMessage(f"Exported PDAT to {exported_paths[0]}")
        except Exception as exc:
            QMessageBox.warning(self, "Export failed", str(exc))

    def _export_p2dat(self):
        """Export the current 2D dataset as a P2DAT file."""
        if self.twoD_dataset is None:
            QMessageBox.information(self, "Export P2DAT", "Load a 2D dataset first.")
            return

        default_dir = self._rootdir_text() or ""
        default_name = "dataset_PROCESSED.p2dat"
        if self._twoD_current_path:
            default_name = Path(self._twoD_current_path).name + "_PROCESSED.p2dat"

        default_path = os.path.join(default_dir, default_name)

        path, _ = QFileDialog.getSaveFileName(
            self, "Export P2DAT data", default_path, "P2DAT (*.p2dat);;All files (*)"
        )
        if not path:
            return
        try:
            Z = np.asarray(self.twoD_dataset.Z)
            pump = np.asarray(self.twoD_dataset.pump)
            probe = np.asarray(self.twoD_dataset.probe)
            delays = np.asarray(self.twoD_dataset.delays)

            PUMP, PROBE = np.meshgrid(pump, probe, indexing="ij")
            pump_flat = PUMP.ravel(order="F")
            probe_flat = PROBE.ravel(order="F")

            n_rows = len(pump_flat) + 1
            n_cols = 2 + len(delays)

            out = np.zeros((n_rows, n_cols))
            out[0, 0] = 0.0
            out[0, 1] = 0.0
            out[0, 2:] = delays
            out[1:, 0] = pump_flat
            out[1:, 1] = probe_flat

            for i in range(len(delays)):
                out[1:, 2 + i] = Z[:, :, i].ravel(order="F")

            np.savetxt(path, out, delimiter=",", fmt="%.18g")

            QMessageBox.information(self, "Export complete", f"P2DAT file exported at: {path}")
            self.statusBar().showMessage(f"Exported P2DAT to {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Export failed", str(exc))

    def _on_main_tabs_changed(self, index: int):
        """Enable / disable PDAT / P2DAT / oneD actions based on current tab index."""
        act_1d = getattr(self, "actionExportPDAT", None)
        act_2d = getattr(self, "actionExportP2DAT", None)
        act_deriv = getattr(self, "actionTimeDerivative", None)
        if act_1d is not None:
            act_1d.setEnabled(index == 0)
        if act_2d is not None:
            act_2d.setEnabled(index == 1)
        if act_deriv is not None:
            act_deriv.setEnabled(index == 0 and getattr(self, "dataset", None) is not None)
        if index == 0 and getattr(self, "plot_controls", None) is not None:
            self.plot_controls.sync_from_settings()
        elif index == 2:
            self._ensure_calibration_initialized()
        # Keep the dataset browser of the newly shown tab in sync with the
        # (shared) root folder and its current contents on disk.
        self._sync_dataset_lists(index)

    def _open_manual(self):
        """Open the bundled manual PDF with the system default application."""
        manual = Path(pm.__file__).resolve().parent.parent.parent / "docs" / "main.pdf"
        if not manual.is_file():
            QMessageBox.information(self, "Manual", f"Manual not found at {manual}.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(manual)))

    def _about(self):
        """Show the modern dark-themed About dialog."""
        from pymorgan.__about__ import __author__, __email__

        from .about_dialog import ModernAboutDialog

        email = __email__.replace("[at]", "@")
        icons_dir = Path(__file__).with_name("icons")
        banner_path = icons_dir / "window.png"
        icon_path = icons_dir / "pirate-hat.png"
        manual_path = Path(pm.__file__).resolve().parent.parent.parent / "docs" / "main.pdf"

        dlg = ModernAboutDialog(
            parent=self,
            title="About PyMORGAN",
            app_name="PyMORGAN",
            version=pm.__version__,
            subtitle="Multidimensional Optical Spectroscopy Graphical Analysis Interface",
            description=(
                "PyMORGAN provides interactive plotting and analysis of ultrafast "
                "time-resolved spectroscopy (1-D, 2-D and steady-state) data."
            ),
            author=__author__,
            department="Department of Physical Chemistry",
            institution="University of Geneva, Switzerland",
            contact_email=email,
            website_url="https://www.unige.ch/sciences/chifi/fernandez-teran/",
            license_name="AGPL-3.0 License",
            github_url="https://github.com/RJFernandezTeran/PyMORGAN",
            banner_path=str(banner_path) if banner_path.exists() else None,
            icon_path=str(icon_path) if icon_path.exists() else None,
            manual_pdf_path=str(manual_path) if manual_path.exists() else None,
            ai_credit="Developed with AI assistance from <b>Google Antigravity</b>.",
        )
        dlg.exec()

    def _retain_hidden_space(self):
        """Reserve layout space for the dataset widgets even while hidden.

        With the responsive layouts, hiding these widgets before a dataset is
        loaded would otherwise collapse the layout and make it jump to the
        correct proportions only once data appears. ``retainSizeWhenHidden``
        keeps the pre-load arrangement identical to the loaded one.
        """
        for name in _DATASET_WIDGETS + _DATASET_WIDGETS_2D:
            widget = getattr(self, name, None)
            if widget is not None:
                sp = widget.sizePolicy()
                sp.setRetainSizeWhenHidden(True)
                widget.setSizePolicy(sp)

    def _set_dataset_widgets_visible(self, visible: bool):
        """Show the dataset-dependent panels and plot area only when loaded."""
        for name in _DATASET_WIDGETS:
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setVisible(visible)

    def _set_twoD_dataset_widgets_visible(self, visible: bool):
        """Show the 2D dataset-dependent panels and plot area only when loaded."""
        for name in _DATASET_WIDGETS_2D:
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setVisible(visible)

    def _init_plot_controls(self):
        """Bind the plot-controls panel (``PC_box``, declared in the .ui)."""
        from .plot_controls import PlotControlsPanel, PlotControlsPanel2D

        # Contour re-plots cannot be updated artist-by-artist (the level set
        # itself changes), so rapid control changes -- a Z-scale slider drag
        # emits one signal per step -- are coalesced into a single render.
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(_RENDER_DEBOUNCE_MS)
        self._render_timer.timeout.connect(self._rerender_preview)
        self._twoD_render_timer = QTimer(self)
        self._twoD_render_timer.setSingleShot(True)
        self._twoD_render_timer.setInterval(_RENDER_DEBOUNCE_MS)
        self._twoD_render_timer.timeout.connect(self._twoD_rerender_preview)

        if getattr(self, "PC_box", None) is None:
            self.plot_controls = None
        else:
            self.plot_controls = PlotControlsPanel(self)
            self.plot_controls.renderRequested.connect(self._render_timer.start)
            self.plot_controls.limitsChanged.connect(self._apply_view_limits)

        if getattr(self, "twoD_PC_box", None) is None:
            self.twoD_plot_controls = None
        else:
            self.twoD_plot_controls = PlotControlsPanel2D(self)
            self.twoD_plot_controls.renderRequested.connect(self._twoD_render_timer.start)
            self.twoD_plot_controls.limitsChanged.connect(self._twoD_apply_view_limits)

    def _rerender_preview(self):
        """Re-draw the embedded contour (Z scale or a global setting changed)."""
        current_tab_idx = self.MainTabs.currentIndex()
        if current_tab_idx == 0:
            if self.plot_controls is not None:
                self.plot_controls.sync_from_settings()
            if self.dataset is not None:
                self._preview_contour()
        elif current_tab_idx == 1:
            if self.twoD_dataset is not None:
                if self.twoD_plot_controls is not None:
                    s = pm.get_settings()
                    if self.twoD_plot_controls.cmb_cmap is not None:
                        self.twoD_plot_controls.cmb_cmap.setCurrentText(str(s.cmap))
                self._twoD_preview_contour()

    # ----------------------------------------------------------------- #
    #                     Shared prompts / helpers                      #
    # ----------------------------------------------------------------- #

    def _ask_values(
        self, title: str, label: str, default: list, all_values: list | np.ndarray | None = None
    ) -> list | None:
        """Prompt for cuts: numbers, MATLAB ranges (``1:0.1:5``) or ``all``.

        The parsing lives in :func:`pymorgan.gui.dialogs.ask_values` so that
        every window asking "which delays?" -- here and in PyRATE-TA -- accepts
        exactly the same syntax.
        """
        from .dialogs import ask_values

        return ask_values(self, title, label, default, all_values)

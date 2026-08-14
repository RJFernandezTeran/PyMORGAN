"""Matplotlib canvas widget for rendering 2x2 detector calibration plot grids."""

from __future__ import annotations

import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QWidget

from pymorgan.gui.theme import adapt_axes_traces, base_line_color, is_dark_palette, style_figure


def _style_cal_figure(fig):
    """Theme an *embedded* calibration figure, brightening its traces if dark.

    The calibration plots are drawn with fixed colours (black reference, red
    convolved reference, purple experimental, ...) that are unreadable on the
    dark background, so :func:`~pymorgan.gui.theme.adapt_axes_traces` maps them
    to brighter equivalents. Text colours are left to that function as well
    (``style_text=False``), because some axis labels are deliberately coloured to
    match their trace. Pop-out windows keep the light figure and the original
    colours.
    """
    dark = is_dark_palette()
    style_figure(fig, dark, style_text=False)
    for ax in fig.axes:
        adapt_axes_traces(ax, dark)
    return fig


class CalibrationCanvas(FigureCanvas):
    """Canvas widget containing a 2x2 grid of subplots for detector calibration visualizer.

    Grid arrangement:
      - (1,1) Top-Left: Raw Intensities vs Pixel (shares X with 2,1)
      - (2,1) Bottom-Left: Absorbance Spectrum vs Pixel
      - (1,2) Top-Right: Exp (Red) vs Reference (Black) Spectrum
      - (2,2) Bottom-Right: Calibration Fit Curve (Pixel vs Wavelength/Wavenumber)
    """

    def __init__(self, parent=None, detector_name: str = "Detector 1"):
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.fig.patch.set_facecolor("white")

        super().__init__(self.fig)
        self.setParent(parent)

        self.detector_name = detector_name
        self.unit_mode = "nm"  # "nm" or "cm-1"
        self.abs_mode = "calculated"  # "calculated" or "corrected_ref"

        # Create 2x2 subplots with shared X for column 1
        self.ax_raw = self.fig.add_subplot(2, 2, 1)
        self.ax_abs = self.fig.add_subplot(2, 2, 3, sharex=self.ax_raw)
        self.ax_ref = self.fig.add_subplot(2, 2, 2)
        self.ax_fit = self.fig.add_subplot(2, 2, 4)

        self.fig.subplots_adjust(left=0.08, right=0.96, top=0.96, bottom=0.08, hspace=0.08, wspace=0.18)
        self.draw_placeholder()

    def set_unit_mode(self, mode: str):
        """Set unit mode to 'nm' or 'cm-1' without clearing 1st column data."""
        self.unit_mode = mode
        is_ir = self.unit_mode in ("cm-1", "cm1", "wavenumber")
        fit_axis_label = r"Wavenumber ($\mathrm{cm}^{-1}$)" if is_ir else "Wavelength (nm)"
        self.ax_fit.set_ylabel(fit_axis_label, fontweight="bold", fontsize=8)
        self.make_draggable_legend(self.ax_fit, title="Calibration Fit")
        self.draw()

    def set_abs_mode(self, mode: str):
        """Set absorbance mode to 'calculated' or 'corrected_ref' without clearing 1st column data."""
        self.abs_mode = mode
        abs_title = (
            "Calculated Absorbance"
            if self.abs_mode == "calculated"
            else "Corrected Absorbance"
        )
        self.make_draggable_legend(self.ax_abs, title=abs_title)
        self.draw()

    def make_draggable_legend(self, ax, title: str, loc: str = "best"):
        """Add or update a legend on ax with the given title and make it draggable."""
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            leg = ax.legend(handles, labels, title=title, loc=loc, fontsize=7, title_fontsize=8)
        else:
            from matplotlib.patches import Rectangle
            dummy = Rectangle((0, 0), 0, 0, fill=False, edgecolor="none", visible=False)
            leg = ax.legend([dummy], [""], title=title, loc=loc, fontsize=7, title_fontsize=8)

        if leg is not None:
            leg.set_draggable(True)
        return leg

    def draw(self):
        """Re-apply the embedded theme (and dark-mode trace colours), then draw."""
        _style_cal_figure(self.fig)
        super().draw()

    def draw_placeholder(self):
        """Draw clean empty 2x2 axes grid with theme awareness, legend titles, and draggable legends."""
        _style_cal_figure(self.fig)

        all_axes = (self.ax_raw, self.ax_abs, self.ax_ref, self.ax_fit)
        for ax in all_axes:
            ax.clear()
            ax.set_title("")
            ax.grid(False)
            ax.tick_params(axis="both", which="major", labelsize=8, labelbottom=True)

        # Zero reference line only for raw, abs, and ref plots (not dispersion fit)
        for ax in (self.ax_raw, self.ax_abs, self.ax_ref):
            ax.axhline(0, color="0.75", lw=0.75, zorder=0)

        # (1,1) Top-Left: Raw Intensities vs Pixel
        self.ax_raw.set_xlabel("Pixel", fontweight="bold", fontsize=8)
        self.ax_raw.set_ylabel("Counts", fontweight="bold", fontsize=8)
        self.make_draggable_legend(self.ax_raw, title="Raw Intensities")

        # (2,1) Bottom-Left: Absorbance Spectrum vs Pixel
        abs_title = (
            "Calculated Absorbance"
            if self.abs_mode == "calculated"
            else "Corrected Absorbance"
        )
        self.ax_abs.set_xlabel("Pixel", fontweight="bold", fontsize=8)
        self.ax_abs.set_ylabel("Absorbance (OD)", fontweight="bold", fontsize=8)
        self.make_draggable_legend(self.ax_abs, title=abs_title)

        # (1,2) Top-Right: Exp vs Reference Spectrum Overlay
        is_ir = self.unit_mode in ("cm-1", "cm1", "wavenumber")
        ref_axis_label = r"Wavenumber ($\mathrm{cm}^{-1}$)" if is_ir else "Wavelength (nm)"
        self.ax_ref.set_xlabel(ref_axis_label, fontweight="bold", fontsize=8)
        self.ax_ref.set_ylabel("Norm. Absorbance", fontweight="bold", fontsize=8)
        self.make_draggable_legend(self.ax_ref, title="Reference Spectrum")

        # (2,2) Bottom-Right: Dispersion Fit Curve (Pixel vs nm / cm-1)
        fit_axis_label = r"Wavenumber ($\mathrm{cm}^{-1}$)" if is_ir else "Wavelength (nm)"
        self.ax_fit.set_xlabel("Pixel", fontweight="bold", fontsize=8)
        self.ax_fit.set_ylabel(fit_axis_label, fontweight="bold", fontsize=8)
        self.make_draggable_legend(self.ax_fit, title="Calibration Fit")

        # Re-apply theme styling and adjust hspace & wspace for clear layout
        _style_cal_figure(self.fig)
        self.fig.subplots_adjust(left=0.09, right=0.96, top=0.96, bottom=0.12, hspace=0.30, wspace=0.22)
        self.draw()


class SubplotPopUpDialog(QDialog):
    """Pop-up dialog displaying a single enlarged subplot with an independent Matplotlib toolbar."""

    def __init__(self, source_ax, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{title} (Pop-up)")
        self.resize(850, 600)

        fig = Figure(figsize=(8, 5.5), dpi=100)

        canvas = FigureCanvas(fig)
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
        toolbar = NavToolbar(canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        ax_pop = fig.add_subplot(111)

        # Check if source_ax is the host axis for a twin Y-axis (e.g. ax_ref with _ref_twin)
        twin_source = None
        if source_ax.figure is not None:
            for ax_other in source_ax.figure.axes:
                if (
                    ax_other is not source_ax
                    and getattr(ax_other, "get_label", lambda: "")() == "_ref_twin"
                    and getattr(ax_other, "_sharex", None) is source_ax
                ):
                    twin_source = ax_other
                    break

        # Clone primary curves
        for line in source_ax.get_lines():
            ax_pop.plot(
                line.get_xdata(),
                line.get_ydata(),
                color=base_line_color(line),
                linewidth=line.get_linewidth(),
                alpha=line.get_alpha(),
                linestyle=line.get_linestyle(),
                label=line.get_label(),
                zorder=line.get_zorder(),
            )

        ax_pop.set_xlim(source_ax.get_xlim())
        ax_pop.set_ylim(source_ax.get_ylim())
        ax_pop.set_xlabel(source_ax.get_xlabel(), fontweight="bold")
        ax_pop.set_ylabel(source_ax.get_ylabel(), fontweight="bold")
        ax_pop.grid(False)
        ax_pop.axhline(0, color="0.75", lw=0.75, zorder=0)

        # Clone twin axis if present
        if twin_source is not None:
            ax_twin_pop = ax_pop.twinx()
            ax_twin_pop.set_label("_ref_twin")
            for line in twin_source.get_lines():
                ax_twin_pop.plot(
                    line.get_xdata(),
                    line.get_ydata(),
                    color=base_line_color(line),
                    linewidth=line.get_linewidth(),
                    alpha=line.get_alpha(),
                    linestyle=line.get_linestyle(),
                    label=line.get_label(),
                    zorder=line.get_zorder(),
                )
            ax_twin_pop.set_ylim(twin_source.get_ylim())
            ax_twin_pop.set_ylabel(twin_source.get_ylabel(), fontweight="bold", color="tab:purple")
            ax_twin_pop.tick_params(axis="y", labelsize=9, labelcolor="tab:purple")

            l1, lab1 = ax_pop.get_legend_handles_labels()
            l2, lab2 = ax_twin_pop.get_legend_handles_labels()
            leg = ax_pop.legend(l1 + l2, lab1 + lab2, title=title, fontsize=8, loc="best", framealpha=0.85)
            if leg:
                leg.set_draggable(True)
        else:
            handles, labels = ax_pop.get_legend_handles_labels()
            if handles:
                leg = ax_pop.legend(handles, labels, title=title, fontsize=8, loc="best", framealpha=0.85)
                if leg:
                    leg.set_draggable(True)

        # Pop-outs stay light: the cloned traces keep their original colours.
        style_figure(fig, dark=False)
        fig.tight_layout()
        canvas.draw()


class CalibrationCanvasWidget(QWidget):
    """Container bundling a CalibrationCanvas figure, toolbar, and subplot pop-up handler."""

    def __init__(self, parent=None, detector_name: str = "Detector 1"):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar

        self.canvas = CalibrationCanvas(self, detector_name=detector_name)
        self.toolbar = NavToolbar(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Expose figure, subplots, and canvas attributes directly for convenience
        self.fig = self.canvas.fig
        self.ax_raw = self.canvas.ax_raw
        self.ax_abs = self.canvas.ax_abs
        self.ax_ref = self.canvas.ax_ref
        self.ax_fit = self.canvas.ax_fit

    def _popup_subplot(self, source_ax, title: str):
        """Open a dedicated QDialog window displaying the selected subplot in full resolution."""
        dlg = SubplotPopUpDialog(source_ax, f"{self.canvas.detector_name} - {title}", parent=self)
        dlg.exec()

    def draw(self):
        self.canvas.draw()

    def draw_placeholder(self):
        self.canvas.draw_placeholder()

    def make_draggable_legend(self, ax, title: str):
        return self.canvas.make_draggable_legend(ax, title)

    @property
    def unit_mode(self):
        return self.canvas.unit_mode

    @unit_mode.setter
    def unit_mode(self, val):
        self.canvas.unit_mode = val

    @property
    def abs_mode(self):
        return self.canvas.abs_mode

    @abs_mode.setter
    def abs_mode(self, val):
        self.canvas.abs_mode = val

    def set_unit_mode(self, mode: str):
        self.canvas.set_unit_mode(mode)

    def set_abs_mode(self, mode: str):
        self.canvas.set_abs_mode(mode)


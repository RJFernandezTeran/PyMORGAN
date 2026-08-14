"""Progress dialog for the chirp/dispersion fit, with an optional live preview.

Used by ``main_window.py`` while the Automatic/Step per-pixel fits run (the
Manual mode has no per-pixel loop and uses interactive point-picking instead,
see ``picker.py``). Exposes the one method (``wasCanceled``) that
``QProgressDialog`` callers rely on, so it drops in as a replacement, but adds
an embedded preview plot of the current pixel's data vs. fitted model when
the user checked "Preview single-pixel results" in
:class:`pymorgan.gui.chirp_dialog.ChirpFitOptionsDialog`.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QProgressBar, QVBoxLayout


class ChirpFitProgressDialog(QDialog):
    """Modal progress dialog for the per-pixel chirp fit, with an optional preview plot."""

    def __init__(self, parent=None, *, show_preview: bool = False):
        super().__init__(parent)
        self.setWindowTitle("Fit chirp")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self._canceled = False
        self._show_preview = show_preview

        layout = QVBoxLayout(self)
        self.label = QLabel("Fitting chirp correction...", self)
        self.label.setWordWrap(True)
        layout.addWidget(self.label)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        layout.addWidget(self.progress_bar)

        self._canvas = None
        self._ax = None
        self._preview_artists = None  # (data_line, fit_line, title) -- built on first set_preview
        if show_preview:
            from .canvas import MplCanvas

            self._preview = MplCanvas(self)
            self._preview.toolbar.setVisible(False)  # keep the dialog compact
            self._canvas = self._preview.canvas
            self._ax = self._preview.ax
            self._canvas.setMinimumHeight(260)
            layout.addWidget(self._preview)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel, self)
        buttons.rejected.connect(self._cancel)
        layout.addWidget(buttons)

        self.resize(440, 440 if show_preview else 150)

    def _cancel(self) -> None:
        self._canceled = True

    def wasCanceled(self) -> bool:
        """Mirror ``QProgressDialog.wasCanceled()`` so callers don't need to branch."""
        return self._canceled

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override naming)
        self._canceled = True
        super().closeEvent(event)

    def set_progress(self, i: int, n: int, msg: str) -> None:
        """Update the progress bar and label, appending a 1-based pixel count."""
        self.progress_bar.setMaximum(max(1, n))
        self.progress_bar.setValue(i)
        self.label.setText(f"{msg}\n(Pixel {i} of {n})")

    def set_preview(self, wavelength: float, delays, y, model, t0: float = 0.0) -> None:
        """Redraw the data-vs-fit preview for the pixel just processed. No-op if disabled.

        Plotted as ``delays - t0`` (not absolute delay), zoomed to +/-0.5 ps
        around the peak, so the IRF shape is what's visible regardless of
        where the dispersion places ``t0`` for this pixel. Falls back to an
        unshifted axis (and flags it in the title) if ``t0`` is non-finite,
        e.g. a skipped/failed pixel.

        Two things keep this fast enough to redraw every pixel. The data and
        fit ``Line2D``/title artists are created once and updated in place via
        ``set_data``/``set_text`` on every later call, rather than
        ``ax.clear()`` + replot -- profiling showed ``clear()`` rebuilding the
        whole axes (spines, ticks, legend) is the dominant cost, far more than
        the data itself (~10-15 ms vs <1 ms here for a few hundred points). The
        draw itself runs under matplotlib's ``"fast"`` style (relaxed path
        simplification / antialiasing thresholds meant for exactly this kind
        of frequent redraw), scoped to this call only so the rest of the app's
        plots are unaffected.
        """
        import matplotlib.style as mpl_style
        import numpy as np

        if not self._show_preview or self._ax is None:
            return
        ax = self._ax
        failed = not np.isfinite(t0)
        u = np.asarray(delays, dtype=float) - (0.0 if failed else t0)
        y = np.asarray(y, dtype=float)
        model = np.asarray(model, dtype=float)

        title = f"Pixel preview — {wavelength:.2f} nm"
        if failed:
            title += " (fit failed)"

        with mpl_style.context("fast"):
            if self._preview_artists is None:
                (data_line,) = ax.plot(u, y, "o", ms=6, mfc="none", color="0.4", label="Data")
                (fit_line,) = ax.plot(u, model, "-r", lw=2.5, alpha=0.5, label="Fit")
                ax.axvline(0, color="0.75", linewidth=0.75)
                ax.axhline(0, color="0.75", linewidth=0.75)
                ax.set_xlabel("t - t0 (ps)", fontsize=8)
                ax.set_ylabel(r"$\Delta$A", fontsize=8)
                ax.set_xlim(-0.5, 0.5)
                ax.tick_params(labelsize=9)
                ax.legend(loc="best", fontsize=9)
                title_artist = ax.set_title(title, fontsize=10)
                self._preview_artists = (data_line, fit_line, title_artist)
            else:
                data_line, fit_line, title_artist = self._preview_artists
                data_line.set_data(u, y)
                fit_line.set_data(u, model)
                title_artist.set_text(title)

            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
            self._canvas.draw()

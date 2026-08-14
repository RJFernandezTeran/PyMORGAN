"""Options dialog for the chirp/dispersion fit ("Fit Chirp" button).

Presents the three fit modes -- Automatic (VARPRO coherent-artefact fit), Manual
(interactively-picked dispersion points) and Step Function (Auto)
(Gaussian+erf-step fit) -- with their relevant options on a per-mode page. This
module only builds the Qt widgets and reads back the chosen options; the fit itself
is delegated to :mod:`pymorgan.oneD.chirp` (called from ``main_window.py``).
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

# (label, (include_1st_derivative, include_2nd_derivative))
_DERIV_OPTIONS = [
    ("None", (False, False)),
    ("1st derivative", (True, False)),
    ("2nd derivative", (False, True)),
    ("1st + 2nd derivatives", (True, True)),
]

MODE_AUTOMATIC = "Automatic"
MODE_MANUAL = "Manual"
MODE_STEP = "Step Function (Auto)"
MODE_WAVELET = "Edge Detection (Wavelet)"


class ChirpFitOptionsDialog(QDialog):
    """Mode selection + per-mode fit options for the chirp/dispersion fit."""

    def __init__(self, parent=None, *, n_detectors: int = 1, active_detector: int = 0):
        super().__init__(parent)
        self.setWindowTitle("Fit Chirp Correction")
        self._n_detectors = max(1, int(n_detectors))
        self._active_detector = max(0, min(int(active_detector), self._n_detectors - 1))

        outer = QVBoxLayout(self)

        top_form = QFormLayout()
        self.mode_cbx = QComboBox(self)
        self.mode_cbx.addItems([MODE_AUTOMATIC, MODE_MANUAL, MODE_STEP, MODE_WAVELET])
        top_form.addRow("Method:", self.mode_cbx)
        outer.addLayout(top_form)

        self.stack = QStackedWidget(self)
        self.stack.addWidget(self._build_automatic_page())
        self.stack.addWidget(self._build_manual_page())
        self.stack.addWidget(self._build_step_page())
        self.stack.addWidget(self._build_wavelet_page())
        outer.addWidget(self.stack)

        # Live single-pixel preview while the per-pixel fit runs (Automatic /
        # Step only -- Manual has no per-pixel loop to preview).
        self.preview_chk = QCheckBox("Preview single-pixel results while fitting", self)
        outer.addWidget(self.preview_chk)
        self.mode_cbx.currentIndexChanged.connect(self._on_mode_changed)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def _on_mode_changed(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        is_manual = self.mode_cbx.itemText(index) == MODE_MANUAL
        self.preview_chk.setEnabled(not is_manual)
        if is_manual:
            self.preview_chk.setChecked(False)

    # ------------------------------------------------------------------ #
    #                          Shared row builders                       #
    # ------------------------------------------------------------------ #
    def _detector_row(self, form: QFormLayout) -> QSpinBox | None:
        """Add a "Detector:" row, but only when there is more than one detector.

        Returns ``None`` (rather than an un-parented, never-laid-out spin box)
        when there is nothing to choose, so the dialog never shows a stray
        floating widget for the common single-detector case.
        """
        if self._n_detectors <= 1:
            return None
        spin = QSpinBox(self)
        spin.setRange(0, self._n_detectors - 1)
        spin.setValue(self._active_detector)
        form.addRow("Detector:", spin)
        return spin

    def _cauchy_rows(self, form: QFormLayout):
        n_terms = QSpinBox(self)
        n_terms.setRange(1, 12)
        n_terms.setValue(6)
        form.addRow("Cauchy terms:", n_terms)

        lambda_ref = QLineEdit(self)
        lambda_ref.setPlaceholderText("blank = mean(probe)")
        form.addRow("Reference wavelength (nm):", lambda_ref)
        return n_terms, lambda_ref

    @staticmethod
    def _deriv_combo(page: QWidget) -> QComboBox:
        cbx = QComboBox(page)
        cbx.addItems([label for label, _ in _DERIV_OPTIONS])
        return cbx

    # ------------------------------------------------------------------ #
    #                                Pages                                #
    # ------------------------------------------------------------------ #
    def _build_automatic_page(self) -> QWidget:
        page = QWidget(self)
        form = QFormLayout(page)

        self.auto_deriv_cbx = self._deriv_combo(page)
        self.auto_deriv_cbx.setCurrentIndex(3)  # 1st+2nd derivatives
        form.addRow("IRF derivative terms:", self.auto_deriv_cbx)

        self.auto_2ndpass_chk = QCheckBox(page)
        self.auto_2ndpass_chk.setChecked(True)
        form.addRow("Two-pass refinement:", self.auto_2ndpass_chk)

        self.auto_smooth_spin = QSpinBox(page)
        self.auto_smooth_spin.setRange(1, 200)
        self.auto_smooth_spin.setValue(10)
        form.addRow("Smoothing window (pixels):", self.auto_smooth_spin)

        self.auto_use_exp_chk = QCheckBox(page)
        self.auto_use_exp_chk.setChecked(False)
        form.addRow("Use exponential decay:", self.auto_use_exp_chk)

        self.auto_nterms_spin, self.auto_lambda_ref_edit = self._cauchy_rows(form)
        self.auto_detector_spin = self._detector_row(form)
        return page

    def _build_manual_page(self) -> QWidget:
        page = QWidget(self)
        form = QFormLayout(page)

        note = QLabel(
            "After clicking OK, left-click points on the embedded contour to mark "
            "(wavelength, delay) pairs for the dispersion fit. Right-click removes "
            "the last point, Enter finishes, Esc cancels.",
            page,
        )
        note.setWordWrap(True)
        form.addRow(note)

        self.manual_nterms_spin, self.manual_lambda_ref_edit = self._cauchy_rows(form)
        self.manual_detector_spin = self._detector_row(form)
        return page

    def _build_step_page(self) -> QWidget:
        page = QWidget(self)
        form = QFormLayout(page)

        self.step_deriv_cbx = self._deriv_combo(page)
        self.step_deriv_cbx.setCurrentIndex(3)  # 1st+2nd derivatives
        form.addRow("IRF derivative terms:", self.step_deriv_cbx)

        self.step_tmin_spin = QDoubleSpinBox(page)
        self.step_tmin_spin.setRange(-1000.0, 1000.0)
        self.step_tmin_spin.setDecimals(3)
        self.step_tmin_spin.setValue(-1.0)
        form.addRow("Fit window t_min (ps):", self.step_tmin_spin)

        self.step_tmax_spin = QDoubleSpinBox(page)
        self.step_tmax_spin.setRange(-1000.0, 1000.0)
        self.step_tmax_spin.setDecimals(3)
        self.step_tmax_spin.setValue(3.0)
        form.addRow("Fit window t_max (ps):", self.step_tmax_spin)

        self.step_use_exp_chk = QCheckBox(page)
        self.step_use_exp_chk.setChecked(False)
        form.addRow("Use exponential decay:", self.step_use_exp_chk)

        self.step_nterms_spin, self.step_lambda_ref_edit = self._cauchy_rows(form)
        self.step_detector_spin = self._detector_row(form)
        return page

    def _build_wavelet_page(self) -> QWidget:
        page = QWidget(self)
        form = QFormLayout(page)

        self.wavelet_tmin_spin = QDoubleSpinBox(page)
        self.wavelet_tmin_spin.setRange(-1000.0, 1000.0)
        self.wavelet_tmin_spin.setDecimals(3)
        self.wavelet_tmin_spin.setValue(-1.0)
        form.addRow("Fit window t_min (ps):", self.wavelet_tmin_spin)

        self.wavelet_tmax_spin = QDoubleSpinBox(page)
        self.wavelet_tmax_spin.setRange(-1000.0, 1000.0)
        self.wavelet_tmax_spin.setDecimals(3)
        self.wavelet_tmax_spin.setValue(3.0)
        form.addRow("Fit window t_max (ps):", self.wavelet_tmax_spin)

        self.wavelet_omega_spin = QDoubleSpinBox(page)
        self.wavelet_omega_spin.setRange(1.0, 10000.0)
        self.wavelet_omega_spin.setDecimals(1)
        self.wavelet_omega_spin.setValue(125.0)
        form.addRow("Wavelet oscillation period (fs):", self.wavelet_omega_spin)

        self.wavelet_gamma_spin = QDoubleSpinBox(page)
        self.wavelet_gamma_spin.setRange(1.0, 10000.0)
        self.wavelet_gamma_spin.setDecimals(1)
        self.wavelet_gamma_spin.setValue(225.0)
        form.addRow("Wavelet width (fs):", self.wavelet_gamma_spin)

        self.wavelet_optimise_chk = QCheckBox(page)
        self.wavelet_optimise_chk.setChecked(False)
        form.addRow("Optimise wavelet parameters:", self.wavelet_optimise_chk)

        self.wavelet_opt_maxiter_spin = QSpinBox(page)
        self.wavelet_opt_maxiter_spin.setRange(20, 500)
        self.wavelet_opt_maxiter_spin.setValue(100)
        self.wavelet_opt_maxiter_spin.setEnabled(False)
        form.addRow("Max optimiser iterations:", self.wavelet_opt_maxiter_spin)

        # Enable/disable max-iter spinbox based on the optimise checkbox
        self.wavelet_optimise_chk.toggled.connect(self.wavelet_opt_maxiter_spin.setEnabled)

        self.wavelet_2ndpass_chk = QCheckBox(page)
        self.wavelet_2ndpass_chk.setChecked(True)
        form.addRow("Two-pass refinement:", self.wavelet_2ndpass_chk)

        self.wavelet_smooth_spin = QSpinBox(page)
        self.wavelet_smooth_spin.setRange(1, 200)
        self.wavelet_smooth_spin.setValue(10)
        form.addRow("Smoothing window (pixels):", self.wavelet_smooth_spin)

        self.wavelet_nterms_spin, self.wavelet_lambda_ref_edit = self._cauchy_rows(form)
        self.wavelet_detector_spin = self._detector_row(form)
        return page

    # ------------------------------------------------------------------ #
    #                                Result                               #
    # ------------------------------------------------------------------ #
    def mode(self) -> str:
        """The selected method name (``"Automatic"``, ``"Manual"``, ``"Step Function (Auto)"`` or ``"Edge Detection (Wavelet)"``)."""
        return self.mode_cbx.currentText()

    def preview_enabled(self) -> bool:
        """Whether the live single-pixel preview was requested (Automatic / Step / Wavelet only)."""
        return self.preview_chk.isEnabled() and self.preview_chk.isChecked()

    @staticmethod
    def _lambda_ref(edit: QLineEdit) -> float | None:
        text = edit.text().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    @staticmethod
    def _detector_value(spin: QSpinBox | None) -> int:
        return spin.value() if spin is not None else 0

    def fit_kwargs(self) -> dict:
        """Keyword arguments for the selected mode's ``chirp.fit_chirp_*`` function."""
        mode = self.mode()
        if mode == MODE_AUTOMATIC:
            deriv_mask = _DERIV_OPTIONS[self.auto_deriv_cbx.currentIndex()][1]
            return {
                "detector": self._detector_value(self.auto_detector_spin),
                "deriv_mask": deriv_mask,
                "use_exp": self.auto_use_exp_chk.isChecked(),
                "n_cauchy_terms": self.auto_nterms_spin.value(),
                "cauchy_lambda_ref": self._lambda_ref(self.auto_lambda_ref_edit),
                "do_2nd_pass": self.auto_2ndpass_chk.isChecked(),
                "smooth_window": self.auto_smooth_spin.value(),
            }
        if mode == MODE_STEP:
            deriv_mask = _DERIV_OPTIONS[self.step_deriv_cbx.currentIndex()][1]
            return {
                "detector": self._detector_value(self.step_detector_spin),
                "deriv_mask": deriv_mask,
                "use_exp": self.step_use_exp_chk.isChecked(),
                "t_min": self.step_tmin_spin.value(),
                "t_max": self.step_tmax_spin.value(),
                "n_cauchy_terms": self.step_nterms_spin.value(),
                "cauchy_lambda_ref": self._lambda_ref(self.step_lambda_ref_edit),
            }
        if mode == MODE_WAVELET:
            return {
                "detector": self._detector_value(self.wavelet_detector_spin),
                "t_min": self.wavelet_tmin_spin.value(),
                "t_max": self.wavelet_tmax_spin.value(),
                "omega_w": self.wavelet_omega_spin.value() / 1000.0,
                "gamma_w": self.wavelet_gamma_spin.value() / 1000.0,
                "do_2nd_pass": self.wavelet_2ndpass_chk.isChecked(),
                "smooth_window": self.wavelet_smooth_spin.value(),
                "optimise_wavelet": self.wavelet_optimise_chk.isChecked(),
                "opt_max_iter": self.wavelet_opt_maxiter_spin.value(),
                "n_cauchy_terms": self.wavelet_nterms_spin.value(),
                "cauchy_lambda_ref": self._lambda_ref(self.wavelet_lambda_ref_edit),
            }
        # Manual: only the dispersion-fit options apply (no per-pixel IRF fit).
        return {
            "detector": self._detector_value(self.manual_detector_spin),
            "n_cauchy_terms": self.manual_nterms_spin.value(),
            "cauchy_lambda_ref": self._lambda_ref(self.manual_lambda_ref_edit),
        }

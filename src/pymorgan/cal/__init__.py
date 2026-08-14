"""Spectrometer & Wavelength Calibration routines for PyMORGAN.

This package handles:
  - Optical dispersion mathematical models (Grating, Polynomial Grating, Prism Sellmeier)
  - Non-linear parameter optimisation and axis inversion (scipy.optimize.least_squares)
  - Gaussian probe spectral distribution diagnostics (w0, FWHM)
  - Reference spectra library and experimental spectrum file parsers
  - Multi-detector split & merge calibration tools
"""

from __future__ import annotations

from .fit import CalibrationResult, fit_wavelength_axis, get_default_calibration_params
from .load import (
    ExperimentalData,
    ReferenceSpectrum,
    load_experimental_spectrum,
    load_reference_spectrum,
)
from .models import (
    fit_model_eval,
    linear_grating_pixel_mapping,
    n_sellmeier_sf10,
    polynomial_grating_pixel_mapping,
    prism_output_angle,
    prism_pixel_mapping,
)
from .spectrum_fit import (
    ProbeFitResult,
    SpectrumFitResult,
    fit_probe_spectrum,
    fit_spectrum_gaussian,
)
from .utils import merge_calibration, save_calibration_file, split_calibration

__all__ = [
    "CalibrationResult",
    "SpectrumFitResult",
    "ProbeFitResult",
    "ExperimentalData",
    "ReferenceSpectrum",
    "fit_wavelength_axis",
    "fit_spectrum_gaussian",
    "fit_probe_spectrum",
    "fit_model_eval",
    "linear_grating_pixel_mapping",
    "polynomial_grating_pixel_mapping",
    "prism_pixel_mapping",
    "prism_output_angle",
    "n_sellmeier_sf10",
    "load_reference_spectrum",
    "load_experimental_spectrum",
    "split_calibration",
    "merge_calibration",
    "save_calibration_file",
]

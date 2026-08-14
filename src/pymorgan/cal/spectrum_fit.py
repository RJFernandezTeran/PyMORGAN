"""Spectral distribution Gaussian fitting for pump and probe beam profiles.

Extracts peak amplitude, central frequency w0, and Full-Width at Half-Maximum (FWHM) bandwidth.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.optimize import curve_fit


class SpectrumFitResult(NamedTuple):
    """Result container for pump/probe spectrum Gaussian fit."""

    amplitude: float
    w0: float
    fwhm: float
    fit_curve: np.ndarray
    axis_fit: np.ndarray


def gaussian_1d(x: np.ndarray, amp: float, w0: float, fwhm: float) -> np.ndarray:
    """1D Gaussian profile with full-width at half-maximum (FWHM) parametrization.

    Parameters
    ----------
    x : np.ndarray
        Spectral axis (wavenumber or wavelength).
    amp : float
        Peak amplitude.
    w0 : float
        Centre frequency / wavelength.
    fwhm : float
        Full-Width at Half-Maximum.

    Returns
    -------
    np.ndarray
        Evaluated Gaussian values.
    """
    return amp * np.exp(-4.0 * np.log(2.0) * ((x - w0) / fwhm) ** 2)


def fit_spectrum_gaussian(
    intensity: np.ndarray,
    spectral_axis: np.ndarray,
    w0_guess: float | None = None,
    fwhm_guess: float = 250.0,
) -> SpectrumFitResult:
    """Fit a 1D Gaussian to a pump or probe spectrum intensity array.

    Parameters
    ----------
    intensity : np.ndarray
        Raw or reference intensity array I(x).
    spectral_axis : np.ndarray
        Spectral axis values x (wavenumbers cm-1 or wavelength nm).
    w0_guess : float, optional
        Starting estimate for central frequency/wavelength. If None, uses peak location.
    fwhm_guess : float, default 250.0
        Starting estimate for FWHM bandwidth.

    Returns
    -------
    SpectrumFitResult
        Fit amplitude, w0, FWHM, high-density fit curve, and fit axis.
    """
    y = np.asarray(intensity, dtype=float)
    x = np.asarray(spectral_axis, dtype=float)

    # Normalise intensity
    max_val = np.max(y) if np.max(y) > 0 else 1.0
    y_norm = y / max_val

    if w0_guess is None:
        peak_idx = np.argmax(y_norm)
        w0_guess = float(x[peak_idx])

    p0 = [1.0, w0_guess, fwhm_guess]
    lb = [0.0, np.min(x), 1.0]
    ub = [5.0, np.max(x), (np.max(x) - np.min(x)) * 2.0]

    try:
        popt, _ = curve_fit(
            gaussian_1d,
            x,
            y_norm,
            p0=p0,
            bounds=(lb, ub),
            ftol=1e-4,
            xtol=1e-4,
            maxfev=1000,
        )
        amp_fit, w0_fit, fwhm_fit = popt
    except Exception:
        amp_fit, w0_fit, fwhm_fit = 1.0, w0_guess, fwhm_guess

    x_fit = np.linspace(np.min(x) - 100, np.max(x) + 100, 1000)
    fit_curve = gaussian_1d(x_fit, amp_fit, w0_fit, fwhm_fit)

    return SpectrumFitResult(
        amplitude=float(amp_fit * max_val),
        w0=float(w0_fit),
        fwhm=float(fwhm_fit),
        fit_curve=fit_curve * max_val,
        axis_fit=x_fit,
    )


# Alias for backward compatibility
fit_probe_spectrum = fit_spectrum_gaussian
ProbeFitResult = SpectrumFitResult

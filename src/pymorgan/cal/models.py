"""Mathematical optical dispersion models and fitting functions for spectrometer calibration.

Supports:
  1. Linear Grating Dispersion Model (UV-Vis and Mid-IR)
  2. Non-Linear Polynomial Grating Model (UniGE TRUVIS-II)
  3. Non-Linear Prism Dispersion Model via Sellmeier Equation for SF10 Glass (UniGE NIR-TA)
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

# Sellmeier coefficients for SF10 glass (lambda in micrometers)
SF10_B = np.array([1.62153902, 0.256287842, 1.64447552])
SF10_C = np.array([0.0122241457, 0.0595736775, 147.468793])
SF10_APEX_ANGLE_RAD = np.deg2rad(60.84)
SF10_REF_WL_NM = 1100.0


def n_sellmeier_sf10(wavelength_nm: np.ndarray | float) -> np.ndarray | float:
    """Calculate refractive index n(lambda) for SF10 glass using Sellmeier equation.

    Parameters
    ----------
    wavelength_nm : array-like or float
        Wavelength(s) in nanometers.

    Returns
    -------
    array-like or float
        Refractive index n(lambda).
    """
    wl_um = wavelength_nm / 1000.0
    wl_sq = wl_um**2
    # n(lambda) = sqrt(1 + sum(B_i * wl_um^2 / (wl_um^2 - C_i)))
    terms = (
        SF10_B[0] * wl_sq / (wl_sq - SF10_C[0])
        + SF10_B[1] * wl_sq / (wl_sq - SF10_C[1])
        + SF10_B[2] * wl_sq / (wl_sq - SF10_C[2])
    )
    return np.sqrt(1.0 + terms)


def prism_output_angle(
    wavelength_nm: np.ndarray | float, theta_in_deg: float
) -> np.ndarray | float:
    """Calculate exit angle theta_out(lambda) from SF10 prism via Snell's law.

    Parameters
    ----------
    wavelength_nm : array-like or float
        Wavelength in nanometers.
    theta_in_deg : float
        Input incidence angle in degrees.

    Returns
    -------
    array-like or float
        Output deflection angle theta_out in radians.
    """
    n_sf10 = n_sellmeier_sf10(wavelength_nm)
    th_in_rad = np.deg2rad(theta_in_deg)
    # theta_out = asin( n * sin( A - asin( sin(theta_in) / n ) ) )
    sin_in_refract = np.clip(np.sin(th_in_rad) / n_sf10, -1.0, 1.0)
    angle_inside = SF10_APEX_ANGLE_RAD - np.arcsin(sin_in_refract)
    sin_out = np.clip(n_sf10 * np.sin(angle_inside), -1.0, 1.0)
    return np.arcsin(sin_out)


def prism_pixel_mapping(
    wavelength_nm: np.ndarray,
    focal_mm: float,
    x_shift: float,
    theta_in_deg: float,
) -> np.ndarray:
    """Map reference wavelengths (nm) to pixel indices for SF10 prism spectrometer.

    Parameters
    ----------
    wavelength_nm : np.ndarray
        Wavelength values in nm.
    focal_mm : float
        Effective focal distance in mm (p[0]).
    x_shift : float
        Pixel offset / shift (p[1]).
    theta_in_deg : float
        Input incidence angle in degrees (p[2]).

    Returns
    -------
    np.ndarray
        Corresponding pixel position array.
    """
    th_wl = prism_output_angle(wavelength_nm, theta_in_deg)
    th_ref = prism_output_angle(SF10_REF_WL_NM, theta_in_deg)
    pix = focal_mm * (1000.0 / 25.0) * np.sin(th_wl - th_ref) + x_shift
    return pix


def linear_grating_pixel_mapping(
    wavelength_nm: np.ndarray, central_wl: float, ppnm: float, n_pix: int = 128
) -> np.ndarray:
    """Map reference wavelengths (nm) to pixel indices for a linear diffraction grating.

    Parameters
    ----------
    wavelength_nm : np.ndarray
        Wavelength values in nm.
    central_wl : float
        Central reference wavelength at middle pixel (nm).
    ppnm : float
        Dispersion in pixels per nm.
    n_pix : int, default 128
        Total number of pixels on detector.

    Returns
    -------
    np.ndarray
        Pixel index values.
    """
    dwl = wavelength_nm - central_wl
    c_pix = (1.0 + float(n_pix)) / 2.0
    return dwl * ppnm + c_pix


def polynomial_grating_pixel_mapping(
    wavelength_nm: np.ndarray,
    central_wl: float,
    ppnm: float,
    c2: float,
    c3: float,
    n_pix: int = 128,
) -> np.ndarray:
    """Map reference wavelengths (nm) to pixel indices for higher-order grating (TRUVIS-II).

    Parameters
    ----------
    wavelength_nm : np.ndarray
        Wavelength values in nm.
    central_wl : float
        Central / starting reference wavelength (nm).
    ppnm : float
        Linear dispersion in pixels per nm.
    c2 : float
        Quadratic dispersion coefficient (x 1e-4).
    c3 : float
        Cubic dispersion coefficient (x 1e-7).
    n_pix : int, default 128
        Total number of pixels on detector.

    Returns
    -------
    np.ndarray
        Pixel index values.
    """
    dwl = wavelength_nm - central_wl
    c_pix = (1.0 + float(n_pix)) / 2.0
    pix = dwl * ppnm + 1e-4 * c2 * (dwl**2) + 1e-7 * c3 * (dwl**3) + c_pix
    return pix


def fit_model_eval(
    params: np.ndarray,
    ref_x: np.ndarray,
    meas_pixels: np.ndarray,
    meas_y: np.ndarray,
    cal_type_code: int,
    grating_degree: int = 1,
) -> np.ndarray:
    """Evaluate synthetic measured absorbance from reference spectrum given fit parameters.

    Parameters
    ----------
    params : np.ndarray
        Optimisation parameters vector p.
    ref_x : np.ndarray
        Cut reference spectrum wavelength grid (nm).
    meas_pixels : np.ndarray
        Experimental pixel numbers (1..Npix).
    meas_y : np.ndarray
        Experimental absorbance/intensity array.
    cal_type_code : int
        Calibration type indicator (1..10).
    grating_degree : int, default 1
        Degree of grating dispersion polynomial (1=Linear, 2=Quadratic, 3=Cubic).

    Returns
    -------
    np.ndarray
        Evaluated model spectrum matched to ref_x wavelength points.
    """
    # Create interpolant for measured experimental data vs pixel
    meas_interp = interp1d(
        meas_pixels,
        meas_y,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )

    if cal_type_code == 5:
        # NIR-TA Prism model
        # params: [focal_mm, x_shift, theta_in_deg, y_scale, y_shift]
        pix_calc = prism_pixel_mapping(ref_x, params[0], params[1], params[2])
        model_y = (meas_interp(pix_calc) + params[4]) * params[3]

    else:
        # Grating model (Degree 1, 2, or 3)
        # params[0] is central wavelength lambda_0 (at central pixel c_pix)
        dwl = ref_x - params[0]
        ppnm = params[1]
        c2 = params[6] if len(params) > 6 else 0.0
        c3 = params[7] if len(params) > 7 else 0.0

        n_pix = len(meas_pixels)
        c_pix = (1.0 + float(n_pix)) / 2.0

        if grating_degree == 2 or len(params) == 7:
            pix_calc = dwl * ppnm + 1e-4 * c2 * (dwl**2) + c_pix
        elif grating_degree == 3 or len(params) >= 8:
            pix_calc = dwl * ppnm + 1e-4 * c2 * (dwl**2) + 1e-7 * c3 * (dwl**3) + c_pix
        else:
            pix_calc = dwl * ppnm + c_pix

        baseline = 1e-6 * params[4] * dwl + 1e-6 * params[5] * (dwl**2)
        model_y = meas_interp(pix_calc) * params[2] + params[3] + baseline

    return model_y


def convolve_spectrum_gaussian(y: np.ndarray, sigma: float) -> np.ndarray:
    """Convolve 1D spectrum intensity/absorbance array with a Gaussian kernel of standard deviation sigma.

    Parameters
    ----------
    y : np.ndarray
        Spectral intensity/absorbance values.
    sigma : float
        Standard deviation of Gaussian smoothing kernel (in samples/pixels).

    Returns
    -------
    np.ndarray
        Convolved spectrum array.
    """
    if sigma <= 0.001:
        return np.asarray(y, dtype=float).copy()
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(y, sigma=sigma, mode="nearest")

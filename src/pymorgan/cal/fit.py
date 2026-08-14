"""Spectral axis fitting engine using scipy.optimize.least_squares."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.spatial import KDTree

from ..log import get_logger
from .models import (
    SF10_REF_WL_NM,
    convolve_spectrum_gaussian,
    fit_model_eval,
    prism_output_angle,
)

logger = get_logger(__name__)


class CalibrationResult(NamedTuple):
    """Result container for spectrometer wavelength/wavenumber axis calibration."""

    wavelength_nm: np.ndarray
    wavenumber_cm1: np.ndarray
    fit_params: np.ndarray
    residuals: np.ndarray
    ref_x_cut: np.ndarray
    ref_y_cut: np.ndarray
    model_y_fit: np.ndarray
    convolved_ref_y: np.ndarray | None = None
    sigma_opt: float = 0.0
    exp_corr_y_fit: np.ndarray | None = None
    ref_x_full: np.ndarray | None = None
    ref_y_full: np.ndarray | None = None
    baseline_fit: np.ndarray | None = None  # Baseline evaluated over active ref_x_cut range
    ref_y_corr: np.ndarray | None = None   # Baseline-corrected reference spectrum (active range)


def prealign_peaks_cross_correlation(
    meas_y: np.ndarray,
    ref_x_full: np.ndarray,
    ref_y_full: np.ndarray,
    cwl: float,
    ppnm_guess: float,
) -> tuple[float, float]:
    """Pre-align experimental and reference spectra via template scanning.

    Scans central_wl0 (wavelength at central pixel c_pix) over a wide range and finds the offset
    that maximises the normalised dot-product between the measured absorbance
    and the reference resampled onto the estimated pixel grid.
    """
    n_pix = len(meas_y)
    if n_pix < 10 or len(ref_y_full) < 10 or ppnm_guess <= 0:
        return cwl, ppnm_guess

    from scipy.signal import savgol_filter

    c_pix = (1.0 + float(n_pix)) / 2.0
    pix = np.arange(1, n_pix + 1, dtype=float)

    # Lightly smooth the experimental spectrum to suppress noise before scoring
    win = max(5, min(21, (n_pix // 20) | 1))  # odd window, ~5% of array length
    abs_smooth = savgol_filter(meas_y, win, polyorder=2)
    abs_norm = abs_smooth - np.mean(abs_smooth)
    std_abs = np.std(abs_norm)
    if std_abs > 0:
        abs_norm = abs_norm / std_abs

    # Scan central_wl0 over ±half-detector-width around cwl, at 1-pixel resolution
    half_span_nm = n_pix / (2.0 * ppnm_guess)          # half detector span in nm
    cwl_lo = cwl - half_span_nm
    cwl_hi = cwl + half_span_nm
    n_steps = max(200, int(2 * n_pix))                  # at least 1 step per pixel
    cwl_scan = np.linspace(cwl_lo, cwl_hi, n_steps)

    scores = np.empty(n_steps)
    for i, wl0 in enumerate(cwl_scan):
        wl_grid = (pix - c_pix) / ppnm_guess + wl0
        ref_on_pix = np.interp(wl_grid, ref_x_full, ref_y_full, left=0.0, right=0.0)
        ref_on_pix -= np.mean(ref_on_pix)
        std_r = np.std(ref_on_pix)
        if std_r > 0:
            ref_on_pix /= std_r
        scores[i] = float(np.dot(abs_norm, ref_on_pix))

    best_cwl0 = float(cwl_scan[np.argmax(scores)])
    return best_cwl0, float(ppnm_guess)


def get_default_calibration_params(cal_type_code: int) -> dict:
    """Return default hardware parameters and relative nm bounds for setup (1..10)."""
    defaults = {
        1: {"cwl": 2000.0, "ppnm_guess": 0.30, "min_wl": 1500.0, "max_wl": 2500.0, "rel_min_nm": -1000.0, "rel_max_nm": 1666.0, "grating_degree": 2, "use_derivative": "1st", "use_cross_corr": False},  # UoS TRIR
        2: {"cwl": 535.0,  "ppnm_guess": 1.1,  "min_wl": 350.0,  "max_wl": 740.0,  "rel_min_nm": -180.0,  "rel_max_nm": 210.0,  "grating_degree": 2, "use_derivative": "1st", "use_cross_corr": False},  # UniGE fsTA
        3: {"cwl": 540.0,  "ppnm_guess": 1.4,  "min_wl": 350.0,  "max_wl": 740.0,  "rel_min_nm": -180.0,  "rel_max_nm": 210.0,  "grating_degree": 2, "use_derivative": "1st", "use_cross_corr": False},  # UniGE nsTA
        4: {"cwl": 2000.0, "ppnm_guess": 0.30, "min_wl": 1600.0, "max_wl": 2200.0, "rel_min_nm": -455.0,  "rel_max_nm": 1250.0, "grating_degree": 2, "use_derivative": "1st", "use_cross_corr": False},  # UZH Lab 2
        5: {"cwl": 1000.0, "ppnm_guess": 1.0,  "min_wl": 800.0,  "max_wl": 1600.0, "rel_min_nm": -200.0,  "rel_max_nm": 600.0,  "grating_degree": 1, "use_derivative": "1st", "use_cross_corr": False},  # UniGE NIR-TA (Prism model)
        6: {"cwl": 2000.0, "ppnm_guess": 0.30, "min_wl": -300.0, "max_wl": 300.0,  "rel_min_nm": -653.0,  "rel_max_nm": 882.0,  "grating_degree": 2, "use_derivative": "1st", "use_cross_corr": False},  # RAL Absorbance
        7: {"cwl": 2000.0, "ppnm_guess": 0.30, "min_wl": -300.0, "max_wl": 300.0,  "rel_min_nm": -653.0,  "rel_max_nm": 882.0,  "grating_degree": 2, "use_derivative": "1st", "use_cross_corr": False},  # RAL Intensity
        8: {"cwl": 2000.0, "ppnm_guess": 0.30, "min_wl": 1500.0, "max_wl": 2500.0, "rel_min_nm": -200.0,  "rel_max_nm": 200.0,   "grating_degree": 2, "use_derivative": "1st", "use_cross_corr": False},  # UniGE TRIR Intensity
        9: {"cwl": 2000.0, "ppnm_guess": 0.30, "min_wl": 1500.0, "max_wl": 2500.0, "rel_min_nm": -200.0,  "rel_max_nm": 200.0,   "grating_degree": 2, "use_derivative": "1st", "use_cross_corr": False},  # UniGE TRIR Absorbance
        10: {"cwl": 530.0, "ppnm_guess": 1.2,  "min_wl": 320.0,  "max_wl": 780.0,  "rel_min_nm": -150.0,  "rel_max_nm": 220.0,  "grating_degree": 2, "use_derivative": "1st", "use_cross_corr": False},  # UniGE TRUVIS-II
    }
    return defaults.get(cal_type_code, {"cwl": 535.0, "ppnm_guess": 1.1, "min_wl": 350.0, "max_wl": 740.0, "rel_min_nm": -180.0, "rel_max_nm": 210.0, "grating_degree": 2, "use_derivative": "1st", "use_cross_corr": False})


def fit_wavelength_axis(
    meas_y: np.ndarray,
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    cal_type_code: int = 2,
    cwl: float | None = None,
    ppnm_guess: float | None = None,
    min_wl: float | None = None,
    max_wl: float | None = None,
    rel_min_nm: float | None = None,
    rel_max_nm: float | None = None,
    do_fit: bool = True,
    grating_degree: int | None = None,
    use_cross_corr: bool = False,
    use_derivative: str = "1st",  # "none", "1st", "2nd"
    savgol_window: int = 9,
    convolve_ref: bool = False,
) -> CalibrationResult:
    """Optimise dispersion parameters and compute calibrated wavelength/wavenumber axes."""
    meas_pixels = np.arange(1, len(meas_y) + 1, dtype=float)
    n_pix = len(meas_y)

    spec_defaults = get_default_calibration_params(cal_type_code)
    if cwl is None:
        cwl = spec_defaults["cwl"]
    if ppnm_guess is None:
        ppnm_guess = spec_defaults["ppnm_guess"]
    if rel_min_nm is None:
        rel_min_nm = spec_defaults.get("rel_min_nm", -200.0)
    if rel_max_nm is None:
        rel_max_nm = spec_defaults.get("rel_max_nm", 200.0)
    if grating_degree is None:
        grating_degree = spec_defaults.get("grating_degree", 1)

    # Preserve full reference spectrum arrays before cutting
    ref_x_full = np.asarray(ref_x, dtype=float).ravel().copy()
    ref_y_full = np.asarray(ref_y, dtype=float).ravel().copy()

    ref_x = ref_x_full
    ref_y = ref_y_full

    is_ir = cal_type_code in (1, 4, 6, 7, 8, 9)
    if is_ir:
        cwl_nm = 1e7 / cwl if cwl > 0 else 5000.0
    else:
        cwl_nm = cwl if cwl > 0 else 500.0

    # Calculate cut bounds in nm relative to central wavelength
    min_cut_nm = cwl_nm + min(rel_min_nm, rel_max_nm)
    max_cut_nm = cwl_nm + max(rel_min_nm, rel_max_nm)

    if is_ir:
        # Convert nm bounds to wavenumber bounds (cm-1)
        min_cut_cm = 1e7 / max(max_cut_nm, 100.0)
        max_cut_cm = 1e7 / max(min_cut_nm, 50.0)

        cut_mask = (ref_x >= min_cut_cm) & (ref_x <= max_cut_cm)
        if not np.any(cut_mask):
            cut_mask = np.ones_like(ref_x, dtype=bool)

        ref_x_cut_native = ref_x[cut_mask]
        ref_y_cut = ref_y[cut_mask]

        sort_id = np.argsort(ref_x_cut_native)[::-1]
        ref_x_cut_native = ref_x_cut_native[sort_id]
        ref_y_cut = ref_y_cut[sort_id]
        meas_y = np.flip(meas_y)

        # Convert to nm for internal NLS optimisation
        ref_x_cut = 1e7 / ref_x_cut_native
    else:
        # UV-Vis / NIR setups (ref_x in nm)
        cut_mask = (ref_x >= min_cut_nm) & (ref_x <= max_cut_nm)
        if not np.any(cut_mask):
            cut_mask = np.ones_like(ref_x, dtype=bool)

        ref_x_cut_native = ref_x[cut_mask]
        ref_y_cut = ref_y[cut_mask]

        sort_cut = np.argsort(ref_x_cut_native)
        ref_x_cut_native = ref_x_cut_native[sort_cut]
        ref_y_cut = ref_y_cut[sort_cut]
        ref_x_cut = ref_x_cut_native

    # Scale both active region and full reference spectrum using active region peak height
    ref_max = np.max(ref_y_cut)
    if ref_max > 0:
        ref_y_cut = ref_y_cut / ref_max
        ref_y_full = ref_y_full / ref_max

    # Normalise experimental measured input array to [0, 1] range
    meas_span = np.max(meas_y) - np.min(meas_y)
    if meas_span > 1e-6:
        meas_y = (meas_y - np.min(meas_y)) / meas_span

    # Convert central wavelength to nm for dispersion estimation if in cm-1
    if cal_type_code in (1, 4, 6, 7, 8, 9):
        cwl_nm = 1e7 / cwl
        # Convert ppnm_guess from pixels/cm-1 to pixels/nm at central wavenumber
        ppnm_nm_guess = ppnm_guess * ((cwl**2) / 1e7)
    else:
        cwl_nm = cwl
        ppnm_nm_guess = ppnm_guess

    # Estimate starting reference central wavelength (central_wl0) at middle pixel
    if use_cross_corr:
        central_wl0, ppnm_nm_guess = prealign_peaks_cross_correlation(
            meas_y, ref_x_full, ref_y_full, cwl_nm, ppnm_nm_guess
        )
    else:
        central_wl0 = cwl_nm

    # Define initial guess (p0) and bounds (LB, UB) based on setup type & grating degree
    if cal_type_code == 5:
        # UniGE NIR-TA Prism
        p0 = np.array([248.0, 175.0, 60.0, 1.0, 0.0])
        lb = np.array([100.0, 100.0, 30.0, 0.01, -0.5])
        ub = np.array([370.0, 240.0, 90.0, 100.0, 0.5])
    elif grating_degree == 3:
        # Cubic Grating Model (e.g. TRUVIS-II) - small perturbations c2, c3 around linear model
        p0 = np.array([central_wl0, ppnm_nm_guess, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        lb = np.array([central_wl0 - 150.0, 1e-4, 0.01, -5.0, -1.0, -0.5, -2.0, -1.0])
        ub = np.array([central_wl0 + 150.0, 10.0, 100.0, 5.0, 1.0, 0.5, 2.0, 1.0])
    elif grating_degree == 2:
        # Quadratic Grating Model - small perturbation c2 around linear model
        p0 = np.array([central_wl0, ppnm_nm_guess, 1.0, 0.0, 0.0, 0.0, 0.0])
        lb = np.array([central_wl0 - 150.0, 1e-4, 0.01, -5.0, -1.0, -0.5, -2.0])
        ub = np.array([central_wl0 + 150.0, 10.0, 100.0, 5.0, 1.0, 0.5, 2.0])
    else:
        # Standard Linear Grating Model (Degree 1)
        p0 = np.array([central_wl0, ppnm_nm_guess, 1.0, 0.0, 0.0, 0.0])
        lb = np.array([central_wl0 - 150.0, 1e-4, 0.01, -5.0, -1.0, -0.5])
        ub = np.array([central_wl0 + 150.0, 10.0, 100.0, 5.0, 1.0, 0.5])

    win_len = max(5, savgol_window if savgol_window % 2 == 1 else savgol_window + 1)
    if win_len >= len(ref_x_cut):
        win_len = len(ref_x_cut) - 1 if len(ref_x_cut) % 2 == 0 else len(ref_x_cut) - 2
        win_len = max(3, win_len)

    def residuals_func(params: np.ndarray) -> np.ndarray:
        model_y = fit_model_eval(
            params, ref_x_cut, meas_pixels, meas_y, cal_type_code, grating_degree
        )
        if use_derivative == "1st" and win_len >= 5:
            d_model = savgol_filter(model_y, win_len, polyorder=2, deriv=1)
            d_ref = savgol_filter(ref_y_cut, win_len, polyorder=2, deriv=1)
            return d_model - d_ref
        elif use_derivative == "2nd" and win_len >= 5:
            d2_model = savgol_filter(model_y, win_len, polyorder=2, deriv=2)
            d2_ref = savgol_filter(ref_y_cut, win_len, polyorder=2, deriv=2)
            return d2_model - d2_ref
        else:
            return model_y - ref_y_cut

    if do_fit:
        res = least_squares(
            residuals_func,
            p0,
            bounds=(lb, ub),
            ftol=5e-7,
            xtol=5e-7,
            gtol=5e-7,
            max_nfev=5000,
        )
        p_fit = res.x
    else:
        p_fit = p0

    convolved_ref_y = None
    sigma_opt = 0.0

    if convolve_ref and do_fit:
        def conv_refine_residuals(params_with_sigma: np.ndarray) -> np.ndarray:
            p_curr = params_with_sigma[:-1]
            sig_curr = params_with_sigma[-1]
            m_y = fit_model_eval(
                p_curr, ref_x_cut, meas_pixels, meas_y, cal_type_code, grating_degree
            )
            r_conv = convolve_spectrum_gaussian(ref_y_cut, sig_curr)
            if use_derivative == "1st" and win_len >= 5:
                d_m = savgol_filter(m_y, win_len, polyorder=2, deriv=1)
                d_r = savgol_filter(r_conv, win_len, polyorder=2, deriv=1)
                return d_m - d_r
            elif use_derivative == "2nd" and win_len >= 5:
                d2_m = savgol_filter(m_y, win_len, polyorder=2, deriv=2)
                d2_r = savgol_filter(r_conv, win_len, polyorder=2, deriv=2)
                return d2_m - d2_r
            else:
                return m_y - r_conv

        p0_refine = np.append(p_fit, [2.0])
        lb_refine = np.append(lb, [0.01])
        ub_refine = np.append(ub, [30.0])

        try:
            res_conv = least_squares(
                conv_refine_residuals,
                p0_refine,
                bounds=(lb_refine, ub_refine),
                ftol=5e-7,
                xtol=5e-7,
                gtol=5e-7,
                max_nfev=3000,
            )
            p_fit = res_conv.x[:-1]
            sigma_opt = float(res_conv.x[-1])
            convolved_ref_y = convolve_spectrum_gaussian(ref_y_cut, sigma_opt)
        except Exception:
            sigma_opt = 2.0
            convolved_ref_y = convolve_spectrum_gaussian(ref_y_cut, sigma_opt)

    # Post-fit baseline parameters (b0, b1, b2) in direct absorbance space if derivative fitting was used,
    # because constant offset b0 and linear slope b1 are degenerate or unconstrained in derivative space.
    if do_fit and use_derivative != "standard" and cal_type_code != 5 and len(p_fit) >= 6:
        dwl = ref_x_cut - p_fit[0]
        ppnm = p_fit[1]
        c2 = p_fit[6] if len(p_fit) > 6 else 0.0
        c3 = p_fit[7] if len(p_fit) > 7 else 0.0
        c_pix = (1.0 + float(n_pix)) / 2.0
        if grating_degree == 2 or len(p_fit) == 7:
            pix_calc = dwl * ppnm + 1e-4 * c2 * (dwl**2) + c_pix
        elif grating_degree == 3 or len(p_fit) >= 8:
            pix_calc = dwl * ppnm + 1e-4 * c2 * (dwl**2) + 1e-7 * c3 * (dwl**3) + c_pix
        else:
            pix_calc = dwl * ppnm + c_pix

        meas_interp = interp1d(
            meas_pixels, meas_y, kind="linear", bounds_error=False, fill_value="extrapolate"
        )
        scaled_meas = meas_interp(pix_calc) * p_fit[2]
        target_ref = convolved_ref_y if convolved_ref_y is not None else ref_y_cut
        target_b = target_ref - scaled_meas

        A = np.column_stack([np.ones_like(dwl), 1e-6 * dwl, 1e-6 * (dwl**2)])
        try:
            b_opt, _, _, _ = np.linalg.lstsq(A, target_b, rcond=None)
            p_fit[3] = float(b_opt[0])
            p_fit[4] = float(b_opt[1])
            p_fit[5] = float(b_opt[2])
        except Exception:
            logger.debug(
                "Quadratic refinement of the dispersion coefficients failed; "
                "keeping the initial estimate.", exc_info=True
            )

    # Evaluate final model fit
    model_y_fit = fit_model_eval(
        p_fit, ref_x_cut, meas_pixels, meas_y, cal_type_code, grating_degree
    )
    residuals = model_y_fit - (convolved_ref_y if convolved_ref_y is not None else ref_y_cut)

    # Invert mapping to calculate pixel -> wavelength (lambda) axis
    c_pix = (1.0 + float(n_pix)) / 2.0
    if cal_type_code in (1, 4, 6, 7, 8, 9):
        # IR setups
        lam_axis = np.flip((meas_pixels - c_pix) / p_fit[1] + p_fit[0])
    elif cal_type_code == 5:
        # NIR-TA Prism numerical inversion
        wl_grid = 1e7 / np.linspace(5000, 20000, 15000)
        th_wl = prism_output_angle(wl_grid, p_fit[2])
        th_ref = prism_output_angle(SF10_REF_WL_NM, p_fit[2])
        pix_grid = p_fit[0] * (1000.0 / 25.0) * np.sin(th_wl - th_ref) + p_fit[1]

        tree = KDTree(pix_grid[:, None])
        _, nearest_idx = tree.query(meas_pixels[:, None])
        lam_axis = wl_grid[nearest_idx]
    elif grating_degree >= 2:
        # Higher-order Grating inversion (Degree 2 or 3)
        x_grid = np.linspace(250.0, 1000.0, 20000)
        dwl = x_grid - p_fit[0]
        c2 = p_fit[6] if len(p_fit) > 6 else 0.0
        c3 = p_fit[7] if len(p_fit) > 7 else 0.0
        pix_grid = dwl * p_fit[1] + 1e-4 * c2 * (dwl**2) + 1e-7 * c3 * (dwl**3) + c_pix
        lam_axis = np.interp(meas_pixels, pix_grid, x_grid)
    else:
        # Standard UV-Vis setups (Linear Grating)
        lam_axis = (meas_pixels - c_pix) / p_fit[1] + p_fit[0]

    cm_axis = 1e7 / lam_axis

    # --- Baseline in reference-scale units (active fitting range only) ---
    # The model is:  model_y = meas_interp(pix) * scale + b0 + b1·Δλ + b2·Δλ²
    # baseline_fit  = the additive polynomial offset (b0 + b1·Δλ + b2·Δλ²)
    # exp_corr_y_fit = model_y - baseline  = meas_interp(pix) * scale  (signal in ref units)
    # ref_y_corr     = ref - baseline  (reference without the offset)
    dwl_ref = ref_x_cut - p_fit[0]
    b0 = p_fit[3] if len(p_fit) > 3 else 0.0
    b1 = p_fit[4] if len(p_fit) > 4 else 0.0
    b2 = p_fit[5] if len(p_fit) > 5 else 0.0
    baseline_fit = b0 + 1e-6 * b1 * dwl_ref + 1e-6 * b2 * (dwl_ref**2)

    # Baseline-corrected experimental: signal in reference scale, no offset
    exp_corr_y_fit = model_y_fit - baseline_fit

    # Baseline-corrected reference (offset subtracted; keep sign for proper display)
    active_ref = convolved_ref_y if convolved_ref_y is not None else ref_y_cut
    ref_y_corr = active_ref - baseline_fit

    return CalibrationResult(
        wavelength_nm=lam_axis,
        wavenumber_cm1=cm_axis,
        fit_params=p_fit,
        residuals=residuals,
        ref_x_cut=ref_x_cut_native,
        ref_y_cut=ref_y_cut,
        model_y_fit=model_y_fit,
        convolved_ref_y=convolved_ref_y,
        sigma_opt=sigma_opt,
        exp_corr_y_fit=exp_corr_y_fit,
        ref_x_full=ref_x_full,
        ref_y_full=ref_y_full,
        baseline_fit=baseline_fit,
        ref_y_corr=ref_y_corr,
    )

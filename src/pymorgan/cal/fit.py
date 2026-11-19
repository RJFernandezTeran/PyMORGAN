"""Spectral axis fitting engine using scipy.optimize.least_squares and VARPRO."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.interpolate import PchipInterpolator, make_interp_spline
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from ..log import get_logger
from .baseline import estimate_baseline
from .models import (
    convolve_spectrum_gaussian,
    fit_model_eval,
    prism_pixel_mapping,
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
    ref_y_corr: np.ndarray | None = None    # Baseline-corrected reference spectrum (active range)
    sample_baseline_px: np.ndarray | None = None  # Sample baseline in input units on pixels 1..N
    ref_baseline_cut: np.ndarray | None = None    # Reference baseline in normalized ref units
    baseline_method: str = "arpls"
    baseline_target: str = "both"
    exp_raw_y_fit: np.ndarray | None = None  # Raw normalized sample mapped onto ref_x_cut


def prealign_peaks_cross_correlation(
    meas_y: np.ndarray,
    ref_x_full: np.ndarray,
    ref_y_full: np.ndarray,
    cwl: float,
    ppnm_guess: float,
) -> tuple[float, float]:
    """Pre-align experimental and reference spectra via template scanning.

    Combines normalized 0th-order and 1st-derivative template correlation over
    ``±half-detector-width`` around ``cwl`` so pre-alignment remains robust
    even under residual baseline curvature.
    """
    n_pix = len(meas_y)
    if n_pix < 10 or len(ref_y_full) < 10 or ppnm_guess <= 0:
        return cwl, ppnm_guess

    c_pix = (1.0 + float(n_pix)) / 2.0
    pix = np.arange(1, n_pix + 1, dtype=float)

    # Lightly smooth the experimental spectrum to suppress noise before scoring
    win = max(5, min(21, (n_pix // 20) | 1))  # odd window, ~5% of array length
    win = min(win, (n_pix - 1) | 1)
    if win >= 5:
        abs_smooth = savgol_filter(meas_y, win, polyorder=2)
    else:
        abs_smooth = np.asarray(meas_y, dtype=float)

    abs_norm = abs_smooth - np.mean(abs_smooth)
    std_abs = float(np.std(abs_norm))
    if std_abs > 0:
        abs_norm = abs_norm / std_abs

    use_d1 = n_pix >= 15 and win >= 5
    if use_d1:
        d1_abs = savgol_filter(meas_y, win, polyorder=2, deriv=1)
        d1_abs = d1_abs - np.mean(d1_abs)
        std_d1 = float(np.std(d1_abs))
        if std_d1 > 0:
            d1_abs = d1_abs / std_d1
        else:
            use_d1 = False

    # Scan central_wl0 over ±half-detector-width around cwl, at sub-pixel resolution
    half_span_nm = n_pix / (2.0 * ppnm_guess)
    cwl_lo = cwl - half_span_nm
    cwl_hi = cwl + half_span_nm
    n_steps = max(250, int(3 * n_pix))
    cwl_scan = np.linspace(cwl_lo, cwl_hi, n_steps)

    scores = np.empty(n_steps)
    for i, wl0 in enumerate(cwl_scan):
        wl_grid = (pix - c_pix) / ppnm_guess + wl0
        ref_on_pix = np.interp(wl_grid, ref_x_full, ref_y_full, left=0.0, right=0.0)
        ref_z = ref_on_pix - np.mean(ref_on_pix)
        std_r = float(np.std(ref_z))
        sc0 = float(np.dot(abs_norm, ref_z / std_r)) if std_r > 1e-12 else 0.0
        if use_d1:
            d1_ref = savgol_filter(ref_on_pix, win, polyorder=2, deriv=1)
            d1_ref = d1_ref - np.mean(d1_ref)
            std_dr = float(np.std(d1_ref))
            sc1 = float(np.dot(d1_abs, d1_ref / std_dr)) if std_dr > 1e-12 else 0.0
            scores[i] = 0.5 * sc0 + 0.5 * sc1
        else:
            scores[i] = sc0

    best_cwl0 = float(cwl_scan[int(np.argmax(scores))])
    return best_cwl0, float(ppnm_guess)


def get_default_calibration_params(
    cal_type_code: int,
    spectral_range: str | None = None,
) -> dict:
    """Return default hardware parameters and relative nm bounds for setup (1..12)."""
    from .load import normalize_spectral_range

    norm_range = normalize_spectral_range(spectral_range)
    common_bc = {
        "use_derivative": "global",
        "baseline_method": "arpls",
        "baseline_target": "both",
        "baseline_smoothness": 0.20,
        "convolve_ref": True,
        "use_snr_weights": True,
    }
    defaults = {
        1: {"cwl": 2000.0, "ppnm_guess": 0.30, "min_wl": 1500.0, "max_wl": 2500.0, "rel_min_nm": -1000.0, "rel_max_nm": 1666.0, "grating_degree": 2, "use_cross_corr": False, **common_bc},  # UoS TRIR
        2: {"cwl": 535.0,  "ppnm_guess": 1.1,  "min_wl": 350.0,  "max_wl": 740.0,  "rel_min_nm": -180.0,  "rel_max_nm": 210.0,  "grating_degree": 2, "use_cross_corr": False, **common_bc},  # UniGE fsTA
        3: {"cwl": 540.0,  "ppnm_guess": 1.4,  "min_wl": 350.0,  "max_wl": 740.0,  "rel_min_nm": -180.0,  "rel_max_nm": 150.0,  "grating_degree": 2, "use_cross_corr": True,  **common_bc},  # UniGE nsTA
        4: {"cwl": 2000.0, "ppnm_guess": 0.30, "min_wl": 1600.0, "max_wl": 2200.0, "rel_min_nm": -455.0,  "rel_max_nm": 1250.0, "grating_degree": 2, "use_cross_corr": False, **common_bc},  # UZH Lab 2
        5: {"cwl": 1000.0, "ppnm_guess": 1.0,  "min_wl": 800.0,  "max_wl": 1600.0, "rel_min_nm": -200.0,  "rel_max_nm": 600.0,  "grating_degree": 1, "use_cross_corr": False, **common_bc},  # UniGE NIR-TA (Prism model)
        6: {"cwl": 2000.0, "ppnm_guess": 0.30, "min_wl": -300.0, "max_wl": 300.0,  "rel_min_nm": -653.0,  "rel_max_nm": 882.0,  "grating_degree": 2, "use_cross_corr": False, **common_bc},  # RAL Absorbance
        7: {"cwl": 2000.0, "ppnm_guess": 0.30, "min_wl": -300.0, "max_wl": 300.0,  "rel_min_nm": -653.0,  "rel_max_nm": 882.0,  "grating_degree": 2, "use_cross_corr": False, **common_bc},  # RAL Intensity
        8: {"cwl": 2000.0, "ppnm_guess": 0.30, "min_wl": 1500.0, "max_wl": 2500.0, "rel_min_nm": -200.0,  "rel_max_nm": 200.0,  "grating_degree": 2, "use_cross_corr": False, **common_bc},  # UniGE TRIR Intensity
        9: {"cwl": 2000.0, "ppnm_guess": 0.30, "min_wl": 1500.0, "max_wl": 2500.0, "rel_min_nm": -200.0,  "rel_max_nm": 200.0,  "grating_degree": 2, "use_cross_corr": False, **common_bc},  # UniGE TRIR Absorbance
        10: {"cwl": 530.0, "ppnm_guess": 1.2,  "min_wl": 320.0,  "max_wl": 780.0,  "rel_min_nm": -190.0,  "rel_max_nm": 220.0,  "grating_degree": 3, "use_cross_corr": False, **common_bc},  # UniGE TRUVIS-II
        11: {"cwl": 430.0, "ppnm_guess": 1.23, "min_wl": 360.0,  "max_wl": 515.0,  "rel_min_nm": -40.0,   "rel_max_nm": 115.0,  "grating_degree": 2, "use_cross_corr": False, **common_bc},  # Light Conversion HARPIA-TA (Int.) UV
        12: {"cwl": 2000.0, "ppnm_guess": 0.30, "min_wl": 1500.0, "max_wl": 2500.0, "rel_min_nm": -200.0, "rel_max_nm": 200.0,  "grating_degree": 2, "use_cross_corr": False, **common_bc},  # PyMESS (defaults to Mid-IR TRIR)
    }

    if cal_type_code == 11 and norm_range == "vis":
        # Light Conversion HARPIA-TA (Int.) Vis Range
        return {
            "cwl": 685.0,
            "ppnm_guess": 0.62,
            "min_wl": 480.0,
            "max_wl": 890.0,
            "rel_min_nm": -190.0,
            "rel_max_nm": 190.0,
            "grating_degree": 2,
            "use_cross_corr": False,
            **common_bc,
        }

    if cal_type_code == 12:
        if norm_range == "uv-vis":
            return dict(defaults[10])  # Follow UniGE TRUVIS-II pathway
        if norm_range == "nir":
            return dict(defaults[5])   # Follow UniGE NIR-TA pathway
        return dict(defaults[8])       # Follow UniGE TRIR pathway

    return dict(
        defaults.get(
            cal_type_code,
            {
                "cwl": 535.0,
                "ppnm_guess": 1.1,
                "min_wl": 350.0,
                "max_wl": 740.0,
                "rel_min_nm": -180.0,
                "rel_max_nm": 210.0,
                "grating_degree": 2,
                "use_cross_corr": False,
                **common_bc,
            },
        )
    )


def _central_monotonic_branch(
    pix_grid: np.ndarray,
    x_grid: np.ndarray,
    mid_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the longest strictly increasing segment of ``pix_grid`` containing ``mid_idx``."""
    dp = np.diff(pix_grid)
    lo = mid_idx
    while lo > 0 and dp[lo - 1] > 0.0:
        lo -= 1
    hi = mid_idx
    while hi < len(dp) and dp[hi] > 0.0:
        hi += 1
    return pix_grid[lo : hi + 1], x_grid[lo : hi + 1]


def _invert_dispersion_axis(
    meas_pixels: np.ndarray,
    p_fit: np.ndarray,
    cal_type_code: int,
    grating_degree: int,
    is_ir: bool,
) -> np.ndarray:
    """Invert pixel-to-wavelength dispersion model cleanly for both UV-Vis/NIR and Mid-IR."""
    n_pix = len(meas_pixels)
    c_pix = (1.0 + float(n_pix)) / 2.0

    if cal_type_code == 5:
        wl_grid = np.linspace(500.0, 2200.0, 30000)
        pix_grid = prism_pixel_mapping(wl_grid, float(p_fit[0]), float(p_fit[1]), float(p_fit[2]))
        order = np.argsort(pix_grid)
        lam_inv = np.interp(meas_pixels, pix_grid[order], wl_grid[order])
    elif grating_degree >= 2 and len(p_fit) >= 7:
        wl0 = float(p_fit[0])
        ppnm = max(float(p_fit[1]), 1e-4)
        c2 = float(p_fit[6])
        c3 = float(p_fit[7]) if (grating_degree >= 3 and len(p_fit) > 7) else 0.0
        half_span_nm = max(250.0, float(n_pix) / ppnm)
        x_lo = max(50.0, wl0 - 2.5 * half_span_nm)
        x_hi = wl0 + 2.5 * half_span_nm
        x_grid = np.linspace(x_lo, x_hi, 40000)
        dwl = x_grid - wl0
        pix_grid = dwl * ppnm + 1e-4 * c2 * (dwl**2) + 1e-7 * c3 * (dwl**3) + c_pix
        mid_idx = int(np.argmin(np.abs(x_grid - wl0)))
        pix_mono, x_mono = _central_monotonic_branch(pix_grid, x_grid, mid_idx)
        if len(pix_mono) >= 2:
            lam_inv = np.interp(meas_pixels, pix_mono, x_mono)
        else:
            lam_inv = (meas_pixels - c_pix) / ppnm + wl0
    else:
        lam_inv = (meas_pixels - c_pix) / p_fit[1] + p_fit[0]

    return np.flip(lam_inv) if is_ir else lam_inv


def _prepare_cut_and_baselines(
    meas_y_raw: np.ndarray,
    ref_x_full: np.ndarray,
    ref_y_full: np.ndarray,
    is_ir: bool,
    min_cut_nm: float,
    max_cut_nm: float,
    baseline_method: str,
    baseline_target: str,
    baseline_smoothness: float,
    baseline_asym_p: float,
    probe_weights: np.ndarray | None,
) -> dict:
    """Slice reference on a padded window for edge-safe baseline correction and prepare sample."""
    bm = (baseline_method or "arpls").strip().lower()
    bt = (baseline_target or "both").strip().lower()
    apply_sample_bc = bm not in ("legacy_poly", "legacy") and bt in ("sample", "both")
    apply_ref_bc = bm not in ("legacy_poly", "legacy") and bt in ("standard", "both")

    pad_nm = 0.5 * max(20.0, max_cut_nm - min_cut_nm)
    if is_ir:
        pad_lo = 1e7 / max(max_cut_nm + pad_nm, 100.0)
        pad_hi = 1e7 / max(min_cut_nm - pad_nm, 50.0)
        cut_lo = 1e7 / max(max_cut_nm, 100.0)
        cut_hi = 1e7 / max(min_cut_nm, 50.0)
    else:
        pad_lo, pad_hi = min_cut_nm - pad_nm, max_cut_nm + pad_nm
        cut_lo, cut_hi = min_cut_nm, max_cut_nm

    pad_mask = (ref_x_full >= pad_lo) & (ref_x_full <= pad_hi)
    if not np.any(pad_mask):
        pad_mask = np.ones_like(ref_x_full, dtype=bool)
    rx_pad = ref_x_full[pad_mask]
    ry_pad = ref_y_full[pad_mask]
    s_pad = np.argsort(rx_pad)
    rx_pad = rx_pad[s_pad]
    ry_pad = ry_pad[s_pad]

    cut_mask = (rx_pad >= cut_lo) & (rx_pad <= cut_hi)
    if not np.any(cut_mask):
        cut_mask = np.ones_like(rx_pad, dtype=bool)

    if apply_ref_bc:
        len_ratio = float(np.sum(cut_mask)) / max(float(len(ry_pad)), 1.0)
        ref_smooth = float(np.clip(0.5 * baseline_smoothness * len_ratio, 0.03, 0.50))
        b_ref_pad = estimate_baseline(
            ry_pad,
            method=bm,
            smoothness=ref_smooth,
            asym_p=baseline_asym_p,
        )
    else:
        b_ref_pad = np.zeros_like(ry_pad)

    if is_ir:
        # Descending cm-1 (which maps to ascending wavelength nm)
        ref_x_cut_native = rx_pad[cut_mask][::-1]
        ref_y_cut_raw = ry_pad[cut_mask][::-1]
        ref_b_cut_raw = b_ref_pad[cut_mask][::-1]
        ref_x_cut_nm = 1e7 / np.clip(ref_x_cut_native, 1.0, None)
    else:
        ref_x_cut_native = rx_pad[cut_mask]
        ref_y_cut_raw = ry_pad[cut_mask]
        ref_b_cut_raw = b_ref_pad[cut_mask]
        ref_x_cut_nm = ref_x_cut_native

    # Scale active region and full reference spectrum using active region peak height
    ref_max = float(np.max(ref_y_cut_raw)) if len(ref_y_cut_raw) > 0 else 1.0
    if ref_max > 0.0:
        ref_y_cut = ref_y_cut_raw / ref_max
        ref_y_full_scaled = ref_y_full / ref_max
        ref_b_cut = ref_b_cut_raw / ref_max
    else:
        ref_y_cut = ref_y_cut_raw.copy()
        ref_y_full_scaled = ref_y_full.copy()
        ref_b_cut = ref_b_cut_raw.copy()

    meas_native = np.asarray(meas_y_raw, dtype=float).ravel().copy()
    n_pix = len(meas_native)

    w_snr_native: np.ndarray | None = None
    if probe_weights is not None:
        pw = np.asarray(probe_weights, dtype=float).ravel()
        if len(pw) == n_pix:
            pw_clean = np.where(np.isfinite(pw), np.maximum(pw, 0.0), 0.0)
            p90 = float(np.percentile(pw_clean, 90)) if n_pix > 0 else 0.0
            if p90 > 0.0:
                w_snr_native = np.clip(pw_clean / p90, 0.04, 1.0) ** 0.5

    if apply_sample_bc:
        b_samp_native = estimate_baseline(
            meas_native,
            method=bm,
            smoothness=baseline_smoothness,
            asym_p=baseline_asym_p,
            weights=w_snr_native,
        )
        meas_clean_native = meas_native - b_samp_native
    else:
        b_samp_native = None
        meas_clean_native = meas_native - float(np.min(meas_native))

    meas_oriented = np.flip(meas_clean_native).copy() if is_ir else meas_clean_native.copy()
    w_snr_oriented = (
        (np.flip(w_snr_native).copy() if is_ir else w_snr_native.copy())
        if w_snr_native is not None
        else None
    )

    return {
        "apply_sample_bc": apply_sample_bc,
        "apply_ref_bc": apply_ref_bc,
        "ref_x_cut_native": ref_x_cut_native,
        "ref_x_cut_nm": ref_x_cut_nm,
        "ref_y_cut": ref_y_cut,
        "ref_y_full": ref_y_full_scaled,
        "ref_b_cut": ref_b_cut,
        "sample_baseline_px": b_samp_native,
        "meas_oriented": meas_oriented,
        "w_snr_oriented": w_snr_oriented,
    }


def _fit_legacy_wavelength_axis(
    meas_y_raw: np.ndarray,
    ref_x_full: np.ndarray,
    ref_y_full: np.ndarray,
    cal_type_code: int,
    is_ir: bool,
    cwl_nm: float,
    ppnm_nm_guess: float,
    min_cut_nm: float,
    max_cut_nm: float,
    do_fit: bool,
    grating_degree: int,
    use_cross_corr: bool,
    use_derivative: str,
    savgol_window: int,
    convolve_ref: bool,
) -> CalibrationResult:
    """Classic PyMORGAN calibration path with fixed IR quadratic/cubic inversion."""
    n_pix = len(meas_y_raw)
    meas_pixels = np.arange(1, n_pix + 1, dtype=float)
    meas_y = np.flip(np.asarray(meas_y_raw, dtype=float).ravel()) if is_ir else np.asarray(meas_y_raw, dtype=float).ravel().copy()

    if is_ir:
        min_cut_cm = 1e7 / max(max_cut_nm, 100.0)
        max_cut_cm = 1e7 / max(min_cut_nm, 50.0)
        cut_mask = (ref_x_full >= min_cut_cm) & (ref_x_full <= max_cut_cm)
        if not np.any(cut_mask):
            cut_mask = np.ones_like(ref_x_full, dtype=bool)
        ref_x_cut_native = ref_x_full[cut_mask]
        ref_y_cut = ref_y_full[cut_mask]
        sort_id = np.argsort(ref_x_cut_native)[::-1]
        ref_x_cut_native = ref_x_cut_native[sort_id]
        ref_y_cut = ref_y_cut[sort_id]
        ref_x_cut = 1e7 / ref_x_cut_native
    else:
        cut_mask = (ref_x_full >= min_cut_nm) & (ref_x_full <= max_cut_nm)
        if not np.any(cut_mask):
            cut_mask = np.ones_like(ref_x_full, dtype=bool)
        ref_x_cut_native = ref_x_full[cut_mask]
        ref_y_cut = ref_y_full[cut_mask]
        sort_cut = np.argsort(ref_x_cut_native)
        ref_x_cut_native = ref_x_cut_native[sort_cut]
        ref_y_cut = ref_y_cut[sort_cut]
        ref_x_cut = ref_x_cut_native

    ref_max = float(np.max(ref_y_cut)) if len(ref_y_cut) > 0 else 1.0
    if ref_max > 0:
        ref_y_cut = ref_y_cut / ref_max
        ref_y_full_scaled = ref_y_full / ref_max
    else:
        ref_y_full_scaled = ref_y_full.copy()

    meas_span = float(np.max(meas_y) - np.min(meas_y))
    if meas_span > 1e-6:
        meas_y = (meas_y - float(np.min(meas_y))) / meas_span

    if use_cross_corr:
        sort_xc = np.argsort(ref_x_cut)
        central_wl0, ppnm_nm_guess = prealign_peaks_cross_correlation(
            meas_y, ref_x_cut[sort_xc], ref_y_cut[sort_xc], cwl_nm, ppnm_nm_guess
        )
    else:
        central_wl0 = cwl_nm

    if cal_type_code == 5:
        p0 = np.array([248.0, 175.0, 60.0, 1.0, 0.0])
        lb = np.array([100.0, 100.0, 30.0, 0.01, -0.5])
        ub = np.array([370.0, 240.0, 90.0, 100.0, 0.5])
    elif grating_degree == 3:
        p0 = np.array([central_wl0, ppnm_nm_guess, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        lb = np.array([central_wl0 - 150.0, 1e-4, 0.01, -5.0, -1.0, -0.5, -2.0, -1.0])
        ub = np.array([central_wl0 + 150.0, 10.0, 100.0, 5.0, 1.0, 0.5, 2.0, 1.0])
    elif grating_degree == 2:
        p0 = np.array([central_wl0, ppnm_nm_guess, 1.0, 0.0, 0.0, 0.0, 0.0])
        lb = np.array([central_wl0 - 150.0, 1e-4, 0.01, -5.0, -1.0, -0.5, -2.0])
        ub = np.array([central_wl0 + 150.0, 10.0, 100.0, 5.0, 1.0, 0.5, 2.0])
    else:
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
        if use_derivative == "2nd" and win_len >= 5:
            d2_model = savgol_filter(model_y, win_len, polyorder=2, deriv=2)
            d2_ref = savgol_filter(ref_y_cut, win_len, polyorder=2, deriv=2)
            return d2_model - d2_ref
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
        p_fit = p0.copy()

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
            if use_derivative == "2nd" and win_len >= 5:
                d2_m = savgol_filter(m_y, win_len, polyorder=2, deriv=2)
                d2_r = savgol_filter(r_conv, win_len, polyorder=2, deriv=2)
                return d2_m - d2_r
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

    if do_fit and use_derivative not in ("none", "standard") and cal_type_code != 5 and len(p_fit) >= 6:
        dwl = ref_x_cut - p_fit[0]
        ppnm = p_fit[1]
        c2 = p_fit[6] if len(p_fit) > 6 else 0.0
        c3 = p_fit[7] if len(p_fit) > 7 else 0.0
        c_pix = (1.0 + float(n_pix)) / 2.0
        if grating_degree == 2 or len(p_fit) == 7:
            pix_calc = dwl * ppnm + 1e-4 * c2 * (dwl**2) + c_pix
        elif grating_degree >= 3 or len(p_fit) >= 8:
            pix_calc = dwl * ppnm + 1e-4 * c2 * (dwl**2) + 1e-7 * c3 * (dwl**3) + c_pix
        else:
            pix_calc = dwl * ppnm + c_pix

        meas_interp = make_interp_spline(meas_pixels, meas_y, k=1)
        scaled_meas = meas_interp(pix_calc) * p_fit[2]
        target_ref = convolved_ref_y if convolved_ref_y is not None else ref_y_cut
        target_b = target_ref - scaled_meas

        a_mat = np.column_stack([np.ones_like(dwl), 1e-6 * dwl, 1e-6 * (dwl**2)])
        try:
            b_opt, _, _, _ = np.linalg.lstsq(a_mat, target_b, rcond=None)
            p_fit[3] = float(b_opt[0])
            p_fit[4] = float(b_opt[1])
            p_fit[5] = float(b_opt[2])
        except Exception:
            logger.debug(
                "Quadratic refinement of the baseline coefficients failed; "
                "keeping the initial estimate.",
                exc_info=True,
            )

    model_y_fit = fit_model_eval(
        p_fit, ref_x_cut, meas_pixels, meas_y, cal_type_code, grating_degree
    )
    active_ref = convolved_ref_y if convolved_ref_y is not None else ref_y_cut
    residuals = model_y_fit - active_ref

    lam_axis = _invert_dispersion_axis(meas_pixels, p_fit, cal_type_code, grating_degree, is_ir)
    cm_axis = 1e7 / np.clip(lam_axis, 1e-3, None)

    dwl_ref = ref_x_cut - p_fit[0]
    b0 = p_fit[3] if len(p_fit) > 3 else 0.0
    b1 = p_fit[4] if len(p_fit) > 4 else 0.0
    b2 = p_fit[5] if len(p_fit) > 5 else 0.0
    baseline_fit = b0 + 1e-6 * b1 * dwl_ref + 1e-6 * b2 * (dwl_ref**2)
    exp_corr_y_fit = model_y_fit - baseline_fit
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
        ref_y_full=ref_y_full_scaled,
        baseline_fit=baseline_fit,
        ref_y_corr=ref_y_corr,
        sample_baseline_px=None,
        ref_baseline_cut=None,
        baseline_method="legacy_poly",
        baseline_target="none",
    )


def _fit_varpro_wavelength_axis(
    meas_y_raw: np.ndarray,
    ref_x_full: np.ndarray,
    ref_y_full: np.ndarray,
    cal_type_code: int,
    is_ir: bool,
    cwl_nm: float,
    ppnm_nm_guess: float,
    min_cut_nm: float,
    max_cut_nm: float,
    do_fit: bool,
    grating_degree: int,
    use_cross_corr: bool,
    use_derivative: str,
    savgol_window: int,
    convolve_ref: bool,
    baseline_method: str,
    baseline_target: str,
    baseline_smoothness: float,
    baseline_asym_p: float,
    probe_weights: np.ndarray | None,
) -> CalibrationResult:
    """Improved spectroscopic baseline + VARPRO global multi-order calibration solver."""
    n_pix = len(meas_y_raw)
    meas_pixels = np.arange(1, n_pix + 1, dtype=float)
    c_pix = (1.0 + float(n_pix)) / 2.0

    prep = _prepare_cut_and_baselines(
        meas_y_raw=meas_y_raw,
        ref_x_full=ref_x_full,
        ref_y_full=ref_y_full,
        is_ir=is_ir,
        min_cut_nm=min_cut_nm,
        max_cut_nm=max_cut_nm,
        baseline_method=baseline_method,
        baseline_target=baseline_target,
        baseline_smoothness=baseline_smoothness,
        baseline_asym_p=baseline_asym_p,
        probe_weights=probe_weights,
    )

    apply_sample_bc = prep["apply_sample_bc"]
    apply_ref_bc = prep["apply_ref_bc"]
    ref_x_cut_native = prep["ref_x_cut_native"]
    ref_x_cut_nm = prep["ref_x_cut_nm"]
    ref_y_cut = prep["ref_y_cut"]
    ref_y_full_scaled = prep["ref_y_full"]
    ref_b_cut = prep["ref_b_cut"]
    sample_baseline_px = prep["sample_baseline_px"]
    meas_oriented = prep["meas_oriented"]
    w_snr_oriented = prep["w_snr_oriented"]

    # Normalize sample and baseline-subtracted reference to unit peak amplitude
    meas_scale_norm = float(np.max(np.abs(meas_oriented)))
    meas_work = meas_oriented / meas_scale_norm if meas_scale_norm > 1e-9 else meas_oriented.copy()

    ref_corr_native_grid = ref_y_cut - ref_b_cut
    ref_work_scale = float(np.max(np.abs(ref_corr_native_grid)))
    ref_corr_scaled = (
        ref_corr_native_grid / ref_work_scale if ref_work_scale > 1e-9 else ref_corr_native_grid.copy()
    )

    # Resample reference onto a uniform wavelength grid for clean SavGol derivatives and PSF convolution
    s_nm = np.argsort(ref_x_cut_nm)
    rx_nm_asc = ref_x_cut_nm[s_nm]
    ry_corr_asc = ref_corr_scaled[s_nm]
    # Deduplicate strictly increasing nodes if input had repeated points
    uniq = np.concatenate(([True], np.diff(rx_nm_asc) > 1e-9))
    rx_nm_asc = rx_nm_asc[uniq]
    ry_corr_asc = ry_corr_asc[uniq]

    n_grid = int(np.clip(len(rx_nm_asc), 500, 1600))
    if len(rx_nm_asc) >= 2:
        ref_x_u = np.linspace(float(rx_nm_asc[0]), float(rx_nm_asc[-1]), n_grid)
        dx_nm = float(ref_x_u[1] - ref_x_u[0])
        ref_work_u = PchipInterpolator(rx_nm_asc, ry_corr_asc)(ref_x_u)
    else:
        ref_x_u = rx_nm_asc.copy()
        dx_nm = 1.0
        ref_work_u = ry_corr_asc.copy()

    if n_pix >= 2:
        meas_interp = PchipInterpolator(meas_pixels, meas_work, extrapolate=True)
    else:
        def meas_interp(x: np.ndarray) -> np.ndarray:
            return np.full_like(x, meas_work[0], dtype=float)

    if w_snr_oriented is not None and n_pix >= 2:
        weight_interp = PchipInterpolator(meas_pixels, w_snr_oriented, extrapolate=True)
    else:
        weight_interp = None

    if use_cross_corr and len(ref_x_u) >= 10:
        if cal_type_code == 5:
            # Scan prism x_shift across the detector when pre-alignment is enabled
            x_scan = np.linspace(max(10.0, 0.15 * n_pix), max(50.0, 0.90 * n_pix), max(150, 2 * n_pix))
            pix_base = prism_pixel_mapping(ref_x_u, 248.0, 0.0, 60.0)
            m_z = meas_work - float(np.mean(meas_work))
            std_m = float(np.std(m_z))
            best_sc = -np.inf
            best_xs = 175.0
            if std_m > 1e-12:
                m_z = m_z / std_m
                for xs in x_scan:
                    r_p = np.interp(meas_pixels, (pix_base + xs)[::-1], ref_work_u[::-1], left=0.0, right=0.0)
                    r_z = r_p - float(np.mean(r_p))
                    std_r = float(np.std(r_z))
                    if std_r > 1e-12:
                        sc = float(np.dot(m_z, r_z / std_r))
                        if sc > best_sc:
                            best_sc = sc
                            best_xs = float(xs)
            prism_x0 = best_xs
        else:
            central_wl0, ppnm_nm_guess = prealign_peaks_cross_correlation(
                meas_work, ref_x_u, ref_work_u, cwl_nm, ppnm_nm_guess
            )
            prism_x0 = 175.0
    else:
        central_wl0 = cwl_nm
        prism_x0 = 175.0

    half_det_nm = max(20.0, float(n_pix) / max(2.0 * ppnm_nm_guess, 1e-4))
    pts_per_pix = max(1.0, (1.0 / max(ppnm_nm_guess, 1e-4)) / max(dx_nm, 1e-9))
    eff_win = max(savgol_window, int(round(1.5 * pts_per_pix)) | 1)
    if eff_win % 2 == 0:
        eff_win += 1
    if len(ref_x_u) >= 7:
        eff_win = min(eff_win, (len(ref_x_u) - 1) | 1)
        eff_win = max(5, eff_win)
    else:
        eff_win = 3
    poly_sg = 3 if eff_win >= 7 else 2

    # Build nonlinear parameter vector (linear scale + Legendre tilt are projected analytically via VARPRO)
    if cal_type_code == 5:
        p0_list = [248.0, prism_x0, 60.0]
        lb_list = [100.0, min(50.0, prism_x0 - 50.0), 30.0]
        ub_list = [370.0, max(240.0, prism_x0 + 50.0), 90.0]
    else:
        p0_list = [central_wl0, ppnm_nm_guess]
        lb_list = [central_wl0 - 0.8 * half_det_nm, max(1e-4, 0.45 * ppnm_nm_guess)]
        ub_list = [central_wl0 + 0.8 * half_det_nm, min(20.0, 2.2 * ppnm_nm_guess)]
        max_c2_pix = (0.02 * n_pix) if is_ir else (0.10 * n_pix)
        max_c3_pix = (0.005 * n_pix) if is_ir else (0.03 * n_pix)
        if grating_degree >= 2:
            p0_list.append(0.0)
            lb_list.append(-max_c2_pix)
            ub_list.append(max_c2_pix)
        if grating_degree >= 3:
            p0_list.append(0.0)
            lb_list.append(-max_c3_pix)
            ub_list.append(max_c3_pix)

    if convolve_ref:
        sig0_nm = 0.8 / max(ppnm_nm_guess, 1e-4)
        sig_lo = max(1e-4, 0.02 * dx_nm)
        sig_hi = max(sig_lo * 10.0, 3.5 / max(ppnm_nm_guess, 1e-4))
        p0_list.append(float(np.clip(sig0_nm, sig_lo, sig_hi)))
        lb_list.append(sig_lo)
        ub_list.append(sig_hi)

    p0 = np.array(p0_list, dtype=float)
    lb = np.array(lb_list, dtype=float)
    ub = np.array(ub_list, dtype=float)
    p0 = np.clip(p0, lb + 1e-9, ub - 1e-9)

    def map_wl(wl_arr: np.ndarray, p: np.ndarray) -> np.ndarray:
        if cal_type_code == 5:
            return prism_pixel_mapping(wl_arr, float(p[0]), float(p[1]), float(p[2]))
        u = (wl_arr - p[0]) / half_det_nm
        pix = (wl_arr - p[0]) * p[1] + c_pix
        idx = 2
        if grating_degree >= 2:
            pix = pix + p[idx] * (u**2)
            idx += 1
        if grating_degree >= 3:
            pix = pix + p[idx] * (u**3)
        return pix

    both_bc = apply_sample_bc and apply_ref_bc
    reg_diag = np.array(
        [1e-6, 1e-4, 0.05 if both_bc else 1e-4, 0.15 if both_bc else 1e-4],
        dtype=float,
    )

    def varpro_solve(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pix_c = map_wl(ref_x_u, p)
        edge_dist = np.minimum(pix_c - 1.0, float(n_pix) - pix_c)
        w_dom = np.clip(edge_dist / 2.0, 0.0, 1.0)
        if weight_interp is not None:
            w_snr = np.clip(weight_interp(np.clip(pix_c, 1.0, float(n_pix))), 0.15, 1.0)
            w_dom = w_dom * w_snr
        # Fallback if initial guess falls mostly outside detector span
        if float(np.sum(w_dom > 0.05)) < 10:
            w_dom = np.ones_like(pix_c)

        s_raw = meas_interp(np.clip(pix_c, 1.0, float(n_pix)))
        if convolve_ref:
            sig_pts = float(p[-1]) / max(dx_nm, 1e-9)
            r_y = (
                gaussian_filter1d(ref_work_u, sigma=sig_pts, mode="nearest")
                if sig_pts > 0.05
                else ref_work_u
            )
        else:
            r_y = ref_work_u

        ref_center = cwl_nm if cal_type_code == 5 else float(p[0])
        u = (ref_x_u - ref_center) / half_det_nm
        x_mat = np.column_stack([s_raw, np.ones_like(u), u, 0.5 * (3.0 * (u**2) - 1.0)])
        x_w = x_mat * w_dom[:, None]
        y_w = r_y * w_dom
        mean_w2 = max(float(np.mean(w_dom**2)), 1e-6)
        reg_mat = np.diag(reg_diag) * (mean_w2 * len(w_dom))
        try:
            c_lin = np.linalg.solve(x_w.T @ x_w + reg_mat, x_w.T @ y_w)
        except np.linalg.LinAlgError:
            c_lin, _, _, _ = np.linalg.lstsq(x_w, y_w, rcond=None)
        if c_lin[0] < 0.05:
            c_lin[0] = 0.05
        m_y = x_mat @ c_lin
        return m_y, r_y, w_dom, c_lin, u

    def eval_residuals(p: np.ndarray, mode: str) -> np.ndarray:
        m_y, r_y, w_dom, _, _ = varpro_solve(p)
        n_pts = max(len(r_y), 1)
        rms0 = float(np.linalg.norm(w_dom * r_y) / np.sqrt(n_pts)) + 1e-12
        r0 = w_dom * (m_y - r_y) / rms0

        if mode in ("none", "standard") or eff_win < 5 or len(r_y) <= eff_win:
            return r0

        dm = savgol_filter(m_y, eff_win, polyorder=poly_sg, deriv=1, delta=dx_nm)
        dr = savgol_filter(r_y, eff_win, polyorder=poly_sg, deriv=1, delta=dx_nm)
        rms1 = float(np.linalg.norm(w_dom * dr) / np.sqrt(n_pts)) + 1e-12
        r1 = w_dom * (dm - dr) / rms1
        if mode == "1st":
            return r1

        d2m = savgol_filter(m_y, eff_win, polyorder=poly_sg, deriv=2, delta=dx_nm)
        d2r = savgol_filter(r_y, eff_win, polyorder=poly_sg, deriv=2, delta=dx_nm)
        rms2 = float(np.linalg.norm(w_dom * d2r) / np.sqrt(n_pts)) + 1e-12
        r2 = w_dom * (d2m - d2r) / rms2
        if mode == "2nd":
            return r2
        if mode == "1st+2nd":
            return np.concatenate([1.0 * r1, 0.75 * r2])
        # "global": simultaneous normalized 0th + 1st + 2nd derivative objective
        return np.concatenate([0.5 * r0, 1.0 * r1, 0.75 * r2])

    mode_norm = (use_derivative or "global").strip().lower()
    if do_fit:
        if mode_norm in ("2nd", "1st+2nd", "global") and eff_win >= 5:
            # Stage-1 coarse lock-in with 0th + 1st derivative prevents cycle-skipping on periodic bands
            def coarse_res(p: np.ndarray) -> np.ndarray:
                return np.concatenate([0.5 * eval_residuals(p, "none"), 1.0 * eval_residuals(p, "1st")])

            res_coarse = least_squares(
                coarse_res,
                p0,
                bounds=(lb, ub),
                loss="soft_l1",
                f_scale=0.20,
                ftol=1e-6,
                xtol=1e-6,
                gtol=1e-6,
                max_nfev=600,
            )
            p_start = np.clip(res_coarse.x, lb + 1e-9, ub - 1e-9)
        else:
            p_start = p0

        res_final = least_squares(
            lambda p: eval_residuals(p, mode_norm),
            p_start,
            bounds=(lb, ub),
            loss="soft_l1",
            f_scale=0.15,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            max_nfev=2500,
        )
        pf = res_final.x
    else:
        pf = p0.copy()

    _, _, _, c_lin, _ = varpro_solve(pf)

    # Convert fitted parameters back to standard CalibrationResult.fit_params layout
    scale_ref_units = float(c_lin[0] * ref_work_scale)
    d0 = float(c_lin[1] * ref_work_scale)
    d1 = float(c_lin[2] * ref_work_scale)
    d2 = float(c_lin[3] * ref_work_scale)

    # d0 + d1*u + d2*(1.5*u^2 - 0.5) = b0 + b1*1e-6*dwl + b2*1e-6*dwl^2
    b0 = d0 - 0.5 * d2
    b1 = (d1 / half_det_nm) * 1e6
    b2 = (1.5 * d2 / (half_det_nm**2)) * 1e6

    if cal_type_code == 5:
        p_fit = np.array([float(pf[0]), float(pf[1]), float(pf[2]), scale_ref_units, b0 / max(scale_ref_units, 1e-9)])
    else:
        out_params = [float(pf[0]), float(pf[1]), scale_ref_units, b0, b1, b2]
        idx = 2
        if grating_degree >= 2:
            c2_pix = float(pf[idx])
            out_params.append(1e4 * c2_pix / (half_det_nm**2))
            idx += 1
        if grating_degree >= 3:
            c3_pix = float(pf[idx])
            out_params.append(1e7 * c3_pix / (half_det_nm**3))
        p_fit = np.array(out_params, dtype=float)

    # Compute physical PSF width sigma_opt in detector pixels
    sigma_opt = 0.0
    convolved_ref_y = None
    if convolve_ref:
        sigma_nm = float(pf[-1])
        if cal_type_code == 5:
            p_plus = prism_pixel_mapping(cwl_nm + 1.0, float(pf[0]), float(pf[1]), float(pf[2]))
            p_minus = prism_pixel_mapping(cwl_nm - 1.0, float(pf[0]), float(pf[1]), float(pf[2]))
            ppnm_local = max(abs(float(p_plus - p_minus)) / 2.0, 1e-4)
        else:
            ppnm_local = max(float(pf[1]), 1e-4)
        sigma_opt = float(sigma_nm * ppnm_local)

        sig_pts_u = sigma_nm / max(dx_nm, 1e-9)
        if sig_pts_u > 0.05 and len(ref_x_u) >= 5:
            r_conv_u = gaussian_filter1d(ref_work_u, sigma=sig_pts_u, mode="nearest") * ref_work_scale
            r_conv_on_cut = np.interp(ref_x_cut_nm, ref_x_u, r_conv_u)
        else:
            r_conv_on_cut = ref_corr_native_grid.copy()
        convolved_ref_y = r_conv_on_cut + ref_b_cut

    # Evaluate fitted experimental spectrum on the native active ref_x_cut_nm grid
    pix_on_cut = map_wl(ref_x_cut_nm, pf)
    s_on_cut = meas_interp(np.clip(pix_on_cut, 1.0, float(n_pix)))
    ref_center = cwl_nm if cal_type_code == 5 else float(pf[0])
    u_cut = (ref_x_cut_nm - ref_center) / half_det_nm
    poly_trim_cut = d0 + d1 * u_cut + d2 * 0.5 * (3.0 * (u_cut**2) - 1.0)
    exp_corr_y_fit = s_on_cut * scale_ref_units + poly_trim_cut

    # model_y_fit includes ref_b_cut so it directly overlays Raw Ref when subtract_ref_baseline=False
    model_y_fit = exp_corr_y_fit + ref_b_cut
    active_ref = convolved_ref_y if convolved_ref_y is not None else ref_y_cut
    residuals = model_y_fit - active_ref
    ref_y_corr = ref_corr_native_grid

    # Raw [0, 1]-normalized sample trace mapped onto active ref_x_cut_nm for un-corrected display
    raw_oriented = (
        np.flip(np.asarray(meas_y_raw, dtype=float).ravel())
        if is_ir
        else np.asarray(meas_y_raw, dtype=float).ravel()
    )
    raw_min = float(np.min(raw_oriented))
    raw_span = float(np.max(raw_oriented) - raw_min)
    raw_norm = (raw_oriented - raw_min) / raw_span if raw_span > 1e-9 else raw_oriented - raw_min
    if n_pix >= 2:
        exp_raw_y_fit = PchipInterpolator(meas_pixels, raw_norm, extrapolate=True)(
            np.clip(pix_on_cut, 1.0, float(n_pix))
        )
    else:
        exp_raw_y_fit = np.full_like(pix_on_cut, raw_norm[0], dtype=float)

    lam_axis = _invert_dispersion_axis(meas_pixels, p_fit, cal_type_code, grating_degree, is_ir)
    cm_axis = 1e7 / np.clip(lam_axis, 1e-3, None)

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
        ref_y_full=ref_y_full_scaled,
        baseline_fit=ref_b_cut,
        ref_y_corr=ref_y_corr,
        sample_baseline_px=sample_baseline_px,
        ref_baseline_cut=ref_b_cut,
        baseline_method=(baseline_method or "arpls").strip().lower(),
        baseline_target=(baseline_target or "both").strip().lower(),
        exp_raw_y_fit=exp_raw_y_fit,
    )


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
    use_derivative: str = "global",  # "none", "1st", "2nd", "1st+2nd", "global"
    savgol_window: int = 9,
    convolve_ref: bool = False,
    spectral_range: str | None = None,
    baseline_method: str = "arpls",
    baseline_target: str = "both",
    baseline_smoothness: float = 0.20,
    baseline_asym_p: float = 0.005,
    probe_weights: np.ndarray | None = None,
) -> CalibrationResult:
    """Optimise dispersion parameters and compute calibrated wavelength/wavenumber axes."""
    from .load import normalize_spectral_range

    ref_x_full = np.asarray(ref_x, dtype=float).ravel().copy()
    ref_y_full = np.asarray(ref_y, dtype=float).ravel().copy()

    # Resolve PyMESS (cal_type_code == 12) pathway into UniGE TRIR (8), TRUVIS-II (10), or NIR-TA (5)
    norm_range = normalize_spectral_range(spectral_range)
    if cal_type_code == 12:
        if norm_range == "uv-vis":
            cal_type_code = 10
        elif norm_range == "nir":
            cal_type_code = 5
        elif norm_range == "mid-ir":
            cal_type_code = 8
        else:
            ref_mean = float(np.mean(ref_x_full)) if len(ref_x_full) > 0 else 0.0
            if (cwl is not None and cwl > 1600.0) or ref_mean > 1500.0:
                cal_type_code = 8
            elif (cwl is not None and 800.0 <= cwl <= 1600.0) or (850.0 <= ref_mean <= 1500.0):
                cal_type_code = 5
            else:
                cal_type_code = 10

    spec_defaults = get_default_calibration_params(cal_type_code, spectral_range=norm_range)
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

    is_ir = cal_type_code in (1, 4, 6, 7, 8, 9)
    if is_ir:
        cwl_nm = 1e7 / cwl if cwl > 0 else 5000.0
        ppnm_nm_guess = ppnm_guess * ((cwl**2) / 1e7)
    else:
        cwl_nm = cwl if cwl > 0 else 500.0
        ppnm_nm_guess = ppnm_guess

    min_cut_nm = cwl_nm + min(rel_min_nm, rel_max_nm)
    max_cut_nm = cwl_nm + max(rel_min_nm, rel_max_nm)

    bm_key = (baseline_method or "arpls").strip().lower()
    deriv_key = (use_derivative or "global").strip().lower()

    if bm_key in ("legacy_poly", "legacy") and deriv_key in ("none", "standard", "1st", "2nd"):
        return _fit_legacy_wavelength_axis(
            meas_y_raw=meas_y,
            ref_x_full=ref_x_full,
            ref_y_full=ref_y_full,
            cal_type_code=cal_type_code,
            is_ir=is_ir,
            cwl_nm=cwl_nm,
            ppnm_nm_guess=ppnm_nm_guess,
            min_cut_nm=min_cut_nm,
            max_cut_nm=max_cut_nm,
            do_fit=do_fit,
            grating_degree=grating_degree,
            use_cross_corr=use_cross_corr,
            use_derivative=deriv_key,
            savgol_window=savgol_window,
            convolve_ref=convolve_ref,
        )

    return _fit_varpro_wavelength_axis(
        meas_y_raw=meas_y,
        ref_x_full=ref_x_full,
        ref_y_full=ref_y_full,
        cal_type_code=cal_type_code,
        is_ir=is_ir,
        cwl_nm=cwl_nm,
        ppnm_nm_guess=ppnm_nm_guess,
        min_cut_nm=min_cut_nm,
        max_cut_nm=max_cut_nm,
        do_fit=do_fit,
        grating_degree=grating_degree,
        use_cross_corr=use_cross_corr,
        use_derivative=deriv_key,
        savgol_window=savgol_window,
        convolve_ref=convolve_ref,
        baseline_method=bm_key,
        baseline_target=baseline_target,
        baseline_smoothness=baseline_smoothness,
        baseline_asym_p=baseline_asym_p,
        probe_weights=probe_weights,
    )

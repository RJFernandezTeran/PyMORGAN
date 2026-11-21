"""Advanced multi-band scatter and dynamic background removal for 1-D data.

Real pump-probe datasets often suffer from pump scatter that dims, spectrally
walks, jitters delay-to-delay, or varies scan-to-scan alongside broadband
fluorescence / wing drift.  A static pre-zero subtraction cannot track those
changes and leaves large positive/negative artifacts at late delays or around
harmonic wavelengths.

This module isolates localized scatter bands from the broadband pre-zero
background, optionally scales the broadband term by a smooth scatter envelope,
and solves a joint local model (scatter basis + smooth local TA polynomial)
per delay (and per single scan when available) so underlying transient
absorption features and coherent artifacts are preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from ..log import get_logger
from . import process

if TYPE_CHECKING:
    from .dataset import Dataset1D

logger = get_logger(__name__)

VALID_SCATTER_METHODS = ("local_svd", "taylor", "warped")


@dataclass
class ScatterBand:
    """Configuration for a single spectral scatter band (in native probe units).

    Attributes
    ----------
    p_min : float
        Lower spectral bound of the scatter core in native probe units
        (swapped automatically with ``p_max`` if given in reverse order).
    p_max : float
        Upper spectral bound of the scatter core in native probe units.
    wing_width : float or None, optional
        Side-wing margin added on either side of ``[p_min, p_max]`` to form the
        local fitting window. When ``None`` or non-positive, defaults to
        ``max(3 * step, 0.5 * span)``.
    taper_width : float or None, optional
        Raised-cosine taper width extending beyond ``[p_min, p_max]``. When
        ``None`` or non-positive, defaults to ``min(wing_width, max(2 * step,
        0.25 * span))``.
    enabled : bool, default True
        Whether the band is actively modelled and subtracted.
    label : str, default ""
        Human-readable label shown in diagnostics and GUI tables.
    rank : int or None, optional
        Optional per-band override for the scatter basis rank / Taylor order
        (minimum ``1``). When ``None``, inherits the global ``rank`` argument.
    couple_to : int or None, optional
        Optional 0-based index of a primary band whose smoothed amplitude
        envelope drives this harmonic band with a single global scale factor.
    """

    p_min: float
    p_max: float
    wing_width: float | None = None
    taper_width: float | None = None
    enabled: bool = True
    label: str = ""
    rank: int | None = None
    couple_to: int | None = None

    def __post_init__(self) -> None:
        lo = float(self.p_min)
        hi = float(self.p_max)
        if hi < lo:
            lo, hi = hi, lo
        self.p_min = lo
        self.p_max = hi
        if self.wing_width is not None:
            self.wing_width = float(self.wing_width)
        if self.taper_width is not None:
            self.taper_width = float(self.taper_width)
        if self.rank is not None:
            self.rank = max(1, int(self.rank))
        if self.couple_to is not None:
            self.couple_to = int(self.couple_to)


@dataclass
class ScatterRemovalResult:
    """Outputs and diagnostics from :func:`compute_scatter_removal`.

    Attributes
    ----------
    Zavg_C : np.ndarray
        Scatter- and background-corrected averaged signal matching ``Zavg_R``
        shape.
    Zss_C : np.ndarray or None
        Corrected single-scan tensor ``[Ndelays x Npixels x Ndet x Nscans]``
        when ``per_scan=True`` and single scans are available, else ``None``.
    Zstdv : np.ndarray or None
        Recomputed inter-scan standard deviation across cleaned single scans
        when ``update_noise=True`` and multiple scans exist, else ``None``.
    bkg_avg : np.ndarray
        Pre-zero background spectrum extracted from the raw signal.
    bkg_ss : np.ndarray or None
        Per-scan pre-zero background spectra when single scans were processed,
        else ``None``.
    Z_static_C : np.ndarray
        Reference signal corrected solely with standard static pre-zero
        subtraction.
    Z_diff : np.ndarray
        Difference matrix ``Z_static_C - Zavg_C`` isolating the dynamic
        scatter and broadband background component removed by the algorithm.
    bands : list of ScatterBand
        Resolved scatter-band definitions used for the calculation.
    detector : int
        0-based detector index used for diagnostics and quality metrics.
    params : dict
        Full parameter dictionary recording the configuration used.
    band_diagnostics : list of dict
        Per-band basis matrices, pre-zero templates, per-delay coefficients,
        raw/smoothed amplitude envelopes, and centroid shifts for visualization.
    metrics : dict
        Quantitative before/after summary metrics (pre-zero core RMS,
        positive-delay spectral roughness vs. wing reference, inter-scan STD,
        and saturated-pixel count).
    """

    Zavg_C: np.ndarray
    Zss_C: np.ndarray | None
    Zstdv: np.ndarray | None
    bkg_avg: np.ndarray
    bkg_ss: np.ndarray | None
    Z_static_C: np.ndarray
    Z_diff: np.ndarray
    bands: list[ScatterBand]
    detector: int
    params: dict[str, Any]
    band_diagnostics: list[dict[str, Any]]
    metrics: dict[str, Any]


# --------------------------------------------------------------------------- #
#                             Internal utilities                              #
# --------------------------------------------------------------------------- #
def _is_od_scale(units: dict | None) -> bool:
    """Return True when the signal unit convention is OD rather than mOD / x1E3."""
    if not units:
        return False
    uz = str(units.get("unitsZ", "") or "").lower()
    uz_ltx = str(units.get("unitsZ_ltx", "") or "").lower()
    return ("od" in uz) and ("mod" not in uz) and ("1e3" not in uz) and ("10^{3}" not in uz_ltx)


def _is_wavelength_probe(units: dict | None) -> bool:
    """Return True when the dataset probe axis is calibrated in wavelength (nm)."""
    if not units:
        return True
    lbl = str(units.get("unitsL_lbl", "") or "").lower()
    ltx = str(units.get("unitsL_ltx", "") or "").lower()
    if "pixel" in lbl or "px" in ltx:
        return False
    if "wavelength" in lbl or "nm" in ltx:
        return True
    if "wavenumber" in lbl or "cm" in ltx or "energy" in lbl or "ev" in ltx or "thz" in ltx:
        return False
    return True


def _resolve_pre_zero_window(
    delays: np.ndarray,
    tmin: float | None,
    tmax: float | None,
) -> tuple[float, float, np.ndarray]:
    """Resolve the pre-zero delay window ``(tmin, tmax, mask)``."""
    d = np.asarray(delays, dtype=float)
    fin = np.isfinite(d)
    if not np.any(fin):
        raise ValueError("Dataset delays contain no finite values.")

    if tmin is not None and tmax is not None:
        lo, hi = min(float(tmin), float(tmax)), max(float(tmin), float(tmax))
        mask = fin & (d >= lo) & (d <= hi)
        if np.any(mask):
            return lo, hi, mask

    neg_strict = fin & (d <= -1.0)
    if np.count_nonzero(neg_strict) >= 2:
        return float(np.min(d[neg_strict])), float(np.max(d[neg_strict])), neg_strict

    neg_half = fin & (d <= -0.5)
    if np.count_nonzero(neg_half) >= 2:
        return float(np.min(d[neg_half])), float(np.max(d[neg_half])), neg_half

    neg_any = fin & (d < 0.0)
    if np.any(neg_any):
        return float(np.min(d[neg_any])), float(np.max(d[neg_any])), neg_any

    ordered = np.sort(d[fin])
    lo = float(ordered[0])
    hi = float(ordered[min(1, ordered.size - 1)])
    mask = fin & (d >= lo) & (d <= hi)
    return lo, hi, mask


def _extract_2d_probe(probe: np.ndarray, detector: int) -> np.ndarray:
    """Return 1-D probe axis for ``detector``."""
    p = np.asarray(probe, dtype=float)
    if p.ndim > 1:
        d = int(np.clip(detector, 0, p.shape[1] - 1))
        return p[:, d]
    return p


def _band_widths(band: ScatterBand, med_step: float) -> tuple[float, float]:
    """Resolve ``(wing_width, taper_width)`` for ``band``."""
    span = max(band.p_max - band.p_min, med_step)
    wing_w = (
        float(band.wing_width)
        if (band.wing_width is not None and band.wing_width > 0)
        else max(3.0 * med_step, 0.5 * span)
    )
    taper_w = (
        float(band.taper_width)
        if (band.taper_width is not None and band.taper_width > 0)
        else max(2.0 * med_step, 0.25 * span)
    )
    taper_w = min(taper_w, wing_w)
    return wing_w, taper_w


def _cosine_taper_1d(x: np.ndarray, core_min: float, core_max: float, taper_w: float) -> np.ndarray:
    """Raised-cosine taper: 1 inside ``[core_min, core_max]``, smooth ramp to 0 over ``taper_w``."""
    taper = np.zeros_like(x, dtype=float)
    inside = (x >= core_min) & (x <= core_max)
    taper[inside] = 1.0
    if taper_w <= 0.0:
        return taper

    left = (x >= core_min - taper_w) & (x < core_min)
    if np.any(left):
        taper[left] = 0.5 * (1.0 - np.cos(np.pi * (x[left] - (core_min - taper_w)) / taper_w))

    right = (x > core_max) & (x <= core_max + taper_w)
    if np.any(right):
        taper[right] = 0.5 * (1.0 - np.cos(np.pi * ((core_max + taper_w) - x[right]) / taper_w))
    return taper


def _robust_pre_zero_spectrum(Z_2d: np.ndarray, neg_mask: np.ndarray) -> np.ndarray:
    """Compute pre-zero average spectrum safely ignoring NaNs."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pre = np.nanmean(Z_2d[neg_mask, :], axis=0)
    return np.nan_to_num(pre, nan=0.0)


def _second_diff_roughness(M: np.ndarray, row_mask: np.ndarray, col_mask: np.ndarray) -> float:
    """Compute RMS spectral 2nd-difference roughness on finite cells."""
    if not np.any(row_mask) or np.count_nonzero(col_mask) < 3:
        return float("nan")
    sub = M[np.ix_(row_mask, col_mask)]
    d2 = np.diff(sub, n=2, axis=1)
    fin = np.isfinite(d2)
    if not np.any(fin):
        return float("nan")
    return float(np.sqrt(np.mean(d2[fin] ** 2)))


# --------------------------------------------------------------------------- #
#                           Auto Band Detection                               #
# --------------------------------------------------------------------------- #
def detect_scatter_bands(
    Z_2d: np.ndarray,
    delays: np.ndarray,
    probe: np.ndarray,
    *,
    tmin: float | None = None,
    tmax: float | None = None,
    units: dict | None = None,
    allow_positive: bool = False,
) -> list[ScatterBand]:
    """Automatically detect primary and harmonic pump-scatter bands from pre-zero spectra.

    Uses a two-tier strategy:

    1. Median-filter detrending + robust MAD thresholding on the pre-zero
       spectrum to detect strong localised scatter bands (negative by default,
       or signed when ``allow_positive=True``).
    2. For calibrated wavelength (nm) probe axes, checks ``2 * pump_wl``
       (from metadata or the strongest Pass-1 band) for weaker 2nd-order
       grating scatter dips (``>= 3 sigma`` added as coupled suggestions,
       auto-enabled when ``>= 5 sigma``).

    Parameters
    ----------
    Z_2d : np.ndarray
        Two-dimensional transient signal matrix of shape ``[Ndelays x Npixels]``.
    delays : np.ndarray
        Delay axis vector of length ``Ndelays``.
    probe : np.ndarray
        Spectral probe axis vector of length ``Npixels``.
    tmin, tmax : float, optional
        Pre-zero delay window bounds used to average the static background
        spectrum. Defaults to delays ``<= -1.0`` (or earliest negative delays).
    units : dict, optional
        Dataset units dictionary (used to detect OD vs. mOD scaling, wavelength
        calibration, and optional ``pymess_metadata["pump_wl_nm"]``).
    allow_positive : bool, default False
        Whether to detect positive sharp pre-zero artefacts in addition to
        negative scatter dips.

    Returns
    -------
    list of ScatterBand
        Detected primary and optional coupled harmonic scatter bands ordered by
        descending peak amplitude.
    """
    from scipy.ndimage import median_filter

    Z_arr = np.asarray(Z_2d, dtype=float)
    d = np.asarray(delays, dtype=float)
    p = np.asarray(probe, dtype=float)
    if Z_arr.ndim != 2 or p.size < 5 or d.size < 1:
        return []

    order = np.argsort(p)
    w = p[order]
    Z_sorted = Z_arr[:, order]

    _, _, neg_mask = _resolve_pre_zero_window(d, tmin, tmax)
    pre = _robust_pre_zero_spectrum(Z_sorted, neg_mask)

    valid_cols = np.any(np.isfinite(Z_sorted[neg_mask, :]), axis=0) & np.isfinite(w)
    if np.count_nonzero(valid_cols) < 5:
        return []

    w_v = w[valid_cols]
    pre_v = pre[valid_cols]
    npix = w_v.size

    k = max(21, int(round(0.08 * npix)) | 1)
    if k > npix:
        k = npix if npix % 2 == 1 else max(3, npix - 1)
    broad = median_filter(pre_v, size=k, mode="nearest")
    sharp = pre_v - broad

    med_s = float(np.median(sharp))
    sigma = float(1.4826 * np.median(np.abs(sharp - med_s)))
    if sigma <= 1e-12:
        sigma = float(np.std(sharp))
    if sigma <= 1e-12:
        return []

    floor = 5e-4 if _is_od_scale(units) else 0.5
    thresh = max(6.0 * sigma, floor)
    hits = (np.abs(sharp) > thresh) if allow_positive else (sharp < -thresh)

    # Group contiguous hits, bridging small 1-2 pixel gaps
    idx_hits = np.where(hits)[0]
    raw_runs: list[tuple[int, int]] = []
    if idx_hits.size > 0:
        start = int(idx_hits[0])
        prev = int(idx_hits[0])
        for h in idx_hits[1:]:
            hi = int(h)
            if hi - prev > 3:
                if prev - start + 1 >= 3:
                    raw_runs.append((start, prev))
                start = hi
            prev = hi
        if prev - start + 1 >= 3:
            raw_runs.append((start, prev))

    # Expand runs outward to the wing foot while preserving peak sign
    expanded: list[tuple[int, int, float, float]] = []
    for a, b in raw_runs:
        seg = sharp[a : b + 1]
        rel_pk = int(np.argmax(np.abs(seg)))
        pk_idx = a + rel_pk
        pk_val = float(sharp[pk_idx])
        sign = 1.0 if pk_val >= 0.0 else -1.0
        exp_thresh = max(2.0 * sigma, 0.01 * abs(pk_val))

        lo = a
        steps = 0
        while lo > 0 and steps < 25:
            nxt = sharp[lo - 1]
            if nxt * sign > 0 and abs(nxt) > exp_thresh:
                lo -= 1
                steps += 1
            else:
                break

        hi = b
        steps = 0
        while hi < npix - 1 and steps < 25:
            nxt = sharp[hi + 1]
            if nxt * sign > 0 and abs(nxt) > exp_thresh:
                hi += 1
                steps += 1
            else:
                break

        expanded.append((lo, hi, abs(pk_val), float(w_v[pk_idx])))

    # Merge overlapping intervals
    expanded.sort(key=lambda item: item[0])
    merged: list[tuple[int, int, float, float]] = []
    for item in expanded:
        if not merged:
            merged.append(item)
        else:
            p_lo, p_hi, p_amp, p_wl = merged[-1]
            c_lo, c_hi, c_amp, c_wl = item
            if c_lo <= p_hi + 1:
                if c_amp > p_amp:
                    merged[-1] = (p_lo, max(p_hi, c_hi), c_amp, c_wl)
                else:
                    merged[-1] = (p_lo, max(p_hi, c_hi), p_amp, p_wl)
            else:
                merged.append(item)

    # Sort strongest first so Band 0 is the dominant primary scatter band
    merged.sort(key=lambda item: item[2], reverse=True)

    bands: list[ScatterBand] = []
    for idx, (lo, hi, _amp, pk_wl) in enumerate(merged):
        lbl = f"Primary ({pk_wl:.1f})" if idx == 0 else f"Band {idx + 1} ({pk_wl:.1f})"
        bands.append(
            ScatterBand(
                p_min=float(w_v[lo]),
                p_max=float(w_v[hi]),
                enabled=True,
                label=lbl,
            )
        )

    # Pass 2: Check 2nd-order grating harmonic for calibrated wavelength (nm) axes
    if _is_wavelength_probe(units):
        pump_candidates: list[float] = []
        pm_meta = (units or {}).get("pymess_metadata") if isinstance(units, dict) else None
        if isinstance(pm_meta, dict) and "pump_wl_nm" in pm_meta:
            try:
                pwl = float(pm_meta["pump_wl_nm"])
                if np.isfinite(pwl) and pwl > 0.0:
                    pump_candidates.append(pwl)
            except (TypeError, ValueError):
                pass
        if merged:
            pk0_wl = merged[0][3]
            if not any(abs(pk0_wl - c) < 3.0 for c in pump_candidates):
                pump_candidates.append(pk0_wl)

        for pwl in pump_candidates:
            wl2 = 2.0 * pwl
            if not (w_v[0] <= wl2 <= w_v[-1]):
                continue
            if any(b.p_min - 5.0 <= wl2 <= b.p_max + 5.0 for b in bands):
                continue

            in_existing = np.zeros(npix, dtype=bool)
            for b in bands:
                in_existing |= (w_v >= b.p_min) & (w_v <= b.p_max)

            m2 = (w_v >= wl2 - 12.0) & (w_v <= wl2 + 12.0) & (~in_existing)
            if np.count_nonzero(m2) < 3:
                continue

            cand_idx = np.where(m2)[0]
            rel_min = int(np.argmin(sharp[cand_idx]))
            pk2_idx = int(cand_idx[rel_min])
            min_val = float(sharp[pk2_idx])
            snr2 = abs(min_val) / max(sigma, 1e-12)
            if min_val < 0.0 and snr2 >= 3.0:
                pk2_wl = float(w_v[pk2_idx])
                lo2 = pk2_idx
                while lo2 > 0 and (pk2_idx - lo2) < 8 and sharp[lo2 - 1] < -1.5 * sigma:
                    lo2 -= 1
                hi2 = pk2_idx
                while hi2 < npix - 1 and (hi2 - pk2_idx) < 8 and sharp[hi2 + 1] < -1.5 * sigma:
                    hi2 += 1
                p_lo2 = min(float(w_v[lo2]), pk2_wl - 4.0)
                p_hi2 = max(float(w_v[hi2]), pk2_wl + 4.0)
                p_lo2 = max(float(w_v[0]), p_lo2)
                p_hi2 = min(float(w_v[-1]), p_hi2)
                bands.append(
                    ScatterBand(
                        p_min=p_lo2,
                        p_max=p_hi2,
                        enabled=(snr2 >= 5.0),
                        label=f"2nd order ({pk2_wl:.1f})",
                        rank=1,
                        couple_to=0 if len(bands) > 0 else None,
                    )
                )

    return bands


# --------------------------------------------------------------------------- #
#                            2-D Matrix Solver                                #
# --------------------------------------------------------------------------- #
def _group_overlapping_bands(
    indexed_bands: list[tuple[int, ScatterBand]],
    med_step: float,
) -> list[list[tuple[int, ScatterBand, float, float]]]:
    """Merge bands whose fit windows ``[p_min - wing, p_max + wing]`` overlap."""
    items: list[tuple[int, ScatterBand, float, float, float, float]] = []
    for orig_idx, b in indexed_bands:
        wing_w, taper_w = _band_widths(b, med_step)
        items.append((orig_idx, b, b.p_min - wing_w, b.p_max + wing_w, wing_w, taper_w))
    items.sort(key=lambda x: x[2])

    groups: list[list[tuple[int, ScatterBand, float, float]]] = []
    cur_group: list[tuple[int, ScatterBand, float, float]] = []
    cur_end = -np.inf

    for orig_idx, b, f_lo, f_hi, wing_w, taper_w in items:
        if not cur_group or f_lo <= cur_end:
            cur_group.append((orig_idx, b, wing_w, taper_w))
            cur_end = max(cur_end, f_hi)
        else:
            groups.append(cur_group)
            cur_group = [(orig_idx, b, wing_w, taper_w)]
            cur_end = f_hi
    if cur_group:
        groups.append(cur_group)
    return groups


def _compute_centroid_shift(w_sub: np.ndarray, ref_prof: np.ndarray, removed: np.ndarray) -> np.ndarray:
    """Estimate per-delay centroid shift (in probe units) relative to ``ref_prof``."""
    ref_w = np.maximum(np.abs(ref_prof), 0.0)
    ref_sum = float(np.sum(ref_w))
    if ref_sum <= 1e-12:
        return np.zeros(removed.shape[0], dtype=float)
    ref_cen = float(np.sum(ref_w * w_sub) / ref_sum)

    sign = 1.0 if np.sum(ref_prof) >= 0.0 else -1.0
    pos = np.maximum(sign * removed, 0.0)
    sums = np.sum(pos, axis=1)
    shifts = np.zeros(removed.shape[0], dtype=float)
    valid = sums > 0.05 * ref_sum
    if np.any(valid):
        cens = (pos[valid] @ w_sub) / sums[valid]
        shifts[valid] = cens - ref_cen
    return shifts


def _smooth_envelope(values: np.ndarray, neg_mask: np.ndarray) -> np.ndarray:
    """Smooth raw scatter amplitude trajectory for broadband background scaling."""
    from scipy.signal import savgol_filter

    arr = np.asarray(values, dtype=float)
    nt = arr.size
    if nt == 0:
        return arr

    if np.any(neg_mask):
        neg_mean = float(np.nanmedian(arr[neg_mask]))
        if np.isfinite(neg_mean) and abs(neg_mean) > 1e-6:
            arr = arr / neg_mean

    arr = np.clip(np.nan_to_num(arr, nan=1.0), 0.0, 2.5)
    win = min(21, nt if nt % 2 == 1 else nt - 1)
    if win >= 5:
        poly = min(2, win - 1)
        smoothed = savgol_filter(arr, window_length=win, polyorder=poly, mode="nearest")
    elif win >= 3:
        kernel = np.ones(win, dtype=float) / win
        smoothed = np.convolve(np.pad(arr, win // 2, mode="edge"), kernel, mode="valid")
    else:
        smoothed = arr
    if np.any(neg_mask):
        neg_sm = float(np.nanmedian(smoothed[neg_mask]))
        if np.isfinite(neg_sm) and abs(neg_sm) > 1e-6:
            smoothed = smoothed / neg_sm
    return np.clip(smoothed, 0.1, 2.5)


def _solve_linear_per_delay(
    Y_sub: np.ndarray,
    basis: np.ndarray,
    P_poly: np.ndarray,
    fit_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Jointly solve ``[basis, P_poly]`` per delay on finite pixels and return ``(removed, coeffs)``."""
    nt = Y_sub.shape[0]
    n_b = basis.shape[1]
    n_p = P_poly.shape[1]
    X_full = np.column_stack([basis, P_poly])
    removed = np.zeros((nt, fit_idx.size), dtype=float)
    coeffs = np.zeros((nt, n_b), dtype=float)

    Y_fit = Y_sub[:, fit_idx]
    all_finite = np.all(np.isfinite(Y_fit))
    if all_finite:
        C_all, *_ = np.linalg.lstsq(X_full, Y_fit.T, rcond=None)
        coeffs = C_all[:n_b, :].T
        removed = coeffs @ basis.T
        return removed, coeffs

    for k in range(nt):
        yk = Y_fit[k]
        fin = np.isfinite(yk)
        if np.count_nonzero(fin) <= n_b + n_p:
            continue
        c, *_ = np.linalg.lstsq(X_full[fin], yk[fin], rcond=None)
        coeffs[k] = c[:n_b]
        removed[k, fin] = basis[fin] @ c[:n_b]
    return removed, coeffs


def _clean_2d_matrix(
    Z_2d: np.ndarray,
    delays: np.ndarray,
    probe_1d: np.ndarray,
    bands: list[ScatterBand],
    *,
    neg_mask: np.ndarray,
    method: str = "local_svd",
    rank: int = 3,
    poly_deg: int = 2,
    t0_exclude: tuple[float, float] = (-0.5, 1.5),
    scale_broadband: bool = True,
    transmittance_space: bool = False,
    mask_saturated: bool = False,
    saturation_threshold: float = 500.0,
    od_scale: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], int]:
    """Clean one 2-D ``[Ndelays x Npixels]`` matrix and return ``(Z_clean, bkg_pre, diags, n_sat)``."""
    from scipy.interpolate import CubicSpline
    from scipy.optimize import least_squares

    M_orig = np.asarray(Z_2d, dtype=float)
    d = np.asarray(delays, dtype=float)
    p = np.asarray(probe_1d, dtype=float)

    order = np.argsort(p)
    inv_order = np.argsort(order)
    w = p[order]
    M = M_orig[:, order]
    nt, npix = M.shape

    med_step = float(np.median(np.abs(np.diff(w)))) if npix > 1 else 1.0
    if not np.isfinite(med_step) or med_step <= 0.0:
        med_step = 1.0

    # Preserve original NaN map so masked detector gaps never spread NaNs
    orig_nan_mask = ~np.isfinite(M)
    valid_cols = np.any(np.isfinite(M), axis=0) & np.isfinite(w)

    if transmittance_space:
        Y = np.where(orig_nan_mask, np.nan, 10.0 ** (-M / od_scale) - 1.0)
    else:
        Y = M.copy()

    pre = _robust_pre_zero_spectrum(Y, neg_mask)

    enabled_indexed = [(idx, b) for idx, b in enumerate(bands) if b.enabled]
    if not enabled_indexed:
        Y_out = Y - pre[None, :]
        if transmittance_space:
            Z_out = -od_scale * np.log10(np.clip(1.0 + Y_out, 1e-6, None))
        else:
            Z_out = Y_out
        Z_out[orig_nan_mask] = np.nan
        return Z_out[:, inv_order], pre[inv_order], [], 0

    all_cores = np.zeros(npix, dtype=bool)
    for _, b in enabled_indexed:
        all_cores |= (w >= b.p_min) & (w <= b.p_max)

    # Separate uncoupled and coupled enabled bands
    enabled_set = {idx for idx, _ in enabled_indexed}
    uncoupled = [
        (idx, b)
        for idx, b in enabled_indexed
        if b.couple_to is None or b.couple_to not in enabled_set or b.couple_to == idx
    ]
    coupled = [
        (idx, b)
        for idx, b in enabled_indexed
        if b.couple_to is not None and b.couple_to in enabled_set and b.couple_to != idx
    ]
    if not uncoupled:
        uncoupled = [(idx, b) for idx, b in enabled_indexed]
        coupled = []

    groups = _group_overlapping_bands(uncoupled, med_step)

    # Step A: Joint pre-zero split into localized templates P0 and broadband B_broad
    B_broad = pre.copy()
    group_specs: list[dict[str, Any]] = []

    def _fit_local_template(core_lo: float, core_hi: float, fit_mask: np.ndarray, taper: np.ndarray) -> np.ndarray:
        wing_mask = fit_mask & (~all_cores) & valid_cols
        if np.count_nonzero(wing_mask) < 2:
            wider = (
                (w >= core_lo - 3.0 * med_step)
                & (w <= core_hi + 3.0 * med_step)
                & (~all_cores)
                & valid_cols
            )
            wing_mask = wider
        if np.count_nonzero(wing_mask) >= 4:
            deg = 1
            coef = np.polyfit(w[wing_mask], pre[wing_mask], deg)
            bl = np.polyval(coef, w[fit_mask])
        elif np.count_nonzero(wing_mask) >= 1:
            bl = np.full(np.count_nonzero(fit_mask), float(np.median(pre[wing_mask])), dtype=float)
        else:
            bl = np.zeros(np.count_nonzero(fit_mask), dtype=float)
        return (pre[fit_mask] - bl) * taper

    for grp in groups:
        f_lo = min(b.p_min - wing_w for _, b, wing_w, _ in grp)
        f_hi = max(b.p_max + wing_w for _, b, wing_w, _ in grp)
        c_lo = min(b.p_min for _, b, _, _ in grp)
        c_hi = max(b.p_max for _, b, _, _ in grp)

        fit_mask = (w >= f_lo) & (w <= f_hi) & valid_cols
        fit_idx = np.where(fit_mask)[0]
        if fit_idx.size < 4:
            continue

        w_fit = w[fit_idx]
        taper = np.zeros(fit_idx.size, dtype=float)
        grp_cores = np.zeros(fit_idx.size, dtype=bool)
        for _, b, _, taper_w in grp:
            taper = np.maximum(taper, _cosine_taper_1d(w_fit, b.p_min, b.p_max, taper_w))
            grp_cores |= (w_fit >= b.p_min) & (w_fit <= b.p_max)

        P0_grp = _fit_local_template(c_lo, c_hi, fit_mask, taper)
        B_broad[fit_idx] -= P0_grp

        # Effective rank for the group
        explicit_ranks = [b.rank for _, b, _, _ in grp if b.rank is not None]
        eff_rank = max(explicit_ranks) if explicit_ranks else int(rank)
        eff_rank = max(1, min(eff_rank, max(1, fit_idx.size - (poly_deg + 2))))

        group_specs.append(
            {
                "grp": grp,
                "fit_idx": fit_idx,
                "fit_mask": fit_mask,
                "w_fit": w_fit,
                "taper": taper,
                "grp_cores": grp_cores,
                "P0": P0_grp,
                "rank": eff_rank,
                "peak_abs": float(np.max(np.abs(P0_grp))) if P0_grp.size else 0.0,
            }
        )

    coupled_specs: list[dict[str, Any]] = []
    for orig_idx, b in coupled:
        wing_w, taper_w = _band_widths(b, med_step)
        fit_mask = (w >= b.p_min - wing_w) & (w <= b.p_max + wing_w) & valid_cols
        fit_idx = np.where(fit_mask)[0]
        if fit_idx.size < 3:
            continue
        w_fit = w[fit_idx]
        taper = _cosine_taper_1d(w_fit, b.p_min, b.p_max, taper_w)
        P0_c = _fit_local_template(b.p_min, b.p_max, fit_mask, taper)
        B_broad[fit_idx] -= P0_c
        coupled_specs.append(
            {
                "orig_idx": orig_idx,
                "band": b,
                "fit_idx": fit_idx,
                "w_fit": w_fit,
                "taper": taper,
                "P0": P0_c,
            }
        )

    # Step B: Primary envelope estimation and broadband subtraction
    primary_amp_raw = np.ones(nt, dtype=float)
    primary_amp_smooth = np.ones(nt, dtype=float)
    Y_static_nb = Y - B_broad[None, :]

    if group_specs:
        g0 = max(group_specs, key=lambda g: g["peak_abs"])
        w_fit0 = g0["w_fit"]
        span0 = float(np.ptp(w_fit0)) if w_fit0.size > 1 else 1.0
        span0 = span0 if span0 > 0 else 1.0
        w_norm0 = (w_fit0 - float(np.mean(w_fit0))) / (0.5 * span0)
        p_deg0 = min(int(poly_deg), max(0, w_fit0.size - 3))
        P_poly0 = np.column_stack([w_norm0 ** j for j in range(p_deg0 + 1)])
        if w_fit0.size >= 4:
            cs0 = CubicSpline(w_fit0, g0["P0"])
            dP0 = np.nan_to_num(cs0(w_fit0, 1), nan=0.0) * g0["taper"]
            basis0 = np.column_stack([g0["P0"], dP0])
        else:
            basis0 = g0["P0"][:, None]
        _, c0 = _solve_linear_per_delay(Y_static_nb, basis0, P_poly0, g0["fit_idx"])
        primary_amp_raw = c0[:, 0]
        primary_amp_smooth = _smooth_envelope(primary_amp_raw, neg_mask)

    if scale_broadband:
        Y_nb = Y - primary_amp_smooth[:, None] * B_broad[None, :]
    else:
        Y_nb = Y_static_nb

    Y_out = Y_nb.copy()
    t0_lo, t0_hi = min(float(t0_exclude[0]), float(t0_exclude[1])), max(
        float(t0_exclude[0]), float(t0_exclude[1])
    )
    learn_delays = (d <= t0_lo) | (d >= t0_hi)

    band_amps_smooth: dict[int, np.ndarray] = {}
    band_amps_raw: dict[int, np.ndarray] = {}
    band_diags: list[dict[str, Any]] = []

    # Step C1: Solve each uncoupled group
    for g in group_specs:
        fit_idx = g["fit_idx"]
        w_fit = g["w_fit"]
        taper = g["taper"]
        P0_grp = g["P0"]
        eff_rank = g["rank"]

        span = float(np.ptp(w_fit)) if w_fit.size > 1 else 1.0
        span = span if span > 0 else 1.0
        w_norm = (w_fit - float(np.mean(w_fit))) / (0.5 * span)
        p_deg = min(int(poly_deg), max(0, w_fit.size - eff_rank - 1))
        P_poly = np.column_stack([w_norm ** j for j in range(p_deg + 1)])

        if method == "warped" and w_fit.size >= 4:
            cs = CubicSpline(w_fit, P0_grp, extrapolate=False)
            w_center = float(w_fit[int(np.argmax(np.abs(P0_grp)))])
            grp_span = max(med_step, max(b.p_max - b.p_min for _, b, _, _ in g["grp"]))
            max_dl = max(3.0 * med_step, 0.25 * grp_span)
            allow_stretch = eff_rank >= 2

            removed = np.zeros((nt, fit_idx.size), dtype=float)
            amps = np.ones(nt, dtype=float)
            shifts = np.zeros(nt, dtype=float)
            prev_dl, prev_gam = 0.0, 1.0

            Y_fit = Y_nb[:, fit_idx]
            for k in range(nt):
                yk = Y_fit[k]
                fin = np.isfinite(yk)
                if np.count_nonzero(fin) <= P_poly.shape[1] + 2:
                    continue
                yk_f = yk[fin]
                wf_f = w_fit[fin]
                tap_f = taper[fin]
                Pp_f = P_poly[fin]

                def _resid(
                    params,
                    _yk=yk_f,
                    _wf=wf_f,
                    _tap=tap_f,
                    _Pp=Pp_f,
                    _allow=allow_stretch,
                    _wc=w_center,
                    _cs=cs,
                ):
                    dl = params[0]
                    gam = params[1] if _allow else 1.0
                    wq = _wc + (_wf - _wc - dl) / gam
                    p_w = np.nan_to_num(_cs(wq), nan=0.0) * _tap
                    X = np.column_stack([p_w, _Pp])
                    c, *_ = np.linalg.lstsq(X, _yk, rcond=None)
                    return _yk - X @ c

                if allow_stretch:
                    res = least_squares(
                        _resid,
                        [np.clip(prev_dl, -max_dl * 0.9, max_dl * 0.9), np.clip(prev_gam, 0.8, 1.3)],
                        bounds=([-max_dl, 0.75], [max_dl, 1.35]),
                        ftol=1e-4,
                        xtol=1e-4,
                        max_nfev=30,
                    )
                    dl, gam = float(res.x[0]), float(res.x[1])
                else:
                    res = least_squares(
                        _resid,
                        [np.clip(prev_dl, -max_dl * 0.9, max_dl * 0.9)],
                        bounds=([-max_dl], [max_dl]),
                        ftol=1e-4,
                        xtol=1e-4,
                        max_nfev=25,
                    )
                    dl, gam = float(res.x[0]), 1.0

                prev_dl, prev_gam = dl, gam
                wq = w_center + (wf_f - w_center - dl) / gam
                p_w = np.nan_to_num(cs(wq), nan=0.0) * tap_f
                X = np.column_stack([p_w, Pp_f])
                c, *_ = np.linalg.lstsq(X, yk_f, rcond=None)
                removed[k, fin] = c[0] * p_w
                amps[k] = float(c[0])
                shifts[k] = dl

            Y_out[:, fit_idx] -= removed
            basis = P0_grp[:, None]
            coeffs = amps[:, None]
            amp_raw = amps
            amp_smooth = _smooth_envelope(amp_raw, neg_mask)

        elif method == "taylor" and w_fit.size >= 4:
            cs = CubicSpline(w_fit, P0_grp)
            cols = [P0_grp]
            for deriv_order in range(1, eff_rank):
                d_col = np.nan_to_num(cs(w_fit, deriv_order), nan=0.0) * taper
                cols.append(d_col)
            basis = np.column_stack(cols)
            removed, coeffs = _solve_linear_per_delay(Y_nb, basis, P_poly, fit_idx)
            Y_out[:, fit_idx] -= removed
            amp_raw = coeffs[:, 0]
            amp_smooth = _smooth_envelope(amp_raw, neg_mask)
            shifts = _compute_centroid_shift(w_fit, P0_grp, removed)

        else:
            # Adaptive Local SVD (default)
            wing_in_fit = np.where(~g["grp_cores"])[0]
            if wing_in_fit.size >= p_deg + 1:
                Q_w = P_poly[wing_in_fit]
                pinv_w = np.linalg.pinv(Q_w)
                Y_wings = np.nan_to_num(Y_nb[:, fit_idx][:, wing_in_fit], nan=0.0)
                smooth_est = (P_poly @ (pinv_w @ Y_wings.T)).T
            else:
                smooth_est = np.zeros((nt, fit_idx.size), dtype=float)

            sharp = np.nan_to_num(Y_nb[:, fit_idx] - smooth_est, nan=0.0) * taper[None, :]
            learn = learn_delays
            if np.count_nonzero(learn) < max(eff_rank + 2, 4):
                learn = np.ones(nt, dtype=bool)
            _, _, vt = np.linalg.svd(sharp[learn], full_matrices=False)
            r_use = min(eff_rank, vt.shape[0])
            basis = vt[:r_use].T * taper[:, None]
            # Orient basis[:, 0] so its correlation with P0_grp is positive -> intuitive amplitude sign
            if r_use >= 1 and float(np.dot(basis[:, 0], P0_grp)) < 0.0:
                basis[:, 0] *= -1.0

            removed, coeffs = _solve_linear_per_delay(Y_nb, basis, P_poly, fit_idx)
            Y_out[:, fit_idx] -= removed

            # Express amplitude relative to pre-zero template P0_grp
            p0_norm2 = float(np.dot(P0_grp, P0_grp))
            if p0_norm2 > 1e-12:
                amp_raw = (removed @ P0_grp) / p0_norm2
            else:
                amp_raw = coeffs[:, 0]
            amp_smooth = _smooth_envelope(amp_raw, neg_mask)
            shifts = _compute_centroid_shift(w_fit, P0_grp, removed)

        for orig_idx, b, _, _ in g["grp"]:
            band_amps_raw[orig_idx] = amp_raw
            band_amps_smooth[orig_idx] = amp_smooth
            band_diags.append(
                {
                    "band_index": orig_idx,
                    "band": b,
                    "w_fit": w_fit,
                    "template": P0_grp,
                    "basis": basis,
                    "coeffs": coeffs,
                    "amplitude": amp_raw,
                    "amplitude_smooth": amp_smooth,
                    "centroid_shift": shifts,
                    "coupled": False,
                }
            )

    # Step C2: Solve coupled harmonic bands driven by their target band's smooth envelope
    for cs_spec in coupled_specs:
        orig_idx = cs_spec["orig_idx"]
        b = cs_spec["band"]
        fit_idx = cs_spec["fit_idx"]
        w_fit = cs_spec["w_fit"]
        P0_c = cs_spec["P0"]

        tgt_idx = b.couple_to if b.couple_to in band_amps_smooth else next(iter(band_amps_smooth), None)
        if tgt_idx is None:
            a_tgt = primary_amp_smooth
        else:
            a_tgt = band_amps_smooth[tgt_idx]

        span = float(np.ptp(w_fit)) if w_fit.size > 1 else 1.0
        span = span if span > 0 else 1.0
        w_norm = (w_fit - float(np.mean(w_fit))) / (0.5 * span)
        p_deg = min(int(poly_deg), max(0, w_fit.size - 2))
        P_poly = np.column_stack([w_norm ** j for j in range(p_deg + 1)])

        core_in_fit = (w_fit >= b.p_min) & (w_fit <= b.p_max)
        wing_in_fit = np.where(~core_in_fit)[0]
        if wing_in_fit.size >= p_deg + 1:
            Q_w = P_poly[wing_in_fit]
            pinv_w = np.linalg.pinv(Q_w)
            Y_w = np.nan_to_num(Y_nb[:, fit_idx][:, wing_in_fit], nan=0.0)
            smooth_est = (P_poly @ (pinv_w @ Y_w.T)).T
        else:
            smooth_est = np.zeros((nt, fit_idx.size), dtype=float)

        learn = learn_delays if np.count_nonzero(learn_delays) >= 3 else np.ones(nt, dtype=bool)
        R_learn = np.nan_to_num(Y_nb[learn][:, fit_idx] - smooth_est[learn], nan=0.0)
        pred_unit = a_tgt[learn, None] * P0_c[None, :]
        denom = float(np.sum(pred_unit ** 2))
        if denom > 1e-12:
            alpha = float(np.clip(np.sum(R_learn * pred_unit) / denom, 0.0, 2.5))
        else:
            alpha = 1.0

        removed = alpha * a_tgt[:, None] * P0_c[None, :]
        fin_sub = np.isfinite(Y_out[:, fit_idx])
        Y_out[:, fit_idx] = np.where(fin_sub, Y_out[:, fit_idx] - removed, np.nan)

        band_diags.append(
            {
                "band_index": orig_idx,
                "band": b,
                "w_fit": w_fit,
                "template": P0_c,
                "basis": P0_c[:, None],
                "coeffs": (alpha * a_tgt)[:, None],
                "amplitude": alpha * a_tgt,
                "amplitude_smooth": alpha * a_tgt,
                "centroid_shift": np.zeros(nt, dtype=float),
                "coupled": True,
                "coupling_alpha": alpha,
            }
        )

    band_diags.sort(key=lambda item: item["band_index"])

    if transmittance_space:
        Z_out = -od_scale * np.log10(np.clip(1.0 + Y_out, 1e-6, None))
    else:
        Z_out = Y_out

    n_sat = 0
    if mask_saturated and np.any(all_cores):
        sat_mask = (np.abs(M) > float(saturation_threshold)) & all_cores[None, :]
        n_sat = int(np.count_nonzero(sat_mask))
        Z_out[sat_mask] = np.nan

    Z_out[orig_nan_mask] = np.nan

    # Store pre-zero spectrum in the same signal space as M_orig for Dataset1D.bkg_avg
    bkg_orig = _robust_pre_zero_spectrum(M, neg_mask)
    return Z_out[:, inv_order], bkg_orig[inv_order], band_diags, n_sat


# --------------------------------------------------------------------------- #
#                             Public Entry Point                              #
# --------------------------------------------------------------------------- #
def compute_scatter_removal(
    dataset: Dataset1D,
    bands: list[ScatterBand] | None = None,
    *,
    tmin: float | None = None,
    tmax: float | None = None,
    method: str = "local_svd",
    rank: int = 3,
    poly_deg: int = 2,
    t0_exclude: tuple[float, float] = (-0.5, 1.5),
    scale_broadband: bool = True,
    per_scan: bool = True,
    update_noise: bool = True,
    transmittance_space: bool = False,
    mask_saturated: bool = False,
    saturation_threshold: float = 500.0,
    allow_positive: bool = False,
    detector: int | None = None,
) -> ScatterRemovalResult:
    """Run multi-band scatter and dynamic background removal on a :class:`Dataset1D`.

    Parameters
    ----------
    dataset : Dataset1D
        Input dataset.
    bands : list of ScatterBand, optional
        Scatter bands (in native probe units). When ``None``, bands are
        automatically detected via :func:`detect_scatter_bands`.
    tmin, tmax : float, optional
        Pre-zero delay window used for template extraction and baseline
        comparison. Defaults to delays ``<= -1.0`` (or earliest negative delays).
    method : {"local_svd", "taylor", "warped"}, default "local_svd"
        Scatter basis model.
    rank : int, default 3
        Number of local SVD modes or Taylor derivative order + 1 (per-band
        ``band.rank`` overrides this when set).
    poly_deg : int, default 2
        Polynomial degree for the smooth local TA baseline inside fit windows.
    t0_exclude : tuple of (float, float), default (-0.5, 1.5)
        Delay window excluded when learning ``local_svd`` modes so coherent
        artifacts around ``t0`` are preserved.
    scale_broadband : bool, default True
        Scale the non-scatter broadband pre-zero background by the slow
        Savitzky-Golay envelope of the primary scatter amplitude.
    per_scan : bool, default True
        Clean individual scans (``Zss_R``) before averaging when available.
    update_noise : bool, default True
        Recompute ``Zstdv`` across the cleaned single scans when ``per_scan``
        is active and ``Nscans > 1``.
    transmittance_space : bool, default False
        Convert ΔA to transmittance modulation ``10^(-ΔA/scale) - 1`` before
        decomposition (for very strong / optically nonlinear scatter).
    mask_saturated : bool, default False
        Mask raw pixels inside enabled scatter cores exceeding
        ``saturation_threshold`` to ``NaN``.
    saturation_threshold : float, default 500.0
        Absolute raw signal threshold for saturation masking (mOD).
    allow_positive : bool, default False
        Also auto-detect positive sharp pre-zero features.
    detector : int, optional
        If provided, process and compute diagnostics with respect to this
        detector channel (default ``None`` processes all channels and computes
        diagnostics for detector 0).

    Returns
    -------
    ScatterRemovalResult
    """
    method_norm = str(method).strip().lower()
    if method_norm not in VALID_SCATTER_METHODS:
        raise ValueError(
            f"Unknown scatter removal method {method!r}. Expected one of {VALID_SCATTER_METHODS}."
        )

    delays = np.asarray(dataset.delays, dtype=float)
    Zavg_R = np.asarray(dataset.Zavg_R, dtype=float)
    t_lo, t_hi, neg_mask = _resolve_pre_zero_window(delays, tmin, tmax)
    od_scale = 1.0 if _is_od_scale(getattr(dataset, "units", None)) else 1000.0

    diag_det = 0 if detector is None else int(np.clip(detector, 0, max(0, dataset.n_detectors - 1)))

    # Normalize working arrays to 3-D [Nt x Np x Ndet] internally
    was_2d = Zavg_R.ndim == 2
    Zavg_3d = Zavg_R[:, :, np.newaxis] if was_2d else Zavg_R
    nt, n_pix, n_det = Zavg_3d.shape

    if bands is None:
        p_diag = _extract_2d_probe(dataset.probe, diag_det)
        bands_use = detect_scatter_bands(
            Zavg_3d[:, :, diag_det],
            delays,
            p_diag,
            tmin=t_lo,
            tmax=t_hi,
            units=getattr(dataset, "units", None),
            allow_positive=allow_positive,
        )
    else:
        bands_use = [replace(b) for b in bands]

    # Standard static pre-zero baseline for comparison / metrics / difference map
    Z_static_3d, _, bkg_avg_static, bkg_ss_static = process.background_correct(
        Zavg_3d,
        dataset.Zss_R,
        dataset.nscans,
        delays,
        t_lo,
        t_hi,
        do_correct=True,
    )
    Z_static_C_raw = Z_static_3d[:, :, 0] if was_2d else Z_static_3d

    has_ss = bool(getattr(dataset, "has_single_scans", False))
    use_per_scan = bool(per_scan and has_ss)

    Zavg_out_3d = np.empty_like(Zavg_3d)
    bkg_out = np.zeros((n_pix, n_det), dtype=float)
    Zss_out = None
    bkg_ss_out = None
    Zstdv_out = None
    n_sat_total = 0
    diag_list: list[dict[str, Any]] = []
    interscan_std_before = float("nan")
    interscan_std_after = float("nan")

    dets_to_run = [diag_det] if detector is not None else list(range(n_det))

    if use_per_scan:
        Zss_R = np.asarray(dataset.Zss_R, dtype=float)
        n_scans = Zss_R.shape[3]
        Zss_out = np.empty_like(Zss_R)
        bkg_ss_out = np.zeros((n_pix, n_det, n_scans), dtype=float)

        if detector is not None:
            # Keep untouched detectors static-corrected
            for d_idx in range(n_det):
                if d_idx != diag_det:
                    Zavg_out_3d[:, :, d_idx] = Z_static_3d[:, :, d_idx]
                    Zss_out[:, :, d_idx, :] = Zss_R[:, :, d_idx, :]

        for d_idx in dets_to_run:
            p_d = _extract_2d_probe(dataset.probe, d_idx)
            for s_idx in range(n_scans):
                z_clean_s, bkg_s, _, n_sat_s = _clean_2d_matrix(
                    Zss_R[:, :, d_idx, s_idx],
                    delays,
                    p_d,
                    bands_use,
                    neg_mask=neg_mask,
                    method=method_norm,
                    rank=rank,
                    poly_deg=poly_deg,
                    t0_exclude=t0_exclude,
                    scale_broadband=scale_broadband,
                    transmittance_space=transmittance_space,
                    mask_saturated=mask_saturated,
                    saturation_threshold=saturation_threshold,
                    od_scale=od_scale,
                )
                Zss_out[:, :, d_idx, s_idx] = z_clean_s
                bkg_ss_out[:, d_idx, s_idx] = bkg_s
                n_sat_total += n_sat_s

            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                Zavg_out_3d[:, :, d_idx] = np.nanmean(Zss_out[:, :, d_idx, :], axis=-1)
            bkg_out[:, d_idx] = np.nanmean(bkg_ss_out[:, d_idx, :], axis=-1)

        # Compute diagnostics on the averaged cleaned signal for the target detector
        _, _, diag_list, _ = _clean_2d_matrix(
            Zavg_3d[:, :, diag_det],
            delays,
            _extract_2d_probe(dataset.probe, diag_det),
            bands_use,
            neg_mask=neg_mask,
            method=method_norm,
            rank=rank,
            poly_deg=poly_deg,
            t0_exclude=t0_exclude,
            scale_broadband=scale_broadband,
            transmittance_space=transmittance_space,
            mask_saturated=mask_saturated,
            saturation_threshold=saturation_threshold,
            od_scale=od_scale,
        )

        if update_noise and n_scans > 1:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                Zstdv_out = np.nanstd(Zss_out, axis=-1, ddof=1)
            # Fall back cleanly for detectors that were not processed when detector is specified
            if detector is not None and getattr(dataset, "Zstdv", None) is not None:
                orig_std = np.asarray(dataset.Zstdv, dtype=float)
                if orig_std.shape == Zstdv_out.shape:
                    for d_idx in range(n_det):
                        if d_idx != diag_det:
                            Zstdv_out[:, :, d_idx] = orig_std[:, :, d_idx]
    else:
        if detector is not None:
            for d_idx in range(n_det):
                if d_idx != diag_det:
                    Zavg_out_3d[:, :, d_idx] = Z_static_3d[:, :, d_idx]

        for d_idx in dets_to_run:
            p_d = _extract_2d_probe(dataset.probe, d_idx)
            z_clean_d, bkg_d, diags_d, n_sat_d = _clean_2d_matrix(
                Zavg_3d[:, :, d_idx],
                delays,
                p_d,
                bands_use,
                neg_mask=neg_mask,
                method=method_norm,
                rank=rank,
                poly_deg=poly_deg,
                t0_exclude=t0_exclude,
                scale_broadband=scale_broadband,
                transmittance_space=transmittance_space,
                mask_saturated=mask_saturated,
                saturation_threshold=saturation_threshold,
                od_scale=od_scale,
            )
            Zavg_out_3d[:, :, d_idx] = z_clean_d
            bkg_out[:, d_idx] = bkg_d
            n_sat_total += n_sat_d
            if d_idx == diag_det:
                diag_list = diags_d

    Zavg_C = Zavg_out_3d[:, :, 0] if was_2d else Zavg_out_3d
    Z_diff = Z_static_C_raw - Zavg_C
    bkg_avg_res = bkg_out[:, 0] if was_2d else bkg_out

    # Quality metrics on diagnostic detector inside enabled scatter cores
    p_diag = _extract_2d_probe(dataset.probe, diag_det)
    core_mask = np.zeros(n_pix, dtype=bool)
    wing_mask = np.zeros(n_pix, dtype=bool)
    med_step = float(np.median(np.abs(np.diff(np.sort(p_diag))))) if n_pix > 1 else 1.0
    for b in bands_use:
        if b.enabled:
            core_mask |= (p_diag >= b.p_min) & (p_diag <= b.p_max)
            wing_w, _ = _band_widths(b, med_step)
            wing_mask |= (p_diag >= b.p_min - wing_w) & (p_diag <= b.p_max + wing_w)
    wing_only = wing_mask & (~core_mask)
    if np.count_nonzero(wing_only) < 3:
        wing_only = ~core_mask

    eval_neg = delays <= float(t0_exclude[0]) if np.any(delays <= float(t0_exclude[0])) else neg_mask
    eval_pos = delays >= max(2.0, float(t0_exclude[1])) if np.any(delays >= 2.0) else (delays > 0.0)

    M_before = Z_static_3d[:, :, diag_det]
    M_after = Zavg_out_3d[:, :, diag_det]

    if np.any(core_mask):
        neg_b = M_before[np.ix_(eval_neg, core_mask)]
        neg_a = M_after[np.ix_(eval_neg, core_mask)]
        pre_rms_before = float(np.sqrt(np.nanmean(neg_b ** 2))) if np.any(np.isfinite(neg_b)) else float("nan")
        pre_rms_after = float(np.sqrt(np.nanmean(neg_a ** 2))) if np.any(np.isfinite(neg_a)) else float("nan")
        pos_rough_before = _second_diff_roughness(M_before, eval_pos, core_mask)
        pos_rough_after = _second_diff_roughness(M_after, eval_pos, core_mask)
    else:
        pre_rms_before = float(np.sqrt(np.nanmean(M_before[eval_neg] ** 2)))
        pre_rms_after = float(np.sqrt(np.nanmean(M_after[eval_neg] ** 2)))
        pos_rough_before = _second_diff_roughness(M_before, eval_pos, np.ones(n_pix, dtype=bool))
        pos_rough_after = _second_diff_roughness(M_after, eval_pos, np.ones(n_pix, dtype=bool))

    wing_rough_ref = _second_diff_roughness(M_before, eval_pos, wing_only)

    if use_per_scan and Zss_out is not None and Zss_out.shape[3] > 1 and np.any(core_mask):
        import warnings

        Zss_R = np.asarray(dataset.Zss_R, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            Zss_pre_b = np.nanmean(Zss_R[neg_mask, :, diag_det, :], axis=0, keepdims=True)
            std_b = np.nanstd(Zss_R[:, core_mask, diag_det, :] - Zss_pre_b[:, core_mask, :], axis=-1)
            std_a = np.nanstd(Zss_out[:, core_mask, diag_det, :], axis=-1)
            interscan_std_before = float(np.nanmedian(std_b))
            interscan_std_after = float(np.nanmedian(std_a))

    params = {
        "bands": [replace(b) for b in bands_use],
        "tmin": t_lo,
        "tmax": t_hi,
        "method": method_norm,
        "rank": int(rank),
        "poly_deg": int(poly_deg),
        "t0_exclude": (float(t0_exclude[0]), float(t0_exclude[1])),
        "scale_broadband": bool(scale_broadband),
        "per_scan": bool(per_scan),
        "update_noise": bool(update_noise),
        "transmittance_space": bool(transmittance_space),
        "mask_saturated": bool(mask_saturated),
        "saturation_threshold": float(saturation_threshold),
        "allow_positive": bool(allow_positive),
        "detector": detector,
    }

    metrics = {
        "pre_zero_rms_before": pre_rms_before,
        "pre_zero_rms_after": pre_rms_after,
        "pos_roughness_before": pos_rough_before,
        "pos_roughness_after": pos_rough_after,
        "wing_roughness_ref": wing_rough_ref,
        "interscan_std_before": interscan_std_before,
        "interscan_std_after": interscan_std_after,
        "used_per_scan": use_per_scan,
        "n_saturated_masked": n_sat_total,
    }

    return ScatterRemovalResult(
        Zavg_C=Zavg_C,
        Zss_C=Zss_out,
        Zstdv=Zstdv_out,
        bkg_avg=bkg_avg_res,
        bkg_ss=bkg_ss_out,
        Z_static_C=Z_static_C_raw,
        Z_diff=Z_diff,
        bands=bands_use,
        detector=diag_det,
        params=params,
        band_diagnostics=diag_list,
        metrics=metrics,
    )

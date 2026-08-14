"""Analysis stage of the 2-D pipeline (scaffold).

Provides simple slice/diagonal extraction now; centre-line-slope (CLS) and
other dynamics analyses will be added here and exposed through the same stage.
"""

from __future__ import annotations

import numpy as np

from ..log import get_logger

logger = get_logger(__name__)


def slice_at(data, t2):
    """Return ``(pump, probe, map)`` for the 2-D map nearest ``t2``."""
    idx = data.map_index(t2)
    return data.pump, data.probe, data.Z[:, :, idx]


def diagonal(
    data,
    t2: float | None = None,
    offset: float = 0.0,
    method: str = "linear",
    num_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(freq, signal)`` along the diagonal or off-diagonal trajectory (w3 = w1 + offset).

    Parameters
    ----------
    data : Dataset2D
        The 2D dataset.
    t2 : float, optional
        Target delay value. If None, extracts cut across all t2 delays as a 2D matrix [Nfreq x Nt2].
    offset : float, default 0.0
        Off-diagonal wavenumber shift along probe axis (w3 = w1 + offset).
    method : str, default 'linear'
        Interpolation method: 'linear', 'cubic', 'nearest', 'slinear', 'pchip'.
    num_points : int, optional
        Number of interpolation points along the cut.

    Returns
    -------
    freq : ndarray
        Pump frequency vector along the cut.
    signal : ndarray
        1D array [Nfreq] if t2 is specified, or 2D array [Nfreq x Nt2] if t2 is None.
    """
    from scipy.interpolate import RegularGridInterpolator

    p_min, p_max = float(np.min(data.pump)), float(np.max(data.pump))
    r_min, r_max = float(np.min(data.probe)), float(np.max(data.probe))

    lo = max(p_min, r_min - offset)
    hi = min(p_max, r_max - offset)

    if not lo < hi:
        raise ValueError(f"No valid overlap for off-diagonal cut with offset={offset}.")

    if num_points is not None and num_points > 1:
        freq = np.linspace(lo, hi, num_points)
    else:
        mask = (data.pump >= lo) & (data.pump <= hi)
        freq = data.pump[mask] if np.any(mask) else np.linspace(lo, hi, 50)

    probe_vals = freq + offset
    points = np.column_stack([freq, probe_vals])
    interp_method = "linear" if method == "spline" else method

    if t2 is not None:
        idx = data.map_index(t2)
        interp = RegularGridInterpolator(
            (data.pump, data.probe), data.Z[:, :, idx], method=interp_method, bounds_error=False, fill_value=0.0
        )
        signal = interp(points)
    else:
        Nt2 = data.n_maps
        signal = np.zeros((len(freq), Nt2), dtype=float)
        for i in range(Nt2):
            interp = RegularGridInterpolator(
                (data.pump, data.probe), data.Z[:, :, i], method=interp_method, bounds_error=False, fill_value=0.0
            )
            signal[:, i] = interp(points)

    return freq, signal


def antidiagonal(
    data,
    centre: tuple[float, float],
    t2: float | None = None,
    method: str = "linear",
    num_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return anti-diagonal cut passing through centre=(w1_0, w3_0) along w1 + w3 = w1_0 + w3_0.

    Parameters
    ----------
    data : Dataset2D
    centre : tuple (w1_0, w3_0)
        Peak centre coordinates.
    t2 : float, optional
    method : str, default 'linear'
    num_points : int, optional

    Returns
    -------
    rel_disp : ndarray
        Relative frequency displacement delta_w along anti-diagonal axis (in cm⁻¹).
    signal : ndarray
        1D array [Nfreq] if t2 is specified, or 2D array [Nfreq x Nt2] if t2 is None.
    """
    from scipy.interpolate import RegularGridInterpolator

    w1_0, w3_0 = float(centre[0]), float(centre[1])
    C = w1_0 + w3_0

    p_min, p_max = float(np.min(data.pump)), float(np.max(data.pump))
    r_min, r_max = float(np.min(data.probe)), float(np.max(data.probe))

    w1_lo = max(p_min, C - r_max)
    w1_hi = min(p_max, C - r_min)

    if not w1_lo < w1_hi:
        raise ValueError(f"Anti-diagonal cut through centre={centre} falls outside data range.")

    N = num_points if (num_points and num_points > 1) else 50
    w1_pts = np.linspace(w1_lo, w1_hi, N)
    w3_pts = C - w1_pts

    rel_disp = w1_pts - w1_0
    points = np.column_stack([w1_pts, w3_pts])
    interp_method = "linear" if method == "spline" else method

    if t2 is not None:
        idx = data.map_index(t2)
        interp = RegularGridInterpolator(
            (data.pump, data.probe), data.Z[:, :, idx], method=interp_method, bounds_error=False, fill_value=0.0
        )
        signal = interp(points)
    else:
        Nt2 = data.n_maps
        signal = np.zeros((len(rel_disp), Nt2), dtype=float)
        for i in range(Nt2):
            interp = RegularGridInterpolator(
                (data.pump, data.probe), data.Z[:, :, i], method=interp_method, bounds_error=False, fill_value=0.0
            )
            signal[:, i] = interp(points)

    return rel_disp, signal


def compare_diag_antidiag(
    data,
    centre: tuple[float, float],
    t2: float | None = None,
    method: str = "cubic",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract and normalise Diagonal and Anti-Diagonal profiles passing through centre=(w1_0, w3_0).

    Returns
    -------
    rel_disp : ndarray
        Relative frequency displacement delta_w.
    norm_diag : ndarray
        Diagonal profile normalised to peak max = 1.0.
    norm_antidiag : ndarray
        Anti-diagonal profile normalised to peak max = 1.0.
    """
    w1_0, w3_0 = centre
    offset = w3_0 - w1_0

    w1_d, sig_d = diagonal(data, t2=t2, offset=offset, method=method, num_points=100)
    rel_disp_d = w1_d - w1_0

    rel_disp_a, sig_a = antidiagonal(data, centre=centre, t2=t2, method=method, num_points=100)

    disp_min = max(np.min(rel_disp_d), np.min(rel_disp_a))
    disp_max = min(np.max(rel_disp_d), np.max(rel_disp_a))
    rel_disp = np.linspace(disp_min, disp_max, 100)

    norm_diag = np.interp(rel_disp, rel_disp_d, sig_d)
    norm_antidiag = np.interp(rel_disp, rel_disp_a, sig_a)

    max_d = np.max(np.abs(norm_diag)) or 1.0
    max_a = np.max(np.abs(norm_antidiag)) or 1.0

    norm_diag = norm_diag / max_d
    norm_antidiag = norm_antidiag / max_a

    return rel_disp, norm_diag, norm_antidiag


def integral_dynamics(
    data,
    pump_range: tuple[float, float],
    probe_range: tuple[float, float],
    method: str = "trapezoid",
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate 2D signal over a rectangular ROI (pump_range x probe_range) across all t2 delays.

    Parameters
    ----------
    data : Dataset2D
    pump_range : tuple (pump_min, pump_max)
    probe_range : tuple (probe_min, probe_max)
    method : str, default 'trapezoid'

    Returns
    -------
    delays : ndarray
        t2 population delay values.
    I_t2 : ndarray
        1D array of integrated intensity per t2 delay.
    """
    from scipy.integrate import trapezoid

    p_min, p_max = min(pump_range), max(pump_range)
    r_min, r_max = min(probe_range), max(probe_range)

    p_mask = (data.pump >= p_min) & (data.pump <= p_max)
    r_mask = (data.probe >= r_min) & (data.probe <= r_max)

    if not np.any(p_mask) or not np.any(r_mask):
        raise ValueError(f"Selected ROI [{p_min:.1f}, {p_max:.1f}] x [{r_min:.1f}, {r_max:.1f}] contains no data points.")

    sub_pump = data.pump[p_mask]
    sub_probe = data.probe[r_mask]
    sub_Z = data.Z[p_mask, :, :][:, r_mask, :]

    Nt2 = data.n_maps
    I_t2 = np.zeros(Nt2, dtype=float)

    for i in range(Nt2):
        Z_map = sub_Z[:, :, i]
        if len(sub_pump) > 1 and len(sub_probe) > 1:
            int_probe = trapezoid(Z_map, sub_probe, axis=1)
            total_val = trapezoid(int_probe, sub_pump, axis=0)
        else:
            total_val = float(np.sum(Z_map))
        I_t2[i] = total_val

    return data.delays, I_t2


def _extremum_from_poly(coeffs, x_lo, x_hi, x0, want_min: bool):
    """Stationary point of a polynomial inside ``[x_lo, x_hi]``, nearest ``x0``.

    Returns ``None`` when the polynomial has no stationary point of the right
    kind (minimum vs maximum) inside the window.
    """
    deriv = np.polyder(coeffs)
    if deriv.size < 1 or not np.any(deriv):
        return None
    roots = np.roots(deriv)
    second = np.polyder(deriv)
    candidates = []
    for r in roots:
        if abs(r.imag) > 1e-9:
            continue
        xr = float(r.real)
        if not (x_lo <= xr <= x_hi):
            continue
        curvature = float(np.polyval(second, xr)) if second.size else 0.0
        if want_min and curvature <= 0:
            continue
        if not want_min and curvature >= 0:
            continue
        candidates.append(xr)
    if not candidates:
        return None
    return min(candidates, key=lambda xr: abs(xr - x0))


def _fit_extremum(xs, ys, x0: float, method: str, want_min: bool):
    """Stationary point of a local model fitted to ``(xs, ys)``, anchored at ``x0``.

    ``x0`` is the starting estimate (the fine-grid position); the model is fitted
    to the **measured** points in ``xs`` so the result varies continuously and
    the noise is averaged over several samples. Returns ``None`` if no suitable
    extremum is found inside the window.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 3:
        return None
    # Fit in a local frame: absolute wavenumbers (~2000) would make the
    # Vandermonde matrix badly conditioned.
    xl = xs - x0

    order = {"quadratic": 2, "cubic": 3, "quartic": 4}.get(method)
    if order is not None:
        if xl.size < order + 1:
            order = 2
        try:
            coeffs = np.polyfit(xl, ys, order)
        except Exception:
            logger.debug("Polynomial peak refinement failed.", exc_info=True)
            return None
        root = _extremum_from_poly(coeffs, float(xl[0]), float(xl[-1]), 0.0, want_min)
        return None if root is None else x0 + root

    if method in ("gaussian", "lorentzian"):
        import warnings

        from scipy.optimize import curve_fit

        span = float(xl[-1] - xl[0]) or 1.0
        offset = float(np.median([ys[0], ys[-1]]))
        amp = float(ys[int(np.argmin(np.abs(xl)))] - offset) or (-1.0 if want_min else 1.0)

        def gaussian(xv, xc, width, a, c):
            return c + a * np.exp(-0.5 * ((xv - xc) / width) ** 2)

        def lorentzian(xv, xc, width, a, c):
            return c + a / (1.0 + ((xv - xc) / width) ** 2)

        model = gaussian if method == "gaussian" else lorentzian
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # a perfect fit has no covariance
                popt, _ = curve_fit(
                    model, xl, ys, p0=[0.0, 0.25 * span, amp, offset], maxfev=2000
                )
        except Exception:
            logger.debug("%s peak refinement failed.", method, exc_info=True)
            return None
        return x0 + popt[0] if xl[0] <= popt[0] <= xl[-1] else None

    return None


def _extremum_from_poly(coeffs, x_lo, x_hi, x0, want_min: bool):
    """Stationary point of a polynomial inside ``[x_lo, x_hi]``, nearest ``x0``.

    Returns ``None`` when the polynomial has no stationary point of the right
    kind (minimum vs maximum) inside the window.
    """
    deriv = np.polyder(coeffs)
    if deriv.size < 1 or not np.any(deriv):
        return None
    roots = np.roots(deriv)
    second = np.polyder(deriv)
    candidates = []
    for r in roots:
        if abs(r.imag) > 1e-9:
            continue
        xr = float(r.real)
        if not (x_lo <= xr <= x_hi):
            continue
        curvature = float(np.polyval(second, xr)) if second.size else 0.0
        if want_min and curvature <= 0:
            continue
        if not want_min and curvature >= 0:
            continue
        candidates.append(xr)
    if not candidates:
        return None
    return min(candidates, key=lambda xr: abs(xr - x0))


def subpixel_peak(x, y, idx, method="quadratic", window: int = 2, want_min: bool | None = None):
    """Refine the extremum of ``y(x)`` near sample ``idx`` to sub-pixel accuracy.

    A local model is least-squares fitted to the data over ``+/-window`` samples
    around ``idx`` and its stationary point is returned, so the result varies
    continuously with the data instead of snapping to the sampling grid. Using
    several points (rather than the classic three-point formula) also averages
    the noise, which matters for the CLS/IvCLS centre lines of real spectra.

    ``method`` is ``"quadratic"``/``"cubic"``/``"quartic"`` (polynomial of that
    order), ``"gaussian"``/``"lorentzian"`` (peak-shape fit), or ``"none"``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        return float("nan")
    idx = int(np.clip(idx, 0, n - 1))
    method = method.value if hasattr(method, "value") else str(method)
    if method == "none" or n < 3:
        return float(x[idx])
    if want_min is None:
        want_min = bool(y[idx] <= np.median(y))

    order = {"quadratic": 2, "cubic": 3, "quartic": 4}.get(method)
    half = max(int(window), -(-(order + 1) // 2) if order else 2)
    lo, hi = max(0, idx - half), min(n, idx + half + 1)
    peak = _fit_extremum(x[lo:hi], y[lo:hi], float(x[idx]), method, want_min)
    return float(peak) if peak is not None else float(x[idx])


def robust_polyfit(x, y):
    """Robust linear fit (y = m * x + c) using Huber loss minimization."""
    from scipy.optimize import minimize

    if len(x) < 3:
        # Fallback to standard linear fit if not enough points
        return np.polyfit(x, y, 1)

    p0 = np.polyfit(x, y, 1)

    def huber_loss(params, x, y):
        m, c = params
        diff = y - (m * x + c)
        std_dev = np.std(diff) if np.std(diff) > 1e-5 else 1.0
        delta = 1.345 * std_dev
        abs_diff = np.abs(diff)
        loss = np.where(abs_diff <= delta,
                        0.5 * (diff ** 2),
                        delta * (abs_diff - 0.5 * delta))
        return np.sum(loss)

    res = minimize(huber_loss, p0, args=(x, y), method='BFGS')
    if res.success:
        return res.x
    return p0


def _upsample_slice(x, y, factor):
    """Upsample a 1-D slice by *factor* using cubic spline interpolation.

    Returns ``(x_fine, y_fine)`` with ``factor`` times as many points.
    Falls back to the original arrays if *factor* <= 1 or the slice is too
    short for cubic interpolation (< 4 points).
    """
    if factor <= 1 or len(x) < 4:
        return x, y
    from scipy.interpolate import CubicSpline
    try:
        cs = CubicSpline(x, y)
        x_fine = np.linspace(x[0], x[-1], len(x) * factor)
        return x_fine, cs(x_fine)
    except Exception:
        return x, y


def _refine_extremum(x_search, z_search, want_min: bool, method: str, factor: int, window: int = 2):
    """Sub-pixel position (and depth) of the extremum of one slice.

    Two stages, so the answer is never tied to the sampling grid:

    1. the slice is interpolated onto a grid ``factor`` times finer (cubic
       spline) and the extremum is *located* there, already resolving positions
       below the spectral resolution of the axis;
    2. a local model (see :func:`subpixel_peak`) is least-squares fitted to the
       **measured** points around that location and its stationary point is
       returned, which refines the estimate further and averages the noise.

    With ``method="none"`` the fine-grid position is returned as is, so the
    interpolation factor alone still gives sub-pixel resolution.
    """
    x_search = np.asarray(x_search, dtype=float)
    z_search = np.asarray(z_search, dtype=float)
    x_fine, z_fine = _upsample_slice(x_search, z_search, factor)
    fine_idx = int(np.argmin(z_fine)) if want_min else int(np.argmax(z_fine))
    x_peak = float(x_fine[fine_idx])
    z_peak = float(z_fine[fine_idx])

    method = method.value if hasattr(method, "value") else str(method)
    if method == "none" or x_search.size < 3:
        return x_peak, z_peak

    # Measured points within +/-window pixels of the located extremum. The
    # window follows the fine-grid position rather than the nearest sample, so
    # nothing snaps back onto the measured grid.
    step = float(np.median(np.abs(np.diff(x_search)))) or 1.0
    order = {"quadratic": 2, "cubic": 3, "quartic": 4}.get(method, 2)
    half = max(int(window), -(-(order + 1) // 2))
    sel = np.abs(x_search - x_peak) <= half * step
    if np.count_nonzero(sel) < order + 1:  # widen until the fit is determined
        nearest = int(np.argmin(np.abs(x_search - x_peak)))
        lo = max(0, nearest - half)
        sel = np.zeros_like(x_search, dtype=bool)
        sel[lo : lo + 2 * half + 1] = True

    fitted = _fit_extremum(x_search[sel], z_search[sel], x_peak, method, want_min)
    if fitted is not None and float(np.min(x_search)) <= fitted <= float(np.max(x_search)):
        x_peak = float(fitted)
    return x_peak, z_peak


def center_line_slope(data, inverse=False, both=False, pump_range=None, probe_range=None, t2_range=None) -> np.ndarray:
    """Centre-line-slope (CLS) analysis of t2 dynamics."""
    from ..settings import get_settings
    s = get_settings()

    # Retrieve configuration parameters
    sd_intensity_threshold = getattr(s, "sd_intensity_threshold", 0.1)
    sd_peak_range = getattr(s, "sd_peak_range", 10.0)
    sd_peak_type = getattr(s, "sd_peak_type", "min")
    sd_interpolation = getattr(s, "sd_interpolation", "quadratic")
    sd_interpolation_factor = int(getattr(s, "sd_interpolation_factor", 4))
    if hasattr(sd_peak_type, "value"):
        sd_peak_type = sd_peak_type.value
    if hasattr(sd_interpolation, "value"):
        sd_interpolation = sd_interpolation.value

    # Caches for visual overlay rendering
    if not hasattr(data, "_cls_cache"):
        data._cls_cache = {}
    if not hasattr(data, "_ivcls_cache"):
        data._ivcls_cache = {}

    delays = data.delays

    cls_slopes = []
    ivcls_slopes = []

    # Range masks
    if pump_range is not None:
        p_min, p_max = pump_range
        pump_mask = (data.pump >= p_min) & (data.pump <= p_max)
    else:
        pump_mask = np.ones_like(data.pump, dtype=bool)

    if probe_range is not None:
        pr_min, pr_max = probe_range
        probe_mask = (data.probe >= pr_min) & (data.probe <= pr_max)
    else:
        probe_mask = np.ones_like(data.probe, dtype=bool)

    for idx, t2 in enumerate(delays):
        if t2_range is not None:
            t2_min, t2_max = t2_range
            if not (t2_min <= t2 <= t2_max):
                data._cls_cache[t2] = None
                data._ivcls_cache[t2] = None
                cls_slopes.append(np.nan)
                ivcls_slopes.append(np.nan)
                continue
        Z_map = data.Z[:, :, idx]
        if pump_range is not None or probe_range is not None:
            sub_Z = Z_map[np.ix_(pump_mask, probe_mask)]
            A_max = np.nanmax(np.abs(sub_Z)) if sub_Z.size > 0 else np.nanmax(np.abs(Z_map))
        else:
            A_max = np.nanmax(np.abs(Z_map))
        threshold_val = sd_intensity_threshold * A_max

        # 1. CLS (slice along probe axis for each pump)
        cls_pts = []
        for i, w1 in enumerate(data.pump):
            if not pump_mask[i]:
                continue
            Z_slice = Z_map[i, :]
            # Search within ±sd_peak_range from the rough peak in the ROI
            roi_indices = np.nonzero(probe_mask)[0]
            if len(roi_indices) == 0:
                continue
            z_roi = Z_slice[roi_indices]
            rough_pk_idx = roi_indices[np.argmin(z_roi)] if sd_peak_type == "min" else roi_indices[np.argmax(z_roi)]
            w_rough = data.probe[rough_pk_idx]

            search_mask = (data.probe >= w_rough - sd_peak_range) & (data.probe <= w_rough + sd_peak_range)
            search_mask = search_mask & probe_mask
            if not np.any(search_mask):
                continue
            indices = np.nonzero(search_mask)[0]
            probe_search = data.probe[indices]
            z_search = Z_slice[indices]

            probe_peak, z_peak = _refine_extremum(
                probe_search, z_search, sd_peak_type == "min",
                sd_interpolation, sd_interpolation_factor,
            )
            if np.abs(z_peak) >= threshold_val:
                cls_pts.append((w1, probe_peak))

        # Store for rendering
        if len(cls_pts) > 1:
            cls_pts_arr = np.array(cls_pts)
            # Fit line using robust_polyfit: probe_peak = m * w1 + c
            m, c = robust_polyfit(cls_pts_arr[:, 0], cls_pts_arr[:, 1])
            data._cls_cache[t2] = {"points": cls_pts_arr, "fit": (m, c)}
            cls_slopes.append(m)
        else:
            data._cls_cache[t2] = None
            cls_slopes.append(np.nan)

        # 2. IvCLS (slice along pump axis for each probe)
        ivcls_pts = []
        for j, w3 in enumerate(data.probe):
            if not probe_mask[j]:
                continue
            Z_slice = Z_map[:, j]
            # Search within ±sd_peak_range from the rough peak in the ROI
            roi_indices = np.nonzero(pump_mask)[0]
            if len(roi_indices) == 0:
                continue
            z_roi = Z_slice[roi_indices]
            rough_pk_idx = roi_indices[np.argmin(z_roi)] if sd_peak_type == "min" else roi_indices[np.argmax(z_roi)]
            w_rough = data.pump[rough_pk_idx]

            search_mask = (data.pump >= w_rough - sd_peak_range) & (data.pump <= w_rough + sd_peak_range)
            search_mask = search_mask & pump_mask
            if not np.any(search_mask):
                continue
            indices = np.nonzero(search_mask)[0]
            pump_search = data.pump[indices]
            z_search = Z_slice[indices]

            pump_peak, z_peak = _refine_extremum(
                pump_search, z_search, sd_peak_type == "min",
                sd_interpolation, sd_interpolation_factor,
            )
            if np.abs(z_peak) >= threshold_val:
                ivcls_pts.append((pump_peak, w3))

        # Store for rendering
        if len(ivcls_pts) > 1:
            ivcls_pts_arr = np.array(ivcls_pts)
            # Fit line using robust_polyfit: pump_peak = m * w3 + c
            m, c = robust_polyfit(ivcls_pts_arr[:, 1], ivcls_pts_arr[:, 0])
            data._ivcls_cache[t2] = {"points": ivcls_pts_arr, "fit": (m, c)}
            ivcls_slopes.append(m)
        else:
            data._ivcls_cache[t2] = None
            ivcls_slopes.append(np.nan)

    # Prepare return arrays
    if both:
        return np.column_stack([delays, cls_slopes, ivcls_slopes])
    elif inverse:
        return np.column_stack([delays, ivcls_slopes])
    else:
        return np.column_stack([delays, cls_slopes])


def nodal_line_slope(data, pump_range=None, probe_range=None, t2_range=None) -> np.ndarray:
    """Nodal line slope (NLS) analysis of t2 dynamics."""
    from ..settings import get_settings
    s = get_settings()

    sd_peak_range = getattr(s, "sd_peak_range", 10.0)
    sd_interpolation_factor = int(getattr(s, "sd_interpolation_factor", 4))

    # Cache for visual overlay rendering
    if not hasattr(data, "_nls_cache"):
        data._nls_cache = {}

    delays = data.delays
    nls_slopes = []

    # Range masks
    if pump_range is not None:
        p_min, p_max = pump_range
        pump_mask = (data.pump >= p_min) & (data.pump <= p_max)
    else:
        pump_mask = np.ones_like(data.pump, dtype=bool)

    if probe_range is not None:
        pr_min, pr_max = probe_range
        probe_mask = (data.probe >= pr_min) & (data.probe <= pr_max)
    else:
        probe_mask = np.ones_like(data.probe, dtype=bool)

    for idx, t2 in enumerate(delays):
        if t2_range is not None:
            t2_min, t2_max = t2_range
            if not (t2_min <= t2 <= t2_max):
                data._nls_cache[t2] = None
                nls_slopes.append(np.nan)
                continue
        Z_map = data.Z[:, :, idx]

        nls_pts = []
        for i, w1 in enumerate(data.pump):
            if not pump_mask[i]:
                continue
            Z_slice = Z_map[i, :]
            # Search within ±sd_peak_range from the diagonal
            search_mask = (data.probe >= w1 - sd_peak_range) & (data.probe <= w1 + sd_peak_range)
            search_mask = search_mask & probe_mask
            if not np.any(search_mask):
                continue
            indices = np.nonzero(search_mask)[0]
            probe_search = data.probe[indices]
            z_search = Z_slice[indices]

            # Upsample before zero-crossing search for sub-pixel accuracy
            probe_fine, z_fine = _upsample_slice(probe_search, z_search, sd_interpolation_factor)

            # Find zero crossings in search range
            crossings = []
            for k in range(len(probe_fine) - 1):
                v1 = z_fine[k]
                v2 = z_fine[k + 1]
                if v1 * v2 <= 0 and v1 != v2:
                    # Linear interpolation between adjacent fine-grid points
                    w3_val = probe_fine[k] + (-v1 / (v2 - v1)) * (probe_fine[k + 1] - probe_fine[k])
                    crossings.append(w3_val)

            if crossings:
                # Use the crossing closest to the diagonal w1
                closest_w3 = crossings[np.argmin(np.abs(np.array(crossings) - w1))]
                nls_pts.append((w1, closest_w3))

        # Store for rendering
        if len(nls_pts) > 1:
            nls_pts_arr = np.array(nls_pts)
            # Fit line: probe_crossing = m * w1 + c
            m, c = robust_polyfit(nls_pts_arr[:, 0], nls_pts_arr[:, 1])
            data._nls_cache[t2] = {"points": nls_pts_arr, "fit": (m, c)}
            nls_slopes.append(m)
        else:
            data._nls_cache[t2] = None
            nls_slopes.append(np.nan)

    return np.column_stack([delays, nls_slopes])


class SliceDataset1D:
    """A lightweight adapter wrapping a 2D slice/cut as a 1D dataset.

    This allows us to reuse the 1D plotting functions (like plot_spectra and plot_kinetics)
    without rewriting them.
    """
    def __init__(self, delays, probe, Z_matrix, units, source=None):
        self.delays = np.asarray(delays)
        self.probe = np.asarray(probe)
        self.Z_R = np.asarray(Z_matrix)  # shape: [Ndelays x Nprobe]
        self.units = units
        self.source = source

    def _detector_slice(self, detector: int = 0):
        return self.Z_R

    def noise_array(self):
        """A 2-D slice carries no per-point noise; always ``None``."""
        return None

    def _noise_array(self):
        """Deprecated alias of :meth:`noise_array`."""
        return self.noise_array()

    def _resolve(self, settings, label_style, delta_a_units):
        from ..oneD.dataset import DeltaAUnits
        from ..settings import get_settings
        s = settings or get_settings()
        style = label_style if label_style is not None else s.label_style.value
        if isinstance(delta_a_units, str):
            delta_a_units = DeltaAUnits(delta_a_units)
        units = s.units_with_convention(self.units, delta_a_units)
        return s, style, units


def get_slice_at_pump(data, pump_wn) -> SliceDataset1D:
    """Return a SliceDataset1D cut along the probe axis at a fixed pump wavenumber."""
    idx = int(np.argmin(np.abs(data.pump - pump_wn)))
    Z_slice = data.Z[idx, :, :].T  # Transpose [Nprobe x Nt2] to [Nt2 x Nprobe]
    return SliceDataset1D(data.delays, data.probe, Z_slice, data.units, data.source)


def get_slice_at_probe(data, probe_wn) -> SliceDataset1D:
    """Return a SliceDataset1D cut along the pump axis at a fixed probe wavenumber."""
    idx = int(np.argmin(np.abs(data.probe - probe_wn)))
    Z_slice = data.Z[:, idx, :].T  # Transpose [Npump x Nt2] to [Nt2 x Npump]
    return SliceDataset1D(data.delays, data.pump, Z_slice, data.units, data.source)


def get_slice_integrate_pump(data, pump_min, pump_max) -> SliceDataset1D:
    """Integrate along the pump axis between pump_min and pump_max, returning a SliceDataset1D."""
    from scipy.integrate import trapezoid
    mask = (data.pump >= min(pump_min, pump_max)) & (data.pump <= max(pump_min, pump_max))
    if not np.any(mask):
        raise ValueError(f"No pump wavenumbers found in range [{pump_min}, {pump_max}]")

    sub_Z = data.Z[mask, :, :]
    sub_pump = data.pump[mask]

    if len(sub_pump) > 1:
        # Integrate along pump axis (axis 0)
        Z_int = trapezoid(sub_Z, sub_pump, axis=0)
    else:
        Z_int = sub_Z[0, :, :]

    return SliceDataset1D(data.delays, data.probe, Z_int.T, data.units, data.source)


def get_slice_integrate_probe(data, probe_min, probe_max) -> SliceDataset1D:
    """Integrate along the probe axis between probe_min and probe_max, returning a SliceDataset1D."""
    from scipy.integrate import trapezoid
    mask = (data.probe >= min(probe_min, probe_max)) & (data.probe <= max(probe_min, probe_max))
    if not np.any(mask):
        raise ValueError(f"No probe wavenumbers found in range [{probe_min}, {probe_max}]")

    sub_Z = data.Z[:, mask, :]
    sub_probe = data.probe[mask]

    if len(sub_probe) > 1:
        # Integrate along probe axis (axis 1)
        Z_int = trapezoid(sub_Z, sub_probe, axis=1)
    else:
        Z_int = sub_Z[:, 0, :]

    return SliceDataset1D(data.delays, data.pump, Z_int.T, data.units, data.source)


# ----------------------------------------------------------------- #
#                        2D Gaussian Fitting                        #
# ----------------------------------------------------------------- #

def evaluate_2d_gaussian_map(pump, probe, modes, correlated: bool = False) -> np.ndarray:
    """Evaluate a multi-mode 2D Gaussian spectrum (GSB + ESA feature pairs).

    Parameters
    ----------
    pump : array-like, shape (Npump,)
        Pump frequency vector (w1).
    probe : array-like, shape (Nprobe,)
        Probe frequency vector (w3).
    modes : list of dict
        Each dict defines one mode pair with keys:
        - 'w1': pump centre
        - 'w3': probe GSB centre
        - 'anharm': anharmonicity shift along w3 (ESA = w3 - anharm)
        - 'amp_gsb': GSB amplitude
        - 'amp_esa': ESA amplitude
        - 'sigma_w1': Gaussian std dev along pump axis
        - 'sigma_w3': Gaussian std dev along probe axis
        - 'rho': (optional) correlation coefficient in [-0.99, 0.99]
    correlated : bool, default False
        Whether to include correlation (tilted 2D Gaussian) via rho.

    Returns
    -------
    ndarray, shape (Npump, Nprobe)
        Simulated 2D spectrum map.
    """
    P, R = np.meshgrid(np.asarray(pump), np.asarray(probe), indexing="ij")
    Z_sim = np.zeros_like(P, dtype=float)

    for mode in modes:
        w1 = float(mode["w1"])
        w3 = float(mode["w3"])
        anharm = float(mode.get("anharm", 15.0))
        amp_gsb = float(mode.get("amp_gsb", -1.0))
        amp_esa = float(mode.get("amp_esa", 0.8))
        sigma_w1 = max(float(mode.get("sigma_w1", 10.0)), 1e-3)
        sigma_w3 = max(float(mode.get("sigma_w3", 10.0)), 1e-3)
        rho = float(mode.get("rho", 0.0)) if correlated else 0.0
        rho = np.clip(rho, -0.95, 0.95)

        x = P - w1
        y_gsb = R - w3
        y_esa = R - (w3 - anharm)

        if correlated and abs(rho) > 1e-4:
            denom = 1.0 - rho**2
            Q_gsb = (1.0 / denom) * (
                (x / sigma_w1) ** 2
                - (2.0 * rho * x * y_gsb) / (sigma_w1 * sigma_w3)
                + (y_gsb / sigma_w3) ** 2
            )
            Q_esa = (1.0 / denom) * (
                (x / sigma_w1) ** 2
                - (2.0 * rho * x * y_esa) / (sigma_w1 * sigma_w3)
                + (y_esa / sigma_w3) ** 2
            )
        else:
            Q_gsb = (x / sigma_w1) ** 2 + (y_gsb / sigma_w3) ** 2
            Q_esa = (x / sigma_w1) ** 2 + (y_esa / sigma_w3) ** 2

        Z_sim += amp_gsb * np.exp(-0.5 * Q_gsb) + amp_esa * np.exp(-0.5 * Q_esa)

    return Z_sim


def fit_2d_gaussian_map(
    pump,
    probe,
    map_2d: np.ndarray,
    initial_modes: list[dict],
    correlated: bool = False,
    bounds_config: dict | None = None,
    normalize_t2: bool = False,
    progress_callback=None,
    show_progress: bool = False,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Fit a single 2D map with multi-mode GSB/ESA 2D Gaussians.

    Parameters
    ----------
    pump, probe : array-like
        Frequency axes.
    map_2d : ndarray, shape (Npump, Nprobe)
        Experimental 2D spectrum map.
    initial_modes : list of dict
        Initial guesses for mode parameters.
    correlated : bool, default False
        Whether to fit correlation coefficient rho.
    bounds_config : dict, optional
        Custom bounds dictionary with keys 'anharm_min', 'anharm_max', 'sigma_min', 'sigma_max'.
    normalize_t2 : bool, default False
        If True, scale the map by its maximum amplitude before fitting.
    progress_callback : callable, optional
        Callback function `fn(current_step, total_steps, label)` for progress updates.
    show_progress : bool, default False
        Whether to automatically display a GUI/CLI progress bar.

    Returns
    -------
    fitted_modes : list of dict
    fit_map : ndarray, shape (Npump, Nprobe)
    residual_map : ndarray, shape (Npump, Nprobe)
    """
    from scipy.optimize import least_squares
    from .progress import ProgressTracker

    pump = np.asarray(pump)
    probe = np.asarray(probe)
    map_2d = np.asarray(map_2d)

    scale_factor = float(np.max(np.abs(map_2d))) if (normalize_t2 and np.max(np.abs(map_2d)) > 0) else 1.0
    map_2d_fit = map_2d / scale_factor

    # Build initial parameter vector p0 and bounds
    # Per mode: [w1, w3, anharm, amp_gsb, amp_esa, sigma_w1, sigma_w3, (rho)]
    p0 = []
    bounds_lower = []
    bounds_upper = []

    w1_min, w1_max = float(np.min(pump)), float(np.max(pump))
    w3_min, w3_max = float(np.min(probe)), float(np.max(probe))
    w1_span = w1_max - w1_min
    w3_span = w3_max - w3_min
    max_amp = (float(np.max(np.abs(map_2d_fit))) * 5.0 or 10.0)

    cfg = bounds_config or {}

    for m in initial_modes:
        w1 = float(m.get("w1", (w1_min + w1_max) / 2))
        w3 = float(m.get("w3", (w3_min + w3_max) / 2))
        anharm = float(m.get("anharm", 15.0))
        amp_gsb = float(m.get("amp_gsb", -1.0)) / scale_factor
        amp_esa = float(m.get("amp_esa", 0.8)) / scale_factor
        sigma_w1 = float(m.get("sigma_w1", 10.0))
        sigma_w3 = float(m.get("sigma_w3", 10.0))
        rho = float(m.get("rho", 0.0))

        anharm_min = float(m.get("anharm_min", cfg.get("anharm_min", 0.1)))
        anharm_max = float(m.get("anharm_max", cfg.get("anharm_max", w3_span)))
        sigma_min = float(m.get("sigma_min", cfg.get("sigma_min", 0.5)))
        sigma_max = float(m.get("sigma_max", cfg.get("sigma_max", max(w1_span, w3_span))))

        # Clamp initial guesses inside bounds
        anharm = np.clip(anharm, anharm_min, anharm_max)
        sigma_w1 = np.clip(sigma_w1, sigma_min, sigma_max)
        sigma_w3 = np.clip(sigma_w3, sigma_min, sigma_max)

        p0.extend([w1, w3, anharm, amp_gsb, amp_esa, sigma_w1, sigma_w3])
        bounds_lower.extend([w1_min - 0.2 * w1_span, w3_min - 0.2 * w3_span, anharm_min, -max_amp, -max_amp, sigma_min, sigma_min])
        bounds_upper.extend([w1_max + 0.2 * w1_span, w3_max + 0.2 * w3_span, anharm_max, max_amp, max_amp, sigma_max, sigma_max])

        if correlated:
            p0.append(rho)
            bounds_lower.append(-0.95)
            bounds_upper.append(0.95)

    n_params_per_mode = 8 if correlated else 7
    max_nfev = 1000
    eval_counter = 0

    tracker = None
    if show_progress and progress_callback is None:
        tracker = ProgressTracker(total=100, title="2D Gaussian Fit", label="Optimizing 2D Gaussian fit...")

    def residuals(p):
        nonlocal eval_counter
        eval_counter += 1
        pct = min(99, int(100 * eval_counter / 300))
        lbl = f"Fitting 2D map... (eval {eval_counter})"
        if progress_callback is not None:
            progress_callback(pct, 100, lbl)
        elif tracker is not None and eval_counter % 3 == 0:
            tracker.update(pct, lbl)

        modes = []
        for i in range(len(initial_modes)):
            offset = i * n_params_per_mode
            mode_dict = {
                "w1": p[offset],
                "w3": p[offset + 1],
                "anharm": p[offset + 2],
                "amp_gsb": p[offset + 3],
                "amp_esa": p[offset + 4],
                "sigma_w1": p[offset + 5],
                "sigma_w3": p[offset + 6],
            }
            if correlated:
                mode_dict["rho"] = p[offset + 7]
            modes.append(mode_dict)
        sim = evaluate_2d_gaussian_map(pump, probe, modes, correlated=correlated)
        return (sim - map_2d_fit).ravel()

    try:
        res = least_squares(residuals, p0, bounds=(bounds_lower, bounds_upper), max_nfev=max_nfev)
        if tracker is not None:
            tracker.update(100, "Fitting complete.")
    finally:
        if tracker is not None:
            tracker.close()

    p_opt = res.x
    fitted_modes = []
    for i in range(len(initial_modes)):
        offset = i * n_params_per_mode
        mode_dict = {
            "w1": float(p_opt[offset]),
            "w3": float(p_opt[offset + 1]),
            "anharm": float(p_opt[offset + 2]),
            "amp_gsb": float(p_opt[offset + 3]) * scale_factor,
            "amp_esa": float(p_opt[offset + 4]) * scale_factor,
            "sigma_w1": float(p_opt[offset + 5]),
            "sigma_w3": float(p_opt[offset + 6]),
        }
        if correlated:
            mode_dict["rho"] = float(p_opt[offset + 7])
        fitted_modes.append(mode_dict)

    fit_map = evaluate_2d_gaussian_map(pump, probe, fitted_modes, correlated=correlated)
    residual_map = map_2d - fit_map
    return fitted_modes, fit_map, residual_map


def fit_2d_gaussian_global(
    pump,
    probe,
    delays,
    cube_3d: np.ndarray,
    initial_modes: list[dict],
    correlated: bool = False,
    bounds_config: dict | None = None,
    normalize_t2: bool = False,
    progress_callback=None,
    show_progress: bool = False,
) -> tuple[list[dict], list[list[dict]], np.ndarray, np.ndarray]:
    """Fit a 3D dataset cube globally across all population delays t2.

    Shared parameters across all delays: w1, w3, anharm, sigma_w1, sigma_w3, (rho).
    Independent parameters per delay: amp_gsb(t2), amp_esa(t2).

    Parameters
    ----------
    pump, probe, delays : array-like
    cube_3d : ndarray, shape (Npump, Nprobe, Nt2)
    initial_modes : list of dict
    correlated : bool, default False
    bounds_config : dict, optional
        Custom bounds dictionary with keys 'anharm_min', 'anharm_max', 'sigma_min', 'sigma_max'.
    normalize_t2 : bool, default False
        If True, normalise each t2 delay map by its maximum amplitude before fitting.
    progress_callback : callable, optional
        Callback function `fn(current_step, total_steps, label)` for progress updates.
    show_progress : bool, default False

    Returns
    -------
    shared_modes : list of dict
    all_delay_modes : list of list of dict (per delay)
    fit_cube : ndarray, shape (Npump, Nprobe, Nt2)
    residual_cube : ndarray, shape (Npump, Nprobe, Nt2)
    """
    from scipy.optimize import least_squares
    from .progress import ProgressTracker

    pump = np.asarray(pump)
    probe = np.asarray(probe)
    cube_3d = np.asarray(cube_3d)
    Nt2 = cube_3d.shape[2]

    # Pre-calculate per-delay scaling factors
    scales = np.ones(Nt2, dtype=float)
    cube_3d_fit = np.zeros_like(cube_3d)
    for i_t2 in range(Nt2):
        if normalize_t2:
            m_max = float(np.max(np.abs(cube_3d[:, :, i_t2])))
            scales[i_t2] = m_max if m_max > 0 else 1.0
        cube_3d_fit[:, :, i_t2] = cube_3d[:, :, i_t2] / scales[i_t2]

    # Shared per mode: [w1, w3, anharm, sigma_w1, sigma_w3, (rho)] -> 5 or 6 params
    # Independent per mode per delay: [amp_gsb, amp_esa] -> 2 * Nt2 params
    p0 = []
    bounds_lower = []
    bounds_upper = []

    w1_min, w1_max = float(np.min(pump)), float(np.max(pump))
    w3_min, w3_max = float(np.min(probe)), float(np.max(probe))
    w1_span = w1_max - w1_min
    w3_span = w3_max - w3_min
    max_amp = (float(np.max(np.abs(cube_3d_fit))) * 5.0 or 10.0)

    cfg = bounds_config or {}

    n_shared_per_mode = 6 if correlated else 5
    n_amp_per_mode = 2 * Nt2

    for m in initial_modes:
        w1 = float(m.get("w1", (w1_min + w1_max) / 2))
        w3 = float(m.get("w3", (w3_min + w3_max) / 2))
        anharm = float(m.get("anharm", 15.0))
        sigma_w1 = float(m.get("sigma_w1", 10.0))
        sigma_w3 = float(m.get("sigma_w3", 10.0))
        rho = float(m.get("rho", 0.0))

        anharm_min = float(m.get("anharm_min", cfg.get("anharm_min", 0.1)))
        anharm_max = float(m.get("anharm_max", cfg.get("anharm_max", w3_span)))
        sigma_min = float(m.get("sigma_min", cfg.get("sigma_min", 0.5)))
        sigma_max = float(m.get("sigma_max", cfg.get("sigma_max", max(w1_span, w3_span))))

        anharm = np.clip(anharm, anharm_min, anharm_max)
        sigma_w1 = np.clip(sigma_w1, sigma_min, sigma_max)
        sigma_w3 = np.clip(sigma_w3, sigma_min, sigma_max)

        p0.extend([w1, w3, anharm, sigma_w1, sigma_w3])
        bounds_lower.extend([w1_min - 0.2 * w1_span, w3_min - 0.2 * w3_span, anharm_min, sigma_min, sigma_min])
        bounds_upper.extend([w1_max + 0.2 * w1_span, w3_max + 0.2 * w3_span, anharm_max, sigma_max, sigma_max])

        if correlated:
            p0.append(rho)
            bounds_lower.append(-0.95)
            bounds_upper.append(0.95)

        # Amplitudes per t2 (scaled)
        amp_g_init = float(m.get("amp_gsb", -1.0))
        amp_e_init = float(m.get("amp_esa", 0.8))
        for i_t2 in range(Nt2):
            p0.extend([amp_g_init / scales[i_t2], amp_e_init / scales[i_t2]])
            bounds_lower.extend([-max_amp, -max_amp])
            bounds_upper.extend([max_amp, max_amp])

    n_mode_total_params = n_shared_per_mode + n_amp_per_mode
    max_nfev = 1500
    eval_counter = 0

    tracker = None
    if show_progress and progress_callback is None:
        tracker = ProgressTracker(total=100, title="Global 2D Gaussian Fit", label=f"Fitting 3D cube across {Nt2} delays...")

    def unpack_modes_for_t2(p, i_t2):
        modes = []
        for i_m in range(len(initial_modes)):
            m_offset = i_m * n_mode_total_params
            w1 = p[m_offset]
            w3 = p[m_offset + 1]
            anharm = p[m_offset + 2]
            sigma_w1 = p[m_offset + 3]
            sigma_w3 = p[m_offset + 4]

            if correlated:
                rho = p[m_offset + 5]
                amp_offset = m_offset + 6 + 2 * i_t2
            else:
                rho = 0.0
                amp_offset = m_offset + 5 + 2 * i_t2

            amp_gsb = p[amp_offset]
            amp_esa = p[amp_offset + 1]

            modes.append({
                "w1": w1,
                "w3": w3,
                "anharm": anharm,
                "amp_gsb": amp_gsb,
                "amp_esa": amp_esa,
                "sigma_w1": sigma_w1,
                "sigma_w3": sigma_w3,
                "rho": rho,
            })
        return modes

    def residuals(p):
        nonlocal eval_counter
        eval_counter += 1
        pct = min(99, int(100 * eval_counter / 400))
        lbl = f"Global fitting across {Nt2} delays... (eval {eval_counter})"
        if progress_callback is not None:
            progress_callback(pct, 100, lbl)
        elif tracker is not None and eval_counter % 3 == 0:
            tracker.update(pct, lbl)

        res_cube = np.zeros_like(cube_3d_fit)
        for i_t2 in range(Nt2):
            modes_t2 = unpack_modes_for_t2(p, i_t2)
            sim_t2 = evaluate_2d_gaussian_map(pump, probe, modes_t2, correlated=correlated)
            res_cube[:, :, i_t2] = sim_t2 - cube_3d_fit[:, :, i_t2]
        return res_cube.ravel()

    try:
        res = least_squares(residuals, p0, bounds=(bounds_lower, bounds_upper), max_nfev=max_nfev)
        if tracker is not None:
            tracker.update(100, "Global fitting complete.")
    finally:
        if tracker is not None:
            tracker.close()

    p_opt = res.x
    all_delay_modes = [[] for _ in range(Nt2)]
    fit_cube = np.zeros_like(cube_3d)
    residual_cube = np.zeros_like(cube_3d)

    for i_t2 in range(Nt2):
        modes_t2 = unpack_modes_for_t2(p_opt, i_t2)
        all_delay_modes[i_t2] = modes_t2
        sim_t2 = evaluate_2d_gaussian_map(pump, probe, modes_t2, correlated=correlated)
        fit_cube[:, :, i_t2] = sim_t2
        residual_cube[:, :, i_t2] = cube_3d[:, :, i_t2] - sim_t2

    # Construct shared modes summary
    shared_modes = []
    for i_m in range(len(initial_modes)):
        m_offset = i_m * n_mode_total_params
        shared_dict = {
            "w1": float(p_opt[m_offset]),
            "w3": float(p_opt[m_offset + 1]),
            "anharm": float(p_opt[m_offset + 2]),
            "sigma_w1": float(p_opt[m_offset + 3]),
            "sigma_w3": float(p_opt[m_offset + 4]),
        }
        if correlated:
            shared_dict["rho"] = float(p_opt[m_offset + 5])
        shared_modes.append(shared_dict)

    return shared_modes, all_delay_modes, fit_cube, residual_cube

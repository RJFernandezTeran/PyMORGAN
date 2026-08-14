"""Processing stage of the 1-D pipeline.

Basic operations (background subtraction) live here; the module is the home for
future advanced processing (chirp/dispersion correction, coherent-artefact
removal, SVD, global-analysis pre-processing).
"""

from __future__ import annotations

import numpy as np


def background_correct(Zavg_R, Zss_R, nscans, delays, tmin, tmax, do_correct: bool = True):
    """Subtract the pre-zero background averaged over ``[tmin, tmax]``.

    Returns ``(Zavg_C, Zss_C, bkg_avg, bkg_ss)``. The background spectra
    (``bkg_avg``/``bkg_ss``) are always computed over ``[tmin, tmax]`` and stored,
    so they remain available for inspection/plotting regardless of ``do_correct``.
    With ``do_correct=False`` the raw signal arrays are passed through unchanged
    (the background is computed but not subtracted).
    """
    import warnings

    in_window = (delays >= tmin) & (delays <= tmax)
    if np.any(in_window):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            bkg_avg = np.nanmean(Zavg_R[in_window, :], axis=0)
        bkg_avg = np.nan_to_num(bkg_avg, nan=0.0)
    else:
        bkg_avg = np.zeros_like(Zavg_R[0, :])

    # Single-scan background is computed only when ``Zss_R`` actually carries
    # per-scan arrays ([Ndelays x Npixels x Ndetectors x Nscans] with finite
    # values). A NaN placeholder (e.g. averaged-only data, or single scans not
    # loaded) is treated as "no single scans" -- the reported ``nscans`` may
    # still be finite for display, so the scan count is taken from the array
    # itself rather than from ``nscans``.
    Zss = np.asarray(Zss_R, dtype=float) if Zss_R is not None else None
    has_ss = Zss is not None and Zss.ndim == 4 and Zss.shape[3] >= 1 and np.any(np.isfinite(Zss))

    if has_ss:
        nss = Zss.shape[3]
        bkg_ss = np.zeros((Zss.shape[1], Zss.shape[2], nss))
        if np.any(in_window):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for s in range(nss):
                    bkg_ss[:, :, s] = np.nan_to_num(
                        np.nanmean(Zss[in_window, :, :, s], axis=0), nan=0.0
                    )
        else:
            bkg_ss[:] = 0.0
    else:
        bkg_ss = np.zeros_like(np.asarray(Zavg_R[0, :, :], dtype=float))

    if do_correct:
        Zavg_C = Zavg_R - bkg_avg
        if has_ss:
            Zss_C = Zss.copy()
            for s in range(nss):
                Zss_C[:, :, :, s] = Zss[:, :, :, s] - bkg_ss[:, :, s]
        elif Zss_R is None:
            Zss_C = None
        else:
            Zss_C = np.full_like(np.asarray(Zss_R, dtype=float), np.nan)
    else:
        Zavg_C = Zavg_R
        Zss_C = Zss_R

    return Zavg_C, Zss_C, bkg_avg, bkg_ss


def subtract_solvent(Z_sample, delays_sample, Z_solvent, delays_solvent, dt=0.0, scale=1.0):
    """Subtract a scaled, time-shifted solvent background from a sample signal.

    The solvent signal is interpolated onto the sample's delay grid after
    shifting its time axis by ``dt`` (positive ``dt`` means the solvent t₀
    comes *later* than the sample t₀, i.e. the solvent delays are
    effectively shifted forward). Out-of-range delays are filled with the
    nearest boundary value (no extrapolation artefacts).

    Parameters
    ----------
    Z_sample : ndarray, shape [Ndelays × Npixels × Ndet]
        Sample signal (most-processed version available).
    delays_sample : ndarray, shape [Ndelays]
        Sample delay axis (ps).
    Z_solvent : ndarray, shape [Ndelays_s × Npixels × Ndet]
        Solvent signal (most-processed version available).
    delays_solvent : ndarray, shape [Ndelays_s]
        Solvent delay axis (ps).
    dt : float
        Global timing offset in ps.  The effective solvent delays used for
        interpolation are ``delays_solvent - dt`` (i.e. a positive ``dt``
        shifts the solvent signal towards earlier times on the sample grid).
    scale : float or array-like
        Global (or per-pixel) amplitude scaling factor applied to the
        interpolated solvent before subtraction.  A scalar applies the same
        factor to every pixel; a 1-D array of length ``Npixels`` applies a
        per-pixel factor.

    Returns
    -------
    ndarray : corrected signal, same shape as ``Z_sample``
    """
    from scipy.interpolate import interp1d

    Z_sample = np.asarray(Z_sample, dtype=float)
    Z_solvent = np.asarray(Z_solvent, dtype=float)
    delays_sample = np.asarray(delays_sample, dtype=float)
    delays_solvent = np.asarray(delays_solvent, dtype=float)

    Ndelays, Npixels, Ndet = Z_sample.shape

    # Build interpolated solvent array [Ndelays × Npixels × Ndet]
    Z_solvent_interp = np.empty_like(Z_sample)
    for pix in range(Npixels):
        dt_pix = dt[pix] if np.ndim(dt) > 0 else dt
        # Effective solvent time axis after applying the timing offset for this pixel.
        t_solvent = delays_solvent - dt_pix
        for det in range(Ndet):
            col = Z_solvent[:, pix, det]
            finite = np.isfinite(col) & np.isfinite(t_solvent)
            if not np.any(finite):
                Z_solvent_interp[:, pix, det] = 0.0
                continue
            t_f = t_solvent[finite]
            c_f = col[finite]
            if t_f.size == 1:
                Z_solvent_interp[:, pix, det] = c_f[0]
            else:
                f = interp1d(
                    t_f,
                    c_f,
                    kind="linear",
                    bounds_error=False,
                    fill_value=(c_f[0], c_f[-1]),
                )
                Z_solvent_interp[:, pix, det] = f(delays_sample)

    # Apply scaling factor (scalar or per-pixel array broadcast over delays/dets)
    scale_arr = np.broadcast_to(np.asarray(scale, dtype=float), (Npixels,))
    # Reshape for broadcasting: [1 × Npixels × 1]
    scale_arr = scale_arr[np.newaxis, :, np.newaxis]

    return Z_sample - scale_arr * Z_solvent_interp


def fit_solvent_auto(
    Z_sample,
    delays_sample,
    Z_solvent,
    delays_solvent,
    det_idx=0,
    per_pixel_dt=False,
    per_pixel_scale=True,
):
    """Automatically fit solvent time offsets and scaling factor.

    Optimises a global (or per-pixel) timing offset ``dt`` and a global (or per-pixel)
    amplitude scaling factors ``scale`` such that the corrected signal
    ``Z_sample - scale * Z_solvent(t - dt)`` is as close as possible to an
    erf-broadened step-like rise.

    Parameters
    ----------
    Z_sample : ndarray, shape [Ndelays × Npixels × Ndet]
        Sample signal.
    delays_sample : ndarray, shape [Ndelays]
        Sample delay axis (ps).
    Z_solvent : ndarray, shape [Ndelays_s × Npixels × Ndet]
        Solvent signal.
    delays_solvent : ndarray, shape [Ndelays_s]
        Solvent delay axis (ps).
    det_idx : int, default 0
        The detector channel to use for the fitting optimisation.
    per_pixel_dt : bool, default False
        If True, optimizes independent per-pixel timing offsets instead of a single
        global timing offset.
    per_pixel_scale : bool, default True
        If True, optimizes independent per-pixel amplitude scaling factors instead
        of a single global scaling factor.

    Returns
    -------
    scale : float or ndarray, shape [Npixels]
        Optimal global scaling factor (float) or per-pixel scaling factors (ndarray).
    dt : float or ndarray, shape [Npixels]
        Optimal global timing offset (float) or per-pixel timing offsets (ndarray).
    """
    from scipy.interpolate import interp1d
    from scipy.optimize import least_squares, minimize_scalar
    from scipy.signal import savgol_filter
    from scipy.special import erf

    Z_sample = np.asarray(Z_sample, dtype=float)
    Z_solvent = np.asarray(Z_solvent, dtype=float)
    delays_sample = np.asarray(delays_sample, dtype=float)
    delays_solvent = np.asarray(delays_solvent, dtype=float)

    # 1. Filter delays to early-time window around time-zero where the artefact lives
    fit_mask = (delays_sample >= -3.0) & (delays_sample <= 7.0)
    if np.count_nonzero(fit_mask) < 5:
        # fallback to all delays if the range is too narrow
        fit_mask = np.ones_like(delays_sample, dtype=bool)

    ts = delays_sample[fit_mask]
    S = Z_sample[fit_mask, :, det_idx]  # [Nt x Npixels]
    Npixels = S.shape[1]

    def make_fallback_itp(val):
        # Avoids E731 lambda assignment warning
        def fallback_itp(t):
            return np.full_like(t, val if np.isfinite(val) else 0.0)

        return fallback_itp

    # 2. Pre-build interpolator functions for the solvent signal per pixel
    sol_interpolators = []
    for i in range(Npixels):
        sol_y = Z_solvent[:, i, det_idx]
        finite = np.isfinite(sol_y) & np.isfinite(delays_solvent)
        if np.count_nonzero(finite) >= 2:
            itp = interp1d(
                delays_solvent[finite],
                sol_y[finite],
                kind="linear",
                bounds_error=False,
                fill_value=(sol_y[finite][0], sol_y[finite][-1]),
            )
        else:
            val = np.nanmean(sol_y)
            itp = make_fallback_itp(val)
        sol_interpolators.append(itp)

    # 3. Pass 1: Optimise global scale A and global timing offset dt_global
    # using a subset of top 5 highest-variance pixels to avoid slow O(N^2) finite-diff evaluations.
    pixel_vars = np.var(S, axis=0)
    K = min(5, Npixels)
    global_pixels = np.argsort(pixel_vars)[-K:]

    sigma = 0.1  # Fixed reasonable IRF width (ps)

    def global_residuals(params):
        A = params[0]
        dt_g = params[1]
        t0s = params[2:]

        res = []
        for idx, pix in enumerate(global_pixels):
            t0 = t0s[idx]
            sol_shifted = sol_interpolators[pix](ts - dt_g)
            y_corr = S[:, pix] - A * sol_shifted

            valid_mask = np.isfinite(y_corr) & np.isfinite(ts)
            if np.count_nonzero(valid_mask) < 5:
                res.append(np.zeros_like(ts))
                continue

            ts_v = ts[valid_mask]
            y_corr_v = y_corr[valid_mask]

            u = (ts_v - t0) / (np.sqrt(2.0) * sigma)
            col1 = np.ones_like(ts_v)
            col2 = 1.0 + erf(u)
            X = np.column_stack([col1, col2])

            try:
                c = np.linalg.lstsq(X, y_corr_v, rcond=None)[0]
                pred = X @ c
                pix_res = np.zeros_like(ts)
                pix_res[valid_mask] = y_corr_v - pred
                res.append(pix_res)
            except Exception:
                res.append(np.nan_to_num(y_corr))
        return np.concatenate(res)

    initial_global = np.zeros(2 + K)
    initial_global[0] = 1.0  # A
    initial_global[1] = 0.0  # dt_global

    lower_global = np.concatenate([[0.0, -2.0], np.full(K, -2.0)])
    upper_global = np.concatenate([[10.0, 2.0], np.full(K, 2.0)])

    res_global = least_squares(
        global_residuals,
        initial_global,
        bounds=(lower_global, upper_global),
        ftol=1e-3,
        xtol=1e-3,
        max_nfev=50,
    )

    A_opt = float(res_global.x[0])
    dt_global = float(res_global.x[1])

    # 4. Pass 2: Fit per-pixel scale factors A_raw[i] and independent t0[i], and optionally per-pixel dt[i].
    A_raw = np.zeros(Npixels)

    if per_pixel_dt:
        dts_opt = np.zeros(Npixels)

        def make_pixel_residuals(sol_itp, S_i):
            def pixel_residuals(params):
                dt = params[0]
                t0 = params[1]

                sol_shifted = sol_itp(ts - dt)
                if per_pixel_scale:
                    valid_mask = np.isfinite(S_i) & np.isfinite(sol_shifted)
                    if np.count_nonzero(valid_mask) < 5:
                        return np.zeros_like(ts)
                    ts_v = ts[valid_mask]
                    S_i_v = S_i[valid_mask]
                    sol_shifted_v = sol_shifted[valid_mask]
                    u = (ts_v - t0) / (np.sqrt(2.0) * sigma)
                    col1 = sol_shifted_v
                    col2 = np.ones_like(ts_v)
                    col3 = 1.0 + erf(u)
                    X = np.column_stack([col1, col2, col3])
                    y_to_fit = S_i_v
                else:
                    y_corr = S_i - A_opt * sol_shifted
                    valid_mask = np.isfinite(y_corr) & np.isfinite(sol_shifted)
                    if np.count_nonzero(valid_mask) < 5:
                        return np.zeros_like(ts)
                    ts_v = ts[valid_mask]
                    y_corr_v = y_corr[valid_mask]
                    u = (ts_v - t0) / (np.sqrt(2.0) * sigma)
                    col1 = np.ones_like(ts_v)
                    col2 = 1.0 + erf(u)
                    X = np.column_stack([col1, col2])
                    y_to_fit = y_corr_v

                try:
                    c = np.linalg.lstsq(X, y_to_fit, rcond=None)[0]
                    pred = X @ c
                    pix_res = np.zeros_like(ts)
                    pix_res[valid_mask] = y_to_fit - pred
                    return pix_res
                except Exception:
                    return np.nan_to_num(y_to_fit)

            return pixel_residuals

        for i in range(Npixels):
            sol_itp = sol_interpolators[i]
            S_i = S[:, i]

            valid = np.isfinite(S_i)
            if np.count_nonzero(valid) < 5:
                A_raw[i] = A_opt
                dts_opt[i] = dt_global
                continue

            pixel_residuals = make_pixel_residuals(sol_itp, S_i)
            initial_pix = np.array([dt_global, 0.0])
            try:
                res_pix = least_squares(
                    pixel_residuals,
                    initial_pix,
                    bounds=([-2.0, -2.0], [2.0, 2.0]),
                    ftol=1e-2,
                    xtol=1e-2,
                    max_nfev=15,
                )
                dt_i = res_pix.x[0]
                t0_i = res_pix.x[1]

                # Resolve for final A_raw[i]
                sol_shifted = sol_itp(ts - dt_i)
                valid_mask = np.isfinite(S_i) & np.isfinite(sol_shifted)
                ts_v = ts[valid_mask]
                S_i_v = S_i[valid_mask]
                sol_shifted_v = sol_shifted[valid_mask]

                u = (ts_v - t0_i) / (np.sqrt(2.0) * sigma)
                if per_pixel_scale:
                    col1 = sol_shifted_v
                    col2 = np.ones_like(ts_v)
                    col3 = 1.0 + erf(u)
                    X = np.column_stack([col1, col2, col3])
                    c = np.linalg.lstsq(X, S_i_v, rcond=None)[0]
                    A_raw[i] = float(c[0])
                else:
                    A_raw[i] = A_opt
                dts_opt[i] = float(dt_i)
            except Exception:
                A_raw[i] = A_opt
                dts_opt[i] = dt_global

        dt_out = dts_opt
    else:
        if not per_pixel_scale:
            # Both are global, skip per-pixel Pass 2 and return immediately!
            return A_opt, dt_global

        # Pre-calculate shifted solvent for this global dt
        sol_shifted_all = []
        for i in range(Npixels):
            sol_shifted_all.append(sol_interpolators[i](ts - dt_global))

        for i in range(Npixels):
            sol_shifted = sol_shifted_all[i]
            S_i = S[:, i]

            valid = np.isfinite(S_i) & np.isfinite(sol_shifted)
            if np.count_nonzero(valid) < 5:
                # Fallback to global A_opt if not enough valid points
                A_raw[i] = A_opt
                continue

            ts_v = ts[valid]
            S_i_v = S_i[valid]
            sol_shifted_v = sol_shifted[valid]

            # Loop variables are bound as defaults: the closure is only used
            # within this iteration, but this makes that explicit.
            def loss_func(t0, ts_v=ts_v, S_i_v=S_i_v, sol_shifted_v=sol_shifted_v):
                u = (ts_v - t0) / (np.sqrt(2.0) * sigma)
                col1 = sol_shifted_v
                col2 = np.ones_like(ts_v)
                col3 = 1.0 + erf(u)
                X = np.column_stack([col1, col2, col3])
                try:
                    c, residuals, rank, s = np.linalg.lstsq(X, S_i_v, rcond=None)
                    if residuals.size > 0:
                        return float(residuals[0])
                    else:
                        return float(np.sum((S_i_v - X @ c) ** 2))
                except Exception:
                    return 1e9

            try:
                res_pix = minimize_scalar(loss_func, bounds=(-2.0, 2.0), method="bounded")
                best_t0 = res_pix.x

                # Resolve for final parameters at best_t0
                u = (ts_v - best_t0) / (np.sqrt(2.0) * sigma)
                col1 = sol_shifted_v
                col2 = np.ones_like(ts_v)
                col3 = 1.0 + erf(u)
                X = np.column_stack([col1, col2, col3])
                c = np.linalg.lstsq(X, S_i_v, rcond=None)[0]
                A_raw[i] = float(c[0])
            except Exception:
                A_raw[i] = A_opt

        dt_out = dt_global

    # Clip raw coefficients to a physically meaningful range [0.0, 2.0]
    # (can be below 1.0, e.g. for ground-state bleach regions)
    A_raw = np.clip(A_raw, 0.0, 2.0)

    # 5. Smooth the per-pixel scaling factors if per_pixel_scale is enabled
    if per_pixel_scale:
        if Npixels >= 5:
            window_length = 15
            if Npixels < window_length:
                window_length = Npixels if Npixels % 2 == 1 else Npixels - 1
            polyorder = min(2, window_length - 1)
            scale_opt = savgol_filter(
                A_raw, window_length=window_length, polyorder=polyorder, mode="nearest"
            )
        else:
            scale_opt = A_raw
        # Re-clip smoothed factors to make sure no overshoot artefacts violate constraints
        scale_opt = np.clip(scale_opt, 0.0, 2.0)
    else:
        scale_opt = A_opt

    return scale_opt, dt_out


def subtract_shockwave(Z, pixels, one_based: bool = True):
    """Subtract an average kinetic trace calculated over ``pixels`` from all pixels.

    Parameters
    ----------
    Z : ndarray, shape [Ndelays × Npixels × Ndet] or [Ndelays × Npixels × Ndet × Nscans]
        Signal matrix to be corrected.
    pixels : int, sequence of int, slice, or 1D boolean ndarray
        Pixel index or indices over which to compute the average shockwave trace.
    one_based : bool, default True
        Whether ``pixels`` are specified as 1-based pixel indices (1..Npixels).
        Set to False if passing 0-based array indices (0..Npixels-1).

    Returns
    -------
    Z_corrected : ndarray, same shape as ``Z``
        Signal matrix after subtracting the averaged kinetic trace.
    shockwave_trace : ndarray
        The extracted average kinetic trace that was subtracted, with shape
        ``[Ndelays × Ndet]`` (for 3D ``Z``) or ``[Ndelays × Ndet × Nscans]`` (for 4D ``Z``).
    """
    if Z is None:
        return None, None

    Z_arr = np.asarray(Z, dtype=float)
    Npixels = Z_arr.shape[1]

    # Normalise pixels to array of 0-based array indices
    if isinstance(pixels, slice):
        step = pixels.step or 1
        if one_based:
            start = (pixels.start - 1) if pixels.start is not None else 0
            stop = pixels.stop if pixels.stop is not None else Npixels
        else:
            start = pixels.start if pixels.start is not None else 0
            stop = pixels.stop if pixels.stop is not None else Npixels
        pix_idx = np.arange(start, stop, step)
    else:
        pix_idx = np.asarray(pixels)
        if pix_idx.dtype == bool:
            pix_idx = np.where(pix_idx)[0]
        else:
            pix_idx = np.atleast_1d(pix_idx)
            if one_based:
                pix_idx = pix_idx - 1

    if pix_idx.size == 0:
        raise ValueError("No valid pixels specified for shockwave subtraction.")

    if np.any(pix_idx < 0) or np.any(pix_idx >= Npixels):
        err_pix = pix_idx + 1 if one_based else pix_idx
        raise ValueError(f"Pixel index out of bounds for dataset with {Npixels} pixels: {err_pix}")

    if Z_arr.ndim == 2:  # [Ndelays x Npixels]
        shockwave_trace = np.nanmean(Z_arr[:, pix_idx], axis=1)  # [Ndelays]
        Z_corrected = Z_arr - shockwave_trace[:, np.newaxis]
    elif Z_arr.ndim == 3:  # [Ndelays x Npixels x Ndet]
        shockwave_trace = np.nanmean(Z_arr[:, pix_idx, :], axis=1)  # [Ndelays x Ndet]
        Z_corrected = Z_arr - shockwave_trace[:, np.newaxis, :]
    elif Z_arr.ndim == 4:  # [Ndelays x Npixels x Ndet x Nscans]
        shockwave_trace = np.nanmean(Z_arr[:, pix_idx, :, :], axis=1)  # [Ndelays x Ndet x Nscans]
        Z_corrected = Z_arr - shockwave_trace[:, np.newaxis, :, :]
    else:
        raise ValueError(f"Unsupported Z dimension {Z_arr.ndim} for shockwave subtraction.")

    return Z_corrected, shockwave_trace

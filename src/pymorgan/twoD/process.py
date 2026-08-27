"""Processing stage of the 2-D pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..log import get_logger
from .progress import ProgressTracker

logger = get_logger(__name__)


def background_correct(Z, reference_index, do_correct: bool = True):
    """Subtract a reference t2 map from every map in the cube.

    ``reference_index`` selects the t2 map used as background; it is subtracted
    from all other maps (the reference map itself is left unchanged, so it can
    still be inspected). With ``do_correct=False`` or ``reference_index=None``
    the cube is returned unchanged (passthrough).
    """
    Z = np.asarray(Z)
    if not do_correct or reference_index is None:
        return Z.copy()

    n = Z.shape[2]
    ref = int(reference_index) % n
    bkg = Z[:, :, ref]
    Zc = Z.copy()
    for i in range(n):
        if i != ref:
            Zc[:, :, i] = Z[:, :, i] - bkg
    return Zc


def robust_polyfit(x, y, deg, max_iter=50, tol=1e-6, random_state=42):
    """Robust polynomial fit using MSAC consensus sampling followed by Tukey's bisquare IRLS.

    Parameters
    ----------
    x, y : array_like
        1-D coordinate and response arrays.
    deg : int
        Polynomial degree (e.g. 0 for constant, 1 for linear, 2 for quadratic).
    max_iter : int, optional
        Maximum number of IRLS refinement iterations (default 50).
    tol : float, optional
        Convergence tolerance for weight changes in IRLS (default 1e-6).
    random_state : int or None, optional
        Seed for the random consensus sampler to ensure deterministic reproducibility.

    Returns
    -------
    coeffs : np.ndarray
        Polynomial coefficients in descending order of powers (highest degree first),
        matching ``np.polyfit`` conventions.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    n = len(x)

    if deg < 0:
        raise ValueError("deg must be non-negative")
    if n <= deg + 1:
        return np.polyfit(x, y, deg)

    if deg == 0:
        return np.array([np.median(y)])

    data_range = float(np.ptp(y))
    if data_range < 1e-12:
        return np.polyfit(x, y, deg)

    # 1. MSAC / RANSAC robust consensus sampling
    rng = np.random.default_rng(random_state)
    n_samples = deg + 1
    n_trials = max(300, min(2000, 200 * (deg + 1)))

    sample_indices = np.empty((n_trials, n_samples), dtype=int)
    for i in range(n_trials):
        sample_indices[i] = rng.choice(n, size=n_samples, replace=False)

    xs = x[sample_indices]
    ys = y[sample_indices]

    if deg == 1:
        dx = xs[:, 1] - xs[:, 0]
        valid_slopes = np.abs(dx) > 1e-10
        xs = xs[valid_slopes]
        ys = ys[valid_slopes]
        dx = dx[valid_slopes]
        m = (ys[:, 1] - ys[:, 0]) / dx
        c = ys[:, 0] - m * xs[:, 0]
        candidate_coeffs = np.column_stack((m, c))
    elif deg == 2:
        x0, x1, x2 = xs[:, 0], xs[:, 1], xs[:, 2]
        y0, y1, y2 = ys[:, 0], ys[:, 1], ys[:, 2]
        denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
        valid_denom = np.abs(denom) > 1e-10
        x0, x1, x2 = x0[valid_denom], x1[valid_denom], x2[valid_denom]
        y0, y1, y2 = y0[valid_denom], y1[valid_denom], y2[valid_denom]
        denom = denom[valid_denom]

        a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
        b = (x2**2 * (y0 - y1) + x1**2 * (y2 - y0) + x0**2 * (y1 - y2)) / denom
        c = (x1 * x2 * (x1 - x2) * y0 + x2 * x0 * (x2 - x0) * y1 + x0 * x1 * (x0 - x1) * y2) / denom
        candidate_coeffs = np.column_stack((a, b, c))
    else:
        candidate_list = []
        for i in range(n_trials):
            try:
                candidate_list.append(np.polyfit(xs[i], ys[i], deg))
            except (np.linalg.LinAlgError, ValueError):
                continue
        candidate_coeffs = np.array(candidate_list) if len(candidate_list) > 0 else np.empty((0, deg + 1))

    if len(candidate_coeffs) == 0:
        return np.polyfit(x, y, deg)

    X = np.vander(x, deg + 1)
    preds = candidate_coeffs @ X.T
    residuals = np.abs(preds - y[np.newaxis, :])

    med_res = np.median(residuals, axis=1)
    scales = 1.4826 * med_res

    default_threshold = max(1.0, data_range * 0.05)
    thresh = np.clip(3.0 * scales, 1.0, default_threshold)

    losses = np.sum(np.minimum(residuals**2, thresh[:, np.newaxis]**2), axis=1)

    best_idx = int(np.argmin(losses))
    best_coeffs = candidate_coeffs[best_idx]
    best_inliers = residuals[best_idx] < thresh[best_idx]

    # 2. Refinement on inliers using IRLS with Tukey's bisquare weights
    x_in = x[best_inliers]
    y_in = y[best_inliers]

    if len(x_in) > deg + 1:
        X_in = np.vander(x_in, deg + 1)
        coeffs = best_coeffs.copy()
        w = np.ones_like(y_in)

        for _ in range(max_iter):
            sw = np.sqrt(w)
            try:
                new_coeffs, _, _, _ = np.linalg.lstsq(X_in * sw[:, np.newaxis], y_in * sw, rcond=None)
            except np.linalg.LinAlgError:
                break

            resid = y_in - X_in @ new_coeffs
            mad = np.median(np.abs(resid - np.median(resid)))
            s = 1.4826 * mad
            if not np.isfinite(s) or s < 1e-12:
                break
            u = resid / (4.685 * s)
            new_w = np.where(np.abs(u) < 1.0, (1.0 - u**2) ** 2, 0.0)
            if np.sum(new_w > 0) <= deg + 1:
                break
            if np.all(np.abs(new_w - w) < tol):
                coeffs = new_coeffs
                break
            coeffs = new_coeffs
            w = new_w
        return coeffs
    return best_coeffs


def process(
    dataset,
    *,
    apodise_method: str = "0",
    zeropad_enable: bool = True,
    zeropad_factor: int = 1,
    zeropad_next2k: bool = False,
    phase_method: str = "No fit",
    phase_points: int = 10,
    pumpcorrection: bool = False,
    bkg_sub: bool = False,
    bkgIdx: int = 0,
    w0: float | None = None,
    dt1: float | None = None,
    autocalibrate_probe: bool = False,
    cal_probe_vector: np.ndarray | None = None,
    progress_tracker=None,
):
    """Process raw time-domain signal and interferogram data to frequency-domain.

    Performs baseline correction, apodization, zero-padding, FFT, and phasing.
    Updates dataset.Z_R and dataset.pump.
    """
    from scipy.signal import medfilt

    if dataset.raw_signal is None or dataset.raw_interferogram is None or dataset.raw_t1delays is None:
        return

    close_tracker = False
    if progress_tracker is None:
        progress_tracker = ProgressTracker(1, title="Processing 2D Dataset", label="Initializing...")
        close_tracker = True

    raw_signal = np.asarray(dataset.raw_signal, dtype=float)
    raw_interferogram = np.asarray(dataset.raw_interferogram, dtype=float)
    raw_t1delays = np.asarray(dataset.raw_t1delays, dtype=float)

    Nbins, Nprobe, Ndelays = raw_signal.shape

    total_steps = Ndelays * 2
    progress_tracker.set_total(total_steps)

    # 1. Baseline Correction / Preprocessing
    proc_interf = np.zeros((Nbins, Ndelays))
    proc_signal = np.zeros((Nbins, Nprobe, Ndelays))

    for m in range(Ndelays):
        step = m + 1
        if hasattr(progress_tracker, "update_split"):
            progress_tracker.update_split(step, total_steps, "processing")
        else:
            progress_tracker.update(step, label=f"Processing ({step} of {total_steps})")
        interf = raw_interferogram[:, m].copy()
        sig = raw_signal[:, :, m].copy()

        if dataset.datatype == "interferometer":
            # Median filter
            interf = -(interf - np.mean(interf))
            k_size = 101  # filter_points = 10 -> odd 11
            interf = interf - medfilt(interf, kernel_size=k_size)

            sig = sig - np.mean(sig, axis=0)
            for w in range(Nprobe):
                sig[:, w] = sig[:, w] - medfilt(sig[:, w], kernel_size=k_size)
        else: # shaper
            # Mean filter
            sig = sig - np.mean(sig, axis=0)

        proc_interf[:, m] = interf
        proc_signal[:, :, m] = sig

    # 2. Determine rotating frame parameters w0 and dt1 for shaper
    if dataset.datatype == "shaper":
        if w0 is None or dt1 is None:
            w0_csv = None
            if dataset.source:
                folder_path = Path(dataset.source)
                for path in [folder_path / "w0.csv", folder_path.parent / "w0.csv"]:
                    if path.exists():
                        try:
                            vals = np.loadtxt(path)
                            if vals.size >= 2:
                                w0_csv = (float(vals[0]), float(vals[1]))
                                break
                        except Exception:
                            logger.debug(
                                "Could not read the rotating-frame frequency from %s.",
                                path, exc_info=True
                            )
            if w0_csv is not None:
                w0_val, dt1_val = w0_csv
            else:
                w0_val = 0.0
                dt1_val = np.nanmean(np.abs(np.diff(raw_t1delays))) if len(raw_t1delays) > 1 else 1.0

            w0 = w0_val if w0 is None else w0
            dt1 = dt1_val if dt1 is None else dt1
    else:
        w0 = 0.0
        dt1 = 2.11079 # HeNe period in fs

    # 3. Determine FFT points
    if not zeropad_enable:
        n_ft = Nbins
    else:
        n_ft = zeropad_factor * Nbins
        if zeropad_next2k:
            n_ft = 2 ** int(np.ceil(np.log2(n_ft))) if n_ft > 0 else Nbins

    # Store intermediate datasets for TD/PH subplots
    dataset.proc_absFFT_ZPint = np.zeros((n_ft, Ndelays))
    dataset.proc_ZP_phase = np.zeros((n_ft, Ndelays))
    dataset.proc_fittedPhase = np.zeros((n_ft, Ndelays))
    dataset.proc_apod_func = np.zeros((Nbins, Ndelays))
    dataset.proc_binspecmax = np.zeros(Ndelays, dtype=int)
    dataset.proc_phase_points = phase_points
    dataset.proc_phase_coeffs = [None] * Ndelays  # fitted polynomial coefficients per delay
    c_0 = 2.99792458e-5 # cm/fs
    res_m = 1.0 / (n_ft * dt1 * c_0)
    dataset.proc_pump_full = np.arange(n_ft) * res_m + w0

    # 4. Apodization, Zero-Padding, FFT, and Phasing
    phased_FFTZPsig = np.zeros((n_ft, Nprobe, Ndelays))
    phased_FFTZPint = np.zeros((n_ft, Ndelays), dtype=complex)

    for m in range(Ndelays):
        step = Ndelays + m + 1
        if hasattr(progress_tracker, "update_split"):
            progress_tracker.update_split(step, total_steps, "processing")
        else:
            progress_tracker.update(step, label=f"Processing ({step} of {total_steps})")
        interf = proc_interf[:, m]
        sig = proc_signal[:, :, m]

        # Find binzero
        if dataset.datatype == "interferometer":
            bininterfmax = np.argmax(interf)
            fft_interf = np.fft.fft(interf)
            binspecmax = np.argmax(np.abs(fft_interf[19:Nbins//2])) + 19
            dataset.proc_binspecmax[m] = binspecmax

            bins_arr = np.arange(1, 201) + bininterfmax - 100
            diff_arr = np.zeros(200)
            for idx_p, b_val in enumerate(bins_arr):
                shifted = np.roll(interf, -b_val)
                temp_phase = np.unwrap(np.angle(np.fft.fft(shifted)))
                diff_arr[idx_p] = temp_phase[binspecmax + 10] - temp_phase[binspecmax - 10]

            coeff = robust_polyfit(bins_arr, diff_arr, 1)
            binzero = int(np.round(-coeff[1] / coeff[0]))
        else:
            binzero = 0
            binspecmax = 0

        q = np.arange(1, Nbins + 1)
        cosine_sym = np.ones(Nbins)
        cosine_onesided = np.ones(Nbins)

        V = q - (binzero + 1)
        box = np.zeros(Nbins)
        box[V > 0] = 1.0
        box[V == 0] = 0.5

        M = Nbins - binzero - 1
        denom = float(M) if M > 0 else 1.0

        if apodise_method in ("Box", "Cos", "Cos^2", "Cos^3", "Hanning", "Hamming"):
            for idx_q, q_val in enumerate(q):
                v_val = abs(q_val - (binzero + 1))
                if v_val <= M:
                    if apodise_method == "Box":
                        val = 1.0
                    elif apodise_method == "Cos":
                        val = np.cos(np.pi * v_val / (2.0 * denom))
                    elif apodise_method == "Cos^2":
                        val = (np.cos(np.pi * v_val / (2.0 * denom))) ** 2
                    elif apodise_method == "Cos^3":
                        val = (np.cos(np.pi * v_val / (2.0 * denom))) ** 3
                    elif apodise_method == "Hanning":
                        val = 0.5 + 0.5 * np.cos(np.pi * v_val / denom)
                    elif apodise_method == "Hamming":
                        val = 0.54 + 0.46 * np.cos(np.pi * v_val / denom)
                else:
                    val = 0.0
                cosine_sym[idx_q] = val

            cosine_onesided = cosine_sym * box
        else:
            # If "None" or not specified, do not apply any window/Heaviside (keep as ones)
            box = np.ones(Nbins)
            cosine_onesided = np.ones(Nbins)
            cosine_sym = np.ones(Nbins)

        dataset.proc_apod_func[:, m] = cosine_onesided
        apo_interferogram = interf * cosine_sym
        apo_signal = sig * cosine_onesided[:, np.newaxis]

        if apodise_method == "Box":
            mean_interf = np.mean(interf[-10:])
            mean_sig = np.mean(sig[-10:, :], axis=0)
        else:
            mean_interf = 0.0
            mean_sig = np.zeros(Nprobe)

        n_pad = n_ft - Nbins
        if n_pad > 0:
            pad_interf = np.ones(n_pad) * mean_interf
            pad_sig = np.ones((n_pad, Nprobe)) * mean_sig
            zeropad_interf = np.concatenate([apo_interferogram, pad_interf])
            zeropad_sig = np.concatenate([apo_signal, pad_sig], axis=0)
        else:
            zeropad_interf = apo_interferogram
            zeropad_sig = apo_signal

        if dataset.datatype == "interferometer":
            phased_ZPint = np.roll(zeropad_interf, -binzero)
            phased_ZPsig = np.roll(zeropad_sig, -binzero, axis=0)

            absFFT_ZPint = np.abs(np.fft.fft(phased_ZPint))
            binspecmax = np.argmax(absFFT_ZPint[99:n_ft//2]) + 99
            dataset.proc_binspecmax[m] = binspecmax

            FFT_ZPint = np.fft.fft(phased_ZPint)
            FFT_ZPsig = np.fft.fft(phased_ZPsig, axis=0)

            ZP_phase = np.unwrap(np.angle(FFT_ZPint))
            center_freq = dataset.proc_pump_full[binspecmax]
            in_range_indices = np.where((dataset.proc_pump_full >= center_freq - phase_points) &
                                        (dataset.proc_pump_full <= center_freq + phase_points))[0]
            if len(in_range_indices) > 0:
                start_pt = in_range_indices[0]
                end_pt = in_range_indices[-1]
            else:
                start_pt = max(0, binspecmax - 10)
                end_pt = min(n_ft - 1, binspecmax + 10)
            points = np.arange(start_pt, end_pt + 1)

            if len(points) > 0:
                if phase_method == "Constant":
                    coeff = robust_polyfit(points, ZP_phase[points], 0)
                    fittedPhase = np.polyval(coeff, np.arange(n_ft))
                elif phase_method == "Linear":
                    coeff = robust_polyfit(points, ZP_phase[points], 1)
                    fittedPhase = np.polyval(coeff, np.arange(n_ft))
                elif phase_method == "Quadratic":
                    coeff = robust_polyfit(points, ZP_phase[points], 2)
                    fittedPhase = np.polyval(coeff, np.arange(n_ft))
                elif phase_method == "Cubic":
                    coeff = robust_polyfit(points, ZP_phase[points], 3)
                    fittedPhase = np.polyval(coeff, np.arange(n_ft))
                else:
                    coeff = None
                    fittedPhase = ZP_phase
                dataset.proc_phase_coeffs[m] = coeff
            else:
                fittedPhase = ZP_phase
                dataset.proc_phase_coeffs[m] = None

            phasingterm = np.exp(-1j * fittedPhase)
            phased_FFTZPint[:, m] = FFT_ZPint * phasingterm

            dataset.proc_absFFT_ZPint[:, m] = absFFT_ZPint
            dataset.proc_ZP_phase[:, m] = ZP_phase
            dataset.proc_fittedPhase[:, m] = fittedPhase

            if pumpcorrection:
                p_int = FFT_ZPint * phasingterm / np.max(np.abs(FFT_ZPint))
                denom = np.abs(p_int)[:, np.newaxis]
                denom[denom == 0] = 1.0
                phased_FFTZPsig[:, :, m] = np.real(FFT_ZPsig * phasingterm[:, np.newaxis]) / denom
            else:
                phased_FFTZPsig[:, :, m] = np.real(FFT_ZPsig * phasingterm[:, np.newaxis])
        else: # shaper
            FFT_ZPint = np.fft.fft(zeropad_interf)
            FFT_ZPsig = np.fft.fft(apo_signal, n=n_ft, axis=0)

            fittedPhase = np.zeros(n_ft)
            phasingterm = np.exp(-1j * fittedPhase)
            phased_FFTZPint[:, m] = FFT_ZPint * phasingterm

            dataset.proc_absFFT_ZPint[:, m] = np.abs(FFT_ZPint)
            dataset.proc_ZP_phase[:, m] = 0.0
            dataset.proc_fittedPhase[:, m] = 0.0

            if pumpcorrection:
                p_int = FFT_ZPint * phasingterm
                denom = np.abs(p_int)[:, np.newaxis]
                denom[denom == 0] = 1.0
                phased_FFTZPsig[:, :, m] = np.real(FFT_ZPsig * phasingterm[:, np.newaxis]) / denom
            else:
                phased_FFTZPsig[:, :, m] = np.real(FFT_ZPsig)

    # 5. Probe Axis Autocalibration (if requested)
    if autocalibrate_probe:
        if cal_probe_vector is not None and len(cal_probe_vector) == Nprobe:
            dataset.probe = np.asarray(cal_probe_vector, dtype=float)
            dataset.cal_level = "autocalibrated"
        elif dataset.datatype in ("shaper", "interferometer") and getattr(dataset, "proc_pump_full", None) is not None:
            # We locate search range fitrange = binspecmax +/- P
            if dataset.datatype == "interferometer":
                bin_spec = binspecmax
            else:
                # Find maximum of first delay spectrum of shaper
                bin_spec = np.argmax(np.mean(np.abs(phased_FFTZPsig[:, :, 0]), axis=1))

            P = max(1, n_ft // 50)
            fit_start = max(0, bin_spec - P)
            fit_end = min(n_ft - 1, bin_spec + P)

            # Find maximum index along pump axis for each probe pixel
            max_indices = np.argmax(np.abs(phased_FFTZPsig[fit_start:fit_end+1, :, 0]), axis=0) + fit_start
            scattering_maxima = dataset.proc_pump_full[max_indices]

            pixels = np.arange(1, Nprobe + 1)
            # Perform robust quadratic fit
            cal_coeff = robust_polyfit(pixels, scattering_maxima, 2)
            dataset.probe = np.polyval(cal_coeff, pixels)
            dataset.cal_level = "autocalibrated"

    # 6. Background Subtraction (Scattering)
    proc_2d = phased_FFTZPsig
    if bkg_sub:
        bkg_data = phased_FFTZPsig[:, :, bkgIdx]
        proc_2d = phased_FFTZPsig.copy()
        for m in range(Ndelays):
            if m != bkgIdx:
                proc_2d[:, :, m] = phased_FFTZPsig[:, :, m] - bkg_data

    # 7. Scaling
    scale_factor = np.sqrt(c_0 * dt1 / n_ft) * 2.0
    proc_2d = proc_2d * scale_factor

    # 8. Slicing to Positive Half
    n_pos = n_ft // 2
    dataset.Z_R = proc_2d[:n_pos, :, :]
    dataset.pump = dataset.proc_pump_full[:n_pos]

    dataset.Z_C = None
    dataset._is_corrected = False

    if close_tracker:
        progress_tracker.close()

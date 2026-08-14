import concurrent.futures

import numpy as np
from numpy import pi
from scipy.optimize import minimize

from ..log import get_logger

logger = get_logger(__name__)

c_0 = 2.99792458e-2  # Speed of light in cm/ps

def g_function(t, dw_0, dw_1, tau_c, T2):
    """Kubo lineshape function g(t) in ps^-1.
    dw_0, dw_1 are FWHM in cm^-1, converted to standard deviation in rad/ps.
    T2 is in ps.
    """
    # Convert cm^-1 (wavenumbers FWHM) to standard deviation in angular frequency (rad/ps)
    # sigma = FWHM / (2 * sqrt(2 * ln 2))
    sigma_factor = 2 * pi * c_0 / (2 * np.sqrt(2 * np.log(2)))
    g0 = dw_0 * sigma_factor
    g1 = dw_1 * sigma_factor

    # Avoid zero dephasing time division error
    inv_T2 = 1.0 / T2 if T2 > 0 else 0.0

    # standard formula
    val = (g0**2) * (t**2) / 2.0
    if tau_c > 0:
        exp_arg = np.clip(-t / tau_c, -700, 700)
        val += (g1**2) * (tau_c**2) * (np.exp(exp_arg) + t / tau_c - 1.0)
    val += t * inv_T2
    return val

def simulate_2d_spectrum(bands, decays, pump_axis=None, probe_axis=None, t2_delay=0.0,
                         tmax=5.0, nt=64, w_rot=1900.0, undersampling_factor=1.0, fid_data=None,
                         _legacy1=None, _legacy2=None, _legacy3=None, _legacy4=None):
    """Simulate a 2D-IR spectrum at a given t2 delay time (ps) with multiple bands and decay components.
    Supports backward compatibility for scalar arguments:
    simulate_2d_spectrum(w01, dw, dw_0, dw_1, tau_c, dw_T2, pump_axis, probe_axis, t2_delay, tmax, nt, w_rot, undersampling_factor, fid_data)
    """
    if np.isscalar(bands) or (not isinstance(bands, (list, tuple))):
        w01 = bands
        dw = decays
        _dw_0 = pump_axis  # legacy positional slot, unused
        dw_1 = probe_axis
        tau_c = t2_delay
        dw_T2 = tmax

        real_pump = nt
        real_probe = w_rot
        real_t2 = undersampling_factor

        real_tmax = fid_data if fid_data is not None else 5.0
        real_nt = _legacy1 if _legacy1 is not None else 64
        real_w_rot = _legacy2 if _legacy2 is not None else 1900.0
        real_u = _legacy3 if _legacy3 is not None else 1.0
        real_fid = _legacy4 if _legacy4 is not None else None

        bands_list = [(w01, dw)]
        decays_list = [(dw_1, tau_c, dw_T2)]
        return simulate_2d_spectrum(bands=bands_list, decays=decays_list, pump_axis=real_pump, probe_axis=real_probe, t2_delay=real_t2, tmax=real_tmax, nt=real_nt, w_rot=real_w_rot, undersampling_factor=real_u, fid_data=real_fid)

    t2_delay_val = t2_delay
    undersampling_factor_val = undersampling_factor

    # Calculate time step and nt
    if nt is None:
        # Time step calculation based on maximum frequency offset relative to w_rot
        all_freqs = np.concatenate([pump_axis, probe_axis])
        max_offset = np.max(np.abs(all_freqs - w_rot))
        max_offset_ps = max_offset * c_0  # linear frequency (cycles/ps)
        dt_nyq = 1.0 / (2.0 * max_offset_ps + 1e-12)
        dt_calc = dt_nyq * undersampling_factor_val
        nt_calc = int(np.ceil(tmax / dt_calc))
        nt_calc = max(8, nt_calc)
    else:
        nt_calc = int(np.ceil(nt / undersampling_factor_val))
        nt_calc = max(2, nt_calc)
        dt_calc = (tmax / nt) * undersampling_factor_val

    t1 = np.arange(nt_calc) * dt_calc
    t3 = np.arange(nt_calc) * dt_calc

    # Setup dephasing function
    def g_total(t):
        if fid_data is not None:
            fid_t, fid_abs = fid_data
            val = np.interp(t, fid_t, fid_abs)
            return -np.log(val + 1e-12)

        total = np.zeros_like(t, dtype=float)
        for d in decays:
            if len(d) == 3:
                dw_1, tau_c_val, dw_T2 = d
                T2 = 1.0 / (pi * c_0 * dw_T2) if dw_T2 > 0 else 0.0
                total += g_function(t, 0.0, dw_1, tau_c_val, T2)
            elif len(d) == 4:
                dw_0, dw_1, tau_c_val, dw_T2 = d
                T2 = 1.0 / (pi * c_0 * dw_T2) if dw_T2 > 0 else 0.0
                total += g_function(t, dw_0, dw_1, tau_c_val, T2)
        return total

    T1, T3 = np.meshgrid(t1, t3)

    g1 = g_total(T1)
    g2 = g_total(t2_delay_val)
    g3 = g_total(T3)
    g12 = g_total(T1 + t2_delay_val)
    g23 = g_total(t2_delay_val + T3)
    g123 = g_total(T1 + t2_delay_val + T3)

    env_re = -g1 + g2 - g3 - g12 - g23 + g123
    env_nr = -g1 - g2 - g3 + g12 + g23 - g123

    env_re = np.clip(env_re, -700, 700)
    env_nr = np.clip(env_nr, -700, 700)

    R_r = np.zeros_like(T1, dtype=complex)
    R_nr = np.zeros_like(T1, dtype=complex)

    for b in bands:
        if len(b) == 2:
            w01, dw = b
            a_gsb, a_esa = 1.0, 1.0
        elif len(b) == 4:
            w01, dw, a_gsb, a_esa = b
        else:
            w01 = b[0]
            dw = b[1]
            a_gsb = b[2] if len(b) > 2 else 1.0
            a_esa = b[3] if len(b) > 3 else 1.0

        w01_rf = -(w01 - w_rot) * 2 * pi * c_0
        Delta_rad = dw * 2 * pi * c_0

        # Rephasing pathway
        r3_re_gsb = -2.0 * a_gsb * np.exp(-1j * w01_rf * (-T1 + T3)) * np.exp(env_re)
        r3_re_esa = 2.0 * a_esa * np.exp(-1j * w01_rf * (-T1 + T3)) * np.exp(env_re) * np.exp(-1j * Delta_rad * T3)
        R_r += (r3_re_gsb + r3_re_esa)

        # Non-rephasing pathway
        r3_nr_gsb = -2.0 * a_gsb * np.exp(-1j * w01_rf * (T1 + T3)) * np.exp(env_nr)
        r3_nr_esa = 2.0 * a_esa * np.exp(-1j * w01_rf * (T1 + T3)) * np.exp(env_nr) * np.exp(-1j * Delta_rad * T3)
        R_nr += (r3_nr_gsb + r3_nr_esa)

    # First-point correction
    R_r[:, 0] /= 2
    R_r[0, :] /= 2
    R_nr[:, 0] /= 2
    R_nr[0, :] /= 2

    # Correct for double counting at t=0
    R_r[0, 0] *= 2
    R_nr[0, 0] *= 2

    n_zp = nt_calc * 4
    R_r_fft = np.fft.fft2(R_r, s=(n_zp, n_zp))
    R_nr_fft = np.fft.fft2(R_nr, s=(n_zp, n_zp))

    # Combine according to Hamm & Zanni
    R_combined = np.fft.fftshift(np.real(np.fliplr(np.roll(R_r_fft, -1, axis=1)) + R_nr_fft))

    # Frequencies axis
    freqAxis = np.fft.fftshift(np.fft.fftfreq(n_zp, dt_calc)) / c_0 + w_rot

    # Interpolate from (freqAxis, freqAxis) probe/pump to (pump_axis, probe_axis)
    from scipy.interpolate import RegularGridInterpolator
    interp_fun = RegularGridInterpolator(
        (freqAxis, freqAxis), R_combined.T,
        bounds_error=False, fill_value=0.0
    )

    PP, PR = np.meshgrid(pump_axis, probe_axis, indexing='ij')
    points = np.column_stack([PP.ravel(), PR.ravel()])
    Z_interp = interp_fun(points).reshape(PP.shape)

    return Z_interp

def unpack_kubo_params(params, num_bands, num_decays):
    """Unpack parameter vector into bands and decays lists."""
    bands = []
    for i in range(num_bands):
        idx = 4 * i
        w01 = params[idx]
        dw = params[idx + 1]
        a_gsb = params[idx + 2]
        a_esa = params[idx + 3]
        bands.append((w01, dw, a_gsb, a_esa))

    decays = []
    for j in range(num_decays):
        idx = 4 * num_bands + 3 * j
        dw_1 = params[idx]
        tau_c = params[idx + 1]
        dw_T2 = params[idx + 2]
        decays.append((dw_1, tau_c, dw_T2))

    return bands, decays

def compute_jacobian(params, bands_num, decays_num, pump_axis, probe_axis, t2, tmax, nt, w_rot, undersampling_factor, actual_fid):
    n_params = len(params)
    eps = 1e-5

    def get_scaled_sim(p):
        if len(p) == 6:
            w01, dw, dw_0, dw_1, tau_c, dw_T2 = p
            z = simulate_2d_spectrum(
                w01, dw, dw_0, dw_1, tau_c, dw_T2,
                pump_axis, probe_axis, t2,
                tmax, nt, w_rot,
                undersampling_factor, actual_fid
            )
        else:
            bands_opt, decays_opt = unpack_kubo_params(p, bands_num, decays_num)
            z = simulate_2d_spectrum(
                bands_opt, decays_opt, pump_axis, probe_axis, t2,
                tmax=tmax, nt=nt, w_rot=w_rot,
                undersampling_factor=undersampling_factor, fid_data=actual_fid
            )
        z_max = np.max(np.abs(z))
        if z_max > 0:
            z = z / z_max
        return z

    z_base = get_scaled_sim(params)
    M_points = len(pump_axis) * len(probe_axis)
    J = np.zeros((M_points, n_params))

    for j in range(n_params):
        p_perturbed = np.array(params, dtype=float)
        h = eps * max(abs(params[j]), 1.0)
        p_perturbed[j] += h
        z_perturbed = get_scaled_sim(p_perturbed)
        diff = (z_perturbed - z_base) / h
        J[:, j] = diff.ravel()

    return J

def fit_single_delay_worker(args):
    """Picklable top-level worker function to optimise a single delay."""
    if len(args) == 14:
        (Z_exp, pump_axis, probe_axis, t2, p0, bounds, tmax, nt, w_rot,
         undersampling_factor, fid_data, num_bands, num_decays) = args[:-1]
        iteration_callback = args[-1]
    else:
        Z_exp, pump_axis, probe_axis, t2, p0, bounds = args
        tmax = 5.0
        nt = 64
        w_rot = 1900.0
        undersampling_factor = 1.0
        fid_data = None
        num_bands = 1
        num_decays = 1
        iteration_callback = None

    exp_max = np.max(np.abs(Z_exp))
    if exp_max > 0:
        Z_exp_norm = Z_exp / exp_max
    else:
        Z_exp_norm = Z_exp

    actual_fid = fid_data.get("fid") if isinstance(fid_data, dict) else fid_data
    is_legacy = (len(p0) == 6)

    # Track costs for denominator of SIGN
    cost_history = []

    def cost_fun(params):
        if is_legacy:
            w01, dw, dw_0, dw_1, tau_c, dw_T2 = params
            Z_sim = simulate_2d_spectrum(
                w01, dw, dw_0, dw_1, tau_c, dw_T2,
                pump_axis, probe_axis, t2,
                tmax, nt, w_rot,
                undersampling_factor, actual_fid
            )
        else:
            bands_opt, decays_opt = unpack_kubo_params(params, num_bands, num_decays)
            Z_sim = simulate_2d_spectrum(
                bands_opt, decays_opt, pump_axis, probe_axis, t2,
                tmax=tmax, nt=nt, w_rot=w_rot,
                undersampling_factor=undersampling_factor, fid_data=actual_fid
            )

        sim_max = np.max(np.abs(Z_sim))
        if sim_max > 0:
            Z_sim = Z_sim / sim_max

        scale = np.maximum(0.0, np.sum(Z_exp_norm * Z_sim) / (np.sum(Z_sim**2) + 1e-12))
        residuals = Z_exp_norm - scale * Z_sim
        cost = np.sum(residuals**2)
        cost_history.append(cost)

        if iteration_callback is not None:
            iteration_callback(params)

        return cost

    res = minimize(cost_fun, p0, bounds=bounds, method='L-BFGS-B')

    cost_val = res.fun
    grad = getattr(res, "jac", np.zeros_like(p0))

    # Calculate SIGN metric using Robben convention: SIGN_j = grad_j * sigma_j / C_{i-1}
    M = len(pump_axis) * len(probe_axis)
    N = len(res.x)
    C_prev = cost_history[-2] if len(cost_history) >= 2 else cost_val
    if C_prev <= 0:
        C_prev = 1e-12

    try:
        J = compute_jacobian(
            res.x, num_bands, num_decays, pump_axis, probe_axis, t2,
            tmax, nt, w_rot, undersampling_factor, actual_fid
        )
        JTJ = np.dot(J.T, J)
        JTJ += np.eye(N) * 1e-12
        cov = np.linalg.inv(JTJ) * (cost_val / max(1, M - N))
        sigma = np.sqrt(np.clip(np.diagonal(cov), 1e-16, None))
        sign_metrics = (grad * sigma) / C_prev
    except Exception:
        # Fallback if matrix inversion fails
        sigma = np.zeros_like(res.x)
        sign_metrics = (grad * res.x) / (C_prev * (cost_val + 1e-12))

    if is_legacy:
        w01, dw, dw_0, dw_1, tau_c, dw_T2 = res.x
        Z_best = simulate_2d_spectrum(
            w01, dw, dw_0, dw_1, tau_c, dw_T2,
            pump_axis, probe_axis, t2,
            tmax, nt, w_rot,
            undersampling_factor, actual_fid
        )
    else:
        bands_best, decays_best = unpack_kubo_params(res.x, num_bands, num_decays)
        Z_best = simulate_2d_spectrum(
            bands_best, decays_best, pump_axis, probe_axis, t2,
            tmax=tmax, nt=nt, w_rot=w_rot,
            undersampling_factor=undersampling_factor, fid_data=actual_fid
        )

    sim_max = np.max(np.abs(Z_best))
    if sim_max > 0:
        Z_best = Z_best / sim_max
    scale = np.maximum(0.0, np.sum(Z_exp_norm * Z_best) / (np.sum(Z_best**2) + 1e-12))

    Z_best_norm = scale * Z_best
    Z_sim_restored = Z_best_norm * exp_max

    return {
        "delay": t2,
        "params": res.x,
        "success": res.success,
        "cost": cost_val,
        "sign": sign_metrics,
        "errors": sigma,
        "Z_sim": Z_sim_restored,
        "Z_exp": Z_exp
    }

def run_kubo_fit(data_cube, pump_axis, probe_axis, delays, p0, bounds,
                 tmax=5.0, nt=64, w_rot=1900.0, undersampling_factor=1.0,
                 fid_data=None, num_bands=1, num_decays=1, parallel_mode="threadpool",
                 iteration_callback=None):
    """Execute optimisation of Kubo parameters across all delays."""
    tasks = []
    actual_fid = fid_data.get("fid") if isinstance(fid_data, dict) else fid_data
    for idx, t2 in enumerate(delays):
        tasks.append((data_cube[:, :, idx], pump_axis, probe_axis, t2, p0, bounds,
                      tmax, nt, w_rot, undersampling_factor, actual_fid, num_bands, num_decays, iteration_callback))

    results = []
    if parallel_mode == "threadpool" and len(delays) > 1:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(fit_single_delay_worker, tasks))
    elif parallel_mode == "processpool" and len(delays) > 1:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(fit_single_delay_worker, tasks))
    else:
        # Run sequentially
        results = [fit_single_delay_worker(t) for t in tasks]

    logger.info(
        "\n%s\nDirect Kubo Model Fitting Completed.\nMethod Reference:\n"
        "  Robben, K. C.; Cheatum, C. M. Least-Squares Fitting of Multidimensional Spectra\n"
        "  to Kubo Line-Shape Models. J. Phys. Chem. B 2021, 125, 46, 12876-12891.\n"
        "  DOI: 10.1021/acs.jpcb.1c08764\n%s\n",
        "=" * 80,
        "=" * 80,
    )

    return results

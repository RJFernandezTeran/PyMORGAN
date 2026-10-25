"""Direct 2D-IR Kubo line-shape simulation and multi-core nonlinear least-squares fitting.

Supports:
- Generalized Frequency Fluctuation Correlation Functions (FFCF) with homogeneous
  dephasing (Gamma_hom / T2*), static inhomogeneity (Delta_0), and arbitrary
  multi-exponential Kubo decay terms (Delta_m, tau_c,m).
- Shared FFCF across bands or per-band independent FFCFs.
- Diagonal and off-diagonal / cross-peak bands (w1 != w3) with parameter linking
  (link_w1, link_w3, link_anharm) and cross-peak bath correlation factor eta_AB.
- Fast separable 1D outer-product time-domain evaluation, batched 2D FFTs, and
  separable 2D rectilinear grid interpolation.
- Trust-Region Reflective (`scipy.optimize.least_squares`) with Variable Projection
  (VarPro) for analytical linear pathway amplitudes (A_GSB, A_ESA), alongside
  full-parameter TRF and legacy L-BFGS-B solvers.
- Multi-core parallel execution across delays (ProcessPool / ThreadPool), parallel
  finite-difference Jacobian column evaluation, and parallel full-cube reconstruction.
- Analytical 1st-order FTIR simulation for joint 2D-IR + FTIR fitting as well as
  empirical 1st-order FID envelope damping.

Method Reference:
  Robben, K. C.; Cheatum, C. M. Least-Squares Fitting of Multidimensional Spectra
  to Kubo Line-Shape Models. J. Phys. Chem. B 2021, 125, 46, 12876-12891.
  DOI: 10.1021/acs.jpcb.1c08764
"""

from __future__ import annotations

import concurrent.futures
import copy
import os
import time
from typing import Any

import numpy as np
from numpy import pi
from scipy.optimize import least_squares, lsq_linear, minimize

from ..log import get_logger

logger = get_logger(__name__)

c_0 = 2.99792458e-2  # Speed of light in cm/ps
SIGMA_FACTOR = 2.0 * pi * c_0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # FWHM (cm^-1) -> std dev (rad/ps)


# --------------------------------------------------------------------------- #
#                        1. FFCF & Lineshape Functions                        #
# --------------------------------------------------------------------------- #

def g_function(t, dw_0, dw_1, tau_c, T2):
    """Kubo lineshape function g(t) (dimensionless).

    Parameters
    ----------
    t : float or ndarray
        Time array in ps.
    dw_0 : float
        Static inhomogeneous broadening FWHM in cm^-1.
    dw_1 : float
        Kubo fluctuation amplitude FWHM in cm^-1.
    tau_c : float
        Correlation time in ps.
    T2 : float
        Homogeneous pure dephasing time T2* in ps (0 or <=0 means infinite T2*).
    """
    t_arr = np.asarray(t, dtype=float)
    g0 = float(dw_0) * SIGMA_FACTOR
    g1 = float(dw_1) * SIGMA_FACTOR

    inv_T2 = 1.0 / float(T2) if (T2 is not None and float(T2) > 0.0) else 0.0

    val = 0.5 * (g0 ** 2) * (t_arr ** 2)
    if tau_c is not None and float(tau_c) > 0.0 and g1 != 0.0:
        tc = float(tau_c)
        exp_arg = np.clip(-t_arr / tc, -700.0, 700.0)
        val = val + (g1 ** 2) * (tc ** 2) * (np.exp(exp_arg) + t_arr / tc - 1.0)
    if inv_T2 > 0.0:
        val = val + t_arr * inv_T2
    return val


def normalize_ffcf_groups(decays) -> list[dict[str, Any]]:
    """Convert any legacy or structured `decays` representation into a canonical list of FFCF group dicts.

    Canonical FFCF group dict format:
        {
            "dw_T2": float,              # Homogeneous FWHM in cm^-1
            "dw_0": float,               # Static inhomogeneous FWHM in cm^-1
            "terms": [(dw_1, tau_1), ...] # Exponential Kubo terms (cm^-1, ps)
        }
    """
    if decays is None:
        return [{"dw_T2": 0.0, "dw_0": 0.0, "terms": []}]

    if isinstance(decays, dict):
        return [_normalize_single_ffcf_dict(decays)]

    if isinstance(decays, (list, tuple)):
        if len(decays) == 0:
            return [{"dw_T2": 0.0, "dw_0": 0.0, "terms": []}]

        # List of FFCF group dicts
        if isinstance(decays[0], dict) and ("terms" in decays[0] or "dw_T2" in decays[0] or "dw_0" in decays[0]):
            return [_normalize_single_ffcf_dict(d) for d in decays]

        # Legacy list of decay tuples: [(dw_1, tau_c, dw_T2), ...] or [(dw_0, dw_1, tau_c, dw_T2), ...]
        dw_T2_total = 0.0
        dw_0_sq_total = 0.0
        terms = []
        for d in decays:
            if isinstance(d, dict):
                dw_1 = float(d.get("dw_1", d.get("dw", 0.0)))
                tau_c = float(d.get("tau_c", d.get("tau", 1.0)))
                dw_0 = float(d.get("dw_0", 0.0))
                dw_T2 = float(d.get("dw_T2", 0.0))
                dw_T2_total += max(0.0, dw_T2)
                dw_0_sq_total += dw_0 ** 2
                terms.append((dw_1, tau_c))
            elif len(d) == 2:
                terms.append((float(d[0]), float(d[1])))
            elif len(d) == 3:
                dw_1, tau_c, dw_T2 = float(d[0]), float(d[1]), float(d[2])
                dw_T2_total += max(0.0, dw_T2)
                terms.append((dw_1, tau_c))
            elif len(d) >= 4:
                dw_0, dw_1, tau_c, dw_T2 = float(d[0]), float(d[1]), float(d[2]), float(d[3])
                dw_0_sq_total += dw_0 ** 2
                dw_T2_total += max(0.0, dw_T2)
                terms.append((dw_1, tau_c))
        return [{
            "dw_T2": float(dw_T2_total),
            "dw_0": float(np.sqrt(dw_0_sq_total)),
            "terms": terms,
        }]

    return [{"dw_T2": 0.0, "dw_0": 0.0, "terms": []}]


def _normalize_single_ffcf_dict(d: dict[str, Any]) -> dict[str, Any]:
    dw_T2 = float(d.get("dw_T2", d.get("gamma_hom", 0.0)))
    dw_0 = float(d.get("dw_0", d.get("static_inhom", 0.0)))
    raw_terms = d.get("terms", d.get("decays", []))
    terms = []
    for t_item in raw_terms:
        if isinstance(t_item, dict):
            terms.append((float(t_item.get("dw", t_item.get("dw_1", 10.0))), float(t_item.get("tau", t_item.get("tau_c", 1.0)))))
        elif len(t_item) >= 2:
            terms.append((float(t_item[0]), float(t_item[1])))
    return {"dw_T2": dw_T2, "dw_0": dw_0, "terms": terms}


def evaluate_ffcf_gt(t, ffcf_group, fid_tuple=None, ffcf_id: int = 0) -> np.ndarray:
    """Evaluate the line-broadening function g(t) for a single canonical FFCF group or list of groups."""
    t_arr = np.asarray(t, dtype=float)
    if fid_tuple is not None:
        fid_t, fid_abs = fid_tuple
        val = np.interp(t_arr, fid_t, fid_abs)
        return -np.log(np.maximum(val, 1e-12))

    if isinstance(ffcf_group, dict) and "terms" in ffcf_group:
        grp = ffcf_group
    else:
        groups = normalize_ffcf_groups(ffcf_group)
        grp = groups[min(max(0, int(ffcf_id)), len(groups) - 1)]

    dw_T2 = float(grp.get("dw_T2", 0.0))
    dw_0 = float(grp.get("dw_0", 0.0))
    terms = grp.get("terms", [])

    # Homogeneous + static inhomogeneous baseline
    inv_T2 = pi * c_0 * dw_T2 if dw_T2 > 0.0 else 0.0
    g0 = dw_0 * SIGMA_FACTOR
    total = inv_T2 * t_arr + 0.5 * (g0 ** 2) * (t_arr ** 2)

    # Sum exponential Kubo decay components
    for dw_m, tau_m in terms:
        if tau_m > 0.0 and dw_m != 0.0:
            gm = float(dw_m) * SIGMA_FACTOR
            tc = float(tau_m)
            exp_arg = np.clip(-t_arr / tc, -700.0, 700.0)
            total = total + (gm ** 2) * (tc ** 2) * (np.exp(exp_arg) + t_arr / tc - 1.0)

    return total


def evaluate_ffcf_ct(t, decays, ffcf_id: int = 0) -> np.ndarray:
    """Evaluate the Frequency Fluctuation Correlation Function C(t) in cm^-2 (FWHM^2 units).

    C(t) = Delta_0^2 + sum_m Delta_m^2 * exp(-t / tau_c,m)
    """
    t_arr = np.asarray(t, dtype=float)
    groups = normalize_ffcf_groups(decays)
    grp = groups[min(max(0, int(ffcf_id)), len(groups) - 1)]
    c_val = np.full_like(t_arr, float(grp.get("dw_0", 0.0)) ** 2, dtype=float)
    for dw_m, tau_m in grp.get("terms", []):
        if tau_m > 0.0:
            c_val = c_val + (float(dw_m) ** 2) * np.exp(-np.clip(t_arr / float(tau_m), 0.0, 700.0))
    return c_val


def normalize_bands(bands) -> list[dict[str, Any]]:
    """Convert any legacy tuple or dict band list into canonical band dicts.

    Canonical band dict:
        {
            "w1": float,
            "w3": float,
            "anharm": float,
            "amp_gsb": float,
            "amp_esa": float,
            "ffcf_id": int,
            "ffcf_id_w3": int,
            "corr_eta": float,
        }
    """
    if bands is None:
        return []
    norm_list = []
    for b in bands:
        if isinstance(b, dict):
            w1 = float(b.get("w1", b.get("w01", 2000.0)))
            link_w3 = b.get("link_w3", None)
            if link_w3 == "diagonal":
                w3 = w1
            else:
                w3 = float(b.get("w3", w1))
            anharm = float(b.get("anharm", b.get("dw", 15.0)))
            amp_gsb = float(b.get("amp_gsb", b.get("a_gsb", 1.0)))
            amp_esa = float(b.get("amp_esa", b.get("a_esa", 1.0)))
            ffcf_id = int(b.get("ffcf_id", 0))
            ffcf_id_w3 = int(b.get("ffcf_id_w3", ffcf_id))
            default_eta = 1.0 if (abs(w1 - w3) < 1e-6 and ffcf_id == ffcf_id_w3) else 0.5
            corr_eta = float(b.get("corr_eta", default_eta))
            norm_list.append({
                "w1": w1,
                "w3": w3,
                "anharm": anharm,
                "amp_gsb": amp_gsb,
                "amp_esa": amp_esa,
                "ffcf_id": ffcf_id,
                "ffcf_id_w3": ffcf_id_w3,
                "corr_eta": np.clip(corr_eta, -1.0, 1.0),
            })
        elif len(b) == 2:
            w01, dw = float(b[0]), float(b[1])
            norm_list.append({
                "w1": w01, "w3": w01, "anharm": dw,
                "amp_gsb": 1.0, "amp_esa": 1.0,
                "ffcf_id": 0, "ffcf_id_w3": 0, "corr_eta": 1.0,
            })
        elif len(b) == 4:
            w01, dw, a_gsb, a_esa = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            norm_list.append({
                "w1": w01, "w3": w01, "anharm": dw,
                "amp_gsb": a_gsb, "amp_esa": a_esa,
                "ffcf_id": 0, "ffcf_id_w3": 0, "corr_eta": 1.0,
            })
        elif len(b) == 5:
            w1, w3, dw, a_gsb, a_esa = float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4])
            norm_list.append({
                "w1": w1, "w3": w3, "anharm": dw,
                "amp_gsb": a_gsb, "amp_esa": a_esa,
                "ffcf_id": 0, "ffcf_id_w3": 0,
                "corr_eta": 1.0 if abs(w1 - w3) < 1e-6 else 0.5,
            })
        else:
            w01 = float(b[0])
            dw = float(b[1])
            a_gsb = float(b[2]) if len(b) > 2 else 1.0
            a_esa = float(b[3]) if len(b) > 3 else 1.0
            norm_list.append({
                "w1": w01, "w3": w01, "anharm": dw,
                "amp_gsb": a_gsb, "amp_esa": a_esa,
                "ffcf_id": 0, "ffcf_id_w3": 0, "corr_eta": 1.0,
            })
    return norm_list


# --------------------------------------------------------------------------- #
#              2. Fast Separable Interpolation & 2D-IR Simulator              #
# --------------------------------------------------------------------------- #

def _compute_time_grid(pump_axis, probe_axis, tmax: float, nt: int | None, w_rot: float, undersampling_factor: float):
    """Compute time step `dt_calc`, grid length `nt_calc`, and 1D time vector `t_vec`."""
    u_val = max(float(undersampling_factor), 1e-6)
    if nt is None:
        all_freqs = np.concatenate([np.asarray(pump_axis, dtype=float), np.asarray(probe_axis, dtype=float)])
        max_offset = float(np.max(np.abs(all_freqs - w_rot)))
        max_offset_ps = max_offset * c_0
        dt_nyq = 1.0 / (2.0 * max_offset_ps + 1e-12)
        dt_calc = dt_nyq * u_val
        nt_calc = int(np.ceil(float(tmax) / dt_calc))
        nt_calc = max(8, nt_calc)
    else:
        nt_calc = int(np.ceil(float(nt) / u_val))
        nt_calc = max(2, nt_calc)
        dt_calc = (float(tmax) / float(nt)) * u_val

    t_vec = np.arange(nt_calc, dtype=float) * dt_calc
    return t_vec, dt_calc, nt_calc


def _interp2d_separable(freq_axis: np.ndarray, Z_grid_pump_probe: np.ndarray, pump_axis: np.ndarray, probe_axis: np.ndarray) -> np.ndarray:
    """Fast separable bilinear interpolation from uniform `(freq_axis, freq_axis)` to `(pump_axis, probe_axis)`.

    Matches `RegularGridInterpolator((freq_axis, freq_axis), Z_grid_pump_probe, bounds_error=False, fill_value=0.0)`
    for both 2D arrays `(N_freq, N_freq)` and 3D batches `(N_batch, N_freq, N_freq)`.
    """
    n_f = len(freq_axis)
    f0 = float(freq_axis[0])
    df = float(freq_axis[1] - freq_axis[0]) if n_f > 1 else 1.0

    pump_arr = np.asarray(pump_axis, dtype=float)
    probe_arr = np.asarray(probe_axis, dtype=float)

    up = (pump_arr - f0) / df
    ur = (probe_arr - f0) / df

    valid_p = (up >= 0.0) & (up <= float(n_f - 1))
    valid_r = (ur >= 0.0) & (ur <= float(n_f - 1))

    up_clipped = np.clip(up, 0.0, float(max(0, n_f - 1)))
    ur_clipped = np.clip(ur, 0.0, float(max(0, n_f - 1)))

    i0 = np.floor(up_clipped).astype(int)
    j0 = np.floor(ur_clipped).astype(int)
    i1 = np.minimum(i0 + 1, n_f - 1)
    j1 = np.minimum(j0 + 1, n_f - 1)

    wp = up_clipped - i0
    wr = ur_clipped - j0

    if Z_grid_pump_probe.ndim == 2:
        # Interpolate along axis 0 (pump) for the required probe columns
        z_i0_j0 = Z_grid_pump_probe[np.ix_(i0, j0)]
        z_i1_j0 = Z_grid_pump_probe[np.ix_(i1, j0)]
        z_i0_j1 = Z_grid_pump_probe[np.ix_(i0, j1)]
        z_i1_j1 = Z_grid_pump_probe[np.ix_(i1, j1)]

        wp_col = wp[:, None]
        wr_row = wr[None, :]

        z_j0 = (1.0 - wp_col) * z_i0_j0 + wp_col * z_i1_j0
        z_j1 = (1.0 - wp_col) * z_i0_j1 + wp_col * z_i1_j1
        out = (1.0 - wr_row) * z_j0 + wr_row * z_j1

        if not np.all(valid_p):
            out[~valid_p, :] = 0.0
        if not np.all(valid_r):
            out[:, ~valid_r] = 0.0
        return out

    # Batched 3D array: shape (B, n_f, n_f)
    z_i0_j0 = Z_grid_pump_probe[:, i0, :][:, :, j0]
    z_i1_j0 = Z_grid_pump_probe[:, i1, :][:, :, j0]
    z_i0_j1 = Z_grid_pump_probe[:, i0, :][:, :, j1]
    z_i1_j1 = Z_grid_pump_probe[:, i1, :][:, :, j1]

    wp_col = wp[None, :, None]
    wr_row = wr[None, None, :]

    z_j0 = (1.0 - wp_col) * z_i0_j0 + wp_col * z_i1_j0
    z_j1 = (1.0 - wp_col) * z_i0_j1 + wp_col * z_i1_j1
    out = (1.0 - wr_row) * z_j0 + wr_row * z_j1

    if not np.all(valid_p):
        out[:, ~valid_p, :] = 0.0
    if not np.all(valid_r):
        out[:, :, ~valid_r] = 0.0
    return out


def _compute_envelope_pair(
    t_vec: np.ndarray,
    t2_delay: float,
    ffcf_groups: list[dict[str, Any]],
    id1: int,
    id3: int,
    corr_eta: float,
    actual_fid,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute `(exp_env_re, exp_env_nr)` on the `(nt_calc, nt_calc)` grid (axis 0 = t3, axis 1 = t1)."""
    n_groups = len(ffcf_groups)
    grp1 = ffcf_groups[min(max(0, id1), n_groups - 1)]
    grp3 = ffcf_groups[min(max(0, id3), n_groups - 1)]

    # 1D evaluations along t1 and t3
    g1_1d = evaluate_ffcf_gt(t_vec, grp1, fid_tuple=actual_fid)
    g3_1d = g1_1d if (id1 == id3) else evaluate_ffcf_gt(t_vec, grp3, fid_tuple=actual_fid)

    # T1 varies along columns (axis 1), T3 varies along rows (axis 0)
    T1 = t_vec[None, :]
    T3 = t_vec[:, None]

    base_damp = -g1_1d[None, :] - g3_1d[:, None]

    if abs(corr_eta) < 1e-9:
        exp_base = np.exp(np.clip(base_damp, -700.0, 700.0))
        return exp_base, exp_base

    # Waiting-time correlation terms
    t2_val = float(t2_delay)
    if id1 == id3:
        g2 = float(evaluate_ffcf_gt(t2_val, grp1, fid_tuple=actual_fid))
        g12_1d = evaluate_ffcf_gt(t_vec + t2_val, grp1, fid_tuple=actual_fid)
        g23_1d = g12_1d
        g123_2d = evaluate_ffcf_gt(T1 + t2_val + T3, grp1, fid_tuple=actual_fid)
    else:
        g2 = 0.5 * float(
            evaluate_ffcf_gt(t2_val, grp1, fid_tuple=actual_fid)
            + evaluate_ffcf_gt(t2_val, grp3, fid_tuple=actual_fid)
        )
        g12_1d = 0.5 * (
            evaluate_ffcf_gt(t_vec + t2_val, grp1, fid_tuple=actual_fid)
            + evaluate_ffcf_gt(t_vec + t2_val, grp3, fid_tuple=actual_fid)
        )
        g23_1d = g12_1d
        t123 = T1 + t2_val + T3
        g123_2d = 0.5 * (
            evaluate_ffcf_gt(t123, grp1, fid_tuple=actual_fid)
            + evaluate_ffcf_gt(t123, grp3, fid_tuple=actual_fid)
        )

    mem_term = float(corr_eta) * (g2 - g12_1d[None, :] - g23_1d[:, None] + g123_2d)
    env_re = np.clip(base_damp + mem_term, -700.0, 700.0)
    env_nr = np.clip(base_damp - mem_term, -700.0, 700.0)
    return np.exp(env_re), np.exp(env_nr)


def _extract_empirical_fid(fid_data):
    """Return `(fid_t, fid_abs)` only if `fid_data` is configured as an empirical FID override."""
    if fid_data is None:
        return None
    if isinstance(fid_data, dict):
        mode = str(fid_data.get("mode", "empirical_fid")).lower()
        if mode == "joint_fit":
            return None
        return fid_data.get("fid")
    if isinstance(fid_data, (tuple, list)) and len(fid_data) == 2:
        return fid_data
    return None


def simulate_2d_basis_maps(
    bands,
    decays,
    pump_axis: np.ndarray,
    probe_axis: np.ndarray,
    t2_delay: float = 0.0,
    tmax: float = 5.0,
    nt: int | None = 64,
    w_rot: float = 1900.0,
    undersampling_factor: float = 1.0,
    fid_data=None,
    normalize_unit_basis: bool = False,
) -> tuple[np.ndarray, float]:
    """Compute the 2*K basis maps [GSB_0, ESA_0, GSB_1, ESA_1, ...] on `(pump_axis, probe_axis)`.

    Uses batched 2D FFTs and separable bilinear interpolation so all 2*K basis maps
    are computed in a single vectorized pass.

    Returns
    -------
    basis_maps : ndarray, shape (2 * K, N_pump, N_probe)
        Individual unit-amplitude pathway maps for each band's GSB and ESA.
    unit_scale : float
        Reference peak magnitude of the unit-amplitude basis (1.0 if `normalize_unit_basis=False`).
    """
    norm_b = normalize_bands(bands)
    n_bands = len(norm_b)
    pump_arr = np.asarray(pump_axis, dtype=float)
    probe_arr = np.asarray(probe_axis, dtype=float)

    if n_bands == 0:
        return np.zeros((0, len(pump_arr), len(probe_arr)), dtype=float), 1.0

    t_vec, dt_calc, nt_calc = _compute_time_grid(pump_arr, probe_arr, tmax, nt, w_rot, undersampling_factor)
    ffcf_groups = normalize_ffcf_groups(decays)
    actual_fid = _extract_empirical_fid(fid_data)

    # Cache envelope pairs by (ffcf_id, ffcf_id_w3, corr_eta)
    env_cache: dict[tuple[int, int, float], tuple[np.ndarray, np.ndarray]] = {}
    for b in norm_b:
        key = (b["ffcf_id"], b["ffcf_id_w3"], round(float(b["corr_eta"]), 6))
        if key not in env_cache:
            env_cache[key] = _compute_envelope_pair(
                t_vec, t2_delay, ffcf_groups, b["ffcf_id"], b["ffcf_id_w3"], b["corr_eta"], actual_fid
            )

    n_basis = 2 * n_bands
    R_r_batch = np.zeros((n_basis, nt_calc, nt_calc), dtype=complex)
    R_nr_batch = np.zeros((n_basis, nt_calc, nt_calc), dtype=complex)

    for k, b in enumerate(norm_b):
        key = (b["ffcf_id"], b["ffcf_id_w3"], round(float(b["corr_eta"]), 6))
        exp_env_re, exp_env_nr = env_cache[key]

        w1_rf = -(b["w1"] - w_rot) * 2.0 * pi * c_0
        w3_rf = -(b["w3"] - w_rot) * 2.0 * pi * c_0
        delta_rad = b["anharm"] * 2.0 * pi * c_0

        v1_re = np.exp(1j * w1_rf * t_vec)
        v1_nr = np.exp(-1j * w1_rf * t_vec)
        v3_gsb = -2.0 * np.exp(-1j * w3_rf * t_vec)
        v3_esa = 2.0 * np.exp(-1j * (w3_rf + delta_rad) * t_vec)

        # GSB basis (index 2*k)
        osc_re_gsb = np.outer(v3_gsb, v1_re)
        osc_nr_gsb = np.outer(v3_gsb, v1_nr)
        R_r_batch[2 * k] = osc_re_gsb * exp_env_re
        R_nr_batch[2 * k] = osc_nr_gsb * exp_env_nr

        # ESA basis (index 2*k + 1)
        osc_re_esa = np.outer(v3_esa, v1_re)
        osc_nr_esa = np.outer(v3_esa, v1_nr)
        R_r_batch[2 * k + 1] = osc_re_esa * exp_env_re
        R_nr_batch[2 * k + 1] = osc_nr_esa * exp_env_nr

    # First-point trapezoidal correction across batch
    R_r_batch[:, :, 0] *= 0.5
    R_r_batch[:, 0, :] *= 0.5
    R_nr_batch[:, :, 0] *= 0.5
    R_nr_batch[:, 0, :] *= 0.5
    R_r_batch[:, 0, 0] *= 2.0
    R_nr_batch[:, 0, 0] *= 2.0

    n_zp = nt_calc * 4
    R_r_fft = np.fft.fft2(R_r_batch, s=(n_zp, n_zp), axes=(-2, -1))
    R_nr_fft = np.fft.fft2(R_nr_batch, s=(n_zp, n_zp), axes=(-2, -1))

    # Flip and roll along axis -1 (pump axis in FFT output)
    R_r_flipped = np.roll(R_r_fft, -1, axis=-1)[:, :, ::-1]
    R_combined = np.fft.fftshift(np.real(R_r_flipped + R_nr_fft), axes=(-2, -1))

    # Transpose each map in batch from (probe, pump) to (pump, probe)
    Z_grid_batch = np.swapaxes(R_combined, -2, -1)

    freq_axis = np.fft.fftshift(np.fft.fftfreq(n_zp, dt_calc)) / c_0 + w_rot
    basis_maps = _interp2d_separable(freq_axis, Z_grid_batch, pump_arr, probe_arr)

    unit_scale = 1.0
    if normalize_unit_basis:
        gsb_max = float(np.max(np.abs(basis_maps[0::2]))) if basis_maps.size > 0 else 1.0
        unit_scale = gsb_max if gsb_max > 1e-12 else 1.0
        basis_maps = basis_maps / unit_scale

    return basis_maps, unit_scale


def simulate_2d_spectrum(
    bands,
    decays,
    pump_axis=None,
    probe_axis=None,
    t2_delay=0.0,
    tmax=5.0,
    nt=64,
    w_rot=1900.0,
    undersampling_factor=1.0,
    fid_data=None,
    _legacy1=None,
    _legacy2=None,
    _legacy3=None,
    _legacy4=None,
    *,
    normalize_unit_basis: bool = False,
):
    """Simulate a 2D-IR spectrum at a given t2 delay time (ps) with multiple bands and FFCF components.

    Supports backward compatibility for legacy scalar positional arguments:
        simulate_2d_spectrum(w01, dw, dw_0, dw_1, tau_c, dw_T2, pump_axis, probe_axis, t2_delay, ...)
    """
    if np.isscalar(bands) or (not isinstance(bands, (list, tuple))):
        w01 = float(bands)
        dw = float(decays)
        dw_0 = float(pump_axis) if pump_axis is not None else 0.0
        dw_1 = float(probe_axis)
        tau_c = float(t2_delay)
        dw_T2 = float(tmax)

        real_pump = nt
        real_probe = w_rot
        real_t2 = undersampling_factor

        real_tmax = fid_data if fid_data is not None else 5.0
        real_nt = _legacy1 if _legacy1 is not None else 64
        real_w_rot = _legacy2 if _legacy2 is not None else 1900.0
        real_u = _legacy3 if _legacy3 is not None else 1.0
        real_fid = _legacy4 if _legacy4 is not None else None

        bands_list = [(w01, dw)]
        decays_list = [(dw_0, dw_1, tau_c, dw_T2)] if dw_0 != 0.0 else [(dw_1, tau_c, dw_T2)]
        return simulate_2d_spectrum(
            bands=bands_list,
            decays=decays_list,
            pump_axis=real_pump,
            probe_axis=real_probe,
            t2_delay=real_t2,
            tmax=real_tmax,
            nt=real_nt,
            w_rot=real_w_rot,
            undersampling_factor=real_u,
            fid_data=real_fid,
            normalize_unit_basis=normalize_unit_basis,
        )

    pump_arr = np.asarray(pump_axis, dtype=float)
    probe_arr = np.asarray(probe_axis, dtype=float)
    norm_b = normalize_bands(bands)
    if len(norm_b) == 0:
        return np.zeros((len(pump_arr), len(probe_arr)), dtype=float)

    if normalize_unit_basis:
        basis_maps, _ = simulate_2d_basis_maps(
            norm_b,
            decays,
            pump_arr,
            probe_arr,
            t2_delay=t2_delay,
            tmax=tmax,
            nt=nt,
            w_rot=w_rot,
            undersampling_factor=undersampling_factor,
            fid_data=fid_data,
            normalize_unit_basis=True,
        )
        weights = np.empty(2 * len(norm_b), dtype=float)
        for k, b in enumerate(norm_b):
            weights[2 * k] = b["amp_gsb"]
            weights[2 * k + 1] = b["amp_esa"]
        return np.tensordot(weights, basis_maps, axes=(0, 0))

    # Fast grouped evaluation for standard unnormalized simulation
    t_vec, dt_calc, nt_calc = _compute_time_grid(pump_arr, probe_arr, tmax, nt, w_rot, undersampling_factor)
    ffcf_groups = normalize_ffcf_groups(decays)
    actual_fid = _extract_empirical_fid(fid_data)

    # Group bands by envelope key so exp(env_re) and exp(env_nr) are multiplied once per FFCF group
    grouped_bands: dict[tuple[int, int, float], list[dict[str, Any]]] = {}
    for b in norm_b:
        key = (b["ffcf_id"], b["ffcf_id_w3"], round(float(b["corr_eta"]), 6))
        grouped_bands.setdefault(key, []).append(b)

    R_r = np.zeros((nt_calc, nt_calc), dtype=complex)
    R_nr = np.zeros((nt_calc, nt_calc), dtype=complex)

    for (id1, id3, eta), b_list in grouped_bands.items():
        exp_env_re, exp_env_nr = _compute_envelope_pair(t_vec, t2_delay, ffcf_groups, id1, id3, eta, actual_fid)
        osc_re_sum = np.zeros((nt_calc, nt_calc), dtype=complex)
        osc_nr_sum = np.zeros((nt_calc, nt_calc), dtype=complex)

        for b in b_list:
            w1_rf = -(b["w1"] - w_rot) * 2.0 * pi * c_0
            w3_rf = -(b["w3"] - w_rot) * 2.0 * pi * c_0
            delta_rad = b["anharm"] * 2.0 * pi * c_0

            v1_re = np.exp(1j * w1_rf * t_vec)
            v1_nr = np.exp(-1j * w1_rf * t_vec)
            v3_tot = (
                -2.0 * b["amp_gsb"] * np.exp(-1j * w3_rf * t_vec)
                + 2.0 * b["amp_esa"] * np.exp(-1j * (w3_rf + delta_rad) * t_vec)
            )

            osc_re_sum += np.outer(v3_tot, v1_re)
            osc_nr_sum += np.outer(v3_tot, v1_nr)

        R_r += osc_re_sum * exp_env_re
        R_nr += osc_nr_sum * exp_env_nr

    # First-point correction
    R_r[:, 0] *= 0.5
    R_r[0, :] *= 0.5
    R_nr[:, 0] *= 0.5
    R_nr[0, :] *= 0.5
    R_r[0, 0] *= 2.0
    R_nr[0, 0] *= 2.0

    n_zp = nt_calc * 4
    R_r_fft = np.fft.fft2(R_r, s=(n_zp, n_zp))
    R_nr_fft = np.fft.fft2(R_nr, s=(n_zp, n_zp))

    R_combined = np.fft.fftshift(np.real(np.fliplr(np.roll(R_r_fft, -1, axis=1)) + R_nr_fft))
    freq_axis = np.fft.fftshift(np.fft.fftfreq(n_zp, dt_calc)) / c_0 + w_rot

    return _interp2d_separable(freq_axis, R_combined.T, pump_arr, probe_arr)


def simulate_1d_ftir_spectrum(
    bands,
    decays,
    freq_axis: np.ndarray,
    tmax: float = 5.0,
    nt: int | None = 256,
    normalize: bool = True,
) -> np.ndarray:
    """Simulate the 1st-order linear FTIR absorption spectrum from the analytical Kubo FFCF g(t).

    I_FTIR(w) = sum_{k in diagonal bands} A_GSB,k * Re int_0^{tmax} dt exp(-i (w - w1,k) * 2*pi*c_0 * t - g_k(t))
    """
    freq_arr = np.asarray(freq_axis, dtype=float)
    norm_b = normalize_bands(bands)
    if len(norm_b) == 0:
        return np.zeros_like(freq_arr)

    n_steps = int(nt) if (nt is not None and int(nt) >= 16) else 256
    t_ftir = np.linspace(0.0, float(tmax), n_steps)
    dt_ftir = float(t_ftir[1] - t_ftir[0]) if n_steps > 1 else 0.05

    ffcf_groups = normalize_ffcf_groups(decays)
    n_groups = len(ffcf_groups)

    # Precompute exp(-g(t)) for each FFCF group
    exp_neg_g = {}
    for g_idx, grp in enumerate(ffcf_groups):
        g_t = evaluate_ffcf_gt(t_ftir, grp, fid_tuple=None)
        g_t_clipped = np.clip(g_t, 0.0, 700.0)
        fid_env = np.exp(-g_t_clipped)
        fid_env[0] *= 0.5  # Half-weight at t=0 for trapezoidal integration
        exp_neg_g[g_idx] = fid_env

    ftir_sim = np.zeros_like(freq_arr, dtype=float)
    for b in norm_b:
        # Cross-peaks (w1 != w3) do not contribute to 1st-order linear absorption
        if abs(b["w1"] - b["w3"]) > 5.0 and b["ffcf_id"] != b["ffcf_id_w3"]:
            continue
        g_idx = min(max(0, b["ffcf_id"]), n_groups - 1)
        env = exp_neg_g[g_idx]
        w_shift = freq_arr - b["w1"]
        phase = np.cos(w_shift[:, None] * (2.0 * pi * c_0) * t_ftir[None, :])
        band_profile = np.dot(phase, env) * dt_ftir
        ftir_sim += max(0.0, abs(b["amp_gsb"])) * band_profile

    if normalize:
        s_max = float(np.max(np.abs(ftir_sim)))
        if s_max > 1e-12:
            ftir_sim = ftir_sim / s_max

    return ftir_sim


# --------------------------------------------------------------------------- #
#              3. Parameter Unpacking, Linking & Vector Mapping               #
# --------------------------------------------------------------------------- #

def unpack_kubo_params(
    params,
    num_bands: int,
    num_decays: int,
    has_dw0: bool = False,
    ffcf_structure: list[int] | None = None,
):
    """Unpack flat parameter vector into `(bands, decays)` lists.

    Supports:
    - Standard legacy layout: 4 * num_bands + 3 * num_decays
    - Extended layout with shared dT2, dw_0 + 2 * num_decays: 4 * num_bands + 2 + 2 * num_decays
    - Multi-FFCF layout with `ffcf_structure = [n_terms_grp0, n_terms_grp1, ...]`:
      4 * num_bands + sum_g (2 + 2 * n_terms_g)
    """
    params = np.asarray(params, dtype=float)
    bands = []
    for i in range(num_bands):
        idx = 4 * i
        w01 = float(params[idx])
        dw = float(params[idx + 1])
        a_gsb = float(params[idx + 2])
        a_esa = float(params[idx + 3])
        bands.append((w01, dw, a_gsb, a_esa))

    rem = len(params) - 4 * num_bands
    decays = []
    if ffcf_structure is not None and len(ffcf_structure) > 0:
        ptr = 4 * num_bands
        for n_terms in ffcf_structure:
            dw_T2 = float(params[ptr])
            dw_0 = float(params[ptr + 1])
            ptr += 2
            terms = []
            for _ in range(int(n_terms)):
                terms.append((float(params[ptr]), float(params[ptr + 1])))
                ptr += 2
            decays.append({"dw_T2": dw_T2, "dw_0": dw_0, "terms": terms})
    elif has_dw0 or (rem == 2 + 2 * num_decays and rem != 3 * num_decays):
        base = 4 * num_bands
        dw_T2 = float(params[base])
        dw_0 = float(params[base + 1])
        terms = []
        for j in range(num_decays):
            t_idx = base + 2 + 2 * j
            terms.append((float(params[t_idx]), float(params[t_idx + 1])))
        decays = [{"dw_T2": dw_T2, "dw_0": dw_0, "terms": terms}]
    else:
        for j in range(num_decays):
            idx = 4 * num_bands + 3 * j
            dw_1 = float(params[idx])
            tau_c = float(params[idx + 1])
            dw_T2 = float(params[idx + 2])
            decays.append((dw_1, tau_c, dw_T2))

    return bands, decays


def compute_jacobian(
    params,
    bands_num,
    decays_num,
    pump_axis,
    probe_axis,
    t2,
    tmax,
    nt,
    w_rot,
    undersampling_factor,
    actual_fid,
):
    """Compute finite-difference Jacobian of the normalized simulated 2D spectrum wrt `params`."""
    params = np.asarray(params, dtype=float)
    n_params = len(params)
    eps = 1e-5

    def get_scaled_sim(p):
        if len(p) == 6:
            w01, dw, dw_0, dw_1, tau_c, dw_T2 = p
            z = simulate_2d_spectrum(
                w01, dw, dw_0, dw_1, tau_c, dw_T2,
                pump_axis, probe_axis, t2,
                tmax, nt, w_rot,
                undersampling_factor, actual_fid,
            )
        else:
            bands_opt, decays_opt = unpack_kubo_params(p, bands_num, decays_num)
            z = simulate_2d_spectrum(
                bands_opt, decays_opt, pump_axis, probe_axis, t2,
                tmax=tmax, nt=nt, w_rot=w_rot,
                undersampling_factor=undersampling_factor, fid_data=actual_fid,
            )
        z_max = float(np.max(np.abs(z)))
        if z_max > 0:
            z = z / z_max
        return z

    z_base = get_scaled_sim(params)
    M_points = len(pump_axis) * len(probe_axis)
    J = np.zeros((M_points, n_params), dtype=float)

    for j in range(n_params):
        p_perturbed = np.array(params, dtype=float)
        h = eps * max(abs(float(params[j])), 1.0)
        p_perturbed[j] += h
        z_perturbed = get_scaled_sim(p_perturbed)
        J[:, j] = ((z_perturbed - z_base) / h).ravel()

    return J


# --------------------------------------------------------------------------- #
#              4. Multi-Core Parallel Execution & Jacobian Helpers            #
# --------------------------------------------------------------------------- #

def resolve_parallel_config(parallel_mode: str | None = None, max_workers: int | None = None) -> tuple[str, int]:
    """Resolve `(mode, workers)` from arguments or PyMORGAN settings."""
    if parallel_mode is None:
        try:
            from ..settings import get_settings
            s = get_settings()
            pmode = getattr(s, "parallel_fitting", "processpool")
            parallel_mode = pmode.value if hasattr(pmode, "value") else str(pmode)
        except Exception:
            parallel_mode = "threadpool"

    mode = str(parallel_mode).lower().strip()
    if mode not in ("processpool", "threadpool", "disabled"):
        mode = "threadpool"

    cpu_cnt = os.cpu_count() or 4
    workers = int(max_workers) if (max_workers is not None and int(max_workers) >= 1) else cpu_cnt
    workers = max(1, min(workers, cpu_cnt * 2))
    if workers == 1:
        mode = "disabled"
    return mode, workers


def _make_parallel_jacobian(residual_fn, bounds_lower: np.ndarray, bounds_upper: np.ndarray, executor, eps: float = 1e-5):
    """Build a multi-core finite-difference Jacobian callable for `scipy.optimize.least_squares`.

    Evaluates all P perturbed parameter columns concurrently across the worker pool.
    """
    lb = np.asarray(bounds_lower, dtype=float)
    ub = np.asarray(bounds_upper, dtype=float)

    def jac_callable(x, *args, **kwargs):
        x_arr = np.asarray(x, dtype=float)
        f0 = residual_fn(x_arr)
        n_p = len(x_arr)
        if n_p == 0:
            return np.zeros((len(f0), 0), dtype=float)

        perturbed_xs = []
        steps = np.zeros(n_p, dtype=float)
        for j in range(n_p):
            span = ub[j] - lb[j]
            if span <= 1e-12:
                steps[j] = 0.0
                perturbed_xs.append(None)
                continue
            h = eps * max(abs(x_arr[j]), 1.0)
            if x_arr[j] + h > ub[j]:
                if x_arr[j] - h >= lb[j]:
                    h = -h
                else:
                    h = 0.5 * (ub[j] - x_arr[j])
            xp = x_arr.copy()
            xp[j] += h
            steps[j] = h
            perturbed_xs.append(xp)

        J = np.zeros((len(f0), n_p), dtype=float)
        active_indices = [j for j, xp in enumerate(perturbed_xs) if xp is not None and abs(steps[j]) > 1e-15]
        if not active_indices:
            return J

        if executor is not None and len(active_indices) > 1:
            futures = {executor.submit(residual_fn, perturbed_xs[j]): j for j in active_indices}
            for fut in concurrent.futures.as_completed(futures):
                j = futures[fut]
                f_pert = fut.result()
                J[:, j] = (f_pert - f0) / steps[j]
        else:
            for j in active_indices:
                f_pert = residual_fn(perturbed_xs[j])
                J[:, j] = (f_pert - f0) / steps[j]

        return J

    return jac_callable


def _compute_fit_statistics(
    residuals_vec: np.ndarray,
    Z_exp_norm: np.ndarray,
    Z_sim_norm: np.ndarray,
    J: np.ndarray,
    params_full: np.ndarray,
    cost_history: list[float],
) -> dict[str, Any]:
    """Compute covariance, standard errors, 95% CI, R^2, RMSE, reduced chi^2, and Robben SIGN metric."""
    M = max(1, Z_exp_norm.size)
    N = len(params_full)
    dof = max(1, M - N)

    cost_val = float(np.sum(residuals_vec ** 2))
    mse = cost_val / float(dof)
    rmse = float(np.sqrt(np.mean((Z_exp_norm - Z_sim_norm) ** 2)))

    ss_tot = float(np.sum((Z_exp_norm - np.mean(Z_exp_norm)) ** 2))
    ss_res = float(np.sum((Z_exp_norm - Z_sim_norm) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    # Variance of edge corner pixels as noise estimate for reduced chi^2
    corner_h = max(2, Z_exp_norm.shape[0] // 6)
    corner_w = max(2, Z_exp_norm.shape[1] // 6)
    corners = np.concatenate([
        Z_exp_norm[:corner_h, :corner_w].ravel(),
        Z_exp_norm[-corner_h:, -corner_w:].ravel(),
    ])
    noise_var = float(np.var(corners)) if corners.size > 2 else mse
    noise_var = max(noise_var, 1e-5)
    reduced_chi2 = float(mse / noise_var)

    C_prev = cost_history[-2] if len(cost_history) >= 2 else cost_val
    if C_prev <= 1e-12:
        C_prev = max(cost_val, 1e-12)

    try:
        # Gradient of SSR C = sum(r^2) is 2 * J^T * r
        r_2d = (Z_sim_norm - Z_exp_norm).ravel()
        if J.shape[0] == r_2d.size and J.shape[1] == N:
            grad = 2.0 * np.dot(J.T, r_2d)
            JTJ = np.dot(J.T, J)
            cov = np.linalg.pinv(JTJ + np.eye(N) * 1e-12) * mse
            sigma = np.sqrt(np.clip(np.diagonal(cov), 1e-16, None))
            sign_metrics = (grad * sigma) / C_prev
        else:
            sigma = np.zeros(N, dtype=float)
            sign_metrics = np.zeros(N, dtype=float)
    except Exception:
        sigma = np.zeros(N, dtype=float)
        sign_metrics = np.zeros(N, dtype=float)

    ci95 = 1.96 * sigma
    return {
        "cost": cost_val,
        "rmse": rmse,
        "r_squared": r_squared,
        "reduced_chi2": reduced_chi2,
        "errors": sigma,
        "ci95": ci95,
        "sign": sign_metrics,
    }


# --------------------------------------------------------------------------- #
#              5. Structured Model Config & Variable Projection               #
# --------------------------------------------------------------------------- #

def _solve_varpro_amplitudes(
    basis_maps: np.ndarray,
    Z_exp_norm: np.ndarray,
    amp_p0: np.ndarray,
    amp_lb: np.ndarray,
    amp_ub: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve for optimal pathway amplitudes `(A_GSB_1, A_ESA_1, ...)` given basis maps.

    Respects fixed amplitudes (`lb == ub`) and box bounds via `scipy.optimize.lsq_linear`.
    """
    n_basis = basis_maps.shape[0]
    if n_basis == 0:
        return np.zeros(0, dtype=float), np.zeros_like(Z_exp_norm)

    B_mat = basis_maps.reshape(n_basis, -1).T  # shape (M, 2K)
    y_vec = Z_exp_norm.ravel()

    fixed_mask = np.isclose(amp_lb, amp_ub, atol=1e-12)
    amps = np.array(amp_p0, dtype=float)

    if np.all(fixed_mask):
        amps = np.array(amp_lb, dtype=float)
    else:
        y_rem = y_vec.copy()
        if np.any(fixed_mask):
            amps[fixed_mask] = amp_lb[fixed_mask]
            y_rem = y_rem - np.dot(B_mat[:, fixed_mask], amps[fixed_mask])

        free_idx = np.nonzero(~fixed_mask)[0]
        B_free = B_mat[:, free_idx]
        lb_f = amp_lb[free_idx]
        ub_f = amp_ub[free_idx]

        # Fast unconstrained QR solve first; fall back to bounded lsq_linear if outside bounds
        try:
            a_ls, _, _, _ = np.linalg.lstsq(B_free, y_rem, rcond=None)
            if np.all((a_ls >= lb_f - 1e-9) & (a_ls <= ub_f + 1e-9)):
                amps[free_idx] = np.clip(a_ls, lb_f, ub_f)
            else:
                res_lin = lsq_linear(B_free, y_rem, bounds=(lb_f, ub_f))
                amps[free_idx] = res_lin.x
        except Exception:
            amps[free_idx] = np.clip(amp_p0[free_idx], lb_f, ub_f)

    Z_sim_norm = np.tensordot(amps, basis_maps, axes=(0, 0))
    return amps, Z_sim_norm


def _extract_first_kubo_pair(
    params,
    num_bands: int,
    num_decays: int,
    has_dw0: bool = False,
    ffcf_structure: list[int] | None = None,
) -> tuple[float, float]:
    """Return `(dw_1, tau_c)` from a flat parameter vector for trajectory tracking."""
    if len(params) == 6:
        return float(params[3]), float(params[4])
    rem = len(params) - 4 * num_bands
    base = 4 * num_bands
    if ffcf_structure or has_dw0 or (rem == 2 + 2 * num_decays and rem != 3 * num_decays):
        if len(params) >= base + 4:
            return float(params[base + 2]), float(params[base + 3])
    elif len(params) >= base + 2:
        return float(params[base]), float(params[base + 1])
    return 0.0, 0.0


# --------------------------------------------------------------------------- #
#              6. Single-Delay & Multi-Delay Kubo Fit Workers                 #
# --------------------------------------------------------------------------- #

def fit_single_delay_worker(args):
    """Picklable top-level worker function to optimise Kubo parameters for a single t2 delay.

    Supports:
    - Legacy 6-tuple and 14-tuple arguments
    - Extended dictionary configuration with TRF + Variable Projection (`solver="trf_varpro"`),
      full-parameter TRF (`solver="trf"`), or legacy L-BFGS-B (`solver="lbfgsb"`),
      off-diagonal / linked bands, per-band FFCFs, static inhomogeneity `dw_0`,
      joint FTIR fitting, and multi-core Jacobian evaluation.
    """
    t_start = time.perf_counter()

    # Parse arguments (dict or legacy tuple)
    if isinstance(args, dict):
        Z_exp = args["Z_exp"]
        pump_axis = args["pump_axis"]
        probe_axis = args["probe_axis"]
        t2 = float(args["t2"])
        p0 = np.asarray(args["p0"], dtype=float)
        bounds = args["bounds"]
        tmax = float(args.get("tmax", 5.0))
        nt = args.get("nt", 64)
        w_rot = float(args.get("w_rot", 1900.0))
        undersampling_factor = float(args.get("undersampling_factor", 1.0))
        fid_data = args.get("fid_data", None)
        num_bands = int(args.get("num_bands", 1))
        num_decays = int(args.get("num_decays", 1))
        has_dw0 = bool(args.get("has_dw0", False))
        ffcf_structure = args.get("ffcf_structure", None)
        solver = str(args.get("solver", "trf_varpro")).lower()
        use_varpro = bool(args.get("use_varpro", "varpro" in solver))
        band_links = args.get("band_links", None)
        ffcf_Assignments = args.get("ffcf_assignments", None)
        ftir_weight = float(args.get("ftir_weight", 1.0))
        jac_workers = int(args.get("jac_workers", 1))
        iteration_callback = args.get("iteration_callback", None)
    elif len(args) == 14:
        (
            Z_exp, pump_axis, probe_axis, t2, p0, bounds, tmax, nt, w_rot,
            undersampling_factor, fid_data, num_bands, num_decays,
        ) = args[:-1]
        iteration_callback = args[-1]
        p0 = np.asarray(p0, dtype=float)
        has_dw0 = False
        ffcf_structure = None
        solver = "trf_varpro"
        use_varpro = True
        band_links = None
        ffcf_Assignments = None
        ftir_weight = 1.0
        jac_workers = 1
    else:
        Z_exp, pump_axis, probe_axis, t2, p0, bounds = args
        p0 = np.asarray(p0, dtype=float)
        tmax = 5.0
        nt = 64
        w_rot = 1900.0
        undersampling_factor = 1.0
        fid_data = None
        num_bands = 1
        num_decays = 1
        has_dw0 = False
        ffcf_structure = None
        solver = "trf_varpro"
        use_varpro = True
        band_links = None
        ffcf_Assignments = None
        ftir_weight = 1.0
        jac_workers = 1
        iteration_callback = None

    pump_axis = np.asarray(pump_axis, dtype=float)
    probe_axis = np.asarray(probe_axis, dtype=float)
    Z_exp = np.asarray(Z_exp, dtype=float)

    exp_max = float(np.max(np.abs(Z_exp)))
    Z_exp_norm = Z_exp / exp_max if exp_max > 0.0 else Z_exp.copy()

    is_legacy_6 = (len(p0) == 6)
    if is_legacy_6:
        use_varpro = False

    lb_full = np.array([b[0] for b in bounds], dtype=float)
    ub_full = np.array([b[1] for b in bounds], dtype=float)
    p0_clamped = np.clip(p0, lb_full, ub_full)

    # Check if joint FTIR fitting is active
    ftir_target_y = None
    if isinstance(fid_data, dict) and str(fid_data.get("mode", "empirical_fid")).lower() == "joint_fit":
        spec = fid_data.get("spectrum")
        if spec is not None and len(spec) == 2:
            w_ftir, y_ftir = spec
            ftir_target_y = np.interp(pump_axis, w_ftir, y_ftir, left=0.0, right=0.0)
            f_max = float(np.max(np.abs(ftir_target_y)))
            if f_max > 1e-12:
                ftir_target_y = ftir_target_y / f_max

    history: list[tuple[float, float]] = []
    cost_history: list[float] = []

    def _build_structured_bands_and_decays(full_vec: np.ndarray):
        """Construct structured bands (with w3/link resolution) and decays from full parameter vector."""
        if is_legacy_6:
            w01, dw, dw_0, dw_1, tau_c, dw_T2 = full_vec
            b_list = [{"w1": float(w01), "w3": float(w01), "anharm": float(dw), "amp_gsb": 1.0, "amp_esa": 1.0, "ffcf_id": 0, "ffcf_id_w3": 0, "corr_eta": 1.0}]
            d_list = [{"dw_T2": float(dw_T2), "dw_0": float(dw_0), "terms": [(float(dw_1), float(tau_c))]}]
            return b_list, d_list

        raw_bands, d_list = unpack_kubo_params(
            full_vec, num_bands, num_decays, has_dw0=has_dw0, ffcf_structure=ffcf_structure
        )
        b_dicts = []
        for i, (w01, dw, a_gsb, a_esa) in enumerate(raw_bands):
            link_cfg = band_links[i] if (band_links is not None and i < len(band_links)) else {}
            w3_val = float(link_cfg.get("w3", w01)) if link_cfg.get("link_w3") != "diagonal" else float(w01)
            ffcf_id = int(ffcf_Assignments[i]) if (ffcf_Assignments is not None and i < len(ffcf_Assignments)) else int(link_cfg.get("ffcf_id", 0))
            ffcf_id_w3 = int(link_cfg.get("ffcf_id_w3", ffcf_id))
            corr_eta = float(link_cfg.get("corr_eta", 1.0 if abs(w01 - w3_val) < 1e-6 else 0.5))
            b_dicts.append({
                "w1": float(w01),
                "w3": w3_val,
                "anharm": float(dw),
                "amp_gsb": float(a_gsb),
                "amp_esa": float(a_esa),
                "ffcf_id": ffcf_id,
                "ffcf_id_w3": ffcf_id_w3,
                "corr_eta": corr_eta,
                "link_w1": link_cfg.get("link_w1"),
                "link_w3": link_cfg.get("link_w3", "diagonal"),
                "link_anharm": link_cfg.get("link_anharm"),
            })

        # Resolve inter-band parameter links (link_w1, link_w3, link_anharm)
        for i, bd in enumerate(b_dicts):
            lw1 = bd.get("link_w1")
            if isinstance(lw1, int) and 0 <= lw1 < num_bands and lw1 != i:
                bd["w1"] = b_dicts[lw1]["w1"]
            lw3 = bd.get("link_w3")
            if lw3 == "diagonal" or lw3 is None:
                if lw3 == "diagonal":
                    bd["w3"] = bd["w1"]
            elif isinstance(lw3, int) and 0 <= lw3 < num_bands:
                bd["w3"] = b_dicts[lw3]["w1"] if lw3 != i else bd["w1"]
            lanh = bd.get("link_anharm")
            if isinstance(lanh, int) and 0 <= lanh < num_bands and lanh != i:
                bd["anharm"] = b_dicts[lanh]["anharm"]

        return b_dicts, d_list

    # Identify linear amplitude indices vs nonlinear indices when VarPro is enabled
    amp_indices = []
    if not is_legacy_6:
        for i in range(num_bands):
            amp_indices.extend([4 * i + 2, 4 * i + 3])
    amp_indices_arr = np.array(amp_indices, dtype=int)
    nl_mask = np.ones(len(p0_clamped), dtype=bool)
    if use_varpro and len(amp_indices_arr) > 0:
        nl_mask[amp_indices_arr] = False
    nl_indices = np.nonzero(nl_mask)[0]

    # Maintain latest full parameter vector
    best_full_params = p0_clamped.copy()

    def _evaluate_full_residual(nl_or_full_vec: np.ndarray, record_step: bool = True) -> np.ndarray:
        nonlocal best_full_params
        full_vec = best_full_params.copy()
        if use_varpro and len(nl_indices) < len(full_vec):
            full_vec[nl_indices] = nl_or_full_vec
        else:
            full_vec = np.array(nl_or_full_vec, dtype=float)

        b_dicts, d_list = _build_structured_bands_and_decays(full_vec)

        if use_varpro and not is_legacy_6:
            basis_maps, _ = simulate_2d_basis_maps(
                b_dicts, d_list, pump_axis, probe_axis, t2_delay=t2,
                tmax=tmax, nt=nt, w_rot=w_rot,
                undersampling_factor=undersampling_factor, fid_data=fid_data,
                normalize_unit_basis=True,
            )
            amp_p0 = full_vec[amp_indices_arr]
            amp_lb = lb_full[amp_indices_arr]
            amp_ub = ub_full[amp_indices_arr]
            amps_opt, Z_sim_norm = _solve_varpro_amplitudes(basis_maps, Z_exp_norm, amp_p0, amp_lb, amp_ub)
            full_vec[amp_indices_arr] = amps_opt
            for i in range(num_bands):
                b_dicts[i]["amp_gsb"] = float(amps_opt[2 * i])
                b_dicts[i]["amp_esa"] = float(amps_opt[2 * i + 1])
        else:
            Z_sim = simulate_2d_spectrum(
                b_dicts, d_list, pump_axis, probe_axis, t2_delay=t2,
                tmax=tmax, nt=nt, w_rot=w_rot,
                undersampling_factor=undersampling_factor, fid_data=fid_data,
                normalize_unit_basis=(not is_legacy_6),
            )
            if is_legacy_6:
                s_max = float(np.max(np.abs(Z_sim)))
                if s_max > 0.0:
                    Z_sim = Z_sim / s_max
                scale = float(np.maximum(0.0, np.sum(Z_exp_norm * Z_sim) / (np.sum(Z_sim ** 2) + 1e-12)))
                Z_sim_norm = scale * Z_sim
            else:
                Z_sim_norm = Z_sim

        res_2d = (Z_sim_norm - Z_exp_norm).ravel()

        if ftir_target_y is not None and ftir_weight > 0.0:
            ftir_sim = simulate_1d_ftir_spectrum(b_dicts, d_list, pump_axis, tmax=tmax, nt=nt or 256, normalize=True)
            scale_f = float(np.maximum(0.0, np.sum(ftir_target_y * ftir_sim) / (np.sum(ftir_sim ** 2) + 1e-12)))
            # Scale FTIR residual weight relative to 2D pixel count
            w_factor = np.sqrt(ftir_weight * (res_2d.size / max(1, ftir_target_y.size)))
            res_ftir = w_factor * (scale_f * ftir_sim - ftir_target_y)
            res_total = np.concatenate([res_2d, res_ftir])
        else:
            res_total = res_2d

        if record_step:
            best_full_params = full_vec.copy()
            c_val = float(np.sum(res_total ** 2))
            cost_history.append(c_val)
            pair = _extract_first_kubo_pair(
                full_vec, num_bands, num_decays, has_dw0=has_dw0, ffcf_structure=ffcf_structure
            )
            if not history or (abs(history[-1][0] - pair[0]) > 1e-5 or abs(history[-1][1] - pair[1]) > 1e-5):
                history.append(pair)
            if iteration_callback is not None:
                try:
                    iteration_callback(full_vec)
                except Exception:
                    pass

        return res_total

    # Run optimization
    if solver == "lbfgsb":
        def scalar_cost(x_in):
            r = _evaluate_full_residual(x_in, record_step=True)
            return float(np.sum(r ** 2))

        x0_opt = p0_clamped[nl_indices] if use_varpro else p0_clamped
        bounds_opt = [bounds[i] for i in nl_indices] if use_varpro else bounds
        res_opt = minimize(scalar_cost, x0_opt, bounds=bounds_opt, method="L-BFGS-B")
        _evaluate_full_residual(res_opt.x, record_step=True)
        success = bool(res_opt.success)
        nfev = int(getattr(res_opt, "nfev", len(cost_history)))
    else:
        x0_opt = p0_clamped[nl_indices] if use_varpro else p0_clamped
        lb_opt = lb_full[nl_indices] if use_varpro else lb_full
        ub_opt = ub_full[nl_indices] if use_varpro else ub_full

        # Ensure strict lb < ub for least_squares by fixing equal bounds via small epsilon or mask
        free_opt_mask = (ub_opt - lb_opt) > 1e-11
        if not np.any(free_opt_mask):
            _evaluate_full_residual(x0_opt, record_step=True)
            success = True
            nfev = 1
        else:
            free_sub_idx = np.nonzero(free_opt_mask)[0]
            x0_free = np.clip(x0_opt[free_sub_idx], lb_opt[free_sub_idx], ub_opt[free_sub_idx])
            lb_free = lb_opt[free_sub_idx]
            ub_free = ub_opt[free_sub_idx]

            def res_free(x_f):
                x_full_nl = x0_opt.copy()
                x_full_nl[free_sub_idx] = x_f
                return _evaluate_full_residual(x_full_nl, record_step=True)

            def res_free_no_record(x_f):
                x_full_nl = x0_opt.copy()
                x_full_nl[free_sub_idx] = x_f
                return _evaluate_full_residual(x_full_nl, record_step=False)

            if jac_workers > 1 and len(free_sub_idx) > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=jac_workers) as jac_pool:
                    jac_fn = _make_parallel_jacobian(res_free_no_record, lb_free, ub_free, jac_pool)
                    res_ls = least_squares(
                        res_free,
                        x0_free,
                        jac=jac_fn,
                        bounds=(lb_free, ub_free),
                        method="trf",
                        x_scale="jac",
                        ftol=1e-5,
                        xtol=1e-5,
                        gtol=1e-5,
                        max_nfev=80 * max(1, len(free_sub_idx)),
                    )
            else:
                res_ls = least_squares(
                    res_free,
                    x0_free,
                    bounds=(lb_free, ub_free),
                    method="trf",
                    x_scale="jac",
                    ftol=1e-5,
                    xtol=1e-5,
                    gtol=1e-5,
                    max_nfev=80 * max(1, len(free_sub_idx)),
                )

            x_final_nl = x0_opt.copy()
            x_final_nl[free_sub_idx] = res_ls.x
            _evaluate_full_residual(x_final_nl, record_step=True)
            success = bool(res_ls.success or res_ls.status > 0)
            nfev = int(res_ls.nfev)

    # Final evaluation & full Jacobian for parameter errors & Robben SIGN metric
    b_best, d_best = _build_structured_bands_and_decays(best_full_params)
    if is_legacy_6:
        Z_best = simulate_2d_spectrum(
            b_best, d_best, pump_axis, probe_axis, t2_delay=t2,
            tmax=tmax, nt=nt, w_rot=w_rot,
            undersampling_factor=undersampling_factor, fid_data=fid_data,
            normalize_unit_basis=False,
        )
        s_max = float(np.max(np.abs(Z_best)))
        if s_max > 0.0:
            Z_best = Z_best / s_max
        scale = float(np.maximum(0.0, np.sum(Z_exp_norm * Z_best) / (np.sum(Z_best ** 2) + 1e-12)))
        Z_best_norm = scale * Z_best
        unit_scale = s_max / max(scale, 1e-12)
    else:
        basis_maps, unit_scale = simulate_2d_basis_maps(
            b_best, d_best, pump_axis, probe_axis, t2_delay=t2,
            tmax=tmax, nt=nt, w_rot=w_rot,
            undersampling_factor=undersampling_factor, fid_data=fid_data,
            normalize_unit_basis=True,
        )
        weights = best_full_params[amp_indices_arr]
        Z_best_norm = np.tensordot(weights, basis_maps, axes=(0, 0))

    Z_sim_restored = Z_best_norm * exp_max

    # Compute full Jacobian wrt all parameters (nonlinear + linear) at solution
    N_all = len(best_full_params)
    M_pts = Z_exp_norm.size
    J_full = np.zeros((M_pts, N_all), dtype=float)
    if not is_legacy_6 and len(amp_indices_arr) == 2 * num_bands:
        # Exact analytical Jacobian columns for linear pathway amplitudes
        for k_amp, p_idx in enumerate(amp_indices_arr):
            if abs(ub_full[p_idx] - lb_full[p_idx]) > 1e-11:
                J_full[:, p_idx] = basis_maps[k_amp].ravel()

    # Finite-difference columns for nonlinear parameters
    def _sim_norm_for_full_p(p_vec):
        b_tmp, d_tmp = _build_structured_bands_and_decays(p_vec)
        z_tmp = simulate_2d_spectrum(
            b_tmp, d_tmp, pump_axis, probe_axis, t2_delay=t2,
            tmax=tmax, nt=nt, w_rot=w_rot,
            undersampling_factor=undersampling_factor, fid_data=fid_data,
            normalize_unit_basis=(not is_legacy_6),
        )
        if is_legacy_6:
            sm = float(np.max(np.abs(z_tmp)))
            if sm > 0.0:
                z_tmp = z_tmp / sm
            sc = float(np.maximum(0.0, np.sum(Z_exp_norm * z_tmp) / (np.sum(z_tmp ** 2) + 1e-12)))
            return (sc * z_tmp).ravel()
        return z_tmp.ravel()

    z0_flat = Z_best_norm.ravel()
    nl_to_diff = range(N_all) if is_legacy_6 else nl_indices
    for j in nl_to_diff:
        if abs(ub_full[j] - lb_full[j]) <= 1e-11:
            continue
        h = 1e-5 * max(abs(best_full_params[j]), 1.0)
        p_pert = best_full_params.copy()
        p_pert[j] += h
        z_pert = _sim_norm_for_full_p(p_pert)
        J_full[:, j] = (z_pert - z0_flat) / h

    stats = _compute_fit_statistics(
        (Z_best_norm - Z_exp_norm).ravel(),
        Z_exp_norm,
        Z_best_norm,
        J_full,
        best_full_params,
        cost_history,
    )

    elapsed = time.perf_counter() - t_start
    return {
        "delay": t2,
        "params": best_full_params,
        "success": success,
        "cost": stats["cost"],
        "sign": stats["sign"],
        "errors": stats["errors"],
        "ci95": stats["ci95"],
        "r_squared": stats["r_squared"],
        "rmse": stats["rmse"],
        "reduced_chi2": stats["reduced_chi2"],
        "nfev": nfev,
        "elapsed_s": elapsed,
        "history": history,
        "unit_scale": unit_scale,
        "exp_max": exp_max,
        "Z_sim": Z_sim_restored,
        "Z_sim_norm": Z_best_norm,
        "Z_exp": Z_exp,
        "bands_fitted": b_best,
        "ffcf_fitted": d_best,
    }


def run_kubo_fit(
    data_cube,
    pump_axis,
    probe_axis,
    delays,
    p0,
    bounds,
    tmax=5.0,
    nt=64,
    w_rot=1900.0,
    undersampling_factor=1.0,
    fid_data=None,
    num_bands=1,
    num_decays=1,
    parallel_mode="threadpool",
    iteration_callback=None,
    *,
    has_dw0: bool = False,
    ffcf_structure: list[int] | None = None,
    solver: str = "trf_varpro",
    use_varpro: bool | None = None,
    band_links: list[dict] | None = None,
    ffcf_assignments: list[int] | None = None,
    ftir_weight: float = 1.0,
    max_workers: int | None = None,
    progress_callback=None,
    should_cancel=None,
):
    """Execute multi-core optimisation of Kubo parameters across all specified delays."""
    mode, workers = resolve_parallel_config(parallel_mode, max_workers)
    use_vp = ("varpro" in str(solver).lower()) if use_varpro is None else bool(use_varpro)

    n_delays = len(delays)
    # If fitting a single delay, give all CPU workers to the parallel Jacobian evaluator!
    jac_workers = workers if (n_delays == 1 and mode != "disabled") else 1

    tasks = []
    for idx, t2 in enumerate(delays):
        # Only pass iteration_callback when not running in ProcessPool
        ic = iteration_callback if (mode != "processpool" and n_delays == 1) else None
        tasks.append({
            "Z_exp": data_cube[:, :, idx],
            "pump_axis": pump_axis,
            "probe_axis": probe_axis,
            "t2": float(t2),
            "p0": p0,
            "bounds": bounds,
            "tmax": tmax,
            "nt": nt,
            "w_rot": w_rot,
            "undersampling_factor": undersampling_factor,
            "fid_data": fid_data,
            "num_bands": num_bands,
            "num_decays": num_decays,
            "has_dw0": has_dw0,
            "ffcf_structure": ffcf_structure,
            "solver": solver,
            "use_varpro": use_vp,
            "band_links": band_links,
            "ffcf_assignments": ffcf_assignments,
            "ftir_weight": ftir_weight,
            "jac_workers": jac_workers,
            "iteration_callback": ic,
        })

    results = [None] * n_delays
    if mode in ("threadpool", "processpool") and n_delays > 1:
        pool_cls = (
            concurrent.futures.ProcessPoolExecutor
            if mode == "processpool"
            else concurrent.futures.ThreadPoolExecutor
        )
        with pool_cls(max_workers=min(workers, n_delays)) as executor:
            futures = {executor.submit(fit_single_delay_worker, t): idx for idx, t in enumerate(tasks)}
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                if should_cancel is not None and should_cancel():
                    for f in futures:
                        f.cancel()
                    raise InterruptedError("Kubo fitting cancelled by user.")
                orig_idx = futures[future]
                res_item = future.result()
                results[orig_idx] = res_item
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, n_delays, res_item)
    else:
        for idx, t in enumerate(tasks):
            if should_cancel is not None and should_cancel():
                raise InterruptedError("Kubo fitting cancelled by user.")
            res_item = fit_single_delay_worker(t)
            results[idx] = res_item
            if progress_callback is not None:
                progress_callback(idx + 1, n_delays, res_item)

    logger.info(
        "\n%s\nDirect Kubo Model Fitting Completed (%d delay(s), mode=%s, workers=%d).\nMethod Reference:\n"
        "  Robben, K. C.; Cheatum, C. M. Least-Squares Fitting of Multidimensional Spectra\n"
        "  to Kubo Line-Shape Models. J. Phys. Chem. B 2021, 125, 46, 12876-12891.\n"
        "  DOI: 10.1021/acs.jpcb.1c08764\n%s\n",
        "=" * 80,
        n_delays,
        mode,
        workers,
        "=" * 80,
    )

    return results


# --------------------------------------------------------------------------- #
#              7. Global Multi-Delay Kubo Fit (Multi-Core + VarPro)           #
# --------------------------------------------------------------------------- #

def run_kubo_global_fit(
    data_cube: np.ndarray,
    pump_axis: np.ndarray,
    probe_axis: np.ndarray,
    delays: np.ndarray | list[float],
    p0: list[float] | np.ndarray,
    bounds: list[tuple[float, float]],
    *,
    tmax: float = 5.0,
    nt: int | None = 64,
    w_rot: float = 1900.0,
    undersampling_factor: float = 1.0,
    fid_data=None,
    num_bands: int = 1,
    num_decays: int = 1,
    has_dw0: bool = False,
    ffcf_structure: list[int] | None = None,
    solver: str = "trf_varpro",
    use_varpro: bool | None = None,
    band_links: list[dict] | None = None,
    ffcf_assignments: list[int] | None = None,
    ftir_weight: float = 1.0,
    parallel_mode: str | None = None,
    max_workers: int | None = None,
    iteration_callback=None,
    progress_callback=None,
    should_cancel=None,
) -> list[dict[str, Any]]:
    """Global Kubo model optimization across all population delays t2.

    Shared parameters across all t2 delays:
        - Band positions (w1, w3) and anharmonicities (Delta)
        - FFCF line-broadening parameters (dT2, dw_0, dw_m, tau_c,m)
    Per-delay free parameters:
        - Pathway amplitudes A_GSB(t2), A_ESA(t2) for each band.

    When Variable Projection (`use_varpro=True`, default) is enabled, the 2*K*Nt2
    per-delay linear pathway amplitudes are projected out analytically at every
    step across all delays in parallel, reducing the nonlinear parameter dimension
    from O(K * Nt2) to O(K + M) and accelerating convergence by up to 10x.
    """
    t_start = time.perf_counter()
    mode, workers = resolve_parallel_config(parallel_mode, max_workers)
    use_vp = ("varpro" in str(solver).lower()) if use_varpro is None else bool(use_varpro)

    pump_arr = np.asarray(pump_axis, dtype=float)
    probe_arr = np.asarray(probe_axis, dtype=float)
    delays_arr = np.asarray(delays, dtype=float)
    n_delays = len(delays_arr)
    p0_arr = np.asarray(p0, dtype=float)
    lb_arr = np.array([b[0] for b in bounds], dtype=float)
    ub_arr = np.array([b[1] for b in bounds], dtype=float)
    p0_clamped = np.clip(p0_arr, lb_arr, ub_arr)

    # Pre-normalize experimental slices
    exp_max_per_delay = np.zeros(n_delays, dtype=float)
    Z_exp_norm_cube = np.zeros_like(data_cube, dtype=float)
    for idx in range(n_delays):
        em = float(np.max(np.abs(data_cube[:, :, idx])))
        exp_max_per_delay[idx] = em if em > 0.0 else 1.0
        Z_exp_norm_cube[:, :, idx] = data_cube[:, :, idx] / exp_max_per_delay[idx]

    # Pre-crop FTIR target if joint fit is enabled
    ftir_target_y = None
    if isinstance(fid_data, dict) and str(fid_data.get("mode", "empirical_fid")).lower() == "joint_fit":
        spec = fid_data.get("spectrum")
        if spec is not None and len(spec) == 2:
            w_ftir, y_ftir = spec
            ftir_target_y = np.interp(pump_arr, w_ftir, y_ftir, left=0.0, right=0.0)
            f_max = float(np.max(np.abs(ftir_target_y)))
            if f_max > 1e-12:
                ftir_target_y = ftir_target_y / f_max

    # Identify shared nonlinear parameter indices in p0_clamped
    amp_indices = []
    for i in range(num_bands):
        amp_indices.extend([4 * i + 2, 4 * i + 3])
    amp_indices_arr = np.array(amp_indices, dtype=int)
    nl_mask = np.ones(len(p0_clamped), dtype=bool)
    nl_mask[amp_indices_arr] = False
    nl_indices = np.nonzero(nl_mask)[0]

    # Store per-delay optimal amplitudes [n_delays, 2 * num_bands]
    amp_p0_template = p0_clamped[amp_indices_arr]
    amp_lb_template = lb_arr[amp_indices_arr]
    amp_ub_template = ub_arr[amp_indices_arr]
    per_delay_amps = np.tile(amp_p0_template, (n_delays, 1))

    history: list[tuple[float, float]] = []
    cost_history: list[float] = []
    eval_count = 0

    def _unpack_shared_nl(nl_vec: np.ndarray):
        full_template = p0_clamped.copy()
        full_template[nl_indices] = nl_vec
        raw_bands, d_list = unpack_kubo_params(
            full_template, num_bands, num_decays, has_dw0=has_dw0, ffcf_structure=ffcf_structure
        )
        b_dicts = []
        for i, (w01, dw, a_gsb, a_esa) in enumerate(raw_bands):
            link_cfg = band_links[i] if (band_links is not None and i < len(band_links)) else {}
            w3_val = float(link_cfg.get("w3", w01)) if link_cfg.get("link_w3") != "diagonal" else float(w01)
            ffcf_id = int(ffcf_assignments[i]) if (ffcf_assignments is not None and i < len(ffcf_assignments)) else int(link_cfg.get("ffcf_id", 0))
            ffcf_id_w3 = int(link_cfg.get("ffcf_id_w3", ffcf_id))
            corr_eta = float(link_cfg.get("corr_eta", 1.0 if abs(w01 - w3_val) < 1e-6 else 0.5))
            b_dicts.append({
                "w1": float(w01),
                "w3": w3_val,
                "anharm": float(dw),
                "amp_gsb": float(a_gsb),
                "amp_esa": float(a_esa),
                "ffcf_id": ffcf_id,
                "ffcf_id_w3": ffcf_id_w3,
                "corr_eta": corr_eta,
                "link_w1": link_cfg.get("link_w1"),
                "link_w3": link_cfg.get("link_w3", "diagonal"),
                "link_anharm": link_cfg.get("link_anharm"),
            })
        for i, bd in enumerate(b_dicts):
            lw1 = bd.get("link_w1")
            if isinstance(lw1, int) and 0 <= lw1 < num_bands and lw1 != i:
                bd["w1"] = b_dicts[lw1]["w1"]
            lw3 = bd.get("link_w3")
            if lw3 == "diagonal" or lw3 is None:
                if lw3 == "diagonal":
                    bd["w3"] = bd["w1"]
            elif isinstance(lw3, int) and 0 <= lw3 < num_bands:
                bd["w3"] = b_dicts[lw3]["w1"] if lw3 != i else bd["w1"]
            lanh = bd.get("link_anharm")
            if isinstance(lanh, int) and 0 <= lanh < num_bands and lanh != i:
                bd["anharm"] = b_dicts[lanh]["anharm"]
        return full_template, b_dicts, d_list

    # Persistent ThreadPoolExecutor for parallel delay evaluation & parallel Jacobian
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers) if (mode != "disabled" and workers > 1) else None

    try:
        def _eval_single_delay_slice(args_in):
            idx_d, b_dicts, d_list = args_in
            t2_d = float(delays_arr[idx_d])
            basis_maps, _ = simulate_2d_basis_maps(
                b_dicts, d_list, pump_arr, probe_arr, t2_delay=t2_d,
                tmax=tmax, nt=nt, w_rot=w_rot,
                undersampling_factor=undersampling_factor, fid_data=fid_data,
                normalize_unit_basis=True,
            )
            amps_d, Z_sim_d = _solve_varpro_amplitudes(
                basis_maps, Z_exp_norm_cube[:, :, idx_d],
                per_delay_amps[idx_d], amp_lb_template, amp_ub_template,
            )
            res_d = (Z_sim_d - Z_exp_norm_cube[:, :, idx_d]).ravel()
            return idx_d, amps_d, res_d

        def _evaluate_global_residuals(nl_vec: np.ndarray, record_step: bool = True) -> np.ndarray:
            nonlocal eval_count
            if should_cancel is not None and should_cancel():
                raise InterruptedError("Global Kubo fit cancelled by user.")

            full_template, b_dicts, d_list = _unpack_shared_nl(nl_vec)
            slice_tasks = [(idx_d, b_dicts, d_list) for idx_d in range(n_delays)]

            if pool is not None and n_delays > 1 and record_step:
                slice_results = list(pool.map(_eval_single_delay_slice, slice_tasks))
            else:
                slice_results = [_eval_single_delay_slice(st) for st in slice_tasks]

            res_parts = [None] * n_delays
            for idx_d, amps_d, res_d in slice_results:
                res_parts[idx_d] = res_d
                if record_step:
                    per_delay_amps[idx_d] = amps_d

            if ftir_target_y is not None and ftir_weight > 0.0:
                b_ftir = copy.deepcopy(b_dicts)
                for i in range(num_bands):
                    b_ftir[i]["amp_gsb"] = float(per_delay_amps[0, 2 * i])
                ftir_sim = simulate_1d_ftir_spectrum(b_ftir, d_list, pump_arr, tmax=tmax, nt=nt or 256, normalize=True)
                scale_f = float(np.maximum(0.0, np.sum(ftir_target_y * ftir_sim) / (np.sum(ftir_sim ** 2) + 1e-12)))
                w_factor = np.sqrt(ftir_weight * (Z_exp_norm_cube.size / max(1, ftir_target_y.size)))
                res_ftir = w_factor * (scale_f * ftir_sim - ftir_target_y)
                res_parts.append(res_ftir)

            res_all = np.concatenate(res_parts)

            if record_step:
                eval_count += 1
                c_val = float(np.sum(res_all ** 2))
                cost_history.append(c_val)
                pair = _extract_first_kubo_pair(
                    full_template, num_bands, num_decays, has_dw0=has_dw0, ffcf_structure=ffcf_structure
                )
                if not history or (abs(history[-1][0] - pair[0]) > 1e-5 or abs(history[-1][1] - pair[1]) > 1e-5):
                    history.append(pair)
                if iteration_callback is not None:
                    full_cb = full_template.copy()
                    full_cb[amp_indices_arr] = per_delay_amps[0]
                    iteration_callback(full_cb)
                if progress_callback is not None:
                    progress_callback(min(eval_count, 50), 50, None)

            return res_all

        x0_nl = p0_clamped[nl_indices]
        lb_nl = lb_arr[nl_indices]
        ub_nl = ub_arr[nl_indices]

        free_nl_mask = (ub_nl - lb_nl) > 1e-11
        if not np.any(free_nl_mask):
            _evaluate_global_residuals(x0_nl, record_step=True)
            success = True
            nfev = 1
            best_nl = x0_nl.copy()
        else:
            free_sub = np.nonzero(free_nl_mask)[0]
            x0_free = np.clip(x0_nl[free_sub], lb_nl[free_sub], ub_nl[free_sub])
            lb_free = lb_nl[free_sub]
            ub_free = ub_nl[free_sub]

            def res_free(x_f):
                x_tmp = x0_nl.copy()
                x_tmp[free_sub] = x_f
                return _evaluate_global_residuals(x_tmp, record_step=True)

            def res_free_no_record(x_f):
                x_tmp = x0_nl.copy()
                x_tmp[free_sub] = x_f
                return _evaluate_global_residuals(x_tmp, record_step=False)

            jac_fn = _make_parallel_jacobian(res_free_no_record, lb_free, ub_free, pool) if pool is not None else "2-point"
            res_ls = least_squares(
                res_free,
                x0_free,
                jac=jac_fn,
                bounds=(lb_free, ub_free),
                method="trf",
                x_scale="jac",
                ftol=1e-4,
                xtol=1e-4,
                gtol=1e-4,
                max_nfev=60 * max(1, len(free_sub)),
            )
            best_nl = x0_nl.copy()
            best_nl[free_sub] = res_ls.x
            _evaluate_global_residuals(best_nl, record_step=True)
            success = bool(res_ls.success or res_ls.status > 0)
            nfev = int(res_ls.nfev)

        # Assemble per-delay result dictionaries
        full_shared, b_shared, d_shared = _unpack_shared_nl(best_nl)
        elapsed = time.perf_counter() - t_start
        results = []

        for idx_d, t2_d in enumerate(delays_arr):
            row_params = full_shared.copy()
            row_params[amp_indices_arr] = per_delay_amps[idx_d]

            b_d = copy.deepcopy(b_shared)
            for i in range(num_bands):
                b_d[i]["amp_gsb"] = float(per_delay_amps[idx_d, 2 * i])
                b_d[i]["amp_esa"] = float(per_delay_amps[idx_d, 2 * i + 1])

            basis_maps, unit_scale = simulate_2d_basis_maps(
                b_d, d_shared, pump_arr, probe_arr, t2_delay=float(t2_d),
                tmax=tmax, nt=nt, w_rot=w_rot,
                undersampling_factor=undersampling_factor, fid_data=fid_data,
                normalize_unit_basis=True,
            )
            Z_sim_norm = np.tensordot(per_delay_amps[idx_d], basis_maps, axes=(0, 0))
            Z_sim_restored = Z_sim_norm * exp_max_per_delay[idx_d]

            # Per-slice Jacobian for error & SIGN reporting
            N_all = len(row_params)
            M_pts = Z_exp_norm_cube[:, :, idx_d].size
            J_d = np.zeros((M_pts, N_all), dtype=float)
            for k_amp, p_idx in enumerate(amp_indices_arr):
                if abs(ub_arr[p_idx] - lb_arr[p_idx]) > 1e-11:
                    J_d[:, p_idx] = basis_maps[k_amp].ravel()

            z0_flat = Z_sim_norm.ravel()
            for j in nl_indices:
                if abs(ub_arr[j] - lb_arr[j]) <= 1e-11:
                    continue
                h = 1e-5 * max(abs(row_params[j]), 1.0)
                nl_pert = best_nl.copy()
                pos_in_nl = int(np.where(nl_indices == j)[0][0])
                nl_pert[pos_in_nl] += h
                _, b_p, d_p = _unpack_shared_nl(nl_pert)
                for i in range(num_bands):
                    b_p[i]["amp_gsb"] = float(per_delay_amps[idx_d, 2 * i])
                    b_p[i]["amp_esa"] = float(per_delay_amps[idx_d, 2 * i + 1])
                z_p = simulate_2d_spectrum(
                    b_p, d_p, pump_arr, probe_arr, t2_delay=float(t2_d),
                    tmax=tmax, nt=nt, w_rot=w_rot,
                    undersampling_factor=undersampling_factor, fid_data=fid_data,
                    normalize_unit_basis=True,
                ).ravel()
                J_d[:, j] = (z_p - z0_flat) / h

            stats = _compute_fit_statistics(
                (Z_sim_norm - Z_exp_norm_cube[:, :, idx_d]).ravel(),
                Z_exp_norm_cube[:, :, idx_d],
                Z_sim_norm,
                J_d,
                row_params,
                cost_history,
            )

            results.append({
                "delay": float(t2_d),
                "params": row_params,
                "success": success,
                "cost": stats["cost"],
                "sign": stats["sign"],
                "errors": stats["errors"],
                "ci95": stats["ci95"],
                "r_squared": stats["r_squared"],
                "rmse": stats["rmse"],
                "reduced_chi2": stats["reduced_chi2"],
                "nfev": nfev,
                "elapsed_s": elapsed,
                "history": history,
                "unit_scale": unit_scale,
                "exp_max": exp_max_per_delay[idx_d],
                "Z_sim": Z_sim_restored,
                "Z_sim_norm": Z_sim_norm,
                "Z_exp": data_cube[:, :, idx_d],
                "bands_fitted": b_d,
                "ffcf_fitted": d_shared,
                "global_fit": True,
            })

        return results
    finally:
        if pool is not None:
            pool.shutdown(wait=True)


# --------------------------------------------------------------------------- #
#              8. Parallel Full-Cube Reconstruction Helper                    #
# --------------------------------------------------------------------------- #

def _simulate_full_slice_worker(task_dict: dict[str, Any]) -> tuple[int, np.ndarray]:
    """Worker to simulate a single delay on the full dataset `(pump, probe)` axes."""
    idx = int(task_dict["idx"])
    t2 = float(task_dict["t2"])
    bands = task_dict["bands"]
    decays = task_dict["decays"]
    pump_full = task_dict["pump_full"]
    probe_full = task_dict["probe_full"]
    pump_roi = task_dict["pump_roi"]
    probe_roi = task_dict["probe_roi"]
    exp_max = float(task_dict["exp_max"])
    tmax = float(task_dict["tmax"])
    nt = task_dict["nt"]
    w_rot = float(task_dict["w_rot"])
    u = float(task_dict["u"])
    fid_data = task_dict["fid_data"]

    # Reference unit basis scale on ROI so full-map amplitude matches ROI fit identically
    _, roi_unit_scale = simulate_2d_basis_maps(
        bands, decays, pump_roi, probe_roi, t2_delay=t2,
        tmax=tmax, nt=nt, w_rot=w_rot,
        undersampling_factor=u, fid_data=fid_data,
        normalize_unit_basis=True,
    )

    Z_full_raw = simulate_2d_spectrum(
        bands, decays, pump_full, probe_full, t2_delay=t2,
        tmax=tmax, nt=nt, w_rot=w_rot,
        undersampling_factor=u, fid_data=fid_data,
        normalize_unit_basis=False,
    )
    Z_full_scaled = (Z_full_raw / max(roi_unit_scale, 1e-12)) * exp_max
    return idx, Z_full_scaled


def simulate_kubo_cube_parallel(
    delays: np.ndarray | list[float],
    kubo_details: dict[float, dict[str, Any]],
    pump_full: np.ndarray,
    probe_full: np.ndarray,
    pump_roi: np.ndarray,
    probe_roi: np.ndarray,
    Z_exp_roi_cube: np.ndarray,
    delay_indices: list[int],
    *,
    parallel_mode: str | None = None,
    max_workers: int | None = None,
    progress_callback=None,
    should_cancel=None,
) -> np.ndarray:
    """Reconstruct the full 3D simulated dataset cube across all delays in parallel."""
    mode, workers = resolve_parallel_config(parallel_mode, max_workers)
    n_pump = len(pump_full)
    n_probe = len(probe_full)
    n_delays = len(delays)
    Z_sim_cube = np.zeros((n_pump, n_probe, n_delays), dtype=float)

    tasks = []
    for idx, t2 in enumerate(delays):
        # Match float delay key with tolerance
        matched_key = None
        for k_t2 in kubo_details:
            if np.isclose(float(k_t2), float(t2), atol=1e-6):
                matched_key = k_t2
                break
        if matched_key is None:
            continue

        det = kubo_details[matched_key]
        if "bands_fitted" in det and "ffcf_fitted" in det:
            bands = det["bands_fitted"]
            decays = det["ffcf_fitted"]
        else:
            params = det["params"]
            num_bands = det.get("num_bands", 1)
            num_decays = det.get("num_decays", 1)
            has_dw0 = det.get("has_dw0", False)
            ffcf_structure = det.get("ffcf_structure", None)
            bands, decays = unpack_kubo_params(
                params, num_bands, num_decays, has_dw0=has_dw0, ffcf_structure=ffcf_structure
            )

        mapped_idx = delay_indices.index(idx) if idx in delay_indices else min(idx, Z_exp_roi_cube.shape[2] - 1)
        exp_max = float(det.get("exp_max", np.max(np.abs(Z_exp_roi_cube[:, :, mapped_idx]))))

        tasks.append({
            "idx": idx,
            "t2": float(t2),
            "bands": bands,
            "decays": decays,
            "pump_full": pump_full,
            "probe_full": probe_full,
            "pump_roi": pump_roi,
            "probe_roi": probe_roi,
            "exp_max": exp_max,
            "tmax": det.get("tmax", 5.0),
            "nt": det.get("nt", 64),
            "w_rot": det.get("w_rot", 1900.0),
            "u": det.get("u", 1.0),
            "fid_data": det.get("fid_data", None),
        })

    if not tasks:
        return Z_sim_cube

    if mode != "disabled" and len(tasks) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            futures = {pool.submit(_simulate_full_slice_worker, t): t["idx"] for t in tasks}
            done_cnt = 0
            for fut in concurrent.futures.as_completed(futures):
                if should_cancel is not None and should_cancel():
                    for f in futures:
                        f.cancel()
                    raise InterruptedError("Simulation cancelled by user.")
                slice_idx, z_slice = fut.result()
                Z_sim_cube[:, :, slice_idx] = z_slice
                done_cnt += 1
                if progress_callback is not None:
                    progress_callback(done_cnt, len(tasks))
    else:
        for i_t, t in enumerate(tasks):
            if should_cancel is not None and should_cancel():
                raise InterruptedError("Simulation cancelled by user.")
            slice_idx, z_slice = _simulate_full_slice_worker(t)
            Z_sim_cube[:, :, slice_idx] = z_slice
            if progress_callback is not None:
                progress_callback(i_t + 1, len(tasks))

    return Z_sim_cube

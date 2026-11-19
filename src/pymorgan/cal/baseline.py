"""Spectroscopic baseline estimation algorithms for wavelength/wavenumber calibration.

Provides length-invariant baseline estimators suited to both broadband UV-Vis/NIR
detector arrays (e.g. 524 px CCD) and compact mid-IR MCT arrays (e.g. 64 px) as
well as high-resolution FTIR/UV-Vis reference standard spectra:

  - ``arpls``: Asymmetrically Reweighted Penalized Least Squares (Baek et al., 2015)
  - ``asls``: Asymmetric Least Squares Whittaker smoothing (Eilers & Boelens, 2005)
  - ``snip``: Statistics-sensitive Non-linear Iterative Peak-clipping (Ryan et al., 1988)
  - ``modpoly``: Iterative Modified Polynomial peak stripping (Zhao et al., 2007)
  - ``legacy_poly``: No pre-fit baseline subtraction (relies on in-fit polynomial terms)
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

BASELINE_METHODS: dict[str, str] = {
    "arpls": "arPLS (Auto Asymmetric)",
    "asls": "AsLS (Whittaker Smooth)",
    "snip": "SNIP (Peak Clipping)",
    "modpoly": "IModPoly (Iter. Polynomial)",
    "legacy_poly": "Legacy Polynomial (In-Fit)",
}

BASELINE_TARGETS: dict[str, str] = {
    "both": "Sample & Standard",
    "sample": "Sample Only",
    "standard": "Standard Only",
    "none": "None (Off)",
}


def _prepare_1d(
    y: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, bool]:
    """Return a clean finite 1D float copy of ``y``, normalized prior ``weights``, and constant flag."""
    arr = np.asarray(y, dtype=float).ravel().copy()
    n = len(arr)
    if n < 5:
        return arr, None, True

    finite = np.isfinite(arr)
    if not np.all(finite):
        if np.sum(finite) < 5:
            return np.zeros_like(arr), None, True
        idx = np.arange(n, dtype=float)
        arr = np.interp(idx, idx[finite], arr[finite])

    if float(np.ptp(arr)) <= 1e-14:
        return arr, None, True

    prior_w: np.ndarray | None = None
    if weights is not None:
        w_arr = np.asarray(weights, dtype=float).ravel()
        if len(w_arr) == n:
            w_arr = np.where(np.isfinite(w_arr), np.maximum(w_arr, 0.0), 0.0)
            w_max = float(np.max(w_arr))
            if w_max > 0.0:
                prior_w = np.clip(w_arr / w_max, 1e-3, 1.0)

    return arr, prior_w, False


def _whittaker_penalty(n: int, smoothness: float) -> sparse.csc_matrix:
    """Build length-invariant second-difference Whittaker penalty ``lambda * D^T D``.

    Because the discrete second-difference operator scales as ``n^4`` across array
    lengths (e.g. 64 px MCT vs. 524 px CCD vs. 15 000 pt FTIR), defining
    ``lambda = max(10.0, (tau * n)**4)`` preserves the physical smoothing bandwidth
    as a fraction ``tau`` of the spectral window.
    """
    tau = float(np.clip(smoothness, 0.01, 2.0))
    lam = max(10.0, (tau * float(n)) ** 4)
    d2 = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc")
    return lam * (d2.T @ d2)


def baseline_arpls(
    y: np.ndarray,
    smoothness: float = 0.20,
    ratio: float = 1e-5,
    max_iter: int = 60,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate a smooth spectral baseline via arPLS (Baek et al., *Analyst* 2015).

    Parameters
    ----------
    y : np.ndarray
        1D spectrum array (absorbance or intensity).
    smoothness : float, default 0.20
        Dimensionless smoothness fraction in ``[0.02, 1.0]`` controlling ``lambda``.
    ratio : float, default 1e-5
        Relative weight-change stopping tolerance.
    max_iter : int, default 60
        Maximum IRLS iterations.
    weights : np.ndarray, optional
        Prior per-point confidence weights in ``[0, 1]`` (e.g. blank-lamp SNR).

    Returns
    -------
    np.ndarray
        Estimated baseline curve matching ``y.shape``.
    """
    y_work, prior_w, is_const = _prepare_1d(y, weights)
    n = len(y_work)
    if n < 5:
        return np.zeros(n, dtype=float)
    if is_const:
        return y_work

    h_mat = _whittaker_penalty(n, smoothness)
    wt = np.ones(n, dtype=float)
    w_solve = prior_w.copy() if prior_w is not None else wt.copy()
    z = y_work.copy()

    for it in range(max_iter):
        w_mat = sparse.diags(w_solve, 0, shape=(n, n), format="csc")
        z = spsolve(w_mat + h_mat, w_solve * y_work)
        d = y_work - z
        d_neg = d[d < 0.0]
        if len(d_neg) < 2:
            break
        mean_neg = float(np.mean(d_neg))
        std_neg = float(np.std(d_neg))
        if std_neg < 1e-12:
            break
        exponent = np.clip(2.0 * (d - (2.0 * std_neg - mean_neg)) / std_neg, -60.0, 60.0)
        wt_new = 1.0 / (1.0 + np.exp(exponent))
        if it > 0 and np.linalg.norm(wt - wt_new) / (np.linalg.norm(wt) + 1e-12) < ratio:
            wt = wt_new
            break
        wt = wt_new
        w_solve = wt * prior_w if prior_w is not None else wt

    return z


def baseline_asls(
    y: np.ndarray,
    smoothness: float = 0.20,
    p: float = 0.005,
    max_iter: int = 25,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate a smooth spectral baseline via AsLS (Eilers & Boelens, 2005).

    Parameters
    ----------
    y : np.ndarray
        1D spectrum array.
    smoothness : float, default 0.20
        Dimensionless smoothness fraction in ``[0.02, 1.0]``.
    p : float, default 0.005
        Asymmetry parameter favoring negative residuals (``0 < p < 0.5``).
    max_iter : int, default 25
        Maximum IRLS iterations.
    weights : np.ndarray, optional
        Prior per-point confidence weights.

    Returns
    -------
    np.ndarray
        Estimated baseline curve.
    """
    y_work, prior_w, is_const = _prepare_1d(y, weights)
    n = len(y_work)
    if n < 5:
        return np.zeros(n, dtype=float)
    if is_const:
        return y_work

    h_mat = _whittaker_penalty(n, smoothness)
    p_val = float(np.clip(p, 1e-5, 0.5))
    wt = np.ones(n, dtype=float)
    w_solve = prior_w.copy() if prior_w is not None else wt.copy()
    z = y_work.copy()

    for it in range(max_iter):
        w_mat = sparse.diags(w_solve, 0, shape=(n, n), format="csc")
        z = spsolve(w_mat + h_mat, w_solve * y_work)
        wt_new = np.where(y_work > z, p_val, 1.0 - p_val)
        if it > 0 and np.array_equal(wt, wt_new):
            break
        wt = wt_new
        w_solve = wt * prior_w if prior_w is not None else wt

    return z


def baseline_snip(
    y: np.ndarray,
    smoothness: float = 0.20,
    decreasing: bool = True,
) -> np.ndarray:
    """Estimate a spectral baseline via LLS-SNIP peak clipping (Ryan et al., 1988).

    Parameters
    ----------
    y : np.ndarray
        1D spectrum array.
    smoothness : float, default 0.20
        Controls the maximum half-window width as a fraction of spectrum length.
    decreasing : bool, default True
        Use decreasing clipping window schedule (smoother baseline under broad bands).

    Returns
    -------
    np.ndarray
        Estimated baseline curve.
    """
    y_work, _, is_const = _prepare_1d(y)
    n = len(y_work)
    if n < 5:
        return np.zeros(n, dtype=float)
    if is_const:
        return y_work

    tau = float(np.clip(smoothness, 0.01, 2.0))
    max_hw = max(3, min(n // 3, int(round(0.8 * tau * n))))
    y_min = float(np.min(y_work))
    shift = -y_min + 1e-6 if y_min <= 0.0 else 0.0

    # Log-log-sqrt (LLS) variance-stabilizing transform
    v = np.log(np.log(np.sqrt(y_work + shift + 1.0) + 1.0) + 1.0)
    ks = range(max_hw, 0, -1) if decreasing else range(1, max_hw + 1)
    for k in ks:
        if 2 * k >= n:
            continue
        v_next = v.copy()
        v_next[k:-k] = np.minimum(v[k:-k], 0.5 * (v[:-2 * k] + v[2 * k:]))
        v = v_next

    return (np.exp(np.exp(v) - 1.0) - 1.0) ** 2 - 1.0 - shift


def baseline_modpoly(
    y: np.ndarray,
    poly_order: int = 3,
    max_iter: int = 50,
    tol: float = 1e-3,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate a spectral baseline via iterative modified polynomial fitting (IModPoly).

    Parameters
    ----------
    y : np.ndarray
        1D spectrum array.
    poly_order : int, default 3
        Polynomial degree.
    max_iter : int, default 50
        Maximum iterations.
    tol : float, default 1e-3
        Relative residual-STD convergence tolerance.
    weights : np.ndarray, optional
        Prior per-point confidence weights.

    Returns
    -------
    np.ndarray
        Estimated polynomial baseline curve.
    """
    y_work, prior_w, is_const = _prepare_1d(y, weights)
    n = len(y_work)
    if n < 5:
        return np.zeros(n, dtype=float)
    if is_const:
        return y_work

    x = np.linspace(-1.0, 1.0, n)
    deg = max(1, min(int(poly_order), n - 3))
    mask = np.ones(n, dtype=bool)
    prev_dev = np.inf
    z = np.zeros(n, dtype=float)

    for _ in range(max_iter):
        w_fit = prior_w[mask] if prior_w is not None else None
        coeffs = np.polyfit(x[mask], y_work[mask], min(deg, int(np.sum(mask)) - 1), w=w_fit)
        z = np.polyval(coeffs, x)
        res = y_work - z
        dev = float(np.std(res[mask]))
        new_mask = res < dev
        if int(np.sum(new_mask)) <= deg + 2:
            break
        mask = new_mask
        if abs(dev - prev_dev) / (dev + 1e-12) < tol:
            break
        prev_dev = dev

    return z


def estimate_baseline(
    y: np.ndarray,
    method: str = "arpls",
    smoothness: float = 0.20,
    asym_p: float = 0.005,
    poly_order: int = 3,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Dispatch baseline estimation for calibration sample or standard spectra.

    Parameters
    ----------
    y : np.ndarray
        1D spectrum array.
    method : str, default ``"arpls"``
        Algorithm key: ``"arpls"``, ``"asls"``, ``"snip"``, ``"modpoly"``, or
        ``"legacy_poly"`` / ``"none"`` (returns zeros).
    smoothness : float, default 0.20
        Dimensionless smoothness parameter for Whittaker / SNIP methods.
    asym_p : float, default 0.005
        Asymmetry parameter for ``"asls"``.
    poly_order : int, default 3
        Polynomial degree for ``"modpoly"``.
    weights : np.ndarray, optional
        Prior per-point weights in ``[0, 1]``.

    Returns
    -------
    np.ndarray
        Estimated baseline curve matching ``y.ravel().shape``.
    """
    key = (method or "arpls").strip().lower()
    y_arr = np.asarray(y, dtype=float).ravel()

    if key in ("none", "off", "legacy_poly", "legacy"):
        return np.zeros_like(y_arr)
    if key == "arpls":
        return baseline_arpls(y_arr, smoothness=smoothness, weights=weights)
    if key == "asls":
        return baseline_asls(y_arr, smoothness=smoothness, p=asym_p, weights=weights)
    if key == "snip":
        return baseline_snip(y_arr, smoothness=smoothness)
    if key in ("modpoly", "imodpoly"):
        return baseline_modpoly(y_arr, poly_order=poly_order, weights=weights)
    raise ValueError(f"Unknown baseline method {method!r}. Expected one of {list(BASELINE_METHODS)}.")

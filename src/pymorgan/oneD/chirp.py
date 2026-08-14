"""Chirp-correction (dispersion) fitting for the 1-D pipeline.

Kept in its own module, separate from :mod:`pymorgan.oneD.process`, because the
coherent-artefact / dispersion fit is a substantially larger and more
self-contained piece of analysis than the routine background subtraction that
lives there. This is a *fitting* step only: it produces and saves the chirp
(dispersion) coefficients. Loading a saved fit and applying it as a correction
to a dataset (interpolating each pixel's kinetic trace onto a common, chirp-free
delay axis) is not implemented yet and is intentionally out of scope here.

Three fit modes are provided:

* **Automatic** -- VARPRO (variable projection): a Gaussian IRF (optionally with
  its 1st/2nd derivatives) plus a constant offset and an optional erf-broadened
  exponential (coherent artefact + early dynamics) is fit at every probe pixel.
  The linear amplitudes are eliminated in closed form at each iteration, leaving
  only ``(t0, FWHM, tau_exp)`` as nonlinear parameters (bounded
  Trust-Region-Reflective, via :func:`scipy.optimize.least_squares`). A second
  pass re-fits every pixel from smoothed initial guesses (an intermediate
  dispersion fit for ``t0``, moving-median smoothing for ``FWHM``/``tau_exp``).
* **Step function** -- a simpler Gaussian (+ derivatives) plus an erf (Heaviside
  convolved with the same Gaussian) step model, fit jointly (not VARPRO) over an
  early-time window. Useful as a quick/robust alternative when the exponential
  component of the Automatic model is not needed or not well-conditioned.
* **Manual** -- fits the Cauchy dispersion equation directly to a set of
  user-picked ``(wavelength, delay)`` points (see
  :class:`pymorgan.gui.picker.ContourPicker` in ``axis="xy"`` mode), with no
  per-pixel IRF fit.

In all three modes, the resulting ``t0(lambda)`` values are fit to a
generalised Cauchy dispersion equation (truncated at ``n_cauchy_terms``,
quadratic-in-1/lambda terms only), using an iteratively-reweighted (bisquare)
linear least-squares fit -- the dispersion equation is linear in its coefficients, so this is solved as a
bound-constrained (bisquare-)reweighted linear least squares fit rather than a
generic nonlinear solve.

Heavy/optional imports (scipy, pandas, scipy.io) are kept inside the functions
that need them, matching the rest of the 1-D pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ..log import get_logger

logger = get_logger(__name__)

__all__ = [
    "ChirpFit",
    "cauchy_t0",
    "fit_cauchy_dispersion",
    "fit_chirp_automatic",
    "fit_chirp_step",
    "fit_chirp_manual",
    "fit_chirp_wavelet",
    "plot_chirp_diagnostics",
    "save_chirp_fit",
    "load_chirp_fit",
    "default_chirp_filename",
]


# --------------------------------------------------------------------------- #
#                              Result container                              #
# --------------------------------------------------------------------------- #
@dataclass
class ChirpFit:
    """Result of a chirp/dispersion fit.

    ``coeffs`` are the Cauchy dispersion coefficients (see :func:`cauchy_t0`).
    ``Pfit`` holds the per-pixel (or per-point, for ``mode="manual"``) fit
    results: 8 columns ``[t0, FWHM, G0_amp, G1_amp, G2_amp, offset, tau_exp,
    exp_amp]`` for ``mode="automatic"``; 7 columns ``[t0, FWHM, G0_amp,
    G1_amp, G2_amp, offset, step_amp]`` for ``mode="step"`` with
    ``use_exp=False``, or 9 columns (appending ``[tau_exp, exp_amp]``) when
    ``use_exp=True`` -- ``FWHM`` is the genuine full width at half maximum (ps)
    of the Gaussian IRF in both modes, no further conversion needed -- or a
    single ``[t0]`` column for ``mode="manual"``. Pixels skipped during an
    automatic/step fit (non-finite/all-zero data, or fit failure) carry NaN.
    """

    mode: str  # "automatic" | "step" | "manual"
    coeffs: np.ndarray
    fit_wl: np.ndarray
    Pfit: np.ndarray
    equation: str = "Cauchy"
    n_cauchy_terms: int = 6
    lambda_ref: float = float("nan")
    fit_pixels: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    Dfit: np.ndarray | None = None
    mean_irf_fs: float = float("nan")
    detector: int = 0
    deriv_mask: tuple[bool, bool] = (False, False)
    use_exp: bool = False
    do_2nd_pass: bool = True
    smooth_window: int = 10
    t_min: float = float("nan")
    t_max: float = float("nan")
    n_skipped: int = 0
    omega_w: float = float("nan")  # Wavelet oscillation period (ps); NaN if not a wavelet fit
    gamma_w: float = float("nan")  # Wavelet width (ps); NaN if not a wavelet fit
    source: str | None = None
    created: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M%S"))

    def __call__(self, probe):
        """Evaluate the fitted dispersion curve at ``probe`` wavelength(s)."""
        return cauchy_t0(self.coeffs, probe, self.lambda_ref)

    @property
    def fwhm(self) -> np.ndarray:
        """Return the FWHM (in ps) vs wavelength for each fitted pixel.

        Returns an array of NaN if not available (e.g. manual mode).
        """
        if self.mode in ("automatic", "step") and self.Pfit.ndim == 2 and self.Pfit.shape[1] >= 2:
            return self.Pfit[:, 1]
        return np.full(self.fit_wl.shape, np.nan)

    @property
    def t0(self) -> np.ndarray:
        """Return the fitted t0 (in ps) at each fitted wavelength (before Cauchy smoothing)."""
        if self.Pfit.ndim == 2 and self.Pfit.shape[0] > 0:
            return self.Pfit[:, 0]
        return np.full(self.fit_wl.shape, np.nan)

    @property
    def fit_parameters_dict(self) -> dict[str, np.ndarray]:
        """Return a dictionary of the individual pixel/point fit parameters."""
        if self.Pfit.ndim != 2 or self.Pfit.shape[0] == 0:
            return {}

        if self.mode == "automatic":
            names = ["t0", "FWHM", "G0_amp", "G1_amp", "G2_amp", "offset", "tau_exp", "exp_amp"]
        elif self.mode == "step":
            names = ["t0", "FWHM", "G0_amp", "G1_amp", "G2_amp", "offset", "step_amp"]
        elif self.mode in ("manual", "wavelet"):
            names = ["t0"]
        else:
            return {}

        return {name: self.Pfit[:, i] for i, name in enumerate(names) if i < self.Pfit.shape[1]}

    # ------------------------------------------------------------- #
    #                       Save / Load round-trip                  #
    # ------------------------------------------------------------- #
    def to_dict(self) -> dict:
        """Flatten to a dictionary of arrays and scalars for saving via ``save_chirp_fit``.

        Uses compatible variable names (``Cfit``, ``fitPixels``, ``fitWL``, ``Pfit``,
        ``Dfit``) and 1-based indexing for ``fitPixels`` to maintain compatibility
        with downstream analysis scripts. Since ``savemat`` cannot embed a Python callable,
        ``chirpFun_str`` carries the equivalent MATLAB anonymous-function source, and
        ``matlab_usage`` provides instructions on how to evaluate the curve.
        """
        coeffs = np.atleast_1d(np.asarray(self.coeffs, dtype=float))
        fit_pixels = np.atleast_1d(np.asarray(self.fit_pixels, dtype=float))
        fit_pixels_1based = fit_pixels + 1.0 if fit_pixels.size else fit_pixels
        return {
            "mode": self.mode,
            "equation": self.equation,
            "n_cauchy_terms": int(self.n_cauchy_terms),
            "lambda_ref": float(self.lambda_ref),
            "Cfit": coeffs,
            "chirpFun_str": _matlab_chirp_fun_str(coeffs.size, self.lambda_ref),
            "matlab_usage": (
                "S = load(thisfile);  chirpFun = eval(S.chirpFun_str);  "
                "t0 = chirpFun(S.Cfit, probeAxis);"
            ),
            "fitPixels": fit_pixels_1based,
            "fitWL": np.asarray(self.fit_wl, dtype=float),
            "Pfit": np.asarray(self.Pfit, dtype=float),
            "Dfit": np.asarray(self.Dfit, dtype=float)
            if self.Dfit is not None
            else np.zeros((0, 0)),
            "fwhm": np.asarray(self.fwhm, dtype=float),
            "fit_parameters": {
                k: np.asarray(v, dtype=float) for k, v in self.fit_parameters_dict.items()
            },
            "mean_irf_fs": float(self.mean_irf_fs),
            "detector": int(self.detector),
            "deriv_mask": np.asarray([bool(x) for x in self.deriv_mask], dtype=float),
            "use_exp": int(self.use_exp),
            "do_2nd_pass": int(self.do_2nd_pass),
            "smooth_window": int(self.smooth_window),
            "t_min": float(self.t_min),
            "t_max": float(self.t_max),
            "n_skipped": int(self.n_skipped),
            "omega_w": float(self.omega_w),
            "gamma_w": float(self.gamma_w),
            "source": self.source or "",
            "created": self.created,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ChirpFit:
        """Reconstruct from a dict produced by loading (see :func:`load_chirp_fit`).

        Accepts both the standard compatible keys (``Cfit``/``fitPixels``/
        ``fitWL``, 1-based pixels) and the alternative key names
        (``coeffs``/``fit_pixels``/``fit_wl``, 0-based) for backward compatibility.

        Also handles MAT files saved by MATLAB's ``FitChirpCorr.m``, which do not
        store ``mode``, ``lambda_ref`` or ``n_cauchy_terms``.  These are inferred
        automatically:

        * ``lambda_ref`` is reconstructed from ``mean(fitWL)`` -- MATLAB always
          uses this value as the reference wavelength for the Cauchy dispersion
          formula, and it can be verified from the ``scaleVec`` closure embedded
          in the saved function handle.
        * ``n_cauchy_terms`` is inferred from ``len(Cfit)``.
        * ``mode`` is guessed from the column count of ``Pfit``:
          8 → ``"automatic"``, 7 or 9 → ``"step"``, anything else → ``"manual"``.
        """

        def _get(*keys, default=None):
            for key in keys:
                if key in d and d[key] is not None:
                    return d[key]
            return default

        Dfit_raw = np.atleast_1d(np.asarray(_get("Dfit", default=np.zeros((0, 0)))))
        Dfit = None if Dfit_raw.size == 0 else np.atleast_2d(Dfit_raw.astype(float))
        deriv_mask_arr = np.atleast_1d(np.asarray(_get("deriv_mask", default=[0, 0]), dtype=float))
        deriv_mask = (
            tuple(bool(x) for x in deriv_mask_arr[:2])
            if deriv_mask_arr.size >= 2
            else (False, False)
        )
        source = str(_get("source", default="")).strip()

        if "fitPixels" in d and d["fitPixels"] is not None:
            fit_pixels = np.atleast_1d(np.asarray(d["fitPixels"], dtype=float)) - 1.0
        else:
            fit_pixels = np.atleast_1d(np.asarray(_get("fit_pixels", default=[]), dtype=float))

        coeffs = np.atleast_1d(np.asarray(_get("Cfit", "coeffs", default=[]), dtype=float))
        fit_wl = np.atleast_1d(np.asarray(_get("fitWL", "fit_wl", default=[]), dtype=float))
        Pfit_raw = np.asarray(_get("Pfit", default=[]), dtype=float)
        if Pfit_raw.ndim == 1:
            Pfit = Pfit_raw[:, np.newaxis]
        else:
            Pfit = np.atleast_2d(Pfit_raw)

        # --- Infer fields that MATLAB's FitChirpCorr.m does not save ----------
        # n_cauchy_terms: fall back to len(Cfit) if not stored.
        stored_n = _get("n_cauchy_terms")
        if stored_n is None:
            n_cauchy_terms = int(coeffs.size) if coeffs.size > 0 else 1
        else:
            n_cauchy_terms = int(stored_n)

        # lambda_ref: MATLAB always uses mean(fitWL) as the reference wavelength
        # for the scaleVec in the anonymous chirpFun.  Recover it from fitWL
        # when not present or NaN in the file.
        stored_lref = _get("lambda_ref")
        if stored_lref is None or (
            hasattr(stored_lref, "__float__") and not np.isfinite(float(stored_lref))
        ):
            valid_wl = fit_wl[np.isfinite(fit_wl)] if fit_wl.size > 0 else np.array([])
            lambda_ref = float(np.mean(valid_wl)) if valid_wl.size > 0 else float("nan")
        else:
            lambda_ref = float(stored_lref)

        # mode: infer from Pfit column count if not present or empty.
        stored_mode = str(_get("mode", default="")).strip()
        if not stored_mode:
            ncols = Pfit.shape[1] if Pfit.ndim == 2 and Pfit.shape[0] > 0 else 0
            if ncols >= 8:
                stored_mode = "automatic"
            elif ncols in (7, 9):
                stored_mode = "step"
            else:
                stored_mode = "manual"

        return cls(
            mode=stored_mode,
            equation=str(_get("equation", default="Cauchy")).strip(),
            n_cauchy_terms=n_cauchy_terms,
            lambda_ref=lambda_ref,
            coeffs=coeffs,
            fit_pixels=fit_pixels.astype(int),
            fit_wl=fit_wl,
            Pfit=Pfit,
            Dfit=Dfit,
            mean_irf_fs=float(_get("mean_irf_fs", default=float("nan"))),
            detector=int(_get("detector", default=0)),
            deriv_mask=deriv_mask,
            use_exp=bool(_get("use_exp", default=0)),
            do_2nd_pass=bool(_get("do_2nd_pass", default=1)),
            smooth_window=int(_get("smooth_window", default=10)),
            t_min=float(_get("t_min", default=float("nan"))),
            t_max=float(_get("t_max", default=float("nan"))),
            n_skipped=int(_get("n_skipped", default=0)),
            omega_w=float(_get("omega_w", default=float("nan"))),
            gamma_w=float(_get("gamma_w", default=float("nan"))),
            source=source or None,
            created=str(_get("created", default="")).strip(),
        )


def default_chirp_filename(detector: int | None = None, n_detectors: int = 1) -> str:
    """Default save name, e.g. ``chirpCorr_20260622-143000.mat`` or ``chirpCorr_DET1_20260622-143000.mat``."""
    ts = f"{datetime.now():%Y%m%d-%H%M%S}"
    if n_detectors > 1 or (detector is not None and detector > 0):
        det_num = (detector or 0) + 1
        return f"chirpCorr_DET{det_num}_{ts}.mat"
    return f"chirpCorr_{ts}.mat"


def get_det_chirp_path(path, detector: int = 0) -> Path:
    """Ensure path has ``_DET<N>`` suffix for multi-detector chirp files."""
    import re
    from pathlib import Path
    p = Path(path)
    det_num = int(detector) + 1
    stem = p.stem
    pattern = r"_DET\d+"
    if re.search(pattern, stem, flags=re.IGNORECASE):
        new_stem = re.sub(pattern, f"_DET{det_num}", stem, flags=re.IGNORECASE)
    else:
        new_stem = f"{stem}_DET{det_num}"
    return p.with_name(new_stem + p.suffix)


def save_chirp_fit(fit: ChirpFit, path, n_detectors: int = 1) -> None:
    """Save ``fit`` to ``path`` as a ``.h5``/``.hdf5``, ``.json``, or ``.mat`` file based on path suffix."""
    import re
    from pathlib import Path

    path = Path(path)
    det = getattr(fit, "detector", 0)
    if det > 0 or n_detectors > 1 or re.search(r"_DET\d+", path.stem, flags=re.IGNORECASE):
        path = get_det_chirp_path(path, det)

    suffix = path.suffix.lower()

    if suffix in (".h5", ".hdf5"):
        import h5py

        data = fit.to_dict()

        def save_item(parent, key, val):
            if val is None:
                return
            if isinstance(val, dict):
                grp = parent.create_group(key)
                for k, v in val.items():
                    save_item(grp, k, v)
            elif isinstance(val, np.ndarray):
                parent.create_dataset(key, data=val, compression="gzip" if val.size > 100 else None)
            elif isinstance(val, (str, bytes)):
                parent.create_dataset(key, data=val)
            elif isinstance(val, (list, tuple)):
                arr = np.asarray(val)
                parent.create_dataset(key, data=arr)
            else:
                parent.create_dataset(key, data=val)

        with h5py.File(path, "w") as f:
            for k, v in data.items():
                save_item(f, k, v)
    elif suffix == ".json":
        import json

        data = fit.to_dict()

        def serialize(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if hasattr(obj, "item") and callable(obj.item):
                return obj.item()
            if isinstance(obj, (list, tuple)):
                return [serialize(x) for x in obj]
            if isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            return obj

        serialized_data = serialize(data)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serialized_data, f, indent=2)
    else:
        import scipy.io as sio

        sio.savemat(str(path), fit.to_dict(), do_compression=True)


def load_chirp_fit(path) -> ChirpFit:
    """Load a :class:`ChirpFit` previously written by :func:`save_chirp_fit`.

    Also accepts MAT files saved by MATLAB's ``FitChirpCorr.m``.  The MATLAB
    format stores ``Cfit``, ``fitWL``, ``fitPixels``, ``Pfit`` and a
    ``chirpFun`` function-handle object.  The function handle cannot be
    evaluated by scipy, but it is not needed: the Cauchy dispersion curve is
    fully reconstructed from ``Cfit`` and the inferred reference wavelength
    (see :meth:`ChirpFit.from_dict`).
    """
    from pathlib import Path

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".h5", ".hdf5"):
        import h5py

        def load_group(group) -> dict:
            d = {}
            for k in group.keys():
                item = group[k]
                if isinstance(item, h5py.Group):
                    d[k] = load_group(item)
                elif isinstance(item, h5py.Dataset):
                    val = item[()]
                    if isinstance(val, bytes):
                        val = val.decode("utf-8")
                    elif isinstance(val, np.ndarray) and val.dtype.kind == "S":
                        val = np.char.decode(val, "utf-8")
                    d[k] = val
            return d

        with h5py.File(path, "r") as f:
            data = load_group(f)
        return ChirpFit.from_dict(data)
    elif suffix == ".json":
        import json

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return ChirpFit.from_dict(data)
    else:
        import scipy.io as sio

        mat = sio.loadmat(str(path), squeeze_me=True, simplify_cells=True)
        # Strip internal scipy/MATLAB metadata keys and non-array objects
        # (e.g. MatlabFunction for chirpFun) that cannot be coerced to arrays.
        _SKIP_PREFIXES = ("__",)
        _SKIP_TYPES = ()
        try:
            from scipy.io.matlab import MatlabFunction, MatlabOpaque

            _SKIP_TYPES = (MatlabFunction, MatlabOpaque)
        except ImportError:
            logger.debug(
                "scipy.io.matlab types unavailable; MATLAB function/opaque entries "
                "will not be skipped.", exc_info=True
            )
        data = {
            k: v
            for k, v in mat.items()
            if not any(k.startswith(p) for p in _SKIP_PREFIXES) and not isinstance(v, _SKIP_TYPES)
        }
        return ChirpFit.from_dict(data)


# --------------------------------------------------------------------------- #
#                          Cauchy dispersion equation                        #
# --------------------------------------------------------------------------- #
def _cauchy_design(probe: np.ndarray, n_terms: int, lambda_ref: float) -> np.ndarray:
    """Design matrix for ``t0(L) = sum_n coeffs[n] * lambda_ref^(2n) / L^(2n)``."""
    L = np.asarray(probe, dtype=float)
    exps = 2 * np.arange(n_terms)
    scale = lambda_ref**exps
    return scale[None, :] / (L.reshape(-1, 1) ** exps[None, :])


def _matlab_chirp_fun_str(n_terms: int, lambda_ref: float) -> str:
    """Generate MATLAB-syntax anonymous-function source for the dispersion curve.

    Generates self-contained anonymous-function text with ``lambda_ref`` and the
    per-term scaling substituted in as literals. Evaluating this string in MATLAB
    recreates a callable: ``chirpFun(S.Cfit, probeAxis)``.
    """
    exps = 2 * np.arange(max(1, int(n_terms)))
    scale = lambda_ref**exps
    scale_str = "[" + " ".join(f"{v:.12g}" for v in scale) + "]"
    exp_str = "[" + " ".join(str(int(e)) for e in exps) + "]"
    return f"@(p,L) reshape(sum(({scale_str} .* reshape(p,1,[])) ./ (L(:).^{exp_str}), 2), size(L))"


def cauchy_t0(coeffs: Sequence[float], probe, lambda_ref: float) -> np.ndarray:
    """Evaluate the generalised Cauchy dispersion equation at ``probe``.

    ``t0(L) = coeffs[0] + sum_{n=1}^{N-1} coeffs[n] * lambda_ref^(2n) / L^(2n)``.
    The ``lambda_ref`` prefactors render every coefficient dimensionless and of
    comparable magnitude.
    """
    coeffs = np.atleast_1d(np.asarray(coeffs, dtype=float))
    probe_arr = np.asarray(probe, dtype=float)
    X = _cauchy_design(probe_arr.ravel(), coeffs.size, lambda_ref)
    return (X @ coeffs).reshape(probe_arr.shape)


def _bisquare_weights(resid: np.ndarray, c: float = 4.685) -> np.ndarray:
    """Tukey's bisquare weights for one IRLS iteration (scale via MAD)."""
    resid = np.asarray(resid, dtype=float)
    mad = np.median(np.abs(resid - np.median(resid)))
    s = 1.4826 * mad
    if not np.isfinite(s) or s < 1e-12:
        return np.ones_like(resid)
    u = resid / (c * s)
    return np.where(np.abs(u) < 1.0, (1.0 - u**2) ** 2, 0.0)


def fit_cauchy_dispersion(
    fit_wl,
    t0,
    *,
    n_terms: int = 6,
    lambda_ref: float | None = None,
    robust: bool = True,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    max_iter: int = 30,
    tol: float = 1e-9,
) -> tuple[np.ndarray, float]:
    """Fit ``t0(lambda)`` to the Cauchy dispersion equation.

    The equation is *linear* in its coefficients, so this is solved as a bound-constrained
    (bisquare-)reweighted linear least squares fit
    (:func:`scipy.optimize.lsq_linear` per IRLS iteration) rather than a generic
    nonlinear solve. Returns ``(coeffs, lambda_ref)``.

    Default bounds constrain the leading (offset) term to ``[-5, 5]`` ps, and
    higher-order terms to ``[-100, 0]`` (constraining dispersion to decrease with wavelength).
    """
    from scipy.optimize import lsq_linear

    fit_wl = np.asarray(fit_wl, dtype=float)
    t0 = np.asarray(t0, dtype=float)
    valid = np.isfinite(fit_wl) & np.isfinite(t0)
    n_terms = max(1, int(round(n_terms)))
    if lambda_ref is None:
        lambda_ref = (
            float(np.nanmean(fit_wl[valid])) if np.any(valid) else float(np.nanmean(fit_wl))
        )
    if np.count_nonzero(valid) < n_terms:
        raise ValueError(
            f"Need at least {n_terms} valid (finite) point(s) to fit {n_terms} Cauchy term(s); "
            f"got {int(np.count_nonzero(valid))}."
        )
    if bounds is None:
        valid_wl = fit_wl[valid]
        valid_t0 = t0[valid]
        if valid_wl.size >= 2:
            dx = valid_wl - np.mean(valid_wl)
            dy = valid_t0 - np.mean(valid_t0)
            var_x = np.sum(dx**2)
            slope = np.sum(dx * dy) / var_x if var_x > 1e-8 else 0.0
        else:
            slope = 0.0

        if slope <= 0.0:
            lb = np.concatenate(([-5.0], 0.0 * np.ones(n_terms - 1)))
            ub = np.concatenate(([5.0], 1e2 * np.ones(n_terms - 1)))
        else:
            lb = np.concatenate(([-5.0], -1e2 * np.ones(n_terms - 1)))
            ub = np.concatenate(([5.0], 0.0 * np.ones(n_terms - 1)))
    else:
        lb, ub = (np.asarray(b, dtype=float) for b in bounds)

    X = _cauchy_design(fit_wl[valid], n_terms, lambda_ref)
    y = t0[valid]
    w = np.ones_like(y)
    coeffs = np.zeros(n_terms)
    n_passes = max_iter if robust else 1
    for _ in range(n_passes):
        sw = np.sqrt(w)
        res = lsq_linear(X * sw[:, None], y * sw, bounds=(lb, ub), method="bvls")
        new_coeffs = res.x
        if not robust:
            coeffs = new_coeffs
            break
        resid = y - X @ new_coeffs
        new_w = _bisquare_weights(resid)
        converged = np.linalg.norm(new_coeffs - coeffs) < tol * (np.linalg.norm(coeffs) + tol)
        coeffs, w = new_coeffs, new_w
        if converged:
            break
    return coeffs, lambda_ref


def _movmedian(x: np.ndarray, window: int) -> np.ndarray:
    """Centred moving median with a shrinking window at the edges."""
    import pandas as pd

    s = pd.Series(np.asarray(x, dtype=float))
    return s.rolling(window=max(1, int(window)), centre=True, min_periods=1).median().to_numpy()


def _interpolate_nans(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN values in a 1-D array, falling back to nearest neighbor at edges."""
    x = np.asarray(x, dtype=float)
    if not np.any(np.isnan(x)):
        return x
    nans = np.isnan(x)
    if np.all(nans):
        return x
    x_new = x.copy()
    indices = np.arange(len(x))
    x_new[nans] = np.interp(indices[nans], indices[~nans], x[~nans])
    return x_new


# --------------------------------------------------------------------------- #
#                  Automatic (VARPRO coherent-artefact) fit                  #
# --------------------------------------------------------------------------- #
def _gaussian_irf_columns(t: np.ndarray, t0: float, fwhm: float, deriv_mask: tuple[bool, bool]):
    """Gaussian IRF column(s) [+ 1st/2nd derivative] for the VARPRO design matrix.

    Uses the standard normalised Gaussian ``G0 = exp(-u^2/(2*sg^2))`` with
    ``sg = FWHM/(2*sqrt(2*ln2))`` the true standard deviation -- so ``G0`` is
    exactly 0.5 at ``u = +/-FWHM/2``, i.e. the fitted ``fwhm`` parameter *is*
    the actual full width at half maximum (verified numerically; an earlier
    version used ``exp(-ln2*(u/sg)^2)``, which is only correct if ``sg`` is
    itself the half-width-at-half-maximum, not a standard deviation -- that
    made the fitted "FWHM" column understate the true width by a factor of
    ``2*sqrt(2*ln2)/2 = sqrt(2*ln2) ~ 1.18``, i.e. report sigma-like behaviour
    under a FWHM label). The standard form here is also what
    :func:`_erf_exp_column` requires, since its closed-form Gaussian-erf
    convolution assumes ``sg`` is a genuine standard deviation.
    """
    sg = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    u = t - t0
    G0 = np.exp(-(u**2) / (2.0 * sg**2))
    cols = [G0]
    if deriv_mask[0]:
        cols.append(-(u / sg**2) * G0)
    if deriv_mask[1]:
        cols.append((u**2 - sg**2) / sg**4 * G0)
    return cols, sg, u


def _erf_exp_column(u: np.ndarray, sg: float, tau: float) -> np.ndarray:
    """Numerically stable erf-broadened exponential (coherent-artefact decay).

    During the bounded nonlinear search ``tau`` can transiently approach its
    lower bound (``1e-3`` ps); ``0.5*sigma^2/tau^2`` then grows large enough
    that the (otherwise exact) exponential prefactor overflows ``float64``
    before the optimiser rejects the step. The exponent is clipped to
    ``+/-700`` (``exp(700)`` is still finite, ~1e304) purely to avoid that
    overflow/NaN -- it never affects a converged, physically sensible fit.
    """
    from scipy.special import erf, erfcx

    sgsq = sg**2
    E1 = np.zeros_like(u)
    late = u >= 0
    if np.any(late):
        ul = u[late]
        argE = np.clip(-ul / tau + 0.5 * sgsq / tau**2, -700.0, 700.0)
        argB = (ul - sgsq / tau) / (sg * np.sqrt(2.0))
        E1[late] = 0.5 * np.exp(argE) * (1.0 + erf(argB))
    early = ~late
    if np.any(early):
        ue = u[early]
        prefac = np.exp(np.clip(-(ue**2) / (2.0 * sgsq), -700.0, 700.0))
        arg_x = sg / (tau * np.sqrt(2.0)) - ue / (sg * np.sqrt(2.0))
        E1[early] = 0.5 * prefac * erfcx(arg_x)
    return E1


def _varpro_design(pnl, t: np.ndarray, deriv_mask: tuple[bool, bool], use_exp: bool) -> np.ndarray:
    t0, fwhm, tau = pnl
    cols, sg, u = _gaussian_irf_columns(t, t0, fwhm, deriv_mask)
    cols.append(np.ones_like(t))
    if use_exp:
        cols.append(_erf_exp_column(u, sg, tau))
    return np.column_stack(cols)


def _varpro_resid(pnl, t: np.ndarray, y: np.ndarray, deriv_mask, use_exp):
    A = _varpro_design(pnl, t, deriv_mask, use_exp)
    if not np.all(np.isfinite(A)):
        # Defensive: a transient non-finite design matrix (e.g. an extreme,
        # not-yet-rejected trial step) would otherwise reach LAPACK and print
        # noisy "illegal value" warnings. Report a large residual instead so
        # the optimiser simply steps away from this point.
        return np.full_like(y, 1e6), A, np.zeros(A.shape[1])
    amps, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ amps, A, amps


def _fit_one_pixel_varpro(y_data, delays, pnl0, lb_nl, ub_nl, deriv_mask, use_exp):
    """Run one VARPRO pixel fit; returns ``(pfit_row[8], dfit_col, nf_val)``."""
    from scipy.optimize import least_squares

    from pymorgan.oneD.chirp import _varpro_design, _varpro_resid

    pfit_row = np.full(8, np.nan)
    dfit_col = np.zeros_like(delays, dtype=float)
    nf_val = 1.0

    finite = np.isfinite(y_data) & np.isfinite(delays)
    if np.count_nonzero(finite) < 5:
        return pfit_row, dfit_col, nf_val

    nf_test = np.max(np.abs(y_data[finite]))
    if not np.isfinite(nf_test) or nf_test == 0:
        return pfit_row, dfit_col, nf_val
    nf_val = float(nf_test)
    y_n = y_data[finite] / nf_val
    delays_fit = delays[finite]

    pnl0 = np.clip(np.asarray(pnl0, dtype=float), lb_nl, ub_nl)
    if not np.all(np.isfinite(pnl0)):
        return pfit_row, dfit_col, nf_val

    try:
        result = least_squares(
            lambda p: _varpro_resid(p, delays_fit, y_n, deriv_mask, use_exp)[0],
            pnl0,
            bounds=(lb_nl, ub_nl),
            method="trf",
            xtol=5e-10,
            ftol=5e-10,
            gtol=5e-10,
            max_nfev=5000,
        )
        pnl = result.x
    except Exception:
        return pfit_row, dfit_col, nf_val

    _, A_fit, amps = _varpro_resid(pnl, delays_fit, y_n, deriv_mask, use_exp)

    A_full = _varpro_design(pnl, delays, deriv_mask, use_exp)
    if not np.all(np.isfinite(A_full)):
        dfit_col = np.full_like(delays, np.nan)
    else:
        dfit_col = nf_val * (A_full @ amps)

    pfit_row = np.zeros(8)
    pfit_row[0] = pnl[0]  # t0
    pfit_row[1] = pnl[1]  # FWHM
    pfit_row[6] = pnl[2]  # tau_exp

    idx = 0
    pfit_row[2] = amps[idx]
    idx += 1  # G0 amp
    if deriv_mask[0]:
        pfit_row[3] = amps[idx]
        idx += 1
    if deriv_mask[1]:
        pfit_row[4] = amps[idx]
        idx += 1
    pfit_row[5] = amps[idx]
    idx += 1  # offset
    if use_exp:
        pfit_row[7] = amps[idx]

    return pfit_row, dfit_col, nf_val


def _compute_second_pass_inits(Pfit, fit_wl, *, n_cauchy_terms, lambda_ref, smooth_window):
    """Smoothed (t0, FWHM, tau_exp) initial guesses for the 2nd VARPRO pass."""
    fwhm_init = _interpolate_nans(_movmedian(Pfit[:, 1], smooth_window))
    tau_init = _interpolate_nans(_movmedian(Pfit[:, 6], smooth_window))

    valid = np.isfinite(Pfit[:, 0])
    n_terms = max(1, int(round(n_cauchy_terms)))
    if np.count_nonzero(valid) >= n_terms:
        try:
            lb = np.concatenate(([-5.0], -1e2 * np.ones(n_terms - 1)))
            ub = np.concatenate(([5.0], 1e2 * np.ones(n_terms - 1)))
            coeffs, lam_ref = fit_cauchy_dispersion(
                fit_wl[valid],
                Pfit[valid, 0],
                n_terms=n_terms,
                lambda_ref=lambda_ref,
                bounds=(lb, ub),
            )
            t0_init = cauchy_t0(coeffs, fit_wl, lam_ref)
        except Exception:
            t0_init = _interpolate_nans(_movmedian(Pfit[:, 0], smooth_window))
    else:
        t0_init = _interpolate_nans(_movmedian(Pfit[:, 0], smooth_window))
    return t0_init, fwhm_init, tau_init


def _parallel_fit_loop(
    worker_fn,
    tasks,
    n_fit,
    label,
    Pfit,
    Dfit,
    progress_callback,
    preview_callback,
    should_cancel,
    delays_fit,
    fit_wl,
):
    """Executes the pixel-by-pixel fit tasks in parallel or sequentially based on settings."""
    from pymorgan import get_settings
    settings = get_settings()
    mode = getattr(settings, "parallel_fitting", "threadpool")
    if hasattr(mode, "value"):
        mode = mode.value

    if mode == "disabled" or len(tasks) <= 1:
        for idx_in_tasks, (j, task_args) in enumerate(tasks):
            if should_cancel is not None and should_cancel():
                raise InterruptedError("Chirp fit cancelled.")
            res = worker_fn(*task_args)
            row = res[0]
            dcol = res[1]
            Pfit[j, :] = row
            Dfit[:, j] = dcol
            if progress_callback is not None:
                progress_callback(idx_in_tasks + 1, n_fit, label)
            if preview_callback is not None:
                preview_callback(float(fit_wl[j]), delays_fit, task_args[0], dcol, float(row[0]))
    else:
        import concurrent.futures
        Executor = (
            concurrent.futures.ProcessPoolExecutor
            if mode == "processpool"
            else concurrent.futures.ThreadPoolExecutor
        )
        with Executor() as executor:
            futures = {}
            for j, task_args in tasks:
                future = executor.submit(worker_fn, *task_args)
                futures[future] = (j, task_args[0])

            completed_count = 0
            for future in concurrent.futures.as_completed(futures):
                if should_cancel is not None and should_cancel():
                    for f in futures:
                        f.cancel()
                    raise InterruptedError("Chirp fit cancelled.")

                j, y_data = futures[future]
                res = future.result()
                row = res[0]
                dcol = res[1]
                Pfit[j, :] = row
                Dfit[:, j] = dcol
                completed_count += 1
                if progress_callback is not None:
                    progress_callback(completed_count, n_fit, label)
                if preview_callback is not None:
                    preview_callback(float(fit_wl[j]), delays_fit, y_data, dcol, float(row[0]))


def fit_chirp_automatic(
    data,
    *,
    detector: int = 0,
    deriv_mask: tuple[bool, bool] = (False, False),
    use_exp: bool = True,
    n_cauchy_terms: int = 6,
    cauchy_lambda_ref: float | None = None,
    do_2nd_pass: bool = True,
    smooth_window: int = 10,
    fit_pixels: Sequence[int] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    preview_callback: Callable[[float, np.ndarray, np.ndarray, np.ndarray, float], None]
    | None = None,
) -> ChirpFit:
    """VARPRO coherent-artefact fit at every probe pixel + Cauchy dispersion fit.

    ``deriv_mask`` selects which Gaussian-IRF derivative terms to include
    (1st, 2nd); ``use_exp`` adds the erf-broadened exponential (coherent
    artefact / early dynamics) term. With ``do_2nd_pass`` (default), a second
    VARPRO pass re-fits every pixel from an intermediate dispersion fit (for
    ``t0``) and moving-median smoothing (for ``FWHM``/``tau_exp``). Pixels with
    non-finite or all-zero data are
    skipped silently (NaN in ``Pfit``). ``preview_callback``, if given, is
    called after every pixel as ``(wavelength, delays, y_data, model_curve,
    t0)`` -- intended for a live single-pixel data-vs-fit preview (e.g. in the
    GUI's progress dialog, typically plotted as ``delays - t0`` so the IRF
    stays centred regardless of dispersion); it is not needed for
    headless/scripted use.
    """
    delays = np.asarray(data.delays, dtype=float)
    probe = data._detector_probe(detector)
    Z = data._detector_slice(detector)

    idxs = np.arange(probe.size) if fit_pixels is None else np.asarray(fit_pixels, dtype=int)
    fit_wl = probe[idxs]
    n_fit = idxs.size

    Pfit = np.full((n_fit, 8), np.nan)
    Dfit = np.zeros((delays.size, n_fit))

    def _run_pass(label, t0_guess_fn, fwhm_guess_fn, tau_guess_fn):
        tasks = []
        for j, i in enumerate(idxs):
            y = Z[:, i]
            finite = np.isfinite(y)
            idm = int(np.nanargmax(np.abs(y))) if np.any(finite) else 0
            t0_guess = t0_guess_fn(j, idm)
            lb = [t0_guess - 1.0, 1e-3, 1e-3]
            ub = [t0_guess + 1.0, 1.0, 5.0]
            pnl0 = [t0_guess, fwhm_guess_fn(j), tau_guess_fn(j)]
            tasks.append((j, (y, delays, pnl0, lb, ub, deriv_mask, use_exp)))

        _parallel_fit_loop(
            _fit_one_pixel_varpro,
            tasks,
            n_fit,
            label,
            Pfit,
            Dfit,
            progress_callback,
            preview_callback,
            should_cancel,
            delays,
            fit_wl,
        )

    _run_pass(
        "Pass 1: fitting coherent artefact...",
        lambda j, idm: delays[idm],
        lambda j: 0.1,
        lambda j: 1.0,
    )

    if do_2nd_pass:
        t0_init, fwhm_init, tau_init = _compute_second_pass_inits(
            Pfit,
            fit_wl,
            n_cauchy_terms=n_cauchy_terms,
            lambda_ref=cauchy_lambda_ref,
            smooth_window=smooth_window,
        )

        def _t0_guess(j, idm):
            return t0_init[j] if np.isfinite(t0_init[j]) else delays[idm]

        def _fwhm_guess(j):
            return max(fwhm_init[j], 1e-3) if np.isfinite(fwhm_init[j]) else 0.1

        def _tau_guess(j):
            return max(tau_init[j], 1e-3) if np.isfinite(tau_init[j]) else 1.0

        _run_pass(
            "Pass 2: refining with smoothed initial guesses...", _t0_guess, _fwhm_guess, _tau_guess
        )

    valid = np.isfinite(Pfit[:, 0])
    n_skipped = int(np.sum(~valid))
    coeffs, lam_ref = fit_cauchy_dispersion(
        fit_wl[valid], Pfit[valid, 0], n_terms=n_cauchy_terms, lambda_ref=cauchy_lambda_ref
    )
    mean_irf_fs = (
        float(np.nanmean(Pfit[:, 1])) * 1000.0 if np.any(np.isfinite(Pfit[:, 1])) else float("nan")
    )

    return ChirpFit(
        mode="automatic",
        equation="Cauchy",
        n_cauchy_terms=int(max(1, round(n_cauchy_terms))),
        lambda_ref=lam_ref,
        coeffs=coeffs,
        fit_pixels=idxs,
        fit_wl=fit_wl,
        Pfit=Pfit,
        Dfit=Dfit,
        mean_irf_fs=mean_irf_fs,
        detector=detector,
        deriv_mask=(bool(deriv_mask[0]), bool(deriv_mask[1])),
        use_exp=bool(use_exp),
        do_2nd_pass=bool(do_2nd_pass),
        smooth_window=int(smooth_window),
        n_skipped=n_skipped,
        source=getattr(data, "source", None),
    )


# --------------------------------------------------------------------------- #
#                    Step-function (Gaussian + erf step) fit                 #
# --------------------------------------------------------------------------- #
def _gaussian_step_model(p, t: np.ndarray, use_exp: bool = False) -> np.ndarray:
    """Gaussian IRF (+ derivatives) + erf step model, optionally with an exponential term.

    Without ``use_exp`` (default): ``p`` has 7 elements
    ``[t0, FWHM, G0_amp, G1_amp, G2_amp, offset, step_amp]``.
    With ``use_exp=True``: ``p`` has 9 elements
    ``[t0, FWHM, G0_amp, G1_amp, G2_amp, offset, step_amp, tau_exp, exp_amp]``,
    and an erf-broadened exponential coherent-artefact term is added.
    Uses the same sigma-parameterised Gaussian as :func:`_gaussian_irf_columns`
    (``sg = FWHM/(2*sqrt(2*ln2))``).
    """
    from scipy.special import erf

    if use_exp:
        t0, fwhm, amp0, amp1, amp2, offset, step_amp, tau, exp_amp = p
    else:
        t0, fwhm, amp0, amp1, amp2, offset, step_amp = p
    sg = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    u = t - t0
    G0 = np.exp(-(u**2) / (2.0 * sg**2))
    G1 = -(u / sg**2) * G0
    G2 = (u**2 - sg**2) / sg**4 * G0
    HS = step_amp * (1.0 + erf(u / (np.sqrt(2.0) * sg)))
    result = amp0 * G0 + amp1 * G1 + amp2 * G2 + offset + HS
    if use_exp:
        result = result + exp_amp * _erf_exp_column(u, sg, tau)
    return result


def _fit_one_pixel_step(
    y_data: np.ndarray,
    t_fit: np.ndarray,
    deriv_mask: tuple[bool, bool],
    use_exp: bool = False,
):
    """Run one Gaussian+step pixel fit; returns ``(pfit_row, dfit_col)``.

    With ``use_exp=False`` (default) ``pfit_row`` has 7 elements
    ``[t0, FWHM, G0_amp, G1_amp, G2_amp, offset, step_amp]``.
    With ``use_exp=True`` it has 9 elements, appending ``[tau_exp, exp_amp]``.
    """
    from scipy.optimize import least_squares

    from pymorgan.oneD.chirp import _gaussian_step_model

    n_params = 9 if use_exp else 7
    pfit_row = np.full(n_params, np.nan)
    dfit_col = np.zeros_like(t_fit, dtype=float)

    finite = np.isfinite(y_data) & np.isfinite(t_fit)
    if np.count_nonzero(finite) < 5:
        return pfit_row, dfit_col

    y_valid = y_data[finite]
    t_valid = t_fit[finite]

    dy = np.diff(np.abs(y_valid))
    if dy.size == 0 or not np.any(np.isfinite(dy)):
        return pfit_row, dfit_col
    idm = int(np.nanargmax(np.abs(dy)))

    t0_guess = t_valid[idm]
    pre_avg = float(np.mean(y_valid[:idm])) if idm > 0 else float(y_valid[0])
    post_avg = float(np.mean(y_valid[idm:]))
    step_amp_guess = 0.5 * (post_avg - pre_avg)
    offset_guess = pre_avg

    amp_scale = max(float(np.abs(step_amp_guess)), 1e-3)

    P0 = np.array([t0_guess, 0.1, 0.0, 0.0, 0.0, offset_guess, step_amp_guess])
    LB = np.array(
        [
            t0_guess - 1.0,
            0.0,
            -10.0 * amp_scale,
            -10.0 * amp_scale,
            -10.0 * amp_scale,
            offset_guess - 10.0 * amp_scale,
            -10.0 * amp_scale,
        ]
    )
    UB = np.array(
        [
            t0_guess + 1.0,
            1.0,
            10.0 * amp_scale,
            10.0 * amp_scale,
            10.0 * amp_scale,
            offset_guess + 10.0 * amp_scale,
            10.0 * amp_scale,
        ]
    )

    if use_exp:
        P0 = np.append(P0, [1.0, 0.0])  # tau_exp=1 ps, exp_amp=0
        LB = np.append(LB, [1e-3, -10.0 * amp_scale])
        UB = np.append(UB, [5.0, 10.0 * amp_scale])

    if not deriv_mask[0]:
        LB[3] = UB[3] = 0.0
        P0[3] = 0.0
    if not deriv_mask[1]:
        LB[4] = UB[4] = 0.0
        P0[4] = 0.0

    P0 = np.clip(P0, LB, UB)

    try:
        result = least_squares(
            lambda p: _gaussian_step_model(p, t_valid, use_exp) - y_valid,
            P0,
            bounds=(LB, UB),
            method="trf",
            xtol=5e-10,
            ftol=5e-10,
            gtol=5e-10,
            max_nfev=10000,
        )
        p = result.x
    except Exception:
        return pfit_row, dfit_col

    model_full = _gaussian_step_model(p, t_fit, use_exp)
    if not np.all(np.isfinite(model_full)):
        model_full = np.full_like(t_fit, np.nan)

    return p, model_full


def fit_chirp_step(
    data,
    *,
    detector: int = 0,
    deriv_mask: tuple[bool, bool] = (True, True),
    use_exp: bool = False,
    t_min: float = -1.0,
    t_max: float = 3.0,
    n_cauchy_terms: int = 6,
    cauchy_lambda_ref: float | None = None,
    fit_pixels: Sequence[int] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    preview_callback: Callable[[float, np.ndarray, np.ndarray, np.ndarray, float], None]
    | None = None,
) -> ChirpFit:
    """Gaussian(+derivatives)+erf-step fit at every pixel, restricted to ``[t_min, t_max]``.

    A simpler, jointly-fit (non-VARPRO) alternative to :func:`fit_chirp_automatic`.
    When ``use_exp=True`` (default ``False``), an erf-broadened exponential
    coherent-artefact term is added, extending ``Pfit`` from 7 to 9 columns
    ``[t0, FWHM, G0_amp, G1_amp, G2_amp, offset, step_amp, tau_exp, exp_amp]``.
    ``preview_callback``, if given, is called after every pixel as
    ``(wavelength, delays, y_data, model_curve, t0)`` (see
    :func:`fit_chirp_automatic`).
    """
    delays = np.asarray(data.delays, dtype=float)
    probe = data._detector_probe(detector)
    Z = data._detector_slice(detector)

    mask = (delays >= t_min) & (delays <= t_max)
    if not np.any(mask):
        raise ValueError(f"No delays fall inside [{t_min}, {t_max}].")
    t_fit = delays[mask]
    Zfit = Z[mask, :]

    idxs = np.arange(probe.size) if fit_pixels is None else np.asarray(fit_pixels, dtype=int)
    fit_wl = probe[idxs]
    n_fit = idxs.size

    n_pfit_cols = 9 if use_exp else 7
    Pfit = np.full((n_fit, n_pfit_cols), np.nan)
    Dfit = np.zeros((t_fit.size, n_fit))

    tasks = []
    for j, i in enumerate(idxs):
        tasks.append((j, (Zfit[:, i], t_fit, deriv_mask, use_exp)))

    _parallel_fit_loop(
        _fit_one_pixel_step,
        tasks,
        n_fit,
        "Fitting chirp correction...",
        Pfit,
        Dfit,
        progress_callback,
        preview_callback,
        should_cancel,
        t_fit,
        fit_wl,
    )

    valid = np.isfinite(Pfit[:, 0])
    n_skipped = int(np.sum(~valid))
    coeffs, lam_ref = fit_cauchy_dispersion(
        fit_wl[valid], Pfit[valid, 0], n_terms=n_cauchy_terms, lambda_ref=cauchy_lambda_ref
    )

    return ChirpFit(
        mode="step",
        equation="Cauchy",
        n_cauchy_terms=int(max(1, round(n_cauchy_terms))),
        lambda_ref=lam_ref,
        coeffs=coeffs,
        fit_pixels=idxs,
        fit_wl=fit_wl,
        Pfit=Pfit,
        Dfit=Dfit,
        detector=detector,
        deriv_mask=(bool(deriv_mask[0]), bool(deriv_mask[1])),
        use_exp=bool(use_exp),
        t_min=float(t_min),
        t_max=float(t_max),
        n_skipped=n_skipped,
        source=getattr(data, "source", None),
    )

# --------------------------------------------------------------------------- #
#                  Manual (interactively-picked points) fit                  #
# --------------------------------------------------------------------------- #
def fit_chirp_manual(
    data,
    points: Sequence[tuple[float, float]],
    *,
    detector: int = 0,
    n_cauchy_terms: int = 6,
    cauchy_lambda_ref: float | None = None,
) -> ChirpFit:
    """Fit the Cauchy dispersion equation to manually-picked ``(wavelength, delay)`` points.

    ``points`` is typically produced by interactively clicking the embedded
    contour map (see :class:`pymorgan.gui.picker.ContourPicker`, ``axis="xy"``).
    At least ``n_cauchy_terms`` points are required.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be an (N, 2) sequence of (wavelength, delay) pairs.")
    order = np.argsort(pts[:, 0])
    pts = pts[order]
    fit_wl = pts[:, 0]
    t0 = pts[:, 1]

    n_terms = max(1, int(round(n_cauchy_terms)))
    if fit_wl.size < n_terms:
        raise ValueError(
            f"Need at least {n_terms} point(s) to fit {n_terms} Cauchy term(s); got {fit_wl.size}."
        )
    coeffs, lam_ref = fit_cauchy_dispersion(
        fit_wl, t0, n_terms=n_terms, lambda_ref=cauchy_lambda_ref
    )

    return ChirpFit(
        mode="manual",
        equation="Cauchy",
        n_cauchy_terms=n_terms,
        lambda_ref=lam_ref,
        coeffs=coeffs,
        fit_pixels=np.array([], dtype=int),
        fit_wl=fit_wl,
        Pfit=t0.reshape(-1, 1),
        Dfit=None,
        detector=detector,
        source=getattr(data, "source", None),
    )


# --------------------------------------------------------------------------- #
#                 Wavelet (edge-detection based onset) fit                    #
# --------------------------------------------------------------------------- #
def _wavelet_conv_modulus(
    y: np.ndarray,
    delays: np.ndarray,
    t_uniform: np.ndarray,
    dt: float,
    omega_w_val: float,
    gamma_w: float,
) -> np.ndarray:
    """Compute the complex wavelet convolution modulus on a uniform grid with edge padding."""
    finite = np.isfinite(y) & np.isfinite(delays)
    if np.count_nonzero(finite) < 5 or np.allclose(y[finite], 0.0):
        return np.zeros_like(t_uniform)

    y_valid = y[finite]
    delays_valid = delays[finite]

    # Interpolate onto uniform grid
    y_uniform = np.interp(t_uniform, delays_valid, y_valid, left=0.0, right=0.0)

    # Pad the signal at both ends to avoid wrap-around edge effects
    # We pad by 1.5 ps (which is > 6 * gamma_w)
    pad_ps = 1.5
    npad = int(np.ceil(pad_ps / dt))
    y_padded = np.pad(y_uniform, npad, mode="edge")

    M_padded = len(y_padded)
    t_wavelet = np.fft.fftfreq(M_padded, d=1.0) * (M_padded * dt)

    # Complex Morlet wavelet
    omega_w = 2.0 * np.pi / omega_w_val
    psi = np.exp(1j * omega_w * t_wavelet - (t_wavelet / gamma_w) ** 2)

    # FFT convolution
    fft_y = np.fft.fft(y_padded)
    fft_psi = np.fft.fft(psi)
    conv_padded = np.fft.ifft(fft_y * fft_psi) * dt

    # Crop back to the original size
    conv = conv_padded[npad : npad + len(t_uniform)]

    return np.abs(conv)


def _find_peak_subpixel(
    modulus: np.ndarray, t_uniform: np.ndarray, idx_range=None
) -> tuple[float, int]:
    """Find peak of modulus with sub-pixel resolution via parabolic fit."""
    if idx_range is not None and len(idx_range) > 0:
        sub_modulus = modulus[idx_range]
        if len(sub_modulus) == 0:
            return np.nan, -1
        sub_k = np.argmax(sub_modulus)
        k = idx_range[sub_k]
    else:
        k = np.argmax(modulus)

    M = len(modulus)
    if k <= 0 or k >= M - 1:
        return float(t_uniform[k]), k

    y_minus = modulus[k - 1]
    y_zero = modulus[k]
    y_plus = modulus[k + 1]

    denom = y_minus - 2.0 * y_zero + y_plus
    if denom < 0.0:
        delta = 0.5 * (y_minus - y_plus) / denom
        dt = t_uniform[1] - t_uniform[0]
        t0 = t_uniform[k] + delta * dt
    else:
        t0 = t_uniform[k]

    return float(t0), k


def _optimise_wavelet_parameters(
    data,
    *,
    detector: int,
    t_min: float,
    t_max: float,
    omega_w_init: float,
    gamma_w_init: float,
    n_cauchy_terms: int,
    cauchy_lambda_ref: float | None,
    smooth_window: int,
    fit_pixels: Sequence[int] | None,
    should_cancel: Callable[[], bool] | None,
    opt_max_iter: int,
    opt_progress_callback: Callable[[int, str, float], None] | None,
) -> tuple[float, float, float]:
    """Iteratively optimise wavelet parameters (omega_w, gamma_w) to minimise the
    RMS dispersion of the fitted t0 points around the Cauchy model.

    The objective function is: RMS of ``t0[valid] - cauchy_t0(coeffs, wl[valid])``.
    Minimising this is equivalent to finding the wavelet shape that produces the
    smoothest (most continuous) dispersion curve -- jumps or noise in t0 increase
    the residual. Only a single-pass wavelet scan is run per evaluation (no 2nd
    pass), keeping each evaluation fast. The full 2nd-pass refinement is applied
    once after convergence.

    Returns ``(best_omega_w, best_gamma_w, best_rms_fs)``.
    """
    from scipy.optimize import minimize

    delays = np.asarray(data.delays, dtype=float)
    probe = data._detector_probe(detector)
    Z = data._detector_slice(detector)

    mask_window = (delays >= t_min) & (delays <= t_max)
    diffs = np.diff(delays[mask_window])
    dt = float(np.min(diffs)) if diffs.size > 0 else 0.01
    dt = max(dt, 0.001)
    t_uniform = np.arange(t_min, t_max + dt / 2.0, dt)

    idxs = np.arange(probe.size) if fit_pixels is None else np.asarray(fit_pixels, dtype=int)
    fit_wl = probe[idxs]
    n_fit = idxs.size
    n_terms = max(1, int(round(n_cauchy_terms)))

    iter_count = [0]

    def _objective(log_params):
        if should_cancel is not None and should_cancel():
            raise InterruptedError("Chirp fit cancelled.")

        ow = float(np.exp(log_params[0]))
        gw = float(np.exp(log_params[1]))

        t0_vals = np.full(n_fit, np.nan)
        for j, i in enumerate(idxs):
            y = Z[:, i]
            finite = np.isfinite(y) & np.isfinite(delays)
            if np.count_nonzero(finite) < 5 or np.allclose(y[finite], 0.0):
                continue
            modulus = _wavelet_conv_modulus(y, delays, t_uniform, dt, ow, gw)
            t0_val, _ = _find_peak_subpixel(modulus, t_uniform)
            t0_vals[j] = t0_val

        valid = np.isfinite(t0_vals)
        if np.count_nonzero(valid) < n_terms:
            return 1e6  # Not enough points for Cauchy fit

        try:
            lb = np.concatenate(([-5.0], -1e2 * np.ones(n_terms - 1)))
            ub = np.concatenate(([5.0], 1e2 * np.ones(n_terms - 1)))
            coeffs, lam_ref = fit_cauchy_dispersion(
                fit_wl[valid],
                t0_vals[valid],
                n_terms=n_terms,
                lambda_ref=cauchy_lambda_ref,
                bounds=(lb, ub),
            )
            resid = t0_vals[valid] - cauchy_t0(coeffs, fit_wl[valid], lam_ref)
            rms = float(np.sqrt(np.mean(resid**2)))
        except Exception:
            return 1e6

        iter_count[0] += 1
        if opt_progress_callback is not None:
            opt_progress_callback(iter_count[0], f"ω={ow * 1000:.1f} fs, γ={gw * 1000:.1f} fs", rms * 1000.0)

        return rms

    # Search in log-space, bounded to ±1.5 decades around the initial guess
    log_init = np.array([np.log(omega_w_init), np.log(gamma_w_init)])
    log_bounds = [
        (log_init[0] - np.log(10) * 1.5, log_init[0] + np.log(10) * 1.5),
        (log_init[1] - np.log(10) * 1.5, log_init[1] + np.log(10) * 1.5),
    ]

    try:
        result = minimize(
            _objective,
            log_init,
            method="Nelder-Mead",
            options={"maxiter": max(1, int(opt_max_iter)), "xatol": 1e-4, "fatol": 1e-5},
        )
        best_log = result.x
        # Clamp to search bounds
        best_log = np.clip(
            best_log,
            [b[0] for b in log_bounds],
            [b[1] for b in log_bounds],
        )
        best_omega_w = float(np.exp(best_log[0]))
        best_gamma_w = float(np.exp(best_log[1]))
        best_rms_fs = float(result.fun) * 1000.0
    except InterruptedError:
        raise
    except Exception:
        best_omega_w = omega_w_init
        best_gamma_w = gamma_w_init
        best_rms_fs = float("nan")

    return best_omega_w, best_gamma_w, best_rms_fs


def fit_chirp_wavelet(
    data,
    *,
    detector: int = 0,
    t_min: float = -1.0,
    t_max: float = 3.0,
    omega_w: float = 0.125,
    gamma_w: float = 0.225,
    n_cauchy_terms: int = 6,
    cauchy_lambda_ref: float | None = None,
    do_2nd_pass: bool = True,
    smooth_window: int = 10,
    optimise_wavelet: bool = False,
    opt_max_iter: int = 100,
    opt_progress_callback: Callable[[int, str, float], None] | None = None,
    fit_pixels: Sequence[int] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    preview_callback: Callable[[float, np.ndarray, np.ndarray, np.ndarray, float], None]
    | None = None,
) -> ChirpFit:
    """Wavelet-based edge-detection fit at every probe pixel + Cauchy dispersion fit.

    Based on the method by Kefer et al. (Applied Optics 2024). Convolves raw
    kinetics with a Morlet wavelet to isolate edge-like signal onsets (modulus peak).

    When ``optimise_wavelet=True``, the wavelet parameters ``omega_w`` and ``gamma_w``
    are refined iteratively before the main per-pixel scan. A Nelder-Mead simplex
    optimiser minimises the RMS of the ``t0`` residuals around the fitted Cauchy
    dispersion curve -- i.e. it finds the wavelet shape that yields the smoothest,
    most continuous dispersion trace. The optimal parameters replace the user-supplied
    initial values, and are stored in the returned :class:`ChirpFit` object
    (``fit.omega_w``, ``fit.gamma_w``). ``opt_progress_callback``, if given, is called
    after every optimiser iteration as ``(iter_number, param_summary_str, rms_fs)``.
    """
    delays = np.asarray(data.delays, dtype=float)
    probe = data._detector_probe(detector)
    Z = data._detector_slice(detector)

    # Determine uniform grid spacing dt
    mask_window = (delays >= t_min) & (delays <= t_max)
    if not np.any(mask_window):
        raise ValueError(f"No delays fall inside [{t_min}, {t_max}].")

    # Calculate spacing (avoid dividing by zero if only one point)
    diffs = np.diff(delays[mask_window])
    dt = float(np.min(diffs)) if diffs.size > 0 else 0.01
    dt = max(dt, 0.001)  # Ensure at least 1 fs step size

    t_uniform = np.arange(t_min, t_max + dt / 2.0, dt)

    idxs = np.arange(probe.size) if fit_pixels is None else np.asarray(fit_pixels, dtype=int)
    fit_wl = probe[idxs]
    n_fit = idxs.size

    Pfit = np.full((n_fit, 1), np.nan)

    def _run_pass(label, t0_guess_fn=None):
        for j, i in enumerate(idxs):
            if should_cancel is not None and should_cancel():
                raise InterruptedError("Chirp fit cancelled.")

            y = Z[:, i]
            finite = np.isfinite(y) & np.isfinite(delays)
            if np.count_nonzero(finite) < 5 or np.allclose(y[finite], 0.0):
                Pfit[j, 0] = np.nan
                continue

            modulus = _wavelet_conv_modulus(y, delays, t_uniform, dt, omega_w, gamma_w)

            idx_range = None
            if t0_guess_fn is not None:
                t0_guess = t0_guess_fn(j)
                if np.isfinite(t0_guess):
                    # Search within +/- 1.0 ps of guess
                    idx_range = np.where(
                        (t_uniform >= t0_guess - 1.0) & (t_uniform <= t0_guess + 1.0)
                    )[0]

            t0, _ = _find_peak_subpixel(modulus, t_uniform, idx_range)
            Pfit[j, 0] = t0

            if progress_callback is not None:
                progress_callback(j + 1, n_fit, label)

            if preview_callback is not None:
                # Scale modulus for plotting
                y_interp = np.interp(t_uniform, delays[finite], y[finite], left=0.0, right=0.0)
                max_abs_y = float(np.max(np.abs(y_interp)))
                max_mod = float(np.max(modulus))
                dcol = modulus * (max_abs_y / max_mod) if max_mod > 1e-12 else modulus
                preview_callback(float(fit_wl[j]), t_uniform, y_interp, dcol, float(t0))

    # --- Optional wavelet parameter optimisation --------------------------------
    if optimise_wavelet:
        omega_w, gamma_w, _opt_rms_fs = _optimise_wavelet_parameters(
            data,
            detector=detector,
            t_min=t_min,
            t_max=t_max,
            omega_w_init=omega_w,
            gamma_w_init=gamma_w,
            n_cauchy_terms=n_cauchy_terms,
            cauchy_lambda_ref=cauchy_lambda_ref,
            smooth_window=smooth_window,
            fit_pixels=fit_pixels,
            should_cancel=should_cancel,
            opt_max_iter=opt_max_iter,
            opt_progress_callback=opt_progress_callback,
        )

    _run_pass("Pass 1: detecting wavelet edge peaks...")


    if do_2nd_pass:
        valid_p1 = np.isfinite(Pfit[:, 0])
        n_terms = max(1, int(round(n_cauchy_terms)))
        if np.count_nonzero(valid_p1) >= n_terms:
            try:
                lb = np.concatenate(([-5.0], -1e2 * np.ones(n_terms - 1)))
                ub = np.concatenate(([5.0], 1e2 * np.ones(n_terms - 1)))
                coeffs, lam_ref = fit_cauchy_dispersion(
                    fit_wl[valid_p1],
                    Pfit[valid_p1, 0],
                    n_terms=n_terms,
                    lambda_ref=cauchy_lambda_ref,
                    bounds=(lb, ub),
                )
                t0_init = cauchy_t0(coeffs, fit_wl, lam_ref)
            except Exception:
                t0_init = _interpolate_nans(_movmedian(Pfit[:, 0], smooth_window))
        else:
            t0_init = _interpolate_nans(_movmedian(Pfit[:, 0], smooth_window))

        def _t0_guess(j):
            return t0_init[j]

        _run_pass("Pass 2: refining edge peaks...", _t0_guess)

    valid = np.isfinite(Pfit[:, 0])
    n_skipped = int(np.sum(~valid))
    coeffs, lam_ref = fit_cauchy_dispersion(
        fit_wl[valid], Pfit[valid, 0], n_terms=n_cauchy_terms, lambda_ref=cauchy_lambda_ref
    )

    logger.info(
        "\n%s\nWavelet Edge-Detection Chirp Fit Completed.\nMethod Reference:\n"
        "  Kefer, O.; Buckup, T.; Kolesnichenko, P. V. Retroactive correction for\n"
        "  white-light dispersion as an edge-detection problem in ultrafast spectroscopies.\n"
        "  Applied Optics 2024, 63, 15, 4015-4024.\n"
        "  DOI: 10.1364/AO.532878\n%s\n",
        "=" * 80,
        "=" * 80,
    )

    return ChirpFit(
        mode="wavelet",
        equation="Cauchy",
        n_cauchy_terms=int(max(1, round(n_cauchy_terms))),
        lambda_ref=lam_ref,
        coeffs=coeffs,
        fit_pixels=idxs,
        fit_wl=fit_wl,
        Pfit=Pfit,
        Dfit=None,
        detector=detector,
        t_min=float(t_min),
        t_max=float(t_max),
        do_2nd_pass=bool(do_2nd_pass),
        smooth_window=int(smooth_window),
        n_skipped=n_skipped,
        omega_w=float(omega_w),
        gamma_w=float(gamma_w),
        source=getattr(data, "source", None),
    )


# --------------------------------------------------------------------------- #
#                            Diagnostic plotting                             #
# --------------------------------------------------------------------------- #
def plot_chirp_diagnostics(data, fit: ChirpFit, *, detector: int | None = None) -> list:
    """Build the chirp-fit review figures: dispersion overlay, fit+residuals,
    and (for ``mode in {"automatic", "step"}``) IRF FWHM vs wavelength, all
    combined into a single dashboard figure using Matplotlib subfigures.

    Layout:
    [contour] [fig2 (dispersion fit + residuals)]
    [still contour] [fig3 (IRF FWHM vs wavelength)]

    Returns a list containing the new single figure.
    """
    import types
    import warnings

    import matplotlib.pyplot as plt

    detector = fit.detector if detector is None else detector
    probe = data._detector_probe(detector)
    t0 = np.asarray(fit.Pfit[:, 0], dtype=float)
    valid = np.isfinite(t0)
    curve = cauchy_t0(fit.coeffs, probe, fit.lambda_ref)

    has_fwhm = fit.mode in ("automatic", "step") and fit.Pfit.shape[1] >= 2

    # We use two independent GridSpecs via subfigures.
    figsize = (12.6, 7.2) if has_fwhm else (12.6, 4.95)
    fig = plt.figure(figsize=figsize)

    # Divide the figure horizontally into left (contour) and right (fits/FWHM) columns
    subfigs = fig.subfigures(1, 2, width_ratios=[1.65, 1.0], wspace=0.085)
    sub_left = subfigs[0]
    sub_right = subfigs[1]

    # Helper to add dummy layout methods to subfigures to make them safe for core plotting helpers
    def make_subfigure_safe(sf) -> None:
        if not hasattr(sf, "set_layout_engine"):
            sf.set_layout_engine = types.MethodType(lambda self, engine: None, sf)
        if not hasattr(sf, "tight_layout"):
            sf.tight_layout = types.MethodType(lambda self, *args, **kwargs: None, sf)

    make_subfigure_safe(sub_left)
    make_subfigure_safe(sub_right)

    if has_fwhm:
        sub_right_split = sub_right.subfigures(2, 1, height_ratios=[1.3, 1.0], hspace=0.095)
        sub_right_top = sub_right_split[0]
        sub_right_bottom = sub_right_split[1]
        make_subfigure_safe(sub_right_top)
        make_subfigure_safe(sub_right_bottom)
    else:
        sub_right_top = sub_right
        sub_right_bottom = None
        make_subfigure_safe(sub_right_top)

    # --- Left Column: contour + dispersion overlay ---------------------- #
    sub_left.subplots_adjust(left=0.11, right=0.88, bottom=0.11, top=0.93)
    ax1 = sub_left.add_subplot(111)

    # Suppress tight_layout warnings during contour plotting
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*not compatible with tight_layout.*",
        )
        data.plot_contour(ax=ax1, detector=detector, Zscale=10, Yscale="lin")

    ax1.plot(fit.fit_wl[valid], t0[valid], "o", mfc="none", mec="0.5", ms=6, label="Fitted $t_0$")
    ax1.plot(probe, curve, "-", color="gold", lw=3, alpha=0.75, label="Dispersion fit")
    if np.any(valid):
        centre = float(np.nanmean(t0[valid]))
        ax1.set_ylim(centre - 2.0, centre + 2.0)
    ax1.legend(loc="best")
    ax1.set_title("Fitted dispersion curve (contour overlay)")

    # --- Right Column (Top): dispersion fit + residuals ----------------- #
    sub_right_top.subplots_adjust(left=0.185, right=0.935, bottom=0.175, top=0.89)

    ax2a, ax2b = sub_right_top.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}
    )

    ax2a.plot(fit.fit_wl[valid], t0[valid], "o", mfc="none", mec="0.5", ms=8)
    ax2a.plot(probe, curve, "-r", lw=2.5, alpha=0.75)
    ax2a.set_ylabel("Delay (ps)")
    ax2a.set_title("Fitted dispersion curve & residuals")

    res_fs = 1000.0 * (t0[valid] - cauchy_t0(fit.coeffs, fit.fit_wl[valid], fit.lambda_ref))
    ax2b.plot(fit.fit_wl[valid], res_fs, "o", mfc="none", mec="0.5", ms=8)
    ax2b.axhline(0.0, color="0.4", lw=1)
    ax2b.set_ylim(-50, 50)
    if has_fwhm:
        ax2b.set_xlabel("")  # Hide redundant x-label when FWHM plot is present to prevent overlap
    else:
        ax2b.set_xlabel("Wavelength (nm)")
    ax2b.set_ylabel(r"$\Delta t_0^{res}$ (fs)")

    # --- Right Column (Bottom): IRF FWHM vs wavelength ------------------ #
    if has_fwhm and sub_right_bottom is not None:
        sub_right_bottom.subplots_adjust(left=0.185, right=0.935, bottom=0.25, top=0.865)
        ax3 = sub_right_bottom.add_subplot(111)
        fwhm_fs = fit.Pfit[:, 1] * 1000.0  # convert ps->fs for plotting
        fwhm_valid = np.isfinite(fwhm_fs)
        ax3.plot(
            fit.fit_wl[fwhm_valid],
            fwhm_fs[fwhm_valid],
            "o",
            mfc="none",
            mec="0.5",
            ms=8,
            alpha=0.75,
        )
        mean_irf = float(np.nanmean(fwhm_fs)) if np.any(fwhm_valid) else float("nan")
        ax3.axhline(mean_irf, ls="--", lw=2, color="b")
        ax3.set_xlabel("Wavelength (nm)")
        ax3.set_ylabel("IRF FWHM (fs)")
        ax3.set_ylim(0, 500)
        ax3.set_title(f"Fitted IRF FWHM - Avg: {mean_irf:.1f} fs")

    # Manually adjust ax2a and ax2b to be vertically adjacent with a tiny gap
    # (since they share the x-axis), preserving their gridspec width/x-position.
    pos_a = ax2a.get_position()
    pos_b = ax2b.get_position()
    gap = 0.015
    new_bottom = pos_b.y1 + gap
    new_height = pos_a.y1 - new_bottom
    ax2a.set_position([pos_a.x0, new_bottom, pos_a.width, new_height])

    return [fig]

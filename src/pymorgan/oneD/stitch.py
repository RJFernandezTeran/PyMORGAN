"""Dataset stitching algorithms for 1D Transient Absorption Spectroscopy.

Provides:
- Wavelength stitching (e.g. UV/Vis + NIR spectral window merging) with
  linear blending or hard cutover, and overlap scaling factor optimization.
- Timescale stitching (e.g. femtosecond + nanosecond delay merging) with
  unit conversion, linear spectrometer calibration alignment, and
  wavelength-dependent quadratic scaling factor estimation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import scipy.optimize

from pymorgan.log import get_logger

if TYPE_CHECKING:
    from pymorgan.oneD.dataset import Dataset1D

logger = get_logger(__name__)


@dataclass
class StitchResult:
    """Container for dataset stitching results, scaling parameters, and provenance.

    Attributes
    ----------
    dataset : Dataset1D
        The resulting stitched Dataset1D object.
    kind : {"wavelength", "timescale"}
        The type of stitching performed.
    scaling_factor : float | np.ndarray
        Scalar multiplier or 1D array of wavelength-dependent scaling factors.
    scaling_model : {"scalar", "quadratic", "manual"}
        Model used to compute the scaling factor.
    stitch_region : tuple[float, float]
        Overlap interval used for scaling estimation.
    blend_window : tuple[float, float] | None
        Transition window for blending.
    cutover_val : float | None
        Cutover threshold (if hard cutover).
    blend_method : {"cubic", "linear", "quintic", "cosine", "step"}
        Transition weighting function used for blending.
    wl_shift_params : tuple[float, float] | None
        Linear wavelength shift parameters (slope m, intercept b) if applied.
    scaling_params : dict[str, Any]
        Additional fit diagnostics, polynomial coefficients, or SSR.
    """

    dataset: Dataset1D
    kind: Literal["wavelength", "timescale"]
    scaling_factor: float | np.ndarray
    scaling_model: str = "scalar"
    stitch_region: tuple[float, float] = (0.0, 0.0)
    blend_window: tuple[float, float] | None = None
    cutover_val: float | None = None
    blend_method: str = "cubic"
    wl_shift_params: tuple[float, float] | None = None
    scaling_params: dict[str, Any] = field(default_factory=dict)

    @property
    def delays(self) -> np.ndarray:
        """1D delays array of the stitched dataset."""
        return self.dataset.delays

    @property
    def probe(self) -> np.ndarray:
        """1D probe array of the stitched dataset."""
        return self.dataset.probe

    @property
    def Z(self) -> np.ndarray:
        """Stitched 2D data matrix."""
        return self.dataset.Z

    def to_dataset(self) -> Dataset1D:
        """Return the stitched Dataset1D object."""
        return self.dataset

    def export(self, filepath: str | Path, file_format: str | None = None) -> None:
        """Export stitched dataset to file (PDAT, HDF5, or CSV)."""
        path = Path(filepath)
        fmt = (file_format or path.suffix).lower().lstrip(".")
        if fmt in ("pdat", "txt"):
            from pymorgan.oneD.load import export_pdat

            export_pdat(self.dataset, path)
        elif fmt in ("h5", "hdf5"):
            self.dataset.save_hdf5(path)
        elif fmt == "csv":
            import csv

            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Delay / Probe"] + [str(p) for p in self.probe])
                mat = self.dataset._detector_slice(0)
                for t_idx, d in enumerate(self.delays):
                    writer.writerow([str(d)] + [f"{v:.6e}" for v in mat[t_idx, :]])
        else:
            raise ValueError(f"Unsupported export format '{fmt}'. Choose 'pdat', 'h5', or 'csv'.")


def _compute_blend_weight(u: np.ndarray, method: str = "cubic") -> np.ndarray:
    """Compute normalized blend weight w(u) in [0, 1] for dataset 2.

    Parameters
    ----------
    u : np.ndarray
        Normalized coordinate from 0.0 (dataset 1 pure) to 1.0 (dataset 2 pure).
    method : {"cubic", "linear", "quintic", "cosine", "step"}
        Transition weighting function:
        - "cubic": Cubic Hermite polynomial (3u^2 - 2u^3), C^1 smooth with zero boundary slopes.
        - "linear": Linear cross-fade ramp (u).
        - "quintic": Quintic smootherstep polynomial (6u^5 - 15u^4 + 10u^3), C^2 smooth.
        - "cosine": Half-period raised cosine (0.5 * (1 - cos(pi * u))), C^1 smooth.
        - "step": Direct step threshold (0 for u < 0.5, 1 for u >= 0.5).
    """
    u_clamped = np.clip(u, 0.0, 1.0)
    m = str(method).lower().strip()
    if m in ("cubic", "polynomial", "smoothstep"):
        return 3.0 * (u_clamped ** 2) - 2.0 * (u_clamped ** 3)
    elif m in ("quintic", "smootherstep"):
        return 6.0 * (u_clamped ** 5) - 15.0 * (u_clamped ** 4) + 10.0 * (u_clamped ** 3)
    elif m == "cosine":
        return 0.5 * (1.0 - np.cos(np.pi * u_clamped))
    elif m in ("step", "hard", "cutover"):
        return np.where(u_clamped >= 0.5, 1.0, 0.0)
    return u_clamped


def stitch_wavelength(
    ds1: Dataset1D,
    ds2: Dataset1D,
    stitch_region: tuple[float, float] | None = None,
    mode: Literal["blend", "cutover"] = "blend",
    blend_window: tuple[float, float] | None = None,
    cutover_wl: float | None = None,
    time_threshold: float = -0.1,
    scale_override: float | None = None,
    target_delays: Literal["ds2", "ds1"] | np.ndarray = "ds2",
    detector1: int = 0,
    detector2: int = 0,
    blend_method: Literal["cubic", "linear", "quintic", "cosine", "step"] = "cubic",
) -> StitchResult:
    """Stitch two datasets with different spectral ranges (e.g. UV/Vis + NIR).

    Parameters
    ----------
    ds1 : Dataset1D
        Dataset 1 (typically the shorter wavelength window, e.g. UV/Vis).
    ds2 : Dataset1D
        Dataset 2 (typically the longer wavelength window, e.g. NIR).
    stitch_region : tuple[float, float], optional
        Wavelength range ``(wl_min, wl_max)`` over which the scaling factor C is
        fitted (such that C * ds1 ~ ds2). If None, automatically set to the inner
        80% of the overlapping probe range.
    mode : {"blend", "cutover"}, default "blend"
        Stitching transition mode:
        - ``"blend"``: Weighted linear cross-fade ramp across ``blend_window``.
        - ``"cutover"``: Direct concatenation at ``cutover_wl`` without interpolation.
    blend_window : tuple[float, float], optional
        Wavelength transition window ``(blend_start, blend_end)``. Defaults to
        the overlap region.
    cutover_wl : float, optional
        Wavelength threshold below which ds1 is kept and above which ds2 is kept.
        Defaults to the midpoint of the overlap region.
    time_threshold : float, default -0.1
        Minimum delay time for including kinetic points in the scaling fit.
    scale_override : float, optional
        Manual scaling factor to use instead of the least-squares optimal factor.
    target_delays : {"ds2", "ds1"} or np.ndarray, default "ds2"
        Delay grid onto which the non-matching dataset is interpolated.
    detector1, detector2 : int, default 0
        Detector channels if multi-detector datasets.

    Returns
    -------
    StitchResult
        Result containing the stitched Dataset1D and fit metadata.
    """
    from pymorgan.oneD.dataset import Dataset1D

    # 1. Resolve probe and delay axes
    wl1 = np.asarray(ds1._detector_probe(detector1), dtype=float)
    wl2 = np.asarray(ds2._detector_probe(detector2), dtype=float)
    t1 = np.asarray(ds1.delays, dtype=float)
    t2 = np.asarray(ds2.delays, dtype=float)

    z1 = np.asarray(ds1._detector_slice(detector1), dtype=float)
    z2 = np.asarray(ds2._detector_slice(detector2), dtype=float)

    # Sort probe axes ascending if descending
    if np.any(np.diff(wl1) < 0):
        s1 = np.argsort(wl1)
        wl1, z1 = wl1[s1], z1[:, s1]
    if np.any(np.diff(wl2) < 0):
        s2 = np.argsort(wl2)
        wl2, z2 = wl2[s2], z2[:, s2]

    # Ensure ds1 is the lower wavelength dataset
    swapped = False
    if np.mean(wl1) > np.mean(wl2):
        wl1, wl2 = wl2, wl1
        t1, t2 = t2, t1
        z1, z2 = z2, z1
        swapped = True

    # 2. Determine target delays
    if isinstance(target_delays, str):
        t_out = np.copy(t2) if target_delays.lower() == "ds2" else np.copy(t1)
    else:
        t_out = np.asarray(target_delays, dtype=float)

    # 3. Align time axis (interpolate z1 and z2 onto t_out)
    z1_aligned = np.zeros((len(t_out), len(wl1)), dtype=float)
    for c in range(len(wl1)):
        z1_aligned[:, c] = np.interp(t_out, t1, z1[:, c], left=z1[0, c], right=z1[-1, c])

    z2_aligned = np.zeros((len(t_out), len(wl2)), dtype=float)
    for c in range(len(wl2)):
        z2_aligned[:, c] = np.interp(t_out, t2, z2[:, c], left=z2[0, c], right=z2[-1, c])

    # 4. Overlap region determination
    detected_overlap_min = max(float(wl1[0]), float(wl2[0]))
    detected_overlap_max = min(float(wl1[-1]), float(wl2[-1]))

    if detected_overlap_min >= detected_overlap_max:
        raise ValueError(
            f"No spectral overlap detected between Dataset 1 [{wl1[0]:.1f}, {wl1[-1]:.1f}] "
            f"and Dataset 2 [{wl2[0]:.1f}, {wl2[-1]:.1f}]."
        )

    if stitch_region is None:
        span = detected_overlap_max - detected_overlap_min
        stitch_region = (detected_overlap_min + 0.1 * span, detected_overlap_max - 0.1 * span)

    fit_min, fit_max = sorted(stitch_region)

    # 5. Scaling factor optimization
    if scale_override is not None:
        scaling_factor = float(scale_override)
        scale_model = "manual"
    else:
        eval_grid = np.linspace(fit_min, fit_max, 35)
        z1_eval = np.zeros((len(t_out), len(eval_grid)), dtype=float)
        z2_eval = np.zeros((len(t_out), len(eval_grid)), dtype=float)

        for r in range(len(t_out)):
            z1_eval[r, :] = np.interp(eval_grid, wl1, z1_aligned[r, :])
            z2_eval[r, :] = np.interp(eval_grid, wl2, z2_aligned[r, :])

        mask_t = t_out >= time_threshold
        if not np.any(mask_t):
            mask_t = np.ones(len(t_out), dtype=bool)

        a_vec = z1_eval[mask_t, :].ravel()
        b_vec = z2_eval[mask_t, :].ravel()

        # Least squares C * a ~ b
        denom = float(np.dot(a_vec, a_vec))
        if denom > 1e-15:
            scaling_factor = float(np.dot(a_vec, b_vec) / denom)
        else:
            scaling_factor = 1.0
        scale_model = "scalar"

    # Scale ds1
    z1_scaled = z1_aligned * scaling_factor

    # 6. Joining Data
    if mode == "cutover":
        cut_wl = (
            cutover_wl
            if cutover_wl is not None
            else (detected_overlap_min + detected_overlap_max) / 2.0
        )
        idx_keep_1 = wl1 < cut_wl
        idx_keep_2 = wl2 >= cut_wl

        out_wl = np.concatenate([wl1[idx_keep_1], wl2[idx_keep_2]])
        out_z = np.concatenate([z1_scaled[:, idx_keep_1], z2_aligned[:, idx_keep_2]], axis=1)
        actual_blend = None
        actual_cutover = float(cut_wl)

    else:  # mode == "blend"
        if blend_window is None:
            blend_start = detected_overlap_min + 0.15 * (detected_overlap_max - detected_overlap_min)
            blend_end = detected_overlap_max - 0.15 * (detected_overlap_max - detected_overlap_min)
        else:
            blend_start, blend_end = sorted(blend_window)

        idx_pure_1 = wl1 < blend_start
        idx_pure_2 = wl2 > blend_end

        # Blend grid matching median spacing of wl2
        step = float(np.median(np.diff(wl2)))
        n_blend = max(3, int(np.round((blend_end - blend_start) / max(step, 1e-4))))
        grid_blend = np.linspace(blend_start, blend_end, n_blend)

        part_1 = z1_scaled[:, idx_pure_1]
        part_2 = z2_aligned[:, idx_pure_2]

        # Interpolate both onto blend grid
        z1_blend = np.zeros((len(t_out), len(grid_blend)), dtype=float)
        z2_blend = np.zeros((len(t_out), len(grid_blend)), dtype=float)
        for r in range(len(t_out)):
            z1_blend[r, :] = np.interp(grid_blend, wl1, z1_scaled[r, :])
            z2_blend[r, :] = np.interp(grid_blend, wl2, z2_aligned[r, :])

        # Compute weights using selected blend method
        denom_wl = max(blend_end - blend_start, 1e-12)
        u_blend = (grid_blend - blend_start) / denom_wl
        w2 = _compute_blend_weight(u_blend, blend_method)
        w1 = 1.0 - w2
        part_blend = w1[np.newaxis, :] * z1_blend + w2[np.newaxis, :] * z2_blend

        out_wl = np.concatenate([wl1[idx_pure_1], grid_blend, wl2[idx_pure_2]])
        out_z = np.concatenate([part_1, part_blend, part_2], axis=1)
        actual_blend = (float(blend_start), float(blend_end))
        actual_cutover = None

    # 7. Construct Resulting Dataset1D
    units = dict(ds1.units or {})
    src_name = f"Wavelength-Stitched ({getattr(ds1, 'source', 'ds1')} + {getattr(ds2, 'source', 'ds2')})"
    res_ds = Dataset1D(
        Zavg_R=out_z,
        delays=np.copy(t_out),
        probe=np.copy(out_wl),
        units=units,
        source=src_name,
    )
    res_ds.Zavg_C = np.copy(out_z)

    return StitchResult(
        dataset=res_ds,
        kind="wavelength",
        scaling_factor=scaling_factor,
        scaling_model=scale_model,
        stitch_region=(float(fit_min), float(fit_max)),
        blend_window=actual_blend,
        cutover_val=actual_cutover,
        blend_method=blend_method,
        scaling_params={"swapped": swapped, "time_threshold": time_threshold},
    )


def _optimize_linear_wl_shift(
    wl_ref: np.ndarray,
    spec_ref: np.ndarray,
    wl_moving: np.ndarray,
    spec_moving: np.ndarray,
    scatter_region: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Fit optimal slope m and intercept b such that spec_moving(m * wl + b) aligns with spec_ref."""
    mask_ref = np.ones(len(wl_ref), dtype=bool)
    if scatter_region is not None:
        s_min, s_max = sorted(scatter_region)
        mask_ref &= (wl_ref < s_min) | (wl_ref > s_max)

    target_wl = wl_ref[mask_ref]
    target_spec = spec_ref[mask_ref]

    # Normalize reference
    t_min, t_max = np.min(target_spec), np.max(target_spec)
    if t_max - t_min > 1e-12:
        norm_ref = (target_spec - t_min) / (t_max - t_min)
    else:
        norm_ref = target_spec

    def cost(params):
        m, b = params
        shifted_wl = m * wl_moving + b
        interp_spec = np.interp(target_wl, shifted_wl, spec_moving, left=np.nan, right=np.nan)
        valid = ~np.isnan(interp_spec)
        if np.sum(valid) < 5:
            return 1e6
        i_min, i_max = np.min(interp_spec[valid]), np.max(interp_spec[valid])
        if i_max - i_min > 1e-12:
            norm_moving = (interp_spec[valid] - i_min) / (i_max - i_min)
        else:
            norm_moving = interp_spec[valid]
        return float(np.sum((norm_moving - norm_ref[valid]) ** 2))

    res = scipy.optimize.minimize(cost, [1.0, 0.0], method="Nelder-Mead")
    if res.success:
        return float(res.x[0]), float(res.x[1])
    return 1.0, 0.0


def stitch_time(
    ds_fs: Dataset1D,
    ds_ns: Dataset1D,
    time_overlap: tuple[float, float] = (0.5, 2.0),
    blend_window: tuple[float, float] = (1.0, 1.7),
    scale_model: Literal["quadratic", "scalar"] = "quadratic",
    scale_override: float | None = None,
    align_wl: bool = True,
    cut_scatter: bool = False,
    scatter_region: tuple[float, float] = (500.0, 550.0),
    detector_fs: int = 0,
    detector_ns: int = 0,
    blend_method: Literal["cubic", "linear", "quintic", "cosine", "step"] = "cubic",
) -> StitchResult:
    """Stitch femtosecond and nanosecond/microsecond datasets across timescales.

    Parameters
    ----------
    ds_fs : Dataset1D
        Ultrafast (femtosecond) dataset. If delays are in ps, they are automatically
        converted to ns.
    ds_ns : Dataset1D
        Long-delay (nanosecond) dataset with delays in ns.
    time_overlap : tuple[float, float], default (0.5, 2.0)
        Delay window in ns where the scaling factor is calculated.
    blend_window : tuple[float, float], default (1.0, 1.7)
        Delay transition window in ns over which the blend weight ramp is applied.
    scale_model : {"quadratic", "scalar"}, default "quadratic"
        - ``"quadratic"``: Fit a smooth quadratic polynomial S(lambda) across wavelengths.
        - ``"scalar"``: Fit a single global scalar multiplier C.
    scale_override : float, optional
        Manual scalar scaling multiplier.
    align_wl : bool, default True
        Optimize linear spectrometer shift (m * lambda + b) to reconcile slight grating offsets.
    cut_scatter : bool, default False
        Exclude scatter band from wavelength alignment.
    scatter_region : tuple[float, float], default (500.0, 550.0)
        Wavelength band to ignore during alignment if ``cut_scatter`` is True.
    detector_fs, detector_ns : int, default 0
        Detector channel index.
    blend_method : {"cubic", "linear", "quintic", "cosine", "step"}, default "cubic"
        Weighting function used across the blend transition window.

    Returns
    -------
    StitchResult
        Result containing the stitched Dataset1D and scaling diagnostics.
    """
    from pymorgan.oneD.dataset import Dataset1D

    # 1. Delays & Unit Harmonization
    fs_t_raw = np.asarray(ds_fs.delays, dtype=float)
    u_t_fs = str(ds_fs.units.get("unitsT_ltx", "ps")).lower()
    # If units indicate ps (or max delay > 50 with ns expected), convert ps -> ns
    if "ps" in u_t_fs or (float(np.max(fs_t_raw)) > 100.0 and float(np.max(ds_ns.delays)) < 50000.0):
        fs_t = fs_t_raw / 1000.0
    else:
        fs_t = np.copy(fs_t_raw)

    ns_t = np.asarray(ds_ns.delays, dtype=float)

    # 2. Probe Axes & Wavelength Intersection
    fs_wl = np.asarray(ds_fs._detector_probe(detector_fs), dtype=float)
    ns_wl = np.asarray(ds_ns._detector_probe(detector_ns), dtype=float)
    fs_z_raw = np.asarray(ds_fs._detector_slice(detector_fs), dtype=float)
    ns_z_raw = np.asarray(ds_ns._detector_slice(detector_ns), dtype=float)

    # Sort if descending
    if np.any(np.diff(fs_wl) < 0):
        s1 = np.argsort(fs_wl)
        fs_wl, fs_z_raw = fs_wl[s1], fs_z_raw[:, s1]
    if np.any(np.diff(ns_wl) < 0):
        s2 = np.argsort(ns_wl)
        ns_wl, ns_z_raw = ns_wl[s2], ns_z_raw[:, s2]

    # Common wavelength intersection
    wl_min = max(float(np.min(fs_wl)), float(np.min(ns_wl)))
    wl_max = min(float(np.max(fs_wl)), float(np.max(ns_wl)))

    if wl_min >= wl_max:
        raise ValueError(
            f"No spectral overlap between FS [{np.min(fs_wl):.1f}, {np.max(fs_wl):.1f}] "
            f"and NS [{np.min(ns_wl):.1f}, {np.max(ns_wl):.1f}]."
        )

    mask_fs_wl = (fs_wl >= wl_min) & (fs_wl <= wl_max)
    master_wl = fs_wl[mask_fs_wl]
    fs_z = fs_z_raw[:, mask_fs_wl]

    # 3. Optional Linear Wavelength Alignment (m * x + b)
    wl_shift = None
    if align_wl:
        t_al_min, t_al_max = sorted(time_overlap)
        mask_t_fs = (fs_t >= t_al_min) & (fs_t <= t_al_max)
        mask_t_ns = (ns_t >= t_al_min) & (ns_t <= t_al_max)

        if np.any(mask_t_fs) and np.any(mask_t_ns):
            spec_fs = np.nanmean(fs_z[mask_t_fs, :], axis=0)
            spec_ns = np.nanmean(ns_z_raw[mask_t_ns, :], axis=0)
            s_rgn = scatter_region if cut_scatter else None
            m, b = _optimize_linear_wl_shift(master_wl, spec_fs, ns_wl, spec_ns, s_rgn)
            wl_shift = (m, b)
            ns_wl_shifted = m * ns_wl + b
        else:
            ns_wl_shifted = ns_wl
    else:
        ns_wl_shifted = ns_wl

    # Interpolate NS onto master wavelength
    ns_z_interp = np.zeros((len(ns_t), len(master_wl)), dtype=float)
    s_idx = np.argsort(ns_wl_shifted)
    ns_wl_sorted = ns_wl_shifted[s_idx]
    ns_z_sorted = ns_z_raw[:, s_idx]
    for r in range(len(ns_t)):
        ns_z_interp[r, :] = np.interp(master_wl, ns_wl_sorted, ns_z_sorted[r, :])

    # 4. Scaling Factor Optimization in time_overlap [t_scale_min, t_scale_max]
    t_scale_min, t_scale_max = sorted(time_overlap)
    mask_fs_scale = (fs_t >= t_scale_min) & (fs_t <= t_scale_max)
    mask_ns_scale = (ns_t >= t_scale_min) & (ns_t <= t_scale_max)

    if not np.any(mask_fs_scale) or not np.any(mask_ns_scale):
        raise ValueError(
            f"Delay overlap window [{t_scale_min:.2f}, {t_scale_max:.2f}] ns contains no data points."
        )

    t_eval = np.linspace(t_scale_min, t_scale_max, 40)
    surf_fs = np.zeros((len(t_eval), len(master_wl)), dtype=float)
    surf_ns = np.zeros((len(t_eval), len(master_wl)), dtype=float)

    for c in range(len(master_wl)):
        p_fs = np.polyfit(fs_t[mask_fs_scale], fs_z[mask_fs_scale, c], 2)
        surf_fs[:, c] = np.polyval(p_fs, t_eval)
        p_ns = np.polyfit(ns_t[mask_ns_scale], ns_z_interp[mask_ns_scale, c], 2)
        surf_ns[:, c] = np.polyval(p_ns, t_eval)

    if scale_override is not None:
        scaling_factor = float(scale_override)
        s_vector = np.full(len(master_wl), scaling_factor, dtype=float)
        used_model = "manual"
        fit_params: dict[str, Any] = {}
    elif scale_model == "scalar":
        a_vec = surf_fs.ravel()
        b_vec = surf_ns.ravel()
        denom = float(np.dot(a_vec, a_vec))
        scaling_factor = float(np.dot(a_vec, b_vec) / denom) if denom > 1e-15 else 1.0
        s_vector = np.full(len(master_wl), scaling_factor, dtype=float)
        used_model = "scalar"
        fit_params = {}
    else:  # quadratic S(lambda)
        poly_s = np.zeros(len(master_wl), dtype=float)
        for c in range(len(master_wl)):
            v_fs = surf_fs[:, c]
            v_ns = surf_ns[:, c]
            denom = float(np.dot(v_fs, v_fs))
            poly_s[c] = float(np.dot(v_fs, v_ns) / denom) if denom > 1e-15 else 1.0

        norm_wl = (master_wl - np.min(master_wl)) / (np.max(master_wl) - np.min(master_wl))
        p_fit = np.polyfit(norm_wl, poly_s, 2)
        s_vector = np.polyval(p_fit, norm_wl)
        s_vector = np.clip(s_vector, 0.01, 10.0)
        scaling_factor = s_vector
        used_model = "quadratic"
        fit_params = {"poly_coeffs": p_fit.tolist(), "individual_ratios": poly_s}

    # Scale FS data
    fs_z_scaled = fs_z * s_vector[np.newaxis, :]

    # 5. Join Data with Weighted Linear Ramp
    t_blend_start, t_blend_end = sorted(blend_window)
    idx_fs_need = fs_t <= t_blend_end
    idx_ns_need = ns_t >= t_blend_start

    joined_t = np.unique(np.concatenate([fs_t[idx_fs_need], ns_t[idx_ns_need]]))
    joined_t = np.sort(joined_t)

    fs_interp_t = np.zeros((len(joined_t), len(master_wl)), dtype=float)
    ns_interp_t = np.zeros((len(joined_t), len(master_wl)), dtype=float)

    for c in range(len(master_wl)):
        fs_interp_t[:, c] = np.interp(
            joined_t, fs_t, fs_z_scaled[:, c], left=fs_z_scaled[0, c], right=fs_z_scaled[-1, c]
        )
        ns_interp_t[:, c] = np.interp(
            joined_t, ns_t, ns_z_interp[:, c], left=ns_z_interp[0, c], right=ns_z_interp[-1, c]
        )

    weights = np.zeros(len(joined_t), dtype=float)
    weights[joined_t <= t_blend_start] = 1.0
    blend_mask = (joined_t > t_blend_start) & (joined_t < t_blend_end)
    denom_t = max(t_blend_end - t_blend_start, 1e-12)
    u_blend = (joined_t[blend_mask] - t_blend_start) / denom_t
    # u=0 is fs pure, u=1 is ns pure; weights holds fs weight, so 1.0 - w_ns
    weights[blend_mask] = 1.0 - _compute_blend_weight(u_blend, blend_method)
    weights[joined_t >= t_blend_end] = 0.0

    joined_z = weights[:, np.newaxis] * fs_interp_t + (1.0 - weights[:, np.newaxis]) * ns_interp_t

    # 6. Construct Resulting Dataset1D in ns
    units = dict(ds_fs.units or {})
    units["unitsT_lbl"] = "Delay"
    units["unitsT_ltx"] = "ns"
    src_name = f"Timescale-Stitched ({getattr(ds_fs, 'source', 'fs')} + {getattr(ds_ns, 'source', 'ns')})"

    res_ds = Dataset1D(
        Zavg_R=joined_z,
        delays=joined_t,
        probe=np.copy(master_wl),
        units=units,
        source=src_name,
    )
    res_ds.Zavg_C = np.copy(joined_z)

    return StitchResult(
        dataset=res_ds,
        kind="timescale",
        scaling_factor=scaling_factor,
        scaling_model=used_model,
        stitch_region=(float(t_scale_min), float(t_scale_max)),
        blend_window=(float(t_blend_start), float(t_blend_end)),
        blend_method=blend_method,
        wl_shift_params=wl_shift,
        scaling_params=fit_params,
    )

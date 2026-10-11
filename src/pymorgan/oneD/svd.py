"""Singular Value Decomposition (SVD) for 1D Transient Absorption Spectroscopy.

Provides SVD factorization, component extraction, scree analysis, data
recomposition with rank-k truncation, and residual calculation.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import scipy.linalg

if TYPE_CHECKING:
    import pandas as pd

    from pymorgan.oneD.dataset import Dataset1D


CenteringMode = Literal["none", "spectrum", "probe", "kinetic", "time", "both"]


@dataclass
class SVDResult:
    """Container for SVD results with rank-k recomposition and component access.

    Attributes
    ----------
    U : np.ndarray
        Left singular vectors (temporal components), shape ``(M, n_comp)``.
    S : np.ndarray
        Singular values, shape ``(n_comp,)`` in descending order.
    Vt : np.ndarray
        Right singular vectors transposed (spectral components), shape ``(n_comp, N)``.
    delays : np.ndarray
        1D delay axis array of length ``M``.
    probe : np.ndarray
        1D probe axis array of length ``N``.
    k : int
        Default / active rank truncation (number of components for recomposition).
    raw_matrix : np.ndarray
        The input data matrix (cropped and NaN-handled), shape ``(M, N)``.
    centering : str
        Centering applied before SVD ("none", "spectrum", "kinetic", "both").
    offset : np.ndarray
        Centering offset array subtracted before SVD, shape ``(M, N)`` or broadcastable.
    variance_explained : np.ndarray
        Fraction of total variance explained by each singular value:
        :math:`S_i^2 / \\sum S_j^2`.
    cumulative_variance : np.ndarray
        Cumulative sum of fraction of total variance explained.
    units : dict
        Units dictionary associated with the dataset (if available).
    """

    U: np.ndarray
    S: np.ndarray
    Vt: np.ndarray
    delays: np.ndarray
    probe: np.ndarray
    k: int
    raw_matrix: np.ndarray
    centering: str = "none"
    offset: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))
    variance_explained: np.ndarray = field(default_factory=lambda: np.array([]))
    cumulative_variance: np.ndarray = field(default_factory=lambda: np.array([]))
    units: dict = field(default_factory=dict)

    def __post_init__(self):
        self.k = int(np.clip(self.k, 1, len(self.S)))
        if self.variance_explained.size == 0 and self.S.size > 0:
            s2 = self.S**2
            s2_sum = np.sum(s2)
            self.variance_explained = s2 / s2_sum if s2_sum > 0 else np.zeros_like(s2)
            self.cumulative_variance = np.cumsum(self.variance_explained)

    @property
    def n_components(self) -> int:
        """Total number of available singular components."""
        return len(self.S)

    def recompose(self, k: int | None = None) -> np.ndarray:
        """Recompose the data matrix using the first ``k`` components.

        Parameters
        ----------
        k : int, optional
            Number of components to use. If None, uses ``self.k``.

        Returns
        -------
        np.ndarray
            Recomposed 2D matrix of shape ``(M, N)``, including the centering offset.
        """
        k_val = self.k if k is None else int(np.clip(k, 1, self.n_components))
        # (U[:, :k] * S[:k]) @ Vt[:k, :] + offset
        recon = (self.U[:, :k_val] * self.S[:k_val]) @ self.Vt[:k_val, :]
        if self.offset.size > 0:
            recon = recon + self.offset
        return recon

    def residuals(self, k: int | None = None) -> np.ndarray:
        """Compute residuals (raw_matrix - recompose(k)).

        Parameters
        ----------
        k : int, optional
            Number of components to use. If None, uses ``self.k``.

        Returns
        -------
        np.ndarray
            Residuals 2D matrix of shape ``(M, N)``.
        """
        return self.raw_matrix - self.recompose(k)

    def component_time(self, idx: int, scaled: bool = True) -> np.ndarray:
        """Return the temporal trace (left singular vector) for component ``idx``.

        Parameters
        ----------
        idx : int
            0-based component index.
        scaled : bool, default True
            If True, scales by the corresponding singular value :math:`S[idx]`.
        """
        if idx < 0 or idx >= self.n_components:
            raise IndexError(f"Component index {idx} out of range (0 to {self.n_components - 1})")
        u = self.U[:, idx]
        return u * self.S[idx] if scaled else u

    def component_probe(self, idx: int) -> np.ndarray:
        """Return the spectral trace (right singular vector) for component ``idx``.

        Parameters
        ----------
        idx : int
            0-based component index.
        """
        if idx < 0 or idx >= self.n_components:
            raise IndexError(f"Component index {idx} out of range (0 to {self.n_components - 1})")
        return self.Vt[idx, :]

    def variance_at(self, k: int) -> float:
        """Return cumulative variance explained by the first ``k`` components (0 to 1)."""
        k_val = int(np.clip(k, 1, self.n_components))
        return float(self.cumulative_variance[k_val - 1])

    def to_dataset(
        self,
        k: int | None = None,
        name: str | None = None,
        units: dict | None = None,
    ) -> Dataset1D:
        """Construct a new :class:`Dataset1D` containing the recomposed data.

        Parameters
        ----------
        k : int, optional
            Number of components to use. If None, uses ``self.k``.
        name : str, optional
            Source description name.
        units : dict, optional
            Units dictionary. Defaults to ``self.units``.

        Returns
        -------
        Dataset1D
            New dataset with recomposed signal in ``Zavg_R`` and ``Zavg_C``.
        """
        from pymorgan.oneD.dataset import Dataset1D

        k_val = self.k if k is None else int(np.clip(k, 1, self.n_components))
        recomposed = self.recompose(k_val)
        u = dict(self.units) if self.units else {}
        if units:
            u.update(units)

        src = name or f"SVD-recomposed (k={k_val})"
        ds = Dataset1D(
            Zavg_R=recomposed,
            delays=np.copy(self.delays),
            probe=np.copy(self.probe),
            units=u,
            source=src,
        )
        ds.Zavg_C = np.copy(recomposed)
        return ds

    def to_dataframe(self, k: int | None = None) -> dict[str, pd.DataFrame]:
        """Convert components, scree table, recomposed, and residuals to pandas DataFrames.

        Returns a dictionary with keys: ``'scree'``, ``'temporal'``, ``'spectral'``,
        ``'recomposed'``, and ``'residuals'``.
        """
        import pandas as pd

        k_val = self.k if k is None else int(np.clip(k, 1, self.n_components))

        # Scree DataFrame
        scree_df = pd.DataFrame(
            {
                "Component": np.arange(1, self.n_components + 1),
                "Singular_Value": self.S,
                "Variance_Explained_Pct": self.variance_explained * 100.0,
                "Cumulative_Variance_Pct": self.cumulative_variance * 100.0,
            }
        )

        # Temporal DataFrame
        u_dict: dict[str, Any] = {"Delay": self.delays}
        for i in range(k_val):
            u_dict[f"U_{i+1}_scaled"] = self.component_time(i, scaled=True)
            u_dict[f"U_{i+1}_raw"] = self.component_time(i, scaled=False)
        temporal_df = pd.DataFrame(u_dict)

        # Spectral DataFrame
        v_dict: dict[str, Any] = {"Probe": self.probe}
        for i in range(k_val):
            v_dict[f"V_{i+1}"] = self.component_probe(i)
        spectral_df = pd.DataFrame(v_dict)

        # Recomposed & Residuals DataFrames
        recomp_df = pd.DataFrame(self.recompose(k_val), index=self.delays, columns=self.probe)
        resid_df = pd.DataFrame(self.residuals(k_val), index=self.delays, columns=self.probe)

        return {
            "scree": scree_df,
            "temporal": temporal_df,
            "spectral": spectral_df,
            "recomposed": recomp_df,
            "residuals": resid_df,
        }

    def export_csv(self, filepath: str | Path, k: int | None = None) -> None:
        """Export SVD scree summary and first ``k`` components to CSV file(s).

        If ``filepath`` ends with ``.csv``, writes a composite CSV with sections.
        """
        path = Path(filepath)
        k_val = self.k if k is None else int(np.clip(k, 1, self.n_components))

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Scree Header
            writer.writerow(["# SVD Summary & Scree Plot"])
            writer.writerow(["# Centering:", self.centering])
            writer.writerow(["# Selected k:", k_val])
            writer.writerow(["Component", "SingularValue", "VarianceExplained_Pct", "CumulativeVariance_Pct"])
            for i in range(self.n_components):
                writer.writerow(
                    [
                        i + 1,
                        f"{self.S[i]:.6e}",
                        f"{self.variance_explained[i] * 100.0:.4f}",
                        f"{self.cumulative_variance[i] * 100.0:.4f}",
                    ]
                )
            writer.writerow([])

            # Temporal Components Header
            writer.writerow([f"# Temporal Components U(t) * S (k={k_val})"])
            u_hdr = ["Delay"] + [f"U{i+1}_scaled" for i in range(k_val)]
            writer.writerow(u_hdr)
            for r, d in enumerate(self.delays):
                row = [f"{d:.6e}"] + [f"{self.U[r, i] * self.S[i]:.6e}" for i in range(k_val)]
                writer.writerow(row)
            writer.writerow([])

            # Spectral Components Header
            writer.writerow([f"# Spectral Components V(lambda) (k={k_val})"])
            v_hdr = ["Probe"] + [f"V{i+1}" for i in range(k_val)]
            writer.writerow(v_hdr)
            for c, p in enumerate(self.probe):
                row = [f"{p:.6e}"] + [f"{self.Vt[i, c]:.6e}" for i in range(k_val)]
                writer.writerow(row)


def compute_svd(
    data: np.ndarray,
    delays: np.ndarray,
    probe: np.ndarray,
    k: int = 2,
    tmin: float | None = None,
    tmax: float | None = None,
    probe_min: float | None = None,
    probe_max: float | None = None,
    centering: CenteringMode = "none",
    handle_nan: Literal["interpolate", "drop", "raise"] = "interpolate",
    units: dict | None = None,
) -> SVDResult:
    """Compute Singular Value Decomposition on transient absorption data.

    Parameters
    ----------
    data : np.ndarray
        2D data matrix of shape ``(M, N)`` where ``M`` is the number of delays
        and ``N`` is the number of probe points.
    delays : np.ndarray
        1D delay array of length ``M``.
    probe : np.ndarray
        1D probe array of length ``N``.
    k : int, default 2
        Number of principal components to select for recomposition.
    tmin, tmax : float, optional
        Delay range limits for cropping prior to SVD.
    probe_min, probe_max : float, optional
        Probe range limits for cropping prior to SVD.
    centering : {"none", "spectrum", "probe", "kinetic", "time", "both"}, default "none"
        Centering method applied before factorization:
        - ``"none"``: Standard uncentered SVD (physical TA decomposition).
        - ``"spectrum"`` / ``"probe"``: Subtract average spectrum (mean across delays).
        - ``"kinetic"`` / ``"time"``: Subtract average kinetic trace (mean across probe).
        - ``"both"``: Double-centered SVD.
    handle_nan : {"interpolate", "drop", "raise"}, default "interpolate"
        How to handle NaN values in ``data``:
        - ``"interpolate"``: Drop fully-NaN rows/cols, then interpolate isolated NaNs.
        - ``"drop"``: Drop any row or column containing NaN.
        - ``"raise"``: Raise ValueError if NaNs are found.
    units : dict, optional
        Units dictionary associated with the dataset.

    Returns
    -------
    SVDResult
        Result container with components, scree analysis, and recomposition tools.
    """
    mat = np.asarray(data, dtype=float)
    t = np.asarray(delays, dtype=float)
    p = np.asarray(probe, dtype=float)

    if mat.ndim != 2:
        raise ValueError(f"Expected 2D data matrix for SVD, got shape {mat.shape}")
    if mat.shape[0] != len(t):
        raise ValueError(f"Row count ({mat.shape[0]}) does not match delay length ({len(t)})")
    if mat.shape[1] != len(p):
        raise ValueError(f"Column count ({mat.shape[1]}) does not match probe length ({len(p)})")

    # --- 1. Apply time and probe limits ---
    t_mask = np.ones(len(t), dtype=bool)
    if tmin is not None:
        t_mask &= t >= tmin
    if tmax is not None:
        t_mask &= t <= tmax

    p_mask = np.ones(len(p), dtype=bool)
    if probe_min is not None or probe_max is not None:
        p_lo = min(
            probe_min if probe_min is not None else -np.inf,
            probe_max if probe_max is not None else -np.inf,
        )
        p_hi = max(
            probe_min if probe_min is not None else np.inf,
            probe_max if probe_max is not None else np.inf,
        )
        p_mask &= (p >= p_lo) & (p <= p_hi)

    if not np.any(t_mask):
        raise ValueError("Time limits [tmin, tmax] excluded all delay points.")
    if not np.any(p_mask):
        raise ValueError("Probe limits [probe_min, probe_max] excluded all probe points.")

    mat = mat[np.ix_(t_mask, p_mask)]
    t = t[t_mask]
    p = p[p_mask]

    # --- 2. Handle NaNs ---
    if np.any(np.isnan(mat)):
        if handle_nan == "raise":
            raise ValueError("Data matrix contains NaN values. Please crop or clean before SVD.")
        elif handle_nan == "drop":
            valid_rows = ~np.isnan(mat).any(axis=1)
            valid_cols = ~np.isnan(mat).any(axis=0)
            if not np.any(valid_rows) or not np.any(valid_cols):
                raise ValueError("Dropping NaNs eliminated all data.")
            mat = mat[np.ix_(valid_rows, valid_cols)]
            t = t[valid_rows]
            p = p[valid_cols]
        elif handle_nan == "interpolate":
            # 1. Drop entirely NaN rows/cols
            valid_rows = ~np.isnan(mat).all(axis=1)
            valid_cols = ~np.isnan(mat).all(axis=0)
            if not np.any(valid_rows) or not np.any(valid_cols):
                raise ValueError("All data points in selected region are NaN.")
            mat = mat[np.ix_(valid_rows, valid_cols)]
            t = t[valid_rows]
            p = p[valid_cols]

            # 2. Linear interpolation across remaining isolated NaNs
            if np.any(np.isnan(mat)):
                for r in range(mat.shape[0]):
                    row = mat[r, :]
                    nans = np.isnan(row)
                    if np.any(nans) and not np.all(nans):
                        x = np.arange(len(row))
                        row[nans] = np.interp(x[nans], x[~nans], row[~nans])
                # If any NaNs remain (e.g. edge columns with all NaNs), do column-wise
                if np.any(np.isnan(mat)):
                    for c in range(mat.shape[1]):
                        col = mat[:, c]
                        nans = np.isnan(col)
                        if np.any(nans) and not np.all(nans):
                            y = np.arange(len(col))
                            col[nans] = np.interp(y[nans], y[~nans], col[~nans])
                # Final fallback for any boundary corners
                mat = np.nan_to_num(mat, nan=0.0)

    # --- 3. Centering ---
    cent_lower = centering.lower()
    m_rows, n_cols = mat.shape
    offset = np.zeros((m_rows, n_cols), dtype=float)

    if cent_lower in ("spectrum", "probe"):
        mean_spec = np.mean(mat, axis=0, keepdims=True)
        offset = np.repeat(mean_spec, m_rows, axis=0)
    elif cent_lower in ("kinetic", "time"):
        mean_kin = np.mean(mat, axis=1, keepdims=True)
        offset = np.repeat(mean_kin, n_cols, axis=1)
    elif cent_lower == "both":
        mean_spec = np.mean(mat, axis=0, keepdims=True)
        mean_kin = np.mean(mat, axis=1, keepdims=True)
        mean_grand = np.mean(mat)
        offset = mean_spec + mean_kin - mean_grand
    elif cent_lower == "none":
        offset = np.zeros((m_rows, n_cols), dtype=float)
    else:
        raise ValueError(
            f"Unknown centering '{centering}'. Choose from 'none', 'spectrum', 'kinetic', 'both'."
        )

    mat_centered = mat - offset

    # --- 4. SVD Factorization ---
    U, S, Vt = scipy.linalg.svd(mat_centered, full_matrices=False)

    return SVDResult(
        U=U,
        S=S,
        Vt=Vt,
        delays=t,
        probe=p,
        k=k,
        raw_matrix=mat,
        centering=centering,
        offset=offset,
        units=units or {},
    )

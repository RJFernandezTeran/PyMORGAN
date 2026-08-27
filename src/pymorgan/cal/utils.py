"""Utility functions for splitting and merging multi-detector calibration arrays."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np


def split_calibration(
    merged_cm: np.ndarray, split_pixel_idx: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Split a multi-detector merged calibration vector into LHS (Detector 1) and RHS (Detector 2).

    Parameters
    ----------
    merged_cm : np.ndarray
        Continuous 1D array of calibrated wavenumbers/wavelengths for combined detectors.
    split_pixel_idx : int, optional
        Pixel index split boundary. If None, splits at mid-point len(merged_cm) // 2.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (lhs_cal, rhs_cal) split calibration arrays.
    """
    arr = np.asarray(merged_cm, dtype=float).ravel()
    n = len(arr)
    if n % 2 != 0 and split_pixel_idx is None:
        raise ValueError(f"Merged calibration vector has odd length {n}. Specify explicit split_pixel_idx.")

    split_at = split_pixel_idx if split_pixel_idx is not None else n // 2
    lhs = arr[:split_at]
    rhs = arr[split_at:]

    return lhs, rhs


def merge_calibration(det1_cm: np.ndarray, det2_cm: np.ndarray) -> np.ndarray:
    """Merge Detector 1 (LHS) and Detector 2 (RHS) calibration vectors into a single probe array.

    Parameters
    ----------
    det1_cm : np.ndarray
        Detector 1 (LHS) calibrated wavenumber/wavelength array.
    det2_cm : np.ndarray
        Detector 2 (RHS) calibrated wavenumber/wavelength array.

    Returns
    -------
    np.ndarray
        Combined merged calibration vector.
    """
    arr1 = np.asarray(det1_cm, dtype=float).ravel()
    arr2 = np.asarray(det2_cm, dtype=float).ravel()

    if len(arr1) != len(arr2):
        raise ValueError(f"Detector lengths do not match ({len(arr1)} vs {len(arr2)}). Cannot merge.")

    n_pix = len(arr1)
    merged = np.concatenate([arr1[:n_pix], arr2[:n_pix]])
    return merged


def save_calibration_file(
    probe_axis: np.ndarray,
    output_dir: str | Path,
    filename: str = "CalibratedProbe.csv",
    save_timestamped: bool = False,
    cal_type_code: int | None = None,
    save_mat: bool | None = None,
) -> tuple[Path, Path | None]:
    """Save calibrated probe vector to CSV file (CalibratedProbe.csv).

    When save_mat is True or cal_type_code is in (2, 3, 5) (UniGE fsTA / nsTA / NIR-TA),
    also writes pix2lam.mat.

    Parameters
    ----------
    probe_axis : np.ndarray
        Calibrated wavenumber or wavelength vector (in native detector units).
    output_dir : str or Path
        Target export directory.
    filename : str, default "CalibratedProbe.csv"
        Primary file name.
    save_timestamped : bool, default False
        If True, also saves a timestamped copy (e.g. CalibProbe_20260722-2119.csv).
    cal_type_code : int, optional
        Calibration setup type code (1..11).
    save_mat : bool, optional
        Explicit override to save pix2lam.mat. If None, auto-activates for UniGE fsTA/nsTA (codes 2, 3, 5).

    Returns
    -------
    tuple[Path, Path | None]
        (primary_path, timestamped_path) output file paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    primary_file = out_path / filename
    np.savetxt(primary_file, probe_axis, delimiter=",", fmt="%.6f")

    # Save pix2lam.mat only for UniGE fsTA / nsTA datasets (codes 2, 3, 5) or if save_mat=True
    should_save_mat = save_mat if save_mat is not None else (cal_type_code in (2, 3, 5))
    if should_save_mat:
        import scipy.io as sio

        mat_file = out_path / "pix2lam.mat"
        try:
            sio.savemat(mat_file, {"lam": np.asarray(probe_axis, dtype=float).ravel()})
        except Exception:
            pass

    timestamp_file = None
    if save_timestamped:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        timestamp_file = out_path / f"CalibProbe_{stamp}.csv"

        # Format 2-column format [pixel_index, calibrated_value]
        n = len(probe_axis)
        pixels = np.arange(n, dtype=float)
        stacked = np.column_stack([pixels, probe_axis])
        np.savetxt(timestamp_file, stacked, delimiter=",", fmt=["%d", "%.6f"])

    return primary_file, timestamp_file

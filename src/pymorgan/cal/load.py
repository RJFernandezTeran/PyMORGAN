"""File loaders for reference spectra libraries and experimental calibration measurement files."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

from ..log import get_logger

logger = get_logger(__name__)


class ExperimentalData(NamedTuple):
    """Container for loaded experimental calibration raw/absorbance spectra."""

    detector_data: list[np.ndarray]
    gratings: list[float]
    cwl: list[float]
    file_path: Path
    min_wl: float | None = None
    max_wl: float | None = None
    wavenumbers: np.ndarray | None = None
    wavelengths: np.ndarray | None = None


class ReferenceSpectrum(NamedTuple):
    """Container for a standard calibration reference spectrum."""

    name: str
    spectral_axis: np.ndarray  # nm or cm-1
    absorbance: np.ndarray
    axis_unit: str  # "nm" or "cm-1"


def load_reference_spectrum(file_path: str | Path) -> ReferenceSpectrum:
    """Load reference spectrum from CSV/TSV/text file with automatic delimiter and unit detection.

    Parameters
    ----------
    file_path : str or Path
        Path to reference spectrum CSV/text file.

    Returns
    -------
    ReferenceSpectrum
        Reference spectrum object containing name, spectral axis, absorbance, and unit ("nm" or "cm-1").
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Reference spectrum file not found: {path}")

    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    lines = raw_text.splitlines()

    # Detect unit hints in comments or header text
    header_hint_cm = False
    header_hint_nm = False
    for line in lines[:10]:
        l_lower = line.lower()
        if any(term in l_lower for term in ["cm-1", "cm1", "cm^-1", "wavenumber"]):
            header_hint_cm = True
        if any(term in l_lower for term in ["nm", "nanometer", "wavelength"]):
            header_hint_nm = True

    rows = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("%") or s.startswith("//"):
            continue

        if ";" in s:
            parts = [p.strip().replace(",", ".") for p in s.split(";")]
        elif "\t" in s:
            parts = [p.strip() for p in s.split("\t")]
        elif "," in s:
            parts = [p.strip() for p in s.split(",")]
        else:
            parts = s.split()

        try:
            vals = [float(p) for p in parts if p != ""]
            if len(vals) >= 2:
                rows.append(vals[:2])
        except ValueError:
            continue

    if not rows:
        mat = read_numeric_matrix(path)
        if mat.ndim > 1 and mat.shape[1] >= 2:
            x_axis = mat[:, 0]
            y_vals = mat[:, 1]
        else:
            raise ValueError(f"Could not parse 2-column reference spectrum from file: {path}")
    else:
        mat = np.array(rows, dtype=float)
        x_axis = mat[:, 0]
        y_vals = mat[:, 1]

    # Ensure ascending sort along spectral axis
    if len(x_axis) > 1 and x_axis[0] > x_axis[-1]:
        sort_idx = np.argsort(x_axis)
        x_axis = x_axis[sort_idx]
        y_vals = y_vals[sort_idx]

    # Determine units
    if header_hint_cm:
        axis_unit = "cm-1"
    elif header_hint_nm:
        axis_unit = "nm"
    elif np.max(x_axis) > 1500.0 or np.mean(x_axis) > 1000.0:
        axis_unit = "cm-1"
    else:
        axis_unit = "nm"

    return ReferenceSpectrum(
        name=path.stem,
        spectral_axis=x_axis,
        absorbance=y_vals,
        axis_unit=axis_unit,
    )


def read_numeric_matrix(file_path: str | Path, default_delimiter: str | None = None) -> np.ndarray:
    """Read numeric array data from text file, automatically skipping header rows (matching MATLAB readmatrix)."""
    path = Path(file_path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    numeric_lines = []

    for line in lines:
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("%") or s.startswith("//"):
            continue
        # Test if first token is numeric
        parts = s.replace(",", " ").split()
        try:
            float(parts[0])
            numeric_lines.append(s)
        except (ValueError, IndexError):
            continue

    if not numeric_lines:
        raise ValueError(f"No numeric data found in file: {path}")

    sample = numeric_lines[0]
    if default_delimiter is None:
        if "\t" in sample:
            delimiter = "\t"
        elif "," in sample:
            delimiter = ","
        else:
            delimiter = None
    else:
        delimiter = default_delimiter

    import io
    buf = io.StringIO("\n".join(numeric_lines))
    return np.loadtxt(buf, delimiter=delimiter)


def parse_meta_file(meta_path: Path, is_ir: bool) -> dict:
    """Parse MESS or generic setup metadata text file (_meta.txt) for grating position, CWL, and axis arrays."""
    import re

    cwl_val = 0.0
    grating_val = 0.0
    wn_axis = None
    wl_axis = None

    if not meta_path.exists():
        return {"cwl": 0.0, "grating": 0.0, "min_wl": None, "max_wl": None, "wn_axis": None, "wl_axis": None}

    lines = meta_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines:
        l_lower = line.lower()
        if "centre wavelength" in l_lower or "cwl" in l_lower:
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", line[line.find(":") if ":" in line else 0:])
            if nums:
                cwl_val = float(nums[0])
        elif "grating" in l_lower:
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", line[line.find(":") if ":" in line else 0:])
            if nums:
                grating_val = float(nums[0])
        elif "wavenumber axis:" in l_lower:
            nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", line[line.find(":"):])]
            if nums:
                wn_axis = np.array(nums, dtype=float)
        elif "wavelength axis:" in l_lower:
            nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", line[line.find(":"):])]
            if nums:
                wl_axis = np.array(nums, dtype=float)

    min_wl = None
    max_wl = None

    if is_ir:
        if wn_axis is not None and len(wn_axis) > 0:
            min_wl = float(np.min(wn_axis)) - 50.0
            max_wl = float(np.max(wn_axis)) + 50.0
            if cwl_val == 0.0:
                cwl_val = float(wn_axis[len(wn_axis) // 2])
            elif cwl_val > 3000.0:  # given in nm (e.g. 4750.10 nm -> 2105.22 cm-1)
                cwl_val = 1e7 / cwl_val
        elif cwl_val > 3000.0:
            cwl_val = 1e7 / cwl_val
    else:
        if wl_axis is not None and len(wl_axis) > 0:
            min_wl = float(np.min(wl_axis)) - 20.0
            max_wl = float(np.max(wl_axis)) + 20.0
            if cwl_val == 0.0:
                cwl_val = float(wl_axis[len(wl_axis) // 2])

    return {
        "cwl": cwl_val,
        "grating": grating_val,
        "min_wl": min_wl,
        "max_wl": max_wl,
        "wn_axis": wn_axis,
        "wl_axis": wl_axis,
    }


def load_experimental_spectrum(
    file_path: str | Path,
    cal_type_code: int,
    blank_file_path: str | Path | None = None,
) -> ExperimentalData:
    """Parse raw measurement data file based on setup type (1..10).

    Parameters
    ----------
    file_path : str or Path
        Path to experimental spectrum file.
    cal_type_code : int
        Calibration setup type code (1..10).
    blank_file_path : str or Path, optional
        Optional path to blank measurement file for UniGE TA setups.

    Returns
    -------
    ExperimentalData
        Loaded detector data arrays, grating settings, and central wavelengths.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Experimental data file not found: {path}")

    gratings = [0.0, 0.0]
    cwl = [0.0, 0.0]
    min_wl_val = None
    max_wl_val = None
    wn_axis = None
    wl_axis = None
    is_ir = cal_type_code in (1, 4, 6, 7, 8, 9)

    if cal_type_code == 1:
        # UoS TRIR (4 detectors: probe1=96, probe2=96, ref1=32, ref2=32)
        det_sz = [96, 96, 32, 32]
        raw = read_numeric_matrix(path, default_delimiter="\t")
        data_list = []
        start_idx = 0
        for sz in det_sz:
            data_list.append(raw[start_idx : start_idx + sz])
            start_idx += sz

        # Check for metadata (.LG file)
        lg_path = path.with_suffix(".LG")
        if lg_path.exists():
            lines = lg_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if len(lines) >= 77:
                try:
                    g1_st = lines[74].split()
                    w1_st = lines[76].split()
                    g2_st = lines[66].split()
                    w2_st = lines[68].split()
                    gratings = [float(g1_st[-1]), float(g2_st[-1])]
                    cwl = [float(w1_st[-1]), float(w2_st[-1])]
                except (ValueError, IndexError):
                    logger.debug(
                        "Could not parse the grating/CWL metadata from %s.", lg_path, exc_info=True
                    )

    elif cal_type_code in (2, 5):
        # UniGE fsTA / NIR-TA (3rd column of tab-delimited dat)
        raw = read_numeric_matrix(path, default_delimiter="\t")
        col_idx = 2 if raw.ndim > 1 and raw.shape[1] >= 3 else 0
        data_list = [raw[:, col_idx]]

    elif cal_type_code == 3:
        # UniGE nsTA (2nd column of tab-delimited dat)
        raw = read_numeric_matrix(path, default_delimiter="\t")
        col_idx = 1 if raw.ndim > 1 and raw.shape[1] >= 2 else 0
        data_list = [raw[:, col_idx]]

    elif cal_type_code == 4:
        # UZH Lab 2 (1st column of csv)
        raw = read_numeric_matrix(path)
        data_list = [raw[:, 0] if raw.ndim > 1 else raw]

        # Parse _meta.txt if available
        meta_path = path.parent / f"{path.parent.name}_meta.txt"
        if not meta_path.exists():
            meta_path = path.parent / f"{path.stem}_meta.txt"

        if meta_path.exists():
            parsed = parse_meta_file(meta_path, is_ir=True)
            gratings[0] = parsed["grating"]
            cwl[0] = parsed["cwl"]
            min_wl_val = parsed["min_wl"]
            max_wl_val = parsed["max_wl"]
            wn_axis = parsed["wn_axis"]

    elif cal_type_code in (6, 7):
        # RAL LIFEtime (2 detectors of 128 pixels each)
        det_sz = [128, 128]
        raw = read_numeric_matrix(path, default_delimiter=",")
        col_idx = 1 if raw.ndim > 1 and raw.shape[1] >= 2 else 0
        y_vec = raw[:, col_idx] if raw.ndim > 1 else raw

        data_list = [
            y_vec[0 : det_sz[0]],
            y_vec[det_sz[0] : det_sz[0] + det_sz[1]],
        ]

    elif cal_type_code in (8, 9, 10):
        # MESS TRIR + TRUVIS (1st column)
        raw = read_numeric_matrix(path)
        data_list = [raw[:, 0] if raw.ndim > 1 else raw]

        # Parse MESS _meta.txt search hierarchy
        meta_candidates = [
            path.parent.parent / path.parent.name / f"{path.parent.name}_meta.txt",
            path.parent / f"{path.parent.name}_meta.txt",
            path.parent / f"{path.stem}_meta.txt",
        ] + list(path.parent.glob("*_meta.txt"))
        meta_path = None
        for candidate in meta_candidates:
            if candidate.exists():
                meta_path = candidate
                break

        if meta_path is not None:
            parsed = parse_meta_file(meta_path, is_ir=is_ir)
            # TRUVIS-II (type 10) does not use grating information from metafile
            gratings[0] = 0.0 if cal_type_code == 10 else parsed["grating"]
            cwl[0] = parsed["cwl"]
            min_wl_val = parsed["min_wl"]
            max_wl_val = parsed["max_wl"]
            wn_axis = parsed["wn_axis"]
            wl_axis = parsed["wl_axis"]
    else:
        # Default single column text reader
        raw = read_numeric_matrix(path)
        data_list = [raw[:, 0] if raw.ndim > 1 else raw]

    return ExperimentalData(
        detector_data=data_list,
        gratings=gratings,
        cwl=cwl,
        file_path=path,
        min_wl=min_wl_val,
        max_wl=max_wl_val,
        wavenumbers=wn_axis,
        wavelengths=wl_axis,
    )

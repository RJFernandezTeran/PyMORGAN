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
    spectral_range: str | None = None


CALIBRATION_TYPES: dict[int, str] = {
    1: "UoS TRIR",
    2: "UniGE fsTA",
    3: "UniGE nsTA",
    4: "UZH Lab 2",
    5: "UniGE NIR-TA",
    6: "RAL LIFEtime (Absorbance)",
    7: "RAL LIFEtime (Intensity)",
    8: "UniGE TRIR (Intensity)",
    9: "UniGE TRIR (Absorbance)",
    10: "UniGE TRUVIS-II (Intensity)",
    11: "Light Conversion HARPIA-TA (Int.)",
    12: "PyMESS",
}


def available_calibration_types(
    name: str | None = None,
    as_dict: bool = False,
) -> dict[int, str] | list[str] | int:
    """Return available calibration setup types or look up the integer code for a given setup name.

    Parameters
    ----------
    name : str, optional
        Setup display name or search query (e.g. "HARPIA", "fsTA", "UoS TRIR") to look up the magic integer code for.
    as_dict : bool, default False
        If True, return dictionary mapping integer setup code -> display name.

    Returns
    -------
    int, dict[int, str], or list[str]
        - If ``name`` is provided: returns integer calibration code (1..11).
        - If ``as_dict=True``: returns dict mapping ``{code: display_name}``.
        - Otherwise: returns list of display names.
    """
    if name is not None:
        n_lower = name.lower().strip()
        for code, disp_name in CALIBRATION_TYPES.items():
            if n_lower in disp_name.lower() or disp_name.lower() in n_lower:
                return code
        raise KeyError(
            f"Unknown calibration type {name!r}. Available setup types: {list(CALIBRATION_TYPES.values())}"
        )
    if as_dict:
        return dict(CALIBRATION_TYPES)
    return list(CALIBRATION_TYPES.values())


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
    """Read numeric array data from text file, automatically skipping header and metadata rows."""
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


def normalize_spectral_range(spectral_range: str | None) -> str | None:
    """Normalize spectral range string to canonical form ('mid-ir', 'uv-vis', 'nir', 'uv', 'vis')."""
    if spectral_range is None:
        return None
    s = spectral_range.lower().strip()
    if not s:
        return None
    if any(k in s for k in ("mid-ir", "mid_ir", "midir", "trir", "ir-ta")):
        return "mid-ir"
    if "nir" in s:
        return "nir"
    if any(k in s for k in ("uv/vis", "uv-vis", "uv_vis", "uvvis", "truvis")):
        return "uv-vis"
    if s in ("uv", "uv range"):
        return "uv"
    if s in ("vis", "vis range", "visible"):
        return "vis"
    if "ir" in s:
        return "mid-ir"
    return s


def parse_pymess_calibration_h5(h5_path: Path, spectral_range: str | None = None) -> dict:
    """Parse a PyMESS HDF5 (.h5) file for calibration spectrum, grating, CWL, and spectral range.

    Parameters
    ----------
    h5_path : Path
        Path to PyMESS .h5 measurement file.
    spectral_range : str, optional
        Explicit spectral range override ('mid-ir', 'uv-vis', 'nir'). If None, auto-detected from HDF5 metadata and axis.

    Returns
    -------
    dict
        Parsed dictionary containing ``detector_data``, ``grating``, ``cwl``, ``min_wl``,
        ``max_wl``, ``wn_axis``, ``wl_axis``, and ``spectral_range``.
    """
    import re

    import h5py

    def _attr_str(val) -> str:
        if val is None:
            return ""
        if isinstance(val, (bytes, np.bytes_)):
            return val.decode("utf-8", errors="ignore").strip()
        if isinstance(val, np.ndarray) and val.ndim == 0:
            return _attr_str(val.item())
        return str(val).strip()

    with h5py.File(h5_path, "r") as f:
        # 1. Extract 1D detector intensity / spectrum array
        data_grp = f["data"] if "data" in f and isinstance(f["data"], h5py.Group) else f
        spec_arr = None
        if "probe_spectrum" in data_grp:
            spec_arr = np.asarray(data_grp["probe_spectrum"][()], dtype=float).ravel()
        elif "intensity" in data_grp:
            arr = np.asarray(data_grp["intensity"][()], dtype=float)
            spec_arr = np.mean(arr, axis=0).ravel() if arr.ndim > 1 else arr.ravel()
        elif "probe_shots" in data_grp:
            shots = np.asarray(data_grp["probe_shots"][()], dtype=float)
            spec_arr = np.mean(shots, axis=0).ravel() if shots.ndim > 1 else shots.ravel()
            if "calibration" in f and "probe_dark_background" in f["calibration"]:
                dark = np.asarray(f["calibration/probe_dark_background"][()], dtype=float).ravel()
                if len(dark) == len(spec_arr):
                    spec_arr = spec_arr - dark
        elif "delta_A" in data_grp:
            arr = np.asarray(data_grp["delta_A"][()], dtype=float)
            spec_arr = np.mean(arr, axis=0).ravel() if arr.ndim > 1 else arr.ravel()
        elif "signal_mOD" in data_grp:
            arr = np.asarray(data_grp["signal_mOD"][()], dtype=float)
            spec_arr = np.mean(arr, axis=0).ravel() if arr.ndim > 1 else arr.ravel()

        if spec_arr is None or len(spec_arr) == 0:
            raise ValueError(f"No valid spectrum dataset found in PyMESS HDF5 file: {h5_path}")

        # 2. Extract wavelength / wavenumber axis if present
        raw_axis = None
        if "wavelengths" in data_grp:
            raw_axis = np.asarray(data_grp["wavelengths"][()], dtype=float).ravel()

        # 3. Extract metadata attributes
        meta_attrs = f["metadata"].attrs if "metadata" in f else f.attrs
        setup_id = _attr_str(meta_attrs.get("setup_id", ""))
        setup_name = _attr_str(meta_attrs.get("setup_name", ""))
        fast_mod = _attr_str(meta_attrs.get("fast_modulation", ""))
        cwl_nm_str = _attr_str(meta_attrs.get("spectrometer_cwl_nm", ""))
        cwn_cm1_str = _attr_str(meta_attrs.get("spectrometer_cwn_cm1", ""))
        grating_str = _attr_str(meta_attrs.get("spectrometer_grating", ""))
        summary_str = _attr_str(meta_attrs.get("spectrometer_summary", ""))

    # Parse numeric CWL, CWN, and grating from explicit metadata attributes
    cwl_nm_meta = 0.0
    if cwl_nm_str:
        try:
            cwl_nm_meta = float(cwl_nm_str)
        except ValueError:
            pass

    cwn_cm1_meta = 0.0
    if cwn_cm1_str:
        try:
            cwn_cm1_meta = float(cwn_cm1_str)
        except ValueError:
            pass

    grating_val = 0.0
    if grating_str:
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", grating_str)
        if nums:
            grating_val = float(nums[0])
    elif summary_str:
        m_grat = re.search(r"(\d+(?:\.\d+)?)\s*l/mm", summary_str, re.IGNORECASE)
        if m_grat:
            grating_val = float(m_grat.group(1))

    # 4. Determine spectral range ('mid-ir', 'uv-vis', 'nir')
    n_pix = len(spec_arr)
    axis_is_physical = (
        raw_axis is not None
        and len(raw_axis) > 1
        and float(np.max(raw_axis)) > float(len(raw_axis)) + 10.0
    )

    detected_range: str | None = None
    # Priority 1: Explicit spectrometer_cwl_nm attribute (written when spectrometer is active)
    if cwl_nm_meta > 2500.0:
        detected_range = "mid-ir"
    elif 850.0 <= cwl_nm_meta <= 2500.0:
        detected_range = "nir"
    elif 200.0 <= cwl_nm_meta < 850.0:
        detected_range = "uv-vis"

    # Priority 2: Physical values in /data/wavelengths dataset
    if detected_range is None and axis_is_physical and raw_axis is not None:
        mean_ax = float(np.mean(raw_axis))
        if 200.0 <= mean_ax <= 850.0:
            detected_range = "uv-vis"
        elif mean_ax > 1650.0 or (n_pix <= 128 and mean_ax > 1000.0):
            detected_range = "mid-ir"
        elif 850.0 < mean_ax <= 1650.0:
            detected_range = "nir"

    # Priority 3: Setup ID / Name / Fast modulation strings
    if detected_range is None:
        meta_combined = f"{setup_id} {setup_name} {fast_mod}".lower()
        if any(k in meta_combined for k in ("trir", "2dir", "ir_pp", "ir-pp", "mid_ir", "mid-ir")):
            detected_range = "mid-ir"
        elif "nir" in meta_combined:
            detected_range = "nir"
        elif any(k in meta_combined for k in ("truvis", "uvvis", "uv-vis", "fsta")):
            detected_range = "uv-vis"

    norm_override = normalize_spectral_range(spectral_range)
    active_range = norm_override if norm_override in ("mid-ir", "uv-vis", "nir") else (detected_range or "mid-ir")

    # 5. Populate pathway-specific CWL, grating, and axis bounds
    cwl_val = 0.0
    min_wl_val = None
    max_wl_val = None
    wn_axis = None
    wl_axis = None

    if active_range == "mid-ir":
        # Follow UniGE TRIR pathway (wavenumbers cm-1)
        if axis_is_physical and raw_axis is not None:
            if float(np.mean(raw_axis)) > 3000.0:
                wn_axis = 1e7 / np.clip(raw_axis, 1.0, None)
            else:
                wn_axis = raw_axis.copy()
            min_wl_val = float(np.min(wn_axis)) - 50.0
            max_wl_val = float(np.max(wn_axis)) + 50.0

        if cwn_cm1_meta > 0:
            cwl_val = cwn_cm1_meta
        elif cwl_nm_meta > 2500.0:
            cwl_val = 1e7 / cwl_nm_meta
        elif wn_axis is not None and len(wn_axis) > 0:
            cwl_val = float(wn_axis[len(wn_axis) // 2])
    else:
        # Follow UniGE TRUVIS-II ('uv-vis') or UniGE NIR-TA ('nir') pathway (wavelengths nm)
        if axis_is_physical and raw_axis is not None:
            wl_axis = raw_axis.copy()
            min_wl_val = float(np.min(wl_axis)) - 20.0
            max_wl_val = float(np.max(wl_axis)) + 20.0

        # TRUVIS-II and NIR-TA pathways do not use grating metadata to overwrite ppnm_guess
        grating_val = 0.0

        if active_range == "uv-vis":
            if 200.0 <= cwl_nm_meta <= 900.0:
                cwl_val = cwl_nm_meta
            elif wl_axis is not None and len(wl_axis) > 0 and 200.0 <= float(np.mean(wl_axis)) <= 900.0:
                cwl_val = float(wl_axis[len(wl_axis) // 2])
        elif active_range == "nir":
            if 800.0 <= cwl_nm_meta <= 2500.0:
                cwl_val = cwl_nm_meta
            elif wl_axis is not None and len(wl_axis) > 0 and 800.0 <= float(np.mean(wl_axis)) <= 2500.0:
                cwl_val = float(wl_axis[len(wl_axis) // 2])

    return {
        "detector_data": [spec_arr],
        "grating": grating_val,
        "cwl": cwl_val,
        "min_wl": min_wl_val,
        "max_wl": max_wl_val,
        "wn_axis": wn_axis,
        "wl_axis": wl_axis,
        "spectral_range": active_range,
    }


def load_experimental_spectrum(
    file_path: str | Path,
    cal_type_code: int,
    blank_file_path: str | Path | None = None,
    spectral_range: str | None = None,
) -> ExperimentalData:
    """Parse raw measurement data file based on setup type (1..12).

    Parameters
    ----------
    file_path : str or Path
        Path to experimental spectrum file.
    cal_type_code : int
        Calibration setup type code (1..12).
    blank_file_path : str or Path, optional
        Optional path to blank measurement file for UniGE TA setups.
    spectral_range : str, optional
        Optional spectral range override for PyMESS ('mid-ir', 'uv-vis', 'nir') or HARPIA ('uv', 'vis').

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
    detected_range = normalize_spectral_range(spectral_range)
    is_ir = cal_type_code in (1, 4, 6, 7, 8, 9)

    if cal_type_code == 12 or path.suffix.lower() in (".h5", ".hdf5"):
        # PyMESS HDF5 spectrum file (.h5)
        implicit_range = spectral_range
        if implicit_range is None and cal_type_code != 12:
            if is_ir:
                implicit_range = "mid-ir"
            elif cal_type_code == 5:
                implicit_range = "nir"
            elif cal_type_code in (2, 3, 10, 11):
                implicit_range = "uv-vis"
        parsed_h5 = parse_pymess_calibration_h5(path, spectral_range=implicit_range)
        data_list = parsed_h5["detector_data"]
        gratings[0] = parsed_h5["grating"]
        cwl[0] = parsed_h5["cwl"]
        min_wl_val = parsed_h5["min_wl"]
        max_wl_val = parsed_h5["max_wl"]
        wn_axis = parsed_h5["wn_axis"]
        wl_axis = parsed_h5["wl_axis"]
        detected_range = parsed_h5["spectral_range"]

    elif cal_type_code == 1:
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
    elif cal_type_code == 11:
        # Light Conversion HARPIA-TA (Int.) (Col 0 = wavelength nm, Col 1 = detector signal V)
        raw = read_numeric_matrix(path)
        if raw.ndim > 1 and raw.shape[1] >= 2:
            wl_axis = raw[:, 0]
            data_list = [raw[:, 1]]
            min_wl_val = float(np.min(wl_axis))
            max_wl_val = float(np.max(wl_axis))
            cwl[0] = float((min_wl_val + max_wl_val) / 2.0)
        else:
            data_list = [raw[:, 0] if raw.ndim > 1 else raw]
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
        spectral_range=detected_range,
    )


def load_HARPIA_calibration_spectrum(
    file_path: str | Path,
    spectral_range: str | None = None,
) -> ExperimentalData:
    """Load Light Conversion HARPIA-TA spectral calibration measurement text file."""
    return load_experimental_spectrum(file_path, cal_type_code=11, spectral_range=spectral_range)


def load_PyMESS_calibration_spectrum(
    file_path: str | Path,
    spectral_range: str | None = None,
) -> ExperimentalData:
    """Load PyMESS HDF5 (.h5) spectral calibration measurement file."""
    return load_experimental_spectrum(file_path, cal_type_code=12, spectral_range=spectral_range)



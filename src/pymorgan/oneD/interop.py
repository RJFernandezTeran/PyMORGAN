"""Interoperability readers and writers for Glotaran / pyglotaran formats."""

from __future__ import annotations

import re
from os import PathLike
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from pymorgan import helpers as hlp

from ..hdf5_meta import create_compressed_dataset
from ..log import get_logger
from .load import (
    LoaderResult,
    register_dataset_file_filter,
    register_dataset_glob,
    register_loader,
)

logger = get_logger(__name__)


def _attr_str(val: Any) -> str:
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8", errors="replace")
    return str(val)


def _normalize_probe_unit(raw: str | None, probe: np.ndarray) -> str:
    low = (raw or "").strip().lower()
    if any(tok in low for tok in ("cm-1", "cm^-1", "cm^{-1}", "wavenumber")):
        return "cm^{-1}"
    if "ev" in low or "energy" in low:
        return "eV"
    if "nm" in low or "wavelength" in low:
        return "nm"
    finite = probe[np.isfinite(probe)]
    if finite.size and float(np.min(finite)) > 1200.0:
        return "cm^{-1}"
    return "nm"


def _normalize_time_unit(raw: str | None) -> str:
    low = (raw or "").strip().lower()
    for unit in ("fs", "ps", "ns", "us"):
        if re.search(rf"\b{unit}\b", low):
            return unit
    return "ps"


def _parse_float_tokens(line: str) -> list[float]:
    vals: list[float] = []
    for tok in line.strip().split():
        low = tok.lower()
        if low in ("na", "nan"):
            vals.append(float("nan"))
        else:
            vals.append(float(tok))
    return vals


# --------------------------------------------------------------------------- #
#                             Glotaran ASCII (.ascii)                         #
# --------------------------------------------------------------------------- #
register_dataset_glob("Glotaran_ASCII", "*.ascii")


@register_dataset_file_filter("Glotaran_ASCII")
def _glotaran_ascii_is_file(path: Path) -> bool:
    p = Path(path)
    if not p.is_file() or p.suffix.lower() != ".ascii":
        return False
    try:
        with open(p, encoding="utf-8", errors="ignore") as fh:
            for _ in range(16):
                line = fh.readline()
                if not line:
                    break
                low = line.strip().lower()
                if low.startswith(("time explicit", "wavelength explicit")):
                    return True
    except OSError:
        return False
    return False


@register_loader("Glotaran_ASCII")
def read_glotaran_ascii(path: str | PathLike) -> LoaderResult:
    """Read a Glotaran / TIMP ``.ascii`` file (``Time explicit`` or ``Wavelength explicit``)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Glotaran ASCII file not found: {path}")

    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    mode: str | None = None
    mode_idx = -1
    header_lines: list[str] = []

    for idx, line in enumerate(lines):
        stripped = line.strip()
        low = stripped.lower()
        if low.startswith("time explicit"):
            mode = "time"
            mode_idx = idx
            break
        if low.startswith("wavelength explicit"):
            mode = "wavelength"
            mode_idx = idx
            break
        if stripped:
            header_lines.append(stripped)

    if mode is None:
        raise ValueError(f"Missing 'Time explicit' or 'Wavelength explicit' marker in {p}")

    header_blob = "\n".join(header_lines)
    header_low = header_blob.lower()

    # Extract explicit metadata hints when written by PyMORGAN or user headers
    m_time = re.search(r"time_unit\s*=\s*([^\s|]+)", header_blob, re.IGNORECASE)
    m_probe = re.search(r"probe_unit\s*=\s*([^\s|]+)", header_blob, re.IGNORECASE)
    m_sig = re.search(r"signal_unit\s*=\s*([^\s|]+)", header_blob, re.IGNORECASE)

    data_lines = lines[mode_idx + 1 :]
    pos = 0
    while pos < len(data_lines):
        s = data_lines[pos].strip()
        if not s or s.startswith("#") or s.lower().startswith("intervalnr"):
            pos += 1
            continue
        break

    if pos >= len(data_lines):
        raise ValueError(f"No axis/data rows found in Glotaran ASCII file {p}")

    col_axis = np.asarray(_parse_float_tokens(data_lines[pos]), dtype=float)
    pos += 1

    row_axis_vals: list[float] = []
    matrix_rows: list[list[float]] = []
    for raw_ln in data_lines[pos:]:
        s = raw_ln.strip()
        if not s or s.startswith("#"):
            continue
        vals = _parse_float_tokens(s)
        if len(vals) < 2:
            continue
        if len(vals) != len(col_axis) + 1:
            raise ValueError(
                f"Row length {len(vals)} does not match expected {len(col_axis) + 1} in {p}"
            )
        row_axis_vals.append(vals[0])
        matrix_rows.append(vals[1:])

    if not row_axis_vals:
        raise ValueError(f"No data matrix rows found in Glotaran ASCII file {p}")

    row_axis = np.asarray(row_axis_vals, dtype=float)
    mat = np.asarray(matrix_rows, dtype=float)

    if mode == "time":
        delays = col_axis
        probe = row_axis
        z_2d = mat.T  # [Ndelays x Npixels]
    else:
        probe = col_axis
        delays = row_axis
        z_2d = mat  # [Ndelays x Npixels]

    sig_hint = m_sig.group(1).strip().lower() if m_sig else ""
    if sig_hint in ("od", "da", "deltaa") or (
        not sig_hint
        and ("mod" not in header_low)
        and (re.search(r"\b(od|da)\b", header_low) is not None)
    ):
        z_2d = z_2d * 1000.0

    units_L = _normalize_probe_unit(m_probe.group(1) if m_probe else header_blob, probe)
    units_T = _normalize_time_unit(m_time.group(1) if m_time else header_blob)

    Zavg_R = z_2d[..., np.newaxis]
    Nscans = np.nan
    Zss_R = np.full((*Zavg_R.shape, 1), np.nan)
    Zstdv = np.zeros_like(Zavg_R)

    Units = hlp.units2dic(units_L, units_T, "x1E3")
    Units["cal_source"] = "Dataset probe axis (Glotaran ASCII)"
    return Zavg_R, delays, probe, Units, Nscans, Zss_R, Zstdv


def write_glotaran_ascii(
    dataset,
    path: str | PathLike,
    *,
    orientation: str = "time",
    signal_unit: str = "mOD",
    use_corrected: bool = True,
    detector: int = 0,
) -> Path:
    """Export a :class:`~pymorgan.oneD.Dataset1D` to Glotaran / TIMP ``.ascii`` format."""
    target = Path(path)
    if target.suffix.lower() != ".ascii":
        target = target.with_suffix(".ascii")
    target.parent.mkdir(parents=True, exist_ok=True)

    orient = orientation.strip().lower()
    if orient not in ("time", "wavelength"):
        raise ValueError(f"orientation must be 'time' or 'wavelength', got {orientation!r}")

    sig_u = signal_unit.strip()
    sig_low = sig_u.lower()
    if sig_low not in ("mod", "od", "da"):
        raise ValueError(f"signal_unit must be 'mOD' or 'OD', got {signal_unit!r}")

    det = int(np.clip(detector, 0, dataset.n_detectors - 1))
    delays = np.asarray(dataset.delays, dtype=float).ravel()
    probe = np.asarray(dataset._detector_probe(det), dtype=float).ravel()
    z_all = np.asarray(dataset.Z if use_corrected else dataset.Zavg_R, dtype=float)
    z_mat = z_all[:, :, det] if z_all.ndim == 3 else z_all

    if sig_low in ("od", "da"):
        z_mat = z_mat / 1000.0
        sig_token = "OD"
    else:
        sig_token = "mOD"

    units_dict = dataset.units if isinstance(dataset.units, dict) else {}
    time_unit = str(units_dict.get("unitsT_ltx", "ps")).replace("$", "")
    lbl = units_dict.get("unitsL_lbl", "")
    if lbl == "Wavenumber":
        probe_unit = "cm-1"
    elif lbl == "Wavelength":
        probe_unit = "nm"
    elif lbl == "Energy":
        probe_unit = "eV"
    else:
        probe_unit = str(units_dict.get("unitsL_ltx", "nm")).replace("$", "")

    header_1 = (
        f"# PyMORGAN Glotaran ASCII export | "
        f"time_unit={time_unit} | probe_unit={probe_unit} | signal_unit={sig_token}"
    )
    header_2 = f"Original data_type: {getattr(dataset, 'data_type', None) or 'PDAT'}"

    if orient == "time":
        mode_line = "Time explicit"
        col_axis = delays
        row_axis = probe
        table = z_mat.T  # [Npixels x Ndelays]
    else:
        mode_line = "Wavelength explicit"
        col_axis = probe
        row_axis = delays
        table = z_mat  # [Ndelays x Npixels]

    lines = [
        header_1,
        header_2,
        mode_line,
        f"Intervalnr {len(col_axis)}",
        "\t".join(f"{v:.10g}" for v in col_axis),
    ]
    for r_val, row_vals in zip(row_axis, table, strict=True):
        lines.append(f"{r_val:.10g}\t" + "\t".join(f"{v:.10g}" for v in row_vals))

    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


# --------------------------------------------------------------------------- #
#                         Glotaran NetCDF4 / CF (.nc)                         #
# --------------------------------------------------------------------------- #
register_dataset_glob("Glotaran_NC", "*.nc")


@register_dataset_file_filter("Glotaran_NC")
def _glotaran_nc_is_file(path: Path) -> bool:
    p = Path(path)
    if not p.is_file() or p.suffix.lower() != ".nc":
        return False
    try:
        if not h5py.is_hdf5(p):
            return False
        with h5py.File(p, "r") as f:
            keys = set(f.keys())
            has_data = any(
                k in keys for k in ("data", "__xarray_dataarray_variable__", "dataset")
            ) or any(
                isinstance(f[k], h5py.Dataset)
                and f[k].ndim >= 2
                and _attr_str(f[k].attrs.get("CLASS", "")) != "DIMENSION_SCALE"
                for k in keys
            )
            return has_data
    except Exception:
        return False


def _pick_nc_var(f: h5py.File, candidates: tuple[str, ...]) -> h5py.Dataset | None:
    for name in candidates:
        if name in f and isinstance(f[name], h5py.Dataset):
            return f[name]
    return None


@register_loader("Glotaran_NC")
def read_glotaran_nc(path: str | PathLike) -> LoaderResult:
    """Read a Glotaran / pyglotaran NetCDF4 (CF-1.8 / HDF5) dataset (``.nc``)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Glotaran NetCDF file not found: {path}")
    if not h5py.is_hdf5(p):
        raise ValueError(f"File {p.name} is not an HDF5/NetCDF4 file.")

    with h5py.File(p, "r") as f:
        time_ds = _pick_nc_var(f, ("time", "delay", "delays", "t"))
        spec_ds = _pick_nc_var(
            f,
            (
                "spectral",
                "wavelength",
                "wavelengths",
                "wavenumber",
                "wavenumbers",
                "probe",
                "frequency",
            ),
        )
        data_ds = _pick_nc_var(f, ("data", "__xarray_dataarray_variable__", "dataset"))
        if data_ds is None:
            for k in f.keys():
                obj = f[k]
                if (
                    isinstance(obj, h5py.Dataset)
                    and obj.ndim >= 2
                    and _attr_str(obj.attrs.get("CLASS", "")) != "DIMENSION_SCALE"
                    and not k.endswith("_std")
                ):
                    data_ds = obj
                    break
        if data_ds is None:
            raise ValueError(f"No 2D data variable found in NetCDF4 file {p.name}")

        # Resolve dimensions via attached dimension scales if coordinates were named differently
        dim_names: list[str] = []
        try:
            for i in range(data_ds.ndim):
                lbl = data_ds.dims[i].label or ""
                if not lbl and len(data_ds.dims[i]) > 0:
                    lbl = Path(data_ds.dims[i][0].name).name
                dim_names.append(lbl.lower())
        except Exception:
            dim_names = []

        if time_ds is None and dim_names:
            for dn in dim_names:
                if (
                    dn in f
                    and isinstance(f[dn], h5py.Dataset)
                    and dn in ("time", "delay", "delays", "t")
                ):
                    time_ds = f[dn]
                    break
        if spec_ds is None and dim_names:
            for dn in dim_names:
                if (
                    dn in f
                    and isinstance(f[dn], h5py.Dataset)
                    and dn not in ("time", "delay", "delays", "t")
                ):
                    spec_ds = f[dn]
                    break

        z_arr = np.asarray(data_ds[:], dtype=float)
        if time_ds is None:
            delays = np.arange(z_arr.shape[0], dtype=float)
            time_unit_raw = "ps"
        else:
            delays = np.asarray(time_ds[:], dtype=float).ravel()
            time_unit_raw = _attr_str(time_ds.attrs.get("units", "ps"))

        if spec_ds is None:
            probe = np.arange(z_arr.shape[1], dtype=float)
            probe_unit_raw = ""
        else:
            probe = np.asarray(spec_ds[:], dtype=float).ravel()
            probe_unit_raw = (
                _attr_str(spec_ds.attrs.get("units", ""))
                + " "
                + _attr_str(spec_ds.attrs.get("long_name", ""))
            )

        # Orient to [Ndelays x Npixels x Ndet]
        if z_arr.ndim == 2:
            if z_arr.shape == (len(probe), len(delays)) and len(probe) != len(delays):
                z_arr = z_arr.T
            elif len(dim_names) >= 2 and dim_names[0] in (
                "spectral",
                "wavelength",
                "wavenumber",
                "probe",
            ):
                z_arr = z_arr.T
            Zavg_R = z_arr[..., np.newaxis]
        elif z_arr.ndim == 3:
            if (
                z_arr.shape[0] == len(probe)
                and z_arr.shape[1] == len(delays)
                and len(probe) != len(delays)
            ):
                z_arr = np.transpose(z_arr, (1, 0, 2))
            Zavg_R = z_arr
        else:
            raise ValueError(f"Unsupported data dimensionality {z_arr.shape} in {p.name}")

        std_ds = _pick_nc_var(f, ("data_std", "std", "noise", "weight"))
        if std_ds is not None:
            std_arr = np.asarray(std_ds[:], dtype=float)
            if std_arr.ndim == 2:
                if std_arr.shape == (len(probe), len(delays)) and len(probe) != len(delays):
                    std_arr = std_arr.T
                elif len(dim_names) >= 2 and dim_names[0] in (
                    "spectral",
                    "wavelength",
                    "wavenumber",
                    "probe",
                ):
                    std_arr = std_arr.T
                Zstdv = std_arr[..., np.newaxis]
            elif std_arr.ndim == 3:
                Zstdv = std_arr
            else:
                Zstdv = np.zeros_like(Zavg_R)
        else:
            Zstdv = np.zeros_like(Zavg_R)

        data_unit = _attr_str(data_ds.attrs.get("units", "")).strip().lower()
        if data_unit in ("od", "da", "deltaa"):
            Zavg_R = Zavg_R * 1000.0
            if std_ds is not None:
                Zstdv = Zstdv * 1000.0

    units_L = _normalize_probe_unit(probe_unit_raw, probe)
    units_T = _normalize_time_unit(time_unit_raw)
    Nscans = np.nan
    Zss_R = np.full((*Zavg_R.shape, 1), np.nan)
    Units = hlp.units2dic(units_L, units_T, "x1E3")
    Units["cal_source"] = "Dataset probe axis (Glotaran NetCDF4)"
    return Zavg_R, delays, probe, Units, Nscans, Zss_R, Zstdv


def write_glotaran_nc(
    dataset,
    path: str | PathLike,
    *,
    signal_unit: str = "mOD",
    use_corrected: bool = True,
    detector: int = 0,
    write_noise: bool = True,
) -> Path:
    """Export a :class:`~pymorgan.oneD.Dataset1D` as a CF-1.8 / NetCDF4-compatible ``.nc`` file."""
    target = Path(path)
    if target.suffix.lower() != ".nc":
        target = target.with_suffix(".nc")
    target.parent.mkdir(parents=True, exist_ok=True)

    sig_u = signal_unit.strip()
    sig_low = sig_u.lower()
    if sig_low not in ("mod", "od", "da"):
        raise ValueError(f"signal_unit must be 'mOD' or 'OD', got {signal_unit!r}")

    det = int(np.clip(detector, 0, dataset.n_detectors - 1))
    delays = np.asarray(dataset.delays, dtype=float).ravel()
    probe = np.asarray(dataset._detector_probe(det), dtype=float).ravel()
    z_all = np.asarray(dataset.Z if use_corrected else dataset.Zavg_R, dtype=float)
    z_mat = z_all[:, :, det] if z_all.ndim == 3 else z_all

    noise_mat = None
    if write_noise:
        n_arr = dataset.noise_array()
        if n_arr is not None:
            n_full = np.asarray(n_arr, dtype=float)
            noise_mat = n_full[:, :, det] if n_full.ndim == 3 else n_full

    if sig_low in ("od", "da"):
        z_mat = z_mat / 1000.0
        if noise_mat is not None:
            noise_mat = noise_mat / 1000.0
        z_unit_str = "OD"
    else:
        z_unit_str = "mOD"

    units_dict = dataset.units if isinstance(dataset.units, dict) else {}
    time_unit = str(units_dict.get("unitsT_ltx", "ps")).replace("$", "")
    lbl = str(units_dict.get("unitsL_lbl", "Spectral"))
    if lbl == "Wavenumber":
        probe_unit = "cm-1"
    elif lbl == "Wavelength":
        probe_unit = "nm"
    elif lbl == "Energy":
        probe_unit = "eV"
    else:
        probe_unit = str(units_dict.get("unitsL_ltx", "nm")).replace("$", "")

    with h5py.File(target, "w") as f:
        f.attrs["Conventions"] = "CF-1.8"
        f.attrs["pymorgan_format"] = "Glotaran_NC"
        f.attrs["source"] = str(getattr(dataset, "source", None) or "")

        t_ds = f.create_dataset("time", data=delays)
        t_ds.make_scale("time")
        t_ds.attrs["units"] = time_unit
        t_ds.attrs["long_name"] = str(units_dict.get("unitsT_lbl", "Delay"))

        s_ds = f.create_dataset("spectral", data=probe)
        s_ds.make_scale("spectral")
        s_ds.attrs["units"] = probe_unit
        s_ds.attrs["long_name"] = lbl

        data_ds = create_compressed_dataset(
            f,
            "data",
            z_mat,
            units=z_unit_str,
            long_name="Transient absorption signal",
        )
        data_ds.dims[0].label = "time"
        data_ds.dims[1].label = "spectral"
        data_ds.dims[0].attach_scale(t_ds)
        data_ds.dims[1].attach_scale(s_ds)

        if noise_mat is not None:
            std_ds = create_compressed_dataset(
                f,
                "data_std",
                noise_mat,
                units=z_unit_str,
                long_name="Signal standard deviation",
            )
            std_ds.dims[0].label = "time"
            std_ds.dims[1].label = "spectral"
            std_ds.dims[0].attach_scale(t_ds)
            std_ds.dims[1].attach_scale(s_ds)

    return target

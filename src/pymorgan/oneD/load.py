"""Loading stage of the 1-D pipeline: readers and their registry.

A *loader* maps a file path to the seven fields consumed by
:class:`pymorgan.oneD.Dataset1D`::

    Zavg_R, delays, probe, Units, Nscans, Zss_R, Zstdv

==========  ===========================================================
Field       Meaning / shape
==========  ===========================================================
Zavg_R      Averaged signal, ``[Ndelays x Npixels x Ndetectors]``
delays      Delay axis, ``[Ndelays]``
probe       Spectral (probe) axis, ``[Npixels]``
Units       Unit dictionary (see ``helpers.units2dic``)
Nscans      Number of scans, or ``numpy.nan`` if unavailable
Zss_R       Single-scan signal, ``[Ndelays x Npixels x Ndetectors x Nscans]``
Zstdv       Standard deviation of the signal, ``[Ndelays x Npixels]``
==========  ===========================================================

New instrument formats are added by decorating a reader with
:func:`register_loader`.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from pymorgan import helpers as hlp

from ..log import get_logger

logger = get_logger(__name__)

LoaderResult = tuple[Any, Any, Any, dict, Any, Any, Any]
Loader = Callable[[str], LoaderResult]

_LOADERS: dict[str, Loader] = {}


def register_loader(*names: str) -> Callable[[Loader], Loader]:
    """Register ``fn`` under one or more data-type names."""

    def decorator(fn: Loader) -> Loader:
        for name in names:
            _LOADERS[name] = fn
        return fn

    return decorator


def get_loader(name: str) -> Loader:
    """Return the loader registered for ``name`` or raise ``KeyError``."""
    try:
        return _LOADERS[name]
    except KeyError:
        raise KeyError(
            f"No 1-D loader registered for {name!r}. Available: {available_loaders()}"
        ) from None


def available_loaders() -> list[str]:
    """Return the sorted list of registered data-type names."""
    return sorted(_LOADERS)


# --------------------------------------------------------------------------- #
#                    Browser registry: globs / dir detectors                  #
# --------------------------------------------------------------------------- #
# A 1-D data type is either *file-based* (a dataset is a single file, found by a
# glob) or *directory-based* (a dataset is a folder whose name matches a marker
# file inside it). The GUI dataset browser consults this registry to decide
# whether a sub-folder is a loadable dataset (shown as ``<name>``) or a plain
# navigable folder (shown as ``[name]``).
_DATATYPE_GLOBS: dict[str, str] = {"PDAT": "*.pdat"}
_DATASET_DETECTORS: dict[str, Callable[[Path], bool]] = {}
_DATASET_DESCRIBERS: dict[str, Callable[[Path], dict]] = {}
_DATASET_FILE_FILTERS: dict[str, Callable[[Path], bool]] = {}


def register_dataset_glob(name: str, pattern: str) -> None:
    """Declare ``name`` as a file-based format whose datasets match ``pattern``."""
    _DATATYPE_GLOBS[name] = pattern


def register_dataset_detector(
    *names: str,
) -> Callable[[Callable[[Path], bool]], Callable[[Path], bool]]:
    """Register a ``folder -> bool`` predicate marking a directory-based dataset."""

    def decorator(fn: Callable[[Path], bool]) -> Callable[[Path], bool]:
        for name in names:
            _DATASET_DETECTORS[name] = fn
        return fn

    return decorator


def register_dataset_describer(
    *names: str,
) -> Callable[[Callable[[Path], dict]], Callable[[Path], dict]]:
    """Register a ``folder -> dict`` describing per-dataset selection options.

    The dict is advisory metadata the GUI uses to configure selectors (e.g.
    ``{"n_spectra": int, "n_slowmod": int}``). Returns ``None`` when no
    describer is registered for the data type.
    """

    def decorator(fn: Callable[[Path], dict]) -> Callable[[Path], dict]:
        for name in names:
            _DATASET_DESCRIBERS[name] = fn
        return fn

    return decorator


def register_dataset_file_filter(
    *names: str,
) -> Callable[[Callable[[Path], bool]], Callable[[Path], bool]]:
    """Register a ``file_path -> bool`` predicate filtering dataset browser files."""

    def decorator(fn: Callable[[Path], bool]) -> Callable[[Path], bool]:
        for name in names:
            _DATASET_FILE_FILTERS[name] = fn
        return fn

    return decorator


def is_dataset_file(name: str, file_path: str | PathLike) -> bool:
    """Whether ``file_path`` is an averaged dataset file for type ``name``."""
    fn = _DATASET_FILE_FILTERS.get(name)
    if fn is None:
        return True
    try:
        return bool(fn(Path(file_path)))
    except OSError:
        return False


def dataset_glob(name: str) -> str | None:
    """Return the file glob for a file-based format, or ``None``."""
    return _DATATYPE_GLOBS.get(name)


def is_directory_format(name: str) -> bool:
    """Whether ``name`` stores each dataset as a directory (has a detector)."""
    return name in _DATASET_DETECTORS


def is_dataset_dir_status(name: str, folder: str | PathLike) -> tuple[bool, bool]:
    """Whether ``folder`` is a candidate/valid dataset of type ``name``.

    Returns ``(is_candidate, is_valid)``.
    """
    fn = _DATASET_DETECTORS.get(name)
    if not fn:
        return False, False
    try:
        res = fn(Path(folder))
        if isinstance(res, tuple):
            return bool(res[0]), bool(res[1])
        elif res:
            return True, True
        else:
            return False, False
    except OSError:
        return False, False


def is_dataset_dir(name: str, folder: str | PathLike) -> bool:
    """Whether ``folder`` is a loadable dataset of type ``name``."""
    _, valid = is_dataset_dir_status(name, folder)
    return valid


def describe_dataset(name: str, folder: str | PathLike) -> dict | None:
    """Return per-dataset selection metadata, or ``None`` if unavailable."""
    fn = _DATASET_DESCRIBERS.get(name)
    if fn is None:
        return None
    try:
        return fn(Path(folder))
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------- #
#                                  Readers                                     #
# --------------------------------------------------------------------------- #
@register_loader("PDAT")
def read_PDAT(datafilename: str) -> LoaderResult:
    """Read a PyMORGAN ``.pdat`` file (averaged data already in mOD).

    The first line holds the time and probe units separated by ``*``; the first
    data row is the probe axis (leading 0), and every later row is a delay
    followed by the signal at each probe coordinate.
    """
    import pandas as pd

    data = np.array(pd.read_csv(datafilename, skiprows=1, header=None))
    with open(datafilename) as f:
        units = f.readline().rstrip().split("*")

    unitsT = units[0]  # time units
    unitsL = units[1]  # probe-axis units
    unitsZ = "x1E3"

    delays = data[1:, 0]
    probe = data[0, 1:]
    Zavg_R = data[1:, 1:]  # already in mOD, no rescaling

    # Generalise to multiple detectors: [Ndelays x Npixels x Ndetectors].
    Zavg_R = Zavg_R[..., np.newaxis]

    # PDATs hold averaged data only, so there is no single-scan information.
    Nscans = np.nan
    Zss_R = np.nan * np.zeros_like(Zavg_R[..., np.newaxis])

    # Optional sibling ".pdatn" holding the per-point standard deviations in the
    # identical PDAT layout (probe-axis header row, delay-led data rows). When
    # present it supplies the noise; otherwise the standard deviation is unknown.
    nfile = Path(datafilename).with_suffix(".pdatn")
    if nfile.exists():
        ndata = np.array(pd.read_csv(nfile, skiprows=1, header=None))
        Zstdv = ndata[1:, 1:][..., np.newaxis]
    else:
        Zstdv = np.zeros_like(Zavg_R)

    Units = hlp.units2dic(unitsL, unitsT, unitsZ)
    Units["cal_source"] = "Dataset probe axis (PDAT header)"
    return Zavg_R, delays, probe, Units, Nscans, Zss_R, Zstdv


def write_pdat(
    path: str | PathLike,
    delays: np.ndarray,
    probe: np.ndarray,
    z_data: np.ndarray,
    *,
    time_unit: str = "ps",
    probe_unit: str = "cm^{-1}",
    noise: np.ndarray | None = None,
) -> Path:
    """Write a 1-D transient absorption slice to a PyMORGAN ``.pdat`` (and optional ``.pdatn``) file.

    Parameters
    ----------
    path : str or Path
        Target file path (.pdat extension added if omitted).
    delays : ndarray, shape [Ndelays]
        Delay axis values.
    probe : ndarray, shape [Npixels]
        Probe spectral axis values.
    z_data : ndarray, shape [Ndelays x Npixels]
        Signal data matrix.
    time_unit : str, default "ps"
        Delay timescale unit string (e.g. "ps", "ns", "fs").
    probe_unit : str, default "cm^{-1}"
        Probe spectral unit string (e.g. "cm^{-1}", "nm", "eV").
    noise : ndarray, optional
        Standard deviation / noise matrix [Ndelays x Npixels]. If supplied, writes a sibling ``.pdatn`` file.

    Returns
    -------
    Path
        The written .pdat file path.
    """
    target = Path(path)
    if target.suffix.lower() != ".pdat":
        target = target.with_suffix(".pdat")
    target.parent.mkdir(parents=True, exist_ok=True)

    delays_arr = np.asarray(delays, dtype=float).ravel()
    probe_arr = np.asarray(probe, dtype=float).ravel()
    z_mat = np.asarray(z_data, dtype=float)
    if z_mat.ndim == 3 and z_mat.shape[2] == 1:
        z_mat = z_mat[:, :, 0]

    header_line = f"{time_unit}*{probe_unit}"
    table = np.empty((z_mat.shape[0] + 1, z_mat.shape[1] + 1), dtype=float)
    table[0, 0] = 0.0
    table[0, 1:] = probe_arr
    table[1:, 0] = delays_arr
    table[1:, 1:] = z_mat

    with open(target, "w", newline="") as f:
        f.write(f"{header_line}\n")
        np.savetxt(f, table, delimiter=",", fmt="%.18g")

    if noise is not None:
        noise_mat = np.asarray(noise, dtype=float)
        if noise_mat.ndim == 3 and noise_mat.shape[2] == 1:
            noise_mat = noise_mat[:, :, 0]
        n_table = np.empty((noise_mat.shape[0] + 1, noise_mat.shape[1] + 1), dtype=float)
        n_table[0, 0] = 0.0
        n_table[0, 1:] = probe_arr
        n_table[1:, 0] = delays_arr
        n_table[1:, 1:] = noise_mat

        pdatn_path = target.with_suffix(".pdatn")
        with open(pdatn_path, "w", newline="") as f:
            f.write(f"{header_line}\n")
            np.savetxt(f, n_table, delimiter=",", fmt="%.18g")

    return target


def export_pdat(
    dataset,
    path: str | PathLike,
    *,
    detector: int | None = None,
    use_corrected: bool = True,
    write_noise: bool = True,
) -> list[Path]:
    """Export a :class:`pymorgan.oneD.Dataset1D` to ``.pdat`` (and optional ``.pdatn``) files.

    If the dataset has multiple detectors and ``detector`` is None, exports one
    ``.pdat`` per detector with a ``_DET{i}`` filename suffix.

    Returns the list of exported Paths.
    """
    base_pdat_path = Path(path)
    if base_pdat_path.suffix.lower() != ".pdat":
        base_pdat_path = base_pdat_path.with_suffix(".pdat")

    delays = np.asarray(dataset.delays, dtype=float)
    noise_arr = dataset.noise_array() if write_noise else None

    lbl = dataset.units.get("unitsL_lbl", "")
    if lbl == "Wavenumber":
        probe_unit = "cm^{-1}"
    elif lbl == "Wavelength":
        probe_unit = "nm"
    elif lbl == "Energy":
        probe_unit = "eV"
    else:
        probe_unit = dataset.units.get("unitsL_ltx", "nm").replace("$", "")

    time_unit = dataset.units.get("unitsT_ltx", "ps").replace("$", "")

    z_all = np.asarray(dataset.Z if use_corrected else dataset.Zavg_R, dtype=float)
    n_det = dataset.n_detectors

    if detector is not None:
        det_indices = [int(detector)]
    else:
        det_indices = list(range(n_det))

    exported: list[Path] = []
    for d in det_indices:
        det_num = d + 1
        if n_det > 1 and detector is None:
            base_stem = base_pdat_path.stem
            if re.search(r"_DET\d+", base_stem, flags=re.IGNORECASE):
                target_stem = re.sub(r"_DET\d+", f"_DET{det_num}", base_stem, flags=re.IGNORECASE)
            else:
                target_stem = f"{base_stem}_DET{det_num}"
            pdat_path = base_pdat_path.with_name(target_stem + ".pdat")
        else:
            pdat_path = base_pdat_path

        probe_d = np.asarray(dataset._detector_probe(d), dtype=float)
        z_d = z_all[:, :, d] if z_all.ndim == 3 else z_all

        noise_d = None
        if noise_arr is not None:
            n_arr = np.asarray(noise_arr, dtype=float)
            noise_d = n_arr[:, :, d] if n_arr.ndim == 3 else n_arr

        written = write_pdat(
            pdat_path,
            delays,
            probe_d,
            z_d,
            time_unit=time_unit,
            probe_unit=probe_unit,
            noise=noise_d,
        )
        exported.append(written)

    return exported


def save_1D_hdf5(dataset, path: str | PathLike) -> Path:
    """Save a :class:`pymorgan.oneD.Dataset1D` instance to a self-contained HDF5 file."""
    import json

    target = Path(path)
    if target.suffix.lower() not in (".h5", ".hdf5"):
        target = target.with_suffix(".h5")
    target.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(target, "w") as f:
        f.create_dataset("delays", data=np.asarray(dataset.delays, dtype=float))
        f.create_dataset("probe", data=np.asarray(dataset.probe, dtype=float))
        if dataset.Zavg_R is not None:
            f.create_dataset("Zavg_R", data=np.asarray(dataset.Zavg_R, dtype=float))
        if dataset.Zavg_C is not None:
            f.create_dataset("Zavg_C", data=np.asarray(dataset.Zavg_C, dtype=float))
        if dataset.Zss_R is not None:
            f.create_dataset("Zss_R", data=np.asarray(dataset.Zss_R, dtype=float))
        if dataset.Zss_C is not None:
            f.create_dataset("Zss_C", data=np.asarray(dataset.Zss_C, dtype=float))
        if dataset.Zstdv is not None:
            f.create_dataset("Zstdv", data=np.asarray(dataset.Zstdv, dtype=float))
        if dataset.bkg_avg is not None:
            f.create_dataset("bkg_avg", data=np.asarray(dataset.bkg_avg, dtype=float))
        if dataset.bkg_ss is not None:
            f.create_dataset("bkg_ss", data=np.asarray(dataset.bkg_ss, dtype=float))

        f.attrs["nscans"] = float(dataset.nscans) if np.isfinite(dataset.nscans) else -1.0
        f.attrs["data_type"] = str(dataset.data_type or "PDAT")
        f.attrs["source"] = str(dataset.source or "")
        f.attrs["format_version"] = "1.0"

        units_serializable = {
            k: v
            for k, v in dataset.units.items()
            if isinstance(v, (str, int, float, bool, list, dict))
        }
        f.attrs["units_json"] = json.dumps(units_serializable)
        if getattr(dataset, "mask_ranges", None):
            f.attrs["mask_ranges_json"] = json.dumps(dataset.mask_ranges)
        if getattr(dataset, "scan_ids", None):
            f.attrs["scan_ids_json"] = json.dumps(list(dataset.scan_ids))

    return target


@register_loader("HDF5")
def read_1D_hdf5(path: str | PathLike) -> LoaderResult:
    """Read a Dataset1D from an HDF5 file written by save_1D_hdf5."""
    import json

    p = Path(path)
    with h5py.File(p, "r") as f:
        # Auto-delegate if this is a PyMESS acquisition file opened via HDF5 loader
        if "delays" not in f and "data" in f and "data/delays_fs" in f:
            return read_PyMESS_PP(str(p))

        delays = np.asarray(f["delays"][:], dtype=float)
        probe = np.asarray(f["probe"][:], dtype=float)
        Zavg_R = np.asarray(f["Zavg_R"][:], dtype=float) if "Zavg_R" in f else None

        Zss_R = np.asarray(f["Zss_R"][:], dtype=float) if "Zss_R" in f else None
        Zstdv = np.asarray(f["Zstdv"][:], dtype=float) if "Zstdv" in f else np.zeros_like(Zavg_R)

        nscans_attr = f.attrs.get("nscans", -1.0)
        nscans = float(nscans_attr) if nscans_attr > 0 else np.nan

        units_json = f.attrs.get("units_json", "{}")
        try:
            Units = json.loads(units_json)
        except Exception:
            Units = hlp.units2dic("cm^{-1}", "ps", "x1E3")

        scan_ids = None
        if "scan_ids_json" in f.attrs:
            try:
                scan_ids = json.loads(f.attrs["scan_ids_json"])
            except Exception:
                pass

    return Zavg_R, delays, probe, Units, nscans, Zss_R, Zstdv, scan_ids


def load_1D_hdf5(path: str | PathLike):
    """Load an HDF5 archive into a fully restored :class:`pymorgan.oneD.Dataset1D`."""
    import json

    from .dataset import Dataset1D

    p = Path(path)
    with h5py.File(p, "r") as f:
        delays = np.asarray(f["delays"][:], dtype=float)
        probe = np.asarray(f["probe"][:], dtype=float)
        Zavg_R = np.asarray(f["Zavg_R"][:], dtype=float) if "Zavg_R" in f else None
        Zavg_C = np.asarray(f["Zavg_C"][:], dtype=float) if "Zavg_C" in f else None
        Zss_R = np.asarray(f["Zss_R"][:], dtype=float) if "Zss_R" in f else None
        Zss_C = np.asarray(f["Zss_C"][:], dtype=float) if "Zss_C" in f else None
        Zstdv = np.asarray(f["Zstdv"][:], dtype=float) if "Zstdv" in f else None
        bkg_avg = np.asarray(f["bkg_avg"][:], dtype=float) if "bkg_avg" in f else None
        bkg_ss = np.asarray(f["bkg_ss"][:], dtype=float) if "bkg_ss" in f else None

        nscans_attr = f.attrs.get("nscans", -1.0)
        nscans = float(nscans_attr) if nscans_attr > 0 else np.nan
        data_type = str(f.attrs.get("data_type", "HDF5"))
        source = str(f.attrs.get("source", str(p)))

        units_json = f.attrs.get("units_json", "{}")
        try:
            units = json.loads(units_json)
        except Exception:
            units = hlp.units2dic("cm^{-1}", "ps", "x1E3")

        scan_ids = None
        if "scan_ids_json" in f.attrs:
            try:
                scan_ids = json.loads(f.attrs["scan_ids_json"])
            except Exception:
                pass

        mask_ranges = None
        if "mask_ranges_json" in f.attrs:
            try:
                mask_ranges = [tuple(r) for r in json.loads(f.attrs["mask_ranges_json"])]
            except Exception:
                pass

    ds = Dataset1D(
        Zavg_R,
        delays,
        probe,
        units,
        nscans=nscans,
        Zss_R=Zss_R,
        Zstdv=Zstdv,
        source=source,
        data_type=data_type,
        scan_ids=scan_ids,
    )
    ds.Zavg_C = Zavg_C
    ds.Zss_C = Zss_C
    ds.bkg_avg = bkg_avg
    ds.bkg_ss = bkg_ss
    ds.mask_ranges = mask_ranges
    return ds


register_dataset_glob("HDF5", "*.h5")
register_dataset_glob("HDF5", "*.hdf5")


def _pymorgan_hdf5_is_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in {".h5", ".hdf5"}


register_dataset_file_filter("HDF5")(_pymorgan_hdf5_is_file)





def _stub_loader(name: str) -> Loader:
    """Build a placeholder loader that fails loudly until implemented."""

    def loader(path: str) -> LoaderResult:
        raise NotImplementedError(
            f"The {name!r} 1-D loader is not implemented yet (requested file: {path}). "
            f"Implement it here and register it via @register_loader({name!r})."
        )

    loader.__name__ = f"load_{name}"
    return loader


# --------------------------------------------------------------------------- #
#                             UniGE nsTA reader                                #
# --------------------------------------------------------------------------- #
register_dataset_glob("UniGE_nsTA", "*.dat")


@register_dataset_file_filter("UniGE_nsTA")
def _unige_nsta_is_file(path: Path) -> bool:
    """Whether path is a UniGE nsTA transient dataset file.

    Checks if the file starts with header comments (%) and contains nsTA metadata
    (e.g. "ps-ta" or "Delay, pixel, TA signal"). Excludes probe-only calibration
    files (e.g. HOLMIUM.dat, WL.dat).
    """
    if path.suffix.lower() != ".dat":
        return False
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            header_sample = f.read(2048).lower()
        if not header_sample.startswith("%"):
            return False
        # Exclude probe-only / calibration files (% pixel signal...)
        if "pixel\tsignal\trms signal" in header_sample or "pixel  signal" in header_sample:
            return False
        if "reference" in header_sample and "delay" not in header_sample:
            return False
        return (
            ("ps-ta" in header_sample)
            or ("delay, pixel, ta signal" in header_sample)
            or ("fifo-stresing-ccd" in header_sample)
            or ("dl-mach-xo-delay" in header_sample)
        )
    except Exception:
        return False


@register_loader("UniGE_nsTA")
def read_UniGE_nsTA(datafilename: str) -> LoaderResult:
    """Read a UniGE nanosecond Transient Absorption (.dat) file.

    Header lines start with '%' and contain metadata.
    Data rows contain:
    - Column 0: delay value (seconds, converted to ns)
    - Column 1: pixel index (0 to Npixels-1)
    - Column 2: TA signal
    - Column 3: rms
    - Column 4: error (noise)
    - Column 5: n samples / accumulation counts

    Returns
    -------
    Zavg_R, delays, probe, Units, Nscans, Zss_R, Zstdv, scan_ids, counts
    """
    from collections import Counter

    import scipy.io as sio

    path = Path(datafilename)
    if not path.is_file():
        raise FileNotFoundError(f"UniGE nsTA file not found: {datafilename}")

    import pandas as pd

    try:
        df = pd.read_csv(path, comment="%", sep=r"\s+", header=None, engine="c")
        if df.empty or df.shape[1] < 6:
            raise ValueError(f"Insufficient columns or empty data in UniGE nsTA file: {datafilename}")
        alldata = df.to_numpy(dtype=float)
    except Exception:
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read().replace("\x00", "")

        lines = text.splitlines()
        data_lines = [l for l in lines if not l.startswith("%") and l.strip()]
        if not data_lines:
            raise ValueError(f"No data rows found in UniGE nsTA file: {datafilename}") from None

        parsed_rows = [[float(x) for x in l.split()] for l in data_lines]
        col_counts = Counter(len(r) for r in parsed_rows)
        expected_cols = col_counts.most_common(1)[0][0]

        valid_rows = [r for r in parsed_rows if len(r) == expected_cols]
        if not valid_rows or expected_cols < 6:
            raise ValueError(
                f"Could not parse valid data rows from UniGE nsTA file: {datafilename}"
            ) from None

        alldata = np.array(valid_rows, dtype=float)

    if alldata.ndim != 2 or alldata.shape[0] == 0:
        raise ValueError(f"No valid data rows found in UniGE nsTA file: {datafilename}")

    # Delays in seconds -> convert to nanoseconds (x 1e9)
    delays_ns = alldata[:, 0] * 1e9
    delays, u_idx, dID = np.unique(delays_ns, return_index=True, return_inverse=True)
    Ndelays = len(delays)
    if Ndelays == 0:
        raise ValueError(f"No delay points found in UniGE nsTA file: {datafilename}")

    total_rows = alldata.shape[0]
    Npixels = total_rows // Ndelays
    valid_len = Ndelays * Npixels

    # Fast vectorized extraction of signal, rms, noise, and accumulation counts
    tmpsignal = alldata[:valid_len, 2].reshape(Ndelays, Npixels)
    tmprms = alldata[:valid_len, 3].reshape(Ndelays, Npixels)
    tmpnoise = alldata[:valid_len, 4].reshape(Ndelays, Npixels)
    cts = alldata[:valid_len:Npixels, 5]

    # Filter out delays with zero counts (matching LoadData_nsTAUniGE.m)
    rmv_idx = cts == 0
    if np.any(rmv_idx):
        keep_idx = ~rmv_idx
        tmpsignal = tmpsignal[keep_idx, :]
        tmprms = tmprms[keep_idx, :]
        tmpnoise = tmpnoise[keep_idx, :]
        delays = delays[keep_idx]
        cts = cts[keep_idx]
        Ndelays = len(delays)

    # Probe wavelength calibration lookup (pix2lam.mat or CalibratedProbe.csv)
    probe = None
    cal_source = None
    cal_level = None
    csv_candidates = [
        (path.parent / "CalibratedProbe.csv", "data folder", "datadir"),
        (path.parent / "calib" / "CalibratedProbe.csv", "calib folder", "datadir"),
        (path.parent.parent / "CalibratedProbe.csv", "root folder", "rootdir"),
        (path.parent.parent / "calib" / "CalibratedProbe.csv", "parent calib folder", "rootdir"),
    ]
    for cand, loc, lvl in csv_candidates:
        if cand.is_file():
            try:
                cand_probe = np.loadtxt(cand, delimiter=",").ravel().astype(float)
                if len(cand_probe) == Npixels:
                    probe = cand_probe
                    cal_source = f"CalibratedProbe.csv ({loc})"
                    cal_level = lvl
                    break
            except Exception:
                pass

    if probe is None or len(probe) != Npixels:
        mat_candidates = [
            (path.parent / "pix2lam.mat", "data folder", "datadir"),
            (path.parent / "calib" / "pix2lam.mat", "calib folder", "datadir"),
            (path.parent.parent / "pix2lam.mat", "root folder", "rootdir"),
            (path.parent.parent / "calib" / "pix2lam.mat", "parent calib folder", "rootdir"),
        ]
        for cand, loc, lvl in mat_candidates:
            if cand.is_file():
                try:
                    mat = sio.loadmat(cand)
                    if "lam" in mat:
                        cand_probe = mat["lam"].ravel().astype(float)
                        if len(cand_probe) == Npixels:
                            probe = cand_probe
                            cal_source = f"pix2lam.mat ({loc})"
                            cal_level = lvl
                            break
                except Exception:
                    pass

    if probe is None or len(probe) != Npixels:
        probe = np.arange(1, Npixels + 1, dtype=float)
        cal_source = "Pixel indices (uncalibrated)"

    # Automatically cut edge pixels containing noise/nonsense (>100 mOD signal, >20 mOD noise, or empty <0.01 mOD)
    max_s = np.max(np.abs(tmpsignal), axis=0)
    max_n = np.max(np.abs(tmpnoise), axis=0)

    trim_start = 0
    while trim_start < Npixels:
        if max_s[trim_start] < 0.01 or max_s[trim_start] > 100.0 or max_n[trim_start] > 20.0:
            trim_start += 1
        else:
            break

    trim_end = Npixels - 1
    while trim_end > trim_start:
        if max_s[trim_end] < 0.01 or max_s[trim_end] > 100.0 or max_n[trim_end] > 20.0:
            trim_end -= 1
        else:
            break

    if trim_start > 0 or trim_end < Npixels - 1:
        keep_pix = slice(trim_start, trim_end + 1)
        tmpsignal = tmpsignal[:, keep_pix]
        tmprms = tmprms[:, keep_pix]
        tmpnoise = tmpnoise[:, keep_pix]
        probe = probe[keep_pix]

    # Reverse probe & data axes if wavelength decreases with pixel index
    if len(probe) > 1 and probe[0] >= probe[-1]:
        probe = np.flip(probe)
        tmpsignal = np.flip(tmpsignal, axis=1)
        tmprms = np.flip(tmprms, axis=1)
        tmpnoise = np.flip(tmpnoise, axis=1)

    Zavg_R = tmpsignal[..., np.newaxis]
    Zstdv = tmpnoise[..., np.newaxis]
    Nscans = np.nan
    Zss_R = np.full((Ndelays, len(probe), 1, 1), np.nan)
    scan_ids = None

    Units = hlp.units2dic("nm", "ns", "x1E3")
    Units["counts"] = cts
    Units["rms"] = tmprms
    Units["error"] = tmpnoise
    Units["cal_source"] = cal_source
    Units["cal_level"] = cal_level
    return Zavg_R, delays, probe, Units, Nscans, Zss_R, Zstdv, scan_ids, cts




# --------------------------------------------------------------------------- #
#                             UniGE fsTA reader                                #
# --------------------------------------------------------------------------- #
register_dataset_glob("UniGE_fsTA", "*.dat")


@register_dataset_file_filter("UniGE_fsTA")
def _unige_fsta_is_file(path: Path) -> bool:
    """Whether path is a UniGE fsTA transient dataset file.

    Checks if the file starts with header comments (%) and contains transient TA
    delay/program metadata (% time-delays, % program-name, step-scan-ta).
    Returns False for calibration/non-transient files (e.g. WLCORR, BLANK, etc.).
    """
    if path.suffix.lower() != ".dat":
        return False
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            header_sample = f.read(2048)
        if not header_sample.startswith("%"):
            return False
        return (
            ("% time-delays" in header_sample)
            or ("% program-name" in header_sample)
            or ("step-scan-ta" in header_sample)
        )
    except Exception:
        return False


@register_loader("UniGE_fsTA")
def read_UniGE_fsTA(datafilename: str) -> LoaderResult:
    """Read a UniGE step-scan femtosecond Transient Absorption (.dat) file.

    Header lines start with '%' and contain dataset metadata.
    Data rows contain:
    - Column 0: delay value (seconds, converted to ps)
    - Column 1: accumulation count
    - Columns 2..N: alternating Signal (uOD) and Error (uOD) pairs for each pixel

    Returns
    -------
    Zavg_R, delays, probe, Units, Nscans, Zss_R, Zstdv, scan_ids
    """
    from collections import Counter

    import scipy.io as sio

    path = Path(datafilename)
    if not path.is_file():
        raise FileNotFoundError(f"UniGE fsTA file not found: {datafilename}")

    import pandas as pd

    try:
        df = pd.read_csv(path, comment="%", sep=r"\s+", header=None, engine="c")
        if df.empty or df.shape[1] < 4:
            raise ValueError(f"Insufficient columns or empty data in UniGE fsTA file: {datafilename}")
        alldata = df.to_numpy(dtype=float)
    except Exception:
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read().replace("\x00", "")

        lines = text.splitlines()
        data_lines = [l for l in lines if not l.startswith("%") and l.strip()]
        if not data_lines:
            raise ValueError(f"No data rows found in UniGE fsTA file: {datafilename}") from None

        parsed_rows = [[float(x) for x in l.split()] for l in data_lines]
        col_counts = Counter(len(r) for r in parsed_rows)
        expected_cols = col_counts.most_common(1)[0][0]

        valid_rows = [r for r in parsed_rows if len(r) == expected_cols]
        if not valid_rows or expected_cols < 4:
            raise ValueError(
                f"Could not parse valid data rows from UniGE fsTA file: {datafilename}"
            ) from None

        alldata = np.array(valid_rows, dtype=float)

    if alldata.ndim != 2 or alldata.shape[0] == 0:
        raise ValueError(f"No valid data rows found in UniGE fsTA file: {datafilename}")

    # Delays in seconds -> convert to picoseconds (x 1e12)
    delays_ps = alldata[:, 0] * 1e12
    delays, u_idx, dID = np.unique(delays_ps, return_index=True, return_inverse=True)
    Ndelays = len(delays)
    if Ndelays == 0:
        raise ValueError(f"No delay points found in UniGE fsTA file: {datafilename}")

    Ncols = alldata.shape[1]
    Npixels = (Ncols - 2) // 2

    # Probe wavelength calibration lookup (CalibratedProbe.csv preferred, then pix2lam.mat)
    probe = None
    cal_source = None
    cal_level = None
    csv_candidates = [
        (path.parent / "CalibratedProbe.csv", "data folder", "datadir"),
        (path.parent / "calib" / "CalibratedProbe.csv", "calib folder", "datadir"),
        (path.parent.parent / "CalibratedProbe.csv", "root folder", "rootdir"),
        (path.parent.parent / "calib" / "CalibratedProbe.csv", "parent calib folder", "rootdir"),
    ]
    for cand, loc, lvl in csv_candidates:
        if cand.is_file():
            try:
                cand_probe = np.loadtxt(cand, delimiter=",").ravel().astype(float)
                if len(cand_probe) == Npixels:
                    probe = cand_probe
                    cal_source = f"CalibratedProbe.csv ({loc})"
                    cal_level = lvl
                    break
            except Exception:
                pass

    if probe is None or len(probe) != Npixels:
        mat_candidates = [
            (path.parent / "pix2lam.mat", "data folder", "datadir"),
            (path.parent / "calib" / "pix2lam.mat", "calib folder", "datadir"),
            (path.parent.parent / "pix2lam.mat", "root folder", "rootdir"),
            (path.parent.parent / "calib" / "pix2lam.mat", "parent calib folder", "rootdir"),
        ]
        for cand, loc, lvl in mat_candidates:
            if cand.is_file():
                try:
                    mat = sio.loadmat(cand)
                    if "lam" in mat:
                        cand_probe = mat["lam"].ravel().astype(float)
                        if len(cand_probe) == Npixels:
                            probe = cand_probe
                            cal_source = f"pix2lam.mat ({loc})"
                            cal_level = lvl
                            break
                except Exception:
                    pass

    if probe is None or len(probe) != Npixels:
        probe = np.arange(1, Npixels + 1, dtype=float)
        cal_source = "Pixel indices (uncalibrated)"

    # Pixel trimming calculation (0-based indexing)
    if probe[0] >= probe[-1]:
        remove_pix = np.concatenate([np.arange(0, 50), np.arange(Npixels - 96, Npixels)])
    else:
        remove_pix = np.concatenate([np.arange(0, 25), np.arange(Npixels - 11, Npixels)])

    remove_pix = np.unique(remove_pix[(remove_pix >= 0) & (remove_pix < Npixels)])

    total_rows = alldata.shape[0]
    Nscans = total_rows // Ndelays

    if Nscans >= 1:
        n_rows = Nscans * Ndelays
        valid_data = alldata[:n_rows, :]
        scandata = np.zeros((Ndelays, Npixels, Nscans), dtype=float)
        scan_err = np.zeros((Ndelays, Npixels, Nscans), dtype=float)

        for s in range(Nscans):
            IDs = s * Ndelays + np.arange(Ndelays)
            # Signal (col 2, 4, 6...) and Error (col 3, 5, 7...) divided by 1e3 (uOD -> mOD)
            tmp_sig = valid_data[IDs, 2::2] / 1e3
            tmp_err = valid_data[IDs, 3::2] / 1e3
            scandata[dID[IDs], :, s] = tmp_sig
            scan_err[dID[IDs], :, s] = tmp_err

        rawsignal = np.nanmean(scandata, axis=2)
        noise = np.nanmean(scan_err, axis=2)

        keep_pix = np.ones(Npixels, dtype=bool)
        keep_pix[remove_pix] = False

        probe = probe[keep_pix]
        rawsignal = rawsignal[:, keep_pix]
        noise = noise[:, keep_pix]
        scandata = scandata[:, keep_pix, :]
        scan_err = scan_err[:, keep_pix, :]
    else:
        Nscans = np.nan
        rawsignal = np.zeros((Ndelays, Npixels), dtype=float)
        noise = np.zeros((Ndelays, Npixels), dtype=float)
        scandata = np.full((Ndelays, Npixels, 1), np.nan)

    # Reverse probe & data axes if wavelength decreases with pixel index
    if probe[0] >= probe[-1]:
        probe = np.flip(probe)
        rawsignal = np.flip(rawsignal, axis=1)
        noise = np.flip(noise, axis=1)
        if not np.isnan(Nscans):
            scandata = np.flip(scandata, axis=1)

    Zavg_R = rawsignal[..., np.newaxis]  # [Ndelays x Npixels x 1]
    Zstdv = noise[..., np.newaxis]
    Zss_R = (
        scandata[:, :, np.newaxis, :]
        if not np.isnan(Nscans)
        else np.full((Ndelays, len(probe), 1, 1), np.nan)
    )
    scan_ids = list(range(int(Nscans))) if not np.isnan(Nscans) else None

    Units = hlp.units2dic("nm", "ps", "x1E3")
    Units["cal_source"] = cal_source
    Units["cal_level"] = cal_level
    return Zavg_R, delays, probe, Units, Nscans, Zss_R, Zstdv, scan_ids


# --------------------------------------------------------------------------- #
#                             UoS IRpp reader                                 #
# --------------------------------------------------------------------------- #
def _parse_uos_lg(lg_file: Path) -> tuple[list[float], list[float]]:
    """Parse spectrograph grating code and central wavelength from a UoS .LG log file."""
    gratings = []
    cwls = []
    if lg_file.is_file():
        try:
            lines = lg_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            for line in lines:
                if "grating grooves/mm" in line.lower():
                    try:
                        gratings.append(float(line.split(":")[-1].strip()))
                    except ValueError:
                        pass
                elif "central wavelength nm" in line.lower():
                    try:
                        cwls.append(float(line.split(":")[-1].strip()))
                    except ValueError:
                        pass
        except Exception:
            pass
    return gratings, cwls


def _estimate_uos_probe(grating_code: float, cwl: float, n_pix: int = 96) -> np.ndarray:
    """Estimate spectrograph probe wavenumber axis (cm^-1) from grating code and CWL (nm)."""
    if grating_code == 0:
        ppnm = 0.168
    elif grating_code == 1:
        ppnm = 0.14
    elif grating_code == 2:
        ppnm = 0.068
    else:
        ppnm = 0.14

    pix = np.flip(np.arange(1, n_pix + 1, dtype=float))
    probe_nm = pix / ppnm + cwl + 60.0 - n_pix / (2.0 * ppnm)
    return 1e7 / probe_nm


@register_dataset_detector("UoS_IRpp")
def _uos_irpp_is_dataset(folder: Path) -> tuple[bool, bool]:
    """Whether folder is a University of Sheffield TRIR dataset folder.

    Returns (is_candidate, is_valid) tuple. A dataset is candidate if it
    contains .2D and .DT files. It is valid for 1D TRIR if its main .2D file
    (matching the .DT file stem) is non-empty.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return False, False
    twod_files = list(folder.glob("*.2D")) + list(folder.glob("*.2d"))
    dt_files = list(folder.glob("*.DT")) + list(folder.glob("*.dt"))
    if not twod_files or not dt_files:
        return False, False

    dt_stems = {f.stem.lower() for f in dt_files}
    main_twod = [f for f in twod_files if f.stem.lower() in dt_stems]
    if not main_twod:
        main_twod = twod_files

    is_candidate = True
    is_valid = any(f.stat().st_size > 0 for f in main_twod) and any(f.stat().st_size > 0 for f in dt_files)
    return is_candidate, is_valid


@register_loader("UoS_IRpp")
def read_UoS_IRpp(
    path: str,
    *,
    load_single_scans: bool | None = None,
) -> LoaderResult:
    """Read a University of Sheffield TRIR dataset folder or file (.2D / .DT)."""
    p = Path(path)
    folder = p if p.is_dir() else p.parent

    twod_files = sorted(list(folder.glob("*.2D")) + list(folder.glob("*.2d")))
    dt_files = sorted(list(folder.glob("*.DT")) + list(folder.glob("*.dt")))
    lg_files = sorted(list(folder.glob("*.LG")) + list(folder.glob("*.lg")))

    twod_files = [f for f in twod_files if f.stat().st_size > 0]
    dt_files = [f for f in dt_files if f.stat().st_size > 0]

    if not twod_files or not dt_files:
        raise FileNotFoundError(
            f"Could not find valid .2D and .DT data files in UoS dataset folder: {folder}"
        )

    target_stem = p.stem.upper()
    twod_file = next((f for f in twod_files if f.stem.upper() == target_stem), twod_files[0])
    dt_file = next((f for f in dt_files if f.stem.upper() == target_stem), dt_files[0])

    delays_raw = np.loadtxt(dt_file).ravel().astype(float)
    raw_data = np.loadtxt(twod_file).astype(float)

    if raw_data.ndim < 2 or raw_data.shape[0] == 0:
        raise ValueError(f"Invalid data matrix shape in UoS dataset file {twod_file}: {raw_data.shape}")

    # Scan detection (resets in delay vector)
    dT = np.diff(delays_raw)
    new_scan_idx = np.where(dT < 0)[0] + 1
    n_scans = len(new_scan_idx) + 1

    if n_scans > 1:
        n_delays = len(delays_raw) // n_scans
        delays_mat = delays_raw[: n_delays * n_scans].reshape(n_scans, n_delays).T
        delays = np.mean(delays_mat, axis=1)
        raw_data = raw_data[: n_delays * n_scans, :]
    else:
        n_delays = len(delays_raw)
        delays = delays_raw

    # Convert delay timescale if in fs or large ps
    if np.max(delays) > 5000.0:
        delays = delays / 1000.0
        units_T = "ns"
    else:
        units_T = "ps"

    total_pixels = raw_data.shape[1]
    if total_pixels >= 192:
        n_detectors = 2
        pixels_per_det = 96
    else:
        n_detectors = 1
        pixels_per_det = total_pixels

    if n_scans > 1:
        scans_reshaped = raw_data.reshape(n_scans, n_delays, total_pixels).transpose(1, 2, 0)
        Zss_R = np.zeros((n_delays, pixels_per_det, n_detectors, n_scans), dtype=float)
        for d in range(n_detectors):
            Zss_R[:, :, d, :] = scans_reshaped[:, d * pixels_per_det : (d + 1) * pixels_per_det, :]
        Zavg_R = np.nanmean(Zss_R, axis=-1)
        Zstdv = np.nanstd(Zss_R, axis=-1, ddof=1) if n_scans > 1 else np.zeros_like(Zavg_R)
        scan_ids = list(range(n_scans))
    else:
        Zavg_R = np.zeros((n_delays, pixels_per_det, n_detectors), dtype=float)
        for d in range(n_detectors):
            Zavg_R[:, :, d] = raw_data[:, d * pixels_per_det : (d + 1) * pixels_per_det]
        Zstdv = np.zeros_like(Zavg_R)
        Zss_R = np.full((n_delays, pixels_per_det, n_detectors, 1), np.nan)
        scan_ids = None

    Nscans_val = n_scans if n_scans > 1 else np.nan

    # Probe wavenumber calibration lookup
    cal_file = folder / "CalibratedProbe.csv"
    cal_loc = "data folder"
    cal_lvl = "datadir"
    if not cal_file.is_file():
        cal_file = folder.parent / "CalibratedProbe.csv"
        cal_loc = "root folder"
        cal_lvl = "rootdir"

    probe = None
    cal_source = None
    cal_level = None
    if cal_file.is_file():
        try:
            tmp_probe = np.loadtxt(cal_file, delimiter=",").ravel().astype(float)
            if n_detectors == 2 and len(tmp_probe) >= 192:
                probe = np.column_stack([tmp_probe[:96], tmp_probe[96:192]])
                cal_source = f"CalibratedProbe.csv ({cal_loc})"
                cal_level = cal_lvl
            elif len(tmp_probe) >= pixels_per_det:
                probe = tmp_probe[:pixels_per_det]
                cal_source = f"CalibratedProbe.csv ({cal_loc})"
                cal_level = cal_lvl
        except Exception:
            probe = None

    if probe is None:
        lg_file = lg_files[0] if lg_files else None
        gratings, cwls = _parse_uos_lg(lg_file) if lg_file else ([], [])
        if len(gratings) >= n_detectors and len(cwls) >= n_detectors and all(c > 0 for c in cwls[:n_detectors]):
            probe_cols = [
                _estimate_uos_probe(gratings[d], cwls[d], pixels_per_det)
                for d in range(n_detectors)
            ]
            probe = np.column_stack(probe_cols) if n_detectors > 1 else probe_cols[0]
            cal_source = f"Estimated from log file ({lg_file.name})"
        else:
            pix = np.arange(1, pixels_per_det + 1, dtype=float)
            probe = np.column_stack([pix] * n_detectors) if n_detectors > 1 else pix
            cal_source = "Pixel indices (uncalibrated)"

    Units = hlp.units2dic("cm^{-1}", units_T, "x1E3")
    Units["cal_source"] = cal_source
    Units["cal_level"] = cal_level
    return Zavg_R, delays, probe, Units, Nscans_val, Zss_R, Zstdv, scan_ids


# --------------------------------------------------------------------------- #
#                             RAL IRpp reader                                 #
# --------------------------------------------------------------------------- #
register_dataset_glob("RAL_IRpp", "*.csv")


@register_dataset_detector("RAL_IRpp")
def _ral_irpp_is_dataset(folder: Path) -> tuple[bool, bool]:
    """Whether folder is a Rutherford Appleton Laboratory (RAL LIFEtime/Ultra) dataset folder."""
    folder = Path(folder)
    if not folder.is_dir():
        return False, False
    avg_files = list(folder.glob("*Avg Diff spec*.csv")) + list(folder.glob("*avg diff spec*.csv"))
    if not avg_files:
        return False, False
    is_candidate = True
    is_valid = any(f.stat().st_size > 0 for f in avg_files)
    return is_candidate, is_valid


@register_dataset_file_filter("RAL_IRpp")
def _ral_irpp_is_file(path: Path) -> bool:
    """Whether path is a RAL IRpp CSV file."""
    if path.suffix.lower() != ".csv":
        return False
    name_lower = path.name.lower()
    return "avg diff spec" in name_lower or "cycle" in name_lower


@register_loader("RAL_IRpp")
def read_RAL_IRpp(
    path: str | PathLike,
    *,
    load_single_scans: bool | None = None,
) -> LoaderResult:
    """Read a Rutherford Appleton Laboratory (CLF Ultra / LIFEtime) IRpp dataset.

    Supports dual-detector arrays (2 detectors of 128 pixels each, 256 pixels total)
    and single-cycle scans (Run * Cycle*.csv). Delays are extracted from column headers
    and sorted into ascending order.
    """
    import pandas as pd

    from ..settings import get_settings

    p = Path(path)
    folder = p if p.is_dir() else p.parent

    # 1. Locate the average difference spectrum file
    if p.is_file() and "avg diff spec" in p.name.lower():
        avg_file = p
    else:
        avg_files = sorted(
            list(folder.glob("*Avg Diff spec*.csv")) + list(folder.glob("*avg diff spec*.csv"))
        )
        avg_files = [f for f in avg_files if f.stat().st_size > 0]
        if not avg_files:
            raise FileNotFoundError(
                f"Could not find valid 'Avg Diff spec' CSV file in RAL dataset folder: {folder}"
            )
        avg_file = avg_files[0]

    df_avg = pd.read_csv(avg_file)
    if df_avg.shape[1] < 2:
        raise ValueError(f"RAL dataset file {avg_file} must have at least 2 columns: {df_avg.shape}")

    delays_raw = np.array([float(c) for c in df_avg.columns[1:]], dtype=float)
    sort_idx = np.argsort(delays_raw)
    delays = delays_raw[sort_idx]

    # Convert delay timescale if in fs or large ps
    if np.max(delays) > 5000.0:
        delays = delays / 1000.0
        units_T = "ns"
    else:
        units_T = "ps"

    raw_mat = df_avg.iloc[:, 1:].to_numpy(dtype=float)
    raw_mat = np.where(np.isneginf(raw_mat) | np.isposinf(raw_mat), np.nan, raw_mat)

    total_pixels = raw_mat.shape[0]
    if total_pixels >= 256:
        n_detectors = 2
        pixels_per_det = 128
    elif total_pixels >= 192:
        n_detectors = 2
        pixels_per_det = 96
    else:
        n_detectors = 1
        pixels_per_det = total_pixels

    s = get_settings()
    load_single = s.load_single_scans if load_single_scans is None else load_single_scans

    # 2. Check for cycle files (single scans)
    cycle_files = sorted(
        list(folder.glob("*Cycle*.csv")) + list(folder.glob("*cycle*.csv"))
    )
    cycle_files = [f for f in cycle_files if f.stat().st_size > 0]
    n_scans = len(cycle_files)

    if load_single and n_scans > 1:
        n_delays = len(delays)
        Zss_R = np.full((n_delays, pixels_per_det, n_detectors, n_scans), np.nan, dtype=float)
        for s_idx, c_path in enumerate(cycle_files):
            try:
                df_c = pd.read_csv(c_path)
                ion = df_c.iloc[:total_pixels, 1::3].to_numpy(dtype=float)
                ioff = df_c.iloc[:total_pixels, 2::3].to_numpy(dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    da = -np.log10(ion / ioff)
                da = np.where(np.isneginf(da) | np.isposinf(da), np.nan, da)
                if da.shape[1] == n_delays:
                    da_sorted = da[:, sort_idx]
                    for d in range(n_detectors):
                        Zss_R[:, :, d, s_idx] = da_sorted[
                            d * pixels_per_det : (d + 1) * pixels_per_det, :
                        ].T
            except Exception:
                logger.debug("Failed to read cycle file %s.", c_path, exc_info=True)

        Zavg_R = np.zeros((len(delays), pixels_per_det, n_detectors), dtype=float)
        for d in range(n_detectors):
            Zavg_R[:, :, d] = raw_mat[d * pixels_per_det : (d + 1) * pixels_per_det, sort_idx].T
        Zstdv = np.nanstd(Zss_R, axis=-1, ddof=1)
        scan_ids = list(range(n_scans))
        Nscans_val = n_scans
    else:
        Zavg_R = np.zeros((len(delays), pixels_per_det, n_detectors), dtype=float)
        for d in range(n_detectors):
            Zavg_R[:, :, d] = raw_mat[d * pixels_per_det : (d + 1) * pixels_per_det, sort_idx].T
        Zstdv = np.zeros_like(Zavg_R)
        Zss_R = np.full((len(delays), pixels_per_det, n_detectors, 1), np.nan)
        scan_ids = None
        Nscans_val = n_scans if n_scans > 1 else np.nan

    # 3. Probe calibration lookup
    cal_file = folder / "CalibratedProbe.csv"
    cal_loc = "data folder"
    cal_lvl = "datadir"
    if not cal_file.is_file():
        cal_file = folder.parent / "CalibratedProbe.csv"
        cal_loc = "root folder"
        cal_lvl = "rootdir"

    probe = None
    cal_source = None
    cal_level = None
    if cal_file.is_file():
        try:
            tmp_probe = np.loadtxt(cal_file, delimiter=",").ravel().astype(float)
            if n_detectors == 2 and len(tmp_probe) >= 256:
                probe = np.column_stack([tmp_probe[:128], tmp_probe[128:256]])
                cal_source = f"CalibratedProbe.csv ({cal_loc})"
                cal_level = cal_lvl
            elif n_detectors == 2 and len(tmp_probe) >= 192:
                probe = np.column_stack([tmp_probe[:96], tmp_probe[96:192]])
                cal_source = f"CalibratedProbe.csv ({cal_loc})"
                cal_level = cal_lvl
            elif len(tmp_probe) >= pixels_per_det:
                probe = tmp_probe[:pixels_per_det]
                cal_source = f"CalibratedProbe.csv ({cal_loc})"
                cal_level = cal_lvl
        except Exception:
            probe = None

    if probe is None:
        pix = np.arange(1, pixels_per_det + 1, dtype=float)
        probe = np.column_stack([pix] * n_detectors) if n_detectors > 1 else pix
        cal_source = "Pixel indices (uncalibrated)"

    Units = hlp.units2dic("cm^{-1}", units_T, "x1E3")
    Units["cal_source"] = cal_source
    Units["cal_level"] = cal_level

    # Check for metadata in Run * info.txt
    info_files = sorted(list(folder.glob("*info.txt")) + list(folder.glob("*Info.txt")))
    if info_files:
        try:
            lines = info_files[0].read_text(encoding="utf-8", errors="ignore").splitlines()
            for line in lines:
                if ":" in line:
                    k, v = line.split(":", 1)
                    k_clean = k.strip().lower().replace(" ", "_")
                    Units[f"info_{k_clean}"] = v.strip()
        except Exception:
            pass

    return Zavg_R, delays, probe, Units, Nscans_val, Zss_R, Zstdv, scan_ids


register_loader("RAL_TRIR")(read_RAL_IRpp)
register_loader("RAL_LIFEtime")(read_RAL_IRpp)
register_dataset_glob("RAL_TRIR", "*.csv")
register_dataset_glob("RAL_LIFEtime", "*.csv")
register_dataset_detector("RAL_TRIR")(_ral_irpp_is_dataset)
register_dataset_detector("RAL_LIFEtime")(_ral_irpp_is_dataset)
register_dataset_file_filter("RAL_TRIR")(_ral_irpp_is_file)
register_dataset_file_filter("RAL_LIFEtime")(_ral_irpp_is_file)



# --------------------------------------------------------------------------- #
#                         MESS_TRIR / MESS_TRUVIS reader                       #
# --------------------------------------------------------------------------- #
# Directory-based transient-IR format written by the MESS acquisition software
# (UniGE). A dataset is a folder whose files are prefixed with the folder name::
#
#   <name>/<name>_delays.csv            delay axis (fs; converted to ps/ns)
#   <name>/<name>_wavenumbers.csv       TRIR probe axis (cm^-1), one per line
#   <name>/<name>_wavelengths.csv       TRUVIS probe axis (nm), one per line
#   <name>/<name>_signal_sp{S}_sm{M}_du0.csv        signal  [Ndelays x Npixels]
#   <name>/<name>_signal_noise_sp{S}_sm{M}_du0.csv  noise   [Ndelays x Npixels]
#   <name>/<name>_Nspectra.csv          number of spectrometer windows (sp)
#   <name>/<name>_slowModulation.csv    first line = number of slow-mod states (sm)
#   <name>/CalibratedProbe.csv          optional probe override (else parent dir)
#   <name>/temp/<name>_signal_sp{S}_sm{M}_du0_{scan}.csv   per-scan signal
#
# The signal is already a computed Delta-A in mOD (so ``unitsZ='x1E3'``).
# ``spectrum`` (sp) and ``slowmod`` (sm) select the file; ``anisotropy`` combines
# slow-modulation polarisation states.

_MESS_ANISOTROPY_MODES = (
    "NONE",
    "UV anisotropy",
    "IR anisotropy",
    "UV magic angle",
    "IR magic angle",
    "UniGE Magic Angle (calc.)",
    "UniGE Magic Angle (exp.)",
    "UniGE Anisotropy",
    "UniGE MA check",
    "UniGE Pol. diff. (Par-Perp)",
)

# Public alias for the GUI / external callers.
MESS_ANISOTROPY_MODES = _MESS_ANISOTROPY_MODES


def _mess_read_int(path: Path, default: int | None = None) -> int | None:
    """Read a single integer from the first numeric token of ``path``."""
    if not path.is_file():
        return default
    with open(path) as f:
        line = f.readline().strip()
    try:
        return int(float(line.split(",")[0]))
    except (ValueError, IndexError):
        return default


def _mess_state_counts(folder: Path, name: str) -> tuple[int, int]:
    """Return ``(n_spectra, n_slowmod)`` for a MESS_TRIR dataset folder."""
    n_spectra = _mess_read_int(folder / f"{name}_Nspectra.csv")
    if n_spectra is None:
        sad = folder / f"{name}_spectraAndDatastates.csv"
        if sad.is_file():
            col0 = np.atleast_2d(np.loadtxt(sad, delimiter=","))[:, 0]
            n_spectra = int(np.unique(col0).size)
    n_slowmod = _mess_read_int(folder / f"{name}_slowModulation.csv", default=1)
    return max(int(n_spectra or 1), 1), max(int(n_slowmod or 1), 1)


def _mess_load_2d(path: Path) -> np.ndarray:
    """Load a ``[Ndelays x Npixels]`` CSV as a 2-D float array."""
    return np.atleast_2d(np.loadtxt(path, delimiter=",")).astype(float)


def _mess_read_state(folder: Path, name: str, sp: int, sm: int) -> tuple[np.ndarray, np.ndarray]:
    """Read the ``(signal, noise)`` arrays for spectrum ``sp``, slow-mod ``sm``."""
    end = f"_sp{sp}_sm{sm}_du0.csv"
    sig = _mess_load_2d(folder / f"{name}_signal{end}")
    nfile = folder / f"{name}_signal_noise{end}"
    noi = _mess_load_2d(nfile) if nfile.is_file() else np.zeros_like(sig)
    return sig, noi


def _aniso_r(par, perp):
    """Anisotropy ``r = (par - perp) / (par + 2*perp)``."""
    return (par - perp) / (par + 2.0 * perp)


def _aniso_r_noise(par, perp, par_n, perp_n):
    """Propagated noise of :func:`_aniso_r` using analytic error propagation."""
    D = par + 2.0 * perp
    return (
        np.abs((D - (par - perp)) / D**2) * par_n
        + np.abs((-D - 2.0 * (par - perp)) / D**2) * perp_n
    )


def _mess_combine(folder, name, sp, sm, anisotropy, n_slowmod):
    """Return the ``(signal, noise)`` for the requested selection / combination.

    ``anisotropy='NONE'`` reads the single ``(sp, sm)`` state directly. The other
    modes combine slow-modulation polarisation states at a fixed ``sp`` (the
    slow-mod index ``sm`` is then ignored). Magic-angle/anisotropy formulas and
    the first-three-delays baseline subtraction are applied.
    """
    if anisotropy == "NONE":
        return _mess_read_state(folder, name, sp, sm)

    if n_slowmod < 2:
        raise ValueError(
            f"Anisotropy mode {anisotropy!r} needs >= 2 slow-modulation states; "
            f"this dataset has {n_slowmod}."
        )

    def S(m):
        return _mess_read_state(folder, name, sp, m)

    def bl(a):  # subtract the mean of the first three delays (per pixel)
        return a - np.mean(a[:3, :], axis=0)

    if anisotropy == "UV anisotropy":
        (par, par_n), (perp, perp_n) = S(1), S(0)
        return _aniso_r(par, perp), _aniso_r_noise(par, perp, par_n, perp_n)
    if anisotropy == "IR anisotropy":
        (par, par_n), (perp, perp_n) = S(0), S(1)
        return _aniso_r(par, perp), _aniso_r_noise(par, perp, par_n, perp_n)
    if anisotropy == "UV magic angle":
        (par, par_n), (perp, perp_n) = S(1), S(0)
        return (par + 2.0 * perp) / 3.0, (par_n + 2.0 * perp_n) / 3.0
    if anisotropy == "IR magic angle":
        (par, par_n), (perp, perp_n) = S(0), S(1)
        return (par + 2.0 * perp) / 3.0, (par_n + 2.0 * perp_n) / 3.0
    if anisotropy == "UniGE Magic Angle (calc.)":
        (par, par_n), (perp, perp_n) = S(0), S(2)
        par, perp = bl(par), bl(perp)
        return (par + 2.0 * perp) / 3.0, (par_n + 2.0 * perp_n) / 3.0
    if anisotropy == "UniGE Magic Angle (exp.)":
        MA, MA_n = S(1)
        return bl(MA), MA_n
    if anisotropy == "UniGE Anisotropy":
        (par, par_n), (perp, perp_n) = S(0), S(2)
        par, perp = bl(par), bl(perp)
        return _aniso_r(par, perp), _aniso_r_noise(par, perp, par_n, perp_n)
    if anisotropy == "UniGE MA check":
        (par, par_n), (perp, perp_n) = S(0), S(2)
        par, perp = bl(par), bl(perp)
        MA, MA_n = S(1)
        MA = bl(MA)
        return MA - (par + 2.0 * perp) / 3.0, MA_n + (par_n + 2.0 * perp_n) / 3.0
    if anisotropy == "UniGE Pol. diff. (Par-Perp)":
        (par, par_n), (perp, perp_n) = S(0), S(2)
        return par - perp, par_n + perp_n
    raise ValueError(f"Unknown anisotropy mode {anisotropy!r}. Valid: {_MESS_ANISOTROPY_MODES}")


def _delay_reducer(delays: np.ndarray):
    """Return ``(unique_sorted_delays, reduce_fn)``.

    ``reduce_fn`` maps any ``[Ndelays x Npixels]`` array onto the unique, sorted
    delay grid, averaging rows that share a delay (grouping duplicates by mean and sorting).
    """
    uniq, inv = np.unique(delays, return_inverse=True)
    counts = np.bincount(inv, minlength=uniq.size).astype(float)

    def reduce(arr: np.ndarray) -> np.ndarray:
        out = np.zeros((uniq.size, arr.shape[1]), dtype=float)
        np.add.at(out, inv, arr)
        return out / counts[:, None]

    return uniq, reduce


def _scan_index(path: Path) -> int:
    """Trailing integer of a per-scan temp file (``..._du0_{scan}.csv``)."""
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def _mess_single_scans(folder, name, sp, sm, anisotropy, reduce, npix, load_single):
    """Return ``(Nscans, Zss_R)`` from the ``temp/`` per-scan signal files.

    ``Nscans`` is always reported (counted from the temp folder). The per-scan arrays are loaded into ``Zss_R`` only when ``load_single``
    is true and ``anisotropy='NONE'`` (per-scan anisotropy combination is not
    reconstructed here). Otherwise ``Zss_R`` is a NaN placeholder.
    """
    tempdir = folder / "temp"
    if not tempdir.is_dir():
        # Lab-1 layout keeps the temp folder beside the dataset ("<name>temp").
        alt = folder.parent / f"{name}temp"
        tempdir = alt if alt.is_dir() else tempdir
    placeholder = np.full((reduce(np.zeros((1, npix))).shape[0], npix, 1, 1), np.nan)
    if not tempdir.is_dir():
        return np.nan, placeholder, None
    files = sorted(tempdir.glob(f"{name}_signal_sp{sp}_sm{sm}_du0_*.csv"), key=_scan_index)
    nscans = len(files)
    scan_ids = [_scan_index(f) for f in files]
    if nscans == 0:
        return np.nan, placeholder, None
    if not (load_single and anisotropy == "NONE"):
        return nscans, placeholder, scan_ids
    stack = [reduce(_mess_load_2d(f)) for f in files]  # each [Ndelays x Npixels]
    zss = np.stack(stack, axis=-1)[:, :, np.newaxis, :]  # [Ndelays x Npixels x 1 x Nscans]
    return nscans, zss, scan_ids


def _mess_probe_calibration(folder):
    """Locate a ``CalibratedProbe.csv`` override for a MESS dataset folder.

    Returns ``(path, level)`` where ``level`` is ``"datadir"`` for a file inside
    the dataset folder (level 2, preferred — lets each measurement carry its own
    calibration), ``"rootdir"`` for one in the parent/root folder (level 1), or
    ``(None, None)`` when neither exists (the dataset's own wavenumbers /
    wavelengths axis is then used).
    """
    folder = Path(folder)
    cand_data = folder / "CalibratedProbe.csv"
    if cand_data.is_file():
        return cand_data, "datadir"
    cand_root = folder.parent / "CalibratedProbe.csv"
    if cand_root.is_file():
        return cand_root, "rootdir"
    return None, None


def mess_calibration_status(folder) -> str:
    """One-line probe-calibration status for the status bar after a MESS load."""
    _path, level = _mess_probe_calibration(folder)
    if level == "datadir":
        return "Calibration loaded (datadir)"
    if level == "rootdir":
        return "Calibration loaded (rootdir)"
    return "Using dataset probe calibration"


def _find_mess_sample_info_file(path: Path | str) -> Path | None:
    """Locate a sample_info.txt file accompanying a dataset path or folder."""
    p = Path(path)
    folder = p.parent if p.is_file() else p
    if not folder.is_dir():
        return None

    stem = p.stem if p.is_file() else folder.name
    base_name = stem.split("-")[0].split("_CORRECTED")[0]

    candidates = [
        folder / f"{folder.name}_SampleInfo.txt",
        folder / f"{folder.name}_sample_info.txt",
        folder / f"{base_name}_SampleInfo.txt",
        folder / f"{base_name}_sample_info.txt",
        folder / "SampleInfo.txt",
        folder / "sample_info.txt",
        folder / "Sample_Info.txt",
        folder / "sampleinfo.txt",
    ]
    for c in candidates:
        if c.is_file():
            return c

    for child in folder.glob("*.txt"):
        name_lower = child.name.lower()
        if "sample" in name_lower and "info" in name_lower:
            return child

    return None


def parse_mess_sample_info(path: Path | str) -> str:
    """Parse a MESS sample_info.txt file into a clean summary string.

    Only non-empty key-value entries and comments are included.
    """
    p = Path(path)
    if not p.is_file():
        return ""
    try:
        content = p.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return ""
    if not content:
        return ""

    lines = [line.strip() for line in content.splitlines()]
    parsed_parts = []
    in_comments = False
    comments = []

    for line in lines:
        if not line:
            continue
        if line.lower().startswith("additional comments"):
            in_comments = True
            continue
        if in_comments:
            if line:
                comments.append(line)
        else:
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                if v:
                    parsed_parts.append(f"{k}: {v}")
            elif "\t" in line:
                k, v = line.split("\t", 1)
                k = k.strip()
                v = v.strip()
                if v:
                    parsed_parts.append(f"{k}: {v}")

    if comments:
        parsed_parts.append("Additional comments: " + ", ".join(comments))

    return "\n".join(parsed_parts)


def _read_mess(
    path: str,
    *,
    probe_suffix: str,
    units_L: str,
    spectrum: int = 0,
    slowmod: int = 0,
    anisotropy: str = "NONE",
    load_single_scans: bool | None = None,
) -> LoaderResult:
    """Shared reader for the MESS directory formats (TRIR / TRUVIS).

    The two formats are identical apart from the probe axis: TRIR stores it as
    ``<name>_wavenumbers.csv`` (cm^-1) and TRUVIS as ``<name>_wavelengths.csv``
    (nm). ``probe_suffix`` selects the file (``"wavenumbers"`` / ``"wavelengths"``)
    and ``units_L`` the axis unit (``"cm^{-1}"`` / ``"nm"``).

    Parameters
    ----------
    path
        Path to the dataset *folder* (its name prefixes every file inside).
    spectrum, slowmod
        Spectrometer-window (``sp``) and slow-modulation (``sm``) indices to
        load. ``slowmod`` is ignored when ``anisotropy != 'NONE'``.
    anisotropy
        Polarisation combination; one of :data:`_MESS_ANISOTROPY_MODES`.
    load_single_scans
        Whether to load the ``temp/`` per-scan arrays into ``Zss_R``. ``None``
        defers to the active ``Settings.load_single_scans`` (default off).
    """
    folder = Path(path)
    name = folder.name
    delays_file = folder / f"{name}_delays.csv"
    if not delays_file.is_file():
        raise FileNotFoundError(f"{folder} is not a MESS dataset (missing {delays_file.name}).")

    if load_single_scans is None:
        try:
            from pymorgan.settings import get_settings

            load_single_scans = bool(getattr(get_settings(), "load_single_scans", False))
        except Exception:
            load_single_scans = False

    n_spectra, n_slowmod = _mess_state_counts(folder, name)
    sp = int(np.clip(spectrum, 0, n_spectra - 1))
    sm = int(np.clip(slowmod, 0, n_slowmod - 1))

    delays = np.loadtxt(delays_file, delimiter=",").ravel().astype(float)

    # Probe axis: a calibrated-probe file overrides the acquisition axis
    # (dataset folder = level 2, takes precedence over the root folder = level 1);
    # otherwise the format-specific file (wavenumbers / wavelengths) is used.
    cal_path, cal_level = _mess_probe_calibration(folder)
    if cal_path is not None:
        probe = np.loadtxt(cal_path, delimiter=",").ravel().astype(float)
        loc = "data folder" if cal_level == "datadir" else "root folder"
        cal_source = f"CalibratedProbe.csv ({loc})"
    else:
        probe = (
            np.loadtxt(folder / f"{name}_{probe_suffix}.csv", delimiter=",").ravel().astype(float)
        )
        cal_source = f"Dataset probe axis ({name}_{probe_suffix}.csv)"

    rawsignal, noise = _mess_combine(folder, name, sp, sm, anisotropy, n_slowmod)

    # Delay-unit heuristic: large magnitudes mean femtoseconds -> convert
    # to picoseconds; otherwise the axis is assumed to be in nanoseconds.
    tmax = float(np.nanmax(np.abs(delays))) if delays.size else 0.0
    if tmax >= 1000.0:
        delays = delays / 1000.0
        unitsT = "ps"
    else:
        unitsT = "ns"

    # Remove duplicate delays (averaging) and sort ascending, applied identically
    # to signal, noise and (below) each single scan so they stay aligned.
    uniq_delays, reduce = _delay_reducer(delays)
    rawsignal = reduce(rawsignal)
    noise = reduce(noise)
    delays = uniq_delays

    Zavg_R = rawsignal[..., np.newaxis]  # [Ndelays x Npixels x 1]
    Zstdv = noise[..., np.newaxis]

    nscans, Zss_R, scan_ids = _mess_single_scans(
        folder, name, sp, sm, anisotropy, reduce, probe.size, load_single_scans
    )
    Nscans = nscans

    Units = hlp.units2dic(units_L, unitsT, "x1E3")
    Units["cal_source"] = cal_source
    Units["cal_level"] = cal_level
    info_file = _find_mess_sample_info_file(folder)
    if info_file is not None:
        info_str = parse_mess_sample_info(info_file)
        if info_str:
            Units["sample_info"] = info_str
    return Zavg_R, delays, probe, Units, Nscans, Zss_R, Zstdv, scan_ids


@register_loader("MESS_TRIR")
def read_MESS_TRIR(
    path: str,
    *,
    spectrum: int = 0,
    slowmod: int = 0,
    anisotropy: str = "NONE",
    load_single_scans: bool | None = None,
) -> LoaderResult:
    """Read a MESS transient-IR dataset folder (probe axis in cm^-1)."""
    return _read_mess(
        path,
        probe_suffix="wavenumbers",
        units_L="cm^{-1}",
        spectrum=spectrum,
        slowmod=slowmod,
        anisotropy=anisotropy,
        load_single_scans=load_single_scans,
    )


@register_loader("MESS_TRUVIS")
def read_MESS_TRUVIS(
    path: str,
    *,
    spectrum: int = 0,
    slowmod: int = 0,
    anisotropy: str = "NONE",
    load_single_scans: bool | None = None,
) -> LoaderResult:
    """Read a MESS transient-UV/Vis (TA) dataset folder (probe axis in nm)."""
    return _read_mess(
        path,
        probe_suffix="wavelengths",
        units_L="nm",
        spectrum=spectrum,
        slowmod=slowmod,
        anisotropy=anisotropy,
        load_single_scans=load_single_scans,
    )


def _is_mess_dataset(folder: Path) -> bool:
    """A MESS dataset folder contains ``<name>/<name>_delays.csv``."""
    folder = Path(folder)
    return (folder / f"{folder.name}_delays.csv").is_file()


@register_dataset_detector("MESS_TRIR")
def _mess_is_trir(folder: Path) -> bool:
    """A TRIR dataset is a MESS folder with **no** ``_wavelengths.csv``.

    TRIR and TRUVIS share the directory layout and differ only in the probe
    axis (cm⁻¹ vs nm). They are mutually exclusive by that file: a folder
    without a wavelengths axis is TRIR (it carries ``_wavenumbers.csv``).
    """
    folder = Path(folder)
    return (
        _is_mess_dataset(folder)
        and not (folder / f"{folder.name}_wavelengths.csv").is_file()
        and not (folder / f"{folder.name}_bins.csv").is_file()
    )


@register_dataset_detector("MESS_TRUVIS")
def _mess_is_truvis(folder: Path) -> bool:
    """A TRUVIS dataset is a MESS folder with **no** ``_wavenumbers.csv``.

    The complement of :func:`_mess_is_trir`: a folder without a wavenumbers
    axis is TRUVIS (it carries ``_wavelengths.csv``, in nm).
    """
    folder = Path(folder)
    return _is_mess_dataset(folder) and not (folder / f"{folder.name}_wavenumbers.csv").is_file()


@register_dataset_describer("MESS_TRIR", "MESS_TRUVIS")
def _mess_describe(folder: Path) -> dict:
    """Selection metadata for the GUI: spectrum / slow-mod counts and modes."""
    folder = Path(folder)
    n_spectra, n_slowmod = _mess_state_counts(folder, folder.name)
    return {
        "n_spectra": n_spectra,
        "n_slowmod": n_slowmod,
        "anisotropy_modes": list(_MESS_ANISOTROPY_MODES),
    }


# --------------------------------------------------------------------------- #
#                      PyMESS Pump-Probe HDF5 reader                          #
# --------------------------------------------------------------------------- #
register_dataset_glob("PyMESS_PP", "*.h5")


def _pymess_attr_str(val: Any) -> str:
    """Safely convert an HDF5 attribute value to a decoded string."""
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8", errors="replace")
    return str(val)


def classify_pymess_h5(filepath: str | Path) -> str:
    """Classify a PyMESS .h5 file into one of:
      - 'pump_probe': Standard 1D Pump-Probe (delay stage scan)
      - 'shaper_t2_scan': 1D Pump-Probe recorded in the 2D tab via 1-pulse shaper delay scan
      - '2d_time_domain': True 2D Time-Domain spectrum (2-pulse shaper t1 scan)
      - '2d_freq_domain': True 2D Frequency-Domain spectrum (shaper frequency scan)
      - 'other': Calibration, magic angle, find_t0, aborted empty file, or non-PyMESS HDF5
    """
    p = Path(filepath)
    if p.suffix.lower() != ".h5" or not p.is_file():
        return "other"
    try:
        if not h5py.is_hdf5(p):
            return "other"
        with h5py.File(p, "r") as f:
            meta = f["metadata"].attrs if "metadata" in f else {}
            mtype = _pymess_attr_str(meta.get("measurement_type", "")).strip().lower()

            # 1. Standard 1D Pump-Probe tab
            if mtype == "pump_probe":
                return "pump_probe"
            if (
                not mtype
                and "data/delays_fs" in f
                and (
                    "data/delta_A" in f
                    or "data/intensity" in f
                    or "data/individual_scans" in f
                    or "data/slow_modulation" in f
                )
            ):
                if (
                    "2D_Spectra" not in f
                    and "data/omega1_axis" not in f
                    and "data/angles_deg" not in f
                    and "data/interferogram" not in f
                ):
                    return "pump_probe"

            # 2. 2D Spectra tab (can be a 1-pulse shaper t2 scan OR a true 2D spectrum)
            if mtype == "2d_spectra" or "2D_Spectra" in f:
                # Verify that the file contains non-empty measurement data
                has_valid_data = False
                if "data/interferograms_raw" in f:
                    ds_iraw = f["data/interferograms_raw"]
                    if ds_iraw.ndim >= 2 and ds_iraw.size > 0:
                        arr_iraw = np.asarray(ds_iraw)
                        if not np.all(np.isnan(arr_iraw)):
                            has_valid_data = True
                if not has_valid_data and "data/slow_modulation" in f and isinstance(f["data/slow_modulation"], h5py.Group):
                    for k in f["data/slow_modulation"].keys():
                        ds_sm = f["data/slow_modulation"][k]
                        if isinstance(ds_sm, h5py.Dataset) and ds_sm.size > 0 and not np.all(np.isnan(np.asarray(ds_sm))):
                            has_valid_data = True
                            break
                if not has_valid_data and "data/individual_scans" in f:
                    ds_ind = f["data/individual_scans"]
                    if isinstance(ds_ind, h5py.Dataset) and ds_ind.size > 0 and not np.all(np.isnan(np.asarray(ds_ind))):
                        has_valid_data = True
                if not has_valid_data and "scans" in f:
                    for k in f["scans"].keys():
                        grp = f["scans"][k]
                        if isinstance(grp, h5py.Group) and (
                            "interferograms_raw" in grp or "interferogram" in grp
                        ):
                            has_valid_data = True
                            break
                if not has_valid_data and "data/surface_absorptive" in f:
                    surf = np.asarray(f["data/surface_absorptive"])
                    if surf.ndim >= 2 and surf.size > 0 and not np.all(np.isnan(surf)):
                        if "data/omega1_axis" in f and f["data/omega1_axis"].size > 0:
                            has_valid_data = True
                if not has_valid_data:
                    return "other"

                domain_mode = _pymess_attr_str(meta.get("domain_mode", "")).strip().lower()
                scan_mode = _pymess_attr_str(meta.get("scan_mode", "")).strip().lower()
                preset_name = _pymess_attr_str(meta.get("preset_name", "")).strip().lower()
                fast_mod = _pymess_attr_str(meta.get("fast_modulation", "")).strip().lower()

                shaper_attrs = {}
                if "metadata/shaper" in f:
                    shaper_attrs = f["metadata/shaper"].attrs
                elif "2D_Spectra" in f:
                    for sk in sorted(f["2D_Spectra"].keys()):
                        sgrp = f["2D_Spectra"][sk]
                        if isinstance(sgrp, h5py.Group) and "shaper" in sgrp:
                            shaper_attrs = sgrp["shaper"].attrs
                            break

                shaper_mode_attr = _pymess_attr_str(shaper_attrs.get("shaper_mode", "")).strip().lower()
                mode_attr = _pymess_attr_str(shaper_attrs.get("mode", "")).strip().lower()
                domain_attr = _pymess_attr_str(shaper_attrs.get("domain", "")).strip().lower()
                if shaper_mode_attr in ("time", "freq", "frequency") or shaper_mode_attr.startswith("freq"):
                    shaper_mode = shaper_mode_attr
                elif mode_attr in ("time", "freq", "frequency") or mode_attr.startswith("freq"):
                    shaper_mode = mode_attr
                elif domain_attr:
                    shaper_mode = domain_attr
                else:
                    shaper_mode = mode_attr

                try:
                    shaper_form = int(shaper_attrs.get("form", -1))
                except (ValueError, TypeError):
                    shaper_form = -1
                if shaper_form == -1:
                    pform_str = _pymess_attr_str(shaper_attrs.get("pulse_form", "")).strip().lower()
                    if "single" in pform_str or "1-pulse" in pform_str or "compression" in pform_str:
                        shaper_form = 0
                    elif "double" in pform_str or "2-pulse" in pform_str or "t1" in pform_str:
                        shaper_form = 1

                # 2a. Explicit Frequency-Domain 2D
                if (
                    domain_mode == "frequency_domain"
                    or "freq" in domain_mode
                    or "freq" in scan_mode
                    or shaper_mode.startswith("freq")
                    or "frequency" in preset_name
                ):
                    return "2d_freq_domain"

                # Fallback for Frequency-Domain 2D where metadata/shaper was not written:
                # 1) in FD mode, t1_delays_fs holds the scanned pump frequencies == omega1_axis, OR
                # 2) in the new PyMESS schema, t1_delays_fs is empty/[0.0] while omega1_axis holds the pump frequencies
                if "data/omega1_axis" in f:
                    w1_arr = np.asarray(f["data/omega1_axis"]).squeeze()
                    if w1_arr.ndim == 1 and w1_arr.size > 1 and not np.all(np.isnan(w1_arr)):
                        if "data/t1_delays_fs" not in f or f["data/t1_delays_fs"].size <= 1:
                            return "2d_freq_domain"
                        t1_arr = np.asarray(f["data/t1_delays_fs"]).squeeze()
                        if (
                            t1_arr.ndim == 1
                            and t1_arr.shape == w1_arr.shape
                            and np.allclose(t1_arr, w1_arr, equal_nan=True)
                        ):
                            return "2d_freq_domain"

                # 2b. Explicit Time-Domain shaper mode
                if shaper_mode.startswith("time"):
                    if shaper_form == 0:
                        return "shaper_t2_scan"
                    if shaper_form == 1:
                        return "2d_time_domain"

                # 2c. Check rotating frame in shaper_attrs (>100 cm-1 indicates 2-pulse TD 2D scan)
                for rf_key in ("effective_rotating_frame_cm1", "rotating_frame_cm1", "rotating_frame_scanned_cm1"):
                    if rf_key in shaper_attrs:
                        try:
                            if float(shaper_attrs[rf_key]) > 100.0:
                                return "2d_time_domain"
                        except (ValueError, TypeError):
                            pass

                # 2d. Fallbacks when metadata/shaper is absent (e.g. scan stopped early)
                if "data/omega1_axis" in f:
                    w1_arr = np.asarray(f["data/omega1_axis"]).squeeze()
                    if w1_arr.ndim == 1 and w1_arr.size > 1 and not np.all(np.isnan(w1_arr)):
                        # A non-zero rotating frame (e.g. 1800 cm-1) indicates a 2-pulse TD 2D scan
                        if float(w1_arr[0]) > 100.0:
                            return "2d_time_domain"

                stem_low = p.stem.lower()
                if "freqdomain" in stem_low or "freq_domain" in stem_low:
                    return "2d_freq_domain"
                if "timedomain" in stem_low or "time_domain" in stem_low or "td_2d" in stem_low:
                    return "2d_time_domain"
                if "t2scan" in stem_low or "chirp" in stem_low:
                    return "shaper_t2_scan"

                if (
                    "t2" in preset_name
                    or "chirp" in preset_name
                    or "single pulse" in preset_name
                    or "1-pulse" in preset_name
                ):
                    return "shaper_t2_scan"

                if ("pp" in fast_mod or "scatter" in fast_mod) and shaper_form != 1:
                    return "shaper_t2_scan"
                if "2d" in fast_mod or domain_mode == "time_domain" or "time" in scan_mode:
                    return "2d_time_domain"

            return "other"
    except Exception:
        return "other"


def _resolve_pymess_omega3_axis(
    f: h5py.File, path: Path, n_pix: int
) -> tuple[np.ndarray, str]:
    """Resolve the probe spectral axis (omega3 / wavelengths) for a PyMESS HDF5 file.

    Returns ``(probe_axis, source_kind)`` where ``source_kind`` is one of
    ``'dataset'``, ``'sibling'``, ``'csv'``, or ``'pixel'``.
    """
    for ds_key in ("data/omega3_axis", "data/wavelengths", "2D_Spectra/scan_001/omega3_axis_cm1"):
        if ds_key in f:
            arr = np.asarray(f[ds_key], dtype=np.float64).squeeze()
            if arr.ndim == 1 and arr.size == n_pix and not np.all(np.isnan(arr)):
                return arr, "dataset"

    # Fallback 1: check sibling .h5 files in the same folder (useful when a scan was stopped early
    # before PyMESS wrote omega3_axis)
    if path.parent.is_dir():
        try:
            for sib in sorted(path.parent.glob("*.h5")):
                if sib == path or not sib.is_file():
                    continue
                try:
                    with h5py.File(sib, "r") as sf:
                        for ds_key in ("data/omega3_axis", "data/wavelengths"):
                            if ds_key in sf:
                                arr = np.asarray(sf[ds_key], dtype=np.float64).squeeze()
                                if arr.ndim == 1 and arr.size == n_pix and not np.all(np.isnan(arr)):
                                    return arr, "sibling"
                except Exception:
                    continue
        except Exception:
            pass

    # Fallback 2: check CalibratedProbe.csv in current, parent, or Cal+Chirp directory
    if path.parent.is_dir():
        csv_candidates = [
            path.parent / "CalibratedProbe.csv",
            path.parent.parent / "CalibratedProbe.csv",
            path.parent.parent / "Cal+Chirp" / "CalibratedProbe.csv",
        ]
        for cand in csv_candidates:
            if cand.is_file():
                try:
                    arr = np.loadtxt(cand, delimiter=",").ravel().astype(np.float64)
                    if arr.size == n_pix and not np.all(np.isnan(arr)):
                        return arr, "csv"
                except Exception:
                    continue

    return np.arange(1, n_pix + 1, dtype=np.float64), "pixel"


@register_dataset_file_filter("PyMESS_PP")
def _pymess_pp_is_file(path: Path | str) -> bool:
    """Verify if path is a valid PyMESS 1D pump-probe or shaper t2-scan HDF5 file."""
    return classify_pymess_h5(path) in ("pump_probe", "shaper_t2_scan")


@register_dataset_describer("PyMESS_PP")
def _pymess_pp_describe(path: Path) -> dict:
    """Selection metadata for PyMESS pump-probe and shaper t2-scan files."""
    p = Path(path)
    if not p.is_file() or p.suffix.lower() != ".h5":
        return {"n_spectra": 1, "n_slowmod": 1}
    try:
        subtype = classify_pymess_h5(p)
        with h5py.File(p, "r") as f:
            meta = {str(k): _pymess_attr_str(v) for k, v in f.get("metadata", {}).attrs.items()}
            is_slow_mod = str(meta.get("slow_modulation_enabled", "False")).lower() in ("true", "1")
            k_slow = int(meta.get("slow_modulation_num_states", 1)) if is_slow_mod else 1
            raw_state_names = meta.get("slow_modulation_state_names", "")
            state_names = (
                [s.strip() for s in str(raw_state_names).split(",") if s.strip()]
                if raw_state_names
                else []
            )
            if "data/slow_modulation" in f and isinstance(f["data/slow_modulation"], h5py.Group):
                sm_grp = f["data/slow_modulation"]
                st_ds_keys = sorted(
                    [
                        k
                        for k in sm_grp.keys()
                        if k.startswith("state_") and isinstance(sm_grp[k], h5py.Dataset)
                    ]
                )
                if len(st_ds_keys) >= 2:
                    k_slow = max(k_slow, len(st_ds_keys))
                    if not state_names:
                        state_names = [re.sub(r"^state_\d+_", "", k) or k for k in st_ds_keys]
                elif (
                    "delta_A_parallel" in sm_grp
                    and "delta_A_perpendicular" in sm_grp
                ):
                    k_slow = max(k_slow, 2)
                    if not state_names:
                        state_names = ["Parallel", "Perpendicular"]

            for grp_name in ("scans", "data"):
                if grp_name in f and isinstance(f[grp_name], h5py.Group):
                    st_keys = sorted(
                        [k for k in f[grp_name].keys() if k.startswith("state_")]
                    )
                    if len(st_keys) >= 2:
                        k_slow = max(k_slow, len(st_keys))
                        if not state_names:
                            state_names = [
                                _pymess_attr_str(f[grp_name][k].attrs.get("state_name", k))
                                for k in st_keys
                            ]

            g_val = 1.0
            for g_key in ("g_factor", "G_factor", "gfactor"):
                if g_key in meta:
                    try:
                        g_val = float(meta[g_key])
                        break
                    except (ValueError, TypeError):
                        pass

            return {
                "n_spectra": 1,
                "n_slowmod": max(1, k_slow),
                "state_names": state_names,
                "is_pymess": True,
                "pymess_subtype": subtype,
                "g_factor": g_val,
            }
    except Exception:
        return {"n_spectra": 1, "n_slowmod": 1}


def _read_pymess_shaper_t2_scan(
    f: h5py.File,
    path: Path,
    *,
    load_single_scans: bool = True,
) -> LoaderResult:
    """Read a PyMESS 1-pulse shaper t2-delay scan recorded in the 2D tab as a 1D dataset."""
    if "data/t1_delays_fs" not in f and "data/t2_delays_fs" not in f:
        raise ValueError(
            f"File {path.name} is missing /data/t1_delays_fs for shaper t2 scan."
        )

    t1_arr = (
        np.asarray(f["data/t1_delays_fs"][:], dtype=np.float64).ravel()
        if "data/t1_delays_fs" in f
        else np.array([], dtype=np.float64)
    )
    t2_arr = (
        np.asarray(f["data/t2_delays_fs"][:], dtype=np.float64).ravel()
        if "data/t2_delays_fs" in f
        else np.array([], dtype=np.float64)
    )
    if t1_arr.size <= 1 and t2_arr.size > 1:
        raw_delays = t2_arr
        scan_along_t2 = True
    else:
        raw_delays = t1_arr
        scan_along_t2 = False

    delays = raw_delays / 1000.0
    unitsT = "ps"

    def _squeeze_t2_mat(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim == 3:
            if scan_along_t2 and a.shape[1] == 1:
                return a[:, 0, :]
            if a.shape[0] == 1:
                return a[0, :, :]
            if a.shape[2] == 1:
                return a[:, :, 0]
            return a[0, :, :]
        return a

    meta_dict: dict[str, str] = {}
    if "metadata" in f:
        for k, v in f["metadata"].attrs.items():
            meta_dict[str(k)] = _pymess_attr_str(v)

    is_slow_mod = str(meta_dict.get("slow_modulation_enabled", "False")).lower() in ("true", "1")
    k_slow = int(meta_dict.get("slow_modulation_num_states", 1)) if is_slow_mod else 1
    raw_state_names = meta_dict.get("slow_modulation_state_names", "")
    state_names = (
        [s.strip() for s in str(raw_state_names).split(",") if s.strip()]
        if raw_state_names
        else []
    )

    # 1. Check for /data/slow_modulation state_00_*, state_01_* datasets (new PyMESS schema)
    sm_ds_list: list[tuple[str, np.ndarray]] = []
    if "data/slow_modulation" in f and isinstance(f["data/slow_modulation"], h5py.Group):
        sm_grp = f["data/slow_modulation"]
        st_ds_keys = sorted(
            [
                k
                for k in sm_grp.keys()
                if k.startswith("state_") and isinstance(sm_grp[k], h5py.Dataset)
            ]
        )
        if len(st_ds_keys) >= 2:
            for k in st_ds_keys:
                sm_ds_list.append((k, _squeeze_t2_mat(sm_grp[k][:])))

    # 2. Check for explicit state_0, state_1... subgroups in /scans or /data (legacy layout)
    state_groups: list[tuple[str, h5py.Group]] = []
    if not sm_ds_list:
        for parent_key in ("scans", "data"):
            if parent_key in f and isinstance(f[parent_key], h5py.Group):
                st_keys = sorted([k for k in f[parent_key].keys() if k.startswith("state_")])
                if st_keys:
                    state_groups = [
                        (k, f[parent_key][k])
                        for k in st_keys
                        if isinstance(f[parent_key][k], h5py.Group)
                    ]
                    if state_groups:
                        break

    if sm_ds_list:
        is_slow_mod = True
        k_slow = len(sm_ds_list)
        if not state_names or len(state_names) != k_slow:
            state_names = [re.sub(r"^state_\d+_", "", k) or k for k, _ in sm_ds_list]
        Zavg_R = np.stack([mat for _, mat in sm_ds_list], axis=2)
        valid_rows = ~np.all(np.isnan(Zavg_R[:, :, 0]), axis=1)
        if not np.all(valid_rows):
            last_valid = np.where(valid_rows)[0]
            if len(last_valid) > 0:
                cut_idx = int(last_valid[-1]) + 1
                Zavg_R = Zavg_R[:cut_idx, :, :]
                delays = delays[:cut_idx]
        Nscans = 1
        Zss_R = Zavg_R[:, :, :, np.newaxis]
        Zstdv = np.zeros_like(Zavg_R)
    elif state_groups and len(state_groups) >= 2:
        is_slow_mod = True
        k_slow = len(state_groups)
        if not state_names or len(state_names) != k_slow:
            state_names = [
                _pymess_attr_str(grp.attrs.get("state_name", k)) for k, grp in state_groups
            ]
        state_avg_list: list[np.ndarray] = []
        state_scans_list: list[list[np.ndarray]] = []
        for _, grp in state_groups:
            s_scans: list[np.ndarray] = []
            for sk in sorted(grp.keys()):
                if sk.startswith("scan_") and isinstance(grp[sk], h5py.Group):
                    for dname in ("interferogram", "interferograms_raw", "delta_A"):
                        if dname in grp[sk]:
                            s_scans.append(_squeeze_t2_mat(grp[sk][dname][:]))
                            break
            if "interferograms_raw" in grp:
                s_avg = _squeeze_t2_mat(grp["interferograms_raw"][:])
            elif "interferogram" in grp:
                s_avg = _squeeze_t2_mat(grp["interferogram"][:])
            elif "delta_A" in grp:
                s_avg = _squeeze_t2_mat(grp["delta_A"][:])
            elif s_scans:
                s_avg = np.nanmean(np.stack(s_scans, axis=0), axis=0)
            else:
                raise ValueError(f"State group in {path.name} contains no interferogram matrix.")
            state_avg_list.append(s_avg)
            state_scans_list.append(s_scans if s_scans else [s_avg])

        Zavg_R = np.stack(state_avg_list, axis=2)
        valid_rows = ~np.all(np.isnan(Zavg_R[:, :, 0]), axis=1)
        if not np.all(valid_rows):
            last_valid = np.where(valid_rows)[0]
            if len(last_valid) > 0:
                cut_idx = int(last_valid[-1]) + 1
                Zavg_R = Zavg_R[:cut_idx, :, :]
                delays = delays[:cut_idx]

        n_repeats = min(len(sc) for sc in state_scans_list)
        if load_single_scans and n_repeats >= 1:
            Nscans = n_repeats
            Zss_R = np.zeros(
                (Zavg_R.shape[0], Zavg_R.shape[1], k_slow, n_repeats), dtype=np.float64
            )
            for s_idx in range(k_slow):
                for r_idx in range(n_repeats):
                    Zss_R[:, :, s_idx, r_idx] = state_scans_list[s_idx][r_idx][: len(delays), :]
            Zstdv = (
                np.nanstd(Zss_R, axis=3, ddof=1)
                if n_repeats > 1
                else np.zeros_like(Zavg_R)
            )
        else:
            Nscans = 1
            Zss_R = Zavg_R[:, :, :, np.newaxis]
            Zstdv = np.zeros_like(Zavg_R)
    else:
        # Standard single-state shaper t2 scan
        if "data/interferograms_raw" in f:
            z_2d = _squeeze_t2_mat(f["data/interferograms_raw"][:])
        elif "2D_Spectra/scan_001/interferograms_raw" in f:
            z_2d = _squeeze_t2_mat(f["2D_Spectra/scan_001/interferograms_raw"][:])
        else:
            raise ValueError(
                f"File {path.name} is missing /data/interferograms_raw for shaper t2 scan."
            )

        # Collect individual sweeps if available
        scan_list: list[np.ndarray] = []
        if load_single_scans:
            if "data/individual_scans" in f and isinstance(f["data/individual_scans"], h5py.Dataset):
                ind_arr = np.asarray(f["data/individual_scans"][:], dtype=np.float64)
                if ind_arr.ndim in (3, 4) and ind_arr.shape[0] >= 1:
                    for s_i in range(ind_arr.shape[0]):
                        s_mat = _squeeze_t2_mat(ind_arr[s_i])
                        if s_mat.shape == z_2d.shape:
                            scan_list.append(s_mat)
            if not scan_list and "scans" in f and isinstance(f["scans"], h5py.Group):
                for sk in sorted(f["scans"].keys()):
                    if sk.startswith("scan_") and isinstance(f["scans"][sk], h5py.Group):
                        for dname in ("interferogram", "interferograms_raw", "delta_A"):
                            if dname in f["scans"][sk]:
                                arr = _squeeze_t2_mat(f["scans"][sk][dname][:])
                                if arr.shape == z_2d.shape:
                                    scan_list.append(arr)
                                break
            if not scan_list and "data/signals" in f and isinstance(f["data/signals"], h5py.Group):
                sig_grp = f["data/signals"]
                num_keys = sorted(
                    [k for k in sig_grp.keys() if k.isdigit()], key=lambda x: int(x)
                )
                for nk in num_keys:
                    arr = _squeeze_t2_mat(sig_grp[nk][:])
                    if arr.shape == z_2d.shape:
                        scan_list.append(arr)

        valid_rows = ~np.all(np.isnan(z_2d), axis=1)
        if not np.all(valid_rows):
            last_valid = np.where(valid_rows)[0]
            if len(last_valid) > 0:
                cut_idx = int(last_valid[-1]) + 1
                z_2d = z_2d[:cut_idx, :]
                delays = delays[:cut_idx]
                scan_list = [s[:cut_idx, :] for s in scan_list]

        Zavg_R = z_2d[:, :, np.newaxis]
        if load_single_scans and len(scan_list) > 0:
            Nscans = len(scan_list)
            Zss_R = np.stack(scan_list, axis=-1)[:, :, np.newaxis, :]
            Zstdv = (
                np.nanstd(Zss_R, axis=3, ddof=1)
                if Nscans > 1
                else np.zeros_like(Zavg_R)
            )
        else:
            Nscans = 1
            Zss_R = Zavg_R[:, :, :, np.newaxis]
            Zstdv = np.zeros_like(Zavg_R)

    n_pix = Zavg_R.shape[1]
    probe, cal_kind = _resolve_pymess_omega3_axis(f, path, n_pix)

    probe_mean = float(np.nanmean(probe)) if probe.size > 0 else 0.0
    if probe_mean > 1200.0:
        unitsL = "cm^{-1}"
        display_L = "cm-1"
    elif probe_mean > 100.0:
        unitsL = "nm"
        display_L = "nm"
    else:
        unitsL = "px"
        display_L = "px"

    Units = hlp.units2dic(unitsL, unitsT, "x1E3")
    if cal_kind == "dataset" and unitsL != "px":
        Units["cal_source"] = f"PyMESS spectral calibration ({display_L})"
    elif cal_kind == "sibling":
        Units["cal_source"] = f"PyMESS sibling HDF5 calibration ({display_L})"
    elif cal_kind == "csv":
        Units["cal_source"] = f"CalibratedProbe.csv ({display_L})"
    else:
        Units["cal_source"] = "Uncalibrated pixel indices"

    if is_slow_mod and k_slow > 1:
        if state_names and len(state_names) == k_slow:
            Units["detectors"] = state_names
        else:
            Units["detectors"] = [f"State_{i}" for i in range(k_slow)]

    Units["pymess_metadata"] = meta_dict
    Units["pymess_subtype"] = "shaper_t2_scan"
    for g_key in ("g_factor", "G_factor", "gfactor"):
        if g_key in meta_dict:
            try:
                Units["g_factor"] = float(meta_dict[g_key])
                break
            except (ValueError, TypeError):
                pass
    if "sample_id" in meta_dict and meta_dict["sample_id"]:
        Units["sample"] = meta_dict["sample_id"]
    for wl_key in ("pump_wl_nm", "pump_wavelength_nm"):
        if wl_key in meta_dict and meta_dict[wl_key]:
            Units["pump_wl"] = meta_dict[wl_key]
            break

    if "data/intensity" in f:
        counts = _squeeze_t2_mat(f["data/intensity"][:])
        Units["counts"] = counts[: len(delays)]
    elif "data/integrated_channel" in f:
        counts = np.asarray(f["data/integrated_channel"][:], dtype=np.float64).squeeze()
        Units["counts"] = counts[: len(delays)]
    elif "data/pump_interferogram" in f:
        counts = np.asarray(f["data/pump_interferogram"][:], dtype=np.float64).squeeze()
        if counts.ndim == 1:
            Units["counts"] = counts[: len(delays)]

    if "calibration" in f:
        calib_dict = {}
        for k in f["calibration"].keys():
            calib_dict[k] = np.asarray(f["calibration"][k][:], dtype=np.float64)
        Units["pymess_calibration"] = calib_dict

    info_lines = ["Mode: Shaper t2 Scan (2D tab)"]
    if "sample_id" in meta_dict and meta_dict["sample_id"]:
        info_lines.append(f"Sample: {meta_dict['sample_id']}")
    if "solvent" in meta_dict and meta_dict["solvent"]:
        info_lines.append(f"Solvent: {meta_dict['solvent']}")
    if "fast_modulation" in meta_dict and meta_dict["fast_modulation"]:
        info_lines.append(f"Fast Modulation: {meta_dict['fast_modulation']}")
    if "pump_polarization" in meta_dict and meta_dict["pump_polarization"]:
        info_lines.append(f"Pump Polarization: {meta_dict['pump_polarization']}")
    if "probe_polarization" in meta_dict and meta_dict["probe_polarization"]:
        info_lines.append(f"Probe Polarization: {meta_dict['probe_polarization']}")
    if "waveplate_positions" in meta_dict and meta_dict["waveplate_positions"]:
        info_lines.append(f"Waveplates: {meta_dict['waveplate_positions']}")
    if is_slow_mod:
        cadence = meta_dict.get("slow_modulation_cadence", "unknown cadence")
        info_lines.append(f"Slow Modulation: {k_slow} states ({cadence})")
    if "comments" in meta_dict and meta_dict["comments"]:
        info_lines.append(f"Comments: {meta_dict['comments']}")
    if info_lines:
        Units["sample_info"] = "\n".join(info_lines)

    if Nscans > 0:
        Units["scan_ids"] = list(range(Nscans))

    return Zavg_R, delays, probe, Units, Nscans, Zss_R, Zstdv


@register_loader("PyMESS_PP")
def read_PyMESS_PP(
    datafilename: str,
    *,
    load_single_scans: bool = True,
    **kwargs,
) -> LoaderResult:
    """Read a PyMESS pump-probe or shaper t2-scan (.h5) transient absorption file.

    Handles standard single-condition scans as well as 1-pulse shaper t2 scans
    recorded in the 2D tab and multi-state slow modulation (e.g. polarization
    cycling, anisotropy, TOPAS stepping, mapped to Ndetectors).

    Parameters
    ----------
    datafilename : str
        Path to the PyMESS HDF5 (.h5) file.
    load_single_scans : bool, optional
        Whether to load individual scan passes (default: True).

    Returns
    -------
    LoaderResult : tuple
        (Zavg_R, delays, probe, Units, Nscans, Zss_R, Zstdv)
    """
    path = Path(datafilename)
    if not path.is_file():
        raise FileNotFoundError(f"PyMESS HDF5 file not found: {datafilename}")

    subtype = classify_pymess_h5(path)

    try:
        f = h5py.File(path, "r", swmr=True)
    except Exception:
        f = h5py.File(path, "r")

    with f:
        if subtype == "shaper_t2_scan" or (
            "data/t1_delays_fs" in f
            and "data/delays_fs" not in f
            and "data/interferograms_raw" in f
            and subtype not in ("2d_time_domain", "2d_freq_domain")
        ):
            return _read_pymess_shaper_t2_scan(
                f, path, load_single_scans=load_single_scans
            )

        if "data" not in f or "data/delays_fs" not in f:
            # Auto-delegate if this is a PyMORGAN native HDF5 dataset opened via PyMESS loader
            if "delays" in f and "probe" in f:
                return read_1D_hdf5(str(path))
            raise ValueError(
                f"File {path.name} is not a valid PyMESS pump-probe dataset (missing /data/delays_fs)."
            )
        if (
            "data/delta_A" not in f
            and "data/intensity" not in f
            and "data/individual_scans" not in f
            and "data/slow_modulation" not in f
        ):
            raise ValueError(
                f"File {path.name} is not a valid PyMESS pump-probe dataset (missing /data/delta_A)."
            )

        # 1. Delays: PyMESS stores in femtoseconds (fs) -> Convert to picoseconds (ps)
        raw_delays = np.asarray(f["data/delays_fs"][:], dtype=np.float64)
        delays = raw_delays / 1000.0
        unitsT = "ps"

        # 2. Probe / Spectral Axis
        if "data/wavelengths" in f:
            probe = np.asarray(f["data/wavelengths"][:], dtype=np.float64)
        else:
            if "data/delta_A" in f:
                n_pix = f["data/delta_A"].shape[1]
            elif "data/intensity" in f:
                n_pix = f["data/intensity"].shape[1]
            elif "data/individual_scans" in f:
                n_pix = f["data/individual_scans"].shape[2]
            else:
                n_pix = 0
            probe = np.arange(n_pix, dtype=np.float64)

        # Detect wavelength vs wavenumber units
        probe_mean = float(np.nanmean(probe)) if probe.size > 0 else 0.0
        if probe_mean > 1200.0:
            unitsL = "cm^{-1}"
            display_L = "cm-1"
        elif probe_mean > 100.0:
            unitsL = "nm"
            display_L = "nm"
        else:
            unitsL = "px"
            display_L = "px"

        # 3. Check for Slow Modulation
        meta_dict = {}
        if "metadata" in f:
            for k, v in f["metadata"].attrs.items():
                meta_dict[str(k)] = _pymess_attr_str(v)

        is_slow_mod = str(meta_dict.get("slow_modulation_enabled", "False")).lower() in ("true", "1")
        k_slow = int(meta_dict.get("slow_modulation_num_states", 1)) if is_slow_mod else 1
        raw_state_names = meta_dict.get("slow_modulation_state_names", "")
        state_names = (
            [s.strip() for s in str(raw_state_names).split(",") if s.strip()]
            if raw_state_names
            else []
        )

        indiv_scans = (
            np.asarray(f["data/individual_scans"][:], dtype=np.float64)
            if "data/individual_scans" in f and f["data/individual_scans"].ndim == 3
            else None
        )

        # Check /data/slow_modulation for state_00_*, state_01_* datasets (new PyMESS schema)
        sm_state_datasets: list[tuple[str, np.ndarray]] = []
        if "data/slow_modulation" in f and isinstance(f["data/slow_modulation"], h5py.Group):
            sm_grp = f["data/slow_modulation"]
            st_ds_keys = sorted(
                [
                    k
                    for k in sm_grp.keys()
                    if k.startswith("state_") and isinstance(sm_grp[k], h5py.Dataset)
                ]
            )
            if len(st_ds_keys) >= 2:
                for k in st_ds_keys:
                    sm_state_datasets.append((k, np.asarray(sm_grp[k][:], dtype=np.float64)))

        if is_slow_mod and k_slow > 1 and indiv_scans is not None and indiv_scans.shape[0] >= k_slow:
            n_total_scans = indiv_scans.shape[0]
            n_repeats = n_total_scans // k_slow
            Nscans = n_repeats

            # Trim incomplete trailing delays if file is actively acquiring
            valid_rows = ~np.all(np.isnan(indiv_scans[0]), axis=1)
            if not np.all(valid_rows):
                last_valid = np.where(valid_rows)[0]
                if len(last_valid) > 0:
                    cut_idx = int(last_valid[-1]) + 1
                    indiv_scans = indiv_scans[:, :cut_idx, :]
                    delays = delays[:cut_idx]

            Ndelays, Npixels = indiv_scans.shape[1], indiv_scans.shape[2]

            # Allocate Zavg_R: [Ndelays, Npixels, k_slow]
            Zavg_R = np.zeros((Ndelays, Npixels, k_slow), dtype=np.float64)
            # Allocate Zss_R: [Ndelays, Npixels, k_slow, n_repeats]
            Zss_R = np.zeros((Ndelays, Npixels, k_slow, n_repeats), dtype=np.float64)

            for s_idx in range(k_slow):
                sweeps = indiv_scans[s_idx : n_repeats * k_slow : k_slow, :, :]
                if s_idx < len(sm_state_datasets) and sm_state_datasets[s_idx][1].shape == (Ndelays, Npixels):
                    Zavg_R[:, :, s_idx] = sm_state_datasets[s_idx][1]
                else:
                    with np.errstate(all="ignore"):
                        Zavg_R[:, :, s_idx] = np.nanmean(sweeps, axis=0) if n_repeats > 1 else sweeps[0]
                Zss_R[:, :, s_idx, :] = np.transpose(sweeps, (1, 2, 0))

            Zstdv = np.zeros_like(Zavg_R)
            if not load_single_scans:
                Nscans = 1
                Zss_R = Zavg_R[:, :, :, np.newaxis]
        elif len(sm_state_datasets) >= 2:
            is_slow_mod = True
            k_slow = len(sm_state_datasets)
            if not state_names or len(state_names) != k_slow:
                state_names = [re.sub(r"^state_\d+_", "", k) or k for k, _ in sm_state_datasets]
            s_mats = [mat for _, mat in sm_state_datasets]
            valid_rows = ~np.all(np.isnan(s_mats[0]), axis=1)
            if not np.all(valid_rows):
                last_valid = np.where(valid_rows)[0]
                if len(last_valid) > 0:
                    cut_idx = int(last_valid[-1]) + 1
                    s_mats = [m[:cut_idx, :] for m in s_mats]
                    delays = delays[:cut_idx]
            Zavg_R = np.stack(s_mats, axis=2)
            Zstdv = np.zeros_like(Zavg_R)
            Nscans = 1
            Zss_R = Zavg_R[:, :, :, np.newaxis]
        elif (
            "data/slow_modulation/delta_A_parallel" in f
            and "data/slow_modulation/delta_A_perpendicular" in f
        ):
            d_par = np.asarray(f["data/slow_modulation/delta_A_parallel"][:], dtype=np.float64)
            d_perp = np.asarray(f["data/slow_modulation/delta_A_perpendicular"][:], dtype=np.float64)
            valid_rows = ~np.all(np.isnan(d_par), axis=1)
            if not np.all(valid_rows):
                last_valid = np.where(valid_rows)[0]
                if len(last_valid) > 0:
                    cut_idx = int(last_valid[-1]) + 1
                    d_par = d_par[:cut_idx, :]
                    d_perp = d_perp[:cut_idx, :]
                    delays = delays[:cut_idx]

            Zavg_R = np.stack([d_par, d_perp], axis=2)
            Zstdv = np.zeros_like(Zavg_R)
            Nscans = 1
            Zss_R = Zavg_R[:, :, :, np.newaxis]
            is_slow_mod = True
            k_slow = 2
            if not state_names:
                state_names = ["Parallel", "Perpendicular"]
        else:
            if "data/delta_A" in f:
                delta_A = np.asarray(f["data/delta_A"][:], dtype=np.float64)
            elif "data/intensity" in f:
                delta_A = np.asarray(f["data/intensity"][:], dtype=np.float64)
            elif indiv_scans is not None:
                delta_A = np.nanmean(indiv_scans, axis=0)
            else:
                raise ValueError(
                    f"File {path.name} is not a valid PyMESS pump-probe dataset (missing primary matrix)."
                )
            valid_rows = ~np.all(np.isnan(delta_A), axis=1)
            if not np.all(valid_rows):
                last_valid = np.where(valid_rows)[0]
                if len(last_valid) > 0:
                    cut_idx = int(last_valid[-1]) + 1
                    delta_A = delta_A[:cut_idx, :]
                    delays = delays[:cut_idx]

            Zavg_R = delta_A[:, :, np.newaxis]

            noise_key = (
                "data/delta_A_noise"
                if "data/delta_A_noise" in f
                else ("data/intensity_noise" if "data/intensity_noise" in f else None)
            )
            if noise_key is not None:
                noise_mat = np.asarray(f[noise_key][:], dtype=np.float64)
                if not np.all(valid_rows):
                    noise_mat = noise_mat[:len(delays), :]
                Zstdv = noise_mat[:, :, np.newaxis]
            else:
                Zstdv = np.zeros_like(Zavg_R)

            if load_single_scans and indiv_scans is not None:
                if not np.all(valid_rows):
                    indiv_scans = indiv_scans[:, :len(delays), :]
                Nscans = int(indiv_scans.shape[0])
                Zss_R = np.transpose(indiv_scans, (1, 2, 0))[:, :, np.newaxis, :]
            else:
                Nscans = 1
                Zss_R = Zavg_R[:, :, :, np.newaxis]

        # 4. Units Dictionary
        Units = hlp.units2dic(unitsL, unitsT, "x1E3")
        Units["cal_source"] = (
            f"PyMESS spectral calibration ({display_L})"
            if unitsL != "px"
            else "Uncalibrated pixel indices"
        )
        if is_slow_mod and k_slow > 1:
            if state_names and len(state_names) == k_slow:
                Units["detectors"] = state_names
            else:
                Units["detectors"] = [f"State_{i}" for i in range(k_slow)]

        Units["pymess_metadata"] = meta_dict
        for g_key in ("g_factor", "G_factor", "gfactor"):
            if g_key in meta_dict:
                try:
                    Units["g_factor"] = float(meta_dict[g_key])
                    break
                except (ValueError, TypeError):
                    pass
        if "sample_id" in meta_dict and meta_dict["sample_id"]:
            Units["sample"] = meta_dict["sample_id"]
        for wl_key in ("pump_wl_nm", "pump_wavelength_nm"):
            if wl_key in meta_dict and meta_dict[wl_key]:
                Units["pump_wl"] = meta_dict[wl_key]
                break

        # 5. Raw Intensity / Accumulation Counts (if present)
        if "data/intensity" in f:
            counts = np.asarray(f["data/intensity"][:], dtype=np.float64)
            if not np.all(valid_rows):
                counts = counts[:len(delays), :]
            Units["counts"] = counts

        # 6. Calibration metadata (B-matrix and dark noise backgrounds)
        if "calibration" in f:
            calib_dict = {}
            for k in f["calibration"].keys():
                calib_dict[k] = np.asarray(f["calibration"][k][:], dtype=np.float64)
            Units["pymess_calibration"] = calib_dict

        # 7. Format a human-readable sample info summary
        info_lines = []
        if "sample_id" in meta_dict and meta_dict["sample_id"]:
            info_lines.append(f"Sample: {meta_dict['sample_id']}")
        if "solvent" in meta_dict and meta_dict["solvent"]:
            info_lines.append(f"Solvent: {meta_dict['solvent']}")
        for wl_key in ("pump_wl_nm", "pump_wavelength_nm"):
            if wl_key in meta_dict and meta_dict[wl_key]:
                info_lines.append(f"Pump Wavelength: {meta_dict[wl_key]} nm")
                break
        if "pump_power_mw" in meta_dict and meta_dict["pump_power_mw"]:
            info_lines.append(f"Pump Power: {meta_dict['pump_power_mw']} mW")
        if "pump_polarization" in meta_dict and meta_dict["pump_polarization"]:
            info_lines.append(f"Pump Polarization: {meta_dict['pump_polarization']}")
        if "probe_polarization" in meta_dict and meta_dict["probe_polarization"]:
            info_lines.append(f"Probe Polarization: {meta_dict['probe_polarization']}")
        if is_slow_mod:
            cadence = meta_dict.get("slow_modulation_cadence", "unknown cadence")
            info_lines.append(f"Slow Modulation: {k_slow} states ({cadence})")
        if "comments" in meta_dict and meta_dict["comments"]:
            info_lines.append(f"Comments: {meta_dict['comments']}")
        if info_lines:
            Units["sample_info"] = "\n".join(info_lines)

        if Nscans > 0:
            Units["scan_ids"] = list(range(Nscans))

    return Zavg_R, delays, probe, Units, Nscans, Zss_R, Zstdv


def load_pymess_anisotropy(
    datafilename: str | Path,
    G_factor: float = 1.0,
    threshold_mOD: float = 0.05,
) -> dict[str, Any]:
    """Load a slow-modulated polarization scan and unpack into Parallel, Perpendicular, Isotropic, and Anisotropy datasets.

    Parameters
    ----------
    datafilename : str or Path
        Path to the PyMESS HDF5 (.h5) file.
    G_factor : float, optional
        Instrumental G-factor (I_par / I_perp sensitivity ratio). Default is 1.0.
    threshold_mOD : float, optional
        Minimum isotropic signal in mOD below which anisotropy is clamped to NaN
        to prevent noise divergence. Default is 0.05 mOD.

    Returns
    -------
    dict of str -> Dataset1D
        Dictionary containing:
        - 'parallel': Dataset1D of Parallel transient absorption
        - 'perpendicular': Dataset1D of Perpendicular transient absorption
        - 'isotropic': Dataset1D of Isotropic transient absorption
        - 'anisotropy': Dataset1D of Time-resolved polarization anisotropy r(t)
    """
    path = Path(datafilename)
    if not path.is_file():
        raise FileNotFoundError(f"PyMESS HDF5 file not found: {datafilename}")

    with h5py.File(path, "r", swmr=True) as f:
        meta = {str(k): _pymess_attr_str(v) for k, v in f.get("metadata", {}).attrs.items()}
        raw_state_names = meta.get("slow_modulation_state_names", "")
        state_names = (
            [s.strip().lower() for s in str(raw_state_names).split(",")] if raw_state_names else []
        )

        par_idx, perp_idx = None, None
        for idx, nm in enumerate(state_names):
            if "par" in nm and par_idx is None:
                par_idx = idx
            elif "perp" in nm and perp_idx is None:
                perp_idx = idx

        # Default fallback to channels 0 and 1 if not explicitly named
        if par_idx is None or perp_idx is None:
            par_idx, perp_idx = 0, 1

        # Check if pre-calculated surfaces exist in data/slow_modulation
        if (
            "data/slow_modulation/delta_A_parallel" in f
            and "data/slow_modulation/delta_A_perpendicular" in f
            and G_factor == 1.0
        ):
            d_par = np.asarray(f["data/slow_modulation/delta_A_parallel"][:], dtype=np.float64)
            d_perp = np.asarray(f["data/slow_modulation/delta_A_perpendicular"][:], dtype=np.float64)
            raw_delays = np.asarray(f["data/delays_fs"][:], dtype=np.float64)
            delays = raw_delays / 1000.0
            probe = (
                np.asarray(f["data/wavelengths"][:], dtype=np.float64)
                if "data/wavelengths" in f
                else np.arange(d_par.shape[1], dtype=np.float64)
            )
            probe_mean = float(np.nanmean(probe)) if probe.size > 0 else 0.0
            if probe_mean > 1200.0:
                unitsL = "cm^{-1}"
            elif probe_mean > 100.0:
                unitsL = "nm"
            else:
                unitsL = "px"

            d_iso = np.asarray(f["data/slow_modulation/isotropic"][:], dtype=np.float64)
            aniso = np.asarray(f["data/slow_modulation/anisotropy"][:], dtype=np.float64)
            if threshold_mOD > 0:
                with np.errstate(invalid="ignore"):
                    aniso = np.where(np.abs(d_iso) >= threshold_mOD, aniso, np.nan)
        else:
            Zavg_R, delays, probe, Units, Nscans, Zss_R, Zstdv = read_PyMESS_PP(str(path))
            if Zavg_R.shape[2] < 2:
                raise ValueError(
                    f"File {path.name} does not contain at least 2 modulation states for anisotropy."
                )

            d_par = Zavg_R[:, :, par_idx]
            d_perp = Zavg_R[:, :, perp_idx]
            unitsL = Units.get("L", Units.get("unitsL_lbl", "nm"))
            if unitsL in ("Wavenumber", "cm-1"):
                unitsL = "cm^{-1}"
            elif unitsL == "Wavelength":
                unitsL = "nm"
            elif unitsL == "Pixel":
                unitsL = "px"

            # Calculate Isotropic: (Par + 2 * G * Perp) / 3
            d_iso = (d_par + 2.0 * G_factor * d_perp) / 3.0

            # Calculate Anisotropy: (Par - G * Perp) / (Par + 2 * G * Perp)
            denom = d_par + 2.0 * G_factor * d_perp
            with np.errstate(divide="ignore", invalid="ignore"):
                aniso = (d_par - G_factor * d_perp) / denom
                if threshold_mOD > 0:
                    aniso = np.where(np.abs(d_iso) >= threshold_mOD, aniso, np.nan)

    from pymorgan.oneD.dataset import Dataset1D

    def _make_ds(data_2d: np.ndarray, name: str, unit_z: str = "x1E3") -> Dataset1D:
        u = hlp.units2dic(unitsL, "ps", unit_z)
        u["cal_source"] = f"PyMESS {name}"
        u["sample"] = str(meta.get("sample_id", "Unknown"))
        z_3d = data_2d[:, :, np.newaxis]
        z_ss = z_3d[:, :, :, np.newaxis]
        return Dataset1D(
            z_3d,
            delays.copy(),
            probe.copy(),
            u,
            1,
            z_ss,
            np.zeros_like(z_3d),
            data_type="PyMESS_PP",
        )

    return {
        "parallel": _make_ds(d_par, "Parallel ΔA (mOD)", "x1E3"),
        "perpendicular": _make_ds(d_perp, "Perpendicular ΔA (mOD)", "x1E3"),
        "isotropic": _make_ds(d_iso, "Isotropic ΔA (mOD)", "x1E3"),
        "anisotropy": _make_ds(aniso, "Polarization Anisotropy r(t)", "dimensionless"),
    }


# --------------------------------------------------------------------------- #
#                             HARPIA reader                                    #
# --------------------------------------------------------------------------- #
register_dataset_glob("HARPIA_TA", "*.dat")


def _is_harpia_main_file(path: Path) -> bool:
    """Whether ``path`` is a HARPIA-TA dataset file (excluding matrix, stats, FluDep, etc.)."""
    if not path.is_file() or path.suffix.lower() != ".dat":
        return False
    stem = path.stem
    if stem.lower().endswith(("_matrix", "_stats")):
        return False
    if re.search(r"(?:^|_)fludep(?:_\d+)?$", stem, re.IGNORECASE):
        return False
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            header_sample = f.read(2048)
        if "Pump-probe" not in header_sample and "Wavelength" not in header_sample:
            return False
        return True
    except Exception:
        return False


@register_dataset_file_filter("HARPIA_TA")
def _harpia_file_filter(path: Path) -> bool:
    """Filter files for dataset listing so only main HARPIA dataset files are shown."""
    return _is_harpia_main_file(Path(path))


@register_loader("HARPIA_TA")
def read_HARPIA(datafilename: str) -> LoaderResult:
    """Read a Light Conversion HARPIA-TA transient absorption data file (.dat).

    The file format consists of headers followed by repeated blocks:
    Line 0: Mode / reference info (e.g., 'Pump-probe, Not referenced, ...')
    Line 1: Background header specification
    Line 2: Measurement header specification
    Line 3: 'Wavelength: <val1>\\t<val2>...' (probe axis in nm)

    Followed by blocks starting with 'Background, scan S, Delay ...' or
    'Measurement, scan S, Delay ...', a 'Pump=...' line, and 4 array lines for:
    1. Sample Pumped (SmplPumped)
    2. Reference Pumped (RefPumped)
    3. Sample Unpumped (SmplUnpumped)
    4. Reference Unpumped (RefUnpumped)

    Returns
    -------
    Zavg_R, delays, probe, Units, Nscans, Zss_R, Zstdv, scan_ids
    """
    path = Path(datafilename)
    if not path.is_file():
        raise FileNotFoundError(f"HARPIA file not found: {datafilename}")

    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 4:
        raise ValueError(f"File {datafilename} does not contain valid HARPIA header lines.")

    header0 = lines[0]
    if "Pump-probe" not in header0 and "Wavelength" not in lines[3]:
        raise ValueError(f"File {datafilename} is not a valid HARPIA-TA data file.")

    is_referenced = "Not referenced" not in header0

    probe_str = lines[3].split(":", 1)[1] if ":" in lines[3] else lines[3]
    probe = np.fromstring(probe_str, sep="\t").astype(float)
    if probe.size == 0:
        raise ValueError(f"Could not parse wavelength probe axis from line 4 in {datafilename}.")
    Npixels = len(probe)

    bcks: dict[int, dict[str, np.ndarray]] = {}
    meas: dict[int, list[dict[str, Any]]] = {}
    seen_delays_per_scan: dict[int, set[float]] = {}
    skipped_blocks = 0

    header_re = re.compile(
        r"^(Background|Measurement)\s*,\s*scan\s*(\d+)\s*,\s*Delay\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
        re.IGNORECASE,
    )

    idx = 4
    while idx < len(lines):
        line = lines[idx]
        m = header_re.match(line)
        if not m:
            idx += 1
            continue

        kind = m.group(1).capitalize()
        scan_num = int(m.group(2))
        try:
            delay_val = float(m.group(3))
        except ValueError:
            skipped_blocks += 1
            idx += 1
            continue

        cur_idx = idx + 1
        pump_val = 0.0
        if cur_idx < len(lines) and lines[cur_idx].startswith("Pump="):
            try:
                pump_val = float(lines[cur_idx].split("=")[1])
            except (IndexError, ValueError):
                pass
            cur_idx += 1

        data_arrays: list[np.ndarray] = []
        valid_block = True
        for _ in range(4):
            if cur_idx >= len(lines):
                valid_block = False
                break
            dline = lines[cur_idx]
            if header_re.match(dline) or dline.startswith("Background") or dline.startswith("Measurement"):
                valid_block = False
                break
            toks = dline.split("\t")
            if len(toks) != Npixels:
                valid_block = False
                cur_idx += 1
                break
            try:
                arr = np.fromstring(dline, sep="\t").astype(float)
                if len(arr) != Npixels or not np.all(np.isfinite(arr)):
                    valid_block = False
                    cur_idx += 1
                    break
                data_arrays.append(arr)
            except Exception:
                valid_block = False
                cur_idx += 1
                break
            cur_idx += 1

        if not valid_block or len(data_arrays) != 4:
            skipped_blocks += 1
            logger.warning(
                "Skipping corrupted %s block for scan %d (around line %d) in HARPIA file '%s'",
                kind,
                scan_num,
                idx,
                path.name,
            )
            idx = cur_idx
            continue

        sp, rp, su, ru = data_arrays
        entry = {
            "delay": delay_val,
            "pump": pump_val,
            "sp": sp,
            "rp": rp,
            "su": su,
            "ru": ru,
        }

        if kind == "Background":
            if scan_num not in bcks:
                bcks[scan_num] = entry
        else:
            delays_set = seen_delays_per_scan.setdefault(scan_num, set())
            already_seen = any(np.isclose(delay_val, d, atol=1e-5, rtol=1e-5) for d in delays_set)
            if already_seen:
                skipped_blocks += 1
                logger.warning(
                    "Skipping duplicate/spliced delay %.4f in scan %d (around line %d) in HARPIA file '%s'",
                    delay_val,
                    scan_num,
                    idx,
                    path.name,
                )
                idx = cur_idx
                continue
            delays_set.add(delay_val)
            meas.setdefault(scan_num, []).append(entry)

        idx = cur_idx

    if skipped_blocks > 0:
        warnings.warn(
            f"HARPIA file '{path.name}': skipped {skipped_blocks} corrupted or duplicate measurement block(s).",
            UserWarning,
            stacklevel=2,
        )

    if not meas:
        raise ValueError(f"No measurement blocks found in HARPIA file {datafilename}.")

    scans = sorted(meas.keys())
    Nscans = len(scans)

    # Master delay grid: unique delays across all scans, sorted ascending
    all_delays = sorted({m["delay"] for s in scans for m in meas[s]})
    delays = np.array(all_delays, dtype=float)
    Ndelays = len(delays)
    unitsT = "ps"

    # Probe wavelength calibration lookup (CalibratedProbe.csv preferred, then line 4 header)
    cal_source = "Embedded HARPIA probe axis"
    cal_level = None
    cal_candidates = [
        (path.parent / "CalibratedProbe.csv", "data folder", "datadir"),
        (path.parent / "calib" / "CalibratedProbe.csv", "calib folder", "datadir"),
        (path.parent.parent / "CalibratedProbe.csv", "root folder", "rootdir"),
        (path.parent.parent / "calib" / "CalibratedProbe.csv", "parent calib folder", "rootdir"),
    ]
    for cand, loc, lvl in cal_candidates:
        if cand.is_file():
            try:
                cal_probe = np.loadtxt(cand, delimiter=",").ravel().astype(float)
                if len(cal_probe) == Npixels:
                    probe = cal_probe
                    cal_source = f"CalibratedProbe.csv ({loc})"
                    cal_level = lvl
                    break
            except Exception:
                pass

    fallback_bck = next(
        iter(bcks.values()),
        {"sp": np.zeros(Npixels), "rp": np.zeros(Npixels), "su": np.zeros(Npixels), "ru": np.zeros(Npixels)},
    )

    Zss_R = np.full((Ndelays, Npixels, 1, Nscans), np.nan, dtype=float)

    for s_idx, scan in enumerate(scans):
        bck = bcks.get(scan, fallback_bck)
        m_list = meas[scan]
        scan_delay_groups: dict[int, list[np.ndarray]] = {}

        for m in m_list:
            b_sp, b_rp, b_su, b_ru = bck["sp"], bck["rp"], bck["su"], bck["ru"]
            m_sp, m_rp, m_su, m_ru = m["sp"], m["rp"], m["su"], m["ru"]

            if is_referenced:
                denom_p = m_rp - b_rp
                denom_u = m_ru - b_ru
                tp = (m_sp - b_sp) / np.where(denom_p == 0, np.nan, denom_p)
                tu = (m_su - b_su) / np.where(denom_u == 0, np.nan, denom_u)
            else:
                tp = m_sp - b_sp
                tu = m_su - b_su

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where((tp > 0) & (tu > 0), tu / tp, np.nan)
                da = 1000.0 * np.log10(ratio)
                da = np.nan_to_num(da, nan=0.0, posinf=0.0, neginf=0.0)

            d_matches = np.where(np.isclose(delays, m["delay"], atol=1e-5, rtol=1e-5))[0]
            if len(d_matches) > 0:
                d_i = int(d_matches[0])
                scan_delay_groups.setdefault(d_i, []).append(da)

        for d_i, da_list in scan_delay_groups.items():
            if len(da_list) == 1:
                Zss_R[d_i, :, 0, s_idx] = da_list[0]
            else:
                Zss_R[d_i, :, 0, s_idx] = np.mean(da_list, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            Zavg_R = np.nanmean(Zss_R, axis=-1)
        if Nscans > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                Zstdv = np.nanstd(Zss_R, axis=-1, ddof=1)
            Zstdv = np.nan_to_num(Zstdv, nan=0.0)
        else:
            Zstdv = np.zeros_like(Zavg_R)

    # Reverse probe & data axes if wavelength decreases with pixel index
    if len(probe) > 1 and probe[0] >= probe[-1]:
        probe = np.flip(probe)
        Zavg_R = np.flip(Zavg_R, axis=1)
        Zstdv = np.flip(Zstdv, axis=1)
        Zss_R = np.flip(Zss_R, axis=1)

    scan_ids = [s - 1 for s in scans]

    Units = hlp.units2dic("nm", unitsT, "x1E3")
    Units["cal_source"] = cal_source
    Units["cal_level"] = cal_level
    if skipped_blocks > 0:
        Units["corrupted_blocks_skipped"] = skipped_blocks
    return Zavg_R, delays, probe, Units, Nscans, Zss_R, Zstdv, scan_ids




# --------------------------------------------------------------------------- #
#                 Per-scan recombination (noise-weighted average)             #
# --------------------------------------------------------------------------- #
def parse_scan_selection(text: str, nmax: int) -> list[int]:
    """Parse a 1-based scan selection string into sorted 0-based indices.

    Accepts space- and/or comma-separated singles and ``a-b`` ranges, e.g.
    ``"1-4 6 9"`` or ``"1-3, 5"``. Out-of-range or malformed tokens raise
    ``ValueError``. Returns the unique indices in ascending order.
    """
    import re

    wanted: set[int] = set()
    for tok in re.split(r"[\s,]+", text.strip()):
        if not tok:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", tok)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            wanted.update(range(a, b + 1))
        elif re.fullmatch(r"\d+", tok):
            wanted.add(int(tok))
        else:
            raise ValueError(f"Invalid token {tok!r}. Use e.g. '1-4 6 9'.")
    if any(v < 1 or v > nmax for v in wanted):
        raise ValueError(f"Scan numbers must be between 1 and {nmax}.")
    return sorted(v - 1 for v in wanted)


def mess_recalc_average(
    path: str,
    scan_indices,
    *,
    spectrum: int = 0,
    slowmod: int = 0,
    noise_weighted: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute the average over selected MESS scans.

    Reads the per-scan signal and per-scan noise from ``<folder>/temp`` and
    combines the selected ``scan_indices`` (0-based). When ``noise_weighted=True``
    (default), combines scans by inverse-variance weighting (weight ``1/noise**2``),
    matching the acquisition's averaged data: selecting every scan reproduces
    the main-directory ``_signal`` / ``_signal_noise`` files. When ``False``,
    computes the direct arithmetic mean and propagates the per-scan noise.
    Returns ``(Zavg_R, Zstdv)`` shaped ``[Ndelays x Npixels x 1]`` and aligned
    to the dataset's (deduplicated, sorted) delay axis.
    """
    folder = Path(path)
    name = folder.name
    scan_indices = [int(k) for k in scan_indices]
    if not scan_indices:
        raise ValueError("No scans selected.")

    delays = np.loadtxt(folder / f"{name}_delays.csv", delimiter=",").ravel().astype(float)
    # The unique/sort mapping is scale-invariant, so the raw delays reproduce the
    # same row alignment used by the reader (which works on ps/ns delays).
    _, reduce = _delay_reducer(delays)

    tempdir = folder / "temp"
    if not tempdir.is_dir():
        alt = folder.parent / f"{name}temp"
        tempdir = alt if alt.is_dir() else tempdir
    if not tempdir.is_dir():
        raise FileNotFoundError(f"No temp/ folder with per-scan data in {folder}.")

    sp, sm = int(spectrum), int(slowmod)
    files = sorted(tempdir.glob(f"{name}_signal_sp{sp}_sm{sm}_du0_*.csv"), key=_scan_index)

    sigs, nois = [], []
    for k in scan_indices:
        if k < 0 or k >= len(files):
            raise IndexError(f"Scan index {k + 1} is out of bounds (found {len(files)} scan files).")
        sf = files[k]
        nf = tempdir / sf.name.replace("signal", "signal_noise")
        if not sf.is_file() or not nf.is_file():
            raise FileNotFoundError(
                f"Missing per-scan files for scan {k + 1} in {tempdir}."
            )
        sigs.append(reduce(_mess_load_2d(sf)))
        nois.append(reduce(_mess_load_2d(nf)))

    S = np.stack(sigs, axis=-1)  # [Ndelays x Npixels x Ksel]
    Nz = np.stack(nois, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        if noise_weighted:
            w = 1.0 / np.where(Nz == 0, np.nan, Nz) ** 2
            wsum = np.nansum(w, axis=-1)
            avg = np.nansum(S * w, axis=-1) / wsum
            comb = np.sqrt(1.0 / wsum)
        else:
            avg = np.nanmean(S, axis=-1)
            k_valid = np.sum(np.isfinite(Nz) & (Nz > 0), axis=-1)
            if np.any(k_valid > 0):
                comb = np.sqrt(np.nansum(np.where(Nz > 0, Nz, 0.0) ** 2, axis=-1)) / np.where(k_valid == 0, 1.0, k_valid)
            elif len(scan_indices) > 1:
                comb = np.nanstd(S, axis=-1, ddof=1)
            else:
                comb = np.zeros_like(avg)
        avg = np.nan_to_num(avg, nan=0.0)
        comb = np.nan_to_num(comb, nan=0.0)
    return avg[..., np.newaxis], comb[..., np.newaxis]


# --------------------------------------------------------------------------- #
#                             Helios TA reader                                #
# --------------------------------------------------------------------------- #
register_dataset_glob("Helios_TA", "*.csv")


def _is_helios_main_file(path: Path) -> bool:
    """Whether ``path`` is an averaged Helios dataset file (excluding single scans and aux files)."""
    if not path.is_file() or not path.name.lower().endswith(".csv"):
        return False
    stem = path.stem
    if re.search(r"_scan\d+$", stem, re.IGNORECASE):
        return False
    if re.search(r"(-bg|-chirp|-CORRECTED)$", stem, re.IGNORECASE):
        return False
    return True


@register_dataset_file_filter("Helios_TA")
def _helios_file_filter(path: Path) -> bool:
    """Filter files for dataset listing so only averaged dataset files are shown."""
    return _is_helios_main_file(Path(path))


def _find_helios_main_csv(folder: Path, dataset_name: str | None = None) -> Path:
    """Find the main averaged dataset CSV file in ``folder``."""
    folder = Path(folder)
    if dataset_name:
        cand = folder / dataset_name
        if not cand.suffix:
            cand = cand.with_suffix(".csv")
        if cand.is_file():
            return cand

    default_cand = folder / f"{folder.name}.csv"
    if default_cand.is_file() and _is_helios_main_file(default_cand):
        return default_cand

    main_files = sorted(
        [f for f in folder.glob("*.csv") if _is_helios_main_file(f)],
        key=lambda f: f.name.lower(),
    )
    if main_files:
        return main_files[0]

    raise FileNotFoundError(f"No valid Helios TA averaged dataset (.csv) found in {folder}.")


def parse_helios_csv(file_path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a single Helios CSV file into delays, probe, and signal (in mOD, NaN-interpolated)."""
    import pandas as pd

    df = pd.read_csv(file_path, header=None, on_bad_lines="skip", engine="c")
    raw_arr = df.to_numpy()
    num_arr = np.asarray(pd.to_numeric(raw_arr.ravel(), errors="coerce")).reshape(raw_arr.shape)
    num_df = pd.DataFrame(num_arr)
    if num_df.empty or num_df.shape[0] < 2 or num_df.shape[1] < 2:
        raise ValueError(f"Invalid Helios TA data file format: {file_path}")

    delays = num_df.iloc[0, 1:].dropna().to_numpy(dtype=float)
    valid_rows = num_df.iloc[1:, :].dropna(subset=[0])
    probe = valid_rows.iloc[:, 0].to_numpy(dtype=float)
    sig = valid_rows.iloc[:, 1 : 1 + len(delays)].to_numpy(dtype=float)

    if sig.shape != (len(probe), len(delays)):
        raise ValueError(f"Helios TA signal matrix dimensions mismatch in {file_path}")

    # Convert Delta-A to mOD (x 1000) and transpose to [Ndelays x Npixels]
    sig_mod = sig.T * 1000.0

    if np.isnan(sig_mod).any():
        sig_df = pd.DataFrame(sig_mod)
        sig_mod = (
            sig_df.interpolate(method="linear", limit_direction="both", axis=0)
            .interpolate(method="linear", limit_direction="both", axis=1)
            .to_numpy(dtype=float)
        )

    return delays, probe, sig_mod


@register_loader("Helios_TA")
def read_Helios_TA(datafilename: str, dataset_name: str | None = None) -> LoaderResult:
    """Read Helios transient absorption dataset (averaged file and single scans)."""
    path = Path(datafilename)
    if path.is_dir():
        main_file = _find_helios_main_csv(path, dataset_name)
        folder = path
    elif path.is_file():
        main_file = path
        folder = path.parent
    else:
        raise FileNotFoundError(f"Helios dataset path not found: {datafilename}")

    main_stem = main_file.stem
    delays, probe, Zavg_2d = parse_helios_csv(main_file)

    scan_pattern = re.compile(rf"^{re.escape(main_stem)}_scan(\d+)\.csv$", re.IGNORECASE)
    scan_entries: list[tuple[int, Path]] = []
    for f in folder.iterdir():
        if f.is_file():
            m = scan_pattern.match(f.name)
            if m:
                scan_entries.append((int(m.group(1)), f))
    scan_entries.sort(key=lambda x: x[0])

    Zavg_R = Zavg_2d[..., np.newaxis]  # [Ndelays x Npixels x 1]

    if scan_entries:
        Nscans = len(scan_entries)
        scan_signals = []
        scan_ids = []
        for scan_num, scan_file in scan_entries:
            _, _, s_sig = parse_helios_csv(scan_file)
            scan_signals.append(s_sig)
            scan_ids.append(scan_num)

        Zss_3d = np.stack(scan_signals, axis=-1)  # [Ndelays x Npixels x Nscans]
        Zss_R = Zss_3d[:, :, np.newaxis, :]  # [Ndelays x Npixels x 1 x Nscans]

        if Nscans > 1:
            Zstdv_2d = np.nanstd(Zss_3d, axis=-1, ddof=1)
        else:
            Zstdv_2d = np.zeros_like(Zavg_2d)
        Zstdv = Zstdv_2d[..., np.newaxis]
    else:
        Nscans = np.nan
        Zss_R = np.full((*Zavg_R.shape, 1), np.nan)
        Zstdv = np.zeros_like(Zavg_R)
        scan_ids = None

    Units = hlp.units2dic("nm", "ps", "x1E3")
    Units["cal_source"] = "Dataset probe axis (CSV header)"
    return Zavg_R, delays, probe, Units, Nscans, Zss_R, Zstdv, scan_ids


@register_dataset_detector("Helios_TA")
def _helios_is_dataset(folder: Path) -> bool:
    """Whether ``folder`` is a Helios TA dataset directory."""
    try:
        _find_helios_main_csv(folder)
        return True
    except FileNotFoundError:
        return False


# --------------------------------------------------------------------------- #
#                            Exported TXT TA reader                           #
# --------------------------------------------------------------------------- #
def _find_exported_txt_files(path: Path | str) -> tuple[Path, Path, Path]:
    """Locate (signal_file, time_file, probe_file) for an Exported TXT dataset."""
    p = Path(path)
    if p.is_file():
        folder = p.parent
    elif p.is_dir():
        folder = p
    else:
        raise FileNotFoundError(f"Exported TXT dataset path not found: {path}")

    try:
        files = {f.name.lower(): f for f in folder.iterdir() if f.is_file()}
    except OSError:
        raise FileNotFoundError(f"Cannot access directory: {folder}") from None

    sig_file = None
    for cand in ["ta.txt", "signal.txt"]:
        if cand in files:
            sig_file = files[cand]
            break

    time_file = None
    for cand in ["time.txt", "delays.txt", "delay.txt", "times.txt"]:
        if cand in files:
            time_file = files[cand]
            break

    probe_file = None
    for cand in ["wavelength.txt", "wavelengths.txt", "wavenumber.txt", "wavenumbers.txt", "probe.txt"]:
        if cand in files:
            probe_file = files[cand]
            break

    if not sig_file or not time_file or not probe_file:
        missing = []
        if not sig_file:
            missing.append("TA.txt")
        if not time_file:
            missing.append("time.txt")
        if not probe_file:
            missing.append("wavelength.txt/wavenumber.txt")
        raise FileNotFoundError(
            f"Missing required Exported TXT file(s) in {folder}: {', '.join(missing)}"
        )

    return sig_file, time_file, probe_file


@register_loader("Exported_TXT")
def read_Exported_TXT(datafilename: str) -> LoaderResult:
    """Read an Exported TXT transient absorption dataset (TA.txt, time.txt, wavelength.txt)."""
    sig_file, time_file, probe_file = _find_exported_txt_files(datafilename)

    def _parse_unit(filepath: Path, default: str) -> str:
        header_line = ""
        try:
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line_s = line.strip()
                    if line_s.startswith("#"):
                        header_line = line_s
                        break
                    elif line_s:
                        break
        except OSError:
            pass

        unit = default
        if header_line and "/" in header_line:
            parts = header_line.split("/")
            raw_unit = parts[-1].strip().lower()
            if "ps" in raw_unit:
                unit = "ps"
            elif "ns" in raw_unit:
                unit = "ns"
            elif "fs" in raw_unit:
                unit = "fs"
            elif "cm-1" in raw_unit or "cm^-1" in raw_unit:
                unit = "cm^{-1}"
            elif "nm" in raw_unit:
                unit = "nm"
            elif "ev" in raw_unit:
                unit = "eV"

        return unit

    units_T = _parse_unit(time_file, "ps")
    default_L = "cm^{-1}" if "wavenumber" in probe_file.name.lower() else "nm"
    units_L = _parse_unit(probe_file, default_L)

    delays = np.loadtxt(time_file)
    probe = np.loadtxt(probe_file)
    Zavg_2d = np.loadtxt(sig_file)

    if delays.ndim != 1:
        delays = np.squeeze(delays)
    if probe.ndim != 1:
        probe = np.squeeze(probe)

    n_delays = len(delays)
    n_probe = len(probe)

    if Zavg_2d.shape == (n_probe, n_delays):
        Zavg_2d = Zavg_2d.T
    elif Zavg_2d.shape != (n_delays, n_probe):
        raise ValueError(
            f"Shape mismatch in Exported TXT dataset: TA matrix shape {Zavg_2d.shape} "
            f"does not match delays length ({n_delays}) and probe length ({n_probe})."
        )

    Zavg_R = Zavg_2d[..., np.newaxis]  # [Ndelays x Npixels x 1]

    Nscans = np.nan
    Zss_R = np.full((*Zavg_R.shape, 1), np.nan)
    Zstdv = np.zeros_like(Zavg_R)

    Units = hlp.units2dic(units_L, units_T, "x1E3")
    Units["cal_source"] = f"Exported text file ({probe_file.name})"
    return Zavg_R, delays, probe, Units, Nscans, Zss_R, Zstdv


@register_dataset_detector("Exported_TXT")
def _exported_txt_is_dataset(folder: Path) -> bool:
    """Whether ``folder`` is an Exported TXT dataset directory."""
    try:
        _find_exported_txt_files(folder)
        return True
    except (FileNotFoundError, OSError):
        return False



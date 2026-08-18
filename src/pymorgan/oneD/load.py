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

from collections.abc import Callable
from os import PathLike
from pathlib import Path
import re
from typing import Any

import numpy as np

from pymorgan import helpers as hlp

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
    return Zavg_R, delays, probe, Units, Nscans, Zss_R, Zstdv


def _stub_loader(name: str) -> Loader:
    """Build a placeholder loader that fails loudly until implemented."""

    def loader(path: str) -> LoaderResult:
        raise NotImplementedError(
            f"The {name!r} 1-D loader is not implemented yet (requested file: {path}). "
            f"Implement it here and register it via @register_loader({name!r})."
        )

    loader.__name__ = f"load_{name}"
    return loader


# Instrument-specific formats are recognised but not yet implemented.
for _name in (
    "UniGE_FLUPSold",
    "UniGE_FLUPSnew",
):
    register_loader(_name)(_stub_loader(_name))


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
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
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

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().replace("\x00", "")

    lines = text.splitlines()
    data_lines = [l for l in lines if not l.startswith("%") and l.strip()]
    if not data_lines:
        raise ValueError(f"No data rows found in UniGE nsTA file: {datafilename}")

    parsed_rows = [[float(x) for x in l.split()] for l in data_lines]
    col_counts = Counter(len(r) for r in parsed_rows)
    expected_cols = col_counts.most_common(1)[0][0]

    valid_rows = [r for r in parsed_rows if len(r) == expected_cols]
    if not valid_rows:
        raise ValueError(f"Could not parse valid data rows from UniGE nsTA file: {datafilename}")

    alldata = np.array(valid_rows, dtype=float)

    # Delays in seconds -> convert to nanoseconds (x 1e9)
    delays_ns = alldata[:, 0] * 1e9
    delays, u_idx, dID = np.unique(delays_ns, return_index=True, return_inverse=True)
    Ndelays = len(delays)

    total_rows = alldata.shape[0]
    Npixels = total_rows // Ndelays

    tmpsignal = np.zeros((Ndelays, Npixels), dtype=float)
    tmprms = np.zeros((Ndelays, Npixels), dtype=float)
    tmpnoise = np.zeros((Ndelays, Npixels), dtype=float)
    cts = np.zeros(Ndelays, dtype=float)

    for j in range(Ndelays):
        idx_pix = slice(j * Npixels, (j + 1) * Npixels)
        cts[j] = alldata[j * Npixels, 5]
        tmpsignal[j, :] = alldata[idx_pix, 2]
        tmprms[j, :] = alldata[idx_pix, 3]
        tmpnoise[j, :] = alldata[idx_pix, 4]

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
    csv_candidates = [
        path.parent / "CalibratedProbe.csv",
        path.parent / "calib" / "CalibratedProbe.csv",
        path.parent.parent / "CalibratedProbe.csv",
        path.parent.parent / "calib" / "CalibratedProbe.csv",
    ]
    for cand in csv_candidates:
        if cand.is_file():
            try:
                probe = np.loadtxt(cand, delimiter=",").ravel().astype(float)
                break
            except Exception:
                pass

    if probe is None or len(probe) != Npixels:
        mat_candidates = [
            path.parent / "pix2lam.mat",
            path.parent / "calib" / "pix2lam.mat",
            path.parent.parent / "pix2lam.mat",
            path.parent.parent / "calib" / "pix2lam.mat",
        ]
        for cand in mat_candidates:
            if cand.is_file():
                try:
                    mat = sio.loadmat(cand)
                    if "lam" in mat:
                        probe = mat["lam"].ravel().astype(float)
                        break
                except Exception:
                    pass

    if probe is None or len(probe) != Npixels:
        probe = np.arange(1, Npixels + 1, dtype=float)

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
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
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

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().replace("\x00", "")

    lines = text.splitlines()
    data_lines = [l for l in lines if not l.startswith("%") and l.strip()]
    if not data_lines:
        raise ValueError(f"No data rows found in UniGE fsTA file: {datafilename}")

    parsed_rows = [[float(x) for x in l.split()] for l in data_lines]
    col_counts = Counter(len(r) for r in parsed_rows)
    expected_cols = col_counts.most_common(1)[0][0]

    valid_rows = [r for r in parsed_rows if len(r) == expected_cols]
    if not valid_rows:
        raise ValueError(f"Could not parse valid data rows from UniGE fsTA file: {datafilename}")

    alldata = np.array(valid_rows, dtype=float)

    # Delays in seconds -> convert to picoseconds (x 1e12)
    delays_ps = alldata[:, 0] * 1e12
    delays, u_idx, dID = np.unique(delays_ps, return_index=True, return_inverse=True)
    Ndelays = len(delays)

    Ncols = alldata.shape[1]
    Npixels = (Ncols - 2) // 2

    # Probe wavelength calibration lookup (CalibratedProbe.csv preferred, then pix2lam.mat)
    probe = None
    csv_candidates = [
        path.parent / "CalibratedProbe.csv",
        path.parent / "calib" / "CalibratedProbe.csv",
        path.parent.parent / "CalibratedProbe.csv",
        path.parent.parent / "calib" / "CalibratedProbe.csv",
    ]
    for cand in csv_candidates:
        if cand.is_file():
            try:
                probe = np.loadtxt(cand, delimiter=",").ravel().astype(float)
                break
            except Exception:
                pass

    if probe is None or len(probe) != Npixels:
        mat_candidates = [
            path.parent / "pix2lam.mat",
            path.parent / "calib" / "pix2lam.mat",
            path.parent.parent / "pix2lam.mat",
            path.parent.parent / "calib" / "pix2lam.mat",
        ]
        for cand in mat_candidates:
            if cand.is_file():
                try:
                    mat = sio.loadmat(cand)
                    if "lam" in mat:
                        probe = mat["lam"].ravel().astype(float)
                        break
                except Exception:
                    pass

    if probe is None or len(probe) != Npixels:
        probe = np.arange(1, Npixels + 1, dtype=float)

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
    if not cal_file.is_file():
        cal_file = folder.parent / "CalibratedProbe.csv"

    probe = None
    if cal_file.is_file():
        try:
            tmp_probe = np.loadtxt(cal_file, delimiter=",").ravel().astype(float)
            if n_detectors == 2 and len(tmp_probe) >= 192:
                probe = np.column_stack([tmp_probe[:96], tmp_probe[96:192]])
            else:
                probe = tmp_probe[:pixels_per_det]
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
        else:
            pix = np.arange(1, pixels_per_det + 1, dtype=float)
            probe = np.column_stack([pix] * n_detectors) if n_detectors > 1 else pix

    Units = hlp.units2dic("cm^{-1}", units_T, "x1E3")
    return Zavg_R, delays, probe, Units, Nscans_val, Zss_R, Zstdv, scan_ids


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
    cal_path, _cal_level = _mess_probe_calibration(folder)
    if cal_path is not None:
        probe = np.loadtxt(cal_path, delimiter=",").ravel().astype(float)
    else:
        probe = (
            np.loadtxt(folder / f"{name}_{probe_suffix}.csv", delimiter=",").ravel().astype(float)
        )

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
#                             HARPIA reader                                    #
# --------------------------------------------------------------------------- #
register_dataset_glob("HARPIA_TA", "*.dat")


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

    bcks: dict[int, dict[str, np.ndarray]] = {}
    meas: dict[int, list[dict[str, Any]]] = {}

    idx = 4
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("Background") or line.startswith("Measurement"):
            kind, scan_str, delay_str = [p.strip() for p in line.split(",")]
            scan_num = int(scan_str.replace("scan", "").strip())
            delay_val = float(delay_str.split()[1])

            pump_val = 0.0
            if idx + 1 < len(lines) and lines[idx + 1].startswith("Pump="):
                try:
                    pump_val = float(lines[idx + 1].split("=")[1])
                except (IndexError, ValueError):
                    pass
                idx_data = idx + 2
            else:
                idx_data = idx + 1

            if idx_data + 3 >= len(lines):
                break

            sp = np.fromstring(lines[idx_data], sep="\t").astype(float)
            rp = np.fromstring(lines[idx_data + 1], sep="\t").astype(float)
            su = np.fromstring(lines[idx_data + 2], sep="\t").astype(float)
            ru = np.fromstring(lines[idx_data + 3], sep="\t").astype(float)

            entry = {
                "delay": delay_val,
                "pump": pump_val,
                "sp": sp,
                "rp": rp,
                "su": su,
                "ru": ru,
            }

            if kind == "Background":
                bcks[scan_num] = entry
            else:
                meas.setdefault(scan_num, []).append(entry)

            idx = idx_data + 4
        else:
            idx += 1

    if not meas:
        raise ValueError(f"No measurement blocks found in HARPIA file {datafilename}.")

    scans = sorted(meas.keys())
    Nscans = len(scans)

    first_meas = meas[scans[0]]
    raw_delays = np.array([m["delay"] for m in first_meas], dtype=float)

    unitsT = "ps"
    delays = raw_delays

    fallback_bck = next(iter(bcks.values()), {"sp": 0.0, "rp": 0.0, "su": 0.0, "ru": 0.0})

    scan_signals = []
    for scan in scans:
        bck = bcks.get(scan, fallback_bck)
        m_list = meas[scan]
        scan_da = []
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

            scan_da.append(da)
        scan_signals.append(np.array(scan_da))

    Zss_2d = np.stack(scan_signals, axis=-1)
    Zss_R = Zss_2d[:, :, np.newaxis, :]

    Zavg_R = np.mean(Zss_R, axis=-1)
    if Nscans > 1:
        Zstdv = np.std(Zss_R, axis=-1, ddof=1)
    else:
        Zstdv = np.zeros_like(Zavg_R)

    scan_ids = [s - 1 for s in scans]

    Units = hlp.units2dic("nm", unitsT, "x1E3")
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
    path: str, scan_indices, *, spectrum: int = 0, slowmod: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute the noise-weighted average over selected MESS scans.

    Reads the per-scan signal and per-scan noise from ``<folder>/temp`` and
    combines the selected ``scan_indices`` (0-based) by inverse-variance
    weighting (weight ``1/noise**2``), matching the acquisition's averaged data:
    selecting every scan reproduces the main-directory ``_signal`` /
    ``_signal_noise`` files. Returns ``(Zavg_R, Zstdv)`` shaped
    ``[Ndelays x Npixels x 1]`` and aligned to the dataset's (deduplicated,
    sorted) delay axis.
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
        w = 1.0 / np.where(Nz == 0, np.nan, Nz) ** 2
        wsum = np.nansum(w, axis=-1)
        avg = np.nansum(S * w, axis=-1) / wsum
        comb = np.sqrt(1.0 / wsum)
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

    df = pd.read_csv(file_path, header=None, on_bad_lines="skip")
    num_df = df.apply(pd.to_numeric, errors="coerce")
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
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
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
    return Zavg_R, delays, probe, Units, Nscans, Zss_R, Zstdv


@register_dataset_detector("Exported_TXT")
def _exported_txt_is_dataset(folder: Path) -> bool:
    """Whether ``folder`` is an Exported TXT dataset directory."""
    try:
        _find_exported_txt_files(folder)
        return True
    except (FileNotFoundError, OSError):
        return False



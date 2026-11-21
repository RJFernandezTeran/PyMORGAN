"""Universal format auto-detection and dispatching for PyMORGAN."""

from __future__ import annotations

import re
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Literal

from .log import get_logger

logger = get_logger(__name__)

_XY_DELIM_RE = re.compile(r"[\s,;]+")


@dataclass(frozen=True)
class FormatMatch:
    """Result of probing a file or directory for its PyMORGAN dataset format."""

    dimension: Literal["1D", "2D", "steady_state", "project", "recipe"]
    data_type: str
    confidence: float
    reason: str
    resolved_path: Path


def _is_helios_csv_content(path: Path) -> bool:
    """Fast header check distinguishing a 2D-grid Helios TA CSV from 1D calibration CSVs."""
    from .oneD.load import _is_helios_main_file

    if not _is_helios_main_file(path):
        return False
    name_low = path.name.lower()
    if "avg diff" in name_low or name_low.endswith("_2dir.csv"):
        return False
    try:
        rows: list[list[float]] = []
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for _ in range(12):
                line = fh.readline()
                if not line:
                    break
                s = line.strip()
                if not s:
                    continue
                parts = [tok.strip() for tok in s.split(",")]
                if len(parts) < 3:
                    continue
                if not rows:
                    # Row 0: corner may be empty, '0', '0.0', or text; remaining cols must be delays
                    tail = parts[1:]
                    try:
                        vals = [float(x) for x in tail if x != ""]
                    except ValueError:
                        return False
                    if len(vals) >= 2:
                        rows.append(vals)
                else:
                    try:
                        vals = [
                            float(x) if x.lower() not in ("", "nan", "inf", "-inf") else 0.0
                            for x in parts[: min(len(parts), len(rows[0]) + 1)]
                        ]
                    except ValueError:
                        continue
                    if len(vals) >= 3:
                        rows.append(vals)
                        if len(rows) >= 3:
                            return True
    except OSError:
        return False
    return False


def _is_two_col_xy_text(path: Path) -> bool:
    """Return True if ``path`` looks like a two-column X-Y steady-state text spectrum."""
    name_low = path.name.lower()
    if name_low in ("calibratedprobe.csv", "calibratedpump.csv", "w0.csv"):
        return False
    numeric_rows = 0
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for _ in range(60):
                line = fh.readline()
                if not line:
                    break
                s = line.strip()
                if not s or s.startswith(("#", "%")):
                    continue
                parts = [tok for tok in _XY_DELIM_RE.split(s) if tok]
                try:
                    vals = [float(tok) for tok in parts]
                except ValueError:
                    if numeric_rows > 0:
                        break
                    continue
                if len(vals) != 2:
                    return False
                numeric_rows += 1
                if numeric_rows >= 3:
                    return True
    except OSError:
        return False
    return False


def _is_pdat_file(path: Path) -> bool:
    if path.suffix.lower() != ".pdat" or not path.is_file():
        return False
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            line1 = fh.readline().strip()
            line2 = fh.readline().strip()
        if "*" not in line1:
            return False
        parts = [float(x) for x in line2.split(",") if x.strip()]
        return len(parts) >= 2
    except Exception:
        return False


def _is_p2dat_file(path: Path) -> bool:
    if path.suffix.lower() != ".p2dat" or not path.is_file():
        return False
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            line1 = fh.readline().strip()
            line2 = fh.readline().strip()
        row1 = [float(x) for x in line1.split(",") if x.strip()]
        row2 = [float(x) for x in line2.split(",") if x.strip()]
        return len(row1) >= 3 and len(row2) >= 3 and abs(row1[0]) < 1e-6 and abs(row1[1]) < 1e-6
    except Exception:
        return False


def _detect_hdf5_like(path: Path) -> list[FormatMatch]:
    import h5py

    from .oneD.interop import _glotaran_nc_is_file
    from .oneD.load import classify_pymess_h5

    matches: list[FormatMatch] = []
    try:
        if not h5py.is_hdf5(path):
            return matches
        with h5py.File(path, "r") as f:
            fmt = f.attrs.get("pymorgan_format", "")
            if isinstance(fmt, bytes):
                fmt = fmt.decode("utf-8", errors="replace")
            if fmt == "2D" or ("pump" in f and "probe" in f and "delays" in f and "Z_R" in f):
                matches.append(FormatMatch("2D", "HDF5_2D", 0.99, "PyMORGAN 2D HDF5 archive", path))
            elif fmt == "1D" or ("delays" in f and "probe" in f and "Zavg_R" in f):
                matches.append(FormatMatch("1D", "HDF5", 0.99, "PyMORGAN 1D HDF5 archive", path))
    except Exception:
        return matches

    if not matches:
        subtype = classify_pymess_h5(path)
        if subtype in ("pump_probe", "shaper_t2_scan"):
            matches.append(FormatMatch("1D", "PyMESS_PP", 0.98, f"PyMESS HDF5 ({subtype})", path))
        elif subtype in ("2d_time_domain", "2d_freq_domain"):
            matches.append(
                FormatMatch("2D", "PyMESS_2D", 0.98, f"PyMESS 2D HDF5 ({subtype})", path)
            )

    if not matches and (path.suffix.lower() == ".nc" or _glotaran_nc_is_file(path)):
        try:
            with h5py.File(path, "r") as f:
                keys = set(f.keys())
                has_coords = any(
                    k in keys for k in ("time", "delay", "delays", "spectral", "wavelength")
                )
                has_data = any(
                    k in keys for k in ("data", "__xarray_dataarray_variable__", "dataset")
                )
                if has_data and (path.suffix.lower() == ".nc" or has_coords):
                    conf = 0.97 if path.suffix.lower() == ".nc" else 0.88
                    matches.append(
                        FormatMatch(
                            "1D",
                            "Glotaran_NC",
                            conf,
                            "Glotaran / CF-1.8 NetCDF4 dataset",
                            path,
                        )
                    )
        except Exception:
            pass
    return matches


def _detect_directory(folder: Path) -> list[FormatMatch]:
    from .oneD.load import (
        _find_helios_main_csv,
        _helios_ufs_is_dataset,
        classify_quickcontrol_info,
        is_dataset_dir,
    )
    from .twoD.load import is_map_dataset_dir

    matches: list[FormatMatch] = []

    if is_map_dataset_dir("MESS_2DIR", folder):
        matches.append(FormatMatch("2D", "MESS_2DIR", 0.98, "MESS 2D-IR directory", folder))
    if is_map_dataset_dir("UoS_2DIR", folder):
        matches.append(FormatMatch("2D", "UoS_2DIR", 0.97, "UoS 2D-IR directory", folder))
    if is_map_dataset_dir("RAL_Proc", folder):
        matches.append(FormatMatch("2D", "RAL_Proc", 0.97, "RAL processed 2D-IR directory", folder))
    if is_map_dataset_dir("RAL_RAW", folder):
        matches.append(FormatMatch("2D", "RAL_RAW", 0.96, "RAL raw 2D-IR directory", folder))

    if is_dataset_dir("MESS_TRIR", folder):
        matches.append(FormatMatch("1D", "MESS_TRIR", 0.95, "MESS TRIR directory", folder))
    if is_dataset_dir("MESS_TRUVIS", folder):
        matches.append(FormatMatch("1D", "MESS_TRUVIS", 0.95, "MESS TRUVIS directory", folder))
    if is_dataset_dir("UoS_IRpp", folder):
        matches.append(FormatMatch("1D", "UoS_IRpp", 0.95, "UoS IR pump-probe directory", folder))
    if is_dataset_dir("Exported_TXT", folder):
        matches.append(FormatMatch("1D", "Exported_TXT", 0.90, "Exported TXT TA directory", folder))

    # QuickControl folder containing a single/named .info dataset
    try:
        info_files = sorted(
            (f for f in folder.iterdir() if f.is_file() and f.suffix.lower() == ".info"),
            key=lambda p: p.name.lower(),
        )
    except OSError:
        info_files = []
    if info_files:
        primary_info = next((f for f in info_files if f.stem == folder.name), None)
        cands = [primary_info] if primary_info is not None else info_files
        conf = 0.92 if (primary_info is not None or len(info_files) == 1) else 0.90
        for inf in cands:
            kind, has_scans = classify_quickcontrol_info(inf)
            if has_scans and kind == "2d":
                matches.append(
                    FormatMatch("2D", "QuickControl_2D", conf, "QuickControl 2D directory", inf)
                )
            elif has_scans and kind == "1d":
                matches.append(
                    FormatMatch("1D", "QuickControl_PP", conf, "QuickControl 1D directory", inf)
                )

    try:
        main_csv = _find_helios_main_csv(folder)
        if _is_helios_csv_content(main_csv):
            matches.append(FormatMatch("1D", "Helios_TA", 0.92, "Helios TA CSV directory", folder))
    except FileNotFoundError:
        pass

    if _helios_ufs_is_dataset(folder):
        matches.append(
            FormatMatch("1D", "Helios_UFS", 0.91, "Surface Xplorer UFS directory", folder)
        )

    if is_dataset_dir("RAL_IRpp", folder):
        matches.append(FormatMatch("1D", "RAL_IRpp", 0.85, "RAL IR pump-probe directory", folder))

    return matches


def _detect_file(path: Path) -> list[FormatMatch]:
    from .oneD.interop import _glotaran_ascii_is_file
    from .oneD.load import (
        _has_valid_ufs_header,
        _is_harpia_main_file,
        _ral_irpp_is_file,
        _unige_fsta_is_file,
        _unige_nsta_is_file,
        classify_quickcontrol_info,
        is_dataset_dir,
    )
    from .twoD.load import (
        _RAL_PROC_FILE_RE,
        _ral_header_has_encoded_t1,
        _ral_proc_is_dataset,
        _uos_2dir_is_dataset,
        is_map_dataset_dir,
    )

    name_low = path.name.lower()
    suffix = path.suffix.lower()
    matches: list[FormatMatch] = []

    if suffix == ".pmproj":
        return [FormatMatch("project", "PMPROJ", 1.0, "PyMORGAN project file (.pmproj)", path)]
    if name_low.endswith(".pmrecipe.toml"):
        return [
            FormatMatch("recipe", "PMRECIPE", 1.0, "PyMORGAN recipe file (.pmrecipe.toml)", path)
        ]

    if suffix == ".pdat":
        if _is_pdat_file(path):
            matches.append(FormatMatch("1D", "PDAT", 0.99, "PyMORGAN 1D PDAT file", path))
    elif suffix == ".pdatn":
        sibling = path.with_suffix(".pdat")
        if _is_pdat_file(sibling):
            matches.append(FormatMatch("1D", "PDAT", 0.95, "PDAT noise sibling (.pdatn)", sibling))
    elif suffix == ".p2dat":
        if _is_p2dat_file(path):
            matches.append(FormatMatch("2D", "P2DAT", 0.99, "PyMORGAN 2D P2DAT file", path))
    elif suffix == ".ufs":
        if _has_valid_ufs_header(path):
            if re.search(r"_scan\d+$", path.stem, re.IGNORECASE):
                sibling_stem = re.sub(r"_scan\d+$", "", path.stem, flags=re.IGNORECASE)
                sibling = path.with_name(f"{sibling_stem}{path.suffix}")
                if sibling.is_file() and _has_valid_ufs_header(sibling):
                    matches.append(
                        FormatMatch(
                            "1D",
                            "Helios_UFS",
                            0.95,
                            "Surface Xplorer scan sibling (.ufs)",
                            sibling,
                        )
                    )
                else:
                    matches.append(
                        FormatMatch("1D", "Helios_UFS", 0.99, "Surface Xplorer binary (.ufs)", path)
                    )
            else:
                matches.append(
                    FormatMatch("1D", "Helios_UFS", 0.99, "Surface Xplorer binary (.ufs)", path)
                )
    elif suffix == ".ascii":
        if _glotaran_ascii_is_file(path):
            matches.append(
                FormatMatch("1D", "Glotaran_ASCII", 0.98, "Glotaran / TIMP ASCII file", path)
            )
    elif suffix in (".h5", ".hdf5", ".nc"):
        matches.extend(_detect_hdf5_like(path))
    elif suffix == ".info":
        kind, has_scans = classify_quickcontrol_info(path)
        if has_scans and kind == "2d":
            matches.append(
                FormatMatch(
                    "2D", "QuickControl_2D", 0.98, "PhaseTech QuickControl 2D (.info)", path
                )
            )
        elif has_scans and kind == "1d":
            matches.append(
                FormatMatch(
                    "1D", "QuickControl_PP", 0.98, "PhaseTech QuickControl 1D (.info)", path
                )
            )
    elif suffix == ".dat":
        if _is_harpia_main_file(path):
            matches.append(
                FormatMatch("1D", "HARPIA_TA", 0.96, "Light Conversion HARPIA-TA (.dat)", path)
            )
        if _unige_fsta_is_file(path):
            matches.append(FormatMatch("1D", "UniGE_fsTA", 0.96, "UniGE fsTA (.dat)", path))
        if _unige_nsta_is_file(path):
            matches.append(FormatMatch("1D", "UniGE_nsTA", 0.96, "UniGE nsTA (.dat)", path))
        if not matches and _is_two_col_xy_text(path):
            matches.append(
                FormatMatch(
                    "steady_state", "txt", 0.55, "Two-column steady-state spectrum (.dat)", path
                )
            )
    elif suffix == ".csv":
        if _RAL_PROC_FILE_RE.search(path.name) and _ral_proc_is_dataset(path.parent):
            matches.append(
                FormatMatch("2D", "RAL_Proc", 0.96, "RAL processed 2D-IR delay slice", path.parent)
            )
        elif (
            "avgdiff" in name_low
            and "avg diff spec" not in name_low
            and ((path.parent / "w0.csv").is_file() or _ral_header_has_encoded_t1(path))
        ):
            matches.append(
                FormatMatch("2D", "RAL_RAW", 0.96, "RAL raw 2D-IR AvgDiff CSV", path.parent)
            )
        elif _ral_irpp_is_file(path) and (
            "avg diff spec" in name_low or is_dataset_dir("RAL_IRpp", path.parent)
        ):
            matches.append(FormatMatch("1D", "RAL_IRpp", 0.95, "RAL IR pump-probe CSV", path))
        elif is_map_dataset_dir("MESS_2DIR", path.parent) and name_low.startswith(
            path.parent.name.lower() + "_"
        ):
            matches.append(
                FormatMatch("2D", "MESS_2DIR", 0.90, "MESS 2D-IR component CSV", path.parent)
            )
        elif is_dataset_dir("MESS_TRIR", path.parent) and name_low.startswith(
            path.parent.name.lower() + "_"
        ):
            matches.append(
                FormatMatch("1D", "MESS_TRIR", 0.90, "MESS TRIR component CSV", path.parent)
            )
        elif is_dataset_dir("MESS_TRUVIS", path.parent) and name_low.startswith(
            path.parent.name.lower() + "_"
        ):
            matches.append(
                FormatMatch("1D", "MESS_TRUVIS", 0.90, "MESS TRUVIS component CSV", path.parent)
            )
        elif _is_helios_csv_content(path):
            matches.append(FormatMatch("1D", "Helios_TA", 0.92, "Helios TA CSV matrix", path))
        elif _is_two_col_xy_text(path):
            matches.append(
                FormatMatch(
                    "steady_state", "csv", 0.70, "Two-column steady-state CSV spectrum", path
                )
            )
    elif suffix in (".2d", ".dt", ".lg"):
        if _uos_2dir_is_dataset(path.parent):
            matches.append(
                FormatMatch("2D", "UoS_2DIR", 0.95, "UoS 2D-IR dataset file", path.parent)
            )
        elif is_dataset_dir("UoS_IRpp", path.parent):
            matches.append(
                FormatMatch("1D", "UoS_IRpp", 0.95, "UoS IR pump-probe dataset file", path.parent)
            )
    elif suffix == ".txt":
        if is_dataset_dir("Exported_TXT", path.parent) and name_low in (
            "ta.txt",
            "signal.txt",
            "time.txt",
            "delays.txt",
            "delay.txt",
            "times.txt",
            "wavelength.txt",
            "wavelengths.txt",
            "wavenumber.txt",
            "wavenumbers.txt",
            "probe.txt",
        ):
            matches.append(
                FormatMatch("1D", "Exported_TXT", 0.92, "Exported TXT TA dataset", path.parent)
            )
        elif _is_two_col_xy_text(path):
            matches.append(
                FormatMatch(
                    "steady_state", "txt", 0.65, "Two-column steady-state text spectrum", path
                )
            )
    elif suffix == ".xy":
        if _is_two_col_xy_text(path):
            matches.append(
                FormatMatch(
                    "steady_state", "xy", 0.85, "Two-column X-Y steady-state spectrum", path
                )
            )
    elif suffix == ".opus":
        matches.append(FormatMatch("steady_state", "opus", 0.90, "Bruker OPUS spectrum", path))

    return matches


def detect_format(
    path: str | PathLike,
    *,
    return_all: bool = False,
) -> FormatMatch | list[FormatMatch] | None:
    """Detect the PyMORGAN dataset/project/recipe format for ``path``.

    Parameters
    ----------
    path:
        File or directory path to inspect.
    return_all:
        When ``True``, return a list of all matching :class:`FormatMatch` entries
        ordered by descending confidence. When ``False`` (default), return the
        single highest-confidence match or ``None``.
    """
    p = Path(path).expanduser()
    try:
        p = p.resolve()
    except OSError:
        pass

    if not p.exists():
        return [] if return_all else None

    if p.is_dir():
        raw_matches = _detect_directory(p)
    elif p.is_file():
        raw_matches = _detect_file(p)
    else:
        raw_matches = []

    best_by_key: dict[tuple[str, str, Path], FormatMatch] = {}
    for m in raw_matches:
        key = (m.dimension, m.data_type, m.resolved_path)
        prev = best_by_key.get(key)
        if prev is None or m.confidence > prev.confidence:
            best_by_key[key] = m

    ordered = sorted(
        best_by_key.values(),
        key=lambda m: (-m.confidence, m.dimension, m.data_type),
    )
    if return_all:
        return ordered
    return ordered[0] if ordered else None


def load_auto(path: str | PathLike, **kwargs):
    """Auto-detect the format at ``path`` and dispatch to the matching loader."""
    match = detect_format(path)
    if match is None:
        raise ValueError(f"Could not detect a supported PyMORGAN format for {path!r}.")

    if match.dimension == "1D":
        from .oneD import load_1D

        return load_1D(match.resolved_path, data_type=match.data_type, **kwargs)
    if match.dimension == "2D":
        from .twoD import load_2D

        return load_2D(match.resolved_path, data_type=match.data_type, **kwargs)
    if match.dimension == "steady_state":
        from .steadyState import load_spectrum

        return load_spectrum(match.resolved_path, data_type=match.data_type, **kwargs)
    if match.dimension == "project":
        try:
            from .project import load_project  # type: ignore[import-not-found]
        except ImportError as exc:
            raise NotImplementedError(
                "Project (.pmproj) loading is not available in this build."
            ) from exc
        return load_project(match.resolved_path, **kwargs)
    if match.dimension == "recipe":
        try:
            from .recipe import load_recipe  # type: ignore[import-not-found]
        except ImportError as exc:
            raise NotImplementedError(
                "Recipe (.pmrecipe.toml) loading is not available in this build."
            ) from exc
        return load_recipe(match.resolved_path, **kwargs)
    raise ValueError(f"Unsupported detected dimension {match.dimension!r}.")

"""Headless and GUI-callable batch processing engine for PyMORGAN."""

from __future__ import annotations

import csv
import fnmatch
import json
import os
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, Literal

from .log import get_logger
from .recipe import Recipe

logger = get_logger(__name__)

SUPPORTED_EXPORT_FORMATS = (
    "pdat",
    "p2dat",
    "hdf5",
    "glotaran_ascii",
    "glotaran_nc",
    "png",
)

_IGNORED_AUTO_SUFFIXES = {
    ".pdatn",
    ".toml",
    ".pmproj",
    ".json",
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
    ".svg",
    ".py",
    ".md",
    ".txt",
}

_FOLDER_LOADERS_1D = {"UoS_IRpp", "MESS_TRIR", "MESS_TRUVIS", "Exported_TXT", "RAL_IRpp"}
_FOLDER_LOADERS_2D = {"UoS_2DIR", "RAL_RAW", "RAL_Proc", "MESS_2DIR"}
_ALL_FOLDER_LOADERS = _FOLDER_LOADERS_1D | _FOLDER_LOADERS_2D


@dataclass
class BatchItemResult:
    """Result record for a single dataset in a batch run."""

    input_path: str
    data_type: str
    status: Literal["ok", "error", "skipped", "cancelled", "dry_run"]
    elapsed_s: float = 0.0
    output_files: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class BatchReport:
    """Aggregated summary of a batch processing execution."""

    recipe_name: str
    output_dir: str
    total: int
    succeeded: int
    failed: int
    skipped: int
    cancelled: bool = False
    dry_run: bool = False
    elapsed_s: float = 0.0
    items: list[BatchItemResult] = field(default_factory=list)
    report_csv: str | None = None
    report_json: str | None = None


def _guess_format_fallback(
    path: Path,
    *,
    dimension: str | None = None,
) -> tuple[str, str] | None:
    """Fallback format sniffer used when ``pymorgan.detect`` is unavailable."""
    if not path.is_file():
        return None
    name_low = path.name.lower()
    if name_low.endswith(".pmrecipe.toml") or name_low.startswith("batch_report."):
        return None
    suffix = path.suffix.lower()
    if suffix in _IGNORED_AUTO_SUFFIXES:
        return None

    if suffix == ".pdat":
        return ("1D", "PDAT")
    if suffix == ".p2dat":
        return ("2D", "P2DAT")
    if suffix == ".hdf5":
        return ("1D", "HDF5")
    if suffix == ".h5":
        try:
            from .oneD.load import classify_pymess_h5

            kind = classify_pymess_h5(path)
        except Exception:
            kind = "other"
        if kind in ("2d_time_domain", "2d_freq_domain"):
            return ("2D", "PyMESS_2D")
        if kind in ("pump_probe", "shaper_t2_scan"):
            return ("1D", "PyMESS_PP")
        # Check if PyMORGAN saved 1D HDF5 inside a .h5 file
        try:
            import h5py

            with h5py.File(path, "r") as f:
                if "delays" in f and ("Zavg_R" in f or "Zavg_C" in f or "data_type" in f.attrs):
                    return ("1D", "HDF5")
        except Exception:
            pass
        return None
    if suffix == ".info":
        try:
            from .oneD.load import classify_quickcontrol_info

            qc_kind, has_scans = classify_quickcontrol_info(path)
        except Exception:
            return None
        if not has_scans:
            return None
        if qc_kind == "2d":
            return ("2D", "QuickControl_2D")
        if qc_kind == "1d":
            return ("1D", "QuickControl_PP")
        return None
    if dimension == "1D" and suffix == ".dat":
        return ("1D", "UniGE_fsTA")
    if dimension == "1D" and suffix == ".csv":
        return ("1D", "Helios_TA")
    return None


def _detect_via_registry_or_fallback(
    path: Path,
    *,
    dimension: str | None = None,
) -> tuple[Path, str] | None:
    """Resolve ``(resolved_path, data_type)`` via ``pymorgan.detect`` or fallback."""
    if path.name.lower().startswith("batch_report.") or path.name.lower().endswith(
        ".pmrecipe.toml"
    ):
        return None
    if path.suffix.lower() == ".pdatn":
        return None

    try:
        from .detect import detect_format

        match = detect_format(path)
        if match is not None:
            if match.dimension in ("steady_state", "project", "recipe"):
                return None
            if dimension and match.dimension != dimension:
                return None
            return (Path(match.resolved_path), match.data_type)
        # Allow dedicated single-file PyMORGAN extensions (e.g. corrupt .pdat/.p2dat)
        # to be discovered so batch error-handling ("skip"/"stop") reports them.
        if path.suffix.lower() not in {".pdat", ".p2dat"}:
            return None
    except ImportError:
        pass

    guessed = _guess_format_fallback(path, dimension=dimension)
    if guessed is None:
        return None
    dim, dt = guessed
    if dimension and dim != dimension:
        return None
    return (path, dt)


def _loader_glob_for_type(data_type: str) -> str | None:
    """Look up the default filename glob pattern for an explicit loader key."""
    try:
        from .oneD import load as load1d
        from .twoD import load as load2d

        if data_type in getattr(load1d, "_DATATYPE_GLOBS", {}):
            return str(load1d._DATATYPE_GLOBS[data_type])
        if data_type in getattr(load2d, "_MAP_DATATYPE_GLOBS", {}):
            return str(load2d._MAP_DATATYPE_GLOBS[data_type])
    except Exception:
        pass
    return None


def _file_filter_for_type(data_type: str):
    """Look up an optional per-format predicate filtering sibling auxiliary files."""
    try:
        from .oneD import load as load1d
        from .twoD import load as load2d

        fn = getattr(load1d, "_DATASET_FILE_FILTERS", {}).get(data_type)
        if fn is not None:
            return fn
        return getattr(load2d, "_MAP_DATASET_FILE_FILTERS", {}).get(data_type)
    except Exception:
        return None


def _detect_folder_dataset(dir_path: Path):
    """Return a ``FormatMatch`` only when ``dir_path`` itself is a multi-file dataset folder."""
    try:
        from .detect import detect_format
    except ImportError:
        return None
    match = detect_format(dir_path)
    if match is not None and match.data_type in _ALL_FOLDER_LOADERS:
        return match
    return None


def discover_batch_inputs(
    paths_or_dir: str | PathLike | Iterable[str | PathLike],
    *,
    data_type: str = "auto",
    dimension: Literal["1D", "2D"] | None = None,
    glob_pattern: str | None = None,
    recursive: bool = False,
) -> list[tuple[Path, str]]:
    """Discover dataset files/folders suitable for batch processing.

    Parameters
    ----------
    paths_or_dir : path-like or iterable of path-like
        File path(s) or directory path(s) to scan.
    data_type : str, default "auto"
        Explicit PyMORGAN loader key (e.g. ``"PDAT"``, ``"P2DAT"``, ``"PyMESS_PP"``)
        or ``"auto"`` to sniff formats automatically.
    dimension : {"1D", "2D"}, optional
        Filter auto-discovered inputs to a specific dimensionality.
    glob_pattern : str, optional
        Optional filename wildcard pattern (e.g. ``"sample_*.pdat"``).
    recursive : bool, default False
        Recurse into subdirectories when scanning folders.

    Returns
    -------
    list of (Path, str)
        Sorted deduplicated list of ``(resolved_path, data_type)`` pairs.
    """
    if isinstance(paths_or_dir, (str, PathLike)):
        raw_items: list[str | PathLike] = [paths_or_dir]
    else:
        raw_items = list(paths_or_dir)

    explicit_type = data_type if data_type and data_type != "auto" else None
    pattern = glob_pattern or (_loader_glob_for_type(explicit_type) if explicit_type else None)

    results: list[tuple[Path, str]] = []
    seen: set[str] = set()

    def _add(p: Path, dt: str) -> None:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            results.append((p, dt))

    is_folder_loader = explicit_type in _ALL_FOLDER_LOADERS
    file_filter = _file_filter_for_type(explicit_type) if explicit_type is not None else None

    def _scan_dir_auto(cur_dir: Path) -> None:
        try:
            children = sorted(cur_dir.iterdir(), key=lambda x: x.name.lower())
        except OSError:
            return
        for child in children:
            if child.is_dir():
                fm = _detect_folder_dataset(child)
                if fm is not None:
                    if (dimension is None or fm.dimension == dimension) and (
                        not glob_pattern or fnmatch.fnmatch(child.name, glob_pattern)
                    ):
                        _add(Path(fm.resolved_path), fm.data_type)
                elif recursive:
                    _scan_dir_auto(child)
            elif child.is_file():
                if glob_pattern and not fnmatch.fnmatch(child.name, glob_pattern):
                    continue
                detected = _detect_via_registry_or_fallback(child, dimension=dimension)
                if detected is not None:
                    _add(detected[0], detected[1])

    for raw in raw_items:
        p = Path(raw)
        if not p.exists():
            continue

        if p.is_file():
            if glob_pattern and not fnmatch.fnmatch(p.name, glob_pattern):
                continue
            if explicit_type is not None:
                if file_filter is None or file_filter(p):
                    _add(p, explicit_type)
            else:
                detected = _detect_via_registry_or_fallback(p, dimension=dimension)
                if detected is not None:
                    _add(detected[0], detected[1])
            continue

        if p.is_dir():
            if is_folder_loader and explicit_type is not None:
                subdirs = sorted(
                    d for d in (p.rglob("*") if recursive else p.iterdir()) if d.is_dir()
                )
                candidates = subdirs if subdirs else [p]
                for cand in candidates:
                    if glob_pattern and not fnmatch.fnmatch(cand.name, glob_pattern):
                        continue
                    _add(cand, explicit_type)
                continue

            if explicit_type is not None:
                search_pat = pattern or "*"
                iterator = p.rglob(search_pat) if recursive else p.glob(search_pat)
                for child in sorted(iterator):
                    if child.is_file() and (file_filter is None or file_filter(child)):
                        _add(child, explicit_type)
                continue

            fm = _detect_folder_dataset(p)
            if fm is not None:
                if (dimension is None or fm.dimension == dimension) and (
                    not glob_pattern or fnmatch.fnmatch(p.name, glob_pattern)
                ):
                    _add(Path(fm.resolved_path), fm.data_type)
                continue

            _scan_dir_auto(p)

    results.sort(key=lambda item: str(item[0]).lower())
    return results


def _load_dataset_item(path: Path, data_type: str, dimension: str | None) -> Any:
    """Load a 1D or 2D dataset using ``data_type`` (or ``load_auto`` when auto)."""
    import pymorgan as pm

    if not data_type or data_type == "auto":
        try:
            from .detect import load_auto

            return load_auto(path)
        except ImportError:
            detected = _detect_via_registry_or_fallback(path, dimension=dimension)
            if detected is None:
                raise ValueError(f"Unable to auto-detect dataset format for: {path}") from None
            path, data_type = detected

    is_2d = False
    if dimension == "2D":
        is_2d = True
    elif dimension == "1D":
        is_2d = False
    else:
        try:
            from .twoD import load as load2d

            is_2d = data_type in getattr(load2d, "_LOADERS", {})
        except Exception:
            is_2d = data_type in {"P2DAT", "PyMESS_2D", "QuickControl_2D", "MESS_2DIR", "UoS_2DIR"}

    if is_2d:
        return pm.load_2D(path, data_type=data_type)
    return pm.load_1D(path, data_type=data_type)


def _export_glotaran_ascii(ds: Any, target: Path) -> Path:
    """Export dataset using ``to_glotaran_ascii`` if present, or TIMP Time-explicit fallback."""
    if hasattr(ds, "to_glotaran_ascii"):
        res = ds.to_glotaran_ascii(target)
        return Path(res) if res is not None else target

    if not hasattr(ds, "Zavg_R"):
        raise NotImplementedError(
            "Fallback glotaran_ascii export is only implemented for 1D datasets."
        )

    import numpy as np

    delays = np.asarray(ds.delays, dtype=float)
    probe = np.asarray(ds.probe, dtype=float)
    if probe.ndim == 2:
        probe = probe[:, 0]
    Z = np.asarray(ds.Z, dtype=float)
    if Z.ndim == 3:
        Z = Z[:, :, 0]
    # Standardize matrix rows = probe wavelengths/wavenumbers, cols = delays
    if Z.shape == (delays.size, probe.size):
        mat = Z.T
    elif Z.shape == (probe.size, delays.size):
        mat = Z
    else:
        raise ValueError(f"Unexpected 1D signal shape {Z.shape} for glotaran_ascii export.")

    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        fh.write("PyMORGAN export\n")
        fh.write("Time explicit\n")
        fh.write(f"Intervalnr {delays.size}\n")
        fh.write(" ".join(f"{t:.8g}" for t in delays) + "\n")
        for i, wl in enumerate(probe):
            row_vals = " ".join("NaN" if not np.isfinite(v) else f"{v:.8g}" for v in mat[i])
            fh.write(f"{wl:.8g} {row_vals}\n")
    return target


def _export_png_summary(ds: Any, target: Path) -> Path:
    """Render a headless publication-quality summary PNG using Agg."""
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(7.2, 5.2), dpi=150)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    try:
        if hasattr(ds, "Z_R"):
            from .twoD.plot import plot_map

            delays = np.asarray(ds.delays, dtype=float)
            non_neg = delays[delays >= 0] if delays.size else delays
            t2 = float(non_neg[0]) if non_neg.size else (float(delays[0]) if delays.size else 0.0)
            plot_map(ds, t2, ax=ax)
        else:
            from .oneD.plot import plot_contour

            plot_contour(ds, ax=ax)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=150, bbox_inches="tight")
    finally:
        fig.clear()
    return target


def _export_dataset(
    ds: Any,
    out_dir: Path,
    stem: str,
    recipe: Recipe,
) -> list[str]:
    """Export ``ds`` into all requested formats in ``recipe.export_formats``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = f"{stem}{recipe.export_suffix}"
    written: list[Path] = []
    formats = [
        str(f).strip().lower() for f in (recipe.export_formats or ["pdat"]) if str(f).strip()
    ]
    is_2d = hasattr(ds, "Z_R")

    for fmt in formats:
        if fmt == "pdat":
            if is_2d:
                raise ValueError(
                    "Cannot export a 2D dataset in 'pdat' format; use 'p2dat' instead."
                )
            paths = ds.to_pdat(out_dir / f"{out_stem}.pdat", write_noise=bool(recipe.write_noise))
            written.extend(Path(p) for p in paths)
        elif fmt == "p2dat":
            if not is_2d:
                raise ValueError(
                    "Cannot export a 1D dataset in 'p2dat' format; use 'pdat' instead."
                )
            target = out_dir / f"{out_stem}.p2dat"
            ds.to_p2dat(target)
            written.append(target)
        elif fmt == "hdf5":
            target = out_dir / f"{out_stem}.hdf5"
            if hasattr(ds, "to_hdf5"):
                res = ds.to_hdf5(target)
                written.append(Path(res) if res is not None else target)
            else:
                raise NotImplementedError("HDF5 export is not supported for this dataset type.")
        elif fmt == "glotaran_ascii":
            written.append(_export_glotaran_ascii(ds, out_dir / f"{out_stem}.ascii"))
        elif fmt == "glotaran_nc":
            target = out_dir / f"{out_stem}.nc"
            if hasattr(ds, "to_glotaran_nc"):
                res = ds.to_glotaran_nc(target)
                written.append(Path(res) if res is not None else target)
            else:
                raise NotImplementedError(
                    "glotaran_nc export requires Dataset.to_glotaran_nc() support."
                )
        elif fmt == "png":
            written.append(_export_png_summary(ds, out_dir / f"{out_stem}.png"))
        else:
            raise ValueError(
                f"Unsupported batch export format {fmt!r}. Supported: {', '.join(SUPPORTED_EXPORT_FORMATS)}."
            )

    return [str(p) for p in written]


def _planned_outputs(out_dir: Path, stem: str, recipe: Recipe) -> list[str]:
    """Return planned output paths for ``dry_run=True`` previews."""
    ext_map = {
        "pdat": ".pdat",
        "p2dat": ".p2dat",
        "hdf5": ".hdf5",
        "glotaran_ascii": ".ascii",
        "glotaran_nc": ".nc",
        "png": ".png",
    }
    out_stem = f"{stem}{recipe.export_suffix}"
    planned: list[str] = []
    for fmt in recipe.export_formats or ["pdat"]:
        ext = ext_map.get(str(fmt).strip().lower(), f".{fmt}")
        planned.append(str(out_dir / f"{out_stem}{ext}"))
    return planned


def _run_batch_worker(
    path_str: str,
    data_type: str,
    recipe_dict: dict[str, Any],
    out_dir_str: str,
    stem: str,
    base_dir_str: str | None,
) -> dict[str, Any]:
    """Top-level picklable worker function for ``ProcessPoolExecutor`` and serial execution."""
    t0 = time.perf_counter()
    try:
        # Ensure headless plotting inside spawned worker processes
        os.environ.setdefault("MPLBACKEND", "Agg")
        recipe = Recipe.from_dict(recipe_dict)
        p = Path(path_str)
        ds = _load_dataset_item(p, data_type, recipe.dimension)
        base_dir = Path(base_dir_str) if base_dir_str else p.parent
        recipe.apply(ds, base_dir=base_dir)
        out_files = _export_dataset(ds, Path(out_dir_str), stem, recipe)
        return asdict(
            BatchItemResult(
                input_path=path_str,
                data_type=getattr(ds, "data_type", data_type) or data_type,
                status="ok",
                elapsed_s=time.perf_counter() - t0,
                output_files=out_files,
                error=None,
            )
        )
    except Exception as exc:
        return asdict(
            BatchItemResult(
                input_path=path_str,
                data_type=data_type,
                status="error",
                elapsed_s=time.perf_counter() - t0,
                output_files=[],
                error=f"{type(exc).__name__}: {exc}",
            )
        )


def _assign_unique_stems(items: Sequence[tuple[Path, str]]) -> list[tuple[Path, str, str]]:
    """Assign collision-free filename stems while keeping unique names unaltered."""
    raw_stems = [(p, dt, p.stem if p.is_file() else p.name) for p, dt in items]
    counts: dict[str, int] = {}
    for _, _, st in raw_stems:
        counts[st] = counts.get(st, 0) + 1

    seen_used: dict[str, int] = {}
    resolved: list[tuple[Path, str, str]] = []
    for p, dt, st in raw_stems:
        if counts[st] == 1:
            resolved.append((p, dt, st))
        else:
            idx = seen_used.get(st, 0) + 1
            seen_used[st] = idx
            resolved.append((p, dt, f"{st}__{idx}"))
    return resolved


def _write_batch_reports(report: BatchReport, out_dir: Path) -> None:
    """Write ``batch_report.csv`` and ``batch_report.json`` into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "batch_report.csv"
    json_path = out_dir / "batch_report.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["input_path", "data_type", "status", "elapsed_s", "output_files", "error"])
        for item in report.items:
            writer.writerow(
                [
                    item.input_path,
                    item.data_type,
                    item.status,
                    f"{item.elapsed_s:.4f}",
                    ";".join(item.output_files),
                    item.error or "",
                ]
            )

    report.report_csv = str(csv_path)
    report.report_json = str(json_path)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(asdict(report), fh, indent=2)


def run_batch(
    inputs: Sequence[str | PathLike | tuple[str | PathLike, str]],
    recipe: Recipe | str | PathLike,
    output_dir: str | PathLike,
    *,
    workers: int = 1,
    on_error: Literal["skip", "stop"] = "skip",
    dry_run: bool = False,
    base_dir: str | PathLike | None = None,
    progress_callback: Callable[[int, int, BatchItemResult], None] | None = None,
    cancel_event: Any = None,
) -> BatchReport:
    """Apply ``recipe`` across a sequence of dataset inputs and write reports.

    Parameters
    ----------
    inputs : sequence of path-like or ``(path, data_type)`` tuples
        Input datasets to process.
    recipe : Recipe or path to ``.pmrecipe.toml``
        Loaded :class:`Recipe` instance or path to load one from.
    output_dir : str or PathLike
        Directory where processed exports and ``batch_report.{csv,json}`` are saved.
    workers : int, default 1
        Number of worker processes (``workers > 1`` uses :class:`ProcessPoolExecutor`).
    on_error : {"skip", "stop"}, default "skip"
        Whether to continue past item errors (``"skip"``) or abort remaining work.
    dry_run : bool, default False
        If True, discover and report planned outputs without processing or writing files.
    base_dir : str or PathLike, optional
        Base directory for resolving relative auxiliary files inside the recipe.
    progress_callback : callable, optional
        Called as ``progress_callback(completed_count, total_count, item_result)``.
    cancel_event : threading.Event-like, optional
        Object with ``is_set() -> bool`` checked between items for cooperative cancellation.

    Returns
    -------
    BatchReport
    """
    if on_error not in ("skip", "stop"):
        raise ValueError(f"Invalid on_error mode {on_error!r}; expected 'skip' or 'stop'.")

    if isinstance(recipe, (str, PathLike)):
        recipe_path = Path(recipe)
        loaded_recipe = Recipe.load(recipe_path)
        resolved_base = Path(base_dir) if base_dir is not None else recipe_path.parent
    else:
        loaded_recipe = recipe
        resolved_base = Path(base_dir) if base_dir is not None else None

    norm_inputs: list[tuple[Path, str]] = []
    for entry in inputs:
        if isinstance(entry, tuple):
            p, dt = entry
            norm_inputs.append((Path(p), str(dt)))
        else:
            norm_inputs.append((Path(entry), loaded_recipe.data_type or "auto"))

    stems_list = _assign_unique_stems(norm_inputs)
    out_dir = Path(output_dir)
    total = len(stems_list)
    t_start = time.perf_counter()

    if dry_run:
        items: list[BatchItemResult] = []
        for idx, (p, dt, st) in enumerate(stems_list, start=1):
            res = BatchItemResult(
                input_path=str(p),
                data_type=dt,
                status="dry_run",
                elapsed_s=0.0,
                output_files=_planned_outputs(out_dir, st, loaded_recipe),
                error=None,
            )
            items.append(res)
            if progress_callback is not None:
                progress_callback(idx, total, res)
        return BatchReport(
            recipe_name=loaded_recipe.name,
            output_dir=str(out_dir),
            total=total,
            succeeded=0,
            failed=0,
            skipped=total,
            cancelled=False,
            dry_run=True,
            elapsed_s=time.perf_counter() - t_start,
            items=items,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    recipe_dict = loaded_recipe.to_dict()
    base_dir_str = str(resolved_base) if resolved_base is not None else None

    results: list[BatchItemResult] = []
    cancelled = False
    stopped_on_error = False
    completed = 0

    def _is_cancelled() -> bool:
        return bool(cancel_event is not None and cancel_event.is_set())

    n_workers = max(1, int(workers))
    if n_workers == 1 or total <= 1:
        for p, dt, st in stems_list:
            if _is_cancelled():
                cancelled = True
                res = BatchItemResult(
                    input_path=str(p),
                    data_type=dt,
                    status="cancelled",
                )
                results.append(res)
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total, res)
                continue

            if stopped_on_error:
                res = BatchItemResult(
                    input_path=str(p),
                    data_type=dt,
                    status="skipped",
                    error="Stopped due to earlier error (on_error='stop').",
                )
                results.append(res)
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total, res)
                continue

            raw_res = _run_batch_worker(str(p), dt, recipe_dict, str(out_dir), st, base_dir_str)
            res = BatchItemResult(**raw_res)
            results.append(res)
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total, res)
            if res.status == "error" and on_error == "stop":
                stopped_on_error = True
    else:
        ordered_results: dict[int, BatchItemResult] = {}
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures: dict[Any, int] = {}
            for idx, (p, dt, st) in enumerate(stems_list):
                if _is_cancelled():
                    cancelled = True
                    ordered_results[idx] = BatchItemResult(
                        input_path=str(p),
                        data_type=dt,
                        status="cancelled",
                    )
                    continue
                fut = executor.submit(
                    _run_batch_worker,
                    str(p),
                    dt,
                    recipe_dict,
                    str(out_dir),
                    st,
                    base_dir_str,
                )
                futures[fut] = idx

            pending = set(futures.keys())
            while pending:
                if _is_cancelled():
                    cancelled = True
                    for pf in list(pending):
                        if pf.cancel():
                            pending.discard(pf)
                            idx = futures[pf]
                            p, dt, _ = stems_list[idx]
                            res = BatchItemResult(
                                input_path=str(p),
                                data_type=dt,
                                status="cancelled",
                            )
                            ordered_results[idx] = res
                            completed += 1
                            if progress_callback is not None:
                                progress_callback(completed, total, res)
                    if not pending:
                        break

                done, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                for df in done:
                    idx = futures[df]
                    p, dt, _ = stems_list[idx]
                    try:
                        res = BatchItemResult(**df.result())
                    except Exception as exc:
                        res = BatchItemResult(
                            input_path=str(p),
                            data_type=dt,
                            status="error",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    ordered_results[idx] = res
                    completed += 1
                    if progress_callback is not None:
                        progress_callback(completed, total, res)
                    if res.status == "error" and on_error == "stop":
                        stopped_on_error = True
                        for pf in list(pending):
                            if pf.cancel():
                                pending.discard(pf)
                                pidx = futures[pf]
                                pp, pdt, _ = stems_list[pidx]
                                sres = BatchItemResult(
                                    input_path=str(pp),
                                    data_type=pdt,
                                    status="skipped",
                                    error="Stopped due to earlier error (on_error='stop').",
                                )
                                ordered_results[pidx] = sres
                                completed += 1
                                if progress_callback is not None:
                                    progress_callback(completed, total, sres)

        for idx in range(total):
            if idx in ordered_results:
                results.append(ordered_results[idx])
            else:
                p, dt, _ = stems_list[idx]
                results.append(BatchItemResult(input_path=str(p), data_type=dt, status="cancelled"))

    succeeded = sum(1 for r in results if r.status == "ok")
    failed = sum(1 for r in results if r.status == "error")
    skipped = sum(1 for r in results if r.status in ("skipped", "cancelled"))

    report = BatchReport(
        recipe_name=loaded_recipe.name,
        output_dir=str(out_dir),
        total=total,
        succeeded=succeeded,
        failed=failed,
        skipped=skipped,
        cancelled=cancelled,
        dry_run=False,
        elapsed_s=time.perf_counter() - t_start,
        items=results,
    )
    _write_batch_reports(report, out_dir)
    return report


def _build_cli_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="pymorgan batch",
        description="Apply a saved PyMORGAN .pmrecipe.toml processing recipe to a batch of datasets.",
    )
    parser.add_argument(
        "--recipe",
        required=True,
        help="Path to a .pmrecipe.toml recipe file.",
    )
    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        action="append",
        required=True,
        help="Input file(s) and/or directory path(s) to discover.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Destination folder for exported datasets and batch_report.{csv,json}.",
    )
    parser.add_argument(
        "--data-type",
        default=None,
        help="Explicit loader type (default: recipe.data_type or 'auto').",
    )
    parser.add_argument(
        "--glob",
        default=None,
        help="Optional filename wildcard filter when scanning folders.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when scanning input folders.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List discovered datasets and planned outputs without running processing.",
    )
    parser.add_argument(
        "--on-error",
        choices=("skip", "stop"),
        default="skip",
        help="Behavior when an individual dataset fails (default: skip).",
    )
    parser.add_argument(
        "--format",
        action="append",
        dest="formats",
        default=None,
        help=f"Override recipe export formats ({', '.join(SUPPORTED_EXPORT_FORMATS)}).",
    )
    return parser


def cli_main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for ``pymorgan batch`` and ``pymorgan-batch``."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    parser = _build_cli_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    recipe_path = Path(args.recipe)
    recipe = Recipe.load(recipe_path)
    if args.formats:
        recipe.export_formats = [str(f).strip().lower() for f in args.formats if str(f).strip()]

    raw_inputs = [item for group in args.input for item in group]
    eff_data_type = args.data_type or recipe.data_type or "auto"
    discovered = discover_batch_inputs(
        raw_inputs,
        data_type=eff_data_type,
        dimension=recipe.dimension if eff_data_type == "auto" else None,
        glob_pattern=args.glob,
        recursive=bool(args.recursive),
    )

    if not discovered:
        print("No matching datasets found.", file=sys.stderr)
        return 1

    def _on_progress(done: int, total: int, res: BatchItemResult) -> None:
        tag = res.status.upper()
        extra = f" -> {', '.join(res.output_files)}" if res.output_files else ""
        err = f" ({res.error})" if res.error else ""
        print(f"[{done}/{total}] {tag}: {res.input_path}{extra}{err}")

    report = run_batch(
        discovered,
        recipe,
        args.output,
        workers=args.workers,
        on_error=args.on_error,
        dry_run=bool(args.dry_run),
        base_dir=recipe_path.parent,
        progress_callback=_on_progress,
    )
    print(
        f"Batch finished: total={report.total}, succeeded={report.succeeded}, "
        f"failed={report.failed}, skipped={report.skipped}"
    )
    return 1 if report.failed > 0 else 0

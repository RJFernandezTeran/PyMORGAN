"""Portable TOML processing recipes (``.pmrecipe.toml``) for PyMORGAN."""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tomlkit

from .log import get_logger
from .provenance import sanitize_params

logger = get_logger(__name__)

RECIPE_SCHEMA_VERSION = "1.0"
_AUX_PATH_KEYS = ("chirp_file", "solvent_file", "ref_file", "csv_path")


def _relativize_path(raw: str | None, base_dir: Path | None) -> str | None:
    """Convert ``raw`` to a POSIX-style path relative to ``base_dir`` when feasible."""
    if not raw:
        return None
    p = Path(str(raw))
    if base_dir is None or not p.is_absolute():
        return str(raw)
    try:
        rel = os.path.relpath(str(p.resolve()), str(base_dir.resolve()))
    except (OSError, ValueError):
        return str(p)
    # Keep relative if inside base_dir or within 2 parent levels; otherwise keep absolute
    rel_path = Path(rel)
    up_count = sum(1 for part in rel_path.parts if part == "..")
    if up_count <= 2:
        return rel_path.as_posix()
    return str(p)


def resolve_aux_path(
    raw_path: str | PathLike,
    *,
    base_dir: str | PathLike | None = None,
    dataset_source: str | PathLike | None = None,
    fallback_abs: str | None = None,
) -> Path:
    """Resolve an auxiliary file path against ``base_dir``, ``dataset_source``, or ``fallback_abs``."""
    raw_p = Path(raw_path)
    candidates: list[Path] = []
    if raw_p.is_absolute():
        candidates.append(raw_p)
    else:
        if base_dir is not None:
            candidates.append(Path(base_dir) / raw_p)
        if dataset_source:
            src_p = Path(dataset_source)
            src_dir = src_p if src_p.is_dir() else src_p.parent
            candidates.append(src_dir / raw_p)
            if src_p.is_dir():
                candidates.append(src_p.parent / raw_p)
        candidates.append(raw_p)
    if fallback_abs:
        candidates.append(Path(fallback_abs))

    for cand in candidates:
        try:
            if cand.exists():
                return cand
        except OSError:
            continue
    return candidates[0] if candidates else raw_p


def _load_aux_dataset(path: Path, data_type: str | None, dimension: str):
    """Load an auxiliary 1D or 2D dataset using explicit ``data_type`` or ``load_auto`` fallback."""
    import pymorgan as pm

    if data_type and data_type != "auto":
        if dimension == "2D":
            return pm.load_2D(path, data_type=data_type)
        return pm.load_1D(path, data_type=data_type)

    try:
        from .detect import load_auto

        return load_auto(path)
    except ImportError:
        pass

    suffix = path.suffix.lower()
    if dimension == "2D":
        dt = (
            "P2DAT"
            if suffix == ".p2dat"
            else ("PyMESS_2D" if suffix in (".h5", ".hdf5") else "MESS_2DIR")
        )
        return pm.load_2D(path, data_type=dt)

    if suffix == ".pdat":
        dt = "PDAT"
    elif suffix in (".h5", ".hdf5"):
        try:
            from .oneD.load import classify_pymess_h5

            dt = "PyMESS_PP" if classify_pymess_h5(path) != "other" else "HDF5"
        except Exception:
            dt = "HDF5"
    elif suffix == ".info":
        dt = "QuickControl_PP"
    elif path.is_dir():
        dt = "MESS_TRIR"
    else:
        dt = "PDAT"
    return pm.load_1D(path, data_type=dt)


def _toml_clean(obj: Any) -> Any:
    """Recursively strip ``None`` values and convert tuples/arrays for tomlkit."""
    if obj is None:
        return None
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_toml_clean(x) for x in obj if x is not None]
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if v is None:
                continue
            cleaned = _toml_clean(v)
            if cleaned is not None:
                out[str(k)] = cleaned
        return out
    return str(obj)


def _normalize_step_dict(step: Mapping[str, Any]) -> dict[str, Any]:
    if "op" not in step or not step["op"]:
        raise ValueError("Recipe step must include a non-empty 'op' field.")
    op = str(step["op"])
    raw_params = step.get("params") or {}
    if not isinstance(raw_params, Mapping):
        raise ValueError(f"Step {op!r} 'params' must be a mapping.")
    params = sanitize_params(dict(raw_params))
    enabled = bool(step.get("enabled", True))
    label = str(step.get("label") or "")
    out: dict[str, Any] = {"op": op, "enabled": enabled, "params": params}
    if label:
        out["label"] = label
    if step.get("timestamp_utc"):
        out["timestamp_utc"] = str(step["timestamp_utc"])
    return out


def _apply_calibration_file(dataset: Any, cal_path: Path) -> None:
    """Apply a 1-column CSV calibration file (e.g. CalibratedProbe.csv) to ``dataset``."""
    from . import helpers as hlp

    if not cal_path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {cal_path}")
    arr = np.loadtxt(cal_path, delimiter=",").ravel().astype(float)
    probe = np.asarray(dataset.probe)
    if probe.ndim == 1 and arr.size == probe.size:
        dataset.probe = arr
    elif probe.ndim == 2 and arr.size == probe.shape[0]:
        for col in range(probe.shape[1]):
            probe[:, col] = arr
        dataset.probe = probe
    else:
        raise ValueError(
            f"Calibration file length ({arr.size}) does not match dataset probe length ({probe.shape[0]})."
        )
    dataset.cal_source = f"Recipe calibration ({cal_path.name})"
    if isinstance(getattr(dataset, "units", None), dict):
        dataset.units["cal_source"] = dataset.cal_source
        cur_T = dataset.units.get("T") or dataset.units.get("unitsT_ltx", "ps")
        cur_Z = dataset.units.get("Z") or dataset.units.get("unitsZ", "x1E3")
        unit_guess = "nm" if float(np.nanmedian(arr)) < 1200.0 else "cm^{-1}"
        if hasattr(dataset, "Zavg_R"):
            dataset.units.update(hlp.units2dic(unit_guess, cur_T, cur_Z))


@dataclass
class Recipe:
    """Serializable, dataset-agnostic processing recipe (``.pmrecipe.toml``)."""

    schema_version: str = RECIPE_SCHEMA_VERSION
    name: str = "Untitled Recipe"
    dimension: Literal["1D", "2D"] = "1D"
    data_type: str = "auto"
    calibration_file: str | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    export_formats: list[str] = field(default_factory=lambda: ["pdat"])
    export_suffix: str = "_PROCESSED"
    write_noise: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        sv = str(self.schema_version or RECIPE_SCHEMA_VERSION).strip()
        major = sv.split(".", 1)[0]
        if major != "1":
            raise ValueError(f"Unsupported recipe schema_version: {self.schema_version!r}")
        self.schema_version = sv

        dim_norm = str(self.dimension).strip().upper()
        if dim_norm not in ("1D", "2D"):
            raise ValueError(f"Recipe dimension must be '1D' or '2D', got {self.dimension!r}")
        self.dimension = dim_norm  # type: ignore[assignment]

        self.name = str(self.name or "Untitled Recipe")
        self.data_type = str(self.data_type or "auto")
        self.calibration_file = str(self.calibration_file) if self.calibration_file else None
        self.steps = [_normalize_step_dict(s) for s in (self.steps or [])]
        self.export_formats = [
            str(fmt).strip() for fmt in (self.export_formats or []) if str(fmt).strip()
        ]
        self.export_suffix = str(
            self.export_suffix if self.export_suffix is not None else "_PROCESSED"
        )
        self.write_noise = bool(self.write_noise)
        self.description = str(self.description or "")

    @classmethod
    def from_dataset(
        cls,
        ds: Any,
        *,
        name: str | None = None,
        data_type: str | None = None,
        calibration_file: str | PathLike | None = None,
        export_formats: list[str] | None = None,
        export_suffix: str = "_PROCESSED",
        write_noise: bool = True,
        description: str = "",
        base_dir: str | PathLike | None = None,
    ) -> Recipe:
        """Build a :class:`Recipe` from a processed dataset's recorded history."""
        dim: Literal["1D", "2D"] = "2D" if hasattr(ds, "Z_R") else "1D"
        dt = data_type or getattr(ds, "data_type", None) or "auto"
        fmts = (
            list(export_formats)
            if export_formats is not None
            else (["p2dat"] if dim == "2D" else ["pdat"])
        )
        src = getattr(ds, "source", None)
        rec_name = name or (Path(str(src)).stem if src else f"{dim} Processing Recipe")
        base_p = Path(base_dir) if base_dir is not None else None

        steps_out: list[dict[str, Any]] = []
        for raw_step in ds.history_to_dicts():
            st = copy.deepcopy(raw_step)
            params = st.get("params") or {}
            if base_p is not None:
                for key in _AUX_PATH_KEYS:
                    if params.get(key):
                        params[key] = _relativize_path(str(params[key]), base_p)
            st["enabled"] = True
            steps_out.append(st)

        cal_str = str(calibration_file) if calibration_file else None
        if cal_str and base_p is not None:
            cal_str = _relativize_path(cal_str, base_p)

        return cls(
            name=rec_name,
            dimension=dim,
            data_type=str(dt),
            calibration_file=cal_str,
            steps=steps_out,
            export_formats=fmts,
            export_suffix=export_suffix,
            write_noise=write_noise,
            description=description,
        )

    def to_dict(self, *, base_dir: str | PathLike | None = None) -> dict[str, Any]:
        """Serialize this recipe to a plain TOML/JSON-friendly dictionary."""
        base_p = Path(base_dir) if base_dir is not None else None
        steps_clean: list[dict[str, Any]] = []
        for s in self.steps:
            sc = copy.deepcopy(s)
            params = sc.get("params") or {}
            if base_p is not None:
                for key in _AUX_PATH_KEYS:
                    if params.get(key):
                        params[key] = _relativize_path(str(params[key]), base_p)
            sc["params"] = _toml_clean(params)
            steps_clean.append(_toml_clean(sc))

        cal = self.calibration_file
        if cal and base_p is not None:
            cal = _relativize_path(cal, base_p)

        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "name": self.name,
            "dimension": self.dimension,
            "data_type": self.data_type,
            "export_formats": list(self.export_formats),
            "export_suffix": self.export_suffix,
            "write_noise": bool(self.write_noise),
            "steps": steps_clean,
        }
        if cal:
            out["calibration_file"] = cal
        if self.description:
            out["description"] = self.description
        return out

    def to_toml(self, *, base_dir: str | PathLike | None = None) -> str:
        """Render this recipe as a commented TOML string."""
        data = self.to_dict(base_dir=base_dir)
        doc = tomlkit.document()
        doc.add(tomlkit.comment("PyMORGAN Processing Recipe (.pmrecipe.toml)"))
        doc.add(
            tomlkit.comment(
                "Schema v1.0 - replayable via `pymorgan batch` or `Recipe.load(...).apply(ds)`"
            )
        )
        doc.add(tomlkit.nl())

        for key in (
            "schema_version",
            "name",
            "dimension",
            "data_type",
            "calibration_file",
            "description",
            "export_formats",
            "export_suffix",
            "write_noise",
        ):
            if key in data:
                doc[key] = data[key]

        doc.add(tomlkit.nl())
        if data["steps"]:
            aot = tomlkit.aot()
            for step in data["steps"]:
                tbl = tomlkit.table()
                tbl["op"] = step["op"]
                tbl["enabled"] = bool(step.get("enabled", True))
                if step.get("label"):
                    tbl["label"] = step["label"]
                if step.get("timestamp_utc"):
                    tbl["timestamp_utc"] = step["timestamp_utc"]
                params_tbl = tomlkit.table()
                for pk, pv in (step.get("params") or {}).items():
                    params_tbl[pk] = pv
                tbl["params"] = params_tbl
                aot.append(tbl)
            doc["steps"] = aot
        else:
            doc["steps"] = []

        return tomlkit.dumps(doc)

    def save(self, path: str | PathLike) -> Path:
        """Save this recipe to ``path`` (``.pmrecipe.toml``)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_toml(base_dir=target.parent), encoding="utf-8")
        return target

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Recipe:
        """Build a :class:`Recipe` from a parsed TOML/JSON mapping."""
        if "dimension" not in data:
            raise ValueError("Recipe is missing required field 'dimension'.")
        export_sec = data.get("export") if isinstance(data.get("export"), Mapping) else {}
        export_formats = data.get("export_formats")
        if export_formats is None:
            export_formats = export_sec.get(
                "formats", ["pdat" if str(data["dimension"]).upper() == "1D" else "p2dat"]
            )
        export_suffix = data.get("export_suffix", export_sec.get("suffix", "_PROCESSED"))
        write_noise = data.get("write_noise", export_sec.get("write_noise", True))

        return cls(
            schema_version=str(data.get("schema_version", RECIPE_SCHEMA_VERSION)),
            name=str(data.get("name", "Untitled Recipe")),
            dimension=str(data["dimension"]),  # type: ignore[arg-type]
            data_type=str(data.get("data_type", "auto")),
            calibration_file=data.get("calibration_file") or None,
            steps=list(data.get("steps") or []),
            export_formats=list(export_formats),
            export_suffix=str(export_suffix),
            write_noise=bool(write_noise),
            description=str(data.get("description", "")),
        )

    @classmethod
    def load(cls, path: str | PathLike) -> Recipe:
        """Load and validate a :class:`Recipe` from a ``.pmrecipe.toml`` file."""
        p = Path(path)
        raw = tomlkit.parse(p.read_text(encoding="utf-8")).unwrap()
        return cls.from_dict(raw)

    def apply(self, dataset: Any, *, base_dir: str | PathLike | None = None):
        """Replay this recipe's enabled steps onto ``dataset`` in-place and return it."""
        ds_dim = "2D" if hasattr(dataset, "Z_R") else "1D"
        if ds_dim != self.dimension:
            raise ValueError(
                f"Recipe dimension {self.dimension!r} does not match dataset dimension {ds_dim!r}."
            )

        ds_source = getattr(dataset, "source", None)
        if self.calibration_file:
            cal_path = resolve_aux_path(
                self.calibration_file,
                base_dir=base_dir,
                dataset_source=ds_source,
            )
            _apply_calibration_file(dataset, cal_path)

        for step in self.steps:
            if not step.get("enabled", True):
                continue
            op = step["op"]
            p = dict(step.get("params") or {})

            if ds_dim == "1D":
                self._apply_step_1d(dataset, op, p, base_dir=base_dir, ds_source=ds_source)
            else:
                self._apply_step_2d(dataset, op, p, base_dir=base_dir, ds_source=ds_source)

        return dataset

    @staticmethod
    def _apply_step_1d(
        dataset: Any,
        op: str,
        p: dict[str, Any],
        *,
        base_dir: str | PathLike | None,
        ds_source: str | None,
    ) -> None:
        if op == "background_correct":
            dataset.background_correct(
                tmin=p["tmin"],
                tmax=p["tmax"],
                do_correct=bool(p.get("do_correct", True)),
            )
        elif op in ("remove_scatter", "apply_scatter_removal"):
            from .oneD.scatter import ScatterBand

            raw_bands = p.pop("bands", None)
            bands = (
                [ScatterBand(**b) if isinstance(b, Mapping) else b for b in raw_bands]
                if raw_bands is not None
                else None
            )
            if "t0_exclude" in p and isinstance(p["t0_exclude"], list):
                p["t0_exclude"] = tuple(p["t0_exclude"])
            dataset.remove_scatter(bands=bands, **p)
        elif op == "clear_scatter_removal":
            dataset.clear_scatter_removal()
        elif op == "apply_chirp_correction":
            from .oneD.chirp import ChirpFit, load_chirp_fit

            chirp_file = p.get("chirp_file")
            fit = None
            if chirp_file:
                resolved = resolve_aux_path(
                    chirp_file,
                    base_dir=base_dir,
                    dataset_source=ds_source,
                    fallback_abs=p.get("chirp_file_abs"),
                )
                if resolved.exists() or not p.get("inline_chirp"):
                    fit = load_chirp_fit(resolved)
            if fit is None:
                ic = p.get("inline_chirp")
                if not isinstance(ic, Mapping):
                    raise ValueError(
                        "apply_chirp_correction step missing both chirp_file and inline_chirp."
                    )
                fit = ChirpFit(
                    mode=str(ic.get("mode", "manual")),
                    coeffs=np.asarray(ic.get("coeffs", [0.0]), dtype=float),
                    fit_wl=np.asarray(ic.get("fit_wl", []), dtype=float),
                    Pfit=np.empty((0, 1), dtype=float),
                    n_cauchy_terms=int(ic.get("n_cauchy_terms", 6)),
                    lambda_ref=float(ic.get("lambda_ref", 500.0)),
                    detector=int(ic.get("detector", 0)),
                )
            dataset.apply_chirp_correction(
                fit,
                detector=p.get("detector"),
                boundary_handling=p.get("boundary_handling"),
            )
        elif op == "subtract_solvent":
            sol_file = p.get("solvent_file")
            if not sol_file:
                raise ValueError("subtract_solvent step requires 'solvent_file'.")
            resolved = resolve_aux_path(
                sol_file,
                base_dir=base_dir,
                dataset_source=ds_source,
                fallback_abs=p.get("solvent_file_abs"),
            )
            solvent_ds = _load_aux_dataset(resolved, p.get("solvent_data_type"), "1D")
            dt = p.get("dt", 0.0)
            scale = p.get("scale", 1.0)
            if isinstance(dt, list):
                dt = np.asarray(dt, dtype=float)
            if isinstance(scale, list):
                scale = np.asarray(scale, dtype=float)
            dataset.subtract_solvent(solvent_ds, dt=dt, scale=scale)
        elif op == "subtract_shockwave":
            dataset.subtract_shockwave(**p)
        elif op == "mask_probe_regions":
            ranges = [tuple(r) for r in p.get("ranges", [])]
            dataset.mask_probe_regions(ranges, detector=p.get("detector"))
        elif op == "shift_t0":
            dataset.shift_t0(p["t0_shift"])
        elif op == "recalc_average":
            dataset.recalc_average(
                p["scan_indices"],
                spectrum=int(p.get("spectrum", 0)),
                slowmod=int(p.get("slowmod", 0)),
                noise_weighted=bool(p.get("noise_weighted", True)),
            )
        elif op == "crop":
            dataset.crop(**p)
        elif op == "embed_calibrated_probe":
            csv_p = p.get("csv_path")
            resolved_csv = (
                resolve_aux_path(
                    csv_p,
                    base_dir=base_dir,
                    dataset_source=ds_source,
                    fallback_abs=p.get("csv_path_abs"),
                )
                if csv_p
                else None
            )
            dataset.embed_calibrated_probe(
                resolved_csv,
                backup_original=bool(p.get("backup_original", True)),
            )
        else:
            raise ValueError(f"Unsupported 1D recipe operation: {op!r}")

    @staticmethod
    def _apply_step_2d(
        dataset: Any,
        op: str,
        p: dict[str, Any],
        *,
        base_dir: str | PathLike | None,
        ds_source: str | None,
    ) -> None:
        if op == "process":
            if isinstance(p.get("cal_probe_vector"), list):
                p["cal_probe_vector"] = np.asarray(p["cal_probe_vector"], dtype=float)
            dataset.process(**p)
        elif op == "background_correct":
            dataset.background_correct(**p)
        elif op == "recalc_average":
            dataset.recalc_average(p["scan_indices"])
        elif op == "subtract_spectrum":
            ref_file = p.get("ref_file")
            if not ref_file:
                raise ValueError("subtract_spectrum step requires 'ref_file'.")
            resolved = resolve_aux_path(
                ref_file,
                base_dir=base_dir,
                dataset_source=ds_source,
                fallback_abs=p.get("ref_file_abs"),
            )
            ref_ds = _load_aux_dataset(resolved, p.get("ref_data_type"), "2D")
            dataset.subtract_spectrum(
                ref_ds,
                scale=float(p.get("scale", 1.0)),
                ref_t2_idx=p.get("ref_t2_idx"),
            )
        elif op == "reset_subtraction":
            dataset.reset_subtraction()
        elif op == "crop":
            dataset.crop(**p)
        elif op == "set_active_state":
            dataset.set_active_state(str(p["state_name"]))
        elif op == "set_active_detector":
            if hasattr(dataset, "set_active_detector"):
                det_name = p.get("name", p.get("detector"))
                if det_name is not None:
                    dataset.set_active_detector(str(det_name))
        else:
            raise ValueError(f"Unsupported 2D recipe operation: {op!r}")

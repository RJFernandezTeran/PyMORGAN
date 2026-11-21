"""FAIR / NeXus HDF5 metadata and compressed dataset helpers for PyMORGAN."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from .__about__ import __version__

_MAX_SHA256_BYTES = 50 * 1024 * 1024  # 50 MB


def json_safe_value(val: Any) -> tuple[bool, Any]:
    """Recursively convert ``val`` to a JSON-serializable structure.

    Returns ``(ok, converted)`` where ``ok`` is ``False`` for raw multi-element
    arrays or unsupported objects so caller mappings can drop them safely.
    """
    if val is None or isinstance(val, (str, int, float, bool)):
        return True, val
    if isinstance(val, np.generic):
        return True, val.item()
    if isinstance(val, Path):
        return True, str(val)
    if isinstance(val, (list, tuple)):
        items = []
        for item in val:
            ok, clean = json_safe_value(item)
            if not ok:
                return False, None
            items.append(clean)
        return True, items
    if isinstance(val, dict):
        out: dict[str, Any] = {}
        for dk, dv in val.items():
            ok, clean = json_safe_value(dv)
            if ok:
                out[str(dk)] = clean
        return True, out
    return False, None


def json_safe_units(
    mapping: dict[str, Any] | None,
    *,
    drop_keys: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Return a JSON-safe copy of ``mapping``, dropping ndarray/excluded entries."""
    if not isinstance(mapping, dict):
        return {}
    filtered = {k: v for k, v in mapping.items() if k not in drop_keys}
    _, cleaned = json_safe_value(filtered)
    return cleaned if isinstance(cleaned, dict) else {}


def compute_source_sha256(
    source: str | PathLike | None,
    *,
    max_bytes: int = _MAX_SHA256_BYTES,
) -> str | None:
    """Return the hex SHA-256 of ``source`` if it is a regular file under ``max_bytes``."""
    if not source:
        return None
    try:
        p = Path(source)
        if not p.is_file():
            return None
        size = p.stat().st_size
        if size < 0 or size >= max_bytes:
            return None
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def extract_sample_info_metadata(units: dict[str, Any] | None) -> dict[str, Any]:
    """Collect sample-info strings and instrument metadata dictionaries from ``units``."""
    if not isinstance(units, dict):
        return {}
    out: dict[str, Any] = {}
    for key in (
        "sample_info",
        "sample_info_full",
        "pymess_metadata",
        "quickcontrol_info",
        "ufs_metadata",
    ):
        if key in units and units[key] is not None:
            ok, clean = json_safe_value(units[key])
            if ok:
                out[key] = clean
    for k, v in units.items():
        if str(k).startswith("info_") and v is not None:
            ok, clean = json_safe_value(v)
            if ok:
                out[str(k)] = clean
    return out


def write_fair_root_attrs(
    f,
    *,
    pymorgan_format: str,
    source: str | PathLike | None = None,
    original_data_type: str | None = None,
    units: dict[str, Any] | None = None,
    dataset: Any = None,
    format_version: str = "2.0",
) -> None:
    """Populate FAIR / NeXus root HDF5 attributes on ``f``."""
    f.attrs["NX_class"] = "NXroot"
    f.attrs["pymorgan_format"] = str(pymorgan_format)
    f.attrs["format_version"] = str(format_version)
    f.attrs["pymorgan_version"] = str(__version__)
    f.attrs["created_utc"] = datetime.now(UTC).isoformat()

    src_str = str(source or "")
    f.attrs["source"] = src_str

    sha = compute_source_sha256(source)
    if sha is not None:
        f.attrs["source_sha256"] = sha

    orig_dt = str(original_data_type or "")
    if isinstance(units, dict) and units.get("original_data_type"):
        orig_dt = str(units["original_data_type"])
    f.attrs["original_data_type"] = orig_dt

    sample_meta = extract_sample_info_metadata(units)
    if sample_meta:
        f.attrs["sample_info_json"] = json.dumps(sample_meta)

    if dataset is not None and hasattr(dataset, "history_to_dicts"):
        try:
            f.attrs["history_json"] = json.dumps(dataset.history_to_dicts())
        except Exception:
            pass


def restore_dataset_history(f, ds: Any) -> None:
    """Restore processing history from ``history_json`` attr if supported by ``ds``."""
    if "history_json" in f.attrs and hasattr(ds, "load_history_from_dicts"):
        try:
            raw = f.attrs["history_json"]
            if isinstance(raw, (bytes, np.bytes_)):
                raw = raw.decode("utf-8", errors="replace")
            items = json.loads(raw)
            if isinstance(items, list):
                ds.load_history_from_dicts(items)
        except Exception:
            pass


def create_compressed_dataset(
    group,
    name: str,
    data: Any,
    *,
    units: str | None = None,
    long_name: str | None = None,
    dtype=None,
):
    """Create a gzip+shuffle compressed HDF5 dataset with optional CF metadata attrs."""
    arr = np.asarray(data, dtype=dtype) if dtype is not None else np.asarray(data)
    if arr.ndim > 0 and arr.size > 0:
        dset = group.create_dataset(
            name,
            data=arr,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
    else:
        dset = group.create_dataset(name, data=arr)
    if units is not None:
        dset.attrs["units"] = str(units)
    if long_name is not None:
        dset.attrs["long_name"] = str(long_name)
    return dset


def serialize_chirp_fit(fit: Any) -> str | None:
    """Serialize a :class:`~pymorgan.oneD.chirp.ChirpFit` dataclass to JSON."""
    if fit is None:
        return None
    try:
        raw = asdict(fit)
    except TypeError:
        return None
    payload: dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, np.ndarray):
            payload[k] = v.tolist()
        elif isinstance(v, np.generic):
            payload[k] = v.item()
        elif isinstance(v, tuple):
            payload[k] = list(v)
        else:
            payload[k] = v
    return json.dumps(payload, allow_nan=True)


def deserialize_chirp_fit(raw_json: str | bytes):
    """Deserialize JSON created by :func:`serialize_chirp_fit` back into ``ChirpFit``."""
    from .oneD.chirp import ChirpFit

    if isinstance(raw_json, (bytes, np.bytes_)):
        raw_json = raw_json.decode("utf-8", errors="replace")
    data = json.loads(raw_json)
    return ChirpFit(
        mode=str(data["mode"]),
        coeffs=np.asarray(data["coeffs"], dtype=float),
        fit_wl=np.asarray(data["fit_wl"], dtype=float),
        Pfit=np.asarray(data["Pfit"], dtype=float),
        equation=str(data.get("equation", "Cauchy")),
        n_cauchy_terms=int(data.get("n_cauchy_terms", 6)),
        lambda_ref=float(data.get("lambda_ref", float("nan"))),
        fit_pixels=np.asarray(data.get("fit_pixels", []), dtype=int),
        Dfit=(np.asarray(data["Dfit"], dtype=float) if data.get("Dfit") is not None else None),
        mean_irf_fs=float(data.get("mean_irf_fs", float("nan"))),
        detector=int(data.get("detector", 0)),
        deriv_mask=tuple(bool(x) for x in data.get("deriv_mask", (False, False))),
        use_exp=bool(data.get("use_exp", False)),
        do_2nd_pass=bool(data.get("do_2nd_pass", True)),
        smooth_window=int(data.get("smooth_window", 10)),
        t_min=float(data.get("t_min", float("nan"))),
        t_max=float(data.get("t_max", float("nan"))),
        n_skipped=int(data.get("n_skipped", 0)),
        omega_w=float(data.get("omega_w", float("nan"))),
        gamma_w=float(data.get("gamma_w", float("nan"))),
        source=data.get("source"),
        created=str(data.get("created", "")),
    )

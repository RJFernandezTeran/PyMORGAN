"""Processing provenance, undo/redo history, and Python script export for PyMORGAN."""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .log import get_logger

logger = get_logger(__name__)

# Operations that may alter single-scan array shapes or values in-place.
_SINGLE_SCAN_MUTATING_OPS = frozenset({"crop", "mask_probe_regions", "apply_chirp_correction"})

_2D_PROC_ATTRS = (
    "proc_absFFT_ZPint",
    "proc_ZP_phase",
    "proc_fittedPhase",
    "proc_apod_func",
    "proc_binspecmax",
    "proc_phase_points",
    "proc_phase_coeffs",
    "proc_pump_full",
)

_2D_ANALYSIS_CACHES = (
    "_cls_cache",
    "_ivcls_cache",
    "_nls_cache",
    "_sd_results",
    "_kubo_fit_dataset",
)


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


def _sanitize_value(val: Any) -> Any:
    """Recursively convert values into plain JSON/TOML-friendly Python objects."""
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, np.generic):
        return val.item()
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, Path):
        return str(val)
    if hasattr(val, "value") and not isinstance(val, dict):
        # Enum / StrEnum
        return _sanitize_value(val.value)
    if hasattr(val, "__dataclass_fields__"):
        return {
            k: _sanitize_value(getattr(val, k))
            for k in val.__dataclass_fields__
            if getattr(val, k) is not None
        }
    if isinstance(val, (list, tuple)):
        return [_sanitize_value(item) for item in val if item is not None]
    if isinstance(val, Mapping):
        out: dict[str, Any] = {}
        for k, v in val.items():
            if v is None:
                continue
            out[str(k)] = _sanitize_value(v)
        return out
    return str(val)


def sanitize_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a JSON/TOML-safe parameter dictionary with ``None`` entries omitted."""
    if not params:
        return {}
    cleaned = _sanitize_value(dict(params))
    return cleaned if isinstance(cleaned, dict) else {}


def format_step_label(op: str, params: Mapping[str, Any] | None = None) -> str:
    """Build a concise human-readable label for a processing step."""
    p = params or {}
    if op == "background_correct":
        do_c = p.get("do_correct", True)
        if not do_c:
            return "Disable background correction"
        if "t2" in p:
            return f"Background correction (t2={p['t2']:g})"
        if "reference" in p:
            return f"Background correction (ref #{int(p['reference']) + 1})"
        if "tmin" in p and "tmax" in p:
            return f"Background correction ([{p['tmin']:g}, {p['tmax']:g}])"
        return "Background correction"
    if op in ("remove_scatter", "apply_scatter_removal"):
        method = p.get("method", "local_svd")
        return f"Scatter removal ({method})"
    if op == "clear_scatter_removal":
        return "Clear scatter removal"
    if op == "apply_chirp_correction":
        bh = p.get("boundary_handling")
        return f"Chirp correction ({bh})" if bh else "Chirp correction"
    if op == "subtract_solvent":
        src = p.get("solvent_file")
        if src:
            return f"Subtract solvent ({Path(str(src)).name})"
        return "Subtract solvent"
    if op == "subtract_shockwave":
        return "Subtract shockwave"
    if op == "mask_probe_regions":
        n = len(p.get("ranges", ()))
        return f"Mask probe ({n} region{'s' if n != 1 else ''})"
    if op == "shift_t0":
        return f"Shift t0 ({p.get('t0_shift', 0.0):g})"
    if op == "recalc_average":
        n = len(p.get("scan_indices", ()))
        return f"Recalc. average ({n} scans)"
    if op == "crop":
        return "Crop axes"
    if op == "embed_calibrated_probe":
        return "Embed calibrated probe"
    if op == "process":
        rot = int(p.get("shot_rotation", 0) or 0)
        return f"2D processing (rot={rot})" if rot else "2D processing"
    if op == "subtract_spectrum":
        scale = p.get("scale", 1.0)
        return f"Subtract 2D spectrum (scale={scale:g})"
    if op == "reset_subtraction":
        return "Reset 2D subtraction"
    if op == "set_active_state":
        return f"Select state ({p.get('state_name', '')})"
    if op == "set_active_detector":
        det = p.get("name", p.get("detector", ""))
        return f"Select detector ({det})"
    return op.replace("_", " ").capitalize()


@dataclass
class ProcessingStep:
    """Record of a single semantic processing operation applied to a dataset."""

    op: str
    params: dict[str, Any] = field(default_factory=dict)
    timestamp_utc: str = field(default_factory=_utc_now_iso)
    label: str = ""

    def __post_init__(self) -> None:
        self.op = str(self.op)
        self.params = sanitize_params(self.params)
        if not self.timestamp_utc:
            self.timestamp_utc = _utc_now_iso()
        if not self.label:
            self.label = format_step_label(self.op, self.params)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this step to a plain dictionary safe for JSON and TOML."""
        return {
            "op": self.op,
            "params": copy.deepcopy(self.params),
            "timestamp_utc": self.timestamp_utc,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProcessingStep:
        """Reconstruct a :class:`ProcessingStep` from a serialized dictionary."""
        if "op" not in data or not data["op"]:
            raise ValueError("Processing step dictionary must include a non-empty 'op' field.")
        raw_params = data.get("params") or {}
        if not isinstance(raw_params, Mapping):
            raise ValueError(f"Step 'params' must be a mapping, got {type(raw_params).__name__}.")
        return cls(
            op=str(data["op"]),
            params=dict(raw_params),
            timestamp_utc=str(data.get("timestamp_utc") or _utc_now_iso()),
            label=str(data.get("label") or ""),
        )


def _copy_if_array(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        return np.copy(val)
    return copy.deepcopy(val)


def _capture_snapshot(ds: Any, *, include_single_scans: bool = False) -> dict[str, Any]:
    """Capture mutable working fields for undo/redo without copying 4D scan tensors unnecessarily.

    Note: ``Zss_R`` / ``Zss_C`` / ``bkg_ss`` and 2D ``raw_signal_ss`` / ``raw_signal`` are
    stored by reference (never deep-copied unless a caller explicitly mutates a tensor buffer),
    keeping memory overhead minimal while preserving full restore fidelity across all operations.
    """
    if hasattr(ds, "Zavg_R"):
        snap: dict[str, Any] = {
            "_kind": "1D",
            "Zavg_R": np.copy(ds.Zavg_R) if ds.Zavg_R is not None else None,
            "Zavg_C": np.copy(ds.Zavg_C) if ds.Zavg_C is not None else None,
            "Zstdv": np.copy(ds.Zstdv) if ds.Zstdv is not None else None,
            "_orig_Zstdv": (
                np.copy(ds._orig_Zstdv) if getattr(ds, "_orig_Zstdv", None) is not None else None
            ),
            "delays": np.copy(ds.delays) if ds.delays is not None else None,
            "probe": np.copy(ds.probe) if ds.probe is not None else None,
            "bkg_avg": np.copy(ds.bkg_avg) if ds.bkg_avg is not None else None,
            # Store 4D single-scan buffers by reference so routine ops do not duplicate GBs.
            "Zss_R": getattr(ds, "Zss_R", None),
            "Zss_C": getattr(ds, "Zss_C", None),
            "bkg_ss": getattr(ds, "bkg_ss", None),
            "nscans": getattr(ds, "nscans", np.nan),
            "total_scans": getattr(ds, "total_scans", None),
            "cal_source": getattr(ds, "cal_source", None),
            "cal_level": getattr(ds, "cal_level", None),
            "units": dict(ds.units) if isinstance(getattr(ds, "units", None), dict) else {},
            "mask_ranges": copy.deepcopy(getattr(ds, "mask_ranges", None)),
            "mask_ranges_by_det": copy.deepcopy(getattr(ds, "mask_ranges_by_det", None)),
            "_scatter_removal_params": copy.deepcopy(getattr(ds, "_scatter_removal_params", None)),
            "_scatter_removal_result": getattr(ds, "_scatter_removal_result", None),
            "chirp_fit": getattr(ds, "chirp_fit", None),
            "_bkg_params": copy.deepcopy(getattr(ds, "_bkg_params", None)),
            "_solvent_dataset": getattr(ds, "_solvent_dataset", None),
            "_solvent_dt": _copy_if_array(getattr(ds, "_solvent_dt", None)),
            "_solvent_scale": _copy_if_array(getattr(ds, "_solvent_scale", None)),
            "_shockwave_pixels": _copy_if_array(getattr(ds, "_shockwave_pixels", None)),
            "_shockwave_trace": _copy_if_array(getattr(ds, "_shockwave_trace", None)),
        }
        if include_single_scans and getattr(ds, "n_detectors", 1) > 1:
            # Multi-detector mask_probe_regions writes NaNs in-place unless detached.
            if snap["Zss_R"] is not None:
                snap["Zss_R"] = np.copy(snap["Zss_R"])
            if snap["Zss_C"] is not None:
                snap["Zss_C"] = np.copy(snap["Zss_C"])
            if snap["bkg_ss"] is not None:
                snap["bkg_ss"] = np.copy(snap["bkg_ss"])
        return snap

    if hasattr(ds, "Z_R"):
        proc_attrs = {k: getattr(ds, k) for k in _2D_PROC_ATTRS if hasattr(ds, k)}
        return {
            "_kind": "2D",
            "Z_R": np.copy(ds.Z_R) if ds.Z_R is not None else None,
            "Z_C": np.copy(ds.Z_C) if ds.Z_C is not None else None,
            "pump": np.copy(ds.pump) if ds.pump is not None else None,
            "probe": np.copy(ds.probe) if ds.probe is not None else None,
            "delays": np.copy(ds.delays) if ds.delays is not None else None,
            "active_scan_indices": (
                list(ds.active_scan_indices)
                if getattr(ds, "active_scan_indices", None) is not None
                else None
            ),
            "active_state": getattr(ds, "active_state", None),
            "active_detector": getattr(ds, "active_detector", None),
            "shot_rotation": int(getattr(ds, "shot_rotation", 0) or 0),
            "_is_corrected": bool(getattr(ds, "_is_corrected", False)),
            "nscans": int(getattr(ds, "nscans", 1) or 1),
            "datatype": getattr(ds, "datatype", None),
            "cal_level": getattr(ds, "cal_level", None),
            "cal_source": getattr(ds, "cal_source", None),
            "freq_units": getattr(ds, "freq_units", "cm-1"),
            "units": dict(ds.units) if isinstance(getattr(ds, "units", None), dict) else {},
            "subtracted_spectrum": copy.deepcopy(getattr(ds, "subtracted_spectrum", None)),
            "_subtraction_ref": getattr(ds, "_subtraction_ref", None),
            "_subtraction_scale": getattr(ds, "_subtraction_scale", 1.0),
            "_subtraction_ref_t2_idx": getattr(ds, "_subtraction_ref_t2_idx", None),
            "_bkg_params": copy.deepcopy(getattr(ds, "_bkg_params", None)),
            "_process_params": copy.deepcopy(getattr(ds, "_process_params", None)),
            # Store raw/time-domain tensors by reference so undo/redo avoids large copies.
            "raw_signal": getattr(ds, "raw_signal", None),
            "raw_interferogram": getattr(ds, "raw_interferogram", None),
            "_base_raw_signal": getattr(ds, "_base_raw_signal", None),
            "_base_raw_interferogram": getattr(ds, "_base_raw_interferogram", None),
            "raw_signal_ss": getattr(ds, "raw_signal_ss", None),
            "raw_interferogram_ss": getattr(ds, "raw_interferogram_ss", None),
            "proc_attrs": proc_attrs,
        }

    raise TypeError(f"Unsupported dataset type for snapshot: {type(ds).__name__}")


def _restore_snapshot(ds: Any, snap: Mapping[str, Any]) -> None:
    """Restore mutable working fields from ``snap`` onto ``ds``."""
    kind = snap.get("_kind")
    if kind == "1D":
        ds.Zavg_R = np.copy(snap["Zavg_R"]) if snap["Zavg_R"] is not None else None
        ds.Zavg_C = np.copy(snap["Zavg_C"]) if snap["Zavg_C"] is not None else None
        ds.Zstdv = np.copy(snap["Zstdv"]) if snap["Zstdv"] is not None else None
        ds._orig_Zstdv = np.copy(snap["_orig_Zstdv"]) if snap["_orig_Zstdv"] is not None else None
        ds.delays = np.copy(snap["delays"]) if snap["delays"] is not None else None
        ds.probe = np.copy(snap["probe"]) if snap["probe"] is not None else None
        ds.bkg_avg = np.copy(snap["bkg_avg"]) if snap["bkg_avg"] is not None else None
        ds.Zss_R = snap["Zss_R"]
        ds.Zss_C = snap["Zss_C"]
        ds.bkg_ss = snap["bkg_ss"]
        ds.nscans = snap["nscans"]
        ds.total_scans = snap["total_scans"]
        ds.cal_source = snap["cal_source"]
        ds.cal_level = snap["cal_level"]
        ds.units = dict(snap["units"])
        ds.mask_ranges = copy.deepcopy(snap["mask_ranges"])
        if snap.get("mask_ranges_by_det") is not None:
            ds.mask_ranges_by_det = copy.deepcopy(snap["mask_ranges_by_det"])
        elif hasattr(ds, "mask_ranges_by_det"):
            delattr(ds, "mask_ranges_by_det")
        ds._scatter_removal_params = copy.deepcopy(snap["_scatter_removal_params"])
        ds._scatter_removal_result = snap["_scatter_removal_result"]
        ds.chirp_fit = snap["chirp_fit"]
        ds._bkg_params = copy.deepcopy(snap.get("_bkg_params"))
        ds._solvent_dataset = snap["_solvent_dataset"]
        ds._solvent_dt = _copy_if_array(snap["_solvent_dt"])
        ds._solvent_scale = _copy_if_array(snap["_solvent_scale"])
        ds._shockwave_pixels = _copy_if_array(snap["_shockwave_pixels"])
        ds._shockwave_trace = _copy_if_array(snap["_shockwave_trace"])
        return

    if kind == "2D":
        ds.Z_R = np.copy(snap["Z_R"]) if snap["Z_R"] is not None else None
        ds.Z_C = np.copy(snap["Z_C"]) if snap["Z_C"] is not None else None
        ds.pump = np.copy(snap["pump"]) if snap["pump"] is not None else None
        ds.probe = np.copy(snap["probe"]) if snap["probe"] is not None else None
        ds.delays = np.copy(snap["delays"]) if snap["delays"] is not None else None
        ds.active_scan_indices = (
            list(snap["active_scan_indices"]) if snap["active_scan_indices"] is not None else None
        )
        ds.active_state = snap["active_state"]
        if hasattr(ds, "active_detector") or snap.get("active_detector") is not None:
            ds.active_detector = snap.get("active_detector")
        ds.shot_rotation = int(snap["shot_rotation"])
        ds._is_corrected = bool(snap["_is_corrected"])
        ds.nscans = int(snap["nscans"])
        ds.datatype = snap["datatype"]
        ds.cal_level = snap["cal_level"]
        ds.cal_source = snap["cal_source"]
        ds.freq_units = snap["freq_units"]
        ds.units = dict(snap["units"])
        ds.subtracted_spectrum = copy.deepcopy(snap["subtracted_spectrum"])
        ds._subtraction_ref = snap["_subtraction_ref"]
        ds._subtraction_scale = snap["_subtraction_scale"]
        ds._subtraction_ref_t2_idx = snap["_subtraction_ref_t2_idx"]
        ds._bkg_params = copy.deepcopy(snap.get("_bkg_params"))
        ds._process_params = copy.deepcopy(snap.get("_process_params"))
        ds.raw_signal = snap["raw_signal"]
        ds.raw_interferogram = snap["raw_interferogram"]
        ds._base_raw_signal = snap["_base_raw_signal"]
        ds._base_raw_interferogram = snap["_base_raw_interferogram"]
        ds.raw_signal_ss = snap["raw_signal_ss"]
        ds.raw_interferogram_ss = snap["raw_interferogram_ss"]
        proc_attrs = snap.get("proc_attrs") or {}
        for k in _2D_PROC_ATTRS:
            if k in proc_attrs:
                setattr(ds, k, proc_attrs[k])
            elif hasattr(ds, k):
                delattr(ds, k)
        for cache_attr in _2D_ANALYSIS_CACHES:
            if hasattr(ds, cache_attr):
                delattr(ds, cache_attr)
        return

    raise ValueError(f"Unknown snapshot kind: {kind!r}")


class HistoryManager:
    """Bounded undo/redo stack and semantic processing history for a dataset."""

    def __init__(self, *, max_snapshots: int = 12) -> None:
        self.max_snapshots = max(1, int(max_snapshots))
        self._steps: list[ProcessingStep] = []
        self._undo_stack: list[tuple[list[ProcessingStep], dict[str, Any]]] = []
        self._redo_stack: list[tuple[list[ProcessingStep], dict[str, Any]]] = []
        self._baseline: dict[str, Any] | None = None
        self._guard_depth: int = 0
        self._suppress_depth: int = 0

    @property
    def steps(self) -> list[ProcessingStep]:
        """Ordered list of recorded processing steps."""
        return list(self._steps)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    @property
    def undo_label(self) -> str | None:
        if not self._undo_stack or not self._steps:
            return None
        return self._steps[-1].label

    @property
    def redo_label(self) -> str | None:
        if not self._redo_stack:
            return None
        next_steps, _ = self._redo_stack[-1]
        return next_steps[-1].label if next_steps else "Redo"

    @property
    def is_guarded(self) -> bool:
        return self._guard_depth > 0 or self._suppress_depth > 0

    @contextmanager
    def guard(self) -> Iterator[None]:
        """Re-entrancy guard preventing inner helper calls from recording extra steps."""
        self._guard_depth += 1
        try:
            yield
        finally:
            self._guard_depth -= 1

    @contextmanager
    def suppress(self) -> Iterator[None]:
        """Temporarily disable history recording (e.g. during internal GUI init)."""
        self._suppress_depth += 1
        try:
            yield
        finally:
            self._suppress_depth -= 1

    def ensure_baseline(self, ds: Any) -> None:
        """Capture the initial raw state once before the first recorded mutation."""
        if self._baseline is None and not self._steps:
            self._baseline = _capture_snapshot(ds, include_single_scans=False)

    def record_call(
        self,
        ds: Any,
        op: str,
        fn,
        *,
        params_fn=None,
        params: Mapping[str, Any] | None = None,
        label: str = "",
        coalesce: bool = False,
        should_record=None,
    ):
        """Execute ``fn()`` inside the re-entrancy guard and record a step on success.

        If ``fn()`` raises, no snapshot is pushed and history remains untouched.
        """
        if self.is_guarded:
            return fn()

        self.ensure_baseline(ds)
        do_coalesce = bool(
            coalesce and self._steps and self._steps[-1].op == op and self._undo_stack
        )
        pre_snap = None
        pre_steps: list[ProcessingStep] | None = None
        if not do_coalesce:
            mutates_ss = op in _SINGLE_SCAN_MUTATING_OPS
            pre_snap = _capture_snapshot(ds, include_single_scans=mutates_ss)
            pre_steps = list(self._steps)

        with self.guard():
            result = fn()

        if should_record is not None and not should_record(result):
            return result

        step_params = params_fn(result) if params_fn is not None else (params or {})
        step = ProcessingStep(op=op, params=dict(step_params), label=label)

        if do_coalesce:
            self._steps[-1] = step
        else:
            assert pre_snap is not None and pre_steps is not None
            self._undo_stack.append((pre_steps, pre_snap))
            if len(self._undo_stack) > self.max_snapshots:
                self._undo_stack.pop(0)
            self._steps.append(step)

        self._redo_stack.clear()
        return result

    def commit_coalesced_external(
        self,
        ds: Any,
        op: str,
        params: Mapping[str, Any],
        pre_snap: dict[str, Any] | None = None,
        *,
        coalesce: bool = True,
        label: str = "",
    ) -> None:
        """Record or coalesce a step computed on a worker copy onto ``ds``."""
        step = ProcessingStep(op=op, params=dict(params), label=label)
        do_coalesce = bool(
            coalesce and self._steps and self._steps[-1].op == op and self._undo_stack
        )
        if do_coalesce:
            self._steps[-1] = step
        else:
            snap = pre_snap if pre_snap is not None else _capture_snapshot(ds)
            self._undo_stack.append((list(self._steps), snap))
            if len(self._undo_stack) > self.max_snapshots:
                self._undo_stack.pop(0)
            self._steps.append(step)
        self._redo_stack.clear()

    def update_last_step_param(self, op: str, key: str, value: Any) -> bool:
        """Update a parameter on the latest step if it matches ``op`` (e.g. post-save chirp path)."""
        if not self._steps or self._steps[-1].op != op:
            return False
        new_params = dict(self._steps[-1].params)
        if value is None:
            new_params.pop(key, None)
        else:
            new_params[key] = _sanitize_value(value)
        self._steps[-1] = ProcessingStep(
            op=op,
            params=new_params,
            timestamp_utc=self._steps[-1].timestamp_utc,
            label=format_step_label(op, new_params),
        )
        return True

    def undo(self, ds: Any) -> bool:
        """Undo the most recent step (or coalesced step group) on ``ds``."""
        if not self._undo_stack:
            return False
        curr_snap = _capture_snapshot(ds, include_single_scans=True)
        curr_steps = list(self._steps)
        prev_steps, prev_snap = self._undo_stack.pop()
        self._redo_stack.append((curr_steps, curr_snap))
        with self.suppress():
            _restore_snapshot(ds, prev_snap)
        self._steps = list(prev_steps)
        return True

    def redo(self, ds: Any) -> bool:
        """Redo the most recently undone step on ``ds``."""
        if not self._redo_stack:
            return False
        curr_snap = _capture_snapshot(ds, include_single_scans=True)
        curr_steps = list(self._steps)
        next_steps, next_snap = self._redo_stack.pop()
        self._undo_stack.append((curr_steps, curr_snap))
        if len(self._undo_stack) > self.max_snapshots:
            self._undo_stack.pop(0)
        with self.suppress():
            _restore_snapshot(ds, next_snap)
        self._steps = list(next_steps)
        return True

    def reset_to_raw(self, ds: Any) -> bool:
        """Restore the raw initial baseline state and clear recorded history."""
        if self._baseline is None:
            if not self._steps and not self._undo_stack:
                return False
            self._steps.clear()
            self._undo_stack.clear()
            self._redo_stack.clear()
            return False
        with self.suppress():
            _restore_snapshot(ds, self._baseline)
        self._steps.clear()
        self._undo_stack.clear()
        self._redo_stack.clear()
        return True

    def revert_to(self, ds: Any, step_index: int) -> bool:
        """Revert ``ds`` to the state immediately after ``step_index`` (``-1`` = raw)."""
        if step_index < -1 or step_index >= len(self._steps):
            raise IndexError(f"step_index {step_index} out of range [-1, {len(self._steps) - 1}]")
        if step_index == -1:
            return self.reset_to_raw(ds)
        if step_index == len(self._steps) - 1:
            return False

        target_len = step_index + 1
        # Try stepping back through the bounded undo stack first so redo stays intact.
        steps_back = 0
        for pre_steps, _ in reversed(self._undo_stack):
            steps_back += 1
            if len(pre_steps) == target_len:
                for _ in range(steps_back):
                    self.undo(ds)
                return True

        # Fallback when target step predates the bounded snapshot window: replay from baseline.
        if self._baseline is None:
            raise RuntimeError(
                f"Cannot revert to step {step_index}: snapshot evicted and no baseline saved."
            )
        from .recipe import Recipe

        target_dicts = [s.to_dict() for s in self._steps[:target_len]]
        dim = "2D" if hasattr(ds, "Z_R") else "1D"
        recipe = Recipe(name="revert", dimension=dim, steps=target_dicts)
        with self.suppress():
            _restore_snapshot(ds, self._baseline)
        self._steps.clear()
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._baseline = _capture_snapshot(ds, include_single_scans=False)
        recipe.apply(ds)
        return True

    def to_dicts(self) -> list[dict[str, Any]]:
        """Serialize all recorded steps as a list of plain dictionaries."""
        return [s.to_dict() for s in self._steps]

    def load_dicts(self, items: Sequence[Mapping[str, Any]]) -> None:
        """Replace recorded metadata steps from serialized dictionaries."""
        self._steps = [ProcessingStep.from_dict(item) for item in items]
        self._undo_stack.clear()
        self._redo_stack.clear()

    def copy(self, *, deep: bool = True) -> HistoryManager:
        """Clone this manager without sharing mutable stack state."""
        cloned = HistoryManager(max_snapshots=self.max_snapshots)
        cloned._steps = [ProcessingStep.from_dict(s.to_dict()) for s in self._steps]
        if deep:
            cloned._baseline = copy.deepcopy(self._baseline) if self._baseline else None
            cloned._undo_stack = [
                ([ProcessingStep.from_dict(s.to_dict()) for s in st], copy.deepcopy(sn))
                for st, sn in self._undo_stack
            ]
            cloned._redo_stack = [
                ([ProcessingStep.from_dict(s.to_dict()) for s in st], copy.deepcopy(sn))
                for st, sn in self._redo_stack
            ]
        else:
            cloned._baseline = self._baseline
            cloned._undo_stack = list(self._undo_stack)
            cloned._redo_stack = list(self._redo_stack)
        return cloned


def _format_arg_value(val: Any, *, allow_np: bool = False) -> str:
    if isinstance(val, list) and allow_np:
        return f"np.array({val!r}, dtype=float)"
    if isinstance(val, np.ndarray) and allow_np:
        return f"np.array({val.tolist()!r}, dtype=float)"
    if isinstance(val, np.generic):
        return repr(val.item())
    return repr(val)


def _format_scatter_band_call(band: Mapping[str, Any]) -> str:
    keys = ("p_min", "p_max", "wing_width", "taper_width", "enabled", "label", "rank", "couple_to")
    parts = [f"{k}={band[k]!r}" for k in keys if k in band and band[k] is not None]
    return f"ScatterBand({', '.join(parts)})"


def _settings_diff_dict() -> dict[str, Any]:
    """Return active settings fields that differ from default ``Settings()``."""
    from .settings import Settings, get_settings

    cur = get_settings().to_dict()
    defaults = Settings().to_dict()
    skip = {"recent_dirs", "max_recent_dirs", "default_datadir", "styles_dir"}
    diffs: dict[str, Any] = {}
    for k, v in cur.items():
        if k in skip:
            continue
        if defaults.get(k) != v:
            diffs[k] = tuple(v) if isinstance(v, list) else v
    return diffs


def generate_python_script(
    ds: Any,
    *,
    include_plot: bool = True,
    include_settings: bool = True,
    plot_kwargs: Mapping[str, Any] | None = None,
) -> str:
    """Generate a clean, standalone, Ruff-valid Python script reproducing ``ds``.

    Parameters
    ----------
    ds : Dataset1D or Dataset2D
        Processed dataset whose recorded history will be emitted.
    include_plot : bool, default True
        Whether to append ``pm.apply_style()``, a contour/map plot, and ``pm.show_plots()``.
    include_settings : bool, default True
        Whether to emit ``pm.update_settings(...)`` for non-default settings.
    plot_kwargs : mapping, optional
        Extra keyword arguments passed to ``ds.plot_contour(...)`` (1D) or
        ``ds.plot_map(...)`` (2D).
    """
    is_2d = hasattr(ds, "Z_R")
    steps = list(getattr(ds, "history", []) or [])

    need_np = False
    need_scatter_band = False
    need_load_chirp = False
    need_chirp_fit = False

    for step in steps:
        p = step.params
        if step.op in ("remove_scatter", "apply_scatter_removal") and p.get("bands"):
            need_scatter_band = True
        elif step.op == "apply_chirp_correction":
            if p.get("chirp_file"):
                need_load_chirp = True
            else:
                need_chirp_fit = True
                need_np = True
        elif step.op == "subtract_solvent":
            if isinstance(p.get("dt"), list) or isinstance(p.get("scale"), list):
                need_np = True
        elif step.op == "process" and isinstance(p.get("cal_probe_vector"), list):
            need_np = True

    lines: list[str] = [
        '"""Reproducible PyMORGAN processing script."""',
        "",
    ]

    if need_np:
        lines.append("import numpy as np")
        lines.append("")
    lines.append("import pymorgan as pm")
    chirp_imports = []
    if need_chirp_fit:
        chirp_imports.append("ChirpFit")
    if need_load_chirp:
        chirp_imports.append("load_chirp_fit")
    if chirp_imports:
        lines.append(f"from pymorgan.oneD.chirp import {', '.join(chirp_imports)}")
    if need_scatter_band:
        lines.append("from pymorgan.oneD.scatter import ScatterBand")
    lines.append("")

    if include_settings:
        diffs = _settings_diff_dict()
        if diffs:
            lines.append("# Settings overrides")
            if len(diffs) <= 3:
                kv = ", ".join(f"{k}={v!r}" for k, v in diffs.items())
                lines.append(f"pm.update_settings({kv})")
            else:
                lines.append("pm.update_settings(")
                for k, v in diffs.items():
                    lines.append(f"    {k}={v!r},")
                lines.append(")")
            lines.append("")

    source_str = getattr(ds, "source", None)
    data_type = getattr(ds, "data_type", None) or ("P2DAT" if is_2d else "PDAT")
    if source_str:
        src_expr = repr(str(source_str))
    else:
        src_expr = repr("path/to/dataset.p2dat" if is_2d else "path/to/dataset.pdat")
        lines.append("# NOTE: Original dataset had no source path; replace with your input path.")

    loader_fn = "pm.load_2D" if is_2d else "pm.load_1D"
    load_kw = [f"data_type={data_type!r}"]
    loader_kwargs = getattr(ds, "_loader_kwargs", None)
    if isinstance(loader_kwargs, Mapping):
        for k, v in loader_kwargs.items():
            if v is not None:
                load_kw.append(f"{k}={v!r}")
    lines.append("# Load dataset")
    lines.append(f"ds = {loader_fn}({src_expr}, {', '.join(load_kw)})")
    lines.append("")

    if steps:
        lines.append("# Processing pipeline")
        for idx, step in enumerate(steps, start=1):
            op = step.op
            p = step.params
            if op == "background_correct":
                kw_parts = []
                if "tmin" in p and "tmax" in p:
                    kw_parts.append(f"tmin={p['tmin']!r}")
                    kw_parts.append(f"tmax={p['tmax']!r}")
                if "t2" in p:
                    kw_parts.append(f"t2={p['t2']!r}")
                elif "reference" in p:
                    kw_parts.append(f"reference={p['reference']!r}")
                kw_parts.append(f"do_correct={bool(p.get('do_correct', True))!r}")
                lines.append(f"ds.background_correct({', '.join(kw_parts)})")
            elif op in ("remove_scatter", "apply_scatter_removal"):
                bands = p.get("bands")
                other_kw = {k: v for k, v in p.items() if k != "bands"}
                if bands:
                    lines.append(f"bands_{idx} = [")
                    for b in bands:
                        lines.append(f"    {_format_scatter_band_call(b)},")
                    lines.append("]")
                    kw_str = ", ".join(f"{k}={v!r}" for k, v in other_kw.items())
                    args_str = f"bands=bands_{idx}" + (f", {kw_str}" if kw_str else "")
                    lines.append(f"ds.remove_scatter({args_str})")
                else:
                    kw_str = ", ".join(f"{k}={v!r}" for k, v in other_kw.items())
                    lines.append(f"ds.remove_scatter({kw_str})")
            elif op == "clear_scatter_removal":
                lines.append("ds.clear_scatter_removal()")
            elif op == "apply_chirp_correction":
                c_var = f"chirp_fit_{idx}"
                if p.get("chirp_file"):
                    lines.append(f"{c_var} = load_chirp_fit({p['chirp_file']!r})")
                else:
                    ic = p.get("inline_chirp") or {}
                    lines.append(f"{c_var} = ChirpFit(")
                    lines.append(f"    mode={ic.get('mode', 'manual')!r},")
                    for field_name, vals in (
                        ("coeffs", list(ic.get("coeffs", [0.0]))),
                        ("fit_wl", list(ic.get("fit_wl", []))),
                    ):
                        if len(vals) <= 4:
                            lines.append(f"    {field_name}=np.array({vals!r}, dtype=float),")
                        else:
                            lines.append(f"    {field_name}=np.array(")
                            lines.append("        [")
                            for start_i in range(0, len(vals), 3):
                                chunk = ", ".join(repr(x) for x in vals[start_i : start_i + 3])
                                lines.append(f"            {chunk},")
                            lines.append("        ],")
                            lines.append("        dtype=float,")
                            lines.append("    ),")
                    lines.append("    Pfit=np.empty((0, 1), dtype=float),")
                    lines.append(f"    n_cauchy_terms={int(ic.get('n_cauchy_terms', 6))!r},")
                    lines.append(f"    lambda_ref={float(ic.get('lambda_ref', 500.0))!r},")
                    lines.append(f"    detector={int(ic.get('detector', 0))!r},")
                    lines.append(")")
                extra = []
                if "detector" in p:
                    extra.append(f"detector={p['detector']!r}")
                if "boundary_handling" in p:
                    extra.append(f"boundary_handling={p['boundary_handling']!r}")
                extra_s = (", " + ", ".join(extra)) if extra else ""
                lines.append(f"ds.apply_chirp_correction({c_var}{extra_s})")
            elif op == "subtract_solvent":
                s_var = f"solvent_{idx}"
                s_file = p.get("solvent_file") or "path/to/solvent.pdat"
                s_dt = p.get("solvent_data_type") or data_type
                lines.append(f"{s_var} = pm.load_1D({str(s_file)!r}, data_type={s_dt!r})")
                dt_expr = _format_arg_value(p.get("dt", 0.0), allow_np=True)
                sc_expr = _format_arg_value(p.get("scale", 1.0), allow_np=True)
                lines.append(f"ds.subtract_solvent({s_var}, dt={dt_expr}, scale={sc_expr})")
            elif op == "subtract_shockwave":
                kw_str = ", ".join(f"{k}={v!r}" for k, v in p.items())
                lines.append(f"ds.subtract_shockwave({kw_str})")
            elif op == "mask_probe_regions":
                ranges_tuples = [tuple(r) for r in p.get("ranges", [])]
                det_part = f", detector={p['detector']!r}" if "detector" in p else ""
                lines.append(f"ds.mask_probe_regions({ranges_tuples!r}{det_part})")
            elif op == "shift_t0":
                lines.append(f"ds.shift_t0({p.get('t0_shift', 0.0)!r})")
            elif op == "recalc_average":
                if is_2d:
                    lines.append(f"ds.recalc_average({p.get('scan_indices', [])!r})")
                else:
                    sc = p.get("scan_indices", [])
                    sp = p.get("spectrum", 0)
                    sm = p.get("slowmod", 0)
                    nw = p.get("noise_weighted", True)
                    lines.append(
                        f"ds.recalc_average({sc!r}, spectrum={sp!r}, slowmod={sm!r}, noise_weighted={nw!r})"
                    )
            elif op == "crop":
                kw_str = ", ".join(f"{k}={v!r}" for k, v in p.items())
                lines.append(f"ds.crop({kw_str})")
            elif op == "embed_calibrated_probe":
                csv_p = p.get("csv_path")
                backup = bool(p.get("backup_original", True))
                if csv_p:
                    lines.append(
                        f"ds.embed_calibrated_probe({csv_p!r}, backup_original={backup!r})"
                    )
                else:
                    lines.append(f"ds.embed_calibrated_probe(backup_original={backup!r})")
            elif op == "process":
                proc_parts = []
                for k, v in p.items():
                    if k == "cal_probe_vector" and isinstance(v, list):
                        proc_parts.append(f"{k}=np.array({v!r}, dtype=float)")
                    else:
                        proc_parts.append(f"{k}={v!r}")
                if len(proc_parts) <= 2:
                    lines.append(f"ds.process({', '.join(proc_parts)})")
                else:
                    lines.append("ds.process(")
                    for part in proc_parts:
                        lines.append(f"    {part},")
                    lines.append(")")
            elif op == "subtract_spectrum":
                r_var = f"ref_ds_{idx}"
                r_file = p.get("ref_file") or "path/to/reference.p2dat"
                r_dt = p.get("ref_data_type") or data_type
                lines.append(f"{r_var} = pm.load_2D({str(r_file)!r}, data_type={r_dt!r})")
                extra = f", scale={p.get('scale', 1.0)!r}"
                if "ref_t2_idx" in p:
                    extra += f", ref_t2_idx={p['ref_t2_idx']!r}"
                lines.append(f"ds.subtract_spectrum({r_var}{extra})")
            elif op == "reset_subtraction":
                lines.append("ds.reset_subtraction()")
            elif op == "set_active_state":
                lines.append(f"ds.set_active_state({p.get('state_name', '')!r})")
            elif op == "set_active_detector":
                det_name = p.get("name", p.get("detector", ""))
                lines.append(f"ds.set_active_detector({det_name!r})")
        lines.append("")

    if include_plot:
        lines.append("# Visualization")
        lines.append("pm.apply_style()")
        pkw = dict(plot_kwargs or {})
        if is_2d:
            default_t2 = (
                float(ds.delays[0])
                if getattr(ds, "delays", None) is not None and len(ds.delays) > 0
                else 0.0
            )
            t2_val = pkw.pop("t2", default_t2)
            kw_s = ", ".join(f"{k}={v!r}" for k, v in pkw.items())
            args_s = f"{t2_val!r}, {kw_s}" if kw_s else f"{t2_val!r}"
            lines.append(f"ds.plot_map({args_s})")
        else:
            kw_s = ", ".join(f"{k}={v!r}" for k, v in pkw.items())
            lines.append(f"ds.plot_contour({kw_s})")
        lines.append("pm.show_plots()")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"

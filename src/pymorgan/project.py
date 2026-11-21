"""Portable TOML session/project files (``.pmproj``) for PyMORGAN."""

from __future__ import annotations

import copy
import os
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime
from os import PathLike
from pathlib import Path
from typing import Any

import tomlkit

from .log import get_logger
from .provenance import sanitize_params
from .recipe import _AUX_PATH_KEYS, Recipe, _toml_clean, resolve_aux_path

logger = get_logger(__name__)

PROJECT_SCHEMA_VERSION = "1.0"

RelocateCallback = Callable[[str, Path], Path | str | None]


def _select_combo_datatype(combo: Any, dt: str) -> None:
    """Select ``dt`` on a PyQt datatype combobox without emitting change signals."""
    if combo is None or not dt:
        return
    combo.blockSignals(True)
    try:
        try:
            from .gui.mw_common import set_combo_datatype

            set_combo_datatype(combo, dt)
        except Exception:
            idx = combo.findData(dt) if hasattr(combo, "findData") else -1
            if idx < 0 and hasattr(combo, "findText"):
                idx = combo.findText(dt)
            if idx >= 0 and hasattr(combo, "setCurrentIndex"):
                combo.setCurrentIndex(idx)
    finally:
        combo.blockSignals(False)


def _encode_portable_path(
    raw: str | PathLike | None,
    base_dir: Path | None,
) -> tuple[str | None, str | None]:
    """Return ``(portable_rel_or_abs, resolved_abs_str)`` for a path."""
    if not raw:
        return None, None
    p = Path(str(raw))
    try:
        abs_str = str(p.resolve()) if p.is_absolute() or p.exists() else str(p)
    except OSError:
        abs_str = str(p)

    if base_dir is None:
        return abs_str, abs_str

    try:
        rel = os.path.relpath(abs_str, str(base_dir.resolve()))
    except (OSError, ValueError):
        return abs_str, abs_str

    rel_path = Path(rel)
    up_count = sum(1 for part in rel_path.parts if part == "..")
    if up_count <= 4:
        return rel_path.as_posix(), abs_str
    return abs_str, abs_str


def resolve_session_path(
    entry: str | PathLike | None,
    abs_fallback: str | PathLike | None = None,
    *,
    base_dir: Path | None = None,
    role: str = "file",
    relocate_callback: RelocateCallback | None = None,
    warnings_out: list[str] | None = None,
) -> Path | None:
    """Resolve a project file/dir reference using relative first, then abs fallback, then callback."""
    candidates: list[Path] = []
    if entry:
        ep = Path(str(entry))
        if ep.is_absolute():
            candidates.append(ep)
        elif base_dir is not None:
            candidates.append(base_dir / ep)
        else:
            candidates.append(ep)

    if abs_fallback:
        ap = Path(str(abs_fallback))
        if not any(str(c) == str(ap) for c in candidates):
            candidates.append(ap)

    if not candidates:
        return None

    for cand in candidates:
        try:
            if cand.exists():
                return cand
        except OSError:
            continue

    missing_ref = candidates[0]
    if relocate_callback is not None:
        try:
            choice = relocate_callback(role, missing_ref)
        except Exception as exc:
            logger.warning("Relocate callback failed for %s (%s): %s", role, missing_ref, exc)
            choice = None
        if choice:
            relocated = Path(str(choice))
            if relocated.exists():
                return relocated

    msg = f"Missing {role}: {missing_ref}"
    logger.warning(msg)
    if warnings_out is not None:
        warnings_out.append(msg)
    return None


def _encode_steps_portable(
    steps: Sequence[Mapping[str, Any]],
    base_dir: Path | None,
    *,
    prev_base_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Encode processing step dicts with portable relative + ``*_abs`` paths."""
    encoded: list[dict[str, Any]] = []
    for raw in steps:
        st = copy.deepcopy(dict(raw))
        params = sanitize_params(dict(st.get("params") or {}))
        for key in _AUX_PATH_KEYS:
            abs_raw = params.get(f"{key}_abs")
            val = params.get(key)
            chosen = abs_raw or val
            if chosen:
                cand_p = Path(str(chosen))
                if not cand_p.is_absolute() and prev_base_dir is not None:
                    chosen = str((prev_base_dir / cand_p).resolve())
                rel_val, abs_val = _encode_portable_path(str(chosen), base_dir)
                if rel_val:
                    params[key] = rel_val
                if abs_val:
                    params[f"{key}_abs"] = abs_val
        item: dict[str, Any] = {
            "op": str(st["op"]),
            "enabled": bool(st.get("enabled", True)),
            "params": params,
        }
        if st.get("label"):
            item["label"] = str(st["label"])
        if st.get("timestamp_utc"):
            item["timestamp_utc"] = str(st["timestamp_utc"])
        encoded.append(item)
    return encoded


def _resolve_steps_aux_paths(
    steps: Sequence[Mapping[str, Any]],
    *,
    dim: str,
    base_dir: Path | None,
    ds_source: str | PathLike | None,
    relocate_callback: RelocateCallback | None,
    warnings_out: list[str],
) -> list[dict[str, Any]]:
    """Resolve auxiliary paths inside ``steps`` before replay, handling missing files cleanly."""
    resolved_steps: list[dict[str, Any]] = []
    for raw in steps:
        st = copy.deepcopy(dict(raw))
        if not st.get("enabled", True):
            resolved_steps.append(st)
            continue
        params = dict(st.get("params") or {})
        skip_step = False
        for key in _AUX_PATH_KEYS:
            val = params.get(key)
            if not val:
                continue
            abs_fb = params.get(f"{key}_abs")
            cand = resolve_aux_path(
                val,
                base_dir=base_dir,
                dataset_source=ds_source,
                fallback_abs=abs_fb,
            )
            if cand.exists():
                params[key] = str(cand)
                continue

            # Optional inline chirp fallback exists without disk file
            if key == "chirp_file" and isinstance(params.get("inline_chirp"), Mapping):
                params.pop("chirp_file", None)
                params.pop("chirp_file_abs", None)
                continue

            relocated = resolve_session_path(
                val,
                abs_fb,
                base_dir=base_dir,
                role=f"{dim}.step.{st.get('op')}.{key}",
                relocate_callback=relocate_callback,
                warnings_out=warnings_out,
            )
            if relocated is not None:
                params[key] = str(relocated)
            else:
                skip_step = True
                break

        if not skip_step:
            st["params"] = params
            resolved_steps.append(st)
    return resolved_steps


def _safe_table_from_dict(mapping: Mapping[str, Any]) -> Any:
    """Recursively convert dicts/lists into clean tomlkit objects, omitting ``None``."""
    tbl = tomlkit.table()
    for k, v in mapping.items():
        if v is None:
            continue
        if isinstance(v, Mapping):
            tbl[str(k)] = _safe_table_from_dict(v)
        elif k == "steps" and isinstance(v, list):
            if v:
                aot = tomlkit.aot()
                for st in v:
                    aot.append(_safe_table_from_dict(st))
                tbl["steps"] = aot
            else:
                tbl["steps"] = []
        else:
            cleaned = _toml_clean(v)
            if cleaned is not None:
                tbl[str(k)] = cleaned
    return tbl


@dataclass
class ProjectSession:
    """Serializable full workspace session (``.pmproj``) for PyMORGAN."""

    schema_version: str = PROJECT_SCHEMA_VERSION
    root_dir: str | None = None
    root_dir_abs: str | None = None
    active_tab: int = 0
    oneD: dict[str, Any] = field(default_factory=dict)
    twoD: dict[str, Any] = field(default_factory=dict)
    settings_overrides: dict[str, Any] = field(default_factory=dict)
    saved_utc: str = ""
    project_path: Path | None = field(default=None, repr=False, compare=False)

    def to_dict(self, *, base_dir: str | PathLike | None = None) -> dict[str, Any]:
        """Serialize session to a TOML/JSON-ready dictionary."""
        bdir = Path(base_dir) if base_dir is not None else None
        prev_bdir = self.project_path.parent if self.project_path is not None else None

        def _prefer_abs(raw_rel: Any, raw_abs: Any) -> str | None:
            chosen = raw_abs or raw_rel
            if not chosen:
                return None
            p = Path(str(chosen))
            if not p.is_absolute() and prev_bdir is not None:
                return str(prev_bdir / p)
            return str(chosen)

        out: dict[str, Any] = {
            "schema_version": str(self.schema_version),
            "saved_utc": self.saved_utc or datetime.now(UTC).replace(microsecond=0).isoformat(),
            "active_tab": int(self.active_tab),
        }

        root_src = _prefer_abs(self.root_dir, self.root_dir_abs)
        if root_src:
            rel_root, abs_root = _encode_portable_path(root_src, bdir)
            if rel_root:
                out["root_dir"] = rel_root
            if abs_root:
                out["root_dir_abs"] = abs_root

        one_d = copy.deepcopy(dict(self.oneD or {}))
        for pkey in ("path", "ss_abs_path", "ss_em_path"):
            raw_v = _prefer_abs(one_d.get(pkey), one_d.get(f"{pkey}_abs"))
            if raw_v:
                rel_p, abs_p = _encode_portable_path(raw_v, bdir)
                if rel_p:
                    one_d[pkey] = rel_p
                if abs_p:
                    one_d[f"{pkey}_abs"] = abs_p
        if "steps" in one_d:
            one_d["steps"] = _encode_steps_portable(one_d["steps"], bdir, prev_base_dir=prev_bdir)
        if one_d:
            out["oneD"] = _toml_clean(one_d)

        two_d = copy.deepcopy(dict(self.twoD or {}))
        for pkey in ("path", "ftir_overlay_path"):
            raw_v = _prefer_abs(two_d.get(pkey), two_d.get(f"{pkey}_abs"))
            if raw_v:
                rel_p, abs_p = _encode_portable_path(raw_v, bdir)
                if rel_p:
                    two_d[pkey] = rel_p
                if abs_p:
                    two_d[f"{pkey}_abs"] = abs_p
        if "steps" in two_d:
            two_d["steps"] = _encode_steps_portable(two_d["steps"], bdir, prev_base_dir=prev_bdir)
        if two_d:
            out["twoD"] = _toml_clean(two_d)

        if self.settings_overrides:
            ov = {k: v for k, v in self.settings_overrides.items() if k != "recent_dirs"}
            if ov:
                out["settings_overrides"] = _toml_clean(ov)

        return out

    def to_toml(self, *, base_dir: str | PathLike | None = None) -> str:
        """Render this project session as commented TOML."""
        data = self.to_dict(base_dir=base_dir)
        doc = tomlkit.document()
        doc.add(tomlkit.comment("PyMORGAN Project Session (.pmproj)"))
        doc.add(tomlkit.comment(f"Schema v{self.schema_version}"))
        doc.add(tomlkit.nl())

        for key in ("schema_version", "saved_utc", "active_tab", "root_dir", "root_dir_abs"):
            if key in data and data[key] is not None:
                doc[key] = data[key]

        doc.add(tomlkit.nl())
        for sec in ("oneD", "twoD", "settings_overrides"):
            if sec in data and isinstance(data[sec], Mapping) and data[sec]:
                doc[sec] = _safe_table_from_dict(data[sec])
                doc.add(tomlkit.nl())

        return tomlkit.dumps(doc)

    def save(self, path: str | PathLike) -> Path:
        """Save project session to ``path`` (typically ending in ``.pmproj``)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.saved_utc = datetime.now(UTC).replace(microsecond=0).isoformat()
        target.write_text(self.to_toml(base_dir=target.parent), encoding="utf-8")
        self.project_path = target
        return target

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProjectSession:
        """Construct a :class:`ProjectSession` from a parsed TOML dictionary."""
        return cls(
            schema_version=str(data.get("schema_version", PROJECT_SCHEMA_VERSION)),
            root_dir=data.get("root_dir") or None,
            root_dir_abs=data.get("root_dir_abs") or None,
            active_tab=int(data.get("active_tab", 0)),
            oneD=dict(data.get("oneD") or {}),
            twoD=dict(data.get("twoD") or {}),
            settings_overrides=dict(data.get("settings_overrides") or {}),
            saved_utc=str(data.get("saved_utc", "")),
        )

    @classmethod
    def load(cls, path: str | PathLike) -> ProjectSession:
        """Load a :class:`ProjectSession` from a ``.pmproj`` file."""
        p = Path(path)
        raw = tomlkit.parse(p.read_text(encoding="utf-8")).unwrap()
        session = cls.from_dict(raw)
        session.project_path = p
        return session

    @classmethod
    def from_main_window(
        cls,
        win: Any,
        *,
        base_dir: str | PathLike | None = None,
    ) -> ProjectSession:
        """Snapshot full interactive workspace state from a ``MainWindow`` instance."""
        import pymorgan as pm

        bdir = Path(base_dir) if base_dir is not None else None
        root_text = win._rootdir_text() if hasattr(win, "_rootdir_text") else ""
        root_rel, root_abs = _encode_portable_path(root_text, bdir)
        active_tab = int(win.MainTabs.currentIndex()) if hasattr(win, "MainTabs") else 0

        one_d: dict[str, Any] = {}
        ds1 = getattr(win, "dataset", None)
        if ds1 is not None:
            src1 = getattr(ds1, "source", None)
            rel1, abs1 = _encode_portable_path(src1, bdir)
            if rel1:
                one_d["path"] = rel1
            if abs1:
                one_d["path_abs"] = abs1
            one_d["data_type"] = getattr(ds1, "data_type", None) or (
                win._current_datatype() if hasattr(win, "_current_datatype") else "PDAT"
            )
            det_spin = getattr(getattr(win, "plot_controls", None), "detector_spin", None)
            one_d["detector"] = max(0, int(det_spin.value()) - 1) if det_spin is not None else 0
            one_d["slowmod"] = int(getattr(win, "_mess_slowmod", 0) or 0)
            one_d["spectrum"] = int(getattr(win, "_mess_spectrum", 0) or 0)
            one_d["steps"] = ds1.history_to_dicts() if hasattr(ds1, "history_to_dicts") else []

            ss_abs = getattr(win, "_ss_abs_path", None)
            if ss_abs:
                r, a = _encode_portable_path(ss_abs, bdir)
                one_d["ss_abs_path"] = r
                one_d["ss_abs_path_abs"] = a
            ss_em = getattr(win, "_ss_em_path", None)
            if ss_em:
                r, a = _encode_portable_path(ss_em, bdir)
                one_d["ss_em_path"] = r
                one_d["ss_em_path_abs"] = a

            pc1 = getattr(win, "plot_controls", None)
            if pc1 is not None:
                vlim: dict[str, Any] = {}
                if hasattr(pc1, "xlim"):
                    xlim = pc1.xlim()
                    if xlim is not None:
                        vlim["xlim"] = [float(xlim[0]), float(xlim[1])]
                if hasattr(pc1, "ylim"):
                    ylim = pc1.ylim()
                    if ylim is not None:
                        vlim["ylim"] = [float(ylim[0]), float(ylim[1])]
                if hasattr(pc1, "zlimits"):
                    zmin, zmax = pc1.zlimits()
                    if zmin is not None:
                        vlim["zmin"] = float(zmin)
                    if zmax is not None:
                        vlim["zmax"] = float(zmax)
                one_d["detector"] = int(pc1.detector)
                one_d["view_limits"] = vlim

        two_d: dict[str, Any] = {}
        ds2 = getattr(win, "twoD_dataset", None)
        if ds2 is not None:
            src2 = getattr(ds2, "source", None)
            rel2, abs2 = _encode_portable_path(src2, bdir)
            if rel2:
                two_d["path"] = rel2
            if abs2:
                two_d["path_abs"] = abs2
            two_d["data_type"] = getattr(ds2, "data_type", None) or (
                win._current_twoD_datatype() if hasattr(win, "_current_twoD_datatype") else "P2DAT"
            )
            if getattr(ds2, "active_state", None) is not None:
                two_d["active_state"] = str(ds2.active_state)
            active_det = getattr(ds2, "active_detector", None)
            if active_det is not None:
                two_d["active_detector"] = str(active_det)
            t2_lst = getattr(win, "twoD_t2delay_lst", None)
            if t2_lst is not None and t2_lst.currentIndex().isValid():
                two_d["active_t2_index"] = int(t2_lst.currentIndex().row())
            two_d["steps"] = ds2.history_to_dicts() if hasattr(ds2, "history_to_dicts") else []

            ov_path = getattr(win, "_twoD_overlay_path", None)
            if ov_path:
                r, a = _encode_portable_path(ov_path, bdir)
                two_d["ftir_overlay_path"] = r
                two_d["ftir_overlay_path_abs"] = a

            pc2 = getattr(win, "twoD_plot_controls", None)
            if pc2 is not None:
                vlim2: dict[str, Any] = {}
                if hasattr(pc2, "pump_lim"):
                    pl = pc2.pump_lim()
                    if pl is not None:
                        vlim2["pump_lim"] = [float(pl[0]), float(pl[1])]
                if hasattr(pc2, "probe_lim"):
                    prl = pc2.probe_lim()
                    if prl is not None:
                        vlim2["probe_lim"] = [float(prl[0]), float(prl[1])]
                if hasattr(pc2, "zlimits"):
                    zmin, zmax = pc2.zlimits()
                    if zmin is not None:
                        vlim2["zmin"] = float(zmin)
                    if zmax is not None:
                        vlim2["zmax"] = float(zmax)
                if hasattr(pc2, "pump_descending"):
                    vlim2["pump_descending"] = bool(pc2.pump_descending())
                if hasattr(pc2, "probe_descending"):
                    vlim2["probe_descending"] = bool(pc2.probe_descending())
                two_d["view_limits"] = vlim2

        cur_s = pm.get_settings().to_dict()
        def_s = pm.Settings().to_dict()
        overrides = {
            k: v
            for k, v in cur_s.items()
            if k not in ("recent_dirs", "max_recent_dirs") and def_s.get(k) != v
        }

        return cls(
            root_dir=root_rel,
            root_dir_abs=root_abs,
            active_tab=active_tab,
            oneD=one_d,
            twoD=two_d,
            settings_overrides=overrides,
            saved_utc=datetime.now(UTC).replace(microsecond=0).isoformat(),
            project_path=getattr(win, "_current_project_path", None),
        )

    def apply_to_main_window(
        self,
        win: Any,
        *,
        relocate_callback: RelocateCallback | None = None,
    ) -> list[str]:
        """Restore this session onto ``win``, returning any non-fatal warning strings."""
        import pymorgan as pm

        warnings_out: list[str] = []
        base_dir = self.project_path.parent if self.project_path is not None else None

        # 1. Settings overrides
        if self.settings_overrides:
            clean_ov = {
                k: v
                for k, v in self.settings_overrides.items()
                if k not in ("recent_dirs", "max_recent_dirs")
            }
            if clean_ov:
                try:
                    pm.update_settings(**clean_ov)
                    if hasattr(win, "_apply_settings_to_ui"):
                        win._apply_settings_to_ui()
                    if getattr(win, "plot_controls", None) is not None:
                        win.plot_controls.sync_from_settings()
                    if getattr(win, "twoD_plot_controls", None) is not None:
                        win.twoD_plot_controls.sync_from_settings()
                except Exception as exc:
                    warnings_out.append(f"Could not apply settings overrides: {exc}")

        # 2. Root directory
        if self.root_dir or self.root_dir_abs:
            resolved_root = resolve_session_path(
                self.root_dir,
                self.root_dir_abs,
                base_dir=base_dir,
                role="root_dir",
                relocate_callback=relocate_callback,
                warnings_out=warnings_out,
            )
            if resolved_root is not None and hasattr(win, "RootDir_field"):
                win.RootDir_field.setText(str(resolved_root))
                if hasattr(win, "_on_rootdir_changed"):
                    win._on_rootdir_changed()

        # 3. 1D Dataset + History + Overlays + View Limits
        if self.oneD and (self.oneD.get("path") or self.oneD.get("path_abs")):
            ds1_path = resolve_session_path(
                self.oneD.get("path"),
                self.oneD.get("path_abs"),
                base_dir=base_dir,
                role="oneD.path",
                relocate_callback=relocate_callback,
                warnings_out=warnings_out,
            )
            if ds1_path is not None:
                dt1 = self.oneD.get("data_type")
                if dt1 and hasattr(win, "PP_datatype_cbx"):
                    _select_combo_datatype(win.PP_datatype_cbx, str(dt1))
                    if (
                        hasattr(win, "refresh_dataset_list")
                        and hasattr(win, "_rootdir_text")
                        and win._rootdir_text()
                    ):
                        win.refresh_dataset_list()
                win._restoring_session = True
                try:
                    has_explicit_state = (
                        self.oneD.get("slowmod") is not None
                        or self.oneD.get("spectrum") is not None
                    )
                    if self.oneD.get("slowmod") is not None:
                        win._mess_slowmod = int(self.oneD["slowmod"])
                    if self.oneD.get("spectrum") is not None:
                        win._mess_spectrum = int(self.oneD["spectrum"])
                    win.load_path(
                        str(ds1_path),
                        data_type=str(dt1) if dt1 else None,
                        reset_selection=not has_explicit_state,
                        force=True,
                    )
                    ds1 = getattr(win, "dataset", None)
                    if ds1 is not None:
                        steps1 = self.oneD.get("steps") or []
                        if steps1:
                            resolved_steps1 = _resolve_steps_aux_paths(
                                steps1,
                                dim="oneD",
                                base_dir=base_dir,
                                ds_source=ds1_path,
                                relocate_callback=relocate_callback,
                                warnings_out=warnings_out,
                            )
                            ds1.reset_to_raw()
                            if resolved_steps1:
                                Recipe(
                                    name=f"session:{self.project_path.name if self.project_path else '1D'}",
                                    dimension="1D",
                                    data_type=str(dt1 or "auto"),
                                    steps=resolved_steps1,
                                ).apply(ds1, base_dir=base_dir)
                        det_idx = self.oneD.get("detector")
                        det_spin = getattr(
                            getattr(win, "plot_controls", None), "detector_spin", None
                        )
                        if det_idx is not None and det_spin is not None:
                            det_spin.blockSignals(True)
                            det_spin.setValue(int(det_idx) + 1)
                            det_spin.blockSignals(False)
                        if hasattr(win, "_sync_oneD_ui_after_history_change"):
                            win._sync_oneD_ui_after_history_change()
                        self._restore_oneD_overlays_and_views(
                            win, base_dir, relocate_callback, warnings_out
                        )
                finally:
                    win._restoring_session = False

        # 4. 2D Dataset + History + Detector/State + Overlay + View Limits
        if self.twoD and (self.twoD.get("path") or self.twoD.get("path_abs")):
            ds2_path = resolve_session_path(
                self.twoD.get("path"),
                self.twoD.get("path_abs"),
                base_dir=base_dir,
                role="twoD.path",
                relocate_callback=relocate_callback,
                warnings_out=warnings_out,
            )
            if ds2_path is not None:
                dt2 = self.twoD.get("data_type")
                if dt2 and hasattr(win, "twoD_datatype_cbx"):
                    _select_combo_datatype(win.twoD_datatype_cbx, str(dt2))
                    if hasattr(win, "_set_twoD_phasing_tab_visible"):
                        win._set_twoD_phasing_tab_visible(str(dt2) not in ("P2DAT", "RAL_Proc"))
                    if (
                        hasattr(win, "twoD_refresh_dataset_list")
                        and hasattr(win, "_rootdir_text")
                        and win._rootdir_text()
                    ):
                        win.twoD_refresh_dataset_list()
                win._restoring_session = True
                try:
                    win.twoD_load_path(
                        str(ds2_path),
                        data_type=str(dt2) if dt2 else None,
                        force=True,
                    )
                    ds2 = getattr(win, "twoD_dataset", None)
                    if ds2 is not None:
                        st_name = self.twoD.get("active_state")
                        det_name = self.twoD.get("active_detector")
                        steps2 = self.twoD.get("steps") or []
                        hist2 = getattr(ds2, "_history", None)

                        if steps2:
                            resolved_steps2 = _resolve_steps_aux_paths(
                                steps2,
                                dim="twoD",
                                base_dir=base_dir,
                                ds_source=ds2_path,
                                relocate_callback=relocate_callback,
                                warnings_out=warnings_out,
                            )
                            ds2.reset_to_raw()
                            if resolved_steps2:
                                Recipe(
                                    name=f"session:{self.project_path.name if self.project_path else '2D'}",
                                    dimension="2D",
                                    data_type=str(dt2 or "auto"),
                                    steps=resolved_steps2,
                                ).apply(ds2, base_dir=base_dir)

                        suppress_ctx = hist2.suppress() if hist2 is not None else nullcontext()
                        with suppress_ctx:
                            if (
                                st_name
                                and getattr(ds2, "active_state", None) != str(st_name)
                                and hasattr(ds2, "set_active_state")
                            ):
                                try:
                                    ds2.set_active_state(str(st_name))
                                except Exception:
                                    pass
                            if (
                                det_name
                                and getattr(ds2, "active_detector", None) != str(det_name)
                                and hasattr(ds2, "set_active_detector")
                            ):
                                try:
                                    ds2.set_active_detector(str(det_name))
                                except Exception:
                                    pass
                        t2_idx = self.twoD.get("active_t2_index")
                        if t2_idx is not None and hasattr(win, "twoD_t2delay_lst"):
                            lst = win.twoD_t2delay_lst
                            model = lst.model() if hasattr(lst, "model") else None
                            if model is not None and 0 <= int(t2_idx) < model.rowCount():
                                lst.setCurrentIndex(model.index(int(t2_idx), 0))
                        if hasattr(win, "_sync_twoD_ui_after_history_change"):
                            win._sync_twoD_ui_after_history_change()
                        self._restore_twoD_overlay_and_views(
                            win, base_dir, relocate_callback, warnings_out
                        )
                finally:
                    win._restoring_session = False

        # 5. Active tab + project book-keeping
        if hasattr(win, "MainTabs"):
            tab_idx = max(0, min(int(self.active_tab), win.MainTabs.count() - 1))
            win.MainTabs.setCurrentIndex(tab_idx)

        win._current_project_path = self.project_path
        if hasattr(win, "_set_project_dirty"):
            win._set_project_dirty(False)
        if hasattr(win, "_update_edit_menu_state"):
            win._update_edit_menu_state()
        return warnings_out

    def _restore_oneD_overlays_and_views(
        self,
        win: Any,
        base_dir: Path | None,
        relocate_callback: RelocateCallback | None,
        warnings_out: list[str],
    ) -> None:
        ss_abs = self.oneD.get("ss_abs_path") or self.oneD.get("ss_abs_path_abs")
        if ss_abs:
            p = resolve_session_path(
                self.oneD.get("ss_abs_path"),
                self.oneD.get("ss_abs_path_abs"),
                base_dir=base_dir,
                role="oneD.ss_abs_path",
                relocate_callback=relocate_callback,
                warnings_out=warnings_out,
            )
            if p is not None and hasattr(win, "_load_steady_state"):
                try:
                    win._load_steady_state("abs", path=str(p))
                except TypeError:
                    win._ss_abs_path = str(p)

        ss_em = self.oneD.get("ss_em_path") or self.oneD.get("ss_em_path_abs")
        if ss_em:
            p = resolve_session_path(
                self.oneD.get("ss_em_path"),
                self.oneD.get("ss_em_path_abs"),
                base_dir=base_dir,
                role="oneD.ss_em_path",
                relocate_callback=relocate_callback,
                warnings_out=warnings_out,
            )
            if p is not None and hasattr(win, "_load_steady_state"):
                try:
                    win._load_steady_state("em", path=str(p))
                except TypeError:
                    win._ss_em_path = str(p)

        vlim = self.oneD.get("view_limits") or {}
        pc1 = getattr(win, "plot_controls", None)
        if pc1 is not None and vlim:
            if "xlim" in vlim and hasattr(pc1, "x_min") and hasattr(pc1, "x_max"):
                try:
                    x0, x1 = float(vlim["xlim"][0]), float(vlim["xlim"][1])
                    if getattr(pc1, "_x_scaled", False):
                        x0 /= 1000.0
                        x1 /= 1000.0
                    for w, v in ((pc1.x_min, x0), (pc1.x_max, x1)):
                        w.blockSignals(True)
                        w.setValue(v)
                        w.blockSignals(False)
                except Exception:
                    pass
            if "ylim" in vlim and hasattr(pc1, "y_min") and hasattr(pc1, "y_max"):
                try:
                    for w, v in (
                        (pc1.y_min, float(vlim["ylim"][0])),
                        (pc1.y_max, float(vlim["ylim"][1])),
                    ):
                        w.blockSignals(True)
                        w.setValue(v)
                        w.blockSignals(False)
                except Exception:
                    pass
            if "zmin" in vlim and hasattr(pc1, "z_min"):
                try:
                    pc1.z_min.blockSignals(True)
                    pc1.z_min.setValue(float(vlim["zmin"]))
                    pc1.z_min.blockSignals(False)
                except Exception:
                    pass
            if "zmax" in vlim and hasattr(pc1, "z_max"):
                try:
                    pc1.z_max.blockSignals(True)
                    pc1.z_max.setValue(float(vlim["zmax"]))
                    pc1.z_max.blockSignals(False)
                except Exception:
                    pass
        if hasattr(win, "_preview_contour"):
            win._preview_contour()

    def _restore_twoD_overlay_and_views(
        self,
        win: Any,
        base_dir: Path | None,
        relocate_callback: RelocateCallback | None,
        warnings_out: list[str],
    ) -> None:
        ov_raw = self.twoD.get("ftir_overlay_path") or self.twoD.get("ftir_overlay_path_abs")
        if ov_raw:
            p = resolve_session_path(
                self.twoD.get("ftir_overlay_path"),
                self.twoD.get("ftir_overlay_path_abs"),
                base_dir=base_dir,
                role="twoD.ftir_overlay_path",
                relocate_callback=relocate_callback,
                warnings_out=warnings_out,
            )
            if p is not None:
                try:
                    import pymorgan as pm

                    win._twoD_overlay_path = str(p)
                    if p.suffix.lower() == ".pdat":
                        ds_ov = pm.load_1D(p)
                        win._twoD_overlay = (ds_ov.probe, ds_ov.Z.mean(axis=0))
                    else:
                        spec = pm.load_spectrum(p, kind="FTIR")
                        win._twoD_overlay = (spec.x, spec.y)
                    if hasattr(win, "twoD_include_ftir_chk"):
                        win.twoD_include_ftir_chk.blockSignals(True)
                        win.twoD_include_ftir_chk.setChecked(True)
                        win.twoD_include_ftir_chk.blockSignals(False)
                except Exception as exc:
                    warnings_out.append(f"Could not reload FTIR overlay {p.name}: {exc}")

        vlim2 = self.twoD.get("view_limits") or {}
        pc2 = getattr(win, "twoD_plot_controls", None)
        if pc2 is not None and vlim2:
            if "pump_lim" in vlim2 and hasattr(pc2, "set_pump_lim_native"):
                try:
                    pc2.set_pump_lim_native(vlim2["pump_lim"][0], vlim2["pump_lim"][1])
                except Exception:
                    pass
            if "probe_lim" in vlim2 and hasattr(pc2, "set_probe_lim_native"):
                try:
                    pc2.set_probe_lim_native(vlim2["probe_lim"][0], vlim2["probe_lim"][1])
                except Exception:
                    pass
            if "pump_descending" in vlim2 and hasattr(pc2, "set_pump_descending"):
                try:
                    pc2.set_pump_descending(bool(vlim2["pump_descending"]))
                except Exception:
                    pass
            if "probe_descending" in vlim2 and hasattr(pc2, "set_probe_descending"):
                try:
                    pc2.set_probe_descending(bool(vlim2["probe_descending"]))
                except Exception:
                    pass

        if hasattr(win, "_twoD_preview_contour"):
            win._twoD_preview_contour()

"""The :class:`Dataset2D` pipeline object (2D-IR / 2D-EV).

Mirrors :class:`pymorgan.oneD.Dataset1D`: it holds the raw signal cube
``Z[pump, probe, t2]`` plus the pump/probe/t2 axes, and ties the
**load -> process -> plot -> analyse** stages together through thin methods.
Presentation defaults come from the shared :class:`pymorgan.settings.Settings`
(label style, colourmap, and the 2-D ``freq_label`` convention).
"""

from __future__ import annotations

from os import PathLike

import numpy as np

from ..log import get_logger
from ..settings import Settings, get_settings
from .load import get_map_loader

logger = get_logger(__name__)


class Dataset2D:
    """A two-dimensional time-resolved dataset (a t2 stack of 2-D maps).

    Attributes
    ----------
    Z_R:
        Raw signal cube, ``[Npump x Nprobe x Nt2]``.
    pump, probe:
        Pump (omega_1) and probe (omega_3) frequency axes.
    delays:
        Population times t2.
    units, freq_units:
        Unit-label dictionary and the spectral-axis unit string.
    Z_C:
        Background-corrected cube (``None`` until :meth:`background_correct`).
    """

    def __init__(
        self,
        Z,
        pump,
        probe,
        delays,
        units: dict,
        freq_units: str = "cm-1",
        *,
        source: str | None = None,
        data_type: str | None = None,
        datatype: str | None = None,
        in_progress: bool = False,
        raw_signal=None,
        raw_interferogram=None,
        raw_t1delays=None,
        cal_level: str | None = None,
        cal_source: str | None = None,
        is_frequency_domain: bool | None = None,
        w0_cm1: float | None = None,
        dt1_fs: float | None = None,
        state_names: list[str] | None = None,
        states_data: dict[str, np.ndarray] | None = None,
        states_signal_ss: dict[str, np.ndarray] | None = None,
        active_state: str | None = None,
        raw_signal_ss: np.ndarray | None = None,
        raw_interferogram_ss: np.ndarray | None = None,
        nscans: int | None = None,
        active_scan_indices: list[int] | None = None,
    ):
        self.Z_R = np.asarray(Z)
        self.pump = np.asarray(pump)
        self.probe = np.asarray(probe)
        self.delays = np.asarray(delays)
        self.units = units if isinstance(units, dict) else {}
        self.freq_units = freq_units
        self.source = source
        self.data_type = data_type
        self.in_progress = in_progress
        self.cal_level = cal_level
        self.cal_source = cal_source or self.units.get("cal_source")

        self.raw_signal = raw_signal
        self.raw_interferogram = raw_interferogram
        self.raw_t1delays = raw_t1delays

        sig_ss = raw_signal_ss if raw_signal_ss is not None else self.units.get("raw_signal_ss")
        int_ss = (
            raw_interferogram_ss
            if raw_interferogram_ss is not None
            else self.units.get("raw_interferogram_ss")
        )
        self.raw_signal_ss = np.asarray(sig_ss, dtype=float) if sig_ss is not None else None
        self.raw_interferogram_ss = (
            np.asarray(int_ss, dtype=float) if int_ss is not None else None
        )
        if self.raw_signal_ss is not None and self.raw_signal_ss.ndim == 4:
            self.nscans = int(self.raw_signal_ss.shape[-1])
        elif nscans is not None:
            self.nscans = int(nscans)
        else:
            self.nscans = int(self.units.get("nscans", 1) or 1)

        if active_scan_indices is not None:
            self.active_scan_indices = [int(i) for i in active_scan_indices]
        else:
            self.active_scan_indices = list(range(self.nscans))

        # Determine datatype: 'shaper', 'interferometer', 'FreqDomain', or 'processed'
        if datatype is not None:
            self.datatype = datatype
        else:
            if self.data_type in ("P2DAT", "RAL_Proc"):
                self.datatype = "processed"
            else:
                self.datatype = "processed"

        if is_frequency_domain is not None:
            self.is_frequency_domain = bool(is_frequency_domain)
        else:
            self.is_frequency_domain = bool(
                self.units.get("is_frequency_domain", False)
                or self.datatype == "FreqDomain"
            )

        self.w0_cm1 = (
            float(w0_cm1)
            if w0_cm1 is not None
            else float(self.units.get("w0_cm1", 0.0) or 0.0)
        )
        self.dt1_fs = (
            float(dt1_fs)
            if dt1_fs is not None
            else float(self.units.get("dt1_fs", 0.0) or 0.0)
        )
        self.state_names = (
            list(state_names)
            if state_names is not None
            else list(self.units.get("state_names", []))
        )
        self.states_data = (
            dict(states_data)
            if states_data is not None
            else (dict(self.units["states_data"]) if "states_data" in self.units else None)
        )
        self.states_signal_ss = (
            dict(states_signal_ss)
            if states_signal_ss is not None
            else (dict(self.units["states_signal_ss"]) if "states_signal_ss" in self.units else None)
        )
        self.active_state = active_state or self.units.get("active_state")

        self.Z_C = None  # populated by background_correct()
        self._is_corrected = False
        self.subtracted_spectrum = None

    def set_active_state(self, state_name: str) -> bool:
        """Switch the active slow-modulation / polarization state cube if available."""
        if not self.states_data or state_name not in self.states_data:
            return False
        cube = np.asarray(self.states_data[state_name], dtype=float)
        self.active_state = state_name
        if self.states_signal_ss and state_name in self.states_signal_ss:
            ss_cube = np.asarray(self.states_signal_ss[state_name], dtype=float)
            self.raw_signal_ss = ss_cube.copy()
            self.nscans = int(ss_cube.shape[-1])
            valid_active = [
                i for i in getattr(self, "active_scan_indices", []) if 0 <= i < self.nscans
            ]
            if valid_active and len(valid_active) < self.nscans:
                with np.errstate(divide="ignore", invalid="ignore"):
                    cube = np.nan_to_num(
                        np.nanmean(ss_cube[:, :, :, valid_active], axis=-1), nan=0.0
                    )
                self.active_scan_indices = valid_active
            else:
                self.active_scan_indices = list(range(self.nscans))
        if self.is_frequency_domain or self.datatype in ("FreqDomain", "processed"):
            self.Z_R = cube.copy()
            if self.raw_signal is not None:
                self.raw_signal = cube.copy()
        else:
            self.raw_signal = cube.copy()
        self.Z_C = None
        self._is_corrected = False
        return True

    # ----------------------------------------------------------------- #
    #                          Load (construction)                      #
    # ----------------------------------------------------------------- #
    @classmethod
    def from_file(
        cls, path: str | PathLike, data_type: str = "P2DAT", progress_tracker=None
    ) -> Dataset2D:
        """Load a 2-D dataset from ``path`` using the loader for ``data_type``."""
        loader = get_map_loader(data_type)
        if data_type in ("MESS_2DIR", "PyMESS_2D") or getattr(loader, "__name__", None) in (
            "read_MESS_2DIR",
            "read_PyMESS_2D",
        ):
            res = loader(str(path), progress_tracker=progress_tracker)
        else:
            res = loader(str(path))
        if len(res) == 11:
            (
                Z,
                pump,
                probe,
                delays,
                units,
                freq_units,
                datatype,
                in_progress,
                raw_signal,
                raw_interferogram,
                raw_t1delays,
            ) = res
        elif len(res) == 8:
            Z, pump, probe, delays, units, freq_units, datatype, in_progress = res
            raw_signal = None
            raw_interferogram = None
            raw_t1delays = None
        elif len(res) == 7:
            Z, pump, probe, delays, units, freq_units, datatype = res
            in_progress = False
            raw_signal = None
            raw_interferogram = None
            raw_t1delays = None
        else:
            Z, pump, probe, delays, units, freq_units = res
            datatype = None
            in_progress = False
            raw_signal = None
            raw_interferogram = None
            raw_t1delays = None
        units = dict(units or {})
        if "sample_info" not in units:
            try:
                from ..oneD.load import _find_mess_sample_info_file, parse_mess_sample_info

                info_file = _find_mess_sample_info_file(path)
                if info_file is not None:
                    info_str = parse_mess_sample_info(info_file)
                    if info_str:
                        units["sample_info"] = info_str
            except Exception:
                pass

        # Check for CalibratedProbe.csv override (only in data folder for PyMESS_2D unless probe is uncalibrated)
        from pathlib import Path

        folder = Path(path)
        cal_path = None
        if folder.is_dir():
            cand_data = folder / "CalibratedProbe.csv"
            if cand_data.is_file():
                cal_path = cand_data
            else:
                cand_root = folder.parent / "CalibratedProbe.csv"
                if cand_root.is_file():
                    cal_path = cand_root
        else:
            cand_data = folder.parent / "CalibratedProbe.csv"
            if cand_data.is_file():
                cal_path = cand_data
            else:
                cand_root = folder.parent.parent / "CalibratedProbe.csv"
                if cand_root.is_file():
                    cal_path = cand_root

        cal_level = None
        cal_source = units.get("cal_source")
        if cal_path is not None:
            try:
                # Load calibrated probe axis
                calibrated_probe = np.loadtxt(cal_path, delimiter=",").ravel().astype(float)
                # Check that its length matches the probe shape
                if len(calibrated_probe) == len(probe):
                    probe = calibrated_probe
                    is_datadir = (folder.is_dir() and cal_path.parent == folder) or (
                        not folder.is_dir() and cal_path.parent == folder.parent
                    )
                    cal_level = "datadir" if is_datadir else "rootdir"
                    cal_source = f"CalibratedProbe.csv ({'data folder' if is_datadir else 'root folder'})"
            except Exception:
                logger.warning(
                    "Could not read the probe calibration file %s; "
                    "using the uncalibrated probe axis.",
                    cal_path,
                    exc_info=True,
                )

        if cal_source is None:
            if data_type == "MESS_2DIR":
                cal_source = f"Dataset probe axis ({folder.name}_wavenumbers.csv)"
            elif data_type == "P2DAT":
                cal_source = "Dataset probe axis (P2DAT header)"
            elif data_type == "PyMESS_2D":
                cal_source = "PyMESS spectral calibration (cm-1)"

        return cls(
            Z,
            pump,
            probe,
            delays,
            units,
            freq_units,
            source=str(path),
            data_type=data_type,
            datatype=datatype,
            in_progress=in_progress,
            raw_signal=raw_signal,
            raw_interferogram=raw_interferogram,
            raw_t1delays=raw_t1delays,
            cal_level=cal_level,
            cal_source=cal_source,
        )

    # ----------------------------------------------------------------- #
    #                            Properties                             #
    # ----------------------------------------------------------------- #
    @property
    def Z(self):
        """Most-processed cube: corrected if available, else raw."""
        return self.Z_R if self.Z_C is None else self.Z_C

    @property
    def n_maps(self) -> int:
        """Number of population-time (t2) maps."""
        return self.Z_R.shape[2]

    @property
    def n_detectors(self) -> int:
        """Number of detector channels in the signal."""
        if self.Z_R.ndim == 4:
            return self.Z_R.shape[3]
        if self.probe.ndim == 2:
            return self.probe.shape[1]
        return 1

    @property
    def is_corrected(self) -> bool:
        """Whether a background correction has been applied."""
        return self._is_corrected

    @property
    def has_single_scans(self) -> bool:
        """Whether per-scan 2D data is available for previewing and recombining."""
        return (
            self.raw_signal_ss is not None
            and self.raw_signal_ss.ndim == 4
            and self.raw_signal_ss.shape[-1] > 1
        )

    def _format_active_scans_str(self) -> str:
        """Human-readable summary of total and active 1-based scan numbers."""
        n = int(getattr(self, "nscans", 1) or 1)
        active = getattr(self, "active_scan_indices", None)
        if not active or len(active) == n:
            return f"{n} (1\u2013{n})" if n > 1 else "1"
        one_based = sorted(int(i) + 1 for i in active)
        parts: list[str] = []
        start = prev = one_based[0]
        for v in one_based[1:]:
            if v == prev + 1:
                prev = v
            else:
                parts.append(f"{start}" if start == prev else f"{start}\u2013{prev}")
                start = prev = v
        parts.append(f"{start}" if start == prev else f"{start}\u2013{prev}")
        return f"{len(one_based)} of {n} (scans {', '.join(parts)})"

    def map_index(self, t2) -> int:
        """Index of the t2 map nearest ``t2``."""
        return int(np.argmin(np.abs(self.delays - t2)))

    def axis_units(self, display: bool = False, settings=None):
        """Units of the dataset axes: X (pump), Y (probe), Z (signal), t2 (delay).

        Returns a :class:`~pymorgan.helpers.DatasetUnits`::

            >>> d2.axis_units().x.unit
            'cm-1'
            >>> d2.axis_units(display=True).x.label()
            '$10^{3}$ cm$^{-1}$'   # visible-range map

        With ``display=True`` the spectral axes are reported as they would be
        plotted: converted to ``Settings.twoD_freq_unit``, carrying the ``10^3``
        factor when one is applied, and swapped when
        ``Settings.pump_axis = "Vertical"`` puts the probe on X.
        """
        from pymorgan import helpers as hlp

        from ..settings import get_settings

        units = self.units if isinstance(self.units, dict) else {}
        native = units.get("unitsL") or getattr(self, "freq_units", None) or "cm-1"
        unit, scale, vertical = native, 1.0, False
        if display:
            s = settings or get_settings()
            target = getattr(s, "twoD_freq_unit", native)
            unit = target.value if hasattr(target, "value") else str(target)
            span = np.concatenate(
                [np.asarray(self.pump, dtype=float), np.asarray(self.probe, dtype=float)]
            )
            scale = hlp.display_scale(hlp.convert_spectral(span, native, unit))
            pump_axis = getattr(s, "pump_axis", "Horizontal")
            pump_axis = pump_axis.value if hasattr(pump_axis, "value") else str(pump_axis)
            vertical = pump_axis == "Vertical"

        pump_unit = hlp.spectral_axis_unit(unit, scale)
        probe_unit = hlp.spectral_axis_unit(unit, scale)
        return hlp.DatasetUnits(
            x=probe_unit if vertical else pump_unit,
            y=pump_unit if vertical else probe_unit,
            z=hlp.signal_axis_unit(units),
            t2=hlp.delay_axis_unit(units, quantity="Population time"),
        )

    def sample_info(self) -> str:
        """Return an HTML summary of the 2-D dataset shown in the GUI info box."""
        pump = np.asarray(self.pump, dtype=float)
        probe = np.asarray(self.probe, dtype=float)
        delays = np.asarray(self.delays, dtype=float)

        pump_res = float(np.nanmean(np.abs(np.diff(pump)))) if pump.size > 1 else float("nan")
        probe_res = float(np.nanmean(np.abs(np.diff(probe)))) if probe.size > 1 else float("nan")

        def fmt_unit(u):
            return u.replace("cm-1", "cm\u207b\u00b9").replace("cm^-1", "cm\u207b\u00b9")

        probe_unit = fmt_unit(self.units.get("unitsL", "cm-1"))
        delay_unit = self.units.get("unitsT_ltx", "ps")

        # --- Pump line ---
        if self.datatype in ("shaper", "interferometer") and self.raw_t1delays is not None:
            t1 = np.asarray(self.raw_t1delays, dtype=float)
            dt1 = float(np.nanmean(np.abs(np.diff(t1)))) if t1.size > 1 else 0.0
            pump_unit = fmt_unit(self.units.get("unitsL", "cm-1"))
            line_pump = (
                f"<b>Pump (t\u2081):</b> [{t1.min():.1f}, {t1.max():.1f}] fs, dt\u2081: {dt1:.2f} fs, Bins: {len(t1)}<br>"
                f"<b>Pump (\u03c9\u2081):</b> [{pump.min():.1f}, {pump.max():.1f}] {pump_unit}, Res: {pump_res:.2f} {pump_unit}, Bins: {len(pump)}"
            )
        else:
            pump_unit = fmt_unit(self.units.get("unitsL", "cm-1"))
            line_pump = (
                f"<b>Pump (\u03c9\u2081):</b> [{pump.min():.1f}, {pump.max():.1f}] {pump_unit}"
                f", Res: {pump_res:.2f} {pump_unit}, Bins: {len(pump)}"
            )

        # --- Probe line ---
        line_probe = (
            f"<b>Probe (\u03c9\u2083):</b> [{probe.min():.1f}, {probe.max():.1f}] {probe_unit}"
            f", Res: {probe_res:.2f} {probe_unit}, Bins: {len(probe)}"
        )

        # --- t2 line ---
        line_t2 = (
            f"<b>t\u2082 delays:</b> {len(delays)}"
            f", range: [{delays.min():.2f}, {delays.max():.2f}] {delay_unit}"
        )

        # --- Calibration line ---
        cal = getattr(self, "cal_level", None)
        if cal == "datadir":
            cal_str = "External file (data folder)"
        elif cal == "rootdir":
            cal_str = "External file (root folder)"
        elif cal == "autocalibrated":
            cal_str = "Auto-calibrated from data"
        else:
            cal_str = "Dataset probe calibration"
        line_cal = f"<b>Calibration:</b> {cal_str}"

        # --- Data type line (last) ---
        if self.in_progress:
            line_type = f"<b>Data type:</b> {self.datatype} [IN PROGRESS]"
        else:
            line_type = f"<b>Data type:</b> {self.datatype}"

        lines = [line_pump, line_probe, line_t2]
        if self.has_single_scans:
            lines.append(f"<b>Scans:</b> {self._format_active_scans_str()}")
        lines.extend([line_cal, line_type])
        info_extra = self.units.get("sample_info")
        if info_extra:
            formatted_info = info_extra.replace("\n", "<br>")
            lines.append(f"<b>Sample Info:</b><br>{formatted_info}")
        else:
            lines.append("<b>Sample Info:</b> No additional sample information available.")
        return "<br>".join(lines)

    def sample_info_console(self) -> str:
        """Return a plain text, console-friendly summary of the 2-D dataset."""
        pump = np.asarray(self.pump, dtype=float)
        probe = np.asarray(self.probe, dtype=float)
        delays = np.asarray(self.delays, dtype=float)

        pump_res = float(np.nanmean(np.abs(np.diff(pump)))) if pump.size > 1 else float("nan")
        probe_res = float(np.nanmean(np.abs(np.diff(probe)))) if probe.size > 1 else float("nan")

        def fmt_unit(u):
            return (
                u.replace("cm-1", "cm-1").replace("cm^-1", "cm-1").replace("cm\u207b\u00b9", "cm-1")
            )

        probe_unit = fmt_unit(self.units.get("unitsL", "cm-1"))
        delay_unit = self.units.get("unitsT_ltx", "ps")

        # --- Pump line ---
        if self.datatype in ("shaper", "interferometer") and self.raw_t1delays is not None:
            t1 = np.asarray(self.raw_t1delays, dtype=float)
            dt1 = float(np.nanmean(np.abs(np.diff(t1)))) if t1.size > 1 else 0.0
            pump_unit = fmt_unit(self.units.get("unitsL", "cm-1"))
            line_pump = (
                f"Pump (t1): [{t1.min():.1f}, {t1.max():.1f}] fs, dt1: {dt1:.2f} fs, Bins: {len(t1)}\n"
                f"Pump (w1): [{pump.min():.1f}, {pump.max():.1f}] {pump_unit}, Res: {pump_res:.2f} {pump_unit}, Bins: {len(pump)}"
            )
        else:
            pump_unit = fmt_unit(self.units.get("unitsL", "cm-1"))
            line_pump = (
                f"Pump (w1): [{pump.min():.1f}, {pump.max():.1f}] {pump_unit}"
                f", Res: {pump_res:.2f} {pump_unit}, Bins: {len(pump)}"
            )

        # --- Probe line ---
        line_probe = (
            f"Probe (w3): [{probe.min():.1f}, {probe.max():.1f}] {probe_unit}"
            f", Res: {probe_res:.2f} {probe_unit}, Bins: {len(probe)}"
        )

        # --- t2 line ---
        line_t2 = f"t2 delays: {len(delays)}, range: [{delays.min():.2f}, {delays.max():.2f}] {delay_unit}"

        # --- Calibration line ---
        cal = getattr(self, "cal_level", None)
        if cal == "datadir":
            cal_str = "External file (data folder)"
        elif cal == "rootdir":
            cal_str = "External file (root folder)"
        elif cal == "autocalibrated":
            cal_str = "Auto-calibrated from data"
        else:
            cal_str = "Dataset probe calibration"
        line_cal = f"Calibration: {cal_str}"

        # --- Data type line (last) ---
        if self.in_progress:
            line_type = f"Data type: {self.datatype} [IN PROGRESS]"
        else:
            line_type = f"Data type: {self.datatype}"

        lines = [line_pump, line_probe, line_t2]
        if self.has_single_scans:
            lines.append(f"Scans: {self._format_active_scans_str()}")
        lines.extend([line_cal, line_type])
        info_extra = self.units.get("sample_info")
        if info_extra:
            lines.append(f"Sample Info:\n{info_extra}")
        else:
            lines.append("Sample Info: No additional sample information available.")
        return "\n".join(lines)

    def calibration_status(self) -> str:
        """Return a clean, human-readable probe calibration source description."""
        cal_source = getattr(self, "cal_source", None)
        if cal_source:
            return cal_source
        cal = getattr(self, "cal_level", None)
        if cal == "datadir":
            return "CalibratedProbe.csv (data folder)"
        if cal == "rootdir":
            return "CalibratedProbe.csv (root folder)"
        if cal == "autocalibrated":
            return "Auto-calibrated from CAL tab"
        if self.data_type == "MESS_2DIR":
            return "Dataset probe axis (wavenumbers.csv)"
        if self.data_type == "P2DAT":
            return "Dataset probe axis (P2DAT header)"
        return "Dataset probe calibration"

    def print_sample_info(self) -> None:
        """Print the console-friendly summary of the 2-D dataset."""
        print(self.sample_info_console())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Dataset2D(data_type={self.data_type!r}, datatype={self.datatype!r}, pump={self.pump.shape}, "
            f"probe={self.probe.shape}, t2={self.delays.shape}, corrected={self.is_corrected})"
        )

    def copy(self, deep: bool = True) -> Dataset2D:
        """Return a copy of the 2-D dataset.

        When ``deep=True`` (default), all signal and coordinate arrays are cloned independently.
        """
        import copy as _copy

        def _dup(arr):
            if arr is None:
                return None
            return np.copy(arr) if deep else arr

        new_ds = Dataset2D(
            _dup(self.Z_R),
            _dup(self.pump),
            _dup(self.probe),
            _dup(self.delays),
            _copy.deepcopy(self.units) if deep else self.units.copy(),
            freq_units=self.freq_units,
            source=self.source,
            data_type=self.data_type,
            datatype=self.datatype,
            in_progress=self.in_progress,
            raw_signal=_dup(self.raw_signal),
            raw_interferogram=_dup(self.raw_interferogram),
            raw_t1delays=_dup(self.raw_t1delays),
            cal_level=getattr(self, "cal_level", None),
            cal_source=getattr(self, "cal_source", None),
            is_frequency_domain=getattr(self, "is_frequency_domain", False),
            w0_cm1=getattr(self, "w0_cm1", 0.0),
            dt1_fs=getattr(self, "dt1_fs", 0.0),
            state_names=(
                list(self.state_names) if getattr(self, "state_names", None) else None
            ),
            states_data=(
                {k: _dup(v) for k, v in self.states_data.items()}
                if getattr(self, "states_data", None)
                else None
            ),
            states_signal_ss=(
                {k: _dup(v) for k, v in self.states_signal_ss.items()}
                if getattr(self, "states_signal_ss", None)
                else None
            ),
            active_state=getattr(self, "active_state", None),
            raw_signal_ss=_dup(getattr(self, "raw_signal_ss", None)),
            raw_interferogram_ss=_dup(getattr(self, "raw_interferogram_ss", None)),
            nscans=getattr(self, "nscans", 1),
            active_scan_indices=(
                list(self.active_scan_indices)
                if getattr(self, "active_scan_indices", None) is not None
                else None
            ),
        )
        new_ds.Z_C = _dup(self.Z_C)
        sub_spec = getattr(self, "subtracted_spectrum", None)
        new_ds.subtracted_spectrum = (
            _copy.deepcopy(sub_spec) if deep else sub_spec
        )
        return new_ds

    def crop(
        self,
        t2_min: float | None = None,
        t2_max: float | None = None,
        pump_min: float | None = None,
        pump_max: float | None = None,
        probe_min: float | None = None,
        probe_max: float | None = None,
    ) -> Dataset2D:
        """Crop the 2-D dataset along t2, pump, and/or probe axes.

        Mutates the dataset in-place and returns ``self`` for chaining.
        To keep the original dataset intact, call ``dataset.copy().crop(...)``.
        """
        if t2_min is not None or t2_max is not None:
            t_low = -np.inf if t2_min is None else float(t2_min)
            t_high = np.inf if t2_max is None else float(t2_max)
            t_mask = (self.delays >= t_low) & (self.delays <= t_high)
            if not np.any(t_mask):
                raise ValueError(f"No t2 delays fall in range [{t_low}, {t_high}]")
            self.delays = self.delays[t_mask]
            self.Z_R = self.Z_R[:, :, t_mask]
            if self.Z_C is not None:
                self.Z_C = self.Z_C[:, :, t_mask]

        if pump_min is not None or pump_max is not None:
            p_low = -np.inf if pump_min is None else float(pump_min)
            p_high = np.inf if pump_max is None else float(pump_max)
            pu_mask = (self.pump >= p_low) & (self.pump <= p_high)
            if not np.any(pu_mask):
                raise ValueError(f"No pump frequencies fall in range [{p_low}, {p_high}]")
            self.pump = self.pump[pu_mask]
            self.Z_R = self.Z_R[pu_mask, :, :]
            if self.Z_C is not None:
                self.Z_C = self.Z_C[pu_mask, :, :]

        if probe_min is not None or probe_max is not None:
            pr_low = -np.inf if probe_min is None else float(probe_min)
            pr_high = np.inf if probe_max is None else float(probe_max)
            pr_probe = self.probe[:, 0] if self.probe.ndim > 1 else self.probe
            pr_mask = (pr_probe >= pr_low) & (pr_probe <= pr_high)
            if not np.any(pr_mask):
                raise ValueError(f"No probe frequencies fall in range [{pr_low}, {pr_high}]")
            self.probe = self.probe[pr_mask] if self.probe.ndim == 1 else self.probe[pr_mask, :]
            self.Z_R = self.Z_R[:, pr_mask, :]
            if self.Z_C is not None:
                self.Z_C = self.Z_C[:, pr_mask, :]

        return self

    def _repr_html_(self) -> str:
        """Rich HTML summary card for Jupyter notebook display."""
        time_unit = self.units.get("unitsT_ltx", "ps").replace("$", "")
        freq_unit = self.freq_units or self.units.get("unitsL_lbl", "cm-1")
        z_unit = self.units.get("unitsZ_lbl", "\u0394A")

        t2_min = f"{float(np.nanmin(self.delays)):.2f}" if self.delays.size else "n/a"
        t2_max = f"{float(np.nanmax(self.delays)):.2f}" if self.delays.size else "n/a"
        pu_min = f"{float(np.nanmin(self.pump)):.1f}" if self.pump.size else "n/a"
        pu_max = f"{float(np.nanmax(self.pump)):.1f}" if self.pump.size else "n/a"
        pr_min = f"{float(np.nanmin(self.probe)):.1f}" if self.probe.size else "n/a"
        pr_max = f"{float(np.nanmax(self.probe)):.1f}" if self.probe.size else "n/a"

        corr_badge = (
            '<span style="background:#10b981;color:#fff;padding:2px 8px;border-radius:12px;font-size:11px;">Corrected</span>'
            if self.is_corrected
            else '<span style="background:#64748b;color:#fff;padding:2px 8px;border-radius:12px;font-size:11px;">Raw</span>'
        )

        return (
            '<div style="font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, Helvetica, Arial, sans-serif; '
            'border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px 18px; max-width: 540px; '
            'background: #ffffff; color: #1e293b; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin: 8px 0;">'
            '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px;">'
            f'<span style="font-weight: 700; font-size: 15px; color: #0f172a;">PyMORGAN Dataset2D</span>'
            f'<div>{corr_badge}</div>'
            '</div>'
            '<table style="width: 100%; border-collapse: collapse; font-size: 13px;">'
            f'<tr><td style="color:#64748b; padding: 3px 0;">Data Type</td><td style="font-weight:600;">{self.data_type or "Unknown"}</td></tr>'
            f'<tr><td style="color:#64748b; padding: 3px 0;">Cube Dimensions</td><td style="font-weight:600;">{self.pump.shape[0]} pump &times; {self.probe.shape[0]} probe &times; {len(self.delays)} t2</td></tr>'
            f'<tr><td style="color:#64748b; padding: 3px 0;">Population time (t2)</td><td>[{t2_min}, {t2_max}] {time_unit}</td></tr>'
            f'<tr><td style="color:#64748b; padding: 3px 0;">Pump range</td><td>[{pu_min}, {pu_max}] {freq_unit}</td></tr>'
            f'<tr><td style="color:#64748b; padding: 3px 0;">Probe range</td><td>[{pr_min}, {pr_max}] {freq_unit}</td></tr>'
            f'<tr><td style="color:#64748b; padding: 3px 0;">Signal unit</td><td>{z_unit}</td></tr>'
            f'<tr><td style="color:#64748b; padding: 3px 0;">Calibration</td><td style="color:#0369a1;">{self.calibration_status()}</td></tr>'
            '</table>'
            '</div>'
        )

    def process(
        self,
        *,
        apodise_method: str = "0",
        zeropad_enable: bool = True,
        zeropad_factor: int = 1,
        zeropad_next2k: bool = False,
        phase_method: str = "No fit",
        phase_points: int = 10,
        pumpcorrection: bool = False,
        bkg_sub: bool = False,
        bkgIdx: int = 0,
        w0: float | None = None,
        dt1: float | None = None,
        autocalibrate_probe: bool = False,
        cal_probe_vector: np.ndarray | None = None,
        progress_tracker=None,
    ):
        """Process raw time-domain signal and interferogram data to frequency-domain.

        Performs baseline correction, apodization, zero-padding, FFT, and phasing.
        Updates self.Z_R and self.pump.
        """
        from . import process

        return process.process(
            self,
            apodise_method=apodise_method,
            zeropad_enable=zeropad_enable,
            zeropad_factor=zeropad_factor,
            zeropad_next2k=zeropad_next2k,
            phase_method=phase_method,
            phase_points=phase_points,
            pumpcorrection=pumpcorrection,
            bkg_sub=bkg_sub,
            bkgIdx=bkgIdx,
            w0=w0,
            dt1=dt1,
            autocalibrate_probe=autocalibrate_probe,
            cal_probe_vector=cal_probe_vector,
            progress_tracker=progress_tracker,
        )

    # ----------------------------------------------------------------- #
    #                          Shared resolution                        #
    # ----------------------------------------------------------------- #
    def _resolve(self, settings: Settings | None, label_style, freq_label):
        """Resolve active settings and the per-call label/freq overrides."""
        s = settings or get_settings()
        lstyle = label_style if label_style is not None else s.label_style.value
        flabel = freq_label if freq_label is not None else s.freq_label.value
        if hasattr(flabel, "value"):
            flabel = flabel.value
        return s, lstyle, flabel, self.units

    # ----------------------------------------------------------------- #
    #                             Process                               #
    # ----------------------------------------------------------------- #
    def background_correct(self, reference=None, *, t2=None, do_correct: bool = True) -> Dataset2D:
        """Subtract a reference t2 map from every map.

        Choose the reference by ``reference`` (a t2 index) or ``t2`` (a delay
        value, resolved to the nearest map). With neither, this is a no-op
        passthrough. Returns ``self`` for chaining.
        """
        from . import process

        ref = self.map_index(t2) if t2 is not None else reference
        self.Z_C = process.background_correct(self.Z_R, ref, do_correct=do_correct)
        self._is_corrected = do_correct and ref is not None
        return self

    def recalc_average(self, scan_indices) -> Dataset2D:
        """Recompute the averaged 2D dataset across the selected 0-based ``scan_indices``.

        Combines ``raw_signal_ss`` (and ``raw_interferogram_ss`` when available) by
        arithmetic mean across the selected scans, updates ``raw_signal`` (and ``Z_R``
        for frequency-domain datasets), records ``active_scan_indices``, and clears
        ``Z_C``. Call :meth:`process` afterwards for time-domain datasets to update
        the Fourier-transformed ``Z_R`` cube.
        """
        if not self.has_single_scans:
            raise ValueError(
                f"Dataset of type {self.data_type!r} has no per-scan data to recombine."
            )
        indices = [int(k) for k in scan_indices]
        if not indices:
            raise ValueError("No scans selected.")
        if any(k < 0 or k >= self.nscans for k in indices):
            raise ValueError(f"Scan indices must be between 0 and {self.nscans - 1}.")

        with np.errstate(divide="ignore", invalid="ignore"):
            sig_avg = np.nanmean(self.raw_signal_ss[:, :, :, indices], axis=-1)
            sig_avg = np.nan_to_num(sig_avg, nan=0.0)
            self.raw_signal = sig_avg

            if (
                self.raw_interferogram_ss is not None
                and self.raw_interferogram_ss.ndim == 3
                and self.raw_interferogram_ss.shape[-1] == self.nscans
            ):
                int_avg = np.nanmean(self.raw_interferogram_ss[:, :, indices], axis=-1)
                int_avg = np.nan_to_num(int_avg, nan=0.0)
                self.raw_interferogram = int_avg

        if self.is_frequency_domain or self.datatype in ("FreqDomain", "processed"):
            self.Z_R = self.raw_signal.copy()

        self.active_scan_indices = sorted(set(indices))
        self.Z_C = None
        self._is_corrected = False
        return self

    def compute_scan_maps_at_t2(
        self,
        t2: float | None = None,
        *,
        t2_idx: int | None = None,
        bin_size: int = 1,
        **process_kwargs,
    ) -> list[tuple[str, list[int], np.ndarray]]:
        """Compute 2D frequency-domain maps ``(label, scan_indices_0based, Z_map)`` at a single ``t2``.

        For time-domain datasets (``shaper`` / ``interferometer``), only the requested ``t2``
        slice (plus the scatter-subtraction reference slice if ``bkg_sub=True``) is
        Fourier-transformed and phased per scan/bin.
        """
        if not self.has_single_scans:
            raise ValueError("Dataset has no individual scans to preview.")

        if t2_idx is None:
            idx = self.map_index(t2 if t2 is not None else float(self.delays[0]))
        else:
            idx = int(np.clip(t2_idx, 0, len(self.delays) - 1))

        bsize = max(1, int(bin_size))
        n_total = self.nscans

        class _NullTracker:
            def set_total(self, total):
                pass

            def update(self, step, label=""):
                pass

            def update_split(self, step, total, stage):
                pass

            def close(self):
                pass

        bkg_sub = bool(process_kwargs.get("bkg_sub", False))
        bkg_idx = int(process_kwargs.get("bkgIdx", 0) or 0)
        bkg_idx = int(np.clip(bkg_idx, 0, len(self.delays) - 1))

        results: list[tuple[str, list[int], np.ndarray]] = []
        for start in range(0, n_total, bsize):
            indices = list(range(start, min(n_total, start + bsize)))
            if len(indices) == 1:
                label = f"Scan {indices[0] + 1}"
            else:
                label = f"Scans {indices[0] + 1}\u2013{indices[-1] + 1}"

            if (
                self.is_frequency_domain
                or self.datatype in ("FreqDomain", "processed")
                or self.raw_t1delays is None
                or self.raw_interferogram is None
            ):
                with np.errstate(divide="ignore", invalid="ignore"):
                    z_map = np.nanmean(self.raw_signal_ss[:, :, idx, indices], axis=-1)
                    if bkg_sub and bkg_idx != idx:
                        z_bkg = np.nanmean(self.raw_signal_ss[:, :, bkg_idx, indices], axis=-1)
                        z_map = z_map - z_bkg
                results.append((label, indices, np.nan_to_num(z_map, nan=0.0)))
            else:
                if bkg_sub and bkg_idx != idx:
                    slice_indices = [idx, bkg_idx]
                    local_bkg_idx = 1
                else:
                    slice_indices = [idx]
                    local_bkg_idx = 0

                with np.errstate(divide="ignore", invalid="ignore"):
                    sub_sig = np.nanmean(
                        self.raw_signal_ss[:, :, slice_indices, :][:, :, :, indices], axis=-1
                    )
                    sub_sig = np.nan_to_num(sub_sig, nan=0.0)
                    if (
                        self.raw_interferogram_ss is not None
                        and self.raw_interferogram_ss.ndim == 3
                        and self.raw_interferogram_ss.shape[-1] == n_total
                    ):
                        sub_int = np.nanmean(
                            self.raw_interferogram_ss[:, slice_indices, :][:, :, indices], axis=-1
                        )
                        sub_int = np.nan_to_num(sub_int, nan=0.0)
                    else:
                        sub_int = np.asarray(self.raw_interferogram[:, slice_indices], dtype=float).copy()

                mini_ds = Dataset2D(
                    sub_sig.copy(),
                    np.asarray(self.raw_t1delays, dtype=float).copy(),
                    np.asarray(self.probe, dtype=float).copy(),
                    np.asarray(self.delays[slice_indices], dtype=float).copy(),
                    dict(self.units),
                    freq_units=self.freq_units,
                    source=self.source,
                    data_type=self.data_type,
                    datatype=self.datatype,
                    raw_signal=sub_sig,
                    raw_interferogram=sub_int,
                    raw_t1delays=np.asarray(self.raw_t1delays, dtype=float).copy(),
                    w0_cm1=self.w0_cm1,
                    dt1_fs=self.dt1_fs,
                )
                kw = dict(process_kwargs)
                kw["bkg_sub"] = bkg_sub and (bkg_idx != idx)
                kw["bkgIdx"] = local_bkg_idx
                kw["autocalibrate_probe"] = False
                kw["progress_tracker"] = _NullTracker()
                mini_ds.process(**kw)
                z_map = np.asarray(mini_ds.Z[:, :, 0], dtype=float)
                results.append((label, indices, z_map))

        return results

    def subtract_spectrum(
        self,
        ref_dataset: Dataset2D,
        scale: float = 1.0,
        ref_t2_idx: int | None = None,
    ) -> Dataset2D:
        """Subtract a reference 2D dataset from this dataset.

        Parameters
        ----------
        ref_dataset : Dataset2D
            The reference dataset to subtract.
        scale : float, optional
            Scaling factor applied to the reference spectrum (default 1.0).
        ref_t2_idx : int, optional
            If provided, use a single reference t2 map for all delays. If None,
            match corresponding t2 delays (or nearest t2 per map).

        Returns
        -------
        Dataset2D
            self (for chaining).
        """
        from scipy.interpolate import RegularGridInterpolator

        sample_Z = np.copy(self.Z_R if self.Z_C is None else self.Z_C)
        ref_Z = ref_dataset.Z

        # Check grid matching
        pump_match = np.array_equal(self.pump, ref_dataset.pump)
        probe_match = np.array_equal(self.probe, ref_dataset.probe)

        _, _, Nt2 = sample_Z.shape
        subtracted = np.zeros_like(sample_Z)

        for i_t2 in range(Nt2):
            t2_val = self.delays[i_t2]
            if ref_t2_idx is not None:
                r_map = ref_Z[:, :, ref_t2_idx]
            else:
                r_idx = ref_dataset.map_index(t2_val)
                r_map = ref_Z[:, :, r_idx]

            if pump_match and probe_match:
                ref_map_interp = r_map
            else:
                interp = RegularGridInterpolator(
                    (ref_dataset.pump, ref_dataset.probe),
                    r_map,
                    bounds_error=False,
                    fill_value=0.0,
                )
                P_grid, R_grid = np.meshgrid(self.pump, self.probe, indexing="ij")
                ref_map_interp = interp((P_grid, R_grid))

            subtracted[:, :, i_t2] = sample_Z[:, :, i_t2] - scale * ref_map_interp

        self.Z_C = subtracted
        self._is_corrected = True
        self._subtraction_ref = ref_dataset
        self._subtraction_scale = scale
        self._subtraction_ref_t2_idx = ref_t2_idx
        return self

    def reset_subtraction(self) -> Dataset2D:
        """Reset subtracted spectrum and restore un-subtracted data."""
        self._subtraction_ref = None
        self._subtraction_scale = 1.0
        self._subtraction_ref_t2_idx = None
        self.Z_C = None
        self._is_corrected = False
        return self


    # ----------------------------------------------------------------- #
    #                              Plot                                 #
    # ----------------------------------------------------------------- #
    def plot_map(self, t2, ax=None, **kwargs):
        """Plot the 2-D correlation map nearest population time ``t2``.

        Parameters
        ----------
        t2 : float
            Requested population time, snapped to the nearest measured t2 (see
            :meth:`map_index`).
        ax : matplotlib.axes.Axes, optional
            Axis to draw into; ``None`` creates a new 6x6-inch figure. The
            colorbar and any top spectrum are appended to this axis with a
            divider, so the call composes into a ``GridSpec`` layout.
        **kwargs
            Passed to :func:`pymorgan.twoD.plot.plot_map`, which documents them
            all. Commonly used: ``vmin``/``vmax`` (explicit colour limits, for
            putting a t2 series on one common scale), ``show_colorbar``,
            ``ShowLines``, ``Nlevels``/``Nskip``, ``aspect``, ``top_spectrum``
            (steady-state overlay panel), ``pump_lim``/``probe_lim`` with
            ``cut_plot``, ``cls_points``/``nls_points`` overlays and
            ``freq_unit``.

        Returns
        -------
        Map2DAxes
            Namedtuple ``(ax, top, cbar)``; ``top`` and ``cbar`` are ``None``
            when not drawn.

        Examples
        --------
        >>> out = data.plot_map(0.5, ShowLines=True)
        >>> out.ax.set_title("early t2")
        """
        from . import plot

        return plot.plot_map(self, t2, ax, **kwargs)

    def plot_surface(self, t2, ax=None, **kwargs):
        """Plot the 2-D spectrum nearest ``t2`` as a 3D surface.

        Parameters
        ----------
        t2 : float
            Requested population time, snapped to the nearest measured t2.
        ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional
            3-D axis to draw into (``projection="3d"`` required when supplied).
        **kwargs
            Passed to :func:`pymorgan.twoD.plot.plot_surface` (``vmin``/``vmax``,
            ``pump_lim``/``probe_lim``/``zlim``, ``elev``/``azim``,
            ``rstride``/``cstride``, ``alpha``, ...).

        Returns
        -------
        mpl_toolkits.mplot3d.axes3d.Axes3D
        """
        from . import plot

        return plot.plot_surface(self, t2, ax, **kwargs)

    def plot_scan_maps(self, t2, **kwargs):
        """Plot an array of 2-D contour maps across individual (or binned) scans at ``t2``."""
        from . import plot

        return plot.plot_scan_map_array(self, t2, **kwargs)

    # ----------------------------------------------------------------- #
    #                             Analyse                              #
    # ----------------------------------------------------------------- #
    def slice_at(self, t2):
        """Return ``(pump, probe, map)`` for the map nearest ``t2``."""
        from . import analyse

        return analyse.slice_at(self, t2)

    def diagonal(self, t2=None, offset: float = 0.0, method: str = "linear", num_points: int | None = None):
        """Return ``(freq, signal)`` along the diagonal or off-diagonal trajectory (w3 = w1 + offset)."""
        from . import analyse

        return analyse.diagonal(self, t2=t2, offset=offset, method=method, num_points=num_points)

    def antidiagonal(
        self,
        centre: tuple[float, float] | None = None,
        t2=None,
        method: str = "linear",
        num_points: int | None = None,
        *,
        center: tuple[float, float] | None = None,
    ):
        """Return ``(rel_disp, signal)`` along anti-diagonal cut passing through centre."""
        from . import analyse

        return analyse.antidiagonal(self, centre=centre, t2=t2, method=method, num_points=num_points, center=center)

    def compare_diag_antidiag(
        self,
        centre: tuple[float, float] | None = None,
        t2=None,
        method: str = "cubic",
        *,
        center: tuple[float, float] | None = None,
    ):
        """Return ``(rel_disp, norm_diag, norm_antidiag)`` comparing normalised profiles."""
        from . import analyse

        return analyse.compare_diag_antidiag(self, centre=centre, t2=t2, method=method, center=center)

    def integral_dynamics(self, pump_range: tuple[float, float], probe_range: tuple[float, float], method: str = "trapezoid"):
        """Return ``(delays, I_t2)`` for 2D ROI integration across population delays."""
        from . import analyse

        return analyse.integral_dynamics(self, pump_range=pump_range, probe_range=probe_range, method=method)

    def center_line_slope(self, *args, **kwargs):
        """Centre-line-slope analysis."""
        from . import analyse

        return analyse.center_line_slope(self, *args, **kwargs)

    def centre_line_slope(self, *args, **kwargs):
        """Alias for :meth:`center_line_slope`."""
        from . import analyse

        return analyse.center_line_slope(self, *args, **kwargs)

    def doCLS(self, *args, **kwargs):
        """Short CLI name for centre-line-slope analysis."""
        from . import analyse

        return analyse.center_line_slope(self, *args, **kwargs)

    def nodal_line_slope(self, *args, **kwargs):
        """Nodal line slope analysis."""
        from . import analyse

        return analyse.nodal_line_slope(self, *args, **kwargs)

    def doNLS(self, *args, **kwargs):
        """Short CLI name for nodal-line-slope analysis."""
        from . import analyse

        return analyse.nodal_line_slope(self, *args, **kwargs)

    def doKuboFit(self, *args, **kwargs):
        """Perform direct Kubo model fitting on the signal cube."""
        from . import kubo_fit

        return kubo_fit.run_kubo_fit(self.Z, self.pump, self.probe, self.delays, *args, **kwargs)

    def get_slice_at_pump(self, pump_wn):
        """Return a 1D dataset cut along the probe axis at a fixed pump wavenumber."""
        from . import analyse

        return analyse.get_slice_at_pump(self, pump_wn)

    def get_slice_at_probe(self, probe_wn):
        """Return a 1D dataset cut along the pump axis at a fixed probe wavenumber."""
        from . import analyse

        return analyse.get_slice_at_probe(self, probe_wn)

    slice_at_pump = get_slice_at_pump
    slice_at_probe = get_slice_at_probe

    def get_slice_integrate_pump(self, pump_min, pump_max):
        """Integrate along the pump axis between pump_min and pump_max, returning a 1D dataset."""
        from . import analyse

        return analyse.get_slice_integrate_pump(self, pump_min, pump_max)

    def get_slice_integrate_probe(self, probe_min, probe_max):
        """Integrate along the probe axis between probe_min and probe_max, returning a 1D dataset."""
        from . import analyse

        return analyse.get_slice_integrate_probe(self, probe_min, probe_max)

    def to_p2dat(self, path: str | PathLike, use_corrected: bool = True) -> None:
        """Write this 2D dataset to a P2DAT file."""
        from .load import write_P2DAT

        Z_data = self.Z if use_corrected else self.Z_R
        write_P2DAT(path, self.pump, self.probe, self.delays, Z_data)


def load_2D(path: str | PathLike, data_type: str = "P2DAT", progress_tracker=None) -> Dataset2D:
    """Convenience wrapper for :meth:`Dataset2D.from_file`."""
    return Dataset2D.from_file(path, data_type=data_type, progress_tracker=progress_tracker)

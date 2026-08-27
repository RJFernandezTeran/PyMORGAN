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
        freq_units: str,
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
    ):
        self.Z_R = np.asarray(Z)
        self.pump = np.asarray(pump)
        self.probe = np.asarray(probe)
        self.delays = np.asarray(delays)
        self.units = units
        self.freq_units = freq_units
        self.source = source
        self.data_type = data_type
        self.in_progress = in_progress
        self.cal_level = cal_level
        self.cal_source = cal_source

        self.raw_signal = raw_signal
        self.raw_interferogram = raw_interferogram
        self.raw_t1delays = raw_t1delays

        # Determine datatype: 'shaper', 'interferometer', or 'processed'
        if datatype is not None:
            self.datatype = datatype
        else:
            if self.data_type in ("P2DAT", "RAL_Proc"):
                self.datatype = "processed"
            else:
                self.datatype = "processed"

        self.Z_C = None  # populated by background_correct()
        self._is_corrected = False

    # ----------------------------------------------------------------- #
    #                          Load (construction)                      #
    # ----------------------------------------------------------------- #
    @classmethod
    def from_file(
        cls, path: str | PathLike, data_type: str = "P2DAT", progress_tracker=None
    ) -> Dataset2D:
        """Load a 2-D dataset from ``path`` using the loader for ``data_type``."""
        loader = get_map_loader(data_type)
        if data_type == "MESS_2DIR" or getattr(loader, "__name__", None) == "read_MESS_2DIR":
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

        # Check for CalibratedProbe.csv override
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
        cal_source = None
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

        lines = [line_pump, line_probe, line_t2, line_cal, line_type]
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

        lines = [line_pump, line_probe, line_t2, line_cal, line_type]
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

        Npump, Nprobe, Nt2 = sample_Z.shape
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

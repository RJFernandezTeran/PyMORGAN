"""The :class:`Dataset1D` pipeline object.

``Dataset1D`` ties the 1-D pipeline together: **load** (:mod:`.load`),
**process** (:mod:`.process`), **plot** (:mod:`.plot`) and **analyse**
(:mod:`.analyse`). It holds the raw arrays plus provenance and resolves
presentation defaults from the active :class:`pymorgan.settings.Settings`. The
stage modules contain the actual logic; the methods here are thin, well-typed
entry points that pass ``self`` (and the resolved settings) through. Heavy
imports (matplotlib, scipy) stay inside the methods.
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike

import numpy as np

from ..settings import DeltaAUnits, Settings, get_settings
from . import process
from .load import get_loader

# Plain-text probe-axis units (the QPlainTextEdit info box renders no LaTeX).
_PLAIN_PROBE_UNIT = {"Wavenumber": "cm\u207b\u00b9", "Wavelength": "nm", "Energy": "eV"}


def _ltx_to_text(s: str) -> str:
    """Best-effort conversion of a LaTeX unit string to plain Unicode text."""
    return (
        s.replace("$", "")
        .replace(r"\times 10^{3}", "\u00d710\u00b3")
        .replace("^{-1}", "\u207b\u00b9")
        .replace("{", "")
        .replace("}", "")
        .strip()
    )


def _format_1d_delay_val(t: float, unit_in: str) -> str:
    if not np.isfinite(t):
        return "n/a"
    try:
        from .. import helpers as hlp

        val, u_out = hlp.ConvertTimeUnits(float(t), unit_in)
        u_clean = u_out.replace(r"$\mu$s", "μs")
        val_str = f"{val:g}"
        return f"{val_str} {u_clean}"
    except Exception:
        val_str = f"{t:g}"
        return f"{val_str} {unit_in}"


def _format_1d_delay_info(delays, unit_in: str, html: bool = False) -> str:
    arr = np.asarray(delays, dtype=float)
    n = int(arr.size)
    if n == 0:
        range_str = "[n/a, n/a]"
    else:
        dmin = float(np.nanmin(arr))
        dmax = float(np.nanmax(arr))
        min_str = _format_1d_delay_val(dmin, unit_in)
        max_str = _format_1d_delay_val(dmax, unit_in)
        range_str = f"[{min_str}, {max_str}]"

    prefix = "<b>Delays:</b>" if html else "Delays:"
    return f"{prefix} {n}, range: {range_str}"


class Dataset1D:
    """A one-dimensional time-resolved dataset and its processing pipeline.

    Attributes
    ----------
    Zavg_R:
        Averaged signal, ``[Ndelays x Npixels x Ndetectors]``.
    delays, probe:
        Delay and probe (spectral) axes.
    units:
        Unit dictionary (see ``helpers.units2dic``).
    nscans, Zss_R, Zstdv:
        Scan count, single-scan signal and standard deviation.
    Zavg_C:
        Background-corrected signal (``None`` until :meth:`background_correct`).
    """

    def __init__(
        self,
        Zavg_R,
        delays,
        probe,
        units: dict,
        nscans=np.nan,
        Zss_R=None,
        Zstdv=None,
        *,
        source: str | None = None,
        data_type: str | None = None,
        scan_ids=None,
        counts=None,
    ):
        self.Zavg_R = np.asarray(Zavg_R)
        self.delays = np.asarray(delays)
        self.probe = np.asarray(probe)
        default_u = {
            "unitsT_lbl": "Delay",
            "unitsT_ltx": "ps",
            "unitsL_lbl": "Wavenumber",
            "unitsL_ltx": r"$\mathrm{cm^{-1}}$",
            "unitsZ_lbl": r"$\Delta$A",
            "unitsZ_ltx": "mOD",
        }
        self.units = {**default_u, **(units or {})}
        self.nscans = nscans
        self.Zss_R = Zss_R
        self.Zstdv = Zstdv
        self.source = source
        self.data_type = data_type
        self.scan_ids = scan_ids
        self.counts = counts if counts is not None else self.units.get("counts")
        self.cal_source = self.units.get("cal_source")
        self.cal_level = self.units.get("cal_level")

        self.Zavg_C = None
        self.Zss_C = None
        self.bkg_avg = None
        self.bkg_ss = None
        self.chirp_fit = None

    # ----------------------------------------------------------------- #
    #                          Load (construction)                      #
    # ----------------------------------------------------------------- #
    @classmethod
    def from_file(cls, path: str | PathLike, data_type: str = "PDAT", **loader_kwargs) -> Dataset1D:
        """Load a dataset from ``path`` using the loader for ``data_type``.

        Extra keyword arguments are forwarded to the loader (e.g. ``spectrum``,
        ``slowmod`` and ``anisotropy`` for the MESS_TRIR reader). Loaders that
        take no options simply receive none.
        """
        loader = get_loader(data_type)
        res = loader(str(path), **loader_kwargs)
        counts = None
        if len(res) >= 9:
            Zavg_R, delays, probe, units, nscans, Zss_R, Zstdv, scan_ids, counts = res[:9]
        elif len(res) >= 8:
            Zavg_R, delays, probe, units, nscans, Zss_R, Zstdv, scan_ids = res[:8]
        else:
            Zavg_R, delays, probe, units, nscans, Zss_R, Zstdv = res
            scan_ids = None

        units = dict(units or {})
        if counts is not None:
            units["counts"] = counts

        if "sample_info" not in units:
            try:
                from .load import _find_mess_sample_info_file, parse_mess_sample_info

                info_file = _find_mess_sample_info_file(path)
                if info_file is not None:
                    info_str = parse_mess_sample_info(info_file)
                    if info_str:
                        units["sample_info"] = info_str
            except Exception:
                pass

        return cls(
            Zavg_R,
            delays,
            probe,
            units,
            nscans,
            Zss_R,
            Zstdv,
            source=str(path),
            data_type=data_type,
            scan_ids=scan_ids,
            counts=counts,
        )

    # ----------------------------------------------------------------- #
    #                            Properties                             #
    # ----------------------------------------------------------------- #
    @property
    def Z(self):
        """Most-processed signal: corrected if available, else raw."""
        return self.Zavg_R if self.Zavg_C is None else self.Zavg_C

    @property
    def n_detectors(self) -> int:
        """Number of detector channels in the signal."""
        return self.Zavg_R.shape[2] if self.Zavg_R.ndim == 3 else 1

    @property
    def is_corrected(self) -> bool:
        """Whether a background correction has been applied."""
        return self.Zavg_C is not None

    @property
    def has_single_scans(self) -> bool:
        """Whether real per-scan arrays are available in ``Zss_R``.

        True when ``Zss_R`` is a 4-D ``[Ndelays x Npixels x Ndet x Nscans]``
        array carrying finite data (e.g. a MESS dataset loaded with
        ``Settings.load_single_scans``); False for a NaN placeholder.
        """
        z = self.Zss_R
        if z is None:
            return False
        z = np.asarray(z)
        return z.ndim == 4 and z.shape[3] >= 1 and bool(np.any(np.isfinite(z)))

    def scan_signal(self, detector: int = 0) -> np.ndarray:
        """Return the single-scan signal ``[Ndelays x Npixels x Nscans]``.

        Uses the background-corrected single scans (``Zss_C``) when available,
        otherwise the raw single scans (``Zss_R``). Raises if no per-scan data
        is present (see :attr:`has_single_scans`).
        """
        if not self.has_single_scans:
            raise ValueError(
                "No single-scan data available. Load the dataset with "
                "Settings.load_single_scans enabled."
            )
        z = None
        if self.Zss_C is not None:
            zc = np.asarray(self.Zss_C, dtype=float)
            if zc.ndim == 4 and zc.shape[3] >= 1 and np.any(np.isfinite(zc)):
                z = zc
        if z is None:
            z = np.asarray(self.Zss_R, dtype=float)
        return z[:, :, detector, :]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Dataset1D(data_type={self.data_type!r}, "
            f"delays={self.delays.shape}, probe={self.probe.shape}, "
            f"detectors={self.n_detectors}, corrected={self.is_corrected})"
        )

    # ----------------------------------------------------------------- #
    #                          Shared resolution                        #
    # ----------------------------------------------------------------- #
    def _detector_slice(self, detector: int):
        """Return the 2-D ``[Ndelays x Npixels]`` slice for ``detector``."""
        return self.Z[:, :, detector]

    def _detector_probe(self, detector: int = 0) -> np.ndarray:
        """Return 1-D probe array for ``detector``."""
        probe = np.asarray(self.probe, dtype=float)
        if probe.ndim > 1:
            d = int(np.clip(detector, 0, probe.shape[1] - 1))
            return probe[:, d]
        return probe

    # ----------------------------------------------------------------- #
    #                       Sample-info / noise summary                 #
    # ----------------------------------------------------------------- #
    @property
    def rms(self) -> np.ndarray | None:
        """Return the raw single-shot RMS noise matrix if present in units."""
        if hasattr(self, "units") and isinstance(self.units, dict):
            return self.units.get("rms")
        return None

    def noise_array(self):
        """Per-point noise (standard deviation) of the signal, or ``None``.

        Noise is taken from :attr:`Zstdv` when it carries non-trivial values
        (e.g. loaded from a sibling ``.pdatn`` file), otherwise computed as the
        standard deviation across the scan axis of :attr:`Zss_R` when
        single-scan data is present. Returns ``None`` when neither is available.

        Returns
        -------
        numpy.ndarray or None
            Same shape as :attr:`Z` (``[Ndelays x Npixels x Ndetectors]``), in
            the same unit as the signal (mOD), or ``None`` when the dataset
            carries no noise information.

        Notes
        -----
        Public because it is the input to weighted fitting: PyRATE-TA reads it to
        build the ``1/sigma`` weights that turn a fit statistic into a true
        reduced chi-squared. A dataset without a ``.pdatn`` sibling and without
        single scans returns ``None``, and a weighted fit of it is refused
        rather than silently falling back to an unweighted one.
        """
        z = self.Zstdv
        if z is not None:
            z = np.asarray(z, dtype=float)
            if z.size and np.any(np.isfinite(z)) and not np.allclose(np.nan_to_num(z), 0.0):
                return z
        if self.has_single_scans:
            zss = np.asarray(self.Zss_R, dtype=float)
            if zss.size and np.any(np.isfinite(zss)):
                return np.nanstd(zss, axis=-1)
        return None

    @property
    def has_noise(self) -> bool:
        """True if non-trivial noise or single-scan standard deviation is available."""
        return self.noise_array() is not None

    def _noise_array(self):
        """Deprecated alias of :meth:`noise_array`.

        Kept so code written against the private name keeps working; new code
        calls :meth:`noise_array`.
        """
        return self.noise_array()

    def axis_units(self, display: bool = False, settings=None):
        """Units of the dataset axes: X (probe), Y (delay) and Z (signal).

        Returns a :class:`~pymorgan.helpers.DatasetUnits` whose entries carry the
        quantity, the unit token, the mathtext used on plots and the display
        scaling factor::

            >>> data.axis_units().x.unit
            'cm-1'
            >>> print(data.axis_units())
            Axis units
              x: Wavenumber in cm-1
              y: Delay in ps
              z: $\\Delta$A in x1E3

        With ``display=True`` the spectral axis is reported as it would be
        plotted, i.e. converted to ``Settings.x_axis_unit`` and carrying the
        ``10^3`` factor when the plot applies one; otherwise the dataset's own
        (stored) units are reported.
        """
        from pymorgan import helpers as hlp

        from ..settings import get_settings

        units = self.units or {}
        native = hlp.native_x_unit(units)
        scale = 1.0
        unit = native
        if display:
            s = settings or get_settings()
            x_axis_unit = getattr(s, "x_axis_unit", "native")
            x_axis_unit = x_axis_unit.value if hasattr(x_axis_unit, "value") else x_axis_unit
            unit = hlp.resolve_x_unit(x_axis_unit, native)
            # The 1-D plots only rescale wavenumber axes (see plot._resolve_x_axis).
            if unit == "cm-1":
                probe = np.asarray(self.probe, dtype=float)
                scale = hlp.display_scale(hlp.convert_spectral(probe, native, unit))

        return hlp.DatasetUnits(
            x=hlp.spectral_axis_unit(unit, scale),
            y=hlp.delay_axis_unit(units),
            z=hlp.signal_axis_unit(units),
        )

    def sample_info(self) -> str:
        """Return the two-line dataset summary shown in the GUI info box.

        Line 1 reports the average and maximum noise (when single-scan or
        ``.pdatn`` data is available, else a notice). Line 2 reports the
        signal-to-noise ratio, the data kind, the number of delays and the
        probe-axis resolution. Units are plain text (no LaTeX markup).
        """
        u = self.units or {}
        probe_unit = _PLAIN_PROBE_UNIT.get(
            u.get("unitsL_lbl", ""), _ltx_to_text(u.get("unitsL_ltx", ""))
        )
        noise_unit = (
            "mOD" if u.get("unitsZ") in ("mOD", "x1E3") else _ltx_to_text(u.get("unitsZ_ltx", ""))
        )

        probe = np.asarray(self.probe, dtype=float)
        res = float(np.median(np.abs(np.diff(probe)))) if probe.size > 1 else float("nan")
        n_delays = int(np.asarray(self.delays).size)
        scans = f"Avg. of {int(self.nscans)} scans" if np.isfinite(self.nscans) else "Avg. Data"

        sig = np.asarray(self.Z, dtype=float)
        peak = (
            float(np.nanmax(np.abs(sig))) if sig.size and np.any(np.isfinite(sig)) else float("nan")
        )

        noise = self.noise_array()
        if noise is None:
            line1 = "No single scan data available"
            snr = "Inf"
        else:
            avg_noise = float(np.nanmean(noise))
            max_noise = float(np.nanmax(noise))
            line1 = f"<b>Avg. Noise:</b> {avg_noise:.4f} {noise_unit}&nbsp;&nbsp;&nbsp;&nbsp;<b>Max Noise:</b> {max_noise:.4f} {noise_unit}"
            snr = "Inf" if avg_noise == 0 or not np.isfinite(peak) else f"{peak / avg_noise:.0f}"

        res_str = f"{res:.2f} {probe_unit}" if np.isfinite(res) else "n/a"
        line2 = f"<b>SNR:</b> {snr} / <b>Scans:</b> {scans} / <b>Res.:</b> {res_str}"
        line3 = _format_1d_delay_info(self.delays, u.get("unitsT_ltx", "ps"), html=True)
        line_cal = f"<b>Calibration:</b> {self.calibration_status()}"
        lines = [line1, line2, line3, line_cal]
        info_extra = self.units.get("sample_info")
        if info_extra:
            formatted_info = info_extra.replace("\n", "<br>")
            lines.append(f"<b>Sample Info:</b><br>{formatted_info}")
        else:
            lines.append("<b>Sample Info:</b> No additional sample information available.")
        return "<br>".join(lines)

    def sample_info_console(self) -> str:
        """Return a plain text, console-friendly summary of the 1-D dataset."""
        u = self.units or {}
        probe_unit = _PLAIN_PROBE_UNIT.get(
            u.get("unitsL_lbl", ""), _ltx_to_text(u.get("unitsL_ltx", ""))
        )
        # Convert any unicode exponents to plain text
        probe_unit = probe_unit.replace("cm⁻¹", "cm-1")

        noise_unit = (
            "mOD" if u.get("unitsZ") in ("mOD", "x1E3") else _ltx_to_text(u.get("unitsZ_ltx", ""))
        )
        noise_unit = noise_unit.replace("cm⁻¹", "cm-1")

        probe = np.asarray(self.probe, dtype=float)
        res = float(np.median(np.abs(np.diff(probe)))) if probe.size > 1 else float("nan")
        scans = f"Avg. of {int(self.nscans)} scans" if np.isfinite(self.nscans) else "Avg. Data"

        sig = np.asarray(self.Z, dtype=float)
        peak = (
            float(np.nanmax(np.abs(sig))) if sig.size and np.any(np.isfinite(sig)) else float("nan")
        )

        noise = self.noise_array()
        if noise is None:
            line1 = "No single scan data available"
            snr = "Inf"
        else:
            avg_noise = float(np.nanmean(noise))
            max_noise = float(np.nanmax(noise))
            line1 = f"Avg. Noise: {avg_noise:.4f} {noise_unit}    Max Noise: {max_noise:.4f} {noise_unit}"
            snr = "Inf" if avg_noise == 0 or not np.isfinite(peak) else f"{peak / avg_noise:.0f}"

        res_str = f"{res:.2f} {probe_unit}" if np.isfinite(res) else "n/a"
        line2 = f"SNR: {snr} / Scans: {scans} / Res.: {res_str}"
        line3 = _format_1d_delay_info(self.delays, u.get("unitsT_ltx", "ps"), html=False)
        line_cal = f"Calibration: {self.calibration_status()}"
        lines = [line1, line2, line3, line_cal]
        info_extra = self.units.get("sample_info")
        if info_extra:
            lines.append(f"Sample Info:\n{info_extra}")
        else:
            lines.append("Sample Info: No additional sample information available.")
        return "\n".join(lines)

    def calibration_status(self) -> str:
        """Return a clean, human-readable probe calibration source description."""
        cal_source = getattr(self, "cal_source", None) or self.units.get("cal_source")
        if cal_source:
            return cal_source
        cal_level = getattr(self, "cal_level", None) or self.units.get("cal_level")
        if cal_level == "datadir":
            return "CalibratedProbe.csv (data folder)"
        if cal_level == "rootdir":
            return "CalibratedProbe.csv (root folder)"
        if self.data_type == "MESS_TRIR":
            return "Dataset probe axis (wavenumbers.csv)"
        if self.data_type == "MESS_TRUVIS":
            return "Dataset probe axis (wavelengths.csv)"
        if self.data_type == "HARPIA_TA":
            return "Embedded HARPIA probe axis"
        if self.data_type == "PDAT":
            return "Dataset probe axis (PDAT header)"
        if self.data_type == "Helios_TA":
            return "Dataset probe axis (CSV header)"
        return "Dataset probe calibration"

    def print_sample_info(self) -> None:
        """Print the console-friendly summary of the 1-D dataset."""
        print(self.sample_info_console())

    def _resolve(
        self,
        settings: Settings | None,
        label_style: str | None,
        delta_a_units: DeltaAUnits | str | None,
    ) -> tuple[Settings, str, dict]:
        """Resolve active settings and per-call label/unit overrides."""
        s = settings or get_settings()
        style = label_style if label_style is not None else s.label_style.value
        if isinstance(delta_a_units, str):
            delta_a_units = DeltaAUnits(delta_a_units)
        units = s.units_with_convention(self.units, delta_a_units)
        return s, style, units

    # ----------------------------------------------------------------- #
    #                             Process                               #
    # ----------------------------------------------------------------- #
    def background_correct(self, tmin, tmax, do_correct: bool = True) -> Dataset1D:
        """Subtract the pre-zero background averaged over ``[tmin, tmax]``.

        Returns ``self`` so calls can be chained.
        """
        self.Zavg_C, self.Zss_C, self.bkg_avg, self.bkg_ss = process.background_correct(
            self.Zavg_R, self.Zss_R, self.nscans, self.delays, tmin, tmax, do_correct=do_correct
        )
        return self

    def mask_probe_regions(
        self, ranges: list[tuple[float, float]], detector: int | None = None
    ) -> Dataset1D:
        """Remove or mask data in the specified probe ranges (in native units).

        For single-detector datasets (or when n_detectors <= 1), specified probe
        ranges are removed from the probe and data arrays.

        For multi-detector datasets, the masking is applied to the specified
        detector channel only, setting data in the masked range to NaN so each
        detector retains its independent probe axis and grid dimensions.

        Returns ``self`` so calls can be chained.
        """
        if not ranges:
            return self

        if self.n_detectors <= 1:
            # Build a mask of indices to keep
            keep = np.ones(self.probe.shape, dtype=bool)
            for r_min, r_max in ranges:
                keep = keep & ~((self.probe >= r_min) & (self.probe <= r_max))

            self.probe = self.probe[keep]

            if self.Zavg_R is not None:
                if self.Zavg_R.ndim == 3:
                    self.Zavg_R = self.Zavg_R[:, keep, :]
                elif self.Zavg_R.ndim == 2:
                    self.Zavg_R = self.Zavg_R[:, keep]

            if self.Zss_R is not None:
                self.Zss_R = np.asarray(self.Zss_R)
                if self.Zss_R.ndim == 4:
                    self.Zss_R = self.Zss_R[:, keep, :, :]
                elif self.Zss_R.ndim == 3:
                    self.Zss_R = self.Zss_R[:, keep, :]
                elif self.Zss_R.ndim == 2:
                    self.Zss_R = self.Zss_R[:, keep]
                elif self.Zss_R.ndim == 1:
                    self.Zss_R = self.Zss_R[keep]

            if self.Zstdv is not None:
                self.Zstdv = np.asarray(self.Zstdv)
                if self.Zstdv.ndim == 3:
                    self.Zstdv = self.Zstdv[:, keep, :]
                elif self.Zstdv.ndim == 2:
                    self.Zstdv = self.Zstdv[:, keep]
                elif self.Zstdv.ndim == 1:
                    self.Zstdv = self.Zstdv[keep]

            if self.Zavg_C is not None:
                if self.Zavg_C.ndim == 3:
                    self.Zavg_C = self.Zavg_C[:, keep, :]
                elif self.Zavg_C.ndim == 2:
                    self.Zavg_C = self.Zavg_C[:, keep]

            if self.Zss_C is not None:
                if self.Zss_C.ndim == 4:
                    self.Zss_C = self.Zss_C[:, keep, :, :]
                elif self.Zss_C.ndim == 3:
                    self.Zss_C = self.Zss_C[:, keep, :]
                elif self.Zss_C.ndim == 2:
                    self.Zss_C = self.Zss_C[:, keep]

            if self.bkg_avg is not None:
                if self.bkg_avg.ndim == 2:
                    self.bkg_avg = self.bkg_avg[keep, :]
                elif self.bkg_avg.ndim == 1:
                    self.bkg_avg = self.bkg_avg[keep]

            if self.bkg_ss is not None:
                if self.bkg_ss.ndim == 3:
                    self.bkg_ss = self.bkg_ss[keep, :, :]
                elif self.bkg_ss.ndim == 2:
                    self.bkg_ss = self.bkg_ss[keep, :]
                elif self.bkg_ss.ndim == 1:
                    self.bkg_ss = self.bkg_ss[keep]

            self.mask_ranges = ranges
        else:
            det = detector if detector is not None else 0
            det = int(np.clip(det, 0, self.n_detectors - 1))

            probe_det = self._detector_probe(det)
            mask_det = np.zeros(probe_det.shape, dtype=bool)
            for r_min, r_max in ranges:
                mask_det = mask_det | ((probe_det >= r_min) & (probe_det <= r_max))

            if np.any(mask_det):
                if self.Zavg_R is not None and self.Zavg_R.ndim == 3:
                    self.Zavg_R[:, mask_det, det] = np.nan
                if self.Zavg_C is not None and self.Zavg_C.ndim == 3:
                    self.Zavg_C[:, mask_det, det] = np.nan
                if self.Zss_R is not None:
                    self.Zss_R = np.asarray(self.Zss_R)
                    if self.Zss_R.ndim == 4:
                        self.Zss_R[:, mask_det, det, :] = np.nan
                if self.Zss_C is not None:
                    self.Zss_C = np.asarray(self.Zss_C)
                    if self.Zss_C.ndim == 4:
                        self.Zss_C[:, mask_det, det, :] = np.nan
                if self.Zstdv is not None:
                    self.Zstdv = np.asarray(self.Zstdv)
                    if self.Zstdv.ndim == 3:
                        self.Zstdv[:, mask_det, det] = np.nan
                if self.bkg_avg is not None:
                    self.bkg_avg = np.asarray(self.bkg_avg)
                    if self.bkg_avg.ndim == 2:
                        self.bkg_avg[mask_det, det] = np.nan
                if self.bkg_ss is not None:
                    self.bkg_ss = np.asarray(self.bkg_ss)
                    if self.bkg_ss.ndim == 3:
                        self.bkg_ss[mask_det, det, :] = np.nan

            if not hasattr(self, "mask_ranges_by_det"):
                self.mask_ranges_by_det = {}
            self.mask_ranges_by_det[det] = ranges
            self.mask_ranges = ranges

        return self

    def shift_t0(self, t0_shift: float) -> Dataset1D:
        """Shift the delay time axis by subtracting ``t0_shift``.

        Updates ``self.delays = self.delays - t0_shift``.
        Since single-scan data (``Zss_R``) and accumulation counts (``counts``)
        are aligned with ``self.delays``, they are automatically shifted.
        Resetting ``self.Zavg_C`` forces background correction to re-evaluate
        pre-zero delay indices with the updated time axis.
        """
        self.delays = np.asarray(self.delays, dtype=float) - float(t0_shift)
        self.Zavg_C = None
        return self

    def subtract_solvent(
        self, solvent, dt: float | np.ndarray = 0.0, scale: float | np.ndarray = 1.0
    ) -> Dataset1D:
        """Subtract a scaled, time-shifted solvent background from this dataset.

        The solvent signal is interpolated onto this dataset's delay grid after
        applying a timing offset ``dt``.  The corrected signal replaces
        (or initialises) :attr:`Zavg_C`, so it composes naturally with a prior
        :meth:`background_correct` call: the solvent is subtracted from
        whichever signal is currently returned by :attr:`Z`.

        Parameters
        ----------
        solvent : Dataset1D
            The solvent (reference) dataset.  If it has been background-
            corrected (``solvent.Zavg_C`` is not ``None``) its corrected signal
            is used; otherwise the raw signal (``solvent.Zavg_R``) is used.
        dt : float or array-like
            Timing offset in ps (scalar or per-pixel array of length ``Npixels``).
            A positive ``dt`` shifts the solvent signal towards *earlier* times
            on the sample delay grid.
        scale : float or array-like
            Amplitude scaling factor (scalar or per-pixel array of length ``Npixels``).

        Returns
        -------
        Dataset1D
            ``self`` — so calls can be chained.
        """
        Z_solvent = solvent.Zavg_C if solvent.Zavg_C is not None else solvent.Zavg_R

        if not np.array_equal(solvent.probe, self.probe):
            from scipy.interpolate import interp1d

            f = interp1d(
                solvent.probe,
                Z_solvent,
                axis=1,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            Z_solvent_aligned = f(self.probe)
        else:
            Z_solvent_aligned = Z_solvent

        self.Zavg_C = process.subtract_solvent(
            self.Z,
            self.delays,
            Z_solvent_aligned,
            solvent.delays,
            dt=dt,
            scale=scale,
        )
        # Provenance — keep references so the caller can inspect them.
        self._solvent_dataset = solvent
        self._solvent_dt = dt
        self._solvent_scale = scale
        return self

    def subtract_shockwave(
        self,
        pixels: int | Sequence[int] | slice | None = None,
        probe_range: tuple[float, float] | None = None,
        probe_values: float | Sequence[float] | None = None,
        one_based: bool = True,
    ) -> Dataset1D:
        """Subtract an average kinetic trace calculated over selected pixels or probe region.

        Calculates an average kinetic trace over the specified pixel(s) or probe region
        and subtracts it from all pixels in the dataset. Updates :attr:`Zavg_C` (and
        :attr:`Zss_C` if single-scan data is present).

        Parameters
        ----------
        pixels : int, sequence of int, slice, or None
            Pixel index or indices to average over (1-based by default when ``one_based=True``).
        probe_range : tuple (min_probe, max_probe) or None
            Probe axis range (in native units) over which to select pixels.
        probe_values : float or sequence of float or None
            Probe value(s) in native units for which to find the nearest pixel(s).
        one_based : bool, default True
            Whether integer ``pixels`` are 1-based pixel numbers (1..Npixels).

        Returns
        -------
        Dataset1D
            ``self`` — so calls can be chained.
        """
        if pixels is None and probe_range is None and probe_values is None:
            raise ValueError(
                "Must specify at least one of 'pixels', 'probe_range', or 'probe_values'."
            )

        Npixels = self.probe.size
        pix_0based = None

        if pixels is not None:
            if isinstance(pixels, slice):
                step = pixels.step or 1
                if one_based:
                    start = (pixels.start - 1) if pixels.start is not None else 0
                    stop = pixels.stop if pixels.stop is not None else Npixels
                else:
                    start = pixels.start if pixels.start is not None else 0
                    stop = pixels.stop if pixels.stop is not None else Npixels
                pix_0based = np.arange(start, stop, step)
            else:
                arr = np.atleast_1d(np.asarray(pixels))
                if arr.dtype == bool:
                    pix_0based = np.where(arr)[0]
                else:
                    pix_0based = arr - 1 if one_based else arr
        elif probe_range is not None:
            p_min, p_max = min(probe_range), max(probe_range)
            mask = (self.probe >= p_min) & (self.probe <= p_max)
            pix_0based = np.where(mask)[0]
            if pix_0based.size == 0:
                raise ValueError(f"No pixels found in probe range [{p_min}, {p_max}].")
        elif probe_values is not None:
            p_vals = np.atleast_1d(np.asarray(probe_values, dtype=float))
            nearest_indices = [int(np.argmin(np.abs(self.probe - val))) for val in p_vals]
            pix_0based = np.unique(np.array(nearest_indices, dtype=int))

        Z_in = self.Z
        self.Zavg_C, shockwave_trace = process.subtract_shockwave(Z_in, pix_0based, one_based=False)

        if self.has_single_scans:
            Zss_in = self.Zss_C if self.Zss_C is not None else self.Zss_R
            self.Zss_C, _ = process.subtract_shockwave(Zss_in, pix_0based, one_based=False)

        self._shockwave_pixels = pix_0based
        self._shockwave_trace = shockwave_trace
        return self

    def fit_solvent_auto(
        self,
        solvent,
        det_idx: int = 0,
        per_pixel_dt: bool | None = None,
        per_pixel_scale: bool | None = None,
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """Automatically find optimal timing offset and scaling for solvent subtraction.

        Fits a global (or per-pixel) timing offset ``dt`` and a global (or per-pixel)
        amplitude scaling factors ``scale`` to match an erf-broadened step-like rise.

        Parameters
        ----------
        solvent : Dataset1D
            The solvent dataset.
        det_idx : int, default 0
            The detector channel index to optimise for.
        per_pixel_dt : bool or None
            Whether to optimise a separate timing offset for each pixel. If None,
            reads the default from active settings (``solvent_per_pixel_dt``).
        per_pixel_scale : bool or None
            Whether to optimise a separate amplitude scaling factor for each pixel.
            If None, reads the default from active settings (``solvent_per_pixel_scale``).

        Returns
        -------
        scale : float or ndarray
            Optimal global scaling factor (float) or per-pixel scaling factors (ndarray).
        dt : float or ndarray
            Optimal global timing offset (float) or per-pixel timing offsets (ndarray).
        """
        import pymorgan as pm

        if per_pixel_dt is None:
            per_pixel_dt = pm.get_settings().solvent_per_pixel_dt
        if per_pixel_scale is None:
            per_pixel_scale = pm.get_settings().solvent_per_pixel_scale

        Z_solvent = solvent.Zavg_C if solvent.Zavg_C is not None else solvent.Zavg_R

        if not np.array_equal(solvent.probe, self.probe):
            from scipy.interpolate import interp1d

            f = interp1d(
                solvent.probe,
                Z_solvent,
                axis=1,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            Z_solvent_aligned = f(self.probe)
        else:
            Z_solvent_aligned = Z_solvent

        return process.fit_solvent_auto(
            self.Z,
            self.delays,
            Z_solvent_aligned,
            solvent.delays,
            det_idx=det_idx,
            per_pixel_dt=per_pixel_dt,
            per_pixel_scale=per_pixel_scale,
        )

    # ----------------------------------------------------------------- #
    #                              Plot                                 #
    # ----------------------------------------------------------------- #
    def plot_contour(self, ax=None, **kwargs):
        """Delay-vs-probe contour map.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis to draw into; ``None`` creates a new figure. When an axis is
            given the figure layout is left to the caller, so the panel can be
            placed in a ``GridSpec`` composite.
        **kwargs
            Passed to :func:`pymorgan.oneD.plot.plot_contour`, which documents
            them all. The ones reached for most often are ``Zscale`` (colour
            limit as a percentage of the peak), ``Zmin``/``Zmax`` (explicit
            limits, for putting panels on a common scale), ``Yscale``
            (``"symlog"``/``"log"``/``"lin"`` delay axis), ``ShowLines``,
            ``Nlevels``/``Nskip``, ``cmap_ID``, ``x_axis_unit``,
            ``show_colorbar`` and ``show_xlabel``/``show_ylabel``. Anything not
            given follows the active :class:`~pymorgan.settings.Settings`.

        Returns
        -------
        matplotlib.axes.Axes

        Examples
        --------
        >>> ax = data.plot_contour(ShowLines=True, Zscale=30, Yscale="symlog")
        """
        from . import plot

        return plot.plot_contour(self, ax, **kwargs)

    def plot_noise_3d(self, ax=None, **kwargs):
        """Plot the noise array as a rotatable 3D surface.

        Parameters
        ----------
        ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional
            3-D axis to draw into (``projection="3d"`` required when supplied).
        **kwargs
            Passed to :func:`pymorgan.oneD.plot.plot_noise_3d` (``detector``,
            ``x_axis_unit``, ``label_style``, ...).

        Returns
        -------
        mpl_toolkits.mplot3d.axes3d.Axes3D

        Raises
        ------
        ValueError
            If the dataset carries neither a noise array nor single-scan data.
        """
        from . import plot

        return plot.plot_noise_3d(self, ax, **kwargs)

    def plot_counts(self, ax=None, **kwargs):
        """Plot accumulation counts as a function of probe delay.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis to draw into; ``None`` creates a new figure.
        **kwargs
            Passed to :func:`pymorgan.oneD.plot.plot_counts`.

        Returns
        -------
        matplotlib.axes.Axes
        """
        from . import plot

        return plot.plot_counts(self, ax, **kwargs)

    def plot_surface(self, ax=None, **kwargs):
        """Delay-vs-probe 3D surface with colourmap shading.

        Parameters
        ----------
        ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional
            3-D axis to draw into (``projection="3d"`` required when supplied).
        **kwargs
            Passed to :func:`pymorgan.oneD.plot.plot_surface`; see there for the
            full list (``rstride``/``cstride``, ``elev``/``azim``, ``Yscale``,
            ``cmap_ID``, ...).

        Returns
        -------
        mpl_toolkits.mplot3d.axes3d.Axes3D
        """
        from . import plot

        return plot.plot_surface(self, ax, **kwargs)

    def plot_spectra(self, plot_delays, ax=None, fig=None, **kwargs):
        """Transient spectra at the requested delays.

        Parameters
        ----------
        plot_delays : sequence of float
            Delays to cut at, in this dataset's time unit. Each is snapped to
            the nearest measured delay, so approximate values are fine.
        ax : matplotlib.axes.Axes, optional
            Axis to draw into; ``None`` creates a new figure.
        fig : matplotlib.figure.Figure, optional
            Figure owning ``ax``; inferred from ``ax`` when omitted.
        **kwargs
            Passed to :func:`pymorgan.oneD.plot.plot_spectra`, which documents
            them all. Commonly used: ``doSmooth`` (box-filter width; negative
            also draws the raw trace), ``normY``, ``roundT``, ``x_axis_unit``,
            ``secondary_axis``, ``Abs``/``Em`` (steady-state overlays) and
            ``show_xlabel``/``show_ylabel``.

        Returns
        -------
        matplotlib.axes.Axes

        Examples
        --------
        >>> data.plot_spectra([0.5, 1, 5, 50, 500], doSmooth=3, roundT=True)
        """
        from . import plot

        return plot.plot_spectra(self, plot_delays, ax, fig, **kwargs)

    def plot_background(self, ax=None, fig=None, **kwargs):
        """Pre-zero background spectrum, drawn with the transient-spectra rules.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis to draw into; ``None`` creates a new figure.
        fig : matplotlib.figure.Figure, optional
            Figure owning ``ax``; inferred from ``ax`` when omitted.
        **kwargs
            Passed to :func:`pymorgan.oneD.plot.plot_background` (``doSmooth``,
            ``normY``, ``colour``, ``title``, ``x_axis_unit``, ...).

        Returns
        -------
        matplotlib.axes.Axes

        Raises
        ------
        ValueError
            If no background has been computed; run :meth:`background_correct`
            first (``do_correct=False`` stores it without subtracting).
        """
        from . import plot

        return plot.plot_background(self, ax, fig, **kwargs)

    def plot_kinetics(self, plot_wavelengths, ax=None, fig=None, **kwargs):
        """Kinetic traces at the requested probe positions.

        Parameters
        ----------
        plot_wavelengths : sequence of float
            Probe positions to cut at, in this dataset's native probe unit (nm,
            cm-1, ...). Each is snapped to the nearest pixel.
        ax : matplotlib.axes.Axes, optional
            Axis to draw into; ``None`` creates a new figure.
        fig : matplotlib.figure.Figure, optional
            Figure owning ``ax``; inferred from ``ax`` when omitted.
        **kwargs
            Passed to :func:`pymorgan.oneD.plot.plot_kinetics`. Commonly used:
            ``plotStyle`` (format string, e.g. ``"-"``), ``Xscale``, ``normY``,
            ``lw``/``ms`` and ``show_xlabel``/``show_ylabel``.

        Returns
        -------
        ax : matplotlib.axes.Axes
        t : numpy.ndarray
            Delay axis, shape ``[Ndelays]``.
        Y : numpy.ndarray
            Extracted traces, shape ``[Ndelays x Ncuts]`` — the same data that
            was plotted, ready to export or fit.

        Examples
        --------
        >>> ax, t, Y = data.plot_kinetics([2132, 2218], plotStyle="-", lw=1.5)
        """
        from . import plot

        return plot.plot_kinetics(self, plot_wavelengths, ax, fig, **kwargs)

    def plot_scan_kinetics(self, plot_wavelengths, ax=None, fig=None, **kwargs):
        """Per-scan kinetic traces at the requested probe positions.

        Diagnostic for scan-to-scan reproducibility; requires single-scan data
        (see :attr:`has_single_scans`).

        Parameters
        ----------
        plot_wavelengths : sequence of float
            Probe positions to cut at, in the native probe unit.
        ax : matplotlib.axes.Axes, optional
            Axis to draw into; ``None`` creates a new figure.
        fig : matplotlib.figure.Figure, optional
            Figure owning ``ax``; inferred from ``ax`` when omitted.
        **kwargs
            Passed to :func:`pymorgan.oneD.plot.plot_scan_kinetics`; most often
            ``binsize`` (scans averaged per trace), ``Xscale`` and ``normY``.

        Returns
        -------
        ax : matplotlib.axes.Axes
        t : numpy.ndarray
            Delay axis, shape ``[Ndelays]``.
        Y : numpy.ndarray
            Binned traces, shape ``[Ndelays x Npixels x Nbins]``.
        """
        from . import plot

        return plot.plot_scan_kinetics(self, plot_wavelengths, ax, fig, **kwargs)

    def plot_scan_spectra(self, plot_delays, ax=None, fig=None, **kwargs):
        """Per-scan transient spectra at the requested delays.

        Spectral counterpart of :meth:`plot_scan_kinetics`; requires single-scan
        data (see :attr:`has_single_scans`).

        Parameters
        ----------
        plot_delays : sequence of float
            Delays to cut at, in this dataset's time unit.
        ax : matplotlib.axes.Axes, optional
            Axis to draw into; ``None`` creates a new figure.
        fig : matplotlib.figure.Figure, optional
            Figure owning ``ax``; inferred from ``ax`` when omitted.
        **kwargs
            Passed to :func:`pymorgan.oneD.plot.plot_scan_spectra`; most often
            ``binsize``, ``doSmooth`` and ``normY``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        from . import plot

        return plot.plot_scan_spectra(self, plot_delays, ax, fig, **kwargs)

    def plot_species_spectra(
        self, Sfit, Taus, TauErr, isFixTau, modelType, ax=None, fig=None, **kwargs
    ):
        """EAS/SAS from a global-analysis fit.

        Parameters
        ----------
        Sfit : numpy.ndarray
            Component amplitude spectra, shape ``[Npixels x Ncomponents]``.
        Taus : sequence of float
            Fitted lifetime of each component.
        TauErr : sequence of float
            Uncertainty on each lifetime.
        isFixTau : sequence of bool
            Per-component flag marking lifetimes held fixed in the fit.
        modelType : str
            Kinetic model used, which decides the EAS / SAS wording.
        ax : matplotlib.axes.Axes, optional
            Axis to draw into; ``None`` creates a new figure.
        fig : matplotlib.figure.Figure, optional
            Figure owning ``ax``; inferred from ``ax`` when omitted.
        **kwargs
            Passed to :func:`pymorgan.oneD.plot.plot_species_spectra`
            (``printErrors``, ``normY``, ``doSmooth``, ``Abs``/``Em``, ...).

        Returns
        -------
        matplotlib.axes.Axes

        Notes
        -----
        The ``Sfit`` / ``Taus`` / ``TauErr`` inputs come from a global-analysis
        fit, which PyMORGAN does not perform — see the companion project
        **PyRATE-TA**. This method is the rendering half of that seam and takes
        plain arrays, so it is independent of how the fit was obtained.
        """
        from . import plot

        return plot.plot_species_spectra(
            self, Sfit, Taus, TauErr, isFixTau, modelType, ax, fig, **kwargs
        )

    # ----------------------------------------------------------------- #
    #                          Chirp correction                         #
    # ----------------------------------------------------------------- #
    def fit_chirp_automatic(self, **kwargs):
        """VARPRO coherent-artefact + Cauchy dispersion fit.

        See :func:`pymorgan.oneD.chirp.fit_chirp_automatic`.
        """
        from . import chirp

        return chirp.fit_chirp_automatic(self, **kwargs)

    def fit_chirp_step(self, **kwargs):
        """Gaussian(+derivatives)+erf-step fit, restricted to an early-time window.

        See :func:`pymorgan.oneD.chirp.fit_chirp_step`.
        """
        from . import chirp

        return chirp.fit_chirp_step(self, **kwargs)

    def fit_chirp_wavelet(self, **kwargs):
        """Wavelet-based edge-detection fit at every probe pixel + Cauchy dispersion fit.

        See :func:`pymorgan.oneD.chirp.fit_chirp_wavelet`.
        """
        from . import chirp

        return chirp.fit_chirp_wavelet(self, **kwargs)

    def fit_chirp_manual(self, points, **kwargs):
        """Cauchy dispersion fit to manually-picked ``(wavelength, delay)`` points.

        See :func:`pymorgan.oneD.chirp.fit_chirp_manual`.
        """
        from . import chirp

        return chirp.fit_chirp_manual(self, points, **kwargs)

    def plot_chirp_diagnostics(self, fit, **kwargs):
        """Review figures for a chirp fit. See :func:`pymorgan.oneD.chirp.plot_chirp_diagnostics`."""
        from . import chirp

        return chirp.plot_chirp_diagnostics(self, fit, **kwargs)

    def apply_chirp_correction(
        self, fit, detector: int | None = None, boundary_handling: str | None = None
    ) -> None:
        """Apply a loaded :class:`ChirpFit` to correct the dispersion in this dataset.

        Interpolates all active signals (raw, corrected, stdv, and single scans)
        for the given detector onto the same time-delay grid.

        Boundary handling options are:
        - "crop"        -> Option 2: Slices/crops the delays axis to keep only fully finite regions where all pixels are defined.
        - "extrapolate" -> Option 1: Fills out-of-bound values with the nearest valid boundary endpoint value (no NaNs).
        """
        import scipy.interpolate as interp

        import pymorgan.settings as settings_mod

        if boundary_handling is None:
            boundary_handling = settings_mod.get_settings().chirp_boundary_handling.value

        if detector is None:
            detector = getattr(fit, "detector", 0)

        probe_det = self._detector_probe(detector)
        t0fit = fit(probe_det)  # shape: [Npixels]
        Npixels = probe_det.size

        def interpolate_array_3d(arr):
            if arr is None:
                return None
            arr_new = arr.copy()
            if arr.ndim == 2:
                for i in range(Npixels):
                    if boundary_handling == "extrapolate":
                        f = interp.interp1d(
                            self.delays - t0fit[i],
                            arr[:, i],
                            kind="linear",
                            bounds_error=False,
                            fill_value=(arr[0, i], arr[-1, i]),
                        )
                    else:
                        f = interp.interp1d(
                            self.delays - t0fit[i],
                            arr[:, i],
                            kind="linear",
                            bounds_error=False,
                            fill_value=np.nan,
                        )
                    arr_new[:, i] = f(self.delays)
            elif arr.ndim == 3:
                for i in range(Npixels):
                    if boundary_handling == "extrapolate":
                        f = interp.interp1d(
                            self.delays - t0fit[i],
                            arr[:, i, detector],
                            kind="linear",
                            bounds_error=False,
                            fill_value=(arr[0, i, detector], arr[-1, i, detector]),
                        )
                    else:
                        f = interp.interp1d(
                            self.delays - t0fit[i],
                            arr[:, i, detector],
                            kind="linear",
                            bounds_error=False,
                            fill_value=np.nan,
                        )
                    arr_new[:, i, detector] = f(self.delays)
            return arr_new

        def interpolate_array_4d(arr):
            if arr is None:
                return None
            arr_new = arr.copy()
            Nscans = arr.shape[3] if arr.ndim == 4 else arr.shape[2]
            if arr.ndim == 3:
                for s in range(Nscans):
                    for i in range(Npixels):
                        if boundary_handling == "extrapolate":
                            f = interp.interp1d(
                                self.delays - t0fit[i],
                                arr[:, i, s],
                                kind="linear",
                                bounds_error=False,
                                fill_value=(arr[0, i, s], arr[-1, i, s]),
                            )
                        else:
                            f = interp.interp1d(
                                self.delays - t0fit[i],
                                arr[:, i, s],
                                kind="linear",
                                bounds_error=False,
                                fill_value=np.nan,
                            )
                        arr_new[:, i, s] = f(self.delays)
            elif arr.ndim == 4:
                for s in range(Nscans):
                    for i in range(Npixels):
                        if boundary_handling == "extrapolate":
                            f = interp.interp1d(
                                self.delays - t0fit[i],
                                arr[:, i, detector, s],
                                kind="linear",
                                bounds_error=False,
                                fill_value=(arr[0, i, detector, s], arr[-1, i, detector, s]),
                            )
                        else:
                            f = interp.interp1d(
                                self.delays - t0fit[i],
                                arr[:, i, detector, s],
                                kind="linear",
                                bounds_error=False,
                                fill_value=np.nan,
                            )
                        arr_new[:, i, detector, s] = f(self.delays)
            return arr_new

        self.Zavg_R = interpolate_array_3d(self.Zavg_R)
        if self.Zavg_C is not None:
            self.Zavg_C = interpolate_array_3d(self.Zavg_C)
        if self.Zstdv is not None:
            self.Zstdv = interpolate_array_3d(self.Zstdv)
        if self.Zss_R is not None:
            self.Zss_R = interpolate_array_4d(self.Zss_R)
        if self.Zss_C is not None:
            self.Zss_C = interpolate_array_4d(self.Zss_C)

        if boundary_handling == "crop":
            # Crop/slice delays axis to fully finite region (exclude introduced NaNs)
            t0_min = np.nanmin(t0fit) if np.any(np.isfinite(t0fit)) else 0.0
            t0_max = np.nanmax(t0fit) if np.any(np.isfinite(t0fit)) else 0.0
            valid_indices = (self.delays >= np.nanmin(self.delays) - t0_min) & (
                self.delays <= np.nanmax(self.delays) - t0_max
            )

            self.delays = self.delays[valid_indices]
            self.Zavg_R = self.Zavg_R[valid_indices, :, :]
            if self.Zavg_C is not None:
                self.Zavg_C = self.Zavg_C[valid_indices, :, :]
            if self.Zstdv is not None:
                self.Zstdv = self.Zstdv[valid_indices, :, :]
            if self.Zss_R is not None:
                self.Zss_R = self.Zss_R[valid_indices, :, :, :]
            if self.Zss_C is not None:
                self.Zss_C = self.Zss_C[valid_indices, :, :, :]

        self.chirp_fit = fit


def load_1D(path: str | PathLike, data_type: str = "PDAT", **loader_kwargs) -> Dataset1D:
    """Convenience wrapper for :meth:`Dataset1D.from_file`.

    Extra keyword arguments are forwarded to the selected loader (e.g.
    ``spectrum`` / ``slowmod`` / ``anisotropy`` for MESS_TRIR).
    """
    return Dataset1D.from_file(path, data_type=data_type, **loader_kwargs)

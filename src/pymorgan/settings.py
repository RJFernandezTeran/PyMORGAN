"""Centralised plotting/aesthetics settings for PyMORGAN.

This module is the single source of truth for presentation choices that are
*not* intrinsic to the data: the matplotlib style profile, font scaling, axis-
label convention, the delta-absorbance unit convention, the default colourmap
and the default time-axis scale.

Design notes
------------
* **Framework-agnostic.** Nothing here imports Qt. The PyQt6 GUI binds to the
  fields described by :func:`Settings.field_specs` (each field carries its type
  and, for enums, its allowed values), so combo boxes can be generated
  automatically. ``pymorgan`` never depends on the GUI; the GUI depends on
  ``pymorgan``.
* **Wraps the ``.mplstyle`` files, does not replace them.** A profile selects
  one of the tuned style files in ``plot_styles/`` and :meth:`Settings.apply`
  overlays a light ``font_scale`` multiplier on top.
* **matplotlib mathtext only.** :meth:`Settings.apply` always forces
  ``text.usetex = False``; LaTeX text rendering is never enabled.
* **Global default + per-call override.** A module-level *active* settings
  object is read by the plotting methods; individual calls may still override
  any field locally.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from enum import Enum, StrEnum
from pathlib import Path
from typing import Union, get_args, get_origin, get_type_hints

from pymorgan.log import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
#                                  Enums                                       #
# --------------------------------------------------------------------------- #


class LabelStyle(StrEnum):
    """Delimiter convention for axis labels, e.g. ``Wavelength (nm)``."""

    PAREN = "()"  # Wavelength (nm)
    BRACKET = "[]"  # Wavelength [nm]
    SLASH = "/"  # Wavelength / nm
    BARE = ""  # Wavelength nm


class DeltaAUnits(StrEnum):
    """Display convention for the (numerically identical) signal axis.

    The stored data is always in mOD; this toggle only changes the label.
    """

    MOD = "mOD"  # $\\Delta$A / mOD
    SCALED = "x1E3"  # $\\Delta$A $\\times 10^{3}$


class StyleProfile(StrEnum):
    """Named matplotlib style profile, mapped to a file in ``plot_styles/``."""

    REGULAR = "regular"
    POSTER = "poster"
    INSET = "inset"


class TimeAxisScale(StrEnum):
    """Default scaling of the delay (time) axis."""

    SYMLOG = "symlog"
    LOG = "log"
    LINEAR = "lin"


class TimeAxisLabel(StrEnum):
    """Tick-label format for the (log/symlog) delay axis of 1-D contour plots.

    Only affects how powers of ten are rendered; the scale itself is set by
    :class:`TimeAxisScale`.
      POWER   -> 10^0, 10^1, 10^2 ...
      DECIMAL -> 0.1, 1, 10, 100 ...
      MIXED   -> 1, 10^1, 10^2 ... (10^0 shown as a plain "1")
    """

    POWER = "power"
    DECIMAL = "decimal"
    MIXED = "mixed"


class FreqLabel(StrEnum):
    """2-D frequency-axis labelling convention (pump = omega_1, probe = omega_3)."""

    OMEGA_N = "Omega_n"  # omega_1 / omega_3
    OMEGA_PP = "Omega_PP"  # omega_pump / omega_probe
    PUMP_PROBE = "Pump-Probe"  # Pump / Probe wavenumber
    OMEGA_2PIC = "Omega_n/2pic"  # omega_n / 2 pi c0


CALIBRATION_SETUPS: tuple[str, ...] = (
    "UoS TRIR",
    "UniGE fsTA",
    "UniGE nsTA",
    "UZH Lab 2",
    "UniGE NIR-TA",
    "RAL LIFEtime (Absorbance)",
    "RAL LIFEtime (Intensity)",
    "UniGE TRIR (Intensity)",
    "UniGE TRIR (Absorbance)",
    "UniGE TRUVIS-II (Intensity)",
)


class XAxisUnit(StrEnum):
    """Display unit for the spectral (probe/pump) X axis of spectra/contours.

    ``NATIVE`` keeps each dataset's stored unit; the others convert the axis to
    nm / cm-1 / eV / THz (see :func:`helpers.convert_spectral`).
    """

    NATIVE = "native"
    NM = "nm"
    WAVENUMBER = "cm-1"
    EV = "eV"
    THZ = "THz"


class TwoDFreqUnit(StrEnum):
    """Display unit for the pump/probe axes of 2-D spectra.

    The data is stored in wavenumbers; the axes are converted on the fly (see
    :func:`helpers.to_display_axis`). Axes whose values exceed
    ``helpers.DISPLAY_SCALE_THRESHOLD`` are additionally divided by 1000 and the
    label gains a ``10^3`` prefix, so visible/UV maps read ``20`` rather than
    ``20000``.
    """

    WAVENUMBER = "cm-1"
    RECIPROCAL_INCH = "in-1"
    NM = "nm"
    THZ = "THz"
    EV = "eV"


class PumpAxis(StrEnum):
    """Orientation of the pump frequency axis: horizontal (default) or vertical."""

    HORIZONTAL = "Horizontal"
    VERTICAL = "Vertical"


class ChirpBoundaryHandling(StrEnum):
    """Boundary handling convention for chirp correction: crop or extrapolate."""

    CROP = "crop"
    EXTRAPOLATE = "extrapolate"


class ParallelFittingMode(StrEnum):
    """Execution mode for pixel-by-pixel chirp fitting."""

    THREADPOOL = "threadpool"
    PROCESSPOOL = "processpool"
    DISABLED = "disabled"


class OverlayDrawStyle(StrEnum):
    """Draw style for spectral diffusion (CLS/IvCLS/NLS) overlays."""

    POINTS = "Points"
    LINES = "Lines"
    BOTH = "Both"


class PeakType(StrEnum):
    """Target peak type for spectral diffusion analysis: min (bleach/SE) or max (ESA)."""

    MIN = "min"
    MAX = "max"


class PeakInterpolation(StrEnum):
    """Interpolation method to locate sub-pixel peaks."""

    QUADRATIC = "quadratic"
    CUBIC = "cubic"
    QUARTIC = "quartic"
    GAUSSIAN = "gaussian"
    LORENTZIAN = "lorentzian"
    NONE = "none"


class T2LabelStyle(StrEnum):
    """Format of the t₂ delay indicator text box in 2-D contour plots.

    T2_EQ  -> shows ``t₂ = 1.0 ps`` (full equation notation)
    PLAIN  -> shows just ``1.0 ps`` (number and unit only)
    """

    T2_EQ = "t2="
    PLAIN = "plain"


class TracesCmap(StrEnum):
    """Colourmap convention for kinetic and spectral trace plots.

    RAINBOW          -> rainbow
    RAINBOW_REVERSED -> rainbow reversed
    RED_PURPLE_BLUE  -> red-purple-blue
    """

    RAINBOW = "rainbow"
    RAINBOW_REVERSED = "rainbow reversed"
    RED_PURPLE_BLUE = "red-purple-blue"



# Profile -> .mplstyle filename (relative to the styles directory).
_PROFILE_FILES: dict[StyleProfile, str] = {
    StyleProfile.REGULAR: "HLV_plt.mplstyle",
    StyleProfile.POSTER: "HLV_plt_poster.mplstyle",
    StyleProfile.INSET: "HLV_plt_in.mplstyle",
}

# rcParams scaled by ``font_scale`` when it differs from 1.0.
_FONT_RCPARAMS = (
    "axes.titlesize",
    "axes.labelsize",
    "legend.fontsize",
    "xtick.labelsize",
    "ytick.labelsize",
)


# Fallback list of Qt styles offered in the settings panel. The GUI replaces it
# at runtime with the styles this machine actually has (see
# ``register_gui_themes``); a "Dark" suffix switches to the dark palette and
# unavailable styles fall back to Fusion (see pymorgan.gui.app.resolve_theme).
GUI_THEMES: tuple[str, ...] = ("Fusion", "Fusion Dark")
_gui_theme_choices: list[str] = list(GUI_THEMES)


def register_gui_themes(names) -> list[str]:
    """Publish the Qt styles available on this machine to the settings panel.

    Called by the GUI (which owns the Qt dependency) so ``field_specs`` can
    offer real choices instead of a hard-coded list. Keeps ``GUI_THEMES`` as the
    fallback for headless use.
    """
    global _gui_theme_choices
    seen: list[str] = []
    for name in list(names) + list(GUI_THEMES):
        if name and name not in seen:
            seen.append(str(name))
    _gui_theme_choices = seen
    return _gui_theme_choices


def gui_theme_choices() -> list[str]:
    """Qt styles currently offered in the settings panel."""
    return list(_gui_theme_choices)


# Legend placements offered for the kinetics / transient-spectra legends. All
# but the last are matplotlib ``loc`` values, i.e. inside the axis; "outside
# right" parks the legend beside it. Defined here rather than in the plotting
# module so the settings panel does not have to import Matplotlib.
LEGEND_LOCATIONS: tuple[str, ...] = (
    "best",
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "center right",
    "center left",
    "outside right",
)


def _legend_location_choices() -> list[str]:
    """Legend placements offered in the settings panel."""
    return list(LEGEND_LOCATIONS)


def _loader_choices(branch: str) -> list[str]:
    """Registered loader names for ``"oneD"``/``"twoD"``, for the defaults combos.

    Imported lazily: the loader registries import this module, so a top-level
    import would be circular.
    """
    try:
        if branch == "twoD":
            from pymorgan.twoD.load import available_map_loaders

            return list(available_map_loaders())
        from pymorgan.oneD.load import available_loaders

        return list(available_loaders())
    except Exception:  # pragma: no cover - defensive, keeps the panel usable
        return []


def _default_styles_dir() -> Path:
    """Locate the bundled ``plot_styles/`` directory."""
    return Path(__file__).resolve().parent / "plot_styles"


# --------------------------------------------------------------------------- #
#                    Field introspection / value coercion                     #
# --------------------------------------------------------------------------- #
# Serialisation (``to_dict`` / ``from_dict``), the ``__post_init__`` enum
# coercion and :func:`update_settings` are all derived from the dataclass fields
# and their annotations. A new setting therefore needs *one* edit -- the
# dataclass field -- plus an entry in :meth:`Settings.field_specs` if it should
# appear in the GUI settings panel.
_FIELD_TYPES: dict[str, type] | None = None


def _field_types() -> dict[str, type]:
    """Resolved annotations of :class:`Settings` (cached)."""
    global _FIELD_TYPES
    if _FIELD_TYPES is None:
        _FIELD_TYPES = get_type_hints(Settings)
    return _FIELD_TYPES


def _unwrap_optional(tp):
    """Return ``(base_type, is_optional)`` for ``X``, ``X | None`` or ``Optional[X]``."""
    if tp is None:
        return None, False
    origin = get_origin(tp)
    args = get_args(tp)
    if args and (origin is Union or type(tp).__name__ == "UnionType"):
        non_none = [a for a in args if a is not type(None)]
        base = non_none[0] if len(non_none) == 1 else None
        return base, len(non_none) != len(args)
    return tp, False


def _enum_field_types() -> dict[str, type[Enum]]:
    """``{field name: Enum subclass}`` for every enum-valued setting."""
    out = {}
    for name, tp in _field_types().items():
        base, _ = _unwrap_optional(tp)
        if isinstance(base, type) and issubclass(base, Enum):
            out[name] = base
    return out


def _coerce_field(name: str, value):
    """Coerce ``value`` to the declared type of field ``name``.

    Enums accept either a member or its string value; ``tuple[float, float]``
    fields accept any 2-sequence; ``Path`` fields accept strings. Unknown or
    un-annotated fields are passed through untouched.
    """
    base, optional = _unwrap_optional(_field_types().get(name))
    if value is None:
        return None
    if optional and isinstance(value, str) and not value.strip():
        return None  # an empty text field means "unset" (e.g. styles_dir)
    if isinstance(base, type) and issubclass(base, Enum):
        return value if isinstance(value, base) else base(value)
    if base is bool:
        return bool(value)
    if base is float:
        return float(value)
    if base is int:
        return int(value)
    if base is str:
        return str(value)
    if base is Path:
        return value if isinstance(value, Path) else Path(str(value))
    if get_origin(base) is tuple:
        args = get_args(base)
        conv = args[0] if args and isinstance(args[0], type) else float
        if isinstance(value, str):
            # GUI text input: "8.0, 6.0" / "(8, 6)" / "8 6"
            parts = [p for p in value.strip(" ()[]").replace(",", " ").split() if p]
        else:
            parts = list(value)
        if args and Ellipsis not in args and len(parts) != len(args):
            raise ValueError(f"{name} expects {len(args)} values, got {len(parts)}")
        return tuple(conv(v) for v in parts)
    if optional and base is None:
        return value
    return value


# --------------------------------------------------------------------------- #
#                                 Settings                                     #
# --------------------------------------------------------------------------- #


@dataclass
class Settings:
    """Container for all presentation-level settings.

    All fields are plain values or enums so the object round-trips cleanly to
    TOML and is trivially introspectable by the GUI.
    """

    profile: StyleProfile = StyleProfile.REGULAR
    font_scale: float = 1.0
    label_style: LabelStyle = LabelStyle.PAREN
    delta_a_units: DeltaAUnits = DeltaAUnits.SCALED
    cmap: str = "DkRd/Wh/DkBu"
    traces_cmap: TracesCmap = TracesCmap.RAINBOW
    white_levels: float = 2.0
    asinh_pct: float = 5.0
    n_contours: int = 40
    show_uncertainties: bool = True
    round_uncertainties: bool = True
    time_axis_scale: TimeAxisScale = TimeAxisScale.SYMLOG


    time_axis_label: TimeAxisLabel = TimeAxisLabel.POWER
    freq_label: FreqLabel = FreqLabel.OMEGA_N
    pump_axis: PumpAxis = PumpAxis.HORIZONTAL
    # Format of the t₂ delay text box in 2-D contour plots.
    # T2_EQ -> "t₂ = 1.0 ps";  PLAIN -> "1.0 ps"
    t2_label_style: T2LabelStyle = T2LabelStyle.T2_EQ
    # Display unit for the pump/probe axes of 2-D spectra (the data itself stays
    # in wavenumbers). Axes above helpers.DISPLAY_SCALE_THRESHOLD are shown
    # divided by 1000, with a 10^3 prefix in the label.
    twoD_freq_unit: TwoDFreqUnit = TwoDFreqUnit.WAVENUMBER
    # Display unit for the spectral (probe) X axis of spectra/contour plots.
    # NATIVE keeps the dataset's stored unit; otherwise the axis is converted.
    x_axis_unit: XAxisUnit = XAxisUnit.NATIVE
    # When True, spectra/contour plots get a secondary X axis (top) showing the
    # complementary unit (cm-1<->nm; nm for eV/THz). See helpers.complementary_unit.
    secondary_axis: bool = False
    round_labels: bool = True
    # When True, 1-D and 2-D contour plots are drawn as a single pcolormesh
    # image instead of a filled contour set, and the contour-line overlay is
    # skipped. Redraws are much faster -- dragging the colour-scale slider,
    # stepping through t2 delays -- at the cost of the contour appearance, so
    # turn it off before exporting figures. Individual calls can override it
    # with ``plot_contour(..., quick=False)``.
    quick_plots: bool = False
    # When True, kinetic/spectral cut figures opened from the GUI inherit the
    # embedded contour's axis limits: spectra X <- probe limits, kinetics X <-
    # delay limits, and both cuts' signal (Y) axis <- the contour Z (ΔA) limits.
    inherit_cut_limits: bool = True
    # When True, the MESS_TRIR reader loads the per-scan arrays from the
    # dataset's temp/ folder into Zss_R (single-scan exploration); off by
    # default to save memory and load time.
    load_single_scans: bool = True
    # Kinetic-trace marker size and line width (matplotlib points).
    kinetics_marker_size: float = 4.0
    kinetics_line_width: float = 1.5
    # Significant digits kept in a legend label (delays, lifetimes). With
    # ``round_labels`` on the value is first rounded to two significant figures;
    # this is what stops an unrounded fit result printing all its decimals.
    label_digits: int = 3
    # Vertical spacing between legend entries (matplotlib ``labelspacing``, in
    # font-size units) for the draggable kinetics / transient-spectra legends.
    legend_label_spacing: float = 0.25
    # Where the kinetics / transient-spectra legend sits. "outside right" parks
    # it beside the axis, which is what a stand-alone figure wants and the
    # long-standing PyMORGAN look; any matplotlib ``loc`` ("best",
    # "upper right", ...) puts it inside instead. A host drawing into a small
    # embedded panel overrides this per call -- there is no room beside the axis
    # there, and an outside legend would be drawn over its neighbour.
    legend_location: str = "outside right"
    # Default figure size (width, height in inches) for *independent* (new-figure)
    # 1-D plots. They are ignored when a plotter draws into a supplied axis
    # (e.g. the GUI embedded preview); only the stand-alone figures use them.
    contour_figsize: tuple[float, float] = (7.5, 5.5)
    kinetics_figsize: tuple[float, float] = (7.5, 4.375)
    spectra_figsize: tuple[float, float] = (7.5, 4.375)
    # Steady-state (0-D) overlay style for transient-spectra plots. The
    # absorption/emission spectra are drawn on a twin axis as a dashed line
    # plus a shaded fill; defaults reproduce the original PumpProbe scripts
    # (blue absorption, red emission, faint line + fainter fill).
    ss_abs_color: str = "b"
    ss_em_color: str = "r"
    ss_line_width: float = 1.5
    ss_line_alpha: float = 0.15
    ss_fill_alpha: float = 0.05
    ss_line_dashes: tuple[float, float] = (4.0, 3.0)
    # Qt style/theme for the GUI window (see settings.toml for the options).
    gui_theme: str = "Fusion"
    # GUI splash-screen duration in seconds (0 disables the splash).
    splash_duration: float = 0.0
    # Default data directory for the GUI: pre-fills the root folder and is the
    # base for the "Today" button, which appends /YYYYMMDD to it.
    default_datadir: str = ""
    default_oneD_datatype: str = "MESS_TRIR"
    default_twoD_datatype: str = "MESS_2DIR"
    default_calibration_datatype: str = "UniGE TRIR (Intensity)"
    # Optional override for the directory holding the .mplstyle files.
    styles_dir: Path | None = None
    # How to handle boundary NaNs introduced by chirp correction (extrapolate or crop)
    chirp_boundary_handling: ChirpBoundaryHandling = ChirpBoundaryHandling.CROP
    # Execution mode for pixel-by-pixel chirp fitting (threadpool, processpool, or disabled)
    parallel_fitting: ParallelFittingMode = ParallelFittingMode.PROCESSPOOL
    # Whether to prune the legend to keep the legend list within the figure.
    prune_legend: bool = True
    # Maximum number of legend entries to show if pruning is enabled.
    max_legend_entries: float = 15.0
    # Whether to use a per-pixel time-zero offset (dt) for solvent subtraction.
    solvent_per_pixel_dt: bool = False
    # Whether to use a per-pixel amplitude scaling factor for solvent subtraction.
    solvent_per_pixel_scale: bool = True
    # Whether to include shaded fills to +/- one standard deviation in 1-D plots.
    error_shading: bool = False
    # Opacity (alpha) of the standard deviation shaded fills.
    error_shading_alpha: float = 0.2
    # Style settings for 2D CLS/IvCLS/NLS overlays
    cls_color: str = "w"
    ivcls_color: str = "y"
    nls_color: str = "r"
    overlay_draw_style: OverlayDrawStyle = OverlayDrawStyle.BOTH
    overlay_linewidth: float = 1.5
    # CLS/NLS/IvCLS peak detection parameters
    sd_intensity_threshold: float = 0.5
    sd_peak_range: float = 30.0
    sd_peak_type: PeakType = PeakType.MIN
    sd_interpolation: PeakInterpolation = PeakInterpolation.QUADRATIC
    sd_interpolation_factor: int = 2  # axis upsampling factor for sub-pixel peak finding (1 = no upsampling)
    kubo_ftir_weight: float = 1.0  # relative weight of FTIR vs 2D fit in joint optimisation

    def __post_init__(self) -> None:
        """Coerce string values (e.g. straight from TOML) into their enum members."""
        if isinstance(self.traces_cmap, str):
            val = self.traces_cmap.lower().strip()
            if val in ("rainbow_reversed", "rainbow-reversed", "rainbow reversed", "rainbow_r"):
                self.traces_cmap = TracesCmap.RAINBOW_REVERSED
            elif val in ("red_purple_blue", "red-purple-blue", "red purple blue", "redpurpleblue"):
                self.traces_cmap = TracesCmap.RED_PURPLE_BLUE
            elif val in ("rainbow", "rainbow"):
                self.traces_cmap = TracesCmap.RAINBOW
        for name, enum_cls in _enum_field_types().items():
            value = getattr(self, name)
            if value is not None and not isinstance(value, enum_cls):
                setattr(self, name, enum_cls(value))



    # ------------------------------------------------------------------ #
    #                           Style application                        #
    # ------------------------------------------------------------------ #
    @property
    def style_path(self) -> Path:
        """Absolute path to the ``.mplstyle`` file for the active profile."""
        base = self.styles_dir or _default_styles_dir()
        return Path(base) / _PROFILE_FILES[self.profile]

    def apply(self) -> None:
        """Apply the profile to matplotlib's global rcParams.

        Resets to the profile's ``.mplstyle``, overlays the ``font_scale``
        multiplier, and forces ``text.usetex = False`` (mathtext only).
        """
        import logging

        import matplotlib.pyplot as plt

        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

        plt.style.use(str(self.style_path))

        # Warn once if none of the preferred fonts are available.
        from pymorgan.fonts import PREFERRED_FONTS, available_preferred_fonts

        if not available_preferred_fonts():
            import warnings

            warnings.warn(
                f"None of the preferred fonts {PREFERRED_FONTS} were found by Matplotlib. "
                "Run `pymorgan-install-fonts` to install TeX Gyre Heros.",
                stacklevel=3,
            )

        if self.font_scale != 1.0:
            for key in _FONT_RCPARAMS:
                try:
                    plt.rcParams[key] = float(plt.rcParams[key]) * self.font_scale
                except (TypeError, ValueError):
                    # Non-numeric sizes (e.g. 'medium') are left untouched.
                    pass

        # PyMORGAN never uses a LaTeX backend for text rendering.
        plt.rcParams["text.usetex"] = False

    # ------------------------------------------------------------------ #
    #                        Units-convention helper                     #
    # ------------------------------------------------------------------ #
    def units_with_convention(self, units: dict, delta_a_units: DeltaAUnits | None = None) -> dict:
        """Return a copy of ``units`` with the delta-A label convention applied.

        Only the signal-axis *labels* are changed; the data is never rescaled.
        Non delta-absorbance signals (e.g. intensity) are returned unchanged.
        """
        choice = delta_a_units or self.delta_a_units
        out = dict(units)
        if out.get("unitsZ") in ("mOD", "x1E3"):
            out["unitsZ_lbl"] = r"$\Delta$A"
            if choice is DeltaAUnits.MOD:
                out["unitsZ_ltx"] = r"mOD"
                out["unitsZ"] = "mOD"
            else:
                out["unitsZ_ltx"] = r"$\times 10^{3}$"
                out["unitsZ"] = "x1E3"
        return out

    # ------------------------------------------------------------------ #
    #                         Serialisation (TOML)                       #
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        """Return a JSON/TOML-friendly mapping of the settings.

        Derived from the dataclass fields: enums are written as their value,
        paths and tuples as strings / lists. Optional fields that are ``None``
        (``styles_dir``) are omitted so they keep their default
        when read back.
        """
        data: dict = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            if isinstance(value, Enum):
                value = value.value
            elif isinstance(value, Path):
                value = str(value)
            elif isinstance(value, tuple):
                value = list(value)
            data[f.name] = value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Settings:
        """Build a :class:`Settings` from a mapping, ignoring unknown keys.

        Each value is coerced to the field's declared type (see
        :func:`_coerce_field`); ``None`` and empty optional paths fall back to
        the field default.
        """
        kwargs: dict = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            value = data[f.name]
            if value is None:
                continue
            base, optional = _unwrap_optional(_field_types().get(f.name))
            if optional and isinstance(value, str) and not value.strip():
                continue  # e.g. styles_dir = "" means "use the bundled styles"
            kwargs[f.name] = _coerce_field(f.name, value)
        # ``data_dir`` was an older name for ``default_datadir``; honour it so
        # existing settings.toml files keep working.
        legacy = data.get("data_dir")
        if legacy and not kwargs.get("default_datadir"):
            kwargs["default_datadir"] = str(legacy)
        return cls(**kwargs)

    def save(self, path: str | Path) -> None:
        """Write the settings to ``path`` as TOML, preserving any comments."""
        import tomlkit

        path = Path(path)
        if path.exists():
            doc = tomlkit.parse(path.read_text(encoding="utf-8"))
        else:
            try:
                doc = tomlkit.parse(get_default_settings_text())
            except Exception:
                doc = tomlkit.document()

        sections = self.field_sections()
        for key, value in self.to_dict().items():
            section = sections.get(key)
            if section is None:
                doc[key] = value  # unsectioned key: keep it where it was
                continue
            if section not in doc:
                doc[section] = tomlkit.table()
            doc[section][key] = value
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tomlkit.dumps(doc), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path | None = None) -> Settings:
        """Read a :class:`Settings` from a TOML file.

        Ensures the settings file exists (creating from defaults if absent) and
        merges any newly introduced settings while keeping user modifications.
        Accepts both the sectioned layout (``[plot]``, ``[axes]``, ...) and the
        older flat one, so an existing settings.toml keeps working.
        """
        import tomlkit

        target = ensure_settings_file(path)
        data = tomlkit.parse(target.read_text(encoding="utf-8"))
        flat: dict = {}
        for key, value in dict(data).items():
            if hasattr(value, "items"):
                flat.update(dict(value))
            else:
                flat.setdefault(key, value)
        return cls.from_dict(flat)

    # ------------------------------------------------------------------ #
    #                          GUI introspection                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def field_sections() -> dict[str, str]:
        """Which section of ``settings.toml`` each field is written to.

        The grouping is the one the file already used as comment headers, so a
        hand-edited file keeps its shape when the GUI saves over it. A field
        missing from this map is written at the top level, which an older flat
        file also produces and :meth:`load` still reads.
        """
        return {
            # [plot]
            "profile": "plot",
            "font_scale": "plot",
            "label_style": "plot",
            "delta_a_units": "plot",
            "cmap": "plot",
            "traces_cmap": "plot",
            "white_levels": "plot",
            "n_contours": "plot",
            "show_uncertainties": "plot",
            "round_uncertainties": "plot",
            "secondary_axis": "plot",


            "kinetics_marker_size": "plot",
            "kinetics_line_width": "plot",
            "label_digits": "plot",
            "legend_label_spacing": "plot",
            "legend_location": "plot",
            "prune_legend": "plot",
            "max_legend_entries": "plot",
            "error_shading": "plot",
            "error_shading_alpha": "plot",
            # [axes]
            "time_axis_scale": "axes",
            "time_axis_label": "axes",
            "freq_label": "axes",
            "pump_axis": "axes",
            "twoD_freq_unit": "axes",
            "t2_label_style": "axes",
            "x_axis_unit": "axes",
            "quick_plots": "axes",
            "round_labels": "axes",
            "inherit_cut_limits": "axes",
            # [figures]
            "contour_figsize": "figures",
            "kinetics_figsize": "figures",
            "spectra_figsize": "figures",
            # [gui]
            "gui_theme": "gui",
            "splash_duration": "gui",
            "default_datadir": "gui",
            "default_oneD_datatype": "gui",
            "default_twoD_datatype": "gui",
            "default_calibration_datatype": "gui",
            "load_single_scans": "gui",
            # [chirp]
            "chirp_boundary_handling": "chirp",
            "parallel_fitting": "chirp",
            "solvent_per_pixel_dt": "chirp",
            "solvent_per_pixel_scale": "chirp",
            # [spectral_diffusion]
            "cls_color": "spectral_diffusion",
            "ivcls_color": "spectral_diffusion",
            "nls_color": "spectral_diffusion",
            "overlay_draw_style": "spectral_diffusion",
            "overlay_linewidth": "spectral_diffusion",
            "sd_intensity_threshold": "spectral_diffusion",
            "sd_peak_range": "spectral_diffusion",
            "sd_peak_type": "spectral_diffusion",
            "sd_interpolation": "spectral_diffusion",
            "sd_interpolation_factor": "spectral_diffusion",
            "kubo_ftir_weight": "spectral_diffusion",
            # [steady_state]
            "ss_abs_color": "steady_state",
            "ss_em_color": "steady_state",
            "ss_line_width": "steady_state",
            "ss_line_alpha": "steady_state",
            "ss_fill_alpha": "steady_state",
            "ss_line_dashes": "steady_state",
        }

    @classmethod
    def field_specs(cls) -> dict[str, dict]:
        """Describe the editable fields for GUI widget generation.

        Each entry gives a human label, the widget kind, and (for choices) the
        allowed values, so a PyQt6 front-end can build combo boxes / spin boxes
        without hard-coding the option lists. An optional ``"tooltip"`` is shown
        on the generated widget.
        """
        return {
            "profile": {
                "label": "Style profile",
                "kind": "choice",
                "choices": [e.value for e in StyleProfile],
                "tab": "common",
            },
            "font_scale": {
                "label": "Font scale",
                "kind": "float",
                "min": 0.5,
                "max": 3.0,
                "step": 0.05,
                "tab": "common",
            },
            "label_style": {
                "label": "Axis-label style",
                "kind": "choice",
                "choices": [e.value for e in LabelStyle],
                "tab": "common",
            },
            "delta_a_units": {
                "label": "ΔA units",
                "kind": "choice",
                "choices": [e.value for e in DeltaAUnits],
                "tab": "common",
            },
            "cmap": {
                "label": "Colourmap",
                "kind": "choice",
                "choices": ["DkRd/Wh/DkBu", "Rd/Wh/Bu v2", "Seismic", "Jet", "vik", "berlin"],
                "tab": "common",
            },
            "traces_cmap": {
                "label": "Trace colourmap",
                "kind": "choice",
                "choices": [e.value for e in TracesCmap],
                "tab": "common",
                "tooltip": "Colourmap used for kinetic and spectral trace plots.",
            },
            "white_levels": {
                "label": "White/Zero levels",
                "kind": "float",
                "min": 0.0,
                "max": 20.0,
                "step": 1.0,
                "tab": "common",
            },
            "asinh_pct": {
                "label": "arcsinh linear %",
                "kind": "float",
                "min": 0.1,
                "max": 100.0,
                "step": 1.0,
                "tab": "common",
                "tooltip": "Linear threshold for arcsinh color scaling as a percentage of the active amplitude limit (Zscale).",
            },
            "n_contours": {
                "label": "Number of contours",
                "kind": "int",
                "min": 2,
                "max": 500,
                "tab": "common",
            },
            "show_uncertainties": {
                "label": "Show 1-σ uncertainties",
                "kind": "bool",
                "tab": "common",
            },
            "round_uncertainties": {
                "label": "Round to uncertainty",
                "kind": "bool",
                "tab": "common",
            },


            "x_axis_unit": {
                "label": "Spectral X-axis unit",
                "kind": "choice",
                "choices": [e.value for e in XAxisUnit],
                "tab": "common",
            },
            "secondary_axis": {
                "label": "Complementary X axis",
                "kind": "bool",
                "tab": "common",
            },
            "round_labels": {
                "label": "Round trace labels",
                "kind": "bool",
                "tab": "common",
            },
            "quick_plots": {
                "label": "Use quick plots (fast redraws)",
                "kind": "bool",
                "tab": "common",
                "tooltip": (
                    "Draw contour plots as a single image (pcolormesh) instead of a filled "
                    "contour set, and skip the contour lines.\nMuch faster to redraw, but it "
                    "overrides the 'Filled' and 'Show lines' options in the plot controls, so "
                    "turn it off for final figures."
                ),
            },
            "time_axis_scale": {
                "label": "Time-axis scale",
                "kind": "choice",
                "choices": [e.value for e in TimeAxisScale],
                "tab": "oneD",
            },
            "time_axis_label": {
                "label": "Time-axis labels",
                "kind": "choice",
                "choices": [e.value for e in TimeAxisLabel],
                "tab": "oneD",
            },
            "freq_label": {
                "label": "2-D frequency labels",
                "kind": "choice",
                "choices": [e.value for e in FreqLabel],
                "tab": "twoD",
            },
            "pump_axis": {
                "label": "2-D pump axis orientation",
                "kind": "choice",
                "choices": [e.value for e in PumpAxis],
                "tab": "twoD",
            },
            "twoD_freq_unit": {
                "label": "2-D axis unit",
                "kind": "choice",
                "choices": [e.value for e in TwoDFreqUnit],
                "tab": "twoD",
                "tooltip": (
                    "Unit of the pump/probe axes of 2-D spectra. The stored data stays in "
                    "wavenumbers.\nAxes reaching more than 5000 (visible/UV) are shown divided "
                    "by 1000, with a 10^3 prefix in the label."
                ),
            },
            "t2_label_style": {
                "label": "t₂ delay label format",
                "kind": "choice",
                "choices": [e.value for e in T2LabelStyle],
                "tab": "twoD",
            },
            "kinetics_marker_size": {
                "label": "Kinetics marker size",
                "kind": "float",
                "min": 0.0,
                "max": 30.0,
                "step": 0.5,
                "tab": "oneD",
            },
            "kinetics_line_width": {
                "label": "Kinetics line width",
                "kind": "float",
                "min": 0.0,
                "max": 10.0,
                "step": 0.25,
                "tab": "oneD",
            },
            "label_digits": {
                "label": "Legend digits",
                "kind": "int",
                "min": 1,
                "max": 8,
                "tab": "oneD",
            },
            "legend_label_spacing": {
                "label": "Legend entry spacing",
                "kind": "float",
                "min": 0.0,
                "max": 5.0,
                "step": 0.1,
                "tab": "oneD",
            },
            "legend_location": {
                "label": "Legend position",
                "kind": "choice",
                "choices": list(_legend_location_choices()),
                "tab": "oneD",
            },
            "prune_legend": {
                "label": "Prune legend",
                "kind": "bool",
                "tab": "oneD",
            },
            "max_legend_entries": {
                "label": "Max legend entries",
                "kind": "float",
                "min": 1.0,
                "max": 100.0,
                "step": 1.0,
                "tab": "oneD",
            },
            "ss_abs_color": {
                "label": "Steady-state abs. colour",
                "kind": "text",
                "tab": "oneD",
            },
            "ss_em_color": {
                "label": "Steady-state em. colour",
                "kind": "text",
                "tab": "oneD",
            },
            "ss_line_width": {
                "label": "Steady-state line width",
                "kind": "float",
                "min": 0.0,
                "max": 10.0,
                "step": 0.25,
                "tab": "oneD",
            },
            "ss_line_alpha": {
                "label": "Steady-state line alpha",
                "kind": "float",
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "tab": "oneD",
            },
            "ss_fill_alpha": {
                "label": "Steady-state fill alpha",
                "kind": "float",
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "tab": "oneD",
            },
            "inherit_cut_limits": {
                "label": "Cuts inherit plot limits",
                "kind": "bool",
                "tab": "oneD",
            },
            "load_single_scans": {
                "label": "Load single scans",
                "kind": "bool",
                "tab": "oneD",
            },
            "chirp_boundary_handling": {
                "label": "Chirp boundary handling",
                "kind": "choice",
                "choices": [e.value for e in ChirpBoundaryHandling],
                "tab": "oneD",
            },
            "parallel_fitting": {
                "label": "Chirp parallel fitting",
                "kind": "choice",
                "choices": [e.value for e in ParallelFittingMode],
                "tab": "oneD",
            },
            "solvent_per_pixel_dt": {
                "label": "Per-pixel solvent t₀ offset",
                "kind": "bool",
                "tab": "oneD",
            },
            "solvent_per_pixel_scale": {
                "label": "Per-pixel solvent amplitude scale",
                "kind": "bool",
                "tab": "oneD",
            },
            "error_shading": {
                "label": "Error shading (±1 std)",
                "kind": "bool",
                "tab": "oneD",
            },
            "error_shading_alpha": {
                "label": "Error shading alpha",
                "kind": "float",
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "tab": "oneD",
            },
            "cls_color": {
                "label": "CLS colour",
                "kind": "text",
                "tab": "twoD",
            },
            "ivcls_color": {
                "label": "IvCLS colour",
                "kind": "text",
                "tab": "twoD",
            },
            "nls_color": {
                "label": "NLS colour",
                "kind": "text",
                "tab": "twoD",
            },
            "overlay_draw_style": {
                "label": "Overlay draw style",
                "kind": "choice",
                "choices": [e.value for e in OverlayDrawStyle],
                "tab": "twoD",
            },
            "overlay_linewidth": {
                "label": "Overlay linewidth",
                "kind": "float",
                "min": 0.5,
                "max": 10.0,
                "step": 0.25,
                "tab": "twoD",
            },
            "sd_intensity_threshold": {
                "label": "Relative intensity threshold",
                "kind": "float",
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "tab": "twoD",
                "tooltip": (
                    "Minimum relative amplitude threshold for peak fitting, expressed as a "
                    "fraction of maximum 2D intensity (0.0 to 1.0)."
                ),
            },
            "sd_peak_range": {
                "label": "Peak search range (cm⁻¹)",
                "kind": "float",
                "min": 0.0,
                "max": 100.0,
                "step": 1.0,
                "tab": "twoD",
            },
            "sd_peak_type": {
                "label": "Target peak type",
                "kind": "choice",
                "choices": [e.value for e in PeakType],
                "tab": "twoD",
            },
            "sd_interpolation": {
                "label": "Peak interpolation",
                "kind": "choice",
                "choices": [e.value for e in PeakInterpolation],
                "tab": "twoD",
            },
            "sd_interpolation_factor": {
                "label": "Interpolation factor (axis upsampling)",
                "kind": "int",
                "min": 1,
                "max": 64,
                "step": 1,
                "tab": "twoD",
            },
            "kubo_ftir_weight": {
                "label": "Kubo joint fit FTIR vs 2D weight",
                "kind": "float",
                "min": 0.0,
                "max": 100.0,
                "step": 0.1,
                "tab": "twoD",
            },
            # --- Application defaults (GUI tab) ------------------------------ #
            "gui_theme": {
                "label": "GUI theme",
                "kind": "choice",
                "choices": gui_theme_choices(),
                "tab": "gui",
            },
            "splash_duration": {
                "label": "Splash-screen duration (s, 0 disables)",
                "kind": "float",
                "min": 0.0,
                "max": 30.0,
                "step": 0.5,
                "tab": "gui",
            },
            "default_datadir": {
                "label": "Default data directory",
                "kind": "text",
                "tab": "gui",
            },
            "default_oneD_datatype": {
                "label": "Default 1D data type",
                "kind": "choice",
                "choices": _loader_choices("oneD"),
                "tab": "gui",
            },
            "default_twoD_datatype": {
                "label": "Default 2D data type",
                "kind": "choice",
                "choices": _loader_choices("twoD"),
                "tab": "gui",
            },
            "default_calibration_datatype": {
                "label": "Default calibration setup",
                "kind": "choice",
                "choices": CALIBRATION_SETUPS,
                "tab": "gui",
            },
            "styles_dir": {
                "label": "Style directory (blank = bundled)",
                "kind": "text",
                "tab": "gui",
            },
            # --- Figure sizes, in inches (width, height) --------------------- #
            "contour_figsize": {
                "label": "Contour figure size (w, h in)",
                "kind": "text",
                "tab": "gui",
            },
            "kinetics_figsize": {
                "label": "Kinetics figure size (w, h in)",
                "kind": "text",
                "tab": "gui",
            },
            "spectra_figsize": {
                "label": "Spectra figure size (w, h in)",
                "kind": "text",
                "tab": "gui",
            },
            "ss_line_dashes": {
                "label": "Steady-state dash pattern (on, off)",
                "kind": "text",
                "tab": "common",
            },
        }


# --------------------------------------------------------------------------- #
#                          Module-level active settings                       #
# --------------------------------------------------------------------------- #
_active: Settings = Settings()


def get_default_settings_text() -> str:
    """Return the raw text of the canonical default settings template."""
    pkg_default = Path(__file__).resolve().parent / "settings.default.toml"
    if pkg_default.is_file():
        return pkg_default.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Could not locate '{pkg_default}'.")


def get_default_settings_doc():
    """Parse and return the canonical default settings template as a tomlkit document."""
    import tomlkit

    return tomlkit.parse(get_default_settings_text())


def settings_path() -> Path:
    """Locate or determine the standard path for ``settings.toml``.

    Checks the current working directory first. If not found, checks the
    project root (if running from a repository source tree). Defaults to the
    repository root (if in a repo) or ``./settings.toml``.
    """
    cfg = Path("settings.toml")
    if cfg.is_file():
        return cfg.resolve()
    repo_root = Path(__file__).resolve().parent.parent.parent
    if (repo_root / "pyproject.toml").is_file() or (repo_root / ".git").is_dir():
        return (repo_root / "settings.toml").resolve()
    return cfg.resolve()


def ensure_settings_file(path: str | Path | None = None) -> Path:
    """Ensure that ``path`` (defaulting to ``settings_path()``) exists and is up to date.

    If the file does not exist, it is generated from ``settings.default.toml``.
    If the file exists, it is checked for any newly introduced settings from the
    default template. If missing keys are found, they are merged into the file
    along with their section comments while preserving all existing user modifications.
    """
    import tomlkit

    target = Path(path).resolve() if path is not None else settings_path()
    default_text = get_default_settings_text()

    if not target.exists():
        import warnings

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(default_text, encoding="utf-8")
        warnings.warn(
            f"Created default settings file at '{target}'. "
            "You can run `pymorgan-settings` to configure your preferences interactively, "
            "or edit this file directly.",
            UserWarning,
            stacklevel=2,
        )
        logger.info("Wrote default settings file to %s", target)
        return target

    try:
        user_doc = tomlkit.parse(target.read_text(encoding="utf-8"))
    except Exception:
        return target

    default_doc = tomlkit.parse(default_text)

    # Check whether any sections or keys from default_doc are missing in user_doc
    needs_merge = False
    for sec_name, sec_val in default_doc.items():
        if hasattr(sec_val, "items"):
            if sec_name not in user_doc:
                needs_merge = True
                break
            user_sec = user_doc[sec_name]
            if not hasattr(user_sec, "__getitem__"):
                needs_merge = True
                break
            for k in sec_val:
                if k not in user_sec:
                    needs_merge = True
                    break
        else:
            if sec_name not in user_doc:
                needs_merge = True
                break

    if needs_merge:
        # Merge by taking default_doc and overlaying all user-defined values and tables
        merged = tomlkit.parse(default_text)
        sections = Settings.field_sections()
        for key, val in user_doc.items():
            if hasattr(val, "items"):
                if key not in merged:
                    merged[key] = val
                else:
                    for sub_k, sub_v in val.items():
                        merged[key][sub_k] = sub_v
            else:
                # Handle unsectioned/flat key: place in appropriate section if known
                sec = sections.get(key)
                if sec and sec in merged and hasattr(merged[sec], "__setitem__"):
                    merged[sec][key] = val
                else:
                    merged[key] = val
        target.write_text(tomlkit.dumps(merged), encoding="utf-8")

    return target


def get_settings() -> Settings:
    """Return the active :class:`Settings` object."""
    return _active


def set_settings(settings: Settings) -> Settings:
    """Replace the active settings and return it."""
    global _active
    _active = settings
    return _active


def update_settings(**changes) -> Settings:
    """Mutate the active settings in place via keyword overrides.

    Enum fields accept either an enum member or its string value (the GUI
    settings panel passes the string). The mapping of field to enum class is
    derived from the dataclass annotations, so it cannot fall out of date.
    """
    global _active
    known = {f.name for f in fields(Settings)}
    coerced: dict = {}
    for key, value in changes.items():
        if key not in known:
            raise TypeError(f"unknown setting: {key!r}")
        coerced[key] = _coerce_field(key, value)
    _active = replace(_active, **coerced)
    return _active


@contextmanager
def use_settings(**changes) -> Iterator[Settings]:
    """Temporarily override the active settings within a ``with`` block."""
    global _active
    previous = _active
    try:
        yield update_settings(**changes)
    finally:
        _active = previous


def load_settings(path: str | Path | None = None) -> Settings:
    """Load settings from ``path`` (defaulting to ``settings_path()``) and make them active."""
    return set_settings(Settings.load(path))


def save_settings(path: str | Path, settings: Settings | None = None) -> None:
    """Write ``settings`` (or the active settings) to ``path`` as TOML."""
    (settings or _active).save(path)


def apply_style(settings: Settings | None = None) -> None:
    """Apply ``settings`` (or the active settings) to matplotlib."""
    (settings or _active).apply()


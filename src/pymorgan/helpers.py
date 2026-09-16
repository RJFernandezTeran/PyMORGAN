import re
from colorsys import hls_to_rgb, rgb_to_hls
from dataclasses import dataclass

import numpy as np

from pymorgan.log import get_logger

logger = get_logger(__name__)


def find_nearest(X, q, unique=False):
    """
    Find the closest element in vector X from the query vector q.

    Parameters:
    - X: numpy array, the vector to search in.
    - q: scalar or numpy array, the query vector.

    Returns:
    - Xq: scalar or numpy array, the closest element(s) in X to q.
    - ID: integer or numpy array, the index(es) of the closest element(s) in X.
    """
    # Ensure both X and q are numpy arrays
    X = np.array(X)
    q = np.array(q)

    # Initialize arrays to store the closest elements and indices
    Xq = np.zeros_like(q)
    ID = np.zeros_like(q, dtype=int)

    # Iterate over each element in q and find the closest element in X
    for i, q_i in enumerate(q):
        # Calculate the absolute differences between X and the current query element
        # Find the index of the minimum difference
        index = np.argmin(np.abs(X - q_i))

        # Store the closest element and its index
        Xq[i] = X[index]
        ID[i] = index
    # Filter the arrays to keep only unique elements (first occurrence wins, so
    # the caller's requested order is preserved).
    if unique:
        Xq = _unique_stable(Xq)
        ID = _unique_stable(ID)

    return Xq, ID


def _unique_stable(a: np.ndarray) -> np.ndarray:
    """Unique values of ``a`` in order of first appearance (``pandas.unique``)."""
    a = np.asarray(a)
    _, idx = np.unique(a, return_index=True)
    return a[np.sort(idx)]


def setXYlabels(
    where, labelStyle, XUnits, YUnits, normX=False, normY=False, setXLabel=True, setYLabel=True
):
    if setXLabel:
        if normX:
            where.set_xlabel(r"Norm. %s" % (XUnits["lbl"]))
        else:
            match labelStyle:
                case "()":
                    where.set_xlabel(r"%s (%s)" % (XUnits["lbl"], XUnits["ltx"]))
                case "[]":
                    where.set_xlabel(r"%s [%s]" % (XUnits["lbl"], XUnits["ltx"]))
                case "/":
                    where.set_xlabel(r"%s / %s" % (XUnits["lbl"], XUnits["ltx"]))

    if YUnits["ltx"] == r"$\times 10^{3}$":
        labelStyle = ""

    if setYLabel:
        if normY:
            where.set_ylabel(r"Norm. %s" % (YUnits["lbl"]))
        else:
            match labelStyle:
                case "()":
                    where.set_ylabel(r"%s (%s)" % (YUnits["lbl"], YUnits["ltx"]))
                case "[]":
                    where.set_ylabel(r"%s [%s]" % (YUnits["lbl"], YUnits["ltx"]))
                case "/":
                    where.set_ylabel(r"%s / %s" % (YUnits["lbl"], YUnits["ltx"]))
                case "":
                    where.set_ylabel(r"%s %s" % (YUnits["lbl"], YUnits["ltx"]))


def fmtZlabel(labelStyle, unitsZ_lbl, unitsZ_ltx, twoLines=False):
    if twoLines:
        NLspace = "\n"
    else:
        NLspace = ""

    match labelStyle:
        case "()":
            # return r'\boldmath\bfseries %s %s $(%s)$' % (unitsZ_lbl, NLspace, unitsZ_ltx)
            return r"%s %s (%s)" % (unitsZ_lbl, NLspace, unitsZ_ltx)
        case "[]":
            # return r'\boldmath\bfseries %s %s $[%s]$' % (unitsZ_lbl, NLspace, unitsZ_ltx)
            return r"%s %s [%s]" % (unitsZ_lbl, NLspace, unitsZ_ltx)
        case "/":
            # return r'\boldmath\bfseries %s / $%s$' % (unitsZ_lbl, unitsZ_ltx)
            return r"%s / %s" % (unitsZ_lbl, unitsZ_ltx)
        case "":
            # return r'\boldmath\bfseries %s $%s$' % (unitsZ_lbl, unitsZ_ltx)
            return r"%s %s" % (unitsZ_lbl, unitsZ_ltx)


def fmt2Dlabel(labelStyle, freqLabel, unitsF):
    if unitsF == "cm-1":
        unitsF = r"cm$^{-1}$"

    match freqLabel:
        case "Omega_n":
            pumpLabel = r"$\omega_{1}$"
            probeLabel = r"$\omega_{3}$"
        case "Omega_PP":
            pumpLabel = r"$\omega_{\text{pump}}$"
            probeLabel = r"$\omega_{\text{probe}}$"
        case "Pump-Probe":
            pumpLabel = r"$Pump Wavenumber$"
            probeLabel = r"$Probe Wavenumber$"
        case "Omega_n/2pic":
            pumpLabel = r"$\omega_{1}/2{\pi}c_{0}$"
            probeLabel = r"$\omega_{3}/2{\pi}c_{0}$"
    match labelStyle:
        case "()":
            pumpAll = r"%s (%s)" % (pumpLabel, unitsF)
            probeAll = r"%s (%s)" % (probeLabel, unitsF)
        case "[]":
            pumpAll = r"%s [%s]" % (pumpLabel, unitsF)
            probeAll = r"%s [%s]" % (probeLabel, unitsF)
        case "/":
            pumpAll = r"%s / %s" % (pumpLabel, unitsF)
            probeAll = r"%s / %s" % (probeLabel, unitsF)

    return pumpAll, probeAll


def units2dic(unitsL, unitsT, unitsZ):
    match unitsL:
        case "cm^{-1}":
            unitsL_lbl = "Wavenumber"
            unitsL_ltx = r"cm$^{-1}$"
        case "nm":
            unitsL_lbl = "Wavelength"
            unitsL_ltx = r"nm"
        case "eV":
            unitsL_lbl = "Energy"
            unitsL_ltx = r"eV"

    match unitsZ:
        case "mOD":
            unitsZ_lbl = r"$\Delta$A"
            unitsZ_ltx = r"mOD"
        case "int":
            unitsZ_lbl = r"Intensity"
            unitsZ_ltx = r"a.u."
        case "int2D":
            unitsZ_lbl = r"2D Intensity"
            unitsZ_ltx = r"a.u."
        case "x1E3":
            unitsZ_lbl = r"$\Delta$A"
            unitsZ_ltx = r"$\times 10^{3}$"

    Units = {
        "unitsL_lbl": unitsL_lbl,
        "unitsL_ltx": unitsL_ltx,
        "unitsZ_lbl": unitsZ_lbl,
        "unitsZ_ltx": unitsZ_ltx,
        "unitsZ": unitsZ,
        "unitsT_ltx": unitsT,
    }

    return Units


# --------------------------------------------------------------------------- #
#                    Spectral X-axis unit conversion                          #
# --------------------------------------------------------------------------- #
# Supported display units for the spectral (probe/pump) axis. ``nm`` is the only
# reciprocal quantity; ``cm-1``, ``eV`` and ``THz`` are all proportional to one
# another (each is a constant divided by the wavelength in nm).
SPECTRAL_UNITS = ("nm", "cm-1", "in-1", "eV", "THz")

# value[unit] = _NM_PRODUCT[unit] / wavelength_nm  (and wavelength_nm = P / value).
# eV:  hc/e = 1239.841984 eV*nm ; THz: c = 2.99792458e5 nm*THz ; cm-1: 1e7 nm/cm ;
# in-1: 2.54 cm/in x 1e7 = 2.54e7 nm/in (reciprocal inches, 1 in-1 = 2.54 cm-1).
_NM_PRODUCT = {"cm-1": 1.0e7, "in-1": 2.54e7, "eV": 1239.841984, "THz": 2.99792458e5}

# unit token -> (axis-label quantity, mathtext unit string).
_UNIT_LABEL = {
    "nm": ("Wavelength", r"nm"),
    "cm-1": ("Wavenumber", r"cm$^{-1}$"),
    "in-1": ("Wavenumber", r"in$^{-1}$"),
    "eV": ("Energy", r"eV"),
    "THz": ("Frequency", r"THz"),
}

# Above this value an axis is rescaled by 1e3 for display (visible/UV spectra
# reach ~2e4 cm-1, where five-digit ticks are unreadable).
DISPLAY_SCALE_THRESHOLD = 5.0e3

# units2dic label -> canonical spectral-unit token (native unit of a dataset).
_NATIVE_FROM_LBL = {
    "Wavelength": "nm",
    "Wavenumber": "cm-1",
    "Energy": "eV",
    "Frequency": "THz",
}


def convert_spectral(x, unit_in, unit_out):
    """Convert spectral-axis values between nm / cm-1 / eV / THz.

    ``x`` may be a scalar or array. Conversion pivots through the wavelength in
    nm; ``nm`` is reciprocal to the energy-like units, the others are mutually
    proportional. Non-positive entries map to +/-inf (guarded, no warning).
    """
    x = np.asarray(x, dtype=float)
    if unit_in == unit_out:
        return x
    if unit_in not in SPECTRAL_UNITS or unit_out not in SPECTRAL_UNITS:
        raise ValueError("unknown spectral unit: %r -> %r" % (unit_in, unit_out))
    with np.errstate(divide="ignore", invalid="ignore"):
        nm = x if unit_in == "nm" else _NM_PRODUCT[unit_in] / x
        out = nm if unit_out == "nm" else _NM_PRODUCT[unit_out] / nm
    return out


def native_x_unit(Units):
    """Return the dataset's native spectral-axis unit token from its Units dict."""
    return _NATIVE_FROM_LBL.get((Units or {}).get("unitsL_lbl", ""), "nm")


def complementary_unit(unit):
    """Complementary unit for the secondary axis.

    cm-1 <-> nm; every other unit (nm, eV, THz) pairs with nm.
    """
    if unit == "cm-1":
        return "nm"
    if unit == "nm":
        return "cm-1"
    return "nm"


@dataclass(frozen=True)
class AxisUnit:
    """Unit of one dataset axis.

    ``quantity`` is what the axis measures ("Wavenumber", "Delay", ...),
    ``unit`` the machine-readable token ("cm-1", "ps", "mOD"), ``latex`` the
    mathtext used on plots, and ``scale`` the factor the stored values were
    divided by for display (1000 for an axis shown as 10^3 cm-1, otherwise 1).
    """

    quantity: str
    unit: str
    latex: str
    scale: float = 1.0

    def label(self, style: str = "()") -> str:
        """Axis label in the delimiter convention of ``Settings.label_style``."""
        latex = self.latex
        if self.scale and self.scale != 1.0:
            latex = r"$10^{%d}$ %s" % (int(round(np.log10(self.scale))), latex)
        match style:
            case "[]":
                return f"{self.quantity} [{latex}]"
            case "/":
                return f"{self.quantity} / {latex}"
            case "":
                return f"{self.quantity} {latex}"
            case _:
                return f"{self.quantity} ({latex})"

    def __str__(self) -> str:
        scale = "" if self.scale == 1.0 else f" (x1/{self.scale:g})"
        return f"{self.quantity} in {self.unit}{scale}"


@dataclass(frozen=True)
class DatasetUnits:
    """Units of the axes of a dataset, as reported by ``axis_units()``.

    ``x``/``y`` are the plotted axes, ``z`` the signal, and ``t2`` the
    population-time axis of a 2-D dataset (``None`` in 1-D, where the delay is
    already one of the plotted axes).
    """

    x: AxisUnit
    y: AxisUnit
    z: AxisUnit
    t2: AxisUnit | None = None

    def as_dict(self) -> dict[str, dict]:
        """Plain nested mapping, e.g. for export or serialisation."""
        out = {}
        for name in ("x", "y", "z", "t2"):
            axis = getattr(self, name)
            if axis is not None:
                out[name] = {
                    "quantity": axis.quantity,
                    "unit": axis.unit,
                    "latex": axis.latex,
                    "scale": axis.scale,
                }
        return out

    def __str__(self) -> str:
        rows = [f"  {name}: {getattr(self, name)}" for name in ("x", "y", "z", "t2")
                if getattr(self, name) is not None]
        return "Axis units\n" + "\n".join(rows)


def signal_axis_unit(units) -> AxisUnit:
    """Signal (Z) unit of a dataset, from its unit dictionary."""
    units = units if isinstance(units, dict) else {}
    return AxisUnit(
        quantity=units.get("unitsZ_lbl", r"$\Delta$A"),
        unit=units.get("unitsZ", "mOD"),
        latex=units.get("unitsZ_ltx", "mOD"),
    )


def spectral_axis_unit(unit_token: str, scale: float = 1.0) -> AxisUnit:
    """Spectral (probe/pump) axis unit for a unit token such as ``"cm-1"``."""
    quantity, latex = _UNIT_LABEL.get(unit_token, ("Wavenumber", unit_token))
    return AxisUnit(quantity=quantity, unit=unit_token, latex=latex, scale=scale)


def delay_axis_unit(units, quantity: str = "Delay") -> AxisUnit:
    """Delay / population-time axis unit of a dataset."""
    units = units if isinstance(units, dict) else {}
    token = units.get("unitsT_ltx", "ps")
    return AxisUnit(quantity=units.get("unitsT_lbl", quantity), unit=token, latex=token)


def display_scale(values, threshold: float = DISPLAY_SCALE_THRESHOLD):
    """Return the 1e3 display factor for an axis, or 1.0.

    Visible/UV axes run to a few 10^4 cm-1 (or in-1); dividing them by 1000
    keeps the tick labels readable. Wavelength/energy/frequency axes stay well
    below the threshold and are therefore never rescaled.
    """
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    return 1.0e3 if np.max(np.abs(finite)) > threshold else 1.0


def scaled_unit_label(unit, factor: float = 1.0) -> str:
    """Mathtext unit string, prefixed with the 1e3 factor when one is applied."""
    label = _UNIT_LABEL[unit][1]
    if factor and factor != 1.0:
        exponent = int(round(np.log10(factor)))
        return r"$10^{%d}$ %s" % (exponent, label)
    return label


def to_display_axis(values, unit_in, unit_out=None, threshold: float = DISPLAY_SCALE_THRESHOLD):
    """Convert a spectral axis for display and apply the 1e3 rule.

    Returns ``(values_out, factor, unit_ltx)``: the converted and rescaled
    values, the factor they were divided by, and the mathtext unit string to put
    in the axis label. Coordinates that must share the axis (limits, overlay
    points) are converted with :func:`convert_spectral` and divided by the same
    factor.
    """
    unit_out = unit_in if unit_out in (None, "native") else unit_out
    converted = convert_spectral(values, unit_in, unit_out)
    factor = display_scale(converted, threshold)
    return converted / factor, factor, scaled_unit_label(unit_out, factor)


def x_unit_label(unit):
    """Return ``(quantity_label, mathtext_unit)`` for a spectral-unit token."""
    return _UNIT_LABEL[unit]


def resolve_x_unit(x_axis_unit, native_unit):
    """Resolve the effective display unit from a setting and the native unit.

    ``x_axis_unit`` may be ``None`` or ``"native"`` (use the dataset's own unit)
    or one of :data:`SPECTRAL_UNITS`.
    """
    if x_axis_unit in (None, "native"):
        return native_unit
    return x_axis_unit


def _register_scientific_colormaps():
    """Ensure Crameri scientific colour maps (vik, berlin) are registered in matplotlib."""
    import cmcrameri  # noqa: F401
    import cmcrameri.cm as cmc
    import matplotlib.pyplot as plt

    try:
        if "vik" not in plt.colormaps():
            plt.colormaps.register(cmap=cmc.vik, name="vik")
        if "vik_r" not in plt.colormaps():
            plt.colormaps.register(cmap=cmc.vik_r, name="vik_r")
        if "berlin" not in plt.colormaps():
            plt.colormaps.register(cmap=cmc.berlin, name="berlin")
        if "berlin_r" not in plt.colormaps():
            plt.colormaps.register(cmap=cmc.berlin_r, name="berlin_r")
    except Exception:
        pass


def zero_center_cmap(cmap, n, k):
    """Sample ``cmap`` into ``n`` bands and force the central ``k`` to the midpoint/zero colour.

    Can accept a colourmap object or a colourmap name string.
    """
    import matplotlib.colors as col
    import matplotlib.pyplot as plt

    _register_scientific_colormaps()

    _SCIENTIFIC = (
        "vik",
        "cmc.vik",
        "berlin",
        "cmc.berlin",
        "vik_r",
        "cmc.vik_r",
        "berlin_r",
        "cmc.berlin_r",
    )

    if isinstance(cmap, str):
        cmap_str = cmap.lower()
        if cmap_str in _SCIENTIFIC:
            cm, _ = CalcCMAP(cmap, n)
            return cm
        cmap = plt.get_cmap(cmap)
    elif hasattr(cmap, "name") and str(cmap.name).lower() in _SCIENTIFIC:
        return cmap

    colours = cmap(np.linspace(0.0, 1.0, n))
    k = max(0, min(int(k), n))
    if k == 0:
        return col.ListedColormap(colours)
    mid = n // 2
    mid_color = colours[mid].copy()
    lo = max(0, mid - k // 2)
    hi = min(n, lo + k)
    colours[lo:hi] = mid_color
    return col.ListedColormap(colours)


def CalcCMAP(cmap_str="DkRd/Wh/DkBu", Ntot=40, Nwhite=2):
    import cmcrameri.cm as cmc
    import matplotlib.colors as col
    import matplotlib.pyplot as plt

    _register_scientific_colormaps()

    n_white = max(0, int(Nwhite)) if Nwhite is not None else 2

    match cmap_str.lower():
        case "rd/wh/bu v2":
            n_half = int((Ntot - n_white) / 2) if n_white > 0 else int(Ntot / 2)
            n_right = Ntot - n_white - n_half if n_white > 0 else n_half
            blue_cm = plt.get_cmap("Blues", n_half)
            red_cm = plt.get_cmap("Reds", n_right)
            blue_a = blue_cm(np.linspace(1, 0, n_half))
            red_a = red_cm(np.linspace(0, 1, n_right))
            if n_white > 0:
                white_a = np.ones((n_white, 4))
                CArr = np.concatenate((blue_a, white_a, red_a))
            else:
                CArr = np.concatenate((blue_a, red_a))

        case "dkrd/wh/dkbu":
            # Given parameters
            m = 2 / 3  # Position of the "middle" colours
            n_blues = int(Ntot / 2)
            n_reds = int(Ntot / 2)

            # Blue colour interpolation
            color_input_blue = np.array([[0, 0, 0.5], [0, 0, 1], [1, 1, 1]])
            oldsteps = np.array([-1, -m, 0])
            newsteps = np.linspace(-1, 0, n_blues)
            blues_N = np.zeros((n_blues, 3))
            for j in range(3):
                blues_N[:, j] = np.clip(np.interp(newsteps, oldsteps, color_input_blue[:, j]), 0, 1)

            # Red colour interpolation
            color_input_red = np.array([[1, 1, 1], [1, 0, 0], [0.5, 0, 0]])
            oldsteps = np.array([0, m, 1])
            newsteps = np.linspace(0, 1, n_reds)
            reds_N = np.zeros((n_reds, 3))
            for j in range(3):
                reds_N[:, j] = np.clip(np.interp(newsteps, oldsteps, color_input_red[:, j]), 0, 1)

            CArr = np.append(blues_N, reds_N, axis=0)

        case "seismic":
            n_half = int(Ntot / 2)
            cmap_obj = plt.get_cmap("seismic")
            left_half = cmap_obj(np.linspace(0.0, 0.5, n_half))
            right_half = cmap_obj(np.linspace(0.5, 1.0, n_half))
            CArr = np.concatenate((left_half, right_half), axis=0)

        case "jet":
            n_half = int(Ntot / 2)
            cmap_obj = plt.get_cmap("jet")
            left_half = cmap_obj(np.linspace(0.0, 0.5, n_half))
            right_half = cmap_obj(np.linspace(0.5, 1.0, n_half))
            CArr = np.concatenate((left_half, right_half), axis=0)

        case "vik" | "cmc.vik":
            cmap_obj = cmc.vik
            CArr = cmap_obj(np.linspace(0, 1, Ntot))

        case "vik_r" | "cmc.vik_r":
            cmap_obj = cmc.vik_r
            CArr = cmap_obj(np.linspace(0, 1, Ntot))

        case "berlin" | "cmc.berlin":
            cmap_obj = cmc.berlin
            CArr = cmap_obj(np.linspace(0, 1, Ntot))

        case "berlin_r" | "cmc.berlin_r":
            cmap_obj = cmc.berlin_r
            CArr = cmap_obj(np.linspace(0, 1, Ntot))

        case _:
            try:
                cmap_obj = plt.get_cmap(cmap_str)
                CArr = cmap_obj(np.linspace(0, 1, Ntot))
            except ValueError:
                return CalcCMAP("DkRd/Wh/DkBu", Ntot, Nwhite=Nwhite)

    CMAP = col.ListedColormap(CArr, name=cmap_str)
    return CMAP, CArr


def get_trace_cmap(cmap_spec="rainbow", n_colors=10, reverse=False):
    """Return an array of RGBA colours for plotting N trace lines.

    Parameters
    ----------
    cmap_spec : str or TracesCmap, default "rainbow"
        Colourmap specification: ``"rainbow"``, ``"rainbow reversed"``, or
        ``"red-purple-blue"``. Can also be any valid Matplotlib colourmap name.
    n_colors : int, default 10
        Number of colours to sample.
    reverse : bool, default False
        Reverse the colour sampling order.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_colors, 4)`` containing RGBA colours.
    """
    import matplotlib.colors as col
    import matplotlib.pyplot as plt

    if hasattr(cmap_spec, "value"):
        cmap_spec = cmap_spec.value
    cmap_str = str(cmap_spec or "rainbow").lower().strip()

    is_rev = reverse
    if "reversed" in cmap_str or cmap_str.endswith("_r"):
        is_rev = not is_rev
        cmap_str = cmap_str.replace("reversed", "").replace("_r", "").strip()

    if cmap_str in ("red-purple-blue", "red_purple_blue", "red purple blue", "redpurpleblue"):
        cmap_obj = col.LinearSegmentedColormap.from_list(
            "red_purple_blue", ["#e31a1c", "#6a5acd", "#1f78b4"]
        )
    elif cmap_str in ("rainbow", "rainbow"):
        cmap_obj = plt.cm.rainbow
    else:
        try:
            cmap_obj = plt.get_cmap(cmap_str)
        except ValueError:
            cmap_obj = plt.cm.rainbow

    if n_colors <= 0:
        return np.empty((0, 4))
    if n_colors == 1:
        steps = np.array([0.0])
    else:
        steps = np.linspace(1.0, 0.0, n_colors) if is_rev else np.linspace(0.0, 1.0, n_colors)

    return cmap_obj(steps)


#: Time unit -> power of ten, as understood by :func:`ConvertTimeUnits`.
_TIME_UNIT_EXPONENTS: dict[str, int] = {
    "fs": -15,
    "ps": -12,
    "ns": -9,
    r"$\mu$s": -6,
    "ms": -3,
    "s": 0,
}


def ConvertTimeUnits(t_in, Unit_in, Unit_out=None, roundT=False):
    if Unit_in not in _TIME_UNIT_EXPONENTS:
        # Previously this fell through the match and failed later with an
        # UnboundLocalError naming an internal variable, which said nothing
        # about the actual problem.
        raise ValueError(
            f"unknown time unit {Unit_in!r}; expected one of {sorted(_TIME_UNIT_EXPONENTS)}"
        )
    match Unit_in:
        case "fs":
            u_in = -15
        case "ps":
            u_in = -12
        case "ns":
            u_in = -9
        case r"$\mu$s":
            u_in = -6
        case "ms":
            u_in = -3
        case "s":
            u_in = 0

    if Unit_out is None:
        t_out = t_in
        u_shift = 0
        if np.abs(t_in) < 1e-9:
            pass
        else:
            while np.abs(t_out) < 1 or np.abs(t_out) >= 1e3:
                if np.abs(t_out) <= 1:
                    u_shift = u_shift + 3
                elif np.abs(t_out) >= 1e3:
                    u_shift = u_shift - 3
                else:
                    u_shift = 0
                if np.abs(u_shift) > 300:
                    u_shift = 0
                    t_out = t_in
                    break
                t_out = t_in * 10**u_shift
    else:
        match Unit_out:
            case "fs":
                u_shift = u_in + 15
            case "ps":
                u_shift = u_in + 12
            case "ns":
                u_shift = u_in + 9
            case r"$\mu$s":
                u_shift = u_in + 6
            case "ms":
                u_shift = u_in + 3
            case "s":
                u_shift = u_in + 0
        t_out = t_in * 10**u_shift

    if roundT:
        if abs(t_out) <= 1:
            t_out = np.round(t_out, 2)
        elif abs(t_out) <= 10:
            t_out = np.round(t_out / 10, 2) * 10
        elif abs(t_out) <= 100:
            t_out = np.round(t_out / 100, 2) * 100
        elif abs(t_out) <= 1000:
            t_out = np.round(t_out / 1000, 2) * 1000

    u_out = u_in - u_shift
    match u_out:
        case -15:
            Unit_out = "fs"
        case -12:
            Unit_out = "ps"
        case -9:
            Unit_out = "ns"
        case -6:
            Unit_out = r"$\mu$s"
        case -3:
            Unit_out = "ms"
        case 0:
            Unit_out = "s"

    return t_out, Unit_out


def _one_significant_figure(x: float) -> float:
    """``x`` to one significant figure, rounding halves *up* (2.5 -> 3, not 2).

    Python and NumPy both round halves to even, which would report an
    uncertainty of 2.5 as 2 -- smaller than it is. An uncertainty should never
    be rounded down.
    """
    x = float(x)
    if x == 0 or not np.isfinite(x):
        return x
    decade = 10.0 ** np.floor(np.log10(abs(x)))
    return float(np.sign(x) * np.floor(abs(x) / decade + 0.5) * decade)


def round_to_uncertainty(value, err):
    """Round a value and its uncertainty so the pair reads sensibly.

    The uncertainty goes to one significant figure and the value to *that same*
    decimal place, so a fitted constant is never quoted with more precision than
    its own uncertainty supports::

        325.3 +/- 2.5   ->  325 +/- 3
        326.3 +/- 3     ->  326 +/- 3
        297.5 +/- 1.38  ->  298 +/- 1
        8.036 +/- 0.023 ->  8.04 +/- 0.02

    A non-finite or non-positive uncertainty leaves both untouched: there is
    nothing to round *to*.

    Returns
    -------
    (float, float)
        The rounded value and uncertainty.
    """
    value = float(value)
    if err is None:
        return value, err
    err = float(err)
    if not np.isfinite(err) or err <= 0 or not np.isfinite(value):
        return value, err

    err_rounded = _one_significant_figure(err)
    if err_rounded == 0:
        return value, err
    # The value shares the uncertainty's decimal place: quoting it any finer
    # claims precision the uncertainty denies.
    place = 10.0 ** np.floor(np.log10(abs(err_rounded)))
    return float(np.floor(abs(value) / place + 0.5) * place * np.sign(value or 1.0)), err_rounded


def format_uncertainty_pair(value, err) -> tuple[str, str | None]:
    """``(value, uncertainty)`` as strings, rounded together and matched.

    Both are written to the same decimal place, so the pair reads as one
    measurement: ``("325", "3")``, ``("8.04", "0.02")``. Without this the value
    would print as ``8`` beside an uncertainty of ``0.02``, suggesting a
    different precision than it has.
    """
    rounded, err_rounded = round_to_uncertainty(value, err)
    if err_rounded is None or not np.isfinite(err_rounded) or err_rounded <= 0:
        return ("%g" % rounded), None

    decimals = max(0, int(-np.floor(np.log10(abs(err_rounded)))))
    return (f"%.{decimals}f" % rounded), (f"%.{decimals}f" % err_rounded)


def format_lifetime_label(
    tau, unit_in, err=None, *, roundT=None, fixed=False, pair_round=False, settings=None
):
    """A lifetime as it should appear in a legend: ``320 fs``, ``8.0 ± 0.2 ps``.

    Handles the three things a raw ``%s`` of the number does not:

    * **the unit follows the magnitude** (via :func:`ConvertTimeUnits`), so a
      sub-picosecond lifetime of a picosecond dataset reads in fs;
    * **the digits are bounded** -- ``Settings.round_labels`` rounds to two
      significant figures and ``Settings.label_digits`` caps the rest, so a
      fitted ``320.95763496398745`` prints as ``320``, not as the full float;
    * **a non-decaying component prints as** :math:`\\infty`, since it has no
      lifetime to round.

    The uncertainty is converted into the *same* unit as the value before it is
    shown, and is omitted for a fixed parameter -- a fixed lifetime has no
    uncertainty, and printing one would invent a result.

    Parameters
    ----------
    tau : float
        Lifetime in ``unit_in``. Non-finite means non-decaying.
    unit_in : str
        Time unit of ``tau`` (the dataset's ``unitsT_ltx``).
    err : float, optional
        1-sigma uncertainty, in the same unit as ``tau``.
    roundT : bool, optional
        Round to two significant figures. Defaults to ``Settings.round_labels``.
    fixed : bool, default False
        Suppress the uncertainty (the parameter was not fitted).
    pair_round : bool, default False
        Round the value *together with* its uncertainty
        (:func:`round_to_uncertainty`), so neither is quoted more precisely than
        the other supports. Applied after the unit conversion, since the
        rounding depends on the displayed magnitude.
    """
    if settings is None:
        from pymorgan import get_settings

        settings = get_settings()
    if not bool(getattr(settings, "show_uncertainties", True)):
        err = None
    if pair_round is None:
        pair_round = bool(getattr(settings, "round_uncertainties", True))
    if roundT is None:
        roundT = bool(getattr(settings, "round_labels", True))
    digits = max(1, int(getattr(settings, "label_digits", 3)))


    tau = float(tau)
    if not np.isfinite(tau):
        return r"$\infty$"

    # Pair rounding needs an uncertainty to round *to*: a fixed parameter or a
    # missing error falls back to the ordinary rounding rather than printing
    # every decimal it happens to have.
    has_err = err is not None and not fixed and np.isfinite(err) and float(err) > 0
    use_pair = bool(pair_round and has_err)
    plain_round = bool(roundT and not use_pair)

    if unit_in not in _TIME_UNIT_EXPONENTS:
        # An unknown (or absent) unit is not an error here: arrays fitted
        # without a dataset have no unit at all. Format the number and say
        # nothing about units rather than inventing one.
        value = float("%.2g" % tau) if plain_round else tau
        scaled = float(err) if has_err else None
        return _value_pm_error(value, scaled, digits, use_pair)

    value, unit_out = ConvertTimeUnits(tau, unit_in, roundT=plain_round)
    scaled = None
    if has_err:
        scaled, _ = ConvertTimeUnits(float(err), unit_in, Unit_out=unit_out)
        if plain_round:
            scaled = float("%.2g" % scaled)
    return f"{_value_pm_error(value, scaled, digits, use_pair)} {unit_out}"


def _value_pm_error(value, err, digits: int, pair: bool) -> str:
    """``value``, optionally ``$\\pm$ err``, in matching precision."""
    if pair:
        head, tail = format_uncertainty_pair(value, err)
        return head if tail is None else f"{head} $\\pm$ {tail}"
    text = f"%.{digits}g" % value
    if err is not None and np.isfinite(err):
        text += r" $\pm$ " + (f"%.{digits}g" % err)
    return text


def print2Ddelay(t2_ps, onlyNumbers=False):
    """Format a t2 delay for the 2-D contour text box.

    ``onlyNumbers=False`` gives ``t_2 = 1.0 ps``; ``True`` gives ``1.0 ps``
    (see ``Settings.t2_label_style``). The unit follows the magnitude
    (fs / ps / ns / us).
    """
    if not onlyNumbers:
        if abs(t2_ps) < 1:
            t2_text = r"$t_{2} =$ %.3g fs" % (t2_ps * 1000)
        elif abs(t2_ps) >= 1 and abs(t2_ps) < 1e3:
            t2_text = r"$t_{2} =$ %.3g ps" % (t2_ps)
        elif abs(t2_ps) >= 1e3 and abs(t2_ps) < 1e6:
            t2_text = r"$t_{2} =$ %.3g ns" % (t2_ps / 1000)
        else:
            t2_text = r"$t_{2} =$ %.3g $\mu$s" % (t2_ps / 1e6)
    else:
        if abs(t2_ps) < 1:
            t2_text = r"%.3g fs" % (t2_ps * 1000)
        elif abs(t2_ps) >= 1 and abs(t2_ps) < 1e3:
            t2_text = r"%.3g ps" % (t2_ps)
        elif abs(t2_ps) >= 1e3 and abs(t2_ps) < 1e6:
            t2_text = r"%.3g ns" % (t2_ps / 1000)
        else:
            t2_text = r"%.3g $\mu$s" % (t2_ps / 1e6)
    return t2_text




def box_smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


def uniform_smooth(y, npts):
    from scipy.ndimage import uniform_filter1d

    y_smooth = uniform_filter1d(y, size=npts)
    return y_smooth


def adjust_lightness(colour, scale=1):
    """Scale the lightness of an RGB(A) colour, keeping hue and saturation.

    Equivalent to ``seaborn.set_hls_values(colour, l=lightness)``, without the
    seaborn import (which pulls in scipy.stats and pandas at start-up).
    """
    r, g, b = colour[0:3]
    h, lightness, s = rgb_to_hls(r, g, b)
    return hls_to_rgb(h, min(1.0, lightness * scale), s)


def adjust_cmap(cmap, scale=1):
    for i, rgba in enumerate(cmap):
        cmap[i, 0:3] = adjust_lightness(rgba, scale)
    return cmap


def ApplyMasks(mask_rgns, X):
    for region in mask_rgns:
        X = np.ma.masked_where((X >= region[0]) & (X <= region[1]), X)
    return X


def add_glow(ax, x, y, colour=None, base_frac=0.04, *, color=None):
    """
    Adds a soft glow around a curve.
    Thickness adapts automatically to axis y-range.
    """
    col = colour if colour is not None else color
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin

    for alpha, frac in zip([0.05, 0.08, 0.12], [base_frac * 1.6, base_frac, base_frac * 0.6], strict=True):
        width = frac * yrange
        ax.fill_between(x, y - width, y + width, color=col, alpha=alpha, linewidth=0)

    ax.plot(x, y, color=col, linewidth=2, solid_capstyle="round")


def parse_value_list(text: str, all_values=None) -> list[float] | None:
    """Parse the cut-list syntax: numbers, MATLAB ranges, or ``all``.

    Accepts ``1900, 1950``, ``1:0.1:5`` (start:step:stop), ``1:5``
    (unit step, descending allowed) and the word ``all``, which returns every
    value of ``all_values``. Unparseable tokens are logged and skipped rather
    than aborting the whole list -- a typo in one of five cuts should not throw
    away the other four.

    Returns ``None`` when ``all`` was asked for but no ``all_values`` was given.
    """
    text = str(text).strip().strip("'\"")
    if text.lower() == "all":
        return None if all_values is None else [float(v) for v in all_values]

    values: list[float] = []
    for token in re.split(r"[,\s]+", text):
        token = token.strip("'\"")
        if not token:
            continue
        if token.lower() == "all":
            if all_values is None:
                return None
            return [float(v) for v in all_values]
        if ":" in token:
            values.extend(_expand_range(token))
            continue
        try:
            values.append(float(token))
        except ValueError:
            logger.debug("Ignoring unparseable value %r.", token, exc_info=True)
    return values


def _expand_range(token: str) -> list[float]:
    """``start:stop`` or ``start:step:stop`` as an inclusive list."""
    parts = token.split(":")
    try:
        if len(parts) == 2:
            start, stop = float(parts[0]), float(parts[1])
            step = 1.0 if start <= stop else -1.0
        elif len(parts) == 3:
            start, step, stop = float(parts[0]), float(parts[1]), float(parts[2])
        else:
            return []
    except ValueError:
        logger.debug("Ignoring unparseable range token %r.", token, exc_info=True)
        return []
    if step == 0:
        return []

    out: list[float] = []
    tolerance = abs(step) * 1e-9
    value = start
    if step > 0:
        while value <= stop + tolerance:
            out.append(value)
            value += step
    else:
        while value >= stop - tolerance:
            out.append(value)
            value += step
    return out

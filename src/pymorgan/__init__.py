"""PyMORGAN: plotting and analysis of ultrafast time-resolved spectroscopy data.

The 1-D branch is a load/process/plot pipeline built around
:class:`pymorgan.oneD.Dataset1D`; steady-state absorption/emission/excitation
spectra live in :mod:`pymorgan.steadyState`; presentation is centralised in
:class:`pymorgan.settings.Settings`.

    import pymorgan as pm

    pm.apply_style()
    data = pm.load_1D("scan.pdat", data_type="PDAT")
    data.background_correct(tmin=-20, tmax=-5)
    data.plot_contour(Zscale=20)
    data.plot_kinetics([2132, 2218], plotStyle="-")
    pm.show_plots()

**Scope.** PyMORGAN loads, processes and plots; it does not fit kinetics.
Trace extraction and fitting -- multi-exponential, global and target analysis
(rate matrices, EAS/SAS) -- live in the companion project **PyRATE-TA**, which
shares this loader/dataset layer. The one piece of that seam PyMORGAN keeps is
:func:`pymorgan.oneD.plot.plot_species_spectra`, which renders fitted spectra
from plain arrays.

**Lazy public API.** The pipelines pull in matplotlib, scipy and the loader
registries, which costs roughly a second. Importing this package therefore
binds only the names below; the module that provides a name is imported the
first time that name is used (PEP 562). ``import pymorgan as pm`` is thus
nearly free, and the GUI can put its splash screen on screen before paying for
the heavy imports. Everything else -- ``from pymorgan import load_1D``,
``pm.twoD.plot_map``, ``help(pm)`` -- behaves exactly as before.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .__about__ import __author__, __email__, __version__
from .log import configure_logging, get_logger

# Public name -> module that defines it. Kept in sync with ``__all__``; a new
# export needs one entry here.
_EXPORTS: dict[str, str] = {
    # 1-D pipeline
    "Dataset1D": ".oneD",
    "available_loaders": ".oneD",
    "load_1D": ".oneD",
    "register_loader": ".oneD",
    # 2-D pipeline
    "Dataset2D": ".twoD",
    "available_map_loaders": ".twoD",
    "is_map_dataset_dir": ".twoD",
    "load_2D": ".twoD",
    "map_dataset_glob": ".twoD",
    "register_map_loader": ".twoD",
    # steady state
    "Spectrum": ".steadyState",
    "SpectrumKind": ".steadyState",
    "SpectrumSeries": ".steadyState",
    "available_spectrum_loaders": ".steadyState",
    "load_spectrum": ".steadyState",
    "register_spectrum_loader": ".steadyState",
    # calibration
    "available_calibration_types": ".cal",
    # settings
    "DeltaAUnits": ".settings",
    "FreqLabel": ".settings",
    "LabelStyle": ".settings",
    "Settings": ".settings",
    "StyleProfile": ".settings",
    "TimeAxisLabel": ".settings",
    "TimeAxisScale": ".settings",
    "apply_style": ".settings",
    "ensure_settings_file": ".settings",
    "get_settings": ".settings",
    "load_settings": ".settings",
    "save_settings": ".settings",
    "set_settings": ".settings",
    "settings_path": ".settings",
    "update_settings": ".settings",
    "use_settings": ".settings",
    # display
    "add_subplot_labels": ".display",
    "close_plots": ".display",
    "show": ".display",
    "show_plots": ".display",
}

# Submodules reachable as attributes (``pm.twoD.plot_map``) without an import.
_SUBMODULES = ("oneD", "twoD", "steadyState", "settings", "display", "helpers", "cal", "log")

if TYPE_CHECKING:  # keeps type checkers and IDE completion fully informed
    from .cal import available_calibration_types
    from .display import add_subplot_labels, close_plots, show, show_plots
    from .oneD import (
        Dataset1D,
        available_loaders,
        load_1D,
        register_loader,
    )
    from .settings import (
        DeltaAUnits,
        FreqLabel,
        LabelStyle,
        Settings,
        StyleProfile,
        TimeAxisLabel,
        TimeAxisScale,
        apply_style,
        ensure_settings_file,
        get_settings,
        load_settings,
        save_settings,
        set_settings,
        settings_path,
        update_settings,
        use_settings,
    )
    from .steadyState import (
        Spectrum,
        SpectrumKind,
        SpectrumSeries,
        available_spectrum_loaders,
        load_spectrum,
        register_spectrum_loader,
    )
    from .twoD import (
        Dataset2D,
        available_map_loaders,
        is_map_dataset_dir,
        load_2D,
        map_dataset_glob,
        register_map_loader,
    )


def __getattr__(name: str):
    """Import the module providing ``name`` on first use (PEP 562)."""
    if name in _SUBMODULES:
        value = importlib.import_module(f".{name}", __name__)
    elif name in _EXPORTS:
        value = getattr(importlib.import_module(_EXPORTS[name], __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value  # bind it, so this happens once per name
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(_SUBMODULES))


__all__ = [
    "__version__",
    "__author__",
    "__email__",
    # 1-D pipeline
    "Dataset1D",
    "load_1D",
    "register_loader",
    "available_loaders",
    # 2-D pipeline
    "Dataset2D",
    "load_2D",
    "register_map_loader",
    "available_map_loaders",
    "map_dataset_glob",
    "is_map_dataset_dir",
    # steady-state
    "Spectrum",
    "SpectrumSeries",
    "SpectrumKind",
    "load_spectrum",
    "register_spectrum_loader",
    "available_spectrum_loaders",
    # calibration
    "available_calibration_types",
    # settings
    "Settings",
    "StyleProfile",
    "LabelStyle",
    "DeltaAUnits",
    "TimeAxisScale",
    "TimeAxisLabel",
    "FreqLabel",
    "get_settings",
    # logging
    "configure_logging",
    "get_logger",
    "set_settings",
    "update_settings",
    "use_settings",
    "load_settings",
    "save_settings",
    "settings_path",
    "ensure_settings_file",
    "apply_style",
    # display
    "add_subplot_labels",
    "show_plots",
    "show",
    "close_plots",
]


# Console output for informational messages, unless the host configured logging.
configure_logging()

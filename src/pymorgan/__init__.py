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
    "SVDResult": ".oneD",
    "ScatterBand": ".oneD",
    "ScatterRemovalResult": ".oneD",
    "StitchResult": ".oneD",
    "available_loaders": ".oneD",
    "calculate_anisotropy": ".oneD",
    "classify_pymess_h5": ".oneD",
    "classify_quickcontrol_info": ".oneD",
    "compute_scatter_removal": ".oneD",
    "compute_svd": ".oneD",
    "detect_scatter_bands": ".oneD",
    "embed_calibrated_probe_h5": ".oneD",
    "export_pdat": ".oneD",
    "load_1D": ".oneD",
    "load_1D_hdf5": ".oneD",
    "load_pymess_anisotropy": ".oneD",
    "parse_ufs": ".oneD",
    "read_1D_hdf5": ".oneD",
    "read_Helios_UFS": ".oneD",
    "read_glotaran_ascii": ".oneD",
    "read_glotaran_nc": ".oneD",
    "register_loader": ".oneD",
    "save_1D_hdf5": ".oneD",
    "stitch_time": ".oneD",
    "stitch_wavelength": ".oneD",
    "write_glotaran_ascii": ".oneD",
    "write_glotaran_nc": ".oneD",
    "write_pdat": ".oneD",
    # 2-D pipeline
    "Dataset2D": ".twoD",
    "available_map_loaders": ".twoD",
    "is_map_dataset_dir": ".twoD",
    "is_map_dataset_file": ".twoD",
    "load_2D": ".twoD",
    "load_2D_hdf5": ".twoD",
    "map_dataset_glob": ".twoD",
    "read_2D_hdf5": ".twoD",
    "register_map_loader": ".twoD",
    "save_2D_hdf5": ".twoD",
    # auto-detection
    "FormatMatch": ".detect",
    "detect_format": ".detect",
    "load_auto": ".detect",
    # steady state
    "Spectrum": ".steadyState",
    "SpectrumKind": ".steadyState",
    "SpectrumSeries": ".steadyState",
    "available_spectrum_loaders": ".steadyState",
    "load_spectrum": ".steadyState",
    "plot_steady_state": ".steadyState",
    "register_spectrum_loader": ".steadyState",
    # calibration
    "available_calibration_types": ".cal",
    # settings
    "DeltaAUnits": ".settings",
    "FreqLabel": ".settings",
    "LabelStyle": ".settings",
    "Settings": ".settings",
    "SteadyStatePlotMode": ".settings",
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
    # workflows, provenance, recipes, batch, and sessions
    "ProcessingStep": ".provenance",
    "Recipe": ".recipe",
    "BatchReport": ".batch",
    "run_batch": ".batch",
    "ProjectSession": ".project",
}

# Submodules reachable as attributes (``pm.twoD.plot_map``) without an import.
_SUBMODULES = (
    "oneD",
    "twoD",
    "steadyState",
    "settings",
    "display",
    "helpers",
    "cal",
    "log",
    "detect",
    "provenance",
    "recipe",
    "batch",
    "project",
)

if TYPE_CHECKING:  # keeps type checkers and IDE completion fully informed
    from .batch import BatchReport, run_batch
    from .cal import available_calibration_types
    from .detect import FormatMatch, detect_format, load_auto
    from .display import add_subplot_labels, close_plots, show, show_plots
    from .oneD import (
        Dataset1D,
        ScatterBand,
        ScatterRemovalResult,
        StitchResult,
        SVDResult,
        available_loaders,
        calculate_anisotropy,
        classify_pymess_h5,
        classify_quickcontrol_info,
        compute_scatter_removal,
        compute_svd,
        detect_scatter_bands,
        embed_calibrated_probe_h5,
        export_pdat,
        load_1D,
        load_1D_hdf5,
        load_pymess_anisotropy,
        parse_ufs,
        read_1D_hdf5,
        read_glotaran_ascii,
        read_glotaran_nc,
        read_Helios_UFS,
        register_loader,
        save_1D_hdf5,
        stitch_time,
        stitch_wavelength,
        write_glotaran_ascii,
        write_glotaran_nc,
        write_pdat,
    )
    from .project import ProjectSession
    from .provenance import ProcessingStep
    from .recipe import Recipe
    from .settings import (
        DeltaAUnits,
        FreqLabel,
        LabelStyle,
        Settings,
        SteadyStatePlotMode,
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
        plot_steady_state,
        register_spectrum_loader,
    )
    from .twoD import (
        Dataset2D,
        available_map_loaders,
        is_map_dataset_dir,
        is_map_dataset_file,
        load_2D,
        load_2D_hdf5,
        map_dataset_glob,
        read_2D_hdf5,
        register_map_loader,
        save_2D_hdf5,
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
    "load_pymess_anisotropy",
    "classify_pymess_h5",
    "classify_quickcontrol_info",
    "embed_calibrated_probe_h5",
    "register_loader",
    "available_loaders",
    "calculate_anisotropy",
    "compute_svd",
    "SVDResult",
    "ScatterBand",
    "ScatterRemovalResult",
    "detect_scatter_bands",
    "compute_scatter_removal",
    "StitchResult",
    "stitch_wavelength",
    "stitch_time",
    "write_pdat",
    "export_pdat",
    "save_1D_hdf5",
    "load_1D_hdf5",
    "read_1D_hdf5",
    "parse_ufs",
    "read_Helios_UFS",
    "read_glotaran_ascii",
    "write_glotaran_ascii",
    "read_glotaran_nc",
    "write_glotaran_nc",
    # 2-D pipeline
    "Dataset2D",
    "load_2D",
    "save_2D_hdf5",
    "load_2D_hdf5",
    "read_2D_hdf5",
    "register_map_loader",
    "available_map_loaders",
    "map_dataset_glob",
    "is_map_dataset_dir",
    "is_map_dataset_file",
    # auto-detection
    "FormatMatch",
    "detect_format",
    "load_auto",
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
    # workflows, recipes, batch, and sessions
    "ProcessingStep",
    "Recipe",
    "BatchReport",
    "run_batch",
    "ProjectSession",
]


# Console output for informational messages, unless the host configured logging.
configure_logging()

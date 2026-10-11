"""Unified interface for one-dimensional time-resolved data.

The 1-D pipeline is organised by stage:

* :mod:`pymorgan.oneD.load` — readers and the loader registry,
* :mod:`pymorgan.oneD.process` — background subtraction (and future processing),
* :mod:`pymorgan.oneD.plot` — Dataset1D-aware plotters,

tied together by :class:`Dataset1D`.

Kinetic analysis — trace extraction, multi-exponential fitting, global/target
analysis (rate matrices, EAS/SAS) — is not part of PyMORGAN. It lives in the
companion project **PyRATE-TA**, which shares this loader/dataset layer. PyMORGAN
keeps only the renderer for fit output
(:func:`pymorgan.oneD.plot.plot_species_spectra`).
"""

from .chirp import (
    ChirpFit,
    cauchy_t0,
    default_chirp_filename,
    fit_cauchy_dispersion,
    fit_chirp_automatic,
    fit_chirp_manual,
    fit_chirp_step,
    fit_chirp_wavelet,
    load_chirp_fit,
    plot_chirp_diagnostics,
    save_chirp_fit,
)
from .dataset import Dataset1D, load_1D
from .load import (
    available_loaders,
    export_pdat,
    get_loader,
    load_1D_hdf5,
    load_pymess_anisotropy,
    read_1D_hdf5,
    read_HARPIA,
    read_PDAT,
    read_PyMESS_PP,
    read_RAL_IRpp,
    read_UniGE_fsTA,
    register_loader,
    save_1D_hdf5,
    write_pdat,
)
from .process import background_correct, calculate_anisotropy
from .stitch import StitchResult, stitch_time, stitch_wavelength
from .svd import SVDResult, compute_svd

__all__ = [
    "Dataset1D",
    "load_1D",
    "register_loader",
    "get_loader",
    "available_loaders",
    "load_pymess_anisotropy",
    "calculate_anisotropy",
    "compute_svd",
    "SVDResult",
    "StitchResult",
    "stitch_wavelength",
    "stitch_time",
    "read_PDAT",
    "read_HARPIA",
    "read_PyMESS_PP",
    "read_RAL_IRpp",
    "read_UniGE_fsTA",
    "write_pdat",
    "export_pdat",
    "save_1D_hdf5",
    "load_1D_hdf5",
    "read_1D_hdf5",
    "background_correct",

    "ChirpFit",
    "cauchy_t0",
    "fit_cauchy_dispersion",
    "fit_chirp_automatic",
    "fit_chirp_step",
    "fit_chirp_manual",
    "fit_chirp_wavelet",
    "plot_chirp_diagnostics",
    "save_chirp_fit",
    "load_chirp_fit",
    "default_chirp_filename",
]


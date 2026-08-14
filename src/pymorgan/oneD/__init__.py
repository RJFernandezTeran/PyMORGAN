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
    get_loader,
    read_HARPIA,
    read_PDAT,
    read_UniGE_fsTA,
    register_loader,
)
from .process import background_correct

__all__ = [
    "Dataset1D",
    "load_1D",
    "register_loader",
    "get_loader",
    "available_loaders",
    "read_PDAT",
    "read_HARPIA",
    "read_UniGE_fsTA",
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

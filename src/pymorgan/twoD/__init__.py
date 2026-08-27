"""Unified interface for two-dimensional time-resolved data (2D-IR / 2D-EV).

Organised by stage, mirroring :mod:`pymorgan.oneD`:

* :mod:`pymorgan.twoD.load` — readers and the loader registry,
* :mod:`pymorgan.twoD.process` — reference-t2 background subtraction,
* :mod:`pymorgan.twoD.plot` — Dataset2D-aware map plotting,
* :mod:`pymorgan.twoD.analyse` — slice / diagonal extraction (CLS to come),

tied together by :class:`Dataset2D`.
"""

from .analyse import (
    center_line_slope,
    diagonal,
    get_slice_at_probe,
    get_slice_at_pump,
    get_slice_integrate_probe,
    get_slice_integrate_pump,
    nodal_line_slope,
    slice_at,
)
from .dataset import Dataset2D, load_2D
from .load import (
    available_map_loaders,
    get_map_loader,
    is_map_dataset_dir,
    is_map_directory_format,
    map_dataset_glob,
    read_P2DAT,
    register_map_loader,
    save_P2DAT,
    write_P2DAT,
)
from .plot import Map2DAxes, plot_map
from .process import background_correct

__all__ = [
    "Dataset2D",
    "load_2D",
    "register_map_loader",
    "get_map_loader",
    "available_map_loaders",
    "read_P2DAT",
    "write_P2DAT",
    "save_P2DAT",
    "background_correct",
    "plot_map",
    "Map2DAxes",
    "slice_at",
    "diagonal",
    "center_line_slope",
    "nodal_line_slope",
    "get_slice_at_pump",
    "get_slice_at_probe",
    "get_slice_integrate_pump",
    "get_slice_integrate_probe",
    "map_dataset_glob",
    "is_map_dataset_dir",
    "is_map_directory_format",
]



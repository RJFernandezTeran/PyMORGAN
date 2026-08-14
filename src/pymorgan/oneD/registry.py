"""Backwards-compatible shim.

The loader registry now lives in :mod:`pymorgan.oneD.load`. This module
re-exports its public names so that ``from pymorgan.oneD.registry import ...``
keeps working.
"""

from .load import available_loaders, get_loader, register_loader

__all__ = ["register_loader", "get_loader", "available_loaders"]

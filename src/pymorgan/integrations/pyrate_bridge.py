"""Integration bridge between PyMORGAN and companion package PyRATE-TA.

PyMORGAN deliberately maintains zero hard dependencies on PyRATE-TA.
This module dynamically detects whether PyRATE-TA is installed, retrieves
version and status metadata, and provides seamless data hand-off and process
launching for kinetic modeling, multi-exponential fitting, and global/target analysis.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pymorgan.log import get_logger

if TYPE_CHECKING:
    from pymorgan.oneD.dataset import Dataset1D

logger = get_logger(__name__)


def is_pyrate_installed() -> bool:
    """Check if PyRATE-TA is installed and available in the current environment."""
    return importlib.util.find_spec("pyrate_ta") is not None


def get_pyrate_version() -> str | None:
    """Return the installed version of PyRATE-TA, or None if not installed."""
    if not is_pyrate_installed():
        return None
    try:
        return importlib.metadata.version("pyrate-ta")
    except Exception:
        try:
            import pyrate_ta

            return getattr(pyrate_ta, "__version__", "unknown")
        except Exception:
            return "unknown"


def launch_pyrate(
    dataset_or_path: Dataset1D | str | Path | None = None,
    extra_args: list[str] | None = None,
) -> subprocess.Popen:
    """Launch the PyRATE-TA GUI, optionally passing a dataset file to load.

    Parameters
    ----------
    dataset_or_path : Dataset1D, str, Path, optional
        A Dataset1D object to hand off, or a file path to load on startup.
        If a Dataset1D is passed, it is exported to a temporary PDAT file
        so that PyRATE-TA can ingest it directly.
    extra_args : list[str], optional
        Additional CLI flags to pass to PyRATE-TA.

    Returns
    -------
    subprocess.Popen
        The spawned detached process handle.

    Raises
    ------
    RuntimeError
        If PyRATE-TA is not installed.
    """
    if not is_pyrate_installed():
        raise RuntimeError(
            "PyRATE-TA is not installed in the active Python environment.\n\n"
            "Install it using:\n"
            "    uv pip install pyrate-ta\n"
            "or clone the companion repository:\n"
            "    https://github.com/RJFernandezTeran/PyRATE-TA"
        )

    cmd = [sys.executable, "-m", "pyrate_ta.gui.app"]

    target_file: Path | None = None
    if dataset_or_path is not None:
        if isinstance(dataset_or_path, (str, Path)):
            p = Path(dataset_or_path)
            if p.exists():
                target_file = p
        else:
            # Export Dataset1D to temp PDAT file
            from pymorgan.oneD.load import export_pdat

            temp_dir = Path(tempfile.gettempdir()) / "pymorgan_pyrate_handoff"
            temp_dir.mkdir(parents=True, exist_ok=True)
            ds_name = getattr(dataset_or_path, "source", "dataset") or "dataset"
            safe_name = Path(ds_name).stem.replace(" ", "_")
            target_file = temp_dir / f"{safe_name}_for_kinetic_analysis.pdat"
            export_pdat(dataset_or_path, target_file)
            logger.info("Exported active dataset to %s for PyRATE-TA", target_file)

    if target_file is not None:
        cmd.append(str(target_file))

    if extra_args:
        cmd.extend(extra_args)

    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        )
        kwargs["close_fds"] = True
    else:
        kwargs["start_new_session"] = True

    logger.info("Launching PyRATE-TA: %s", " ".join(cmd))
    return subprocess.Popen(cmd, **kwargs)

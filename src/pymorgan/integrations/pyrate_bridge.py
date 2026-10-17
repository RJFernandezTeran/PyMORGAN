"""Integration bridge between PyMORGAN and companion package PyRATE-TA.

PyMORGAN deliberately maintains zero hard dependencies on PyRATE-TA.
This module dynamically detects whether PyRATE-TA is installed, retrieves
version and status metadata, and provides seamless data hand-off and process
launching for kinetic modeling, multi-exponential fitting, and global/target analysis.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pymorgan.log import get_logger

if TYPE_CHECKING:
    from pymorgan.oneD.dataset import Dataset1D

logger = get_logger(__name__)


def find_pyrate_command() -> list[str] | None:
    """Return command argv to execute PyRATE-TA, or None if not found."""
    # 1. If frozen executable (PyInstaller / InnoSetup packaged app)
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent
        for name in ("PyRATE-TA.exe", "pyrate-ta.exe", "PyRATE-TA", "pyrate-ta"):
            cand = exe_dir / name
            if cand.is_file():
                return [str(cand)]
        if importlib.util.find_spec("pyrate_ta") is not None:
            for name in ("PyMORGAN_Suite.exe", "PyMORGAN_Suite"):
                cand = exe_dir / name
                if cand.is_file():
                    return [str(cand), "--pyrate"]

    # 2. Check if importable in current python interpreter
    if importlib.util.find_spec("pyrate_ta") is not None:
        py_exe = sys.executable
        if sys.platform == "win32":
            pyw = Path(py_exe).with_name("pythonw.exe")
            if pyw.is_file():
                py_exe = str(pyw)
        return [py_exe, "-m", "pyrate_ta.gui.app"]

    # 3. Check if pyrate-ta script exists in current venv Scripts/bin directory
    venv_dir = Path(sys.executable).parent
    for name in ("pyrate-ta.exe", "pyrate-ta", "pyrate_ta.exe"):
        cand = venv_dir / name
        if cand.is_file():
            return [str(cand)]

    # 4. Check system PATH for pyrate-ta
    path_exe = shutil.which("pyrate-ta") or shutil.which("pyrate-ta.exe")
    if path_exe:
        return [path_exe]

    # 5. Check sibling PyRATE-TA repository venv (development mode)
    try:
        cur_file = Path(__file__).resolve()
        github_dir = cur_file.parents[4]
        sibling_pyrate = github_dir / "PyRATE-TA"
        if sibling_pyrate.is_dir():
            for py_cand in (
                sibling_pyrate / ".venv" / "Scripts" / "pythonw.exe",
                sibling_pyrate / ".venv" / "Scripts" / "python.exe",
                sibling_pyrate / ".venv" / "bin" / "python",
                sibling_pyrate / ".venv" / "Scripts" / "pyrate-ta.exe",
            ):
                if py_cand.is_file():
                    if "python" in py_cand.name.lower():
                        return [str(py_cand), "-m", "pyrate_ta.gui.app"]
                    return [str(py_cand)]
    except Exception:
        pass

    return None


def is_pyrate_installed() -> bool:
    """Check if PyRATE-TA is installed and available in the current or companion environment."""
    return find_pyrate_command() is not None


def get_pyrate_version() -> str | None:
    """Return the installed version of PyRATE-TA, or None if not installed."""
    if not is_pyrate_installed():
        return None
    try:
        import pyrate_ta

        ver = getattr(pyrate_ta, "__version__", None)
        if ver:
            return str(ver)
    except Exception:
        pass
    try:
        return importlib.metadata.version("pyrate-ta")
    except Exception:
        return "unknown"


def launch_pyrate(
    dataset_or_path: Dataset1D | str | Path | None = None,
    extra_args: list[str] | None = None,
    detector: int | None = None,
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
    detector : int, optional
        Detector channel index to export when handing off a multi-detector dataset.

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
            "PyRATE-TA is not installed in the active or companion Python environment.\n\n"
            "Install it using:\n"
            "    uv pip install pyrate-ta\n"
            "or clone the companion repository:\n"
            "    https://github.com/RJFernandezTeran/PyRATE-TA"
        )

    base_cmd = find_pyrate_command() or [sys.executable, "-m", "pyrate_ta.gui.app"]
    cmd = list(base_cmd)

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
            if detector is not None and getattr(dataset_or_path, "n_detectors", 1) > 1:
                safe_name = f"{safe_name}_det{detector + 1}"
            target_file = temp_dir / f"{safe_name}_for_kinetic_analysis.pdat"
            exported = export_pdat(dataset_or_path, target_file, detector=detector)
            if exported:
                target_file = exported[0]
            logger.info("Exported active dataset to %s for PyRATE-TA", target_file)

    if target_file is not None:
        cmd.append(str(target_file))

    if extra_args:
        cmd.extend(extra_args)

    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200
        )
        kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True

    logger.info("Launching PyRATE-TA: %s", " ".join(cmd))
    return subprocess.Popen(cmd, **kwargs)

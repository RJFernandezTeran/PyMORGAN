"""Local Python Environment Setup utility and GUI wizard for PyMORGAN & PyRATE-TA.

Allows users who installed the standalone PyMORGAN Suite (or cloned the repositories)
to make ``pymorgan`` and ``pyrate_ta`` importable in their native OS Python
environments (Conda/Miniconda, Homebrew/system Python, virtual environments,
Jupyter, Spyder, VS Code) with one click or via the CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# Required third-party packages (import_name, pip_spec)
REQUIRED_THIRD_PARTY_DEPS: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy>=1.26.0"),
    ("scipy", "scipy>=1.12.0"),
    ("matplotlib", "matplotlib>=3.8.0"),
    ("pandas", "pandas>=2.1.0"),
    ("h5py", "h5py>=3.10.0"),
    ("PIL", "Pillow>=10.0.0"),
    ("tomlkit", "tomlkit>=0.12.0"),
    ("mpl_axes_aligner", "mpl-axes-aligner>=1.3"),
    ("cmcrameri", "cmcrameri>=1.8"),
    ("rinoh_typeface_texgyreheros", "rinoh-typeface-texgyreheros"),
    ("PyQt6", "PyQt6>=6.6.0"),
)


@dataclass
class PythonEnvInfo:
    """Metadata for a discovered local Python interpreter."""

    executable: Path
    version: str
    env_type: str  # e.g. "Conda (base)", "System Python", "Virtualenv"
    prefix: Path
    has_pymorgan: bool = False
    pymorgan_version: str = ""
    has_pyrate: bool = False
    pyrate_version: str = ""

    @property
    def display_label(self) -> str:
        status = ""
        if self.has_pymorgan and self.has_pyrate:
            status = f"  [pm v{self.pymorgan_version} ✓]"
        return f"{self.env_type} — Python {self.version}{status}  ({self.executable})"


def _subprocess_no_window_kwargs() -> dict:
    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    return kwargs


def _probe_python_executable(exe: Path, timeout: float = 4.0) -> PythonEnvInfo | None:
    """Inspect a candidate Python executable and return its version and package status."""
    try:
        exe = exe.resolve()
    except Exception:
        pass
    if not exe.is_file():
        return None

    # Skip Windows Store zero-byte stub executables
    if sys.platform == "win32" and "WindowsApps" in str(exe):
        try:
            if exe.stat().st_size == 0:
                return None
        except Exception:
            return None

    # Skip frozen PyMORGAN executables
    if exe.stem.lower() in ("pymorgan", "pymorgan_suite", "pyrate-ta", "pyrate_ta"):
        return None

    probe_code = (
        "import json, sys, importlib.util; "
        "v = f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'; "
        "ok_ver = (sys.version_info >= (3, 11)); "
        "pm_v = ''; pr_v = ''; "
        "exec('try:\\n import pymorgan as pm; pm_v = getattr(pm, \"__version__\", \"installed\")\\nexcept Exception: pass'); "
        "exec('try:\\n import pyrate_ta as pr; pr_v = getattr(pr, \"__version__\", \"installed\")\\nexcept Exception: pass'); "
        "print(json.dumps({'ver': v, 'ok': ok_ver, 'prefix': sys.prefix, 'pm': pm_v, 'pr': pr_v}))"
    )
    try:
        res = subprocess.run(
            [str(exe), "-c", probe_code],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            **_subprocess_no_window_kwargs(),
        )
        if res.returncode != 0 or not res.stdout.strip():
            return None
        last_line = res.stdout.strip().splitlines()[-1]
        data = json.loads(last_line)
        if not data.get("ok"):
            return None
        prefix = Path(data.get("prefix", str(exe.parent)))
        env_type = _classify_env(exe, prefix)
        pm_v = data.get("pm", "")
        pr_v = data.get("pr", "")
        return PythonEnvInfo(
            executable=exe,
            version=data["ver"],
            env_type=env_type,
            prefix=prefix,
            has_pymorgan=bool(pm_v),
            pymorgan_version=pm_v,
            has_pyrate=bool(pr_v),
            pyrate_version=pr_v,
        )
    except Exception:
        return None


def _classify_env(exe: Path, prefix: Path) -> str:
    exe_str = str(exe).lower()
    prefix_str = str(prefix).lower()
    if (prefix / "conda-meta").is_dir() or "conda" in prefix_str or "miniforge" in prefix_str or "mamba" in prefix_str:
        if prefix.parent.name.lower() == "envs":
            return f"Conda ({prefix.name})"
        return f"Conda (base: {prefix.name})"
    if (prefix / "pyvenv.cfg").is_file() or ".venv" in exe_str or "venv" in exe_str:
        return f"Virtualenv ({prefix.name})"
    if "homebrew" in exe_str or "/usr/local" in exe_str:
        return "Homebrew Python"
    if "pyenv" in exe_str:
        return f"pyenv ({prefix.name})"
    return "System Python"


def discover_python_environments() -> list[PythonEnvInfo]:
    """Scan the host OS for Python >= 3.11 installations (Conda, system, Homebrew, pyenv, venv)."""
    candidates: list[Path] = []

    # 1. If currently running in a non-frozen Python, include sys.executable first
    if not getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable))

    # 2. Windows Python Launcher (py -0p)
    if sys.platform == "win32":
        py_launcher = shutil.which("py")
        if py_launcher:
            try:
                res = subprocess.run(
                    [py_launcher, "-0p"],
                    capture_output=True,
                    text=True,
                    timeout=3.0,
                    check=False,
                    **_subprocess_no_window_kwargs(),
                )
                if res.returncode == 0:
                    for line in res.stdout.splitlines():
                        parts = line.strip().split()
                        if parts:
                            cand = Path(parts[-1].strip())
                            if cand.is_file():
                                candidates.append(cand)
            except Exception:
                pass

    # 3. Standard PATH executables
    for name in ("python3", "python", "python3.14", "python3.13", "python3.12", "python3.11"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))

    # 4. Conda environments listed in ~/.conda/environments.txt
    home = Path.home()
    conda_envs_file = home / ".conda" / "environments.txt"
    conda_prefixes: list[Path] = []
    if conda_envs_file.is_file():
        try:
            for line in conda_envs_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                p = Path(line.strip())
                if p.is_dir():
                    conda_prefixes.append(p)
        except Exception:
            pass

    # Common Conda root folders
    common_conda_roots = [
        home / "miniconda3",
        home / "anaconda3",
        home / "miniforge3",
        home / "mambaforge",
        home / "micromamba",
    ]
    if sys.platform == "win32":
        local_app = Path(os.environ.get("LOCALAPPDATA", ""))
        prog_data = Path(os.environ.get("PROGRAMDATA", "C:/ProgramData"))
        for base in (local_app, prog_data):
            if base.is_dir():
                for sub in ("miniconda3", "anaconda3", "miniforge3", "mambaforge"):
                    common_conda_roots.append(base / sub)
    elif sys.platform == "darwin":
        common_conda_roots.extend(
            [
                Path("/opt/homebrew/Caskroom/miniconda/base"),
                Path("/opt/homebrew/Caskroom/miniforge/base"),
                Path("/opt/anaconda3"),
                Path("/opt/miniconda3"),
            ]
        )
    else:
        common_conda_roots.extend([Path("/opt/conda"), Path("/opt/miniconda3"), Path("/opt/anaconda3")])

    for c_root in common_conda_roots:
        if c_root.is_dir():
            conda_prefixes.append(c_root)
            envs_dir = c_root / "envs"
            if envs_dir.is_dir():
                for sub_env in envs_dir.iterdir():
                    if sub_env.is_dir():
                        conda_prefixes.append(sub_env)

    for prefix in conda_prefixes:
        if sys.platform == "win32":
            py_exe = prefix / "python.exe"
        else:
            py_exe = prefix / "bin" / "python3"
            if not py_exe.is_file():
                py_exe = prefix / "bin" / "python"
        if py_exe.is_file():
            candidates.append(py_exe)

    # 5. OS-specific standard install directories
    if sys.platform == "win32":
        local_app = Path(os.environ.get("LOCALAPPDATA", ""))
        for pattern_root in [
            local_app / "Programs" / "Python",
            local_app / "Python",
            Path("C:/"),
        ]:
            if pattern_root.is_dir():
                for sub in pattern_root.glob("Python3*"):
                    if (sub / "python.exe").is_file():
                        candidates.append(sub / "python.exe")
                for sub in pattern_root.glob("pythoncore-3*"):
                    if (sub / "python.exe").is_file():
                        candidates.append(sub / "python.exe")
    elif sys.platform == "darwin":
        for p_str in (
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3",
        ):
            if Path(p_str).is_file():
                candidates.append(Path(p_str))
        fw_root = Path("/Library/Frameworks/Python.framework/Versions")
        if fw_root.is_dir():
            for ver_dir in fw_root.iterdir():
                cand = ver_dir / "bin" / "python3"
                if cand.is_file():
                    candidates.append(cand)

    # 6. pyenv and uv managed Pythons
    for managed_glob in [
        home / ".pyenv" / "versions" / "*" / "bin" / "python3",
        home / ".local" / "share" / "uv" / "python" / "*" / "bin" / "python3",
    ]:
        try:
            for p in managed_glob.parent.parent.glob(managed_glob.parent.name + "/" + managed_glob.name):
                if p.is_file():
                    candidates.append(p)
        except Exception:
            pass

    # Deduplicate and probe
    seen_paths: set[str] = set()
    results: list[PythonEnvInfo] = []
    for cand in candidates:
        try:
            norm = os.path.normcase(str(cand.resolve()))
        except Exception:
            norm = os.path.normcase(str(cand))
        if norm in seen_paths:
            continue
        seen_paths.add(norm)
        info = _probe_python_executable(cand)
        if info is not None:
            results.append(info)

    return results


def _ensure_local_source_tree(log_fn: Callable[[str], None]) -> list[Path]:
    """Return list of source root directories containing ``pymorgan`` and ``pyrate_ta`` .py packages.

    If running from a git checkout / editable source tree, returns those ``src``
    directories alongside ``~/.config/pymorgan/bleeding_edge_src``. If running from
    a frozen installer where ``bleeding_edge_src`` has not yet been populated,
    downloads the latest ``main`` sources into ``~/.config/pymorgan/bleeding_edge_src``.
    """
    from pymorgan.gui.updater import BLEEDING_EDGE_DIR

    source_dirs: list[Path] = [BLEEDING_EDGE_DIR]

    # Check if current process is running from non-frozen source tree with .py files
    if not getattr(sys, "frozen", False):
        try:
            import pymorgan as pm

            pm_init = Path(pm.__file__).resolve()
            if pm_init.suffix == ".py":
                pm_src_root = pm_init.parent.parent
                if pm_src_root not in source_dirs:
                    source_dirs.append(pm_src_root)
                sibling_pyrate = pm_src_root.parent.parent / "PyRATE-TA" / "src"
                if (sibling_pyrate / "pyrate_ta" / "__init__.py").is_file():
                    source_dirs.append(sibling_pyrate)
        except Exception:
            pass

    has_pm_py = any((d / "pymorgan" / "__init__.py").is_file() for d in source_dirs)
    has_pr_py = any((d / "pyrate_ta" / "__init__.py").is_file() for d in source_dirs)

    if not (has_pm_py and has_pr_py):
        log_fn("Syncing pure-Python library sources to ~/.config/pymorgan/bleeding_edge_src...")
        from pymorgan.gui.updater import BleedingEdgeSyncWorker

        worker = BleedingEdgeSyncWorker()
        if not has_pm_py:
            worker._download_and_extract_pkg(
                "RJFernandezTeran/PyMORGAN", "pymorgan", BLEEDING_EDGE_DIR, 10, 50
            )
        if not has_pr_py:
            worker._download_and_extract_pkg(
                "RJFernandezTeran/PyRATE-TA", "pyrate_ta", BLEEDING_EDGE_DIR, 50, 90
            )
        log_fn("Library sources synced to ~/.config/pymorgan/bleeding_edge_src.")

    BLEEDING_EDGE_DIR.mkdir(parents=True, exist_ok=True)
    return source_dirs


def _ensure_pip_in_env(python_exe: Path, log_fn: Callable[[str], None]) -> bool:
    """Ensure ``pip`` is available in ``python_exe`` (running ``ensurepip`` if needed)."""
    res = subprocess.run(
        [str(python_exe), "-m", "pip", "--version"],
        capture_output=True,
        text=True,
        check=False,
        **_subprocess_no_window_kwargs(),
    )
    if res.returncode == 0:
        return True
    log_fn("Bootstrapping pip via ensurepip...")
    ep = subprocess.run(
        [str(python_exe), "-m", "ensurepip", "--upgrade"],
        capture_output=True,
        text=True,
        check=False,
        **_subprocess_no_window_kwargs(),
    )
    return ep.returncode == 0


def configure_python_environment(
    python_exe: str | Path,
    mode: str = "pth_bridge",
    install_fonts: bool = True,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> tuple[bool, str]:
    """Configure ``python_exe`` so ``import pymorgan`` and ``import pyrate_ta`` work natively.

    Parameters
    ----------
    python_exe : str or Path
        Target Python >= 3.11 interpreter executable.
    mode : {"pth_bridge", "pypi", "github_main"}
        - ``"pth_bridge"`` (default): Links ``site-packages/pymorgan_suite.pth`` to
          ``~/.config/pymorgan/bleeding_edge_src`` (and local repo ``src`` if available)
          and installs only missing third-party dependencies.
        - ``"pypi"``: Runs ``pip install -U pymorgan pyrate-ta`` from PyPI.
        - ``"github_main"``: Runs ``pip install -U git+...`` from GitHub ``main``.
    install_fonts : bool, default True
        Whether to install and verify TeX Gyre Heros in the target environment's Matplotlib cache.
    """
    py_path = Path(python_exe)

    def _log(msg: str) -> None:
        if log_callback is not None:
            log_callback(msg)

    def _prog(pct: int, label: str) -> None:
        if progress_callback is not None:
            progress_callback(pct, label)

    _prog(5, "Inspecting target Python environment...")
    info = _probe_python_executable(py_path, timeout=8.0)
    if info is None:
        return False, f"Invalid or unsupported Python interpreter (requires Python >= 3.11):\n{py_path}"

    _log(f"Target Python: {info.executable} (Python {info.version} — {info.env_type})")

    if mode == "pth_bridge":
        _prog(15, "Preparing PyMORGAN & PyRATE-TA source directories...")
        src_dirs = _ensure_local_source_tree(_log)

        _prog(30, "Checking installed scientific dependencies...")
        check_script = (
            "import importlib.util, json; "
            f"mods = {[m for m, _ in REQUIRED_THIRD_PARTY_DEPS]!r}; "
            "missing = [m for m in mods if importlib.util.find_spec(m) is None]; "
            "print(json.dumps(missing))"
        )
        res = subprocess.run(
            [str(py_path), "-c", check_script],
            capture_output=True,
            text=True,
            check=False,
            **_subprocess_no_window_kwargs(),
        )
        missing_mods: list[str] = []
        if res.returncode == 0 and res.stdout.strip():
            try:
                missing_mods = json.loads(res.stdout.strip().splitlines()[-1])
            except Exception:
                missing_mods = [m for m, _ in REQUIRED_THIRD_PARTY_DEPS]
        else:
            missing_mods = [m for m, _ in REQUIRED_THIRD_PARTY_DEPS]

        if missing_mods:
            pip_specs = [spec for mod, spec in REQUIRED_THIRD_PARTY_DEPS if mod in missing_mods]
            _log(f"Installing missing dependencies ({', '.join(missing_mods)})...")
            _prog(45, f"Installing {len(pip_specs)} missing package(s)...")
            _ensure_pip_in_env(py_path, _log)
            pip_cmd = [str(py_path), "-m", "pip", "install", *pip_specs]
            pip_res = subprocess.run(
                pip_cmd,
                capture_output=True,
                text=True,
                check=False,
                **_subprocess_no_window_kwargs(),
            )
            if pip_res.returncode != 0:
                _log(pip_res.stdout[-600:] if pip_res.stdout else "")
                _log(pip_res.stderr[-600:] if pip_res.stderr else "")
                return False, f"Failed to install dependencies via pip:\n{pip_res.stderr[-400:]}"
            _log("Third-party dependencies installed successfully.")
        else:
            _log("All required third-party dependencies are already present in this environment.")

        _prog(65, "Writing pymorgan_suite.pth into site-packages...")
        sp_code = (
            "import site, sysconfig, pathlib; "
            "cands = []; "
            "exec('try:\\n cands.extend(site.getsitepackages())\\nexcept Exception: pass'); "
            "pure = sysconfig.get_path('purelib'); "
            "if pure: cands.insert(0, pure); "
            "exec('try:\\n cands.append(site.getusersitepackages())\\nexcept Exception: pass'); "
            "for c in cands:\n"
            "    p = pathlib.Path(c)\n"
            "    try:\n"
            "        p.mkdir(parents=True, exist_ok=True)\n"
            "        t = p / '.pm_write_test'\n"
            "        t.write_text('ok')\n"
            "        t.unlink()\n"
            "        print(str(p))\n"
            "        break\n"
            "    except Exception:\n"
            "        continue"
        )
        sp_res = subprocess.run(
            [str(py_path), "-c", sp_code],
            capture_output=True,
            text=True,
            check=False,
            **_subprocess_no_window_kwargs(),
        )
        if sp_res.returncode != 0 or not sp_res.stdout.strip():
            return False, "Could not locate a writable site-packages directory in the target environment."

        site_pkg_dir = Path(sp_res.stdout.strip().splitlines()[-1])
        pth_file = site_pkg_dir / "pymorgan_suite.pth"
        pth_lines = [str(d.resolve()) for d in src_dirs]
        pth_file.write_text("\n".join(pth_lines) + "\n", encoding="utf-8")
        _log(f"Created live bridge: {pth_file}")
        for line in pth_lines:
            _log(f"  -> {line}")

    elif mode in ("pypi", "github_main"):
        _ensure_pip_in_env(py_path, _log)
        if mode == "pypi":
            _prog(35, "Installing pymorgan & pyrate-ta from PyPI...")
            _log("Running: pip install -U pymorgan pyrate-ta")
            pip_cmd = [str(py_path), "-m", "pip", "install", "-U", "pymorgan", "pyrate-ta"]
        else:
            _prog(35, "Installing pymorgan & pyrate-ta from GitHub main...")
            _log("Running: pip install -U git+https://github.com/RJFernandezTeran/PyMORGAN.git git+https://github.com/RJFernandezTeran/PyRATE-TA.git")
            pip_cmd = [
                str(py_path),
                "-m",
                "pip",
                "install",
                "-U",
                "git+https://github.com/RJFernandezTeran/PyMORGAN.git",
                "git+https://github.com/RJFernandezTeran/PyRATE-TA.git",
            ]
        pip_res = subprocess.run(
            pip_cmd,
            capture_output=True,
            text=True,
            check=False,
            **_subprocess_no_window_kwargs(),
        )
        if pip_res.returncode != 0:
            _log(pip_res.stderr[-600:] if pip_res.stderr else "")
            return False, f"pip install failed:\n{pip_res.stderr[-400:]}"
        _log("Packages installed via pip.")
    else:
        return False, f"Unknown installation mode: {mode}"

    # Verify imports & optionally configure TeX Gyre Heros fonts
    _prog(85, "Verifying imports & Matplotlib fonts...")
    verify_script = (
        "import json, pymorgan as pm, pyrate_ta as pr; "
        f"do_fonts = {bool(install_fonts)!r}; "
        "fonts = []; "
        "if do_fonts:\n"
        "    from pymorgan.fonts import install_fonts_twice, available_preferred_fonts\n"
        "    install_fonts_twice(quiet=True)\n"
        "    fonts = available_preferred_fonts()\n"
        "print(json.dumps({'pm': pm.__version__, 'pr': pr.__version__, 'fonts': fonts}))"
    )
    v_res = subprocess.run(
        [str(py_path), "-c", verify_script],
        capture_output=True,
        text=True,
        check=False,
        **_subprocess_no_window_kwargs(),
    )
    if v_res.returncode != 0 or not v_res.stdout.strip():
        err_detail = (v_res.stderr or v_res.stdout or "Unknown error").strip()[-400:]
        return False, f"Import verification failed in target environment:\n{err_detail}"

    v_data = json.loads(v_res.stdout.strip().splitlines()[-1])
    pm_v = v_data.get("pm", "unknown")
    pr_v = v_data.get("pr", "unknown")
    fonts_list = v_data.get("fonts", [])
    _log(f"Verified imports! pymorgan v{pm_v}, pyrate_ta v{pr_v}")
    if fonts_list:
        _log(f"Configured Matplotlib preferred fonts: {', '.join(fonts_list)}")

    _prog(100, f"Ready! pymorgan v{pm_v} & pyrate_ta v{pr_v} configured.")
    summary = (
        f"Successfully configured Python {info.version} ({info.env_type})!\n\n"
        f"  • Executable: {info.executable}\n"
        f"  • PyMORGAN: v{pm_v} (import pymorgan as pm)\n"
        f"  • PyRATE-TA: v{pr_v} (import pyrate_ta as pr)\n"
    )
    if mode == "pth_bridge":
        summary += (
            "\nLive-linked via pymorgan_suite.pth: whenever you click 'Sync Bleeding-Edge' "
            "in the GUI, this Python environment will automatically use the updated libraries!"
        )
    return True, summary


def open_local_python_setup_dialog(parent=None) -> None:
    """Open the interactive Qt dialog for configuring a local Python environment."""
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtWidgets import (
        QButtonGroup,
        QCheckBox,
        QComboBox,
        QDialog,
        QFileDialog,
        QFrame,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QRadioButton,
        QTextEdit,
        QVBoxLayout,
    )

    try:
        from pymorgan.gui.theme import is_dark_palette

        dark = bool(is_dark_palette())
    except Exception:
        dark = False

    class _EnvScanWorker(QThread):
        finished = pyqtSignal(list)

        def run(self):
            self.finished.emit(discover_python_environments())

    class _EnvSetupWorker(QThread):
        log_line = pyqtSignal(str)
        progress = pyqtSignal(int, str)
        finished = pyqtSignal(bool, str)

        def __init__(self, exe_path: str, mode: str, install_fonts: bool, parent_w=None):
            super().__init__(parent_w)
            self.exe_path = exe_path
            self.mode = mode
            self.install_fonts = install_fonts

        def run(self):
            ok, msg = configure_python_environment(
                self.exe_path,
                mode=self.mode,
                install_fonts=self.install_fonts,
                log_callback=self.log_line.emit,
                progress_callback=self.progress.emit,
            )
            self.finished.emit(ok, msg)

    dlg = QDialog(parent)
    dlg.setObjectName("LocalPythonSetupDialog")
    dlg.setWindowTitle("Setup PyMORGAN & PyRATE-TA in Local Python Environment")
    dlg.resize(680, 540)
    dlg.setModal(True)

    if dark:
        dlg.setStyleSheet("""
            QDialog#LocalPythonSetupDialog {
                background-color: #181c24;
                color: #f1f5f9;
                font-family: 'Segoe UI', -apple-system, Arial, sans-serif;
            }
            QDialog#LocalPythonSetupDialog QLabel {
                color: #f1f5f9;
                background: transparent;
                border: none;
            }
            QDialog#LocalPythonSetupDialog QFrame#cardBox {
                background-color: #222834;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 10px;
            }
            QDialog#LocalPythonSetupDialog QComboBox {
                background-color: #12161d;
                color: #f8fafc;
                border: 1px solid #475569;
                border-radius: 5px;
                padding: 5px 8px;
                font-size: 12px;
            }
            QDialog#LocalPythonSetupDialog QRadioButton,
            QDialog#LocalPythonSetupDialog QCheckBox {
                color: #e2e8f0;
                font-size: 12px;
                spacing: 6px;
            }
            QDialog#LocalPythonSetupDialog QTextEdit {
                background-color: #12161d;
                color: #cbd5e1;
                border: 1px solid #334155;
                border-radius: 6px;
                font-family: Consolas, 'Courier New', monospace;
                font-size: 11.5px;
                padding: 6px;
            }
            QDialog#LocalPythonSetupDialog QPushButton {
                background-color: #222834;
                color: #e2e8f0;
                border: 1px solid #475569;
                border-radius: 6px;
                padding: 6px 14px;
                font-weight: bold;
                font-size: 12px;
            }
            QDialog#LocalPythonSetupDialog QPushButton:hover {
                background-color: #334155;
                color: #ffffff;
            }
            QDialog#LocalPythonSetupDialog QPushButton#btnInstall {
                background-color: #0d9488;
                color: #ffffff;
                border: 1px solid #2dd4bf;
            }
            QDialog#LocalPythonSetupDialog QPushButton#btnInstall:hover {
                background-color: #0f766e;
            }
            QDialog#LocalPythonSetupDialog QProgressBar {
                border: 1px solid #334155;
                border-radius: 4px;
                background-color: #12161d;
                text-align: center;
                color: #ffffff;
                height: 20px;
            }
            QDialog#LocalPythonSetupDialog QProgressBar::chunk {
                background-color: #0d9488;
                border-radius: 3px;
            }
        """)
    else:
        dlg.setStyleSheet("""
            QDialog#LocalPythonSetupDialog {
                background-color: #f8fafc;
                color: #0f172a;
                font-family: 'Segoe UI', -apple-system, Arial, sans-serif;
            }
            QDialog#LocalPythonSetupDialog QLabel {
                color: #0f172a;
                background: transparent;
                border: none;
            }
            QDialog#LocalPythonSetupDialog QFrame#cardBox {
                background-color: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                padding: 10px;
            }
            QDialog#LocalPythonSetupDialog QComboBox {
                background-color: #ffffff;
                color: #0f172a;
                border: 1px solid #94a3b8;
                border-radius: 5px;
                padding: 5px 8px;
                font-size: 12px;
            }
            QDialog#LocalPythonSetupDialog QRadioButton,
            QDialog#LocalPythonSetupDialog QCheckBox {
                color: #1e293b;
                font-size: 12px;
                spacing: 6px;
            }
            QDialog#LocalPythonSetupDialog QTextEdit {
                background-color: #ffffff;
                color: #1e293b;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                font-family: Consolas, 'Courier New', monospace;
                font-size: 11.5px;
                padding: 6px;
            }
            QDialog#LocalPythonSetupDialog QPushButton {
                background-color: #ffffff;
                color: #1e293b;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 6px 14px;
                font-weight: bold;
                font-size: 12px;
            }
            QDialog#LocalPythonSetupDialog QPushButton:hover {
                background-color: #e2e8f0;
                color: #0f172a;
            }
            QDialog#LocalPythonSetupDialog QPushButton#btnInstall {
                background-color: #0d9488;
                color: #ffffff;
                border: 1px solid #0f766e;
            }
            QDialog#LocalPythonSetupDialog QPushButton#btnInstall:hover {
                background-color: #0f766e;
            }
            QDialog#LocalPythonSetupDialog QProgressBar {
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                background-color: #e2e8f0;
                text-align: center;
                color: #0f172a;
                font-weight: bold;
                height: 20px;
            }
            QDialog#LocalPythonSetupDialog QProgressBar::chunk {
                background-color: #0d9488;
                border-radius: 3px;
            }
        """)

    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(20, 18, 20, 18)
    layout.setSpacing(12)

    title_color = "#60a5fa" if dark else "#1d4ed8"
    muted_color = "#94a3b8" if dark else "#475569"

    lbl_title = QLabel("Use PyMORGAN & PyRATE-TA in Your Own Python Environment")
    lbl_title.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {title_color};")
    layout.addWidget(lbl_title)

    lbl_sub = QLabel(
        "Select any Python >= 3.11 environment on your computer (Conda, Homebrew, system Python, "
        "or a project virtualenv) to enable <b>import pymorgan as pm</b> and <b>import pyrate_ta as pr</b> "
        "in Jupyter Notebooks, VS Code, Spyder, or terminal scripts."
    )
    lbl_sub.setWordWrap(True)
    lbl_sub.setStyleSheet(f"font-size: 12px; color: {muted_color};")
    layout.addWidget(lbl_sub)

    # 1. Environment selector box
    env_card = QFrame()
    env_card.setObjectName("cardBox")
    env_vbox = QVBoxLayout(env_card)
    env_vbox.setSpacing(6)

    lbl_env = QLabel("<b>1. Target Python Environment (Python &gt;= 3.11):</b>")
    env_vbox.addWidget(lbl_env)

    combo_row = QHBoxLayout()
    combo_row.setSpacing(8)
    combo_env = QComboBox()
    combo_env.addItem("Scanning for Python environments on your system...", "")
    combo_env.setEnabled(False)
    combo_row.addWidget(combo_env, 1)

    btn_browse = QPushButton("Browse...")
    btn_browse.setToolTip("Select a custom python.exe / python3 executable")
    combo_row.addWidget(btn_browse)
    env_vbox.addLayout(combo_row)
    layout.addWidget(env_card)

    # 2. Mode selection box
    mode_card = QFrame()
    mode_card.setObjectName("cardBox")
    mode_vbox = QVBoxLayout(mode_card)
    mode_vbox.setSpacing(6)

    lbl_mode = QLabel("<b>2. Setup Method:</b>")
    mode_vbox.addWidget(lbl_mode)

    rb_bridge = QRadioButton(
        "Live-Linked Bridge (.pth) — Recommended (instant, stays auto-synced with Bleeding-Edge updates)"
    )
    rb_bridge.setChecked(True)
    rb_pypi = QRadioButton("Standard PyPI Install (pip install -U pymorgan pyrate-ta)")
    rb_github = QRadioButton("GitHub Main Branch Install (pip install -U git+https://github.com/...)")

    btn_group = QButtonGroup(dlg)
    btn_group.addButton(rb_bridge)
    btn_group.addButton(rb_pypi)
    btn_group.addButton(rb_github)

    mode_vbox.addWidget(rb_bridge)
    mode_vbox.addWidget(rb_pypi)
    mode_vbox.addWidget(rb_github)

    chk_fonts = QCheckBox("Install & verify TeX Gyre Heros fonts in Matplotlib for this environment")
    chk_fonts.setChecked(True)
    mode_vbox.addWidget(chk_fonts)

    layout.addWidget(mode_card)

    # 3. Log & progress
    log_box = QTextEdit()
    log_box.setReadOnly(True)
    log_box.setPlaceholderText("Setup output will appear here...")
    layout.addWidget(log_box, 1)

    progress_bar = QProgressBar()
    progress_bar.setVisible(False)
    layout.addWidget(progress_bar)

    # 4. Action buttons
    btn_row = QHBoxLayout()
    btn_row.setSpacing(10)

    btn_install = QPushButton("Configure Selected Python\nInstall & Verify")
    btn_install.setObjectName("btnInstall")
    btn_install.setMinimumHeight(42)
    btn_install.setCursor(Qt.CursorShape.PointingHandCursor)
    btn_install.setEnabled(False)
    btn_row.addWidget(btn_install)

    btn_row.addStretch()

    btn_close = QPushButton("Close")
    btn_close.setMinimumHeight(42)
    btn_close.clicked.connect(dlg.reject)
    btn_row.addWidget(btn_close)

    layout.addLayout(btn_row)

    def _on_envs_found(envs: list[PythonEnvInfo]):
        combo_env.clear()
        combo_env.setEnabled(True)
        for env in envs:
            combo_env.addItem(env.display_label, str(env.executable))
        if envs:
            btn_install.setEnabled(True)
            log_box.append(f"Discovered {len(envs)} compatible Python >= 3.11 environment(s).")
        else:
            combo_env.addItem("No Python >= 3.11 found automatically — click 'Browse...' to select python", "")
            log_box.append(
                "No Python >= 3.11 environment was found on standard paths. "
                "Click 'Browse...' to select your python executable, or install Python / Miniconda first."
            )

    def _on_browse():
        filt = "Python Executable (python.exe pythonw.exe);;All Files (*)" if sys.platform == "win32" else "Python Executable (python*)"
        path, _ = QFileDialog.getOpenFileName(dlg, "Select Python Interpreter", str(Path.home()), filt)
        if not path:
            return
        info = _probe_python_executable(Path(path))
        if info is None:
            QMessageBox.warning(
                dlg,
                "Invalid Python Interpreter",
                f"The selected file is not a valid Python >= 3.11 executable:\n{path}",
            )
            return
        if combo_env.count() == 1 and not combo_env.itemData(0):
            combo_env.clear()
        combo_env.insertItem(0, info.display_label, str(info.executable))
        combo_env.setCurrentIndex(0)
        btn_install.setEnabled(True)

    def _on_run_setup():
        exe_path = combo_env.currentData()
        if not exe_path:
            return
        if rb_bridge.isChecked():
            mode = "pth_bridge"
        elif rb_pypi.isChecked():
            mode = "pypi"
        else:
            mode = "github_main"

        btn_install.setEnabled(False)
        btn_browse.setEnabled(False)
        combo_env.setEnabled(False)
        progress_bar.setVisible(True)
        progress_bar.setValue(0)
        log_box.clear()

        worker = _EnvSetupWorker(exe_path, mode, chk_fonts.isChecked(), dlg)
        dlg._setup_worker = worker

        def _on_prog(pct: int, msg: str):
            progress_bar.setValue(pct)
            progress_bar.setFormat(f"{pct}% - {msg}")

        def _on_done(ok: bool, summary: str):
            btn_install.setEnabled(True)
            btn_browse.setEnabled(True)
            combo_env.setEnabled(True)
            if ok:
                progress_bar.setValue(100)
                QMessageBox.information(dlg, "Python Environment Configured", summary)
            else:
                progress_bar.setVisible(False)
                QMessageBox.warning(dlg, "Configuration Error", summary)

        worker.log_line.connect(log_box.append)
        worker.progress.connect(_on_prog)
        worker.finished.connect(_on_done)
        worker.start()

    btn_browse.clicked.connect(_on_browse)
    btn_install.clicked.connect(_on_run_setup)

    scan_worker = _EnvScanWorker(dlg)
    dlg._scan_worker = scan_worker
    scan_worker.finished.connect(_on_envs_found)
    scan_worker.start()

    dlg.exec()


def cli_main(argv: list[str] | None = None) -> int:
    """Command-line entry point to link or install PyMORGAN & PyRATE-TA into a local Python."""
    parser = argparse.ArgumentParser(
        description="Configure PyMORGAN & PyRATE-TA in a local Python >= 3.11 environment."
    )
    parser.add_argument(
        "--python",
        type=str,
        default="",
        help="Path to target python executable (defaults to first discovered environment).",
    )
    parser.add_argument(
        "--mode",
        choices=["pth_bridge", "pypi", "github_main"],
        default="pth_bridge",
        help="Setup mode: 'pth_bridge' (default live-linked .pth), 'pypi', or 'github_main'.",
    )
    parser.add_argument(
        "--no-fonts",
        action="store_true",
        help="Skip TeX Gyre Heros font installation in the target environment.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered Python >= 3.11 environments and exit.",
    )
    args = parser.parse_args(argv)

    envs = discover_python_environments()
    if args.list:
        if not envs:
            print("No Python >= 3.11 environments found.")
            return 1
        print("Discovered Python >= 3.11 environments:")
        for idx, e in enumerate(envs, 1):
            print(f"  [{idx}] {e.display_label}")
        return 0

    target_exe = args.python
    if not target_exe:
        if not envs:
            print("Error: No Python >= 3.11 environment found. Pass --python <path/to/python>.")
            return 1
        target_exe = str(envs[0].executable)

    ok, summary = configure_python_environment(
        target_exe,
        mode=args.mode,
        install_fonts=not args.no_fonts,
        log_callback=lambda m: print(f"[setup-env] {m}"),
    )
    print("\n" + summary)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(cli_main())

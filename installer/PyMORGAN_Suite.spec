# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification file for PyMORGAN Suite.

Bundles PyMORGAN, PyRATE-TA, and the unified Suite Launcher into a single
shared-runtime directory distribution with zero command-line console windows.
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

SPEC_DIR = Path(SPECPATH).resolve()
REPO_ROOT = SPEC_DIR.parent
PYMORGAN_SRC = REPO_ROOT / "src"
PYRATE_SRC = REPO_ROOT.parent / "PyRATE-TA" / "src"

sys.path.insert(0, str(PYMORGAN_SRC))
if PYRATE_SRC.exists():
    sys.path.insert(0, str(PYRATE_SRC))

# Collect data files
datas = []
datas += collect_data_files("pymorgan")
datas += collect_data_files("rinoh_typeface_texgyreheros")
datas += collect_data_files("cmcrameri")

if PYRATE_SRC.exists():
    datas += collect_data_files("pyrate_ta")

# Icons
icons_dir = PYMORGAN_SRC / "pymorgan" / "gui" / "icons"
datas.append((str(icons_dir), "pymorgan/gui/icons"))

if PYRATE_SRC.exists():
    pyrate_icons = PYRATE_SRC / "pyrate_ta" / "gui" / "icons"
    datas.append((str(pyrate_icons), "pyrate_ta/gui/icons"))

# Hidden imports
hiddenimports = [
    "scipy.special",
    "scipy.signal",
    "scipy.optimize",
    "scipy.interpolate",
    "scipy.linalg",
    "matplotlib.backends.backend_qtagg",
    "matplotlib.style.core",
    "cmcrameri",
    "pandas",
    "h5py",
    "tomlkit",
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    "pymorgan",
    "pymorgan.launcher",
    "pymorgan.gui.app",
    "pymorgan.gui.suite_launcher",
    "pymorgan.gui.console",
    "pymorgan.integrations.pyrate_bridge",
]

if PYRATE_SRC.exists():
    hiddenimports += [
        "pyrate_ta",
        "pyrate_ta.launcher",
        "pyrate_ta.gui.app",
        "pyrate_ta.io.pymorgan_bridge",
    ]
    hiddenimports += collect_submodules("pyrate_ta")

hiddenimports += collect_submodules("pymorgan")

is_win = sys.platform.startswith("win")
icon_suite = str(icons_dir / "pymorgan.ico") if is_win else None
icon_morgan = str(icons_dir / "pymorgan.ico") if is_win else None
icon_pyrate = str(icons_dir / "pyrate_ta.ico") if is_win else None

# Analysis 1: PyMORGAN Suite Launcher
a_suite = Analysis(
    [str(SPEC_DIR / "entrypoints" / "run_suite.py")],
    pathex=[str(PYMORGAN_SRC), str(PYRATE_SRC)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter"],
    noarchive=False,
)
pyz_suite = PYZ(a_suite.pure)
exe_suite = EXE(
    pyz_suite,
    a_suite.scripts,
    [],
    exclude_binaries=True,
    name="PyMORGAN_Suite",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # HIDE TERMINAL WINDOW
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_suite,
)

# Analysis 2: PyMORGAN Standalone
a_morgan = Analysis(
    [str(SPEC_DIR / "entrypoints" / "run_pymorgan.py")],
    pathex=[str(PYMORGAN_SRC), str(PYRATE_SRC)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter"],
    noarchive=False,
)
pyz_morgan = PYZ(a_morgan.pure)
exe_morgan = EXE(
    pyz_morgan,
    a_morgan.scripts,
    [],
    exclude_binaries=True,
    name="PyMORGAN",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # HIDE TERMINAL WINDOW
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_morgan,
)

# Analysis 3: PyRATE-TA Standalone (if present)
if PYRATE_SRC.exists():
    a_pyrate = Analysis(
        [str(SPEC_DIR / "entrypoints" / "run_pyrate.py")],
        pathex=[str(PYMORGAN_SRC), str(PYRATE_SRC)],
        binaries=[],
        datas=datas,
        hiddenimports=hiddenimports,
        hookspath=[],
        hooksconfig={},
        runtime_hooks=[],
        excludes=["tkinter"],
        noarchive=False,
    )
    pyz_pyrate = PYZ(a_pyrate.pure)
    exe_pyrate = EXE(
        pyz_pyrate,
        a_pyrate.scripts,
        [],
        exclude_binaries=True,
        name="PyRATE-TA",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,  # HIDE TERMINAL WINDOW
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=icon_pyrate,
    )

    coll = COLLECT(
        exe_suite,
        a_suite.binaries,
        a_suite.datas,
        exe_morgan,
        a_morgan.binaries,
        a_morgan.datas,
        exe_pyrate,
        a_pyrate.binaries,
        a_pyrate.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name="PyMORGAN_Suite",
    )
else:
    coll = COLLECT(
        exe_suite,
        a_suite.binaries,
        a_suite.datas,
        exe_morgan,
        a_morgan.binaries,
        a_morgan.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name="PyMORGAN_Suite",
    )
